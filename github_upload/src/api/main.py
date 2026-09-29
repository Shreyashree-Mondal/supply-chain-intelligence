"""REST API.

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    # no GPU?  SCP_LLM_BACKEND=extractive SCP_RAG_BACKEND=tfidf uvicorn src.api.main:app

Endpoints: GET /health, POST /predict-late-risk, GET /products, GET /inventory/{product_id},
           POST /inventory-scenario, POST /ask
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from typing import Literal

from pydantic import BaseModel, Field

from src import config
from src.llm.context import build_context
from src.llm.prompts import PROMPT_VERSION, STYLES, build_messages, detect_task, estimate_tokens
from src.optimization.inventory import run_inventory_scenario, summarize_scenario

log = logging.getLogger("scp.api")


class LateRiskRequest(BaseModel):
    shipping_mode: str = Field(examples=["Standard Class"])
    scheduled_days: int = Field(ge=0, le=30, examples=[4])
    payment_type: str = Field(examples=["DEBIT"])
    customer_segment: str = Field(examples=["Consumer"])
    order_region: str = Field(examples=["Western Europe"])


class ScenarioRequest(BaseModel):
    demand_change_pct: float = Field(0, ge=-50, le=200)
    lead_time_change_pct: float = Field(0, ge=-50, le=200)
    safety_stock_change_pct: float = Field(0, ge=-100, le=200)
    top_n: int = Field(5, ge=1, le=25)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    use_rag: bool = True
    use_tools: bool = True
    use_adapter: bool = True
    prompt_style: Literal["auto", "basic", "engineered"] = "auto"   # auto: basic for the fine-tuned model, engineered otherwise
    top_k: int = Field(3, ge=1, le=8)
    max_new_tokens: int = Field(200, ge=16, le=512)


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))      # NaN -> null


@asynccontextmanager
async def lifespan(app: FastAPI):
    st = app.state
    st.policy = pd.read_csv(config.POLICY_CSV) if config.POLICY_CSV.exists() else None
    st.predictor, st.known, st.retriever, st.lookup = None, {}, None, None
    if config.LATE_MODEL_PATH.exists():
        from src.risk_models.late_delivery import LateDeliveryPredictor
        st.predictor = LateDeliveryPredictor()
        st.known = json.loads(config.LATE_METRICS_JSON.read_text()).get("known_values", {})
    if (config.RAG_INDEX_DIR / "meta.json").exists():
        from src.llm.rag import Retriever
        st.retriever = Retriever.load()
    if st.policy is not None:
        from src.llm.tools import ProductLookup
        st.lookup = ProductLookup(st.policy)
    try:
        from src.llm.generation import get_backend
        st.backend = get_backend()
    except Exception as exc:            # e.g. no GPU / model not downloaded -> keep the API up, retrieval-only
        log.warning("LLM backend failed to load (%s); falling back to extractive backend", exc)
        from src.llm.generation import ExtractiveBackend
        st.backend = ExtractiveBackend()
    yield


app = FastAPI(title="Supply Chain Copilot", version="1.0", lifespan=lifespan)


def _need_policy():
    if app.state.policy is None:
        raise HTTPException(503, "Inventory policy table not found. Run `python -m src.pipeline` first.")
    return app.state.policy


@app.get("/health")
def health():
    st = app.state
    return {"status": "ok", "products": None if st.policy is None else int(len(st.policy)),
            "late_risk_model_loaded": st.predictor is not None, "rag_index_loaded": st.retriever is not None,
            "llm_backend": st.backend.name}


@app.post("/predict-late-risk")
def predict_late_risk(req: LateRiskRequest):
    st = app.state
    if st.predictor is None:
        raise HTTPException(503, "Late-delivery model not found. Run `python -m src.pipeline` first.")
    rec = {"Shipping Mode": req.shipping_mode, "Days for shipment (scheduled)": req.scheduled_days, "Type": req.payment_type,
           "Customer Segment": req.customer_segment, "Order Region": req.order_region}
    for col, val in [("Shipping Mode", req.shipping_mode), ("Type", req.payment_type),
                     ("Customer Segment", req.customer_segment), ("Order Region", req.order_region)]:
        if st.known.get(col) and val not in st.known[col]:
            raise HTTPException(422, f"Unknown {col} '{val}'. Known values: {st.known[col]}")
    return st.predictor.predict(rec)


@app.get("/products")
def list_products(active_only: bool = True, abc_class: str | None = Query(None, pattern="^[ABC]$"), limit: int = Query(50, ge=1, le=500)):
    p = _need_policy()
    if active_only:
        p = p[p["Active_Status"] == "Active"]
    if abc_class:
        p = p[p["ABC_Class"] == abc_class]
    cols = ["Product Card Id", "Product Name", "Active_Status", "ABC_Class", "Data_Confidence", "Latest_Forecast", "Reorder_Point", "EOQ", "Inventory_Risk"]
    return {"count": int(len(p)), "products": _records(p[cols].head(limit))}


@app.get("/inventory/{product_id}")
def inventory(product_id: int):
    p = _need_policy()
    row = p[p["Product Card Id"] == product_id]
    if row.empty:
        raise HTTPException(404, f"Product {product_id} not found")
    return _records(row)[0]


@app.post("/inventory-scenario")
def scenario(req: ScenarioRequest):
    p = _need_policy()
    sc = run_inventory_scenario(p, req.demand_change_pct, req.lead_time_change_pct, req.safety_stock_change_pct)
    if sc.empty:
        raise HTTPException(422, "No active products to run a scenario on.")
    top = sc.head(req.top_n)[["Product Card Id", "Product Name", "Reorder_Point", "Scenario_Reorder_Point", "ROP_Increase_Pct", "Additional_ROP_Value"]]
    return {"parameters": req.model_dump(exclude={"top_n"}), "summary": summarize_scenario(sc), "top_increases": _records(top)}


@app.post("/ask")
def ask(req: AskRequest):
    st = app.state
    t0 = time.perf_counter()
    c = build_context(req.question, st.retriever, st.lookup, req.use_rag, req.use_tools, req.top_k)
    t1 = time.perf_counter()
    tuned = bool(req.use_adapter and getattr(st.backend, "has_adapter", False))
    style = req.prompt_style if req.prompt_style != "auto" else ("basic" if tuned else "engineered")
    msgs = build_messages(req.question, c["context"], style=style)
    g = st.backend.generate(msgs, max_new_tokens=req.max_new_tokens, use_adapter=req.use_adapter)
    t2 = time.perf_counter()
    log.info("ask: style %s retrieval %.0fms generation %.0fms prompt_tokens %s new_tokens %s", style, (t1 - t0) * 1e3,
             (t2 - t1) * 1e3, g.get("prompt_tokens"), g["new_tokens"])
    return {"answer": g["text"], "sources": c["sources"], "product_ids": c["product_ids"], "backend": st.backend.name,
            "latency_ms": {"retrieval": round((t1 - t0) * 1e3, 1), "generation": round((t2 - t1) * 1e3, 1), "total": round((t2 - t0) * 1e3, 1)},
            "prompt": {"style": style, "version": PROMPT_VERSION, "task": detect_task(req.question, c["context"])},
            "usage": {"prompt_tokens": g.get("prompt_tokens", estimate_tokens(msgs)), "new_tokens": g["new_tokens"]}}
