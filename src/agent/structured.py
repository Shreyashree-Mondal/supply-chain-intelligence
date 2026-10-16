"""Structured, validated and logged wrapper around the LangGraph inventory agent.

    python -m src.agent.structured
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

LOG_PATH = Path("reports/agent_calls.jsonl")
# USD per 1M tokens. Fill these from the Groq pricing page for your model; 0.0 means free tier / unknown.
PRICE_IN_PER_M = 0.0
PRICE_OUT_PER_M = 0.0

FIELD_FOR_TOOL = {
    "economic_order_quantity": "eoq_units",
    "safety_stock": "safety_stock_units",
    "reorder_point": "reorder_point_units",
}


class ToolCallRecord(BaseModel):
    name: str
    args: dict
    result: float


class AgentAnswer(BaseModel):
    answer: str = Field(min_length=1)
    tool_calls: list[ToolCallRecord]
    eoq_units: Optional[float] = Field(default=None, gt=0)
    safety_stock_units: Optional[float] = Field(default=None, ge=0)
    reorder_point_units: Optional[float] = Field(default=None, ge=0)


class CallMetrics(BaseModel):
    latency_s: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    est_cost_usd: float


def _this_turn(messages: list) -> list:
    """Messages after the last human message, so memory from earlier turns is not counted."""
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    return messages[last_human + 1:]


def _build(messages: list, latency: float) -> tuple[AgentAnswer, CallMetrics]:
    turn = _this_turn(messages)
    results = {}
    for m in turn:
        if m.type == "tool":
            try:
                results[m.tool_call_id] = float(m.content)
            except (TypeError, ValueError):
                pass
    calls, fields = [], {}
    for m in turn:
        if m.type == "ai":
            for tc in m.tool_calls:
                if tc["id"] in results:
                    calls.append(ToolCallRecord(name=tc["name"], args=tc["args"], result=results[tc["id"]]))
                    if tc["name"] in FIELD_FOR_TOOL:
                        fields[FIELD_FOR_TOOL[tc["name"]]] = results[tc["id"]]
    ai = [m for m in turn if m.type == "ai"]
    tin = sum((m.usage_metadata or {}).get("input_tokens", 0) for m in ai)
    tout = sum((m.usage_metadata or {}).get("output_tokens", 0) for m in ai)
    metrics = CallMetrics(latency_s=round(latency, 3), llm_calls=len(ai), input_tokens=tin, output_tokens=tout,
                          total_tokens=tin + tout,
                          est_cost_usd=round(tin / 1e6 * PRICE_IN_PER_M + tout / 1e6 * PRICE_OUT_PER_M, 6))
    answer = AgentAnswer(answer=str(messages[-1].content).strip(), tool_calls=calls, **fields)  # validates here
    return answer, metrics


def _log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def ask(question: str, thread_id: str = "structured-demo") -> tuple[AgentAnswer, CallMetrics]:
    from src.agent.agent import graph  # lazy: importing it creates the Groq client, which needs an API key
    call_id, t0 = uuid.uuid4().hex[:8], time.perf_counter()
    result = graph.invoke({"messages": [("user", question)]}, {"configurable": {"thread_id": thread_id}})
    latency = time.perf_counter() - t0
    base = {"ts": datetime.now(timezone.utc).isoformat(), "call_id": call_id, "question": question}
    try:
        answer, metrics = _build(result["messages"], latency)
    except ValidationError as e:
        _log({**base, "status": "invalid", "latency_s": round(latency, 3), "error": str(e)[:500]})
        raise
    _log({**base, "status": "ok", "metrics": metrics.model_dump(), "answer": answer.model_dump()})
    return answer, metrics


if __name__ == "__main__":
    qs = [
        "A product has annual demand of 12000 units, order cost of 50, and holding cost of 2 per unit per year. What is the EOQ?",
        "Average daily demand is 40, lead time is 7 days, daily demand std dev is 10, and I want a 95% service level. What is the reorder point?",
    ]
    for i, q in enumerate(qs):
        a, m = ask(q, thread_id=f"demo-{i}")
        print(json.dumps({"answer": a.model_dump(), "metrics": m.model_dump()}, indent=2), "\n")
