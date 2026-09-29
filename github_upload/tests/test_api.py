import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(built):
    from src.api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client, policy):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["products"] == len(policy) and r["late_risk_model_loaded"] and r["rag_index_loaded"]


def test_predict_late_risk(client):
    body = {"shipping_mode": "First Class", "scheduled_days": 1, "payment_type": "DEBIT", "customer_segment": "Consumer", "order_region": "Oceania"}
    r = client.post("/predict-late-risk", json=body)
    assert r.status_code == 200 and 0 <= r.json()["late_probability"] <= 1
    assert client.post("/predict-late-risk", json={**body, "shipping_mode": "Teleport"}).status_code == 422
    assert client.post("/predict-late-risk", json={**body, "scheduled_days": -3}).status_code == 422


def test_inventory_endpoints(client, policy):
    pid = int(policy.iloc[0]["Product Card Id"])
    r = client.get(f"/inventory/{pid}")
    assert r.status_code == 200 and r.json()["Reorder_Point"] == policy.iloc[0]["Reorder_Point"]
    assert client.get("/inventory/999999").status_code == 404
    lst = client.get("/products", params={"active_only": False, "limit": 500}).json()
    assert lst["count"] == len(policy)
    assert all(p["ABC_Class"] == "A" for p in client.get("/products", params={"abc_class": "A"}).json()["products"])


def test_scenario_endpoint(client):
    r = client.post("/inventory-scenario", json={"demand_change_pct": 10, "lead_time_change_pct": 20, "top_n": 3}).json()
    assert r["summary"]["additional_rop_value"] > 0 and len(r["top_increases"]) == 3
    assert client.post("/inventory-scenario", json={"demand_change_pct": 900}).status_code == 422


def test_ask_uses_exact_product_record(client, policy):
    row = policy[policy["Active_Status"] == "Active"].iloc[0]
    r = client.post("/ask", json={"question": f"What is the reorder point for {row['Product Name']}?"}).json()
    assert f"Reorder point: {int(row['Reorder_Point'])} units" in r["answer"]
    assert r["product_ids"] == [int(row["Product Card Id"])] and r["sources"] and r["latency_ms"]["total"] >= 0
    nothing = client.post("/ask", json={"question": "hello there"}).json()
    assert "answer" in nothing


def test_ask_prompt_style_and_usage(client, policy):
    row = policy[policy["Active_Status"] == "Active"].iloc[0]
    q = f"What is the reorder point for {row['Product Name']}?"
    auto = client.post("/ask", json={"question": q}).json()
    assert auto["prompt"]["style"] == "engineered"                     # extractive backend has no adapter -> engineered
    assert auto["prompt"]["task"] == "product" and auto["prompt"]["version"]
    basic = client.post("/ask", json={"question": q, "prompt_style": "basic"}).json()
    assert basic["prompt"]["style"] == "basic"
    assert auto["usage"]["prompt_tokens"] > basic["usage"]["prompt_tokens"]      # few-shot + rules cost tokens
    assert client.post("/ask", json={"question": q, "prompt_style": "wild"}).status_code == 422
