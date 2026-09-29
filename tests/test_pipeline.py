import numpy as np
import pandas as pd
import pytest

from src import config
from src.forecasting.demand import forecast_product
from src.optimization.inventory import run_inventory_scenario, summarize_scenario


def test_every_product_gets_a_policy(built, raw_df, policy):
    assert len(policy) == raw_df["Product Card Id"].nunique()      # all products, not only the long-history ones
    assert set(policy["Data_Confidence"]) == {"High", "Medium", "Low"}
    for col in ["Safety_Stock", "Reorder_Point", "EOQ", "Latest_Forecast", "Lead_Time_Days"]:
        assert policy[col].notna().all(), col


def test_policy_math(policy):
    assert (policy["Safety_Stock"] >= 0).all()
    assert (policy["Reorder_Point"] + 1 >= policy["Safety_Stock"]).all()
    r = policy.iloc[0]
    eoq = np.sqrt(2 * r["Annual_Demand"] * config.ORDERING_COST / (config.HOLDING_RATE * r["Product Price"]))
    assert abs(eoq - r["EOQ"]) <= 1
    lt = r["Lead_Time_Days"] / 30
    ss = config.Z_SCORE * r["Monthly_Sigma"] * np.sqrt(lt)
    assert abs(ss - r["Safety_Stock"]) <= 1
    assert abs(r["Latest_Forecast"] * lt + ss - r["Reorder_Point"]) <= 1.5


def test_abc_only_on_active(policy):
    assert set(policy.loc[policy["Active_Status"] == "Inactive", "ABC_Class"]) <= {"Unclassified"}
    assert set(policy.loc[policy["Active_Status"] == "Active", "ABC_Class"]) <= {"A", "B", "C"}
    top = policy[policy["Active_Status"] == "Active"].sort_values("Reorder_Point_Value", ascending=False).iloc[0]
    assert top["ABC_Class"] == "A"


def test_scenarios(policy):
    base = run_inventory_scenario(policy, 0, 0, 0)
    assert summarize_scenario(base)["additional_rop_value"] == pytest.approx(0, abs=1e-6)   # no shock -> no change
    up = summarize_scenario(run_inventory_scenario(policy, 10, 20, 0))
    assert up["additional_rop_value"] > 0
    more = summarize_scenario(run_inventory_scenario(policy, 10, 20, 15))
    assert more["scenario_rop_value"] > up["scenario_rop_value"]
    down = summarize_scenario(run_inventory_scenario(policy, -20, 0, 0))
    assert down["additional_rop_value"] < 0


def test_forecast_product_tiers():
    flat = forecast_product(np.full(20, 100.0))
    assert flat["Latest_Forecast"] == pytest.approx(100) and flat["Data_Confidence"] == "High" and flat["Monthly_Sigma"] == pytest.approx(0)
    assert forecast_product(np.array([50.0, 60.0]))["Data_Confidence"] == "Low"
    assert forecast_product(np.array([50.0, 60, 55, 58, 62, 57, 59, 61]))["Data_Confidence"] == "Medium"
    assert forecast_product(np.array([0.0, 0, 0]))["Latest_Forecast"] == 0


def test_late_model(built):
    import json
    from src.risk_models.late_delivery import LateDeliveryPredictor
    m = json.loads(config.LATE_METRICS_JSON.read_text())
    assert m["roc_auc"] > 0.6
    p = LateDeliveryPredictor().predict({"Shipping Mode": "First Class", "Days for shipment (scheduled)": 1, "Type": "DEBIT",
                                         "Customer Segment": "Consumer", "Order Region": "Oceania"})
    assert 0.0 <= p["late_probability"] <= 1.0 and len(p["drivers"]) == 3
    slow = LateDeliveryPredictor().predict({"Shipping Mode": "Standard Class", "Days for shipment (scheduled)": 4, "Type": "DEBIT",
                                            "Customer Segment": "Consumer", "Order Region": "Oceania"})
    assert p["late_probability"] > slow["late_probability"]        # First Class is riskier than Standard
