"""Safety stock, reorder point, EOQ, ABC and what-if scenarios for every product."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def _lead_time_days(df: pd.DataFrame) -> pd.Series:
    if config.FIXED_LEAD_TIME_DAYS is not None:
        ids = df["Product Card Id"].unique()
        return pd.Series(config.FIXED_LEAD_TIME_DAYS, index=ids)
    lt = df.groupby("Product Card Id")["Days for shipping (real)"].mean()
    return lt.clip(lower=config.MIN_LEAD_TIME_DAYS)


def _policy_math(demand_m: pd.Series, sigma_m: pd.Series, lead_days: pd.Series,
                 annual: pd.Series, price: pd.Series, ss_mult: float = 1.0) -> pd.DataFrame:
    lt_months = lead_days / 30.0
    ss = config.Z_SCORE * sigma_m * np.sqrt(lt_months) * ss_mult
    ltd = demand_m * lt_months
    rop = ltd + ss
    holding = config.HOLDING_RATE * price
    eoq = np.sqrt(2 * annual * config.ORDERING_COST / holding.replace(0, np.nan)).fillna(0)
    return pd.DataFrame({"Lead_Time_Demand": ltd, "Safety_Stock": ss, "Reorder_Point": rop,
                         "Holding_Cost": holding, "EOQ": eoq})


def build_policy(df: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """One row per product in `forecasts` (i.e. every product that ever had demand)."""
    info = df.groupby("Product Card Id").agg(**{
        "Product Name": ("Product Name", "first"), "Category Name": ("Category Name", "first"),
        "Department Name": ("Department Name", "first"), "Product Price": ("Product Price", "median")})
    f = forecasts.set_index("Product Card Id")
    p = f.join(info).join(_lead_time_days(df).rename("Lead_Time_Days"))
    p["Lead_Time_Days"] = p["Lead_Time_Days"].fillna(config.MIN_LEAD_TIME_DAYS)
    p["Annual_Demand"] = p["Mean_Monthly_Demand"] * 12
    math = _policy_math(p["Latest_Forecast"], p["Monthly_Sigma"], p["Lead_Time_Days"],
                        p["Annual_Demand"], p["Product Price"])
    p = p.join(math)
    for c in ["Safety_Stock", "Reorder_Point", "EOQ", "Lead_Time_Demand"]:
        p[c] = p[c].round(0)
    p["Reorder_Point_Value"] = p["Reorder_Point"] * p["Product Price"]
    p["Safety_Stock_Value"] = p["Safety_Stock"] * p["Product Price"]
    p["EOQ_Value"] = p["EOQ"] * p["Product Price"]

    # Risk tier = forecast-error variability (CV = error std / forecast). Absolute thresholds, so the tier
    # means the same thing for every product; low-confidence products are never rated "Low".
    p["Forecast_CV"] = (p["Monthly_Sigma"] / p["Latest_Forecast"].replace(0, np.nan)).fillna(0).round(3)
    p["Inventory_Risk"] = pd.cut(p["Forecast_CV"], bins=[-np.inf, 0.10, 0.25, np.inf],
                                 labels=["Low", "Medium", "High"]).astype(str)
    p.loc[(p["Data_Confidence"] == "Low") & (p["Inventory_Risk"] == "Low"), "Inventory_Risk"] = "Medium"

    # ABC class on ACTIVE products only (cumulative share of reorder-point value 70/90 as in the notebook)
    p["ABC_Class"] = "Unclassified"
    act = p[p["Active_Status"] == "Active"].sort_values("Reorder_Point_Value", ascending=False)
    if len(act) and act["Reorder_Point_Value"].sum() > 0:
        cum = act["Reorder_Point_Value"].cumsum() / act["Reorder_Point_Value"].sum() * 100
        prev = cum.shift(fill_value=0)   # class by where the product STARTS, so the top item is always "A"
        p.loc[act.index, "ABC_Class"] = np.select([prev < 70, prev < 90], ["A", "B"], "C")
    p["Recommended_Action"] = p.apply(_action, axis=1)
    p = p.reset_index()
    cols = ["Product Card Id", "Product Name", "Category Name", "Department Name", "Product Price",
            "Active_Status", "Data_Confidence", "History_Months", "First_Month", "Last_Month",
            "Forecast_Method", "Latest_Forecast", "Mean_Monthly_Demand", "Annual_Demand",
            "Monthly_Sigma", "Sigma_Imputed", "Backtest_MAPE", "Lead_Time_Days", "Lead_Time_Demand",
            "Safety_Stock", "Reorder_Point", "EOQ", "Reorder_Point_Value", "Safety_Stock_Value",
            "EOQ_Value", "Forecast_CV", "Inventory_Risk", "ABC_Class", "Recommended_Action"]
    return p[cols].sort_values(["Active_Status", "Reorder_Point_Value"], ascending=[True, False]) \
        .reset_index(drop=True)


def _action(r: pd.Series) -> str:
    if r["Active_Status"] != "Active":
        return "No recent demand - review whether the product is discontinued before replenishing"
    caveat = " (low data confidence - treat as provisional)" if r["Data_Confidence"] == "Low" else ""
    if r["ABC_Class"] == "A":
        base = "High-value item: review replenishment weekly and protect service level"
    elif r["ABC_Class"] == "B":
        base = "Mid-value item: review replenishment monthly"
    else:
        base = "Low-value item: standard monitoring"
    if r["Inventory_Risk"] == "High":
        base += "; forecast error is high, so keep extra safety stock and review the forecast"
    return base + caveat


# ------------------------------------------------------------------ scenarios
def run_inventory_scenario(policy: pd.DataFrame, demand_change_pct: float = 0,
                           lead_time_change_pct: float = 0, safety_stock_change_pct: float = 0,
                           active_only: bool = True) -> pd.DataFrame:
    """Recompute ROP under demand / lead-time / safety-stock shocks (percent changes)."""
    s = policy[policy["Active_Status"] == "Active"].copy() if active_only else policy.copy()
    d = 1 + demand_change_pct / 100
    l = 1 + lead_time_change_pct / 100
    ss_mult = 1 + safety_stock_change_pct / 100
    lead = s["Lead_Time_Days"] * l
    math = _policy_math(s["Latest_Forecast"] * d, s["Monthly_Sigma"] * d, lead,
                        s["Annual_Demand"] * d, s["Product Price"], ss_mult)
    s["Scenario_Reorder_Point"] = math["Reorder_Point"].round(0)
    s["Scenario_Safety_Stock"] = math["Safety_Stock"].round(0)
    s["Scenario_ROP_Value"] = s["Scenario_Reorder_Point"] * s["Product Price"]
    s["ROP_Increase_Units"] = s["Scenario_Reorder_Point"] - s["Reorder_Point"]
    s["Additional_ROP_Value"] = s["Scenario_ROP_Value"] - s["Reorder_Point_Value"]
    s["ROP_Increase_Pct"] = (s["ROP_Increase_Units"] / s["Reorder_Point"].replace(0, np.nan) * 100).round(1)
    return s.sort_values("Additional_ROP_Value", ascending=False).reset_index(drop=True)


def summarize_scenario(s: pd.DataFrame) -> dict:
    base, scen = float(s["Reorder_Point_Value"].sum()), float(s["Scenario_ROP_Value"].sum())
    return {"products_in_scope": int(len(s)), "baseline_rop_value": round(base, 2),
            "scenario_rop_value": round(scen, 2), "additional_rop_value": round(scen - base, 2),
            "increase_pct": round((scen - base) / base * 100, 2) if base else 0.0}
