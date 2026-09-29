"""Step 1: raw CSV -> trained late-delivery model + demand forecasts + inventory policy for ALL products.

    python -m src.pipeline                      # everything
    python -m src.pipeline --skip-model         # only forecasting/inventory
    python -m src.pipeline --forecast-end 2017-09   # optional: ignore months after this (see README)
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src import config
from src.data_ingestion.load_data import load_raw
from src.forecasting.demand import build_demand_panel, forecast_all
from src.optimization.inventory import build_policy, run_inventory_scenario, summarize_scenario
from src.risk_models import late_delivery


def reference_stats(df: pd.DataFrame) -> dict:
    """Late-delivery rate and volume by shipping mode / market / region / segment / payment type."""
    def table(col):
        g = df.groupby(col).agg(order_items=("Late_delivery_risk", "size"),
                                late_rate_pct=("Late_delivery_risk", lambda s: round(s.mean() * 100, 2)),
                                avg_scheduled_days=("Days for shipment (scheduled)", "mean"),
                                avg_actual_days=("Days for shipping (real)", "mean"),
                                sales=("Sales", "sum"), profit=("Order Profit Per Order", "sum"))
        g = g.round(2).sort_values("late_rate_pct", ascending=False)
        return {str(k): {c: (float(v) if not isinstance(v, int) else v) for c, v in row.items()}
                for k, row in g.to_dict("index").items()}
    return {
        "overview": {"order_items": int(len(df)), "products": int(df["Product Card Id"].nunique()),
                     "first_order": str(df["order_date"].min().date()), "last_order": str(df["order_date"].max().date()),
                     "overall_late_rate_pct": round(float(df["Late_delivery_risk"].mean() * 100), 2),
                     "total_sales": round(float(df["Sales"].sum()), 2)},
        "shipping_mode": table("Shipping Mode"), "market": table("Market"),
        "order_region": table("Order Region"), "customer_segment": table("Customer Segment"),
        "payment_type": table("Type"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--forecast-end", default=None, help="YYYY-MM; ignore demand after this month")
    a = ap.parse_args(argv)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = load_raw(a.csv)
    print(f"Loaded {len(df):,} rows, {df['Product Card Id'].nunique()} products, "
          f"{df['order_date'].min().date()} -> {df['order_date'].max().date()}")

    if not a.skip_model:
        m = late_delivery.train(df)
        print(f"Late-delivery XGBoost | ROC-AUC {m['roc_auc']} | acc {m['accuracy']} | "
              f"precision {m['precision']} | recall {m['recall']} | test rows {m['test_rows']:,}")

    panel = build_demand_panel(df, a.forecast_end)
    last_month = panel["Month"].max()
    tail = panel.groupby("Month")["Demand"].sum().tail(6)
    print("Total monthly demand, last 6 months (check for an anomalous drop, see README):")
    print(tail.to_string())

    fc = forecast_all(panel, last_month)
    policy = build_policy(df, fc)
    policy.to_csv(config.POLICY_CSV, index=False)
    fc.to_csv(config.BACKTEST_CSV, index=False)
    (config.REFERENCE_STATS_JSON).write_text(json.dumps(reference_stats(df), indent=2))

    print(f"\nInventory policy built for {len(policy)} products (notebook covered 9):")
    print(policy.groupby(["Active_Status", "Data_Confidence"]).size().rename("products").to_string())
    print("\nForecast method mix:\n" + policy["Forecast_Method"].value_counts().to_string())
    s = summarize_scenario(run_inventory_scenario(policy, 10, 20, 0))
    print("\nStress test (+10% demand, +20% lead time):", s)
    print(f"\nWrote {config.POLICY_CSV}, {config.BACKTEST_CSV}, {config.REFERENCE_STATS_JSON}")


if __name__ == "__main__":
    main()
