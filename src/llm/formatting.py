"""Text renderings of structured data.

The SAME functions are used to (1) write knowledge-base documents, (2) build training contexts and
(3) build serving-time context, so the fine-tuned model always sees one consistent format.
"""
from __future__ import annotations

import math


def num(x, nd: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{float(x):.{nd}f}" if nd else str(int(round(float(x))))


def format_product_record(r) -> str:
    conf = f"{r['Data_Confidence']} confidence, {int(r['History_Months'])} months of history"
    mape = "" if r.get("Backtest_MAPE") is None or (isinstance(r.get("Backtest_MAPE"), float) and math.isnan(r.get("Backtest_MAPE"))) \
        else f" (backtest MAPE {num(r['Backtest_MAPE'], 1)}%)"
    return "\n".join([
        f"Product record (ID {int(r['Product Card Id'])}): {r['Product Name']}",
        f"- Unit price: ${num(r['Product Price'], 2)}",   # category/department omitted: raw labels are inconsistent
        f"- Status: {r['Active_Status']} (last demand month {r['Last_Month']}); data: {conf}",
        f"- Next-month demand forecast: {num(r['Latest_Forecast'])} units using {r['Forecast_Method']}{mape}",
        f"- Lead time: {num(r['Lead_Time_Days'], 1)} days",
        f"- Safety stock: {num(r['Safety_Stock'])} units",
        f"- Reorder point: {num(r['Reorder_Point'])} units",
        f"- Economic order quantity (EOQ): {num(r['EOQ'])} units",
        f"- ABC class: {r['ABC_Class']}; forecast-variability risk: {r['Inventory_Risk']}",
        f"- Recommended action: {r['Recommended_Action']}",
    ])


def format_group_table(title: str, table: dict, key_label: str) -> str:
    lines = [f"{title} (late-delivery rate = share of order items delivered late)"]
    for k, v in table.items():
        lines.append(f"- {key_label} {k}: late rate {num(v['late_rate_pct'], 2)}%, {num(v['order_items'])} order items, "
                     f"average scheduled {num(v['avg_scheduled_days'], 2)} days, average actual {num(v['avg_actual_days'], 2)} days")
    return "\n".join(lines)


def format_prediction(inputs: dict, result: dict) -> str:
    drivers = "; ".join(f"{d['feature']}={d['value']} ({d['direction']}, {d['effect_log_odds']:+.2f})"
                        for d in result["drivers"])
    ins = "; ".join(f"{k}={v}" for k, v in inputs.items())
    return (f"Late-delivery prediction: probability {result['late_probability']:.3f} ({result['risk_band']} risk)\n"
            f"Inputs: {ins}\nTop drivers: {drivers}")


def format_scenario(params: dict, summary: dict, top: list[tuple[str, float]]) -> str:
    lines = [f"Inventory scenario: demand {params['demand_change_pct']:+g}%, lead time {params['lead_time_change_pct']:+g}%, "
             f"safety stock {params['safety_stock_change_pct']:+g}%",
             f"- Active products in scope: {summary['products_in_scope']}",
             f"- Baseline reorder-point value: ${num(summary['baseline_rop_value'])}",
             f"- Scenario reorder-point value: ${num(summary['scenario_rop_value'])}",
             f"- Additional inventory value required: ${num(summary['additional_rop_value'])} ({summary['increase_pct']:+.2f}%)"]
    if top:
        lines.append("- Largest increases: " + "; ".join(f"{n} (+${num(v)})" for n, v in top))
    return "\n".join(lines)
