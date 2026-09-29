"""Builds the RAG knowledge base: hand-written concept notes + documents generated from YOUR data.

    python -m src.llm.knowledge_base        # after `python -m src.pipeline`
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src import config
from src.llm.formatting import format_group_table, format_product_record, num

# term, definition, why it matters, formula (or ""), worked example (or "")
CONCEPTS = [
    ("lead time", "the time between placing a replenishment order (or shipping an order) and receiving the goods",
     "Longer or more variable lead times force a planner to hold more stock during the replenishment window, so lead time directly drives reorder points and safety stock.",
     "Lead-time demand = average daily demand x lead time in days", "With 20 units of daily demand and a 5-day lead time, lead-time demand is 100 units."),
    ("safety stock", "extra inventory held to protect against demand or lead-time variability during replenishment",
     "It is the buffer that keeps the stockout probability at the chosen service level; too little causes lost sales, too much ties up cash.",
     "Safety stock = z x standard deviation of demand per day x square root of lead time in days",
     "For a 95% service level (z = 1.65), daily demand standard deviation 8 and a 9-day lead time: 1.65 x 8 x 3 = 39.6, about 40 units."),
    ("reorder point", "the inventory level at which a new replenishment order should be placed",
     "Ordering at the right level means stock arrives just before it runs out; ordering late causes stockouts and ordering early inflates inventory.",
     "Reorder point = average daily demand x lead time + safety stock", "With 20 units/day, a 5-day lead time and 30 units of safety stock, the reorder point is 130 units."),
    ("economic order quantity (EOQ)", "the order size that minimises the sum of ordering cost and holding cost",
     "It balances many small expensive orders against few large orders that raise holding cost.",
     "EOQ = square root of (2 x annual demand x cost per order / annual holding cost per unit)",
     "Annual demand 12,000, ordering cost 50, holding cost 4 per unit per year: sqrt(2 x 12000 x 50 / 4) = about 548 units."),
    ("service level", "the target probability of not stocking out during a replenishment cycle",
     "It sets the z-value used in the safety-stock formula; a higher service level needs disproportionately more safety stock.",
     "Common z-values: 90% = 1.28, 95% = 1.65, 97.5% = 1.96, 99% = 2.33", ""),
    ("fill rate", "the share of demand that is met immediately from stock",
     "It measures how much demand was actually satisfied, and can differ from cycle service level because it counts units rather than cycles.",
     "Fill rate = units shipped from stock / units demanded", ""),
    ("stockout", "a situation where demand occurs but no inventory is available to serve it",
     "Stockouts cause lost sales, backorders and unhappy customers, and are the main cost that safety stock is designed to prevent.", "", ""),
    ("backorder", "customer demand that could not be filled immediately and is fulfilled later when stock arrives",
     "Backorders keep the sale but hurt delivery performance and raise late-delivery rates.", "", ""),
    ("cycle stock", "the part of inventory that is used up between replenishments under normal conditions",
     "It is driven by order size, so larger EOQ-type orders increase average cycle stock.",
     "Average cycle stock = order quantity / 2", "An order quantity of 480 units gives an average cycle stock of 240 units."),
    ("holding cost", "the yearly cost of keeping inventory, including capital, storage, insurance and obsolescence",
     "It is the force that pushes order sizes down in EOQ; it is often estimated as a percentage of unit price.",
     "Annual holding cost per unit = holding rate x unit price", "A 20% holding rate on a $50 item gives $10 per unit per year."),
    ("ordering cost", "the fixed cost of placing and receiving one replenishment order, regardless of its size",
     "High ordering cost pushes toward larger, less frequent orders.", "", ""),
    ("inventory turnover", "how many times inventory is sold and replaced during a period",
     "Low turnover means capital is stuck in slow-moving stock; very high turnover can signal stockout risk.",
     "Inventory turnover = cost of goods sold / average inventory value", "COGS of 1,200,000 with average inventory of 200,000 gives a turnover of 6 times per year."),
    ("days of supply", "how many days the current stock will last at the current demand rate",
     "It converts an inventory count into a time buffer that planners can compare to lead time.",
     "Days of supply = inventory on hand / average daily demand", "300 units on hand with 20 units/day demand is 15 days of supply."),
    ("ABC analysis", "a classification of items by their share of total value, so that attention goes to the items that matter most",
     "Class A items (roughly the top 70% of value) get tight control and frequent review, B items moderate control and C items simple rules.",
     "Sort items by value, compute cumulative share; A = first ~70%, B = next ~20%, C = last ~10%", ""),
    ("demand forecasting", "estimating future demand from historical data and other signals",
     "Forecasts feed reorder points, safety stock and purchasing; forecast error is what safety stock has to absorb.", "", ""),
    ("forecast error", "the gap between actual demand and the forecast, commonly summarised as MAE, RMSE or MAPE",
     "The standard deviation of forecast error is the right variability input for safety stock, because that is the uncertainty left after forecasting.",
     "MAE = mean(|actual - forecast|); RMSE = sqrt(mean((actual - forecast)^2)); MAPE = mean(|actual - forecast| / actual) x 100",
     "Forecasts of 90 and 110 against actuals of 100 and 100 give MAE 10, RMSE 10 and MAPE 10%."),
    ("exponential smoothing", "a forecasting method that weights recent observations more heavily using a smoothing parameter alpha",
     "It reacts to level changes while smoothing noise, and needs only a short history, which suits many products.",
     "Level_t = alpha x actual_t + (1 - alpha) x Level_(t-1); next forecast = latest level", ""),
    ("moving average", "a forecast equal to the average of the most recent n periods",
     "It is a simple, robust baseline that works with very short histories but lags behind trends.",
     "3-month moving average forecast = (D_t + D_(t-1) + D_(t-2)) / 3", "Demand of 100, 110 and 120 gives a forecast of 110."),
    ("bullwhip effect", "the amplification of demand variability as orders move upstream from retailer to supplier",
     "It leads to excess inventory and capacity swings upstream; sharing demand data and shorter lead times reduce it.", "", ""),
    ("just-in-time", "a replenishment approach that delivers materials only when needed, minimising inventory",
     "It cuts holding cost but leaves little buffer, so it depends on reliable lead times.", "", ""),
    ("lead-time variability", "the fluctuation of actual lead times around their average",
     "Unreliable lead times require extra safety stock; reducing variability is often cheaper than holding buffer inventory.", "", ""),
    ("late delivery risk", "the probability that an order is delivered later than its scheduled shipping window",
     "It lets planners intervene early on risky orders, for example by expediting or switching the shipping mode.",
     "In the DataCo data an order item counts as late when actual shipping days exceed scheduled shipping days", ""),
    ("shipping mode", "the delivery service level chosen for an order: Standard Class (4 scheduled days), Second Class (2), First Class (1) or Same Day (0)",
     "The scheduled window is very short for premium modes, so they are missed more often; in the DataCo data First Class has the highest late rate and Standard Class the lowest.", "", ""),
    ("shipping delay", "actual shipping days minus scheduled shipping days, where scheduled days are the promised window and actual days are what happened",
     "A positive delay means the order was late, but it is only known after shipment, so it cannot be used as a feature to predict lateness in advance.",
     "Shipping delay = actual shipping days - scheduled shipping days", "Scheduled 2 days and actual 5 days gives a delay of 3 days."),
    ("coefficient of variation", "the ratio of the standard deviation to the mean, a scale-free measure of variability",
     "It makes variability comparable across products with very different volumes and is used here to rate forecast risk.",
     "CV = standard deviation / mean", "A standard deviation of 20 on a mean of 100 gives a CV of 0.20."),
    ("(s, Q) reorder policy", "a continuous-review policy that orders a fixed quantity Q whenever inventory position falls to the reorder point s",
     "It pairs naturally with EOQ for Q and the reorder-point formula for s.", "", ""),
    ("data leakage", "using information in a model that would not be available at prediction time",
     "For late-delivery prediction, features such as delivery status, actual shipping days or shipping delay reveal the outcome, so they must be excluded or the model looks unrealistically accurate.", "", ""),
    ("ROC-AUC", "the probability that a model ranks a random positive case above a random negative case, summarising ranking quality across thresholds",
     "It is threshold-free, so it compares classifiers fairly; 0.5 is random and 1.0 is perfect.", "", ""),
    ("cycle service level", "the probability of having no stockout in a replenishment cycle, which differs from fill rate, the fraction of units served",
     "Two policies with the same cycle service level can have different fill rates, so always state which measure a target refers to.", "", ""),
]


def concept_doc(c) -> str:
    term, definition, why, formula, example = c
    T = term[0].upper() + term[1:]          # keep acronyms intact ("ABC analysis", not "Abc analysis")
    parts = [f"# {T}", f"Definition: {T} is {definition}.", f"Why it matters: {why}"]
    if formula:
        parts.append(f"Formula: {formula}.")
    if example:
        parts.append(f"Example: {example}")
    return "\n".join(parts)


def _write(kb: Path, name: str, text: str):
    (kb / name).write_text(text.strip() + "\n", encoding="utf-8")


def build_knowledge_base(kb_dir: Path = config.KB_DIR) -> int:
    if kb_dir.exists():
        shutil.rmtree(kb_dir)
    kb_dir.mkdir(parents=True)
    n = 0
    for i, c in enumerate(CONCEPTS):
        _write(kb_dir, f"concept_{i:02d}.md", concept_doc(c)); n += 1

    stats = json.loads(config.REFERENCE_STATS_JSON.read_text())
    policy = pd.read_csv(config.POLICY_CSV)
    ov = stats["overview"]
    _write(kb_dir, "data_overview.md",
           f"# DataCo dataset overview\nThe dataset has {ov['order_items']:,} order items across {ov['products']} products, "
           f"from {ov['first_order']} to {ov['last_order']}. The overall late-delivery rate is {ov['overall_late_rate_pct']}%. "
           f"Total sales are ${ov['total_sales']:,.0f}."); n += 1

    for key, title, label in [("shipping_mode", "Late-delivery rate by shipping mode", "shipping mode"),
                              ("market", "Late-delivery rate by market", "market"),
                              ("order_region", "Late-delivery rate by order region", "region"),
                              ("customer_segment", "Late-delivery rate by customer segment", "segment"),
                              ("payment_type", "Late-delivery rate by payment type", "payment type")]:
        t = stats[key]
        hi, lo = next(iter(t)), list(t)[-1]
        body = format_group_table(title, t, label)
        body += f"\nHighest late rate: {hi} ({num(t[hi]['late_rate_pct'], 2)}%). Lowest late rate: {lo} ({num(t[lo]['late_rate_pct'], 2)}%)."
        _write(kb_dir, f"data_{key}.md", f"# {title}\n{body}"); n += 1

    if config.LATE_METRICS_JSON.exists():
        m = json.loads(config.LATE_METRICS_JSON.read_text())
        _write(kb_dir, "model_late_delivery.md",
               f"# Late-delivery prediction model\nAn XGBoost classifier predicts whether an order item will be late using five "
               f"features known at order time: {', '.join(m['features'])}. It is trained on the past and tested on the most recent "
               f"orders ({m['test_rows']:,} test rows). Test ROC-AUC {m['roc_auc']}, accuracy {m['accuracy']}, precision {m['precision']}, "
               f"recall {m['recall']}. Post-outcome columns (delivery status, actual shipping days, shipping delay) are excluded to avoid data leakage."); n += 1

    act = policy[policy["Active_Status"] == "Active"]
    abc = act.groupby("ABC_Class")["Reorder_Point_Value"].agg(["count", "sum"])
    tot = abc["sum"].sum() or 1
    abc_txt = "; ".join(f"class {k}: {int(r['count'])} products, {r['sum'] / tot * 100:.1f}% of reorder-point value" for k, r in abc.iterrows())
    top = act.sort_values("Reorder_Point_Value", ascending=False).head(5)
    top_txt = "; ".join(f"{r['Product Name']} (${num(r['Reorder_Point_Value'])})" for _, r in top.iterrows())
    conf = policy.groupby("Data_Confidence").size().to_dict()
    _write(kb_dir, "data_inventory_summary.md",
           f"# Inventory policy coverage and ABC summary\nInventory policies exist for all {len(policy)} products: {len(act)} active and "
           f"{len(policy) - len(act)} inactive (no demand in the last {config.ACTIVE_WINDOW_MONTHS} months of data). "
           f"Data confidence: {', '.join(f'{k} {v}' for k, v in conf.items())}. ABC classes (active products only): {abc_txt}. "
           f"Top five active products by reorder-point value: {top_txt}."); n += 1
    _write(kb_dir, "policy_assumptions.md",
           f"# Inventory policy assumptions\nService level {config.SERVICE_LEVEL * 100:.0f}% (z = {config.Z_SCORE}); ordering cost ${config.ORDERING_COST:.0f} per order; "
           f"holding cost {config.HOLDING_RATE * 100:.0f}% of unit price per year. The dataset contains outbound shipping days, not supplier lead times, so each "
           f"product's average shipping days (minimum {config.MIN_LEAD_TIME_DAYS:g} day) is used as its replenishment lead time. "
           f"Safety stock = z x forecast-error standard deviation x sqrt(lead time in months). Reorder point = monthly forecast x lead time in months + safety stock. "
           f"Products with too little history use the pooled coefficient of variation of well-forecast products, and are marked with lower data confidence."); n += 1
    fm = policy.groupby("Forecast_Method").size().to_dict()
    hi = policy[policy["Data_Confidence"] == "High"]["Backtest_MAPE"].dropna()
    _write(kb_dir, "data_forecast_quality.md",
           f"# Forecast methods and quality\nEach product gets the best method its history supports. Methods in use: "
           f"{', '.join(f'{k} ({v} products)' for k, v in fm.items())}. For high-confidence products (12+ months of history) the median "
           f"rolling one-step-ahead backtest MAPE is {hi.median():.1f}%." if len(hi) else "# Forecast methods and quality\nNo high-confidence products in this run."); n += 1

    for _, r in policy.iterrows():
        _write(kb_dir, f"product_{int(r['Product Card Id'])}.md", "# " + r["Product Name"] + "\n" + format_product_record(r)); n += 1
    return n


if __name__ == "__main__":
    print(f"Wrote {build_knowledge_base()} knowledge-base documents to {config.KB_DIR}")
