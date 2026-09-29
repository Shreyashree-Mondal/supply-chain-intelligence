"""Builds the supply-chain instruction-tuning dataset (JSONL) - no external LLM/API needed.

Every answer is computed from your real outputs (policy table, reference stats, trained risk model) or from
standard formulas, so the numbers are correct by construction. The *wording* is template-based, so this is a
synthetic dataset: review the sample in review_sample.csv and say so in your README.

    python -m src.llm.build_instruction_dataset
Outputs (data/processed/instructions/): train.jsonl, val.jsonl, test.jsonl, review_sample.csv, dataset_card.json
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter

import pandas as pd

from src import config
from src.llm.formatting import format_group_table, format_prediction, format_product_record, format_scenario, num
from src.llm.knowledge_base import CONCEPTS, concept_doc
from src.optimization.inventory import run_inventory_scenario, summarize_scenario

KNOWLEDGE_TASKS = {"product_fact", "group_fact", "concept_qa"}   # evaluated closed-book vs RAG
Z = {90: 1.28, 95: 1.65, 97.5: 1.96, 99: 2.33}


def ex(task, instruction, response, context="", key_facts=None, expected_number=None, split=None, group=None):
    return {"task_type": task, "instruction": instruction, "context": context, "response": response,
            "key_facts": [str(k) for k in (key_facts or [])], "expected_number": expected_number,
            "rag_eval": task in KNOWLEDGE_TASKS, "_split": split, "_group": group}


def pick(rng, options):
    return options[rng.randrange(len(options))]


# =========================================================== product-based examples
def product_examples(policy: pd.DataFrame, rng: random.Random, split_of: dict[int, str]):
    out = []
    ids = list(policy["Product Card Id"])
    for _, r in policy.iterrows():
        pid, name = int(r["Product Card Id"]), r["Product Name"]
        sp, ctx = split_of[pid], format_product_record(r)
        active = r["Active_Status"] == "Active"
        rop, eoq, ss, fc = num(r["Reorder_Point"]), num(r["EOQ"]), num(r["Safety_Stock"]), num(r["Latest_Forecast"])

        qa = [
            ("What is the reorder point for {n}?", "At what inventory level should {n} be reordered?", "How low can stock of {n} fall before we place a new order?"),
            ("What is the economic order quantity for {n}?", "How many units of {n} should we order each time?", "What is the EOQ of {n}?"),
            ("How much safety stock should we hold for {n}?", "What is the safety stock for {n}?", "How big is the safety-stock buffer for {n}?"),
            ("What is the next-month demand forecast for {n}?", "How many units of {n} do we expect to sell next month?", "Forecast demand for {n} for the coming month."),
            ("What is the ABC class and status of {n}?", "Is {n} an A, B or C item, and is it still active?", "Tell me the ABC classification and status of {n}."),
        ]
        answers = [
            (f"The reorder point for {name} is {rop} units, which includes {ss} units of safety stock.", [rop]),
            (f"The economic order quantity for {name} is {eoq} units per order, based on an ordering cost of ${config.ORDERING_COST:.0f} and a holding rate of {config.HOLDING_RATE * 100:.0f}% of the unit price.", [eoq]),
            (f"The safety stock for {name} is {ss} units, sized for a {config.SERVICE_LEVEL * 100:.0f}% service level given its forecast-error variability and {num(r['Lead_Time_Days'], 1)}-day lead time.", [ss]),
            (f"The next-month demand forecast for {name} is {fc} units, produced with {r['Forecast_Method'].replace('_', ' ')}. Data confidence is {r['Data_Confidence']} based on {int(r['History_Months'])} months of history.", [fc]),
            ((f"{name} is an active class {r['ABC_Class']} item with {r['Inventory_Risk']} forecast-variability risk." if active else
              f"{name} is inactive: its last demand month was {r['Last_Month']}, so it is not given an ABC class."), [r["ABC_Class"]] if active else ["inactive"]),
        ]
        for (phr, (a, kf)) in zip(qa, answers):
            out.append(ex("product_fact", pick(rng, phr).format(n=name), a, ctx, kf, split=sp, group=pid))

        # recommendation
        if active:
            rec = (f"Reorder {eoq} units of {name} whenever inventory falls to {rop} units, and keep {ss} units as safety stock. "
                   + {"A": "It is a class A item, so review replenishment weekly and protect its service level. ",
                      "B": "It is a class B item, so a monthly review is enough. ",
                      "C": "It is a class C item, so standard monitoring is enough. "}.get(r["ABC_Class"], "")
                   + (f"Forecast variability is {r['Inventory_Risk'].lower()}, so keep an eye on forecast accuracy. " if r["Inventory_Risk"] != "Low" else "")
                   + ("Data confidence is low, so treat these figures as provisional." if r["Data_Confidence"] == "Low" else "")).strip()
        else:
            rec = (f"{name} has had no demand recently (last demand month {r['Last_Month']}), so do not replenish it automatically. "
                   f"Confirm whether the product is discontinued first; the calculated reorder point of {rop} units is only indicative.")
        out.append(ex("product_recommendation", pick(rng, ["What should the planner do about {n}?", "Give a replenishment recommendation for {n}.", "How should we manage inventory for {n}?"]).format(n=name),
                      rec, ctx, [rop, eoq] if active else ["discontinued"], split=sp, group=pid))

        # no context -> refuse; wrong context -> refuse
        q = pick(rng, ["What is the reorder point for {n}?", "How much safety stock does {n} need?", "What is the EOQ of {n}?"]).format(n=name)
        out.append(ex("refusal_no_context", q, pick(rng, [
            f"I don't have the current inventory figures for {name} in this conversation, so I can't give a number without inventing one. Please provide its product record or query the inventory system. In general the reorder point is average daily demand x lead time + safety stock.",
            f"The figures for {name} are not in the context I was given, so I can't state them reliably. Share its product record and I will answer. As a reminder, safety stock protects against demand and lead-time variability and EOQ balances ordering and holding cost.",
            f"I can't answer that for {name} because no data for this product was provided. Please supply its record from the inventory policy table."]),
            "", split=sp, group=pid))
        other = int(rng.choice([i for i in ids if i != pid]))
        orow = policy[policy["Product Card Id"] == other].iloc[0]
        out.append(ex("refusal_wrong_context", f"What is the reorder point for {name}?",
                      f"The provided context is for {orow['Product Name']}, not {name}, so it does not contain the reorder point for {name}. Please provide the record for {name}.",
                      format_product_record(orow), split=sp, group=pid))
    return out


# =========================================================== group facts (shipping mode, market, ...)
def group_examples(stats: dict, rng: random.Random):
    out = []
    specs = [("shipping_mode", "shipping mode", "Late-delivery rate by shipping mode"), ("market", "market", "Late-delivery rate by market"),
             ("customer_segment", "customer segment", "Late-delivery rate by customer segment"),
             ("payment_type", "payment type", "Late-delivery rate by payment type"), ("order_region", "order region", "Late-delivery rate by order region")]
    for key, label, title in specs:
        t = stats[key]
        ctx = format_group_table(title, t, label)
        keys = list(t)
        hi, lo = keys[0], keys[-1]
        out.append(ex("group_fact", pick(rng, [f"Which {label} has the highest late-delivery rate?", f"Where is late delivery most common by {label}?"]),
                      f"{hi} has the highest late-delivery rate at {num(t[hi]['late_rate_pct'], 2)}%.", ctx, [num(t[hi]['late_rate_pct'], 2), hi]))
        out.append(ex("group_fact", pick(rng, [f"Which {label} has the lowest late-delivery rate?", f"Which {label} is most reliable on delivery?"]),
                      f"{lo} has the lowest late-delivery rate at {num(t[lo]['late_rate_pct'], 2)}%.", ctx, [num(t[lo]['late_rate_pct'], 2), lo]))
        for k in (keys if len(keys) <= 6 else rng.sample(keys, 6)):
            v = t[k]
            out.append(ex("group_fact", pick(rng, [f"What is the late-delivery rate for {label} {k}?", f"How often are {k} orders late?"]),
                          f"For {label} {k}, {num(v['late_rate_pct'], 2)}% of order items were delivered late, out of {num(v['order_items'])} order items.",
                          ctx, [num(v['late_rate_pct'], 2)]))
    return out


# =========================================================== concept Q&A
def concept_examples(rng: random.Random):
    out = []
    for ci, c in enumerate(CONCEPTS):
        holdout = ci % 3 == 0          # only a third of concepts contribute held-out paraphrases (keeps val/test balanced)
        term, definition, why, formula, example = c
        T = term[0].upper() + term[1:]
        kinds = [
            ([f"What is {term}?", f"Define {term} in a supply-chain context.", f"Can you explain {term}?", f"Give me a short definition of {term}."],
             f"{T} is {definition}. {why}"),
            ([f"Why does {term} matter?", f"How does {term} affect supply-chain performance?", f"Why should a planner care about {term}?"],
             why),
            ([f"Explain {term} to a new analyst.", f"Break down {term} in simple terms.", f"I'm new to supply chain - what does {term} mean?"],
             f"In simple terms, {term} is {definition}. {why}"),
        ]
        if formula:
            kinds.append(([f"What is the formula for {term}?", f"How is {term} calculated?", f"Show me how {term} is computed."],
                          f"{formula}." + (f" Example: {example}" if example else "")))
        for phrasings, resp in kinds:
            n = len(phrasings)
            for i, p in enumerate(phrasings):
                sp = ("test" if i == n - 1 else "val" if i == n - 2 else "train") if holdout else "train"
                out.append(ex("concept_qa", p, resp, "", split=sp, group=term))
                if i == 0:   # grounded variant teaches the model to use provided context
                    out.append(ex("concept_qa", p, resp, concept_doc(c), split=sp, group=term))
    return out


# =========================================================== calculations
def calc_examples(rng: random.Random, n_each: int = 60):
    out, seen = [], set()

    def add(kind, i, q, resp, ans):
        split = "test" if i % 10 == 0 else "val" if i % 10 == 1 else "train"
        out.append(ex("calculation", q, resp, "", [], expected_number=float(ans), split=split, group=f"{kind}{i}"))

    i = 0
    while i < n_each:                                    # EOQ
        D, S = rng.randrange(1200, 60001, 100), pick(rng, [20, 30, 40, 50, 75, 100, 150])
        if rng.random() < 0.5:
            H = round(rng.uniform(1.5, 12), 2)
            q = f"Annual demand is {D} units, each order costs ${S} to place, and holding costs ${H} per unit per year. What is the EOQ?"
            hs = f"holding cost H = ${H}"
        else:
            P, rate = rng.randrange(10, 151, 5), pick(rng, [15, 20, 25])
            H = round(P * rate / 100, 2)
            q = f"A product costs ${P}, the annual holding rate is {rate}%, annual demand is {D} units and ordering cost is ${S} per order. Calculate the economic order quantity."
            hs = f"holding cost H = {rate}% x ${P} = ${H}"
        if (D, S, H) in seen: continue
        seen.add((D, S, H))
        ratio = 2 * D * S / H
        e = math.sqrt(ratio)
        resp = (f"Step 1: EOQ = sqrt(2 x D x S / H), with D = {D}, S = {S} and {hs}.\nStep 2: 2 x {D} x {S} / {H} = {ratio:.1f}.\n"
                f"Step 3: sqrt({ratio:.1f}) = {e:.1f}.\nFinal answer: the EOQ is about {round(e)} units per order.")
        add("eoq", i, q, resp, round(e)); i += 1
    i = 0
    while i < n_each:                                    # safety stock
        sl, sd, L = pick(rng, list(Z)), round(rng.uniform(2, 60), 1), rng.randrange(2, 31)
        if (sl, sd, L) in seen: continue
        seen.add((sl, sd, L))
        z, sq = Z[sl], round(math.sqrt(L), 2)
        ss = z * sd * sq
        q = f"Daily demand has a standard deviation of {sd} units, the lead time is {L} days and the target service level is {sl}%. What safety stock is needed?"
        resp = (f"Step 1: The z-value for a {sl}% service level is {z}.\nStep 2: sqrt({L}) = {sq}.\n"
                f"Step 3: Safety stock = z x sigma x sqrt(L) = {z} x {sd} x {sq} = {ss:.1f}.\nFinal answer: hold about {round(ss)} units of safety stock.")
        add("ss", i, q, resp, round(ss)); i += 1
    i = 0
    while i < n_each:                                    # reorder point
        d, L, ss = rng.randrange(5, 401, 5), rng.randrange(1, 31), rng.randrange(10, 501, 5)
        if (d, L, ss, "r") in seen: continue
        seen.add((d, L, ss, "r"))
        q = pick(rng, [f"Average demand is {d} units per day, lead time is {L} days and safety stock is {ss} units. What is the reorder point?",
                       f"Compute the reorder point when daily demand is {d}, the supplier lead time is {L} days, and we keep {ss} units of safety stock."])
        resp = (f"Step 1: Lead-time demand = {d} x {L} = {d * L}.\nStep 2: Reorder point = lead-time demand + safety stock = {d * L} + {ss} = {d * L + ss}.\n"
                f"Final answer: reorder when inventory reaches {d * L + ss} units.")
        add("rop", i, q, resp, d * L + ss); i += 1
    i = 0
    while i < n_each:                                    # days of supply
        I, d = rng.randrange(100, 20001, 50), rng.randrange(5, 501, 5)
        if (I, d, "s") in seen: continue
        seen.add((I, d, "s"))
        v = I / d
        q = f"We have {I} units on hand and sell {d} units per day. How many days of supply is that?"
        resp = f"Step 1: Days of supply = inventory on hand / daily demand.\nStep 2: {I} / {d} = {v:.1f}.\nFinal answer: about {v:.1f} days of supply."
        add("dos", i, q, resp, round(v, 1)); i += 1
    return out


# =========================================================== risk-model interpretation
def risk_examples(df: pd.DataFrame, stats: dict, rng: random.Random, n: int = 170):
    from src.risk_models.late_delivery import FEATURES, LateDeliveryPredictor
    pred = LateDeliveryPredictor()
    combos = df[FEATURES].drop_duplicates()
    combos = combos.sample(min(n, len(combos)), random_state=config.SEED)
    out = []
    for i, (_, r) in enumerate(combos.iterrows()):
        inputs = {"Shipping Mode": r["Shipping Mode"], "Scheduled days": int(r["Days for shipment (scheduled)"]),
                  "Payment type": r["Type"], "Customer segment": r["Customer Segment"], "Region": r["Order Region"]}
        rec = {f: r[f] for f in FEATURES}
        res = pred.predict(rec)
        ctx = format_prediction(inputs, res)
        p, band = res["late_probability"], res["risk_band"]
        d0 = res["drivers"][0]
        mode_rate = stats["shipping_mode"].get(str(r["Shipping Mode"]), {}).get("late_rate_pct")
        action = {"High": "Flag it for proactive handling: consider expediting, confirming carrier capacity, or warning the customer about a possible delay.",
                  "Medium": "Monitor it and check carrier status before the scheduled ship date.",
                  "Low": "No special action is needed; keep standard tracking."}[band]
        base = (f"This order has a {p * 100:.1f}% predicted probability of arriving late, which is {band} risk. "
                f"The biggest driver is {d0['feature']} = {d0['value']}, which {d0['direction']}. ")
        if mode_rate is not None:
            base += f"Historically {num(mode_rate, 2)}% of {r['Shipping Mode']} order items were late. "
        split = "test" if i % 10 == 0 else "val" if i % 10 == 1 else "train"
        out.append(ex("risk_interpretation", pick(rng, ["Explain this late-delivery prediction and recommend an action.", "What does this prediction mean and what should we do?", "Interpret this risk score for the operations team."]),
                      base + action, ctx, [f"{p * 100:.1f}", band], split=split, group=f"risk{i}"))
        if i % 2 == 0:
            js = {"risk_band": band, "late_probability": round(p, 3), "main_driver": d0["feature"], "recommended_action": action.split(":")[0].split(";")[0].rstrip(".")}
            out.append(ex("structured_json", "Return a JSON object with keys risk_band, late_probability, main_driver and recommended_action for this prediction. Output only JSON.",
                          json.dumps(js), ctx, [band], split=split, group=f"risk{i}"))
    return out


# =========================================================== scenarios
def scenario_examples(policy: pd.DataFrame, rng: random.Random, n: int = 60):
    out, seen = [], set()
    while len(out) < n:
        d, l, s = rng.choice([-20, -10, 0, 5, 10, 15, 20, 30]), rng.choice([0, 10, 20, 30, 50]), rng.choice([-10, 0, 10, 15, 25])
        if (d, l, s) in seen or (d, l, s) == (0, 0, 0): continue
        seen.add((d, l, s))
        sc = run_inventory_scenario(policy, d, l, s)
        if sc.empty: break
        summ = summarize_scenario(sc)
        topn = [(r["Product Name"], r["Additional_ROP_Value"]) for _, r in sc.head(3).iterrows()]
        ctx = format_scenario({"demand_change_pct": d, "lead_time_change_pct": l, "safety_stock_change_pct": s}, summ, topn)
        why = []
        if d: why.append("higher demand raises both lead-time demand and safety stock" if d > 0 else "lower demand reduces both lead-time demand and safety stock")
        if l: why.append("a longer lead time raises lead-time demand and the safety stock needed" if l > 0 else "a shorter lead time lowers lead-time demand and safety stock")
        if s: why.append("the extra safety-stock uplift adds buffer on top" if s > 0 else "the safety-stock cut removes buffer")
        pct = summ["increase_pct"]
        verdict = ("This is a large change: prioritise class A items and consider negotiating shorter lead times." if pct > 15 else
                   "The impact is moderate and can likely be absorbed within normal replenishment planning." if pct > 0 else
                   "This releases working capital but lowers the buffer, so watch service levels.")
        resp = (f"Under this scenario the reorder-point value of the {summ['products_in_scope']} active products moves from ${num(summ['baseline_rop_value'])} to "
                f"${num(summ['scenario_rop_value'])}, a change of ${num(summ['additional_rop_value'])} ({pct:+.2f}%). The main reasons: {'; '.join(why)}. "
                f"The largest single increase is {topn[0][0]} (+${num(topn[0][1])}). {verdict}")
        i = len(out)
        split = "test" if i % 10 == 0 else "val" if i % 10 == 1 else "train"
        out.append(ex("scenario_analysis", pick(rng, ["Interpret this inventory scenario for a planner.", "What does this what-if scenario mean for our inventory investment?"]),
                      resp, ctx, [f"{pct:+.2f}"], split=split, group=f"sc{i}"))
    return out


# =========================================================== assemble
def main():
    rng = random.Random(config.SEED)
    from src.data_ingestion.load_data import load_raw
    df = load_raw()
    policy = pd.read_csv(config.POLICY_CSV)
    stats = json.loads(config.REFERENCE_STATS_JSON.read_text())

    ids = sorted(policy["Product Card Id"].astype(int))
    rng.shuffle(ids)
    n = len(ids)
    n_test, n_val = max(1, round(n * 0.1)), max(1, round(n * 0.1))
    split_of = {p: ("test" if k < n_test else "val" if k < n_test + n_val else "train") for k, p in enumerate(ids)}

    allx = (product_examples(policy, rng, split_of) + concept_examples(rng) + calc_examples(rng)
            + risk_examples(df, stats, rng) + scenario_examples(policy, rng))
    for i, e in enumerate(group_examples(stats, rng)):    # group facts: random 80/10/10
        e["_split"] = "test" if i % 10 == 0 else "val" if i % 10 == 1 else "train"
        allx.append(e)

    # validation: no empty/NaN responses
    bad = [e for e in allx if not e["response"].strip() or "nan" in e["response"].lower().split()]
    assert not bad, f"{len(bad)} bad examples, first: {bad[0]}"

    config.INSTRUCTION_DIR.mkdir(parents=True, exist_ok=True)
    buckets = {"train": [], "val": [], "test": []}
    for k, e in enumerate(allx):
        e["id"] = f"ex{k:05d}"
        buckets[e.pop("_split")].append(e)
        e.pop("_group", None)
    for name, rows in buckets.items():
        rng.shuffle(rows)
        with open(config.INSTRUCTION_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    card = {"total": len(allx), "splits": {k: len(v) for k, v in buckets.items()},
            "by_task": {k: dict(Counter(e["task_type"] for e in v)) for k, v in buckets.items()},
            "note": "Template-generated from computed values; wording is synthetic. Product-based examples are split by product "
                    "(held-out products never appear in train). Concept examples hold out paraphrases, not concepts."}
    (config.INSTRUCTION_DIR / "dataset_card.json").write_text(json.dumps(card, indent=2))
    sample = pd.DataFrame(rng.sample([e for v in buckets.values() for e in v], min(60, len(allx))))
    sample[["id", "task_type", "instruction", "context", "response"]].assign(reviewer_ok="", reviewer_notes="").to_csv(
        config.INSTRUCTION_DIR / "review_sample.csv", index=False)
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
