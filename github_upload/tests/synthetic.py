"""Synthetic DataCo-like data so tests / CI / smoke runs work without the real CSV.

NOT real data - it only mimics the columns and the history patterns (some products run the whole
period, some stop early, some start late) so every code path gets exercised.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODES = {"Standard Class": (4, 0.38), "Second Class": (2, 0.77), "First Class": (1, 0.95), "Same Day": (0, 0.46)}
REGIONS = ["Western Europe", "Central America", "South America", "Oceania", "Southeast Asia", "West of USA"]
MARKETS = {"Western Europe": "Europe", "Central America": "LATAM", "South America": "LATAM",
           "Oceania": "Pacific Asia", "Southeast Asia": "Pacific Asia", "West of USA": "USCA"}


def make_synthetic(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2015-01-01", "2018-01-01", freq="MS")          # 37 months
    plans = []
    pid = 100
    for kind, count, span in [("full", 6, (0, 37)), ("ended", 8, (0, 26)), ("mid", 5, (31, 37)),
                              ("new", 6, (35, 37)), ("mid_early", 4, (10, 18))]:
        for _ in range(count):
            plans.append((pid, f"Product {kind.title()} {pid}", span, rng.uniform(15, 120), rng.uniform(15, 150)))
            pid += 1
    rows = []
    for pid, name, (a, b), price, base in plans:
        for mi in range(a, b):
            n_orders = max(1, int(rng.poisson(base / 3)))
            for _ in range(n_orders):
                mode = rng.choice(list(MODES))
                sched, p_late = MODES[mode]
                late = int(rng.random() < p_late)
                real = sched + (rng.integers(1, 3) if late else -rng.integers(0, 2) if sched > 0 else 0)
                region = rng.choice(REGIONS)
                day = rng.integers(1, 28)
                rows.append({
                    "order date (DateOrders)": f"{months[mi].month}/{day}/{months[mi].year} {rng.integers(0, 24)}:{rng.integers(0, 60):02d}",
                    "Product Card Id": pid, "Product Name": name,
                    "Category Name": "Cleats" if pid % 2 else "Fishing",
                    "Department Name": "Footwear" if pid % 2 else "Outdoors",
                    "Order Item Quantity": int(rng.integers(1, 5)), "Product Price": round(price, 2),
                    "Sales": round(price * 2, 2), "Order Profit Per Order": round(price * 0.2, 2),
                    "Days for shipping (real)": max(int(real), 0), "Days for shipment (scheduled)": sched,
                    "Late_delivery_risk": late, "Shipping Mode": mode,
                    "Type": rng.choice(["DEBIT", "TRANSFER", "PAYMENT", "CASH"]),
                    "Customer Segment": rng.choice(["Consumer", "Corporate", "Home Office"]),
                    "Order Region": region, "Market": MARKETS[region]})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/synthetic_dataco.csv")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    make_synthetic().to_csv(a.out, index=False, encoding="latin1")
    print("wrote", a.out)
