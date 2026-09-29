"""Load the raw DataCo CSV and keep only the columns this project uses."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config

DATE_COL = "order date (DateOrders)"
DATE_FORMAT = "%m/%d/%Y %H:%M"

REQUIRED_COLUMNS = [
    DATE_COL,
    "Product Card Id",
    "Product Name",
    "Category Name",
    "Department Name",
    "Order Item Quantity",
    "Product Price",
    "Sales",
    "Order Profit Per Order",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk",
    "Shipping Mode",
    "Type",
    "Customer Segment",
    "Order Region",
    "Market",
]


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the CSV (latin1, as in the notebook), validate columns, parse dates."""
    path = Path(path) if path else config.DATA_RAW
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Put DataCoSupplyChainDataset.csv in data/raw/ "
            "or set SCP_RAW_CSV to its location."
        )
    df = pd.read_csv(path, encoding="latin1", usecols=lambda c: c in REQUIRED_COLUMNS)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    # The raw data has stray spaces (e.g. "South of  USA "); normalise so labels are clean and consistent everywhere.
    for c in ["Product Name", "Category Name", "Department Name", "Shipping Mode", "Type",
              "Customer Segment", "Order Region", "Market"]:
        df[c] = df[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["order_date"] = pd.to_datetime(df[DATE_COL], format=DATE_FORMAT)
    df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    return df.sort_values("order_date").reset_index(drop=True)
