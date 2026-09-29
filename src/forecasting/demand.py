"""Demand forecasting for EVERY product (not just the ones with full history).

Each product gets the best method its history can support:
  * >= 12 months : rolling one-step-ahead backtest picks the best of naive / 3-month MA / SES
  * 6-11 months  : 3-month moving average (backtested where possible)
  * < 6 months   : mean of what exists, with error taken from the pooled coefficient of variation
Every row carries a `Data_Confidence` tier so downstream users know how much to trust it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

ALPHAS = np.round(np.arange(0.1, 1.0, 0.1), 2)


# ------------------------------------------------------------------ demand panel
def build_demand_panel(df: pd.DataFrame, end_month: str | None = None) -> pd.DataFrame:
    """Monthly demand per product; gaps between a product's first and last month are filled with 0."""
    d = df
    if end_month:
        d = d[d["order_month"] <= pd.Timestamp(end_month)]
    m = (d.groupby(["Product Card Id", "order_month"])["Order Item Quantity"].sum().rename("Demand")
         .reset_index())
    frames = []
    for pid, g in m.groupby("Product Card Id"):
        idx = pd.date_range(g["order_month"].min(), g["order_month"].max(), freq="MS")
        s = g.set_index("order_month")["Demand"].reindex(idx, fill_value=0)
        frames.append(pd.DataFrame({"Product Card Id": pid, "Month": idx, "Demand": s.values}))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ point forecasters (history -> next value)
def f_naive(h: np.ndarray) -> float:
    return float(h[-1])


def f_ma3(h: np.ndarray) -> float:
    return float(np.mean(h[-3:]))


def _ses_level(h: np.ndarray, alpha: float) -> float:
    level = h[0]
    for y in h[1:]:
        level = alpha * y + (1 - alpha) * level
    return float(level)


def f_ses(h: np.ndarray) -> float:
    """Simple exponential smoothing; alpha chosen by in-sample one-step SSE (no optimizer, no warnings)."""
    if len(h) < 3:
        return float(np.mean(h))
    best_a, best_sse = 0.5, np.inf
    for a in ALPHAS:
        level, sse = h[0], 0.0
        for y in h[1:]:
            sse += (y - level) ** 2
            level = a * y + (1 - a) * level
        if sse < best_sse:
            best_a, best_sse = a, sse
    return _ses_level(h, best_a)


METHODS = {"naive": f_naive, "moving_average_3": f_ma3, "exp_smoothing": f_ses}


def _rolling_errors(y: np.ndarray, fn, points: int, min_hist: int = 3):
    """One-step-ahead errors for the last `points` observations."""
    start = max(min_hist, len(y) - points)
    preds, actual = [], []
    for t in range(start, len(y)):
        preds.append(fn(y[:t]))
        actual.append(y[t])
    return np.array(actual, float), np.array(preds, float)


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    if len(actual) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "n": 0}
    err = actual - pred
    nz = actual > 0
    mape = float(np.mean(np.abs(err[nz]) / actual[nz]) * 100) if nz.any() else np.nan
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mape": mape, "n": int(len(actual))}


def forecast_product(y: np.ndarray) -> dict:
    """Choose method, forecast next month, estimate forecast-error std (None if it must be imputed)."""
    n = len(y)
    if n >= config.MIN_HISTORY_HIGH:
        scored = {}
        for name, fn in METHODS.items():
            a, p = _rolling_errors(y, fn, config.BACKTEST_POINTS)
            scored[name] = (_metrics(a, p), a, p)
        method = min(scored, key=lambda k: scored[k][0]["mae"])
        m = scored[method][0]
        sigma = m["rmse"] if m["n"] >= 4 else None
        conf = "High" if sigma is not None else "Medium"
    elif n >= config.MIN_HISTORY_MEDIUM:
        method = "moving_average_3"
        a, p = _rolling_errors(y, f_ma3, config.BACKTEST_POINTS)
        m = _metrics(a, p)
        sigma = m["rmse"] if m["n"] >= 3 else None
        conf = "Medium"
    else:
        method = "mean_of_history" if n < 3 else "moving_average_3"
        m = {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "n": 0}
        sigma, conf = None, "Low"
    fc = float(np.mean(y)) if method == "mean_of_history" else float(METHODS[method](y))
    return {"Forecast_Method": method, "Latest_Forecast": max(fc, 0.0), "Monthly_Sigma": sigma,
            "Data_Confidence": conf, "Backtest_MAE": m["mae"], "Backtest_MAPE": m["mape"],
            "Backtest_Points": m["n"]}


def forecast_all(panel: pd.DataFrame, last_data_month: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for pid, g in panel.groupby("Product Card Id"):
        g = g.sort_values("Month")
        y = g["Demand"].to_numpy(float)
        r = forecast_product(y)
        last = g["Month"].max()
        months_since = (last_data_month.year - last.year) * 12 + (last_data_month.month - last.month)
        r.update({
            "Product Card Id": pid, "History_Months": int(len(y)),
            "First_Month": g["Month"].min().strftime("%Y-%m"), "Last_Month": last.strftime("%Y-%m"),
            "Mean_Monthly_Demand": float(np.mean(y[-12:])),
            "Active_Status": "Active" if months_since < config.ACTIVE_WINDOW_MONTHS else "Inactive",
        })
        rows.append(r)
    out = pd.DataFrame(rows)
    # Impute error std for products without a usable backtest using the pooled CV of well-backtested ones.
    good = out[out["Monthly_Sigma"].notna() & (out["Mean_Monthly_Demand"] > 0)]
    pooled_cv = float((good["Monthly_Sigma"] / good["Mean_Monthly_Demand"]).median()) if len(good) \
        else config.DEFAULT_POOLED_CV
    miss = out["Monthly_Sigma"].isna()
    out.loc[miss, "Monthly_Sigma"] = out.loc[miss, "Mean_Monthly_Demand"] * pooled_cv
    out["Sigma_Imputed"] = miss
    out.attrs["pooled_cv"] = pooled_cv
    return out
