"""Late-delivery risk model (XGBoost) - same 5 features and time-based split as the EDA notebook."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src import config

CATEGORICAL = ["Shipping Mode", "Type", "Customer Segment", "Order Region"]
NUMERIC = ["Days for shipment (scheduled)"]
FEATURES = ["Shipping Mode", "Days for shipment (scheduled)", "Type", "Customer Segment", "Order Region"]
TARGET = "Late_delivery_risk"

# Best parameters found by the grid search in the notebook.
XGB_PARAMS = dict(n_estimators=300, learning_rate=0.1, max_depth=5, subsample=0.8,
                  colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
                  random_state=config.SEED, n_jobs=-1)


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
         ("num", "passthrough", NUMERIC)],
        sparse_threshold=0.0,
    )
    return Pipeline([("prep", pre), ("clf", XGBClassifier(**XGB_PARAMS))])


def time_split(df: pd.DataFrame, test_fraction: float = config.LATE_TEST_FRACTION):
    """Train on the past, test on the most recent `test_fraction` of orders (no shuffling)."""
    df = df.sort_values("order_date")
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut], df.iloc[cut:]


def train(df: pd.DataFrame, model_path: Path = config.LATE_MODEL_PATH,
          metrics_path: Path = config.LATE_METRICS_JSON) -> dict:
    train_df, test_df = time_split(df)
    pipe = build_pipeline()
    pipe.fit(train_df[FEATURES], train_df[TARGET])
    prob = pipe.predict_proba(test_df[FEATURES])[:, 1]
    pred = (prob >= 0.5).astype(int)
    y = test_df[TARGET]
    metrics = {
        "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
        "test_period_start": str(test_df["order_date"].min()),
        "test_period_end": str(test_df["order_date"].max()),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, prob)), 4) if y.nunique() > 1 else None,
        "features": FEATURES,
        "known_values": {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL},
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def risk_band(p: float) -> str:
    if p >= config.LATE_BAND_HIGH:
        return "High"
    if p >= config.LATE_BAND_MEDIUM:
        return "Medium"
    return "Low"


class LateDeliveryPredictor:
    """Loads the saved pipeline; returns probability, band and per-feature contributions."""

    def __init__(self, model_path: Path = config.LATE_MODEL_PATH):
        self.pipe: Pipeline = joblib.load(model_path)
        self.names = list(self.pipe.named_steps["prep"].get_feature_names_out())

    def predict(self, record: dict) -> dict:
        X = pd.DataFrame([{k: record[k] for k in FEATURES}])
        p = float(self.pipe.predict_proba(X)[0, 1])
        return {"late_probability": round(p, 4), "risk_band": risk_band(p),
                "drivers": self._drivers(X)}

    def _drivers(self, X: pd.DataFrame, top: int = 3) -> list[dict]:
        """Aggregate XGBoost SHAP-style contributions (log-odds) back to the 5 original features."""
        enc = self.pipe.named_steps["prep"].transform(X)
        booster = self.pipe.named_steps["clf"].get_booster()
        contrib = booster.predict(xgb.DMatrix(enc), pred_contribs=True)[0][:-1]
        agg: dict[str, float] = {f: 0.0 for f in FEATURES}
        for name, val in zip(self.names, contrib):
            for f in FEATURES:
                if name.startswith(f"cat__{f}_") or name == f"num__{f}":
                    agg[f] += float(val)
                    break
        ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top]
        return [{"feature": f, "value": str(X.iloc[0][f]), "effect_log_odds": round(v, 3),
                 "direction": "raises risk" if v > 0 else "lowers risk"} for f, v in ranked]
