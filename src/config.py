"""Central configuration. Every path/assumption lives here so nothing is hard-coded elsewhere.

Most values can be overridden with environment variables (see the os.getenv calls).
"""
from __future__ import annotations

import os
from pathlib import Path

# SCP_ROOT lets the tests write into a temp folder instead of the real project folders
ROOT = Path(os.getenv("SCP_ROOT") or Path(__file__).resolve().parents[1])

# ---------------------------------------------------------------- paths
DATA_RAW = Path(os.getenv("SCP_RAW_CSV", ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"))
DATA_PROCESSED = ROOT / "data" / "processed"
KB_DIR = ROOT / "data" / "knowledge_base"
INSTRUCTION_DIR = DATA_PROCESSED / "instructions"
RAG_INDEX_DIR = ROOT / "models" / "rag_index"
MODELS_DIR = ROOT / "models"
ADAPTER_DIR = Path(os.getenv("SCP_ADAPTER_DIR", MODELS_DIR / "lora_adapter"))
REPORTS_DIR = ROOT / "reports"

POLICY_CSV = DATA_PROCESSED / "inventory_policy.csv"
BACKTEST_CSV = DATA_PROCESSED / "forecast_backtest.csv"
REFERENCE_STATS_JSON = DATA_PROCESSED / "reference_stats.json"
LATE_MODEL_PATH = MODELS_DIR / "late_delivery_xgb.joblib"
LATE_METRICS_JSON = MODELS_DIR / "late_delivery_metrics.json"

# ---------------------------------------------------------------- inventory assumptions
# (same values as the EDA notebook)
SERVICE_LEVEL = 0.95
Z_SCORE = 1.65            # z for a 95% cycle service level
ORDERING_COST = 50.0      # cost per purchase order
HOLDING_RATE = 0.20       # annual holding cost as a fraction of unit price
MIN_LEAD_TIME_DAYS = 1.0  # floor so "Same Day" products do not get a zero lead time
# The dataset only has *outbound shipping days* (not supplier lead time). By default we use
# the product's average shipping days as the replenishment lead time (this is what the final
# scenario section of the notebook did). Set to a number (e.g. 30) to use a fixed lead time.
FIXED_LEAD_TIME_DAYS = os.getenv("SCP_FIXED_LEAD_TIME_DAYS")
FIXED_LEAD_TIME_DAYS = float(FIXED_LEAD_TIME_DAYS) if FIXED_LEAD_TIME_DAYS else None

# ---------------------------------------------------------------- forecasting
ACTIVE_WINDOW_MONTHS = 3       # product is "Active" if it had demand in the last N months of data
BACKTEST_POINTS = 6            # rolling one-step-ahead backtest length for products with long history
MIN_HISTORY_HIGH = 12          # months needed for "High" data confidence
MIN_HISTORY_MEDIUM = 6         # months needed for "Medium" data confidence
DEFAULT_POOLED_CV = 0.30       # fallback coefficient of variation if nothing better is available

# ---------------------------------------------------------------- late-delivery model
LATE_TEST_FRACTION = 0.05      # most recent 5% of orders (by date) are the held-out "future" test set
LATE_BAND_HIGH = 0.75
LATE_BAND_MEDIUM = 0.50

# ---------------------------------------------------------------- LLM / RAG
BASE_MODEL = os.getenv("SCP_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
EMBED_MODEL = os.getenv("SCP_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RAG_BACKEND = os.getenv("SCP_RAG_BACKEND", "sbert")       # "sbert" (default) or "tfidf" (no downloads)
LLM_BACKEND = os.getenv("SCP_LLM_BACKEND", "hf")           # "hf" or "extractive" (no GPU/LLM needed)
USE_ADAPTER = os.getenv("SCP_USE_ADAPTER", "1") == "1"
LOAD_IN_4BIT = os.getenv("SCP_LOAD_IN_4BIT", "1") == "1"
SEED = 42
