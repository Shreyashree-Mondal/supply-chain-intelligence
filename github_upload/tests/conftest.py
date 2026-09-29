"""Test fixtures. Everything runs on SYNTHETIC data inside a temp folder - the real data/ folders are never touched."""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="scp_test_"))
os.environ["SCP_ROOT"] = str(_TMP)                       # must be set before any `src` import
os.environ["SCP_RAW_CSV"] = str(_TMP / "data" / "raw" / "synthetic.csv")
os.environ["SCP_RAG_BACKEND"] = "tfidf"
os.environ["SCP_LLM_BACKEND"] = "extractive"

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from tests.synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    from src import config
    config.DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    df = make_synthetic()
    df.to_csv(config.DATA_RAW, index=False, encoding="latin1")
    return df


@pytest.fixture(scope="session")
def built(raw_df):
    """Runs the whole offline chain once: pipeline -> knowledge base -> RAG index -> instruction dataset."""
    from src import config, pipeline
    from src.llm import build_instruction_dataset, knowledge_base, rag
    pipeline.main([])
    knowledge_base.build_knowledge_base()
    rag.build_index("tfidf")
    build_instruction_dataset.main()
    return config


@pytest.fixture(scope="session")
def policy(built):
    return pd.read_csv(built.POLICY_CSV)
