"""pgvector tests. Skip cleanly when psycopg, sentence-transformers or a running Postgres are unavailable (e.g. CI)."""
import pytest

from src.llm.rag import Chunk, Retriever


def test_default_load_is_not_pgvector(built, monkeypatch):
    monkeypatch.delenv("SCP_RETRIEVER", raising=False)
    assert type(Retriever.load()).__name__ == "Retriever"


def test_env_switch_returns_pgvector_retriever(built, monkeypatch):
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    monkeypatch.setenv("SCP_RETRIEVER", "pgvector")
    assert type(Retriever.load()).__name__ == "PgVectorRetriever"      # no DB connection is made here


@pytest.fixture
def store(monkeypatch):
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    pytest.importorskip("sentence_transformers")
    from src.llm import pgvector_store as pg
    monkeypatch.setattr(pg, "TABLE", "rag_chunks_test")                # never touch the real table
    try:
        conn = pg.connect()
    except Exception as e:
        pytest.skip(f"no PostgreSQL available: {e}")
    yield pg
    conn.execute("DROP TABLE IF EXISTS rag_chunks_test")
    conn.close()


def test_load_and_search_roundtrip(store):
    chunks = [
        Chunk("a#0", "Safety stock", "Safety stock is a buffer held against demand variability.", "doc.pdf", 3),
        Chunk("b#0", "Fruit", "Bananas and apples are sold at the market on Sundays."),
    ]
    assert store.load_index(chunks) == 2
    hits = store.PgVectorRetriever().search("buffer inventory for demand variability", k=2)
    assert hits[0]["id"] == "a#0"
    assert (hits[0]["source"], hits[0]["page"]) == ("doc.pdf", 3)
    assert hits[0]["score"] > hits[1]["score"]
