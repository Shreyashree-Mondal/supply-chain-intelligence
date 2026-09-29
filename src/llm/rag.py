"""Retrieval-Augmented Generation: chunking, embeddings, FAISS (or TF-IDF) index, search.

    python -m src.llm.rag build        # build index from data/knowledge_base
    python -m src.llm.rag search "reorder point for Perfect Rip Deck"
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from src import config

MAX_CHARS = 900


@dataclass
class Chunk:
    id: str
    title: str
    text: str


def load_chunks(kb_dir: Path = config.KB_DIR) -> list[Chunk]:
    """One chunk per short document; long documents are split on line boundaries."""
    chunks: list[Chunk] = []
    for f in sorted(kb_dir.glob("*.md")):
        raw = f.read_text(encoding="utf-8").strip()
        lines = raw.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else f.stem
        body_lines, cur, n = lines[1:], [], 0
        parts = []
        for ln in body_lines:
            if n + len(ln) > MAX_CHARS and cur:
                parts.append("\n".join(cur)); cur, n = [], 0
            cur.append(ln); n += len(ln) + 1
        if cur:
            parts.append("\n".join(cur))
        for i, p in enumerate(parts or [""]):
            chunks.append(Chunk(id=f"{f.stem}#{i}", title=title, text=f"{title}\n{p}".strip()))
    return chunks


class Retriever:
    def __init__(self, backend: str = config.RAG_BACKEND):
        self.backend = backend
        self.chunks: list[Chunk] = []
        self._index = None
        self._model = None
        self._vec = None
        self._mat = None

    # ---------------------------------------------------------------- build / save / load
    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(config.EMBED_MODEL)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32")

    def build(self, chunks: list[Chunk]) -> "Retriever":
        self.chunks = chunks
        texts = [f"{c.title}. {c.title}. {c.text}" for c in chunks]   # title repeated so it weighs more
        if self.backend == "sbert":
            import faiss
            emb = self._embed(texts)
            self._index = faiss.IndexFlatIP(emb.shape[1])   # cosine similarity (embeddings are normalised)
            self._index.add(emb)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            self._mat = self._vec.fit_transform(texts)
        return self

    def save(self, out: Path = config.RAG_INDEX_DIR):
        out.mkdir(parents=True, exist_ok=True)
        (out / "chunks.jsonl").write_text("\n".join(json.dumps(asdict(c)) for c in self.chunks), encoding="utf-8")
        (out / "meta.json").write_text(json.dumps({"backend": self.backend, "embed_model": config.EMBED_MODEL}))
        if self.backend == "sbert":
            import faiss
            faiss.write_index(self._index, str(out / "index.faiss"))
        else:
            import joblib
            joblib.dump((self._vec, self._mat), out / "tfidf.joblib")

    @classmethod
    def load(cls, path: Path = config.RAG_INDEX_DIR) -> "Retriever":
        meta = json.loads((path / "meta.json").read_text())
        r = cls(meta["backend"])
        r.chunks = [Chunk(**json.loads(l)) for l in (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l]
        if r.backend == "sbert":
            import faiss
            r._index = faiss.read_index(str(path / "index.faiss"))
        else:
            import joblib
            r._vec, r._mat = joblib.load(path / "tfidf.joblib")
        return r

    # ---------------------------------------------------------------- search
    def search(self, query: str, k: int = 4) -> list[dict]:
        if self.backend == "sbert":
            scores, ids = self._index.search(self._embed([query]), k)
            pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        else:
            sims = (self._mat @ self._vec.transform([query]).T).toarray().ravel()
            top = np.argsort(-sims)[:k]
            pairs = [(int(i), float(sims[i])) for i in top if sims[i] > 0]
        return [{"id": self.chunks[i].id, "title": self.chunks[i].title, "score": round(s, 4),
                 "text": self.chunks[i].text} for i, s in pairs]


def build_index(backend: str | None = None) -> Retriever:
    r = Retriever(backend or config.RAG_BACKEND).build(load_chunks())
    r.save()
    return r


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        r = build_index()
        print(f"Indexed {len(r.chunks)} chunks with backend '{r.backend}' -> {config.RAG_INDEX_DIR}")
    elif cmd == "search":
        for h in Retriever.load().search(" ".join(sys.argv[2:]), 4):
            print(f"[{h['score']}] {h['id']}: {h['text'][:160]!r}")
