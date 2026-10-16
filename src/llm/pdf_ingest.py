"""PDF ingestion for the RAG: extract text per page, chunk it, keep source + page for citations.

    python -m src.llm.pdf_ingest        # preview the chunks found in data/documents
"""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from src import config
from src.llm.rag import Chunk

DOCS_DIR = config.KB_DIR.parent / "documents"
MAX_CHARS = 900
MIN_PAGE_CHARS = 40          # skip blank / image-only pages


def _clean(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)      # re-join words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip()


def _split(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Pack whole sentences into chunks of at most max_chars."""
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(cur) + len(sent) + 1 > max_chars:
            parts.append(cur)
            cur = ""
        cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


def extract_pdf_chunks(pdf_path: Path) -> list[Chunk]:
    reader = PdfReader(str(pdf_path))
    title = pdf_path.stem.replace("_", " ").title()
    chunks: list[Chunk] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if len(text) < MIN_PAGE_CHARS:
            continue
        for i, part in enumerate(_split(text)):
            chunks.append(Chunk(id=f"{pdf_path.stem}_p{page_no}#{i}", title=title, text=f"{title}\n{part}",
                                source=pdf_path.name, page=page_no))
    return chunks


def load_pdf_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf in sorted(docs_dir.glob("*.pdf")):
        chunks.extend(extract_pdf_chunks(pdf))
    return chunks


if __name__ == "__main__":
    cs = load_pdf_chunks()
    print(f"{len(cs)} PDF chunks from {DOCS_DIR}")
    for c in cs[:3]:
        print(f"[{c.source} p.{c.page}] {c.id}: {c.text[:120]!r}")
