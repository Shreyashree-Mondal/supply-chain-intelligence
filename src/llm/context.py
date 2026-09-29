"""Builds the context block that is put in front of the question (exact product records + retrieved notes)."""
from __future__ import annotations

from src.llm.rag import Retriever
from src.llm.tools import ProductLookup


def build_context(question: str, retriever: Retriever | None, lookup: ProductLookup | None,
                  use_rag: bool = True, use_tools: bool = True, top_k: int = 3) -> dict:
    blocks, sources, product_ids = [], [], []
    if use_tools and lookup is not None:
        product_ids, records = lookup.context_for(question)
        blocks += records
        sources += [{"id": f"tool:product_record:{p}", "title": "Inventory policy table", "score": 1.0} for p in product_ids]
    if use_rag and retriever is not None:
        exact = bool(product_ids)               # an exact product record is already in the context
        extras = 1 if exact else top_k          # then add at most one supporting note (e.g. a concept), not filler
        for h in retriever.search(question, top_k + 8):
            if len(sources) - len(product_ids) >= extras:
                break
            if exact and h["id"].startswith("product_"):
                continue                        # other products' records would only be noise
            blocks.append(h["text"])
            sources.append({"id": h["id"], "title": h["title"], "score": h["score"]})
    return {"context": "\n\n".join(blocks), "sources": sources, "product_ids": product_ids}
