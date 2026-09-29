"""Prompt lab: see exactly what the model would receive, with no model or GPU needed.

    python -m src.llm.prompt_lab "What is the reorder point for Perfect Fitness Perfect Rip Deck?"
    python -m src.llm.prompt_lab "What is EOQ?" --style engineered --no-rag
    python -m src.llm.prompt_lab "..." --compare        # basic vs engineered side by side (size, task, shots)
"""
from __future__ import annotations

import argparse

from src import config
from src.llm.context import build_context
from src.llm.prompts import PROMPT_VERSION, build_messages, detect_task, estimate_tokens


def render(question: str, style: str = "engineered", use_rag: bool = True, use_tools: bool = True, top_k: int = 3) -> dict:
    retriever = lookup = None
    if use_rag and (config.RAG_INDEX_DIR / "meta.json").exists():
        from src.llm.rag import Retriever
        retriever = Retriever.load()
    if use_tools and config.POLICY_CSV.exists():
        from src.llm.tools import ProductLookup
        lookup = ProductLookup()
    c = build_context(question, retriever, lookup, use_rag, use_tools, top_k)
    msgs = build_messages(question, c["context"], style=style)
    return {"style": style, "version": PROMPT_VERSION, "task": detect_task(question, c["context"]), "sources": c["sources"],
            "messages": msgs, "approx_tokens": estimate_tokens(msgs), "few_shot_pairs": sum(m["role"] == "assistant" for m in msgs)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--style", choices=["basic", "engineered"], default="engineered")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args(argv)
    if a.compare:
        for st in ("basic", "engineered"):
            r = render(a.question, st, not a.no_rag, not a.no_tools)
            print(f"{st:11s} task={r['task']:11s} messages={len(r['messages'])} few-shot pairs={r['few_shot_pairs']} ~tokens={r['approx_tokens']} sources={len(r['sources'])}")
        return
    r = render(a.question, a.style, not a.no_rag, not a.no_tools)
    print(f"# prompt v{r['version']} | style={r['style']} | detected task={r['task']} | ~{r['approx_tokens']} tokens | sources={[s['id'] for s in r['sources']]}\n")
    for m in r["messages"]:
        print(f"--- {m['role'].upper()} " + "-" * 60)
        print(m["content"] + "\n")


if __name__ == "__main__":
    main()
