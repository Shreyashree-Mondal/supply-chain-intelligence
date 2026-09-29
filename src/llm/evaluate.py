"""Evaluation: base vs fine-tuned vs base+RAG vs fine-tuned+RAG, with latency and a human-review sheet.

    python -m src.llm.evaluate                      # runs on the GPU machine
    python -m src.llm.evaluate --max-per-task 15    # quicker
    python -m src.llm.evaluate --summarize-human reports/human_review.csv   # after you fill in the scores

Configs (name = model + optional RAG + optional engineered prompt):
  base            base model, basic prompt                     tuned           fine-tuned adapter, basic prompt (its training format)
  base+prompt     base model, engineered prompt (prompt engineering ONLY, no fine-tuning)
  base+rag        base + retrieved context, basic prompt       tuned+rag       fine-tuned + retrieved context
  base+rag+prompt base + retrieved context + engineered prompt
  --include-tuned-engineered adds tuned+prompt / tuned+rag+prompt (measures the train/serve prompt mismatch)

Knowledge tasks (product/group facts, concepts) are answered WITHOUT the gold context: closed-book or with
context retrieved by RAG + the product lookup tool. Skill tasks (calculations, interpretation, JSON, refusals)
get their gold context as in training and compare the model/prompt variants only.
"""
from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from src import config
from src.llm.build_instruction_dataset import KNOWLEDGE_TASKS
from src.llm.context import build_context
from src.llm.finetune import read_jsonl
from src.llm.metrics import score_example
from src.llm.prompts import PROMPT_VERSION, build_messages
from src.llm.rag import Retriever
from src.llm.tools import ProductLookup


def sample_test(rows, max_per_task, seed=config.SEED):
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["task_type"]].append(r)
    out = []
    for t, v in by.items():
        rng.shuffle(v)
        out += v[:max_per_task]
    return out


CONFIG_ORDER = ["base", "base+prompt", "base+rag", "base+rag+prompt", "tuned", "tuned+rag", "tuned+prompt", "tuned+rag+prompt"]


def summarize(df: pd.DataFrame) -> str:
    order = [c for c in CONFIG_ORDER if c in set(df["config"])]
    piv = df.pivot_table(index="task_type", columns="config", values="primary", aggfunc="mean")[order].round(3)
    lat = df.groupby("config").agg(latency_mean_s=("latency_s", "mean"), latency_p50_s=("latency_s", lambda s: s.quantile(.5)),
                                   latency_p95_s=("latency_s", lambda s: s.quantile(.95)), new_tokens_mean=("new_tokens", "mean"),
                                   prompt_tokens_mean=("prompt_tokens", "mean")).round(2).loc[order]
    ref = df[df["task_type"].isin(["product_fact", "group_fact"])].pivot_table(index="config", values="refused", aggfunc="mean").round(3)
    overall = df.groupby("config")["primary"].mean().round(3).loc[order].to_frame("mean_primary_metric")
    md = ["# Evaluation summary", "", f"Prompt version: {df['prompt_version'].iloc[0]}.", "",
          "Primary metric per task (0-1, higher is better): calculation = numeric answer correct; structured_json = valid JSON with the "
          "correct risk band; refusal_* = correctly declined; concept_qa = ROUGE-L vs reference; all others = fraction of required facts "
          "present in the answer.", "", piv.to_markdown(), "", "## Average across tasks (unweighted mean of examples)", "", overall.to_markdown(), ""]
    pairs = [("base+prompt", "base", "prompt engineering alone (no fine-tuning)"), ("base+rag+prompt", "base+rag", "prompt engineering on top of RAG"),
             ("tuned+prompt", "tuned", "engineered prompt on the fine-tuned model (train/serve mismatch)")]
    rows = []
    for a_, b_, label in pairs:
        if a_ in piv.columns and b_ in piv.columns:
            d = (piv[a_] - piv[b_]).dropna().round(3)
            rows.append(pd.DataFrame({label: d}))
    if rows:
        md += ["## Effect of the engineered prompt (difference in primary metric; positive = engineered prompt is better)", "",
               pd.concat(rows, axis=1).to_markdown(), ""]
    md += ["## Latency and tokens (per generation)", "", lat.to_markdown(), "",
           "## Share of product/group-fact questions where the model declined to answer", "", ref.to_markdown(), ""]
    return "\n".join(md)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(config.INSTRUCTION_DIR / "test.jsonl"))
    ap.add_argument("--max-per-task", type=int, default=25)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--human-n", type=int, default=20)
    ap.add_argument("--include-tuned-engineered", action="store_true")
    ap.add_argument("--summarize-human", default=None)
    a = ap.parse_args(argv)

    if a.summarize_human:
        h = pd.read_csv(a.summarize_human)
        cols = [c for c in ["correctness_1_5", "groundedness_1_5"] if c in h.columns]
        print(h.groupby("config")[cols].mean().round(2).to_string())
        return

    from src.llm.generation import HFBackend
    backend = HFBackend()
    if not backend.has_adapter:
        raise SystemExit(f"No adapter found at {config.ADAPTER_DIR}. Run the fine-tuning first.")
    retriever = Retriever.load()
    lookup = ProductLookup()
    rows = sample_test(read_jsonl(__import__("pathlib").Path(a.test)), a.max_per_task)
    print(f"Evaluating {len(rows)} test examples")

    recs = []
    for i, e in enumerate(rows):
        knowledge = e["task_type"] in KNOWLEDGE_TASKS
        variants = [("base", False, False, "basic"), ("base+prompt", False, False, "engineered"), ("tuned", True, False, "basic")]
        if knowledge:
            variants += [("base+rag", False, True, "basic"), ("base+rag+prompt", False, True, "engineered"), ("tuned+rag", True, True, "basic")]
        if a.include_tuned_engineered:
            variants.append(("tuned+prompt", True, False, "engineered"))
            if knowledge:
                variants.append(("tuned+rag+prompt", True, True, "engineered"))
        for name, tuned, rag, style in variants:
            t0 = time.perf_counter()
            if knowledge:
                ctx = build_context(e["instruction"], retriever, lookup, top_k=a.top_k)["context"] if rag else ""
            else:
                ctx = e["context"]
            retr_s = time.perf_counter() - t0
            msgs = build_messages(e["instruction"], ctx, style=style)
            g = backend.generate(msgs, max_new_tokens=a.max_new_tokens, use_adapter=tuned)
            recs.append({"id": e["id"], "task_type": e["task_type"], "config": name, "prompt_style": style, "prompt_version": PROMPT_VERSION,
                         "instruction": e["instruction"], "reference": e["response"], "output": g["text"], "latency_s": g["latency_s"],
                         "retrieval_s": retr_s if rag else 0.0, "new_tokens": g["new_tokens"], "prompt_tokens": g["prompt_tokens"],
                         **score_example(e["task_type"], g["text"], e)})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)}")

    df = pd.DataFrame(recs)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.REPORTS_DIR / "eval_results.csv", index=False)
    (config.REPORTS_DIR / "eval_summary.md").write_text(summarize(df))
    ids = random.Random(config.SEED).sample(sorted(df["id"].unique()), min(a.human_n, df["id"].nunique()))
    hr = df[df["id"].isin(ids)][["id", "task_type", "config", "instruction", "reference", "output"]].copy()
    hr["correctness_1_5"], hr["groundedness_1_5"], hr["notes"] = "", "", ""
    hr.sort_values(["id", "config"]).to_csv(config.REPORTS_DIR / "human_review.csv", index=False)
    print(summarize(df))
    print(f"\nWrote reports/eval_results.csv, eval_summary.md, human_review.csv (fill in the 1-5 scores, then run --summarize-human)")


if __name__ == "__main__":
    main()
