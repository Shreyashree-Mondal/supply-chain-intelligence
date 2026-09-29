import json
import re

import pytest

from src import config
from src.llm import metrics
from src.llm.finetune import build_features, encode_example, read_jsonl
from src.llm.rag import Retriever
from src.llm.tools import ProductLookup


def test_knowledge_base_has_all_products(built, policy):
    files = {p.name for p in config.KB_DIR.glob("product_*.md")}
    assert len(files) == len(policy)
    assert (config.KB_DIR / "policy_assumptions.md").exists()


def test_retrieval(built, policy):
    r = Retriever.load()
    row = policy.iloc[3]
    hits = r.search(f"reorder point for {row['Product Name']}", 3)
    assert f"product_{int(row['Product Card Id'])}#0" in [h["id"] for h in hits]
    ids = [h["id"] for h in r.search("safety stock", 3)]
    assert any(i.startswith("concept_") for i in ids)


def test_product_lookup(built, policy):
    pl = ProductLookup(policy)
    row = policy.iloc[5]
    assert pl.find(f"What is the EOQ of {row['Product Name']}?") == [int(row["Product Card Id"])]
    assert pl.find(f"details for product {int(row['Product Card Id'])}") == [int(row["Product Card Id"])]
    assert pl.find("what is the weather today") == []


def test_instruction_dataset_schema(built):
    for split in ["train", "val", "test"]:
        rows = read_jsonl(config.INSTRUCTION_DIR / f"{split}.jsonl")
        assert rows
        for e in rows:
            assert e["instruction"].strip() and e["response"].strip()
            assert set(["id", "task_type", "context", "key_facts", "rag_eval"]) <= set(e)
    tasks = {e["task_type"] for e in read_jsonl(config.INSTRUCTION_DIR / "train.jsonl")}
    assert {"calculation", "product_fact", "concept_qa", "risk_interpretation", "structured_json", "scenario_analysis",
            "refusal_no_context", "refusal_wrong_context", "group_fact", "product_recommendation"} <= tasks


def test_no_product_leakage(built):
    ids = lambda split: {re.search(r"ID (\d+)", e["context"]).group(1) for e in read_jsonl(config.INSTRUCTION_DIR / f"{split}.jsonl")
                         if e["task_type"] in ("product_fact", "product_recommendation")}
    assert ids("test") and ids("train")
    assert not (ids("test") & ids("train")) and not (ids("val") & ids("train"))


def test_calculation_answers_are_consistent(built):
    rows = [e for s in ["train", "val", "test"] for e in read_jsonl(config.INSTRUCTION_DIR / f"{s}.jsonl") if e["task_type"] == "calculation"]
    assert len(rows) >= 200
    assert all(metrics.numeric_correct(e["response"], e["expected_number"]) == 1.0 for e in rows)


def test_json_examples_parse(built):
    rows = [e for e in read_jsonl(config.INSTRUCTION_DIR / "train.jsonl") if e["task_type"] == "structured_json"]
    assert rows and all(set(json.loads(e["response"])) == {"risk_band", "late_probability", "main_driver", "recommended_action"} for e in rows)


def test_metrics():
    assert metrics.key_fact_recall("The reorder point is 114 units", ["14"]) == 0.0      # no substring false positives
    assert metrics.key_fact_recall("The reorder point is 1,140 units", ["1140"]) == 1.0
    assert metrics.numeric_correct("Final answer: about 548 units.", 548) == 1.0
    assert metrics.numeric_correct("Final answer: about 600 units.", 548) == 0.0
    assert metrics.is_refusal("I don't have the current figures for this product.") == 1.0
    assert metrics.is_refusal("The reorder point is 120 units.") == 0.0
    assert metrics.rouge_l("safety stock protects against variability", "safety stock protects against variability") == pytest.approx(1)
    assert metrics.json_score('Here: {"risk_band": "High"}', '{"risk_band": "High"}') == (1.0, 1.0)


class _CharTok:                                                     # stand-in tokenizer: no downloads needed
    eos_token = "</s>"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_label_masking():
    e = {"instruction": "Q?", "context": "C", "response": "ANSWER"}
    f = encode_example(_CharTok(), e, 500)
    n_resp = len("ANSWER") + len("</s>")
    assert sum(l != -100 for l in f["labels"]) == n_resp                      # loss only on answer (+eos)
    assert f["labels"][-n_resp:] == f["input_ids"][-n_resp:]
    assert encode_example(_CharTok(), e, 10) is None                          # too long -> dropped, not truncated
    feats, dropped = build_features(_CharTok(), [e, e], 10)
    assert dropped == 2 and feats == []


def test_context_has_no_filler_when_product_record_is_exact(built, policy):
    from src.llm.context import build_context
    row = policy.iloc[2]
    c = build_context(f"What is the EOQ of {row['Product Name']}?", Retriever.load(), ProductLookup(policy), top_k=3)
    assert c["product_ids"] == [int(row["Product Card Id"])]
    other_products = [s for s in c["sources"] if s["id"].startswith("product_")]
    assert not other_products                                   # only the exact tool record, never other products' docs
    assert len(c["sources"]) <= 2
    generic = build_context("what is safety stock", Retriever.load(), ProductLookup(policy), top_k=3)
    assert 1 <= len(generic["sources"]) <= 3
