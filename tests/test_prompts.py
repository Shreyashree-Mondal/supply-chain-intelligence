import json
import math

import pandas as pd
import pytest

from src import config
from src.llm import metrics
from src.llm.finetune import read_jsonl
from src.llm.prompts import (BASIC_SYSTEM, ENGINEERED_SYSTEM, PROMPT_VERSION, TASK_HINTS, _shots, build_messages,
                             build_user_message, detect_task, estimate_tokens, few_shot_instructions, select_shots, trim_context)


def test_basic_style_is_the_training_format():
    """Fine-tuning and serving must see identical prompts - lock the basic format."""
    m = build_messages("Q?", "some context")
    assert m == [{"role": "system", "content": BASIC_SYSTEM},
                 {"role": "user", "content": "Context:\nsome context\n\nQuestion: Q?"}]
    assert build_messages("Q?")[1]["content"] == "Q?"
    assert build_messages("Q?", "c", "A")[-1] == {"role": "assistant", "content": "A"}


def test_engineered_structure():
    m = build_messages("What is the reorder point for Example Trail Shoe?", None, style="engineered")
    assert m[0]["role"] == "system" and ENGINEERED_SYSTEM in m[0]["content"] and TASK_HINTS["product"] in m[0]["content"]
    assert [x["role"] for x in m[1:]] == ["user", "assistant", "user"]           # one few-shot pair, then the real question
    assert m[-1]["content"] == "What is the reorder point for Example Trail Shoe?"
    assert "don't have" in m[2]["content"]                                          # no context -> the refusal shot is chosen
    with_ctx = build_messages("What is the reorder point for Example Trail Shoe?", "Product record (ID 1): X", style="engineered")
    assert "Reorder point: 82 units" in with_ctx[1]["content"]                     # context present -> the answering shot is chosen
    with pytest.raises(ValueError):
        build_messages("q", style="fancy")


@pytest.mark.parametrize("q,ctx,expected", [
    ("What is the reorder point for Perfect Rip Deck?", "", "product"),
    ("What is the EOQ of Product New 120?", "", "product"),
    ("How should we manage inventory for Nike Golf Polo?", "", "product"),
    ("What is the reorder point for X?", "Product record (ID 5): X", "product"),
    ("What is safety stock?", "", "concept"),
    ("What is the formula for economic order quantity (EOQ)?", "", "concept"),
    ("Why should a planner care about safety stock?", "", "concept"),
    ("How is safety stock calculated?", "", "concept"),
    ("Annual demand is 5000 units, each order costs $50 and holding costs $2 per unit per year. What is the EOQ?", "", "calculation"),
    ("Explain this late-delivery prediction and recommend an action.", "Late-delivery prediction: probability 0.5", "risk"),
    ("Return a JSON object with keys a and b. Output only JSON.", "x", "json"),
    ("Interpret this inventory scenario for a planner.", "Inventory scenario: demand +10%", "scenario"),
    ("Which shipping mode has the highest late-delivery rate?", "", "group_fact"),
    ("hello", "", "general"),
])
def test_task_detection(q, ctx, expected):
    assert detect_task(q, ctx) == expected


def test_task_detection_on_dataset(built):
    """Detector accuracy on the real (synthetic-data) test split, for the tasks it is meant to route."""
    want = {"calculation": "calculation", "risk_interpretation": "risk", "structured_json": "json", "scenario_analysis": "scenario",
            "product_recommendation": "product", "product_fact": "product"}
    rows = [e for e in read_jsonl(config.INSTRUCTION_DIR / "train.jsonl") if e["task_type"] in want]
    acc = sum(detect_task(e["instruction"], e["context"]) == want[e["task_type"]] for e in rows) / len(rows)
    assert acc >= 0.95, acc


def test_few_shot_examples_do_not_leak_into_the_dataset(built):
    shots = few_shot_instructions()
    for split in ["train", "val", "test"]:
        assert not (shots & {e["instruction"] for e in read_jsonl(config.INSTRUCTION_DIR / f"{split}.jsonl")}), split
    names = " ".join(e["context"] + e["instruction"] for e in read_jsonl(config.INSTRUCTION_DIR / "test.jsonl"))
    assert "Example Trail Shoe" not in names and "cross-docking" not in names.lower()


def test_few_shot_arithmetic_is_correct():
    calc = _shots()["calculation"][0][2]
    assert metrics.numeric_correct(calc, round(math.sqrt(2 * 10000 * 40 / 2))) == 1.0
    js = json.loads(_shots()["json"][0][2])
    assert set(js) == {"risk_band", "late_probability", "main_driver"} and js["risk_band"] == "Medium"
    assert "82 units" in _shots()["product"][0][2] and "40 units" in _shots()["product"][0][2]      # matches the shot's own record


def test_every_task_has_a_hint_and_a_shot():
    for t in TASK_HINTS:
        assert select_shots(t, True), t


def test_trim_context_keeps_whole_blocks_within_budget():
    blocks = [f"Block {i}: " + "x" * 300 for i in range(20)]
    out = trim_context("\n\n".join(blocks), 1000)
    assert len(out) <= 1000 and out.startswith("Block 0") and "\n\n" in out and out.split("\n\n")[-1].startswith("Block")
    assert trim_context("short", 1000) == "short"
    huge = trim_context("y" * 5000, 1000)
    assert len(huge) == 1000                                                       # a single oversize block is cut, never dropped


def test_engineered_costs_more_tokens_than_basic():
    b = estimate_tokens(build_messages("What is EOQ?"))
    e = estimate_tokens(build_messages("What is EOQ?", style="engineered"))
    assert e > b


def test_prompt_lab_renders(built):
    from src.llm.prompt_lab import render
    r = render("What is the reorder point for Product Full 100?", "engineered")
    assert r["task"] == "product" and r["messages"][-1]["role"] == "user" and "Product record (ID 100)" in r["messages"][-1]["content"]
    assert r["version"] == PROMPT_VERSION and r["few_shot_pairs"] == 1
    assert render("What is EOQ?", "basic", use_rag=False, use_tools=False)["few_shot_pairs"] == 0


def test_evaluate_summary_and_loop_with_fake_backend(built, monkeypatch):
    """Runs the whole evaluation loop (all configs, scoring, reports) with a fake LLM - no GPU needed."""
    import src.llm.generation as gen
    from src.llm import evaluate

    class Fake:
        has_adapter = True
        name = "fake"

        def generate(self, messages, max_new_tokens=200, use_adapter=True):
            user = messages[-1]["content"]
            return {"text": ("I don't have that data. " if "Context:" not in user else "") + user[:200],
                    "latency_s": 0.01, "new_tokens": 5, "prompt_tokens": sum(len(m["content"]) for m in messages) // 4}

    monkeypatch.setattr(gen, "HFBackend", Fake)
    evaluate.main(["--max-per-task", "3", "--human-n", "4", "--include-tuned-engineered"])
    df = pd.read_csv(config.REPORTS_DIR / "eval_results.csv")
    assert {"base", "base+prompt", "tuned", "base+rag", "base+rag+prompt", "tuned+rag", "tuned+prompt", "tuned+rag+prompt"} == set(df["config"])
    assert set(df.loc[df["config"].str.contains("prompt"), "prompt_style"]) == {"engineered"}
    assert set(df.loc[~df["config"].str.contains("prompt"), "prompt_style"]) == {"basic"}
    md = (config.REPORTS_DIR / "eval_summary.md").read_text()
    assert "Effect of the engineered prompt" in md and "Latency and tokens" in md
    hr = pd.read_csv(config.REPORTS_DIR / "human_review.csv")
    assert {"correctness_1_5", "groundedness_1_5"} <= set(hr.columns)
    # engineered prompts are longer, and the report says by how much
    tok = df.groupby("config")["prompt_tokens"].mean()
    assert tok["base+prompt"] > tok["base"]
