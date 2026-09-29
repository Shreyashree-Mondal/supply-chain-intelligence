"""Small, dependency-free scoring functions for the evaluation."""
from __future__ import annotations

import json
import re

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
REFUSAL = re.compile(
    r"(don't|do not|doesn't|does not|cannot|can't|can not|unable to|no)\b[^.]{0,60}\b(have|contain|provide[d]?|include[d]?|know|access|data|information|figures?)\b"
    r"|not in the (provided )?context|not provided|no data", re.I)


def norm(s: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3})", "", s.lower())


def key_fact_recall(output: str, facts: list[str]) -> float:
    if not facts:
        return float("nan")
    o = norm(output)
    hit = 0
    for f in facts:
        pat = r"(?<![\w.])" + re.escape(norm(f)) + r"(?![\w]|\.\d)"
        hit += bool(re.search(pat, o))
    return hit / len(facts)


def last_number(text: str):
    nums = _NUM.findall(text.replace("$", ""))
    for n in reversed(nums):
        try:
            return float(n.replace(",", "").rstrip("."))
        except ValueError:
            continue
    return None


def numeric_correct(output: str, expected: float) -> float:
    got = last_number(output)
    if got is None:
        return 0.0
    tol = max(1.0, 0.005 * abs(expected)) if float(expected).is_integer() else 0.11
    return float(abs(got - expected) <= tol)


def parse_json_obj(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def json_score(output: str, reference: str) -> tuple[float, float]:
    """(valid_json, risk_band matches reference)."""
    got, ref = parse_json_obj(output), parse_json_obj(reference)
    if got is None:
        return 0.0, 0.0
    return 1.0, float(bool(ref) and got.get("risk_band") == ref.get("risk_band"))


def is_refusal(output: str) -> float:
    return float(bool(REFUSAL.search(output)))


def rouge_l(pred: str, ref: str) -> float:
    a, b = re.findall(r"\w+", pred.lower()), re.findall(r"\w+", ref.lower())
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def score_example(task: str, output: str, ex: dict) -> dict:
    """Returns rouge_l, recall, primary (the task's headline metric, 0-1) and refused."""
    res = {"rouge_l": rouge_l(output, ex["response"]), "recall": key_fact_recall(output, ex["key_facts"]),
           "refused": is_refusal(output)}
    if task == "calculation":
        res["primary"] = numeric_correct(output, ex["expected_number"])
    elif task == "structured_json":
        valid, match = json_score(output, ex["response"])
        res["primary"] = valid * match
    elif task.startswith("refusal"):
        res["primary"] = res["refused"]
    elif task == "concept_qa":
        res["primary"] = res["rouge_l"]
    else:
        res["primary"] = res["recall"]
    return res
