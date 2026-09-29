"""Rule-based 'tool context': find the products a question is about and fetch their exact records.

This is deliberately simple lookup (not LLM function calling). Exact numbers come from the policy table,
so the LLM never has to recall or invent them.
"""
from __future__ import annotations

import re

import pandas as pd

from src import config
from src.llm.formatting import format_product_record

_STOP = {"the", "and", "for", "with", "men", "mens", "women", "womens", "girls", "boys", "kids", "s", "a", "of", "in", "to"}


def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP and len(t) > 1]


class ProductLookup:
    def __init__(self, policy: pd.DataFrame | None = None):
        self.policy = policy if policy is not None else pd.read_csv(config.POLICY_CSV)
        self._name_tokens = {int(r["Product Card Id"]): set(_tokens(r["Product Name"])) for _, r in self.policy.iterrows()}

    def find(self, question: str, max_products: int = 2) -> list[int]:
        q = question.lower()
        ids = []
        for m in re.finditer(r"(?:product|item|sku|id)\s*#?\s*(\d{1,5})", q):
            pid = int(m.group(1))
            if pid in self._name_tokens and pid not in ids:
                ids.append(pid)
        qt = set(_tokens(question))
        scored = []
        for pid, nt in self._name_tokens.items():
            hit = len(nt & qt)
            if hit >= 2 and hit / max(len(nt), 1) >= 0.5:
                scored.append((hit / len(nt), hit, pid))
        scored.sort(reverse=True)
        if scored and scored[0][0] >= 0.99:      # a full-name match beats partial matches
            scored = [t for t in scored if t[0] >= 0.99]
        for _, _, pid in scored:
            if pid not in ids:
                ids.append(pid)
        return ids[:max_products]

    def record_text(self, pid: int) -> str:
        row = self.policy[self.policy["Product Card Id"] == pid].iloc[0]
        return format_product_record(row)

    def context_for(self, question: str) -> tuple[list[int], list[str]]:
        ids = self.find(question)
        return ids, [self.record_text(i) for i in ids]
