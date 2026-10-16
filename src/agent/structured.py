"""Structured, validated and logged wrapper around the LangGraph inventory agent.

    python -m src.agent.structured
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

LOG_PATH = Path("reports/agent_calls.jsonl")
# USD per 1M tokens. Fill these from the Groq pricing page for your model; 0.0 means free tier / unknown.
PRICE_IN_PER_M = 0.0
PRICE_OUT_PER_M = 0.0

FIELD_FOR_TOOL = {
    "economic_order_quantity": "eoq_units",
    "safety_stock": "safety_stock_units",
    "reorder_point": "reorder_point_units",
}


class ToolCallRecord(BaseModel):
    name: str
    args: dict
    result: float


class AgentAnswer(BaseModel):
    answer: str = Field(min_length=1)
    tool_calls: list[ToolCallRecord]
    eoq_units: Optional[float] = Field(default=None, gt=0)
    safety_stock_units: Optional[float] = Field(default=None, ge=0)
    reorder_point_units: Optional[float] = Field(default=None, ge=0)


class CallMetrics(BaseModel):
    latency_s: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    est_cost_usd: float


def _this_turn(messages: list) -> list:
    """Messages after the last human message, so memory from earlier turns is not counted."""
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    return messages[last_human + 1:]


def _build(messages: list, latency: float) -> tuple[AgentAnswer, CallMetrics]:
    turn = _this_turn(messages)
    results = {}
    for m in turn:
        if m.type == "tool":
            try:
                results[m.tool_call_id] = float(m.content)
            except (TypeError, ValueError):
                pass
    calls, fields = [], {}
    for m in turn:
        if m.type == "ai":
            for tc in m.tool_calls:
                if tc["id"] in results:
                    calls.append(ToolCallRecord(name=tc["name"], args=tc["args"], result=results[tc["id"]]))
                    if tc["name"] in FIELD_FOR_TOOL:
                        fields[FIELD_FOR_TOOL[tc["name"]]] = results[tc["id"]]
    ai = [m for m in turn if m.type == "ai"]
    tin = sum((m.usage_metadata or
