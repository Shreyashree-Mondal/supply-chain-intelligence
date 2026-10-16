"""Tests for the structured agent wrapper. Uses fake messages and a fake graph: no Groq key, no network."""
import json
import sys
import types
from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

from src.agent import structured


def human(text):
    return NS(type="human", content=text)


def ai(text="", tool_calls=None, tin=0, tout=0):
    return NS(type="ai", content=text, tool_calls=tool_calls or [], usage_metadata={"input_tokens": tin, "output_tokens": tout})


def tool(call_id, value):
    return NS(type="tool", content=str(value), tool_call_id=call_id)


def eoq_turn(result=774.6):
    return [
        human("What is the EOQ?"),
        ai(tool_calls=[{"id": "c1", "name": "economic_order_quantity", "args": {"annual_demand": 12000}}], tin=100, tout=20),
        tool("c1", result),
        ai("The EOQ is about 775 units.", tin=150, tout=30),
    ]


def test_answer_is_built_from_tool_results():
    answer, metrics = structured._build(eoq_turn(), latency=0.5)
    assert answer.eoq_units == 774.6
    assert answer.safety_stock_units is None
    assert [c.name for c in answer.tool_calls] == ["economic_order_quantity"]
    assert answer.answer.startswith("The EOQ")
    assert (metrics.llm_calls, metrics.input_tokens, metrics.output_tokens, metrics.total_tokens) == (2, 250, 50, 300)


def test_tokens_count_only_the_current_turn():
    earlier = [human("old question"), ai("old answer", tin=9999, tout=9999)]
    answer, metrics = structured._build(earlier + eoq_turn(), latency=0.1)
    assert metrics.total_tokens == 300           # the old turn's tokens are excluded


def test_invalid_tool_result_is_rejected():
    with pytest.raises(ValidationError):
        structured._build(eoq_turn(result=-5), latency=0.1)       # EOQ must be > 0


def test_ask_logs_one_line_per_call(tmp_path, monkeypatch):
    fake_graph = NS(invoke=lambda payload, config: {"messages": eoq_turn()})
    fake_module = types.ModuleType("src.agent.agent")
    fake_module.graph = fake_graph
    monkeypatch.setitem(sys.modules, "src.agent.agent", fake_module)
    monkeypatch.setattr(structured, "LOG_PATH", tmp_path / "calls.jsonl")

    answer, metrics = structured.ask("What is the EOQ?")
    lines = (tmp_path / "calls.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["status"] == "ok" and rec["metrics"]["total_tokens"] == 300
    assert rec["answer"]["eoq_units"] == 774.6
