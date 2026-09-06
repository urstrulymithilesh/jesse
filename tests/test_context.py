"""Tests for build_messages — the strict read-budget assembler."""

from jesse.context import build_messages
from jesse.core.interfaces import Fact, Message


def test_persona_then_facts_then_turns_ending_on_current_message():
    facts = [Fact(text="the user's sister is named Anya", confidence=0.9, subject="sister")]
    history = [Message("user", "what's his name again?")]
    msgs = build_messages("PERSONA", facts, history, recent_limit=12, char_budget=5000)
    assert msgs[0] == Message("system", "PERSONA")
    assert msgs[1].role == "system" and "Anya" in msgs[1].content   # recalled fact injected
    assert msgs[-1].content == "what's his name again?"             # ends on current message


def test_no_system_line_when_no_persona_and_no_facts():
    msgs = build_messages("", [], [Message("user", "hi")], recent_limit=12, char_budget=5000)
    assert msgs == [Message("user", "hi")]


def test_recent_limit_caps_turn_count():
    history = [Message("user" if i % 2 == 0 else "assistant", f"m{i}") for i in range(10)]
    msgs = build_messages("P", [], history, recent_limit=4, char_budget=100000)
    turns = [m for m in msgs if m.role != "system"]
    assert len(turns) == 4 and turns[-1].content == "m9"           # only the last 4


def test_char_budget_drops_oldest_turns_but_keeps_current():
    history = [Message("user", "x" * 100) for _ in range(10)]      # 1000 chars total
    msgs = build_messages("P", [], history, recent_limit=12, char_budget=250)
    turns = [m for m in msgs if m.role != "system"]
    assert 0 < len(turns) < 10                                     # oldest dropped
    assert sum(len(m.content) for m in turns) <= 250
