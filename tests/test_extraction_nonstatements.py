"""A question or isolated acknowledgment is not evidence of a personal fact."""

import asyncio

import pytest

from jesse.core.interfaces import Message
from jesse.memory.extraction import FactExtractor
from jesse.memory.store import SqliteMemoryStore
from tests.test_memory import FakeEmbedder
from tests.test_orchestrator import _build_mem


class InventsPreference:
    def __init__(self):
        self.inputs = []

    def chat(self, messages, *, stream=True):
        self.inputs.append(messages[-1].content)
        yield '[{"subject":"coffee preference","text":"the user likes Nes","confidence":0.9}]'


@pytest.mark.parametrize("text", [
    "Nes?", "Nes", "Yes.", "No thanks.", "Yes, please.", "",
    "Do I like coffee?", "Do I like coffee", "what is my favorite drink",
    "What's my birthday plan", "How long do they sleep?", "Nes? Do I like coffee?",
])
def test_nonstatements_never_reach_a_model_that_would_invent_a_fact(text):
    llm = InventsPreference()
    assert FactExtractor(llm).extract(text) == "[]"
    assert llm.inputs == []


@pytest.mark.parametrize("text", [
    "I like tea", "I'm vegetarian", "My dog is Rex.",
    "Remember that my birthday plan is skydiving.",
    "I live in Rome. What is the weather there?",
    "What I like is coffee.",
])
def test_declarations_and_mixed_turns_still_reach_extraction_unchanged(text):
    llm = InventsPreference()
    FactExtractor(llm).extract(text)
    assert llm.inputs == [f"The user said: {text}"]


@pytest.mark.parametrize("catchup", [False, True])
def test_no_fact_is_written_and_nonstatement_is_not_retried(catchup):
    orch, _, _, _ = _build_mem([])
    llm = InventsPreference()
    orch.extractor = FactExtractor(llm)
    store = SqliteMemoryStore(":memory:", FakeEmbedder())
    orch.store = store

    async def scenario():
        if catchup:
            store.append_turn(Message("user", "Nes?"))
            store.append_turn(Message("assistant", "Are you asking about coffee?"))
            await orch._catch_up_extractions()
        else:
            orch._remember_turn("Nes?", "Are you asking about coffee?")
            await orch._extract_task

    try:
        asyncio.run(scenario())
        assert store.all_facts() == []
        assert store.unprocessed_exchanges() == []
        assert len(store.recent()) == 2  # keep the real conversation, not an invented fact
        assert llm.inputs == []
    finally:
        store.close()
