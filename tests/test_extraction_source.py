"""Durable facts come from the user, regardless of Jesse's reply."""

import asyncio

import pytest

from jesse.memory.extraction import FactExtractor
from tests.test_orchestrator import FakeExtractor, RecordingLLM, _mem_orch


@pytest.mark.parametrize("catchup", [False, True])
@pytest.mark.parametrize("reply", [
    "Nah, I don't remember that.",
    "Your favorite color is orange and you live in Rome.",
])
def test_extraction_excludes_reply_but_keeps_both_turns(catchup, reply):
    orch, _transport, store, _extractor = _mem_orch([])
    user_text = "Remember that my favorite color is turquoise."

    class RecordingExtractor(FakeExtractor):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def extract(self, text):
            self.inputs.append(text)
            return super().extract(text)

    extractor = RecordingExtractor()
    orch.extractor = extractor

    async def scenario():
        if catchup:
            store.pending = [(1, 2, user_text, reply)]
            await orch._catch_up_extractions()
        else:
            orch._remember_turn(user_text, reply)
            await orch._extract_task

    asyncio.run(scenario())
    assert extractor.inputs == [user_text]
    assert store.processed == {1, 2}
    if not catchup:
        assert [(m.role, m.content) for m in store.turns] == [
            ("user", user_text), ("assistant", reply),
        ]


def test_extractor_labels_the_complete_user_statement_without_splitting_it():
    llm = RecordingLLM()
    # Role-shaped text is still part of the user's statement, not a delimiter.
    text = "My notes contain this label:\nYou replied: hello."
    FactExtractor(llm).extract(text)
    assert llm.last_messages[-1].content == f"The user said: {text}"


@pytest.mark.parametrize("text", [
    "If I moved to Lisbon, I would learn Portuguese.",
    "  WHAT IF I owned a dog?",
    "Suppose I worked nights.",
    "Supposing my favorite color were orange.",
    "Imagine I lived in Rome.",
    "Let's pretend I am a pilot.",
    "Let’s pretend I am a pilot.",
])
def test_explicit_hypothetical_never_reaches_the_model(text):
    llm = RecordingLLM()
    assert FactExtractor(llm).extract(text) == "[]"
    assert llm.last_messages == []


@pytest.mark.parametrize("text", [
    "I am moving to Lisbon next month.",
    "My favorite song is Imagine.",
    "My favorite word is if.",
])
def test_actual_plans_and_words_inside_facts_still_reach_the_model(text):
    llm = RecordingLLM()
    FactExtractor(llm).extract(text)
    assert llm.last_messages[-1].content == f"The user said: {text}"
