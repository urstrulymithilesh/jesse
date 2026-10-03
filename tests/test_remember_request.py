"""An explicit memory request needs an honest acknowledgement, not a persona reply."""

import asyncio

import pytest

from tests.test_orchestrator import _build_mem, RecordingLLM
from jesse.memory.extraction import FactExtractor
from jesse.memory.remember_parse import parse_remember_request
from jesse.core.state import ConversationState
from jesse.ui.channel import TextChannel


@pytest.mark.parametrize("via", ["voice", "text"])
def test_memory_request_acknowledges_user_and_keeps_background_extraction(via):
    raw = '[{"subject":"favorite color","text":"the user likes turquoise best","confidence":0.9}]'
    orch, transport, store, extractor = _build_mem([], raw)
    orch.text_channel = TextChannel()
    orch._engaged = True
    class WrongPersonaReply(RecordingLLM):
        def chat(self, messages, *, stream=True):
            self.last_messages = list(messages)
            yield "I don't have a favorite color."

    llm = WrongPersonaReply()
    orch.llm = llm

    async def scenario():
        await orch._handle_utterance("Remember that my favorite color is turquoise.", via=via)
        await orch._extract_task

    asyncio.run(scenario())
    assert transport.spoken == ["Got it. I'll try to remember that."]
    assert llm.last_messages == []
    assert extractor.calls == 1
    assert store.facts[0].text == "the user likes turquoise best"
    assert [m.role for m in store.turns] == ["user", "assistant"]
    assert orch.state is ConversationState.LISTENING
    assert [line.role for line in orch.text_channel.transcript] == ["you", "jesse"]
    assert orch.text_channel.transcript[-1].text == transport.spoken[-1]


@pytest.mark.parametrize("text, statement", [
    ("Remember that my favorite color is turquoise.", "my favorite color is turquoise."),
    ("Please remember that I am moving next month!", "I am moving next month!"),
    ("  REMEMBER THAT my sister is named Anya.  ", "my sister is named Anya."),
])
def test_explicit_declarations(text, statement):
    assert parse_remember_request(text) == statement


@pytest.mark.parametrize("text", [
    "Do you remember that my birthday is next month?",
    "Remember that my birthday is next month?",
    "What do you remember about my birthday?",
    "Remember my birthday", "Remember to call Mom tomorrow",
    "Remind me to call Mom tomorrow", "Forget that my favorite color is blue",
    "Remember that", "Remember that ...",
])
def test_questions_reminders_forgetting_and_empty_statements_are_not_intercepted(text):
    assert parse_remember_request(text) is None


@pytest.mark.parametrize("missing", ["store", "extractor"])
def test_memory_unavailable_is_said_explicitly(missing):
    orch, transport, _store, _extractor = _build_mem([])
    setattr(orch, missing, None)
    asyncio.run(orch._handle_utterance("Remember that my sister is named Anya.", via="text"))
    assert "can't save a lasting memory right now" in transport.spoken[-1]
    assert orch._extract_task is None


def test_extraction_failure_does_not_turn_acknowledgement_into_a_saved_claim():
    orch, transport, store, _extractor = _build_mem([])

    class FailedExtractor:
        def extract(self, text):
            raise RuntimeError("model offline")

    orch.extractor = FailedExtractor()

    async def scenario():
        await orch._handle_utterance("Remember that my sister is named Anya.", via="text")
        await orch._extract_task

    asyncio.run(scenario())
    assert transport.spoken == ["Got it. I'll try to remember that."]
    assert store.facts == [] and store.processed == set()
    assert len(store.turns) == 2  # unfinished extraction can be retried on restart


@pytest.mark.parametrize("text", [
    "Remember that if I moved to Lisbon, I would learn Portuguese.",
    "Please remember that imagine I lived in Rome.",
])
def test_memory_directive_does_not_hide_a_hypothetical_from_the_extractor(text):
    llm = RecordingLLM()
    assert FactExtractor(llm).extract(text) == "[]"
    assert llm.last_messages == []


def test_recall_question_still_reaches_the_model():
    orch, _transport, _store, _extractor = _build_mem([])
    llm = RecordingLLM()
    orch.llm = llm

    async def scenario():
        await orch._handle_utterance("Do you remember my sister's name?", via="text")
        await orch._extract_task

    asyncio.run(scenario())
    assert llm.last_messages[-1].content == "Do you remember my sister's name?"
