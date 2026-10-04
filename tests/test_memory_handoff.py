"""An acknowledged memory request must survive the next turn in this session."""

import asyncio
import json
import threading

import pytest

from jesse.core.state import ConversationState
from jesse.memory.store import SqliteMemoryStore
from tests.test_memory import FakeEmbedder
from tests.test_orchestrator import _build_mem, RecordingLLM

FACT = '[{"subject":"dog name","text":"the user has a dog named Rex","confidence":0.9}]'


@pytest.mark.parametrize("fresh_wake", [False, True])
def test_immediate_recall_waits_for_requested_fact_before_building_context(fresh_wake):
    orch, _transport, _store, _extractor = _build_mem([], FACT)
    store = SqliteMemoryStore(":memory:", FakeEmbedder())
    orch.store = store
    llm = RecordingLLM()
    orch.llm = llm

    async def scenario():
        await orch._llm_lock.acquire()  # extraction has not reached the model yet
        await orch._handle_utterance("Remember that my dog is named Rex.", via="text")
        if fresh_wake:
            orch._begin_listening()
        orch._enter(ConversationState.THINKING)
        question = asyncio.create_task(orch._handle_utterance("What is my dog's name?", via="text"))
        await asyncio.sleep(0)  # let the follow-up reach the model gate
        orch._llm_lock.release()
        await question
        if orch._extract_task is not None:
            await orch._extract_task

    try:
        asyncio.run(scenario())
        assert any("Rex" in m.content for m in llm.last_messages if m.role == "system")
        processed = store._conn.execute("SELECT processed FROM turns WHERE id IN (1,2)").fetchall()
        assert processed == [(1,), (1,)]
    finally:
        store.close()


@pytest.mark.parametrize("forget", [False, True])
def test_pending_requests_finish_before_broad_recall_or_forgetting(forget):
    orch, _transport, _store, extractor = _build_mem([])
    store = SqliteMemoryStore(":memory:", FakeEmbedder())
    orch.store = store
    llm = RecordingLLM()
    orch.llm = llm

    def extract(text):
        if not text.startswith("Remember that"):
            return "[]"
        return FACT if "Rex" in text else json.dumps([
            {"subject": "sister name", "text": "the user's sister is named Anya", "confidence": 0.9},
        ])

    extractor.extract = extract

    async def scenario():
        await orch._llm_lock.acquire()
        await orch._handle_utterance("Remember that my dog is named Rex.", via="text")
        await orch._handle_utterance("Remember that my sister is named Anya.", via="text")
        orch._begin_listening()
        orch._enter(ConversationState.THINKING)
        text = "forget my dog" if forget else "what do you remember about us"
        turn = asyncio.create_task(orch._handle_utterance(text, via="text"))
        await asyncio.sleep(0)
        orch._llm_lock.release()
        await turn
        await orch._extract_task

    try:
        asyncio.run(scenario())
        facts = " ".join(f.text for f in store.all_facts())
        assert "Anya" in facts
        assert ("Rex" in facts) is not forget
        assert not orch._requested_extractions
        assert store.unprocessed_exchanges() == []
        context = " ".join(m.content for m in llm.last_messages)
        assert "Rex" in context and "Anya" in context
        if forget:
            assert "permanently deleted" in context
    finally:
        store.close()


def test_cancelled_incidental_extraction_holds_model_lock_until_thread_finishes():
    orch, _transport, store, extractor = _build_mem([])
    started, release = threading.Event(), threading.Event()
    finished = threading.Event()

    def extract(text):
        started.set()
        assert release.wait(5), "test did not release the extraction worker"
        finished.set()
        return FACT

    extractor.extract = extract

    class NoOverlap(RecordingLLM):
        def chat(self, messages, *, stream=True):
            assert finished.is_set(), "reply overlapped a cancelled extraction's HTTP worker"
            yield from super().chat(messages, stream=stream)

    orch.llm = NoOverlap()

    async def scenario():
        extraction = asyncio.create_task(orch._extract_facts("my dog is Rex", turn_ids=(1, 2)))
        response = None
        try:
            assert await asyncio.to_thread(started.wait, 2)
            extraction.cancel()
            await asyncio.sleep(0)
            assert orch._llm_lock.locked()
            extraction.cancel()  # a second wake must not release the gate either
            await asyncio.sleep(0)
            assert orch._llm_lock.locked()
            response = asyncio.create_task(orch._think([]))
            await asyncio.sleep(0)
        finally:
            release.set()
            await asyncio.gather(extraction, return_exceptions=True)
            if response is not None:
                await response
        assert extraction.cancelled()
        assert not orch._llm_lock.locked()

    asyncio.run(scenario())
    assert store.processed == set() and store.facts == []  # cancelled job stays retryable


def test_cancelling_the_followup_does_not_cancel_the_requested_memory():
    orch, _transport, store, _extractor = _build_mem([], FACT)

    async def scenario():
        await orch._llm_lock.acquire()
        await orch._handle_utterance("Remember that my dog is named Rex.", via="text")
        extraction = orch._extract_task
        question = asyncio.create_task(orch._handle_utterance("What is my dog's name?", via="text"))
        await asyncio.sleep(0)
        question.cancel()
        with pytest.raises(asyncio.CancelledError):
            await question
        assert not extraction.done()
        orch._llm_lock.release()
        await extraction

    asyncio.run(scenario())
    assert store.processed == {1, 2}
    assert "Rex" in store.facts[0].text
