"""A corrected plan must beat the clock and unrelated semantic neighbours."""

import asyncio
from datetime import datetime, timedelta

import pytest
from types import SimpleNamespace

from jesse.core.interfaces import Fact, Message
from jesse.memory.recall import asks_personal_memory, apply_explicit_corrections
from jesse.memory.store import SqliteMemoryStore
from tests.test_memory import FakeEmbedder
from tests.test_orchestrator import _build_mem, RecordingLLM

QUERY = "what is my birthday plan?"
CLOCK = "the current date and time is Sunday, 20 September 2026, 01:17"
PLAN = Fact(subject="birthday plans", text="the user wants to be guide-eyeing next month",
            confidence=0.8)


class ClockFirst(FakeEmbedder):
    def embed(self, texts):
        return super().embed([CLOCK if t == QUERY else t for t in texts])


def populate(store):
    store.add_fact(Fact(subject="date and time", text=CLOCK, confidence=1))
    store.add_fact(PLAN)
    store.add_fact(Fact(subject="skydiving", text="the user wants to do skydiving",
                        confidence=0.8))
    store.append_turn(Message("user", "Next month my birthday is coming up; I want guide-eyeing."))
    store.append_turn(Message("assistant", "There is no birthday plan."))
    store.append_turn(Message("user", "That's not guide-eyeing, it's sky-diving."))


def test_subject_match_beats_wrong_semantic_neighbour_after_restart(tmp_path):
    path = tmp_path / "memory.db"
    with_store = SqliteMemoryStore(path, ClockFirst())
    populate(with_store)
    with_store.close()
    store = SqliteMemoryStore(path, ClockFirst())
    try:
        assert store.recall(QUERY, k=1) == [PLAN]
        evidence = store.recall_evidence(QUERY)
        assert all(m.role == "user" for m in evidence)
        corrected = apply_explicit_corrections(store.recall(QUERY), evidence)
        assert "sky-diving" in corrected[0].text
        assert PLAN in store.all_facts()  # no destructive consolidation
        assert len(store.all_facts()) == 3
    finally:
        store.close()


def test_recall_prompt_uses_correction_without_clock_or_assistant_claims():
    store = SqliteMemoryStore(":memory:", ClockFirst())
    populate(store)
    orch, _transport, _fake_store, _extractor = _build_mem([])
    orch.store, orch.extractor = store, None
    llm = RecordingLLM()
    orch.llm = llm
    orch._history = [Message("assistant", "You have no birthday plan. It is Friday.")]
    try:
        asyncio.run(orch._handle_utterance(QUERY, via="text"))
        context = "\n".join(m.content for m in llm.last_messages)
        assert "[birthday plans] the user wants to be sky-diving next month" in context
        assert CLOCK not in context and "Right now it is" not in context
        assert "You have no birthday plan. It is Friday." not in context
        assert "Here is the register." not in context
        assert llm.last_messages[-1] == Message("user", QUERY)
    finally:
        store.close()


def test_evidence_is_bounded_and_does_not_follow_an_anchor_across_days():
    store = SqliteMemoryStore(":memory:", ClockFirst())
    populate(store)
    try:
        store._conn.execute("UPDATE turns SET ts=? WHERE id=3",
                            ((datetime.now() + timedelta(days=1)).isoformat(),))
        assert len(store.recall_evidence(QUERY)) == 1
        assert store.recall_evidence(QUERY, char_budget=3) == []
        assert len(store.recall_evidence(QUERY, limit=1)) == 1
    finally:
        store.close()


def test_forgetting_cannot_reconstruct_a_detail_through_a_surviving_topic():
    store = SqliteMemoryStore(":memory:", ClockFirst())
    populate(store)
    try:
        store.forget("skydiving")
        assert PLAN in store.all_facts()
        assert store.recall_evidence(QUERY) == []
        store.append_turn(Message("user", "My birthday plan is dinner."))
        assert store.recall_evidence(QUERY) == [Message("user", "My birthday plan is dinner.")]
    finally:
        store.close()


def test_no_retained_topic_means_no_history_resurrection():
    store = SqliteMemoryStore(":memory:", ClockFirst())
    try:
        store.append_turn(Message("user", "My birthday plan is skydiving."))
        assert store.recall_evidence(QUERY) == []
    finally:
        store.close()


def test_personal_wording_does_not_steal_a_named_document_question():
    orch, _transport, _store, _extractor = _build_mem([])
    orch.extractor = None
    llm = RecordingLLM()
    orch.llm = llm
    passage = SimpleNamespace(text="Standard tuning is E A D G B E.", source="guitar.md",
                              corpus="guitar", distance=0.1)
    orch.corpus = SimpleNamespace(names=lambda: ["guitar"], search=lambda *a, **k: [passage])
    asyncio.run(orch._handle_utterance("what is my guitar tuning?", via="text"))
    context = "\n".join(m.content for m in llm.last_messages)
    assert "guitar.md" in context and "E A D G B E" in context
    assert "He is asking about his own stored information" not in context


def test_correction_does_not_guess_or_rewrite_protected_identity():
    protected = Fact(subject="identity", text="the user is guide-eyeing", confidence=1,
                     origin="core")
    unrelated = Fact(subject="activity", text="the user is guide-eyeingish", confidence=1)
    evidence = [Message("user", "That's not guide-eyeing, it's sky-diving.")]
    assert apply_explicit_corrections([protected, unrelated], evidence) == [protected, unrelated]
    assert apply_explicit_corrections([PLAN], [Message("user", "I meant skydiving.")]) == [PLAN]


@pytest.mark.parametrize("question", [QUERY, "What's my favorite color?",
    "When is my birthday?", "Do you remember my dog's name?"])
def test_specific_memory_questions_use_recall_mode(question):
    assert asks_personal_memory(question)


@pytest.mark.parametrize("text", ["What time is it?", "What is the date?",
    "What should I do for my birthday?", "I want to go skydiving.", "Open my files"])
def test_clock_advice_and_commands_are_not_personal_recall(text):
    assert not asks_personal_memory(text)
