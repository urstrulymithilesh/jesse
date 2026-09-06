"""Continuous conversation: wake once, talk freely, stand down on request.

The spec says he is always listening by default and goes quiet only when told. He
used to demand the wake word every single turn.
"""

from __future__ import annotations

import asyncio

from jesse.core.state import ConversationState
from jesse.orchestrator import Orchestrator, _asks_to_go_quiet

from tests.test_orchestrator import (END, SPEECH, STOP, WAKE, FakeTransport, FakeVad,
                                     FakeWake, TextSynth)
from tests.test_streaming import ScriptedLLM


class Says:
    """Whatever he said this turn."""

    def __init__(self, text="hello world"):
        self.text = text

    def transcribe(self, pcm):
        return self.text


def _orch(frames, transcriber=None):
    transport = FakeTransport(list(frames))
    orch = Orchestrator(
        transport=transport, wake=FakeWake(WAKE), stopword=FakeWake(STOP),
        vad=FakeVad(), transcriber=transcriber or Says(),
        llm=ScriptedLLM("Alright."), synthesizer=TextSynth())
    return orch, transport


# -- the quiet command (pure) -----------------------------------------------


def test_recognises_ways_of_asking_her_to_stop():
    for text in ("go to sleep", "stop listening", "go quiet", "that's all for now",
                 "we're done", "stand down", "be quiet"):
        assert _asks_to_go_quiet(text), text


def test_ordinary_talk_does_not_stand_her_down():
    for text in ("I need to stop working at five", "the baby is asleep",
                 "I'm done with this project", "quiet day today"):
        assert not _asks_to_go_quiet(text), text


# -- engagement --------------------------------------------------------------


def test_the_wake_word_engages_her():
    orch, _t = _orch([])
    assert not orch._engaged
    asyncio.run(orch._handle_frame(WAKE))
    assert orch._engaged
    assert orch.state is ConversationState.LISTENING


def test_a_turn_ends_listening_not_idle_once_engaged():
    """The point of the whole feature: no wake word for the follow-up."""
    orch, transport = _orch([WAKE, SPEECH, END])
    asyncio.run(orch.run())
    assert orch.state is ConversationState.LISTENING
    assert orch._engaged


def test_a_second_turn_needs_no_wake_word():
    """ONE wake word, TWO utterances. Each turn is awaited before the next, because
    frames arriving mid-THINKING are ignored by design."""
    orch, transport = _orch([])

    async def scenario():
        await orch._handle_frame(WAKE)            # the ONLY wake word
        for _ in range(2):
            await orch._handle_frame(SPEECH)
            await orch._handle_frame(END)
            await orch._turn_task
            assert orch.state is ConversationState.LISTENING

    asyncio.run(scenario())
    assert len(transport.spoken) == 2, "the follow-up turn never ran"


def test_going_quiet_returns_her_to_wake_word_mode():
    orch, transport = _orch([WAKE, SPEECH, END],
                            transcriber=Says("go to sleep"))
    asyncio.run(orch.run())
    assert not orch._engaged
    assert orch.state is ConversationState.IDLE


def test_after_going_quiet_speech_alone_does_nothing():
    orch, transport = _orch([WAKE, SPEECH, END, SPEECH, END],
                            transcriber=Says("stop listening"))
    asyncio.run(orch.run())
    # He said goodbye to the first turn; the later speech must NOT start a turn.
    assert len(transport.spoken) == 1
    assert orch.state is ConversationState.IDLE


def test_a_new_wake_word_re_engages_her():
    orch, _t = _orch([])

    async def scenario():
        await orch._handle_frame(WAKE)
        orch._engaged = False
        orch._enter(ConversationState.IDLE)
        await orch._handle_frame(WAKE)

    asyncio.run(scenario())
    assert orch._engaged and orch.state is ConversationState.LISTENING


# -- the timeout, which is deliberately not infinite -------------------------


def test_engaged_uses_the_long_window_not_the_short_one():
    """A pause mid-conversation is normal; 8 seconds would drop his out constantly."""
    orch, _t = _orch([])
    orch._listen_timeout_frames = 2
    orch._continuous_timeout_frames = 50

    async def scenario():
        await orch._handle_frame(WAKE)            # engages his
        for _ in range(10):                       # well past the SHORT window
            await orch._handle_frame(b"quiet")

    asyncio.run(scenario())
    assert orch.state is ConversationState.LISTENING, "dropped out on the short window"


def test_she_does_eventually_stand_down_when_ignored():
    """Not infinite on purpose: a false VAD trigger would otherwise run forever, and a
    mic that never closes is a privacy problem."""
    orch, _t = _orch([])
    orch._continuous_timeout_frames = 3

    async def scenario():
        await orch._handle_frame(WAKE)
        for _ in range(6):
            await orch._handle_frame(b"quiet")

    asyncio.run(scenario())
    assert orch.state is ConversationState.IDLE
    assert not orch._engaged, "must need the wake word again after standing down"


# -- the starvation guard still holds over long stretches --------------------


def test_both_detectors_stay_fed_through_a_long_listening_stretch():
    """Continuous mode means LISTENING for minutes at a time. If the detectors only
    got frames in their 'own' state they would go cold — the bug that made barge-in
    silently stop working."""
    wake, stop = FakeWake(WAKE), FakeWake(STOP)
    seen = {"wake": 0, "stop": 0}
    wake.process = lambda f, _w=wake: (seen.__setitem__("wake", seen["wake"] + 1),
                                       f == WAKE)[1]
    stop.process = lambda f, _s=stop: (seen.__setitem__("stop", seen["stop"] + 1),
                                       f == STOP)[1]
    transport = FakeTransport([])
    orch = Orchestrator(transport=transport, wake=wake, stopword=stop, vad=FakeVad(),
                        transcriber=Says(), llm=ScriptedLLM("ok."),
                        synthesizer=TextSynth())
    orch._continuous_timeout_frames = 10_000

    async def scenario():
        await orch._handle_frame(WAKE)
        for _ in range(200):                      # a long quiet stretch while engaged
            await orch._handle_frame(b"quiet")

    asyncio.run(scenario())
    assert seen["wake"] == 201 and seen["stop"] == 201, \
        "a detector was starved during continuous listening"
