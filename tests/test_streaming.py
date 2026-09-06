"""Streaming TTS: segmentation, the generate-while-speaking bridge, the held last
sentence, and barge-in across a now multi-sentence reply.

The interrupt cases matter most. Speaking used to be one blob; it is now several
plays with generation still running underneath, so there are more places an interrupt
could be missed or a lock could leak. Every case here asserts the same two invariants:
the state returns to IDLE, and the LLM lock is released.
"""

from __future__ import annotations

import asyncio
import time

from jesse.core.interfaces import LLMError
from jesse.core.state import ConversationState
from jesse.orchestrator import Orchestrator
from jesse.tts.sentences import split_complete_sentences

from tests.test_orchestrator import (END, SPEECH, STOP, WAKE, FakeTransport, FakeVad,
                                     FakeWake, TextSynth)


# -- segmentation (pure) ----------------------------------------------------


def test_splits_on_terminators_and_keeps_the_partial():
    done, rest = split_complete_sentences("Hey you. Good to hear your voice. Still typ")
    assert done == ["Hey you.", "Good to hear your voice."]
    assert rest == "Still typ"


def test_a_sentence_is_only_complete_once_whitespace_follows():
    """Mid-stream "3." could still become "3.5", so we wait for the space."""
    done, rest = split_complete_sentences("It costs 3.")
    assert done == [] and rest == "It costs 3."
    done, rest = split_complete_sentences("It costs 3.5 dollars. And ")
    assert done == ["It costs 3.5 dollars."]


def test_does_not_split_inside_an_abbreviation():
    done, _rest = split_complete_sentences("Dr. Smith called. ")
    assert done == ["Dr. Smith called."]


def test_keeps_multi_character_terminators_together():
    done, _rest = split_complete_sentences("Really?! That's wild. ")
    assert done == ["Really?!", "That's wild."]


def test_empty_and_plain_text_are_safe():
    assert split_complete_sentences("") == ([], "")
    assert split_complete_sentences("no terminator yet") == ([], "no terminator yet")


# -- the streaming bridge ---------------------------------------------------


class ScriptedLLM:
    """Emits tokens one at a time, like a real streaming model."""

    supports_tools = False

    def __init__(self, text: str, *, on_token=None, delay: float = 0.0):
        self._tokens = [w + " " for w in text.split(" ")]
        self._on_token = on_token
        self._delay = delay
        self.tokens_emitted = 0

    def chat(self, messages, *, stream=True):
        for token in self._tokens:
            if self._delay:
                time.sleep(self._delay)   # a real model is not instant; without this the
                                          # producer thread can outrun the consumer and
                                          # any interleaving assertion becomes a race
            self.tokens_emitted += 1
            if self._on_token:
                self._on_token(self)
            yield token


def _orch(llm, frames=(), **kw):
    transport = FakeTransport(list(frames))
    orch = Orchestrator(
        transport=transport, wake=FakeWake(WAKE), stopword=FakeWake(STOP),
        vad=FakeVad(), transcriber=None, llm=llm, synthesizer=TextSynth(), **kw)
    return orch, transport


def _assert_clean(orch):
    assert orch.state is ConversationState.IDLE, "must return to idle"
    assert not orch._llm_lock.locked(), "the LLM lock must always be released"


def test_each_sentence_is_spoken_separately_as_it_becomes_ready():
    llm = ScriptedLLM("First thing. Second thing. Third thing.")
    orch, transport = _orch(llm)
    said = asyncio.run(orch._think_and_speak([]))

    assert transport.spoken == ["First thing.", "Second thing.", "Third thing."]
    assert said == "First thing. Second thing. Third thing."
    _assert_clean(orch)


def test_a_single_sentence_reply_still_works():
    orch, transport = _orch(ScriptedLLM("Just the one."))
    said = asyncio.run(orch._think_and_speak([]))
    assert transport.spoken == ["Just the one."] and said == "Just the one."
    _assert_clean(orch)


def test_speaking_starts_before_generation_finishes():
    """The whole point: sentence one is in the air while later tokens are still coming."""
    progress = {}

    def watch(llm):
        if "spoke_first_at" not in progress and llm.orch.state is ConversationState.SPEAKING:
            progress["spoke_first_at"] = llm.tokens_emitted

    llm = ScriptedLLM("One. Two. Three. Four. Five. Six.", on_token=watch, delay=0.02)
    orch, _t = _orch(llm)
    llm.orch = orch
    asyncio.run(orch._think_and_speak([]))

    assert "spoke_first_at" in progress, "he never started speaking mid-generation"
    assert progress["spoke_first_at"] < llm.tokens_emitted, \
        "speaking only began after every token — that is not streaming"


# -- the held last sentence (trimmer fix) -----------------------------------


def test_a_trailing_question_can_still_be_trimmed_after_streaming():
    """The reason the last sentence is held: it must be judged once the reply is
    complete, and a streamed sentence would already have been spoken."""
    orch, transport = _orch(ScriptedLLM("That sounds lovely. How was your day?"))
    orch.__dict__["_history"] = []
    said = asyncio.run(orch._think_and_speak([]))

    assert transport.spoken[0] == "That sounds lovely."
    # keep_rate is random, so either outcome is valid — but they must AGREE
    if len(transport.spoken) == 1:
        assert "How was your day?" not in said        # dropped, and never spoken
    else:
        assert transport.spoken[1] == "How was your day?"
        assert said.endswith("How was your day?")
    _assert_clean(orch)


def test_an_all_question_reply_is_never_gutted():
    orch, transport = _orch(ScriptedLLM("What happened? Are you alright?"))
    asyncio.run(orch._think_and_speak([]))
    assert transport.spoken == ["What happened?", "Are you alright?"]


# -- barge-in across a multi-sentence reply ---------------------------------


def test_interrupt_before_any_sentence_lands_speaks_nothing():
    """He cuts in while he is still forming the first sentence.

    (A stale interrupt from the PREVIOUS reply can't do this — _think_and_speak
    clears the flag on entry, or one barge-in would mute every reply after it.)
    """
    llm = ScriptedLLM("One. Two. Three. Four. Five.")
    orch, transport = _orch(llm)
    llm._on_token = lambda model: orch._interrupt.set() if model.tokens_emitted == 1 else None

    said = asyncio.run(orch._think_and_speak([]))
    assert transport.spoken == [], "nothing should be spoken after an interrupt"
    assert said == ""
    _assert_clean(orch)


def test_interrupt_mid_reply_keeps_what_was_already_said():
    """Interrupt arrives while he is speaking; earlier sentences stand, later ones stop."""
    llm = ScriptedLLM("One. Two. Three. Four. Five. Six. Seven.")
    orch, transport = _orch(llm)

    original = orch._speak_sentence

    async def speak_then_interrupt(text):
        await original(text)
        if len(transport.spoken) == 2:
            orch._interrupt.set()      # he cuts in after the second sentence
    orch._speak_sentence = speak_then_interrupt

    said = asyncio.run(orch._think_and_speak([]))
    assert transport.spoken[:2] == ["One.", "Two."]
    assert len(transport.spoken) < 7, "later sentences should not have been spoken"
    assert said.startswith("One. Two.")
    _assert_clean(orch)


def test_interrupt_while_generation_is_still_running_stops_the_model_early():
    """The producer checks the flag per token, so an interrupt halts generation too."""
    llm = ScriptedLLM(" ".join(f"Sentence {i}." for i in range(60)))
    orch, _t = _orch(llm)

    def stop_early(model):
        if model.tokens_emitted == 5:
            orch._interrupt.set()
    llm._on_token = stop_early

    asyncio.run(orch._think_and_speak([]))
    assert llm.tokens_emitted < 60, "generation should stop once interrupted"
    _assert_clean(orch)


def test_a_brain_failure_mid_stream_is_surfaced_not_swallowed():
    """It happens in a worker thread now; it must still reach the caller."""
    class Broken:
        supports_tools = False

        def chat(self, messages, *, stream=True):
            yield "This part is fine. "
            raise LLMError("stream died")

    orch, _t = _orch(Broken())
    try:
        asyncio.run(orch._think_and_speak([]))
        raised = False
    except LLMError:
        raised = True
    assert raised, "a mid-stream failure must propagate so the turn can apologise"
    _assert_clean(orch)


def test_the_lock_is_free_for_extraction_once_the_reply_ends():
    """Overlap gating: extraction waits for generation, and generation really ends."""
    orch, _t = _orch(ScriptedLLM("All done. Bye."))
    asyncio.run(orch._think_and_speak([]))
    assert not orch._llm_lock.locked()


# -- what the fakes missed: detector starvation and barge-in recovery --------


class WarmingWake:
    """A wake detector that behaves like the REAL one: it needs a run of recent
    frames before it can fire, because openWakeWord builds its mel/embedding buffers
    over about a second of CONTINUOUS audio.

    The stateless FakeWake ("frame == trigger") can never show the starvation bug —
    which is exactly why the unit tests passed while live barge-in went silent.
    """

    WARMUP = 5

    def __init__(self, trigger):
        self._trigger = trigger
        self.frames_seen = 0
        self._since_gap = 0

    def process(self, frame):
        self.frames_seen += 1
        self._since_gap += 1
        if frame != self._trigger:
            return False
        return self._since_gap >= self.WARMUP      # cold buffer -> missed wake

    def starve(self):
        """Simulate the detector receiving nothing for a while."""
        self._since_gap = 0


def test_both_detectors_are_fed_every_frame_whatever_the_state():
    """The starvation fix: neither model may go cold while the other is in use."""
    wake, stop = WarmingWake(WAKE), WarmingWake(STOP)
    transport = FakeTransport([b"a", b"b", b"c"])
    orch = Orchestrator(
        transport=transport, wake=wake, stopword=stop, vad=FakeVad(),
        transcriber=None, llm=ScriptedLLM("x."), synthesizer=TextSynth())

    async def feed():
        for st in (ConversationState.IDLE, ConversationState.SPEAKING,
                   ConversationState.LISTENING):
            orch._enter(st)
            await orch._handle_frame(b"filler")

    asyncio.run(feed())
    assert wake.frames_seen == 3, "the wake detector went cold in some state"
    assert stop.frames_seen == 3, "the stop detector went cold in some state"


def test_the_wake_word_still_fires_right_after_a_reply():
    """Regression for the live silent failure: interrupt, then wake again."""
    wake = WarmingWake(WAKE)
    transport = FakeTransport([])
    orch = Orchestrator(
        transport=transport, wake=wake, stopword=WarmingWake(STOP), vad=FakeVad(),
        transcriber=None, llm=ScriptedLLM("x."), synthesizer=TextSynth())

    async def scenario():
        orch._enter(ConversationState.SPEAKING)
        for _ in range(10):                       # a long reply plays
            await orch._handle_frame(b"his voice")
        orch._enter(ConversationState.IDLE)
        await orch._handle_frame(WAKE)            # he wakes his immediately after

    asyncio.run(scenario())
    assert orch.state is ConversationState.LISTENING, \
        "the wake word was missed after a reply — the detector had gone cold"


def test_a_barge_in_ends_the_turn_listening_not_idle():
    """He said the wake word to cut him off; that word is spent. Going idle would
    make his say it twice before he heard what he interrupted his to say."""
    orch, transport = _orch(ScriptedLLM("One. Two. Three. Four."))

    original = orch._speak_sentence

    async def speak_then_barge(text):
        await original(text)
        if len(transport.spoken) == 1:
            orch._barge_in = True                 # as _handle_frame would set it
            orch._interrupt.set()
    orch._speak_sentence = speak_then_barge

    async def scenario():
        await orch._think_and_speak([])
        # mimic the tail of _run_turn
        if orch._barge_in:
            orch._barge_in = False
            orch._begin_listening()

    asyncio.run(scenario())
    assert orch.state is ConversationState.LISTENING
    assert not orch._llm_lock.locked()


def test_the_stop_word_sets_barge_in_only_once():
    orch, _t = _orch(ScriptedLLM("x."))
    orch._enter(ConversationState.SPEAKING)
    asyncio.run(orch._handle_frame(STOP))
    assert orch._barge_in and orch._interrupt.is_set()


def test_a_wake_with_no_speech_gives_up_instead_of_listening_forever():
    """A false wake used to hang in LISTENING permanently — the VAD can't end a turn
    that never started, so from outside he looked dead."""
    orch, _t = _orch(ScriptedLLM("x."))
    orch._listen_timeout_frames = 3
    orch._continuous_timeout_frames = 3      # engaged by the wake; both windows short here

    class SilentVad(FakeVad):
        def is_endpoint(self, frame):
            return False                          # nothing ever ends the turn
    orch.vad = SilentVad()

    async def scenario():
        await orch._handle_frame(WAKE)
        assert orch.state is ConversationState.LISTENING
        for _ in range(5):
            await orch._handle_frame(b"silence")

    asyncio.run(scenario())
    assert orch.state is ConversationState.IDLE, "he should have gone back to sleep"


def test_each_sentence_is_printed_as_it_starts_playing(capsys):
    orch, _t = _orch(ScriptedLLM("First one. Second one."))
    asyncio.run(orch._think_and_speak([]))
    out = capsys.readouterr().out
    assert 'jesse: "First one."' in out
    assert 'jesse: "Second one."' in out
