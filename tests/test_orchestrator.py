"""End-to-end tests of the walking-skeleton loop, driven by fakes.

The orchestrator is component-agnostic, so we can exercise the FULL state machine
(wake -> listen -> think -> speak, plus stop-word interrupt and alert timing) with
zero audio hardware and zero models. Async tests are wrapped in asyncio.run so they
need only plain pytest.
"""

from __future__ import annotations

import asyncio

from jesse.audio.vad import Vad
from jesse.core.interfaces import Fact, LLMError
from jesse.core.state import ConversationState
from jesse.llm.echo import EchoLLM
from jesse.orchestrator import Orchestrator

WAKE, SPEECH, END, STOP = b"WAKE", b"speech", b"END", b"STOP"


class FakeTransport:
    def __init__(self, frames: list[bytes]):
        self._frames = frames
        self.played: list[list[bytes]] = []
        self.mute_calls: list[str] = []

    async def capture(self):
        for f in self._frames:
            await asyncio.sleep(0)
            yield f

    async def play(self, frames, *, sample_rate: int = 16000):
        chunks = []
        for c in frames:
            chunks.append(c)
            await asyncio.sleep(0)
        self.played.append(chunks)

    def mute_input(self):
        self.mute_calls.append("mute")

    def unmute_input(self):
        self.mute_calls.append("unmute")

    @property
    def spoken(self) -> list[str]:
        return [b"".join(cl).decode() for cl in self.played]


class FakeWake:
    def __init__(self, trigger: bytes):
        self._trigger = trigger

    def process(self, frame: bytes) -> bool:
        return frame == self._trigger


class FakeVad(Vad):
    def is_speech(self, frame: bytes) -> bool:
        return frame != END

    def is_endpoint(self, frame: bytes) -> bool:
        return frame == END

    def reset(self) -> None:
        pass


class FakeTranscriber:
    def transcribe(self, pcm: bytes) -> str:
        return "hello world"


class TextSynth:
    """Yields the reply text as a single chunk so tests can assert what was spoken."""

    sample_rate = 16000

    def synthesize(self, text: str):
        yield text.encode()


def _build(frames: list[bytes]) -> tuple[Orchestrator, FakeTransport]:
    transport = FakeTransport(frames)
    orch = Orchestrator(
        transport=transport, wake=FakeWake(WAKE), stopword=FakeWake(STOP),
        vad=FakeVad(), transcriber=FakeTranscriber(), llm=EchoLLM(), synthesizer=TextSynth(),
    )
    return orch, transport


# -- full loop --------------------------------------------------------------


class RecordingTranscriber:
    """Captures the exact audio bytes handed to STT so we can assert what was sent."""

    def __init__(self):
        self.last = b""

    def transcribe(self, pcm: bytes) -> str:
        self.last = pcm
        return "hello world"


def test_preroll_is_prepended_so_sentence_start_is_not_lost():
    # Two frames of speech arrive JUST before the wake fires (the detection-latency gap).
    transport = FakeTransport([b"PRE1", b"PRE2", WAKE, SPEECH, END])
    rec = RecordingTranscriber()
    orch = Orchestrator(
        transport=transport, wake=FakeWake(WAKE), stopword=FakeWake(STOP),
        vad=FakeVad(), transcriber=rec, llm=EchoLLM(), synthesizer=TextSynth(),
        preroll_frames=4,
    )
    asyncio.run(orch.run())
    # the pre-wake audio made it into the captured turn
    assert b"PRE1" in rec.last and b"PRE2" in rec.last


def test_happy_path_full_loop():
    orch, transport = _build([WAKE, SPEECH, END])
    asyncio.run(orch.run())

    # visited the whole cycle and returned to idle
    seq = [s for s in orch.states_visited]
    for expected in (ConversationState.LISTENING, ConversationState.THINKING,
                     ConversationState.SPEAKING):
        assert expected in seq
    # Continuous mode: the wake engaged his, so the turn ends LISTENING for a
    # follow-up rather than making his say the wake word again.
    assert orch.state is ConversationState.LISTENING
    assert orch._engaged
    assert transport.spoken == ["I heard you say: hello world"]


def test_no_wake_no_turn():
    orch, transport = _build([SPEECH, SPEECH])
    asyncio.run(orch.run())
    assert orch.state in (ConversationState.IDLE, ConversationState.LISTENING)
    assert transport.spoken == []  # never woke, never spoke


# -- stop-word interrupt (Finding #3) --------------------------------------


def test_interruptible_stops_when_flagged():
    orch, _ = _build([])
    orch._interrupt.set()
    assert list(orch._interruptible(iter([b"a", b"b", b"c"]))) == []


def test_interruptible_passes_through_when_clear():
    orch, _ = _build([])
    assert list(orch._interruptible(iter([b"a", b"b"]))) == [b"a", b"b"]


def test_stopword_during_speaking_sets_interrupt():
    orch, _ = _build([])
    orch._enter(ConversationState.SPEAKING)
    asyncio.run(orch._handle_frame(STOP))
    assert orch._interrupt.is_set()


# -- alert timing (pending-alert overlay) ----------------------------------


def test_alert_while_idle_is_spoken_now():
    orch, transport = _build([b"x"])  # one non-wake frame
    orch.notify("your timer is done")
    asyncio.run(orch.run(max_frames=1))
    assert "your timer is done" in transport.spoken


class BrokenLLM:
    supports_tools = False

    def chat(self, messages, *, stream=True):
        raise LLMError("simulated Ollama 500")


class MarkdownLLM:
    """Simulates a 3B model slipping into a bulleted, markdown reply."""

    supports_tools = False

    def chat(self, messages, *, stream=True):
        yield "Here's the plan:\n- first thing\n- second thing\n**all set**"


def test_reply_is_cleaned_for_speech_before_playback():
    orch, transport = _build([WAKE, SPEECH, END])
    orch.llm = MarkdownLLM()
    asyncio.run(orch.run())
    spoken = transport.spoken[-1]
    assert "\n" not in spoken and "*" not in spoken   # voice-shaped, not markdown
    assert not spoken.split()[0].startswith("-")
    assert "first thing" in spoken and "all set" in spoken


def test_brain_failure_does_not_hang_and_speaks_error():
    orch, transport = _build([WAKE, SPEECH, END])
    orch.llm = BrokenLLM()
    asyncio.run(orch.run())
    # did not stall in THINKING/SPEAKING — recovered and is listening again
    assert orch.state in (ConversationState.IDLE, ConversationState.LISTENING)
    # spoke a clear apology instead of a reply, and left no dangling user turn
    assert transport.spoken and "went wrong" in transport.spoken[-1].lower()
    assert [m for m in orch._history if m.role == "user"] == []


# -- Phase 2: idle-gap extraction + overlap gating -------------------------


class FakeStore:
    def __init__(self):
        self.facts = []
        self.turns = []
        self.recall_facts = []   # what recall() returns (set per test)
        self.processed = set()   # turn ids whose extraction completed
        self.pending = []        # exchanges unprocessed_exchanges() hands back

    def add_fact(self, f):
        self.facts.append(f)

    def append_turn(self, m):
        self.turns.append(m)
        return len(self.turns)

    def recall(self, q, *, k=3, include_history=False):
        return self.recall_facts[:k]

    def recent(self, *, limit=20):
        return []

    def mark_processed(self, ids):
        self.processed.update(ids)

    def unprocessed_exchanges(self, *, limit=5):
        return self.pending[:limit]


class RecordingLLM:
    supports_tools = False

    def __init__(self):
        self.last_messages = []

    def chat(self, messages, *, stream=True):
        self.last_messages = list(messages)
        yield "sure thing"


class FakeExtractor:
    def __init__(self, out="[]"):
        self.out = out
        self.calls = 0

    def extract(self, exchange):
        self.calls += 1
        return self.out


def _build_mem(frames, extractor_out="[]"):
    transport = FakeTransport(frames)
    store, extractor = FakeStore(), FakeExtractor(extractor_out)
    orch = Orchestrator(
        transport=transport, wake=FakeWake(WAKE), stopword=FakeWake(STOP),
        vad=FakeVad(), transcriber=FakeTranscriber(), llm=EchoLLM(), synthesizer=TextSynth(),
        store=store, extractor=extractor,
    )
    return orch, transport, store, extractor


def test_extraction_stores_facts_and_persists_turns():
    out = '[{"subject":"sister","text":"the user\'s sister is named Anya","confidence":0.9}]'
    orch, _t, store, extractor = _build_mem([WAKE, SPEECH, END], out)
    asyncio.run(orch.run())
    assert extractor.calls == 1
    assert any("Anya" in f.text for f in store.facts)
    assert [m.role for m in store.turns] == ["user", "assistant"]   # exchange persisted


def test_malformed_extraction_stores_nothing_but_still_persists_turns():
    orch, _t, store, _e = _build_mem([WAKE, SPEECH, END], "this is not json")
    asyncio.run(orch.run())
    assert store.facts == []
    assert len(store.turns) == 2


def test_new_turn_cancels_pending_extraction_cleanly():
    orch, *_ = _build_mem([])

    async def scenario():
        never = asyncio.Event()
        orch._extract_task = asyncio.create_task(never.wait())  # pretend extraction in flight
        await asyncio.sleep(0)
        orch._begin_listening()                                  # a new turn begins
        await asyncio.sleep(0)
        assert orch._extract_task.cancelled()                   # cancelled, no crash
        assert orch.state is ConversationState.LISTENING        # new turn proceeds

    asyncio.run(scenario())


def test_extraction_skipped_when_a_turn_is_actively_running():
    orch, _t, store, extractor = _build_mem([])
    orch._enter(ConversationState.THINKING)   # a turn is genuinely mid-flight
    asyncio.run(orch._extract_facts("The user said: hi\nYou replied: hey"))
    # acquired the lock, saw a live turn, and bailed before touching Ollama or the store
    assert extractor.calls == 0
    assert store.facts == []


def test_extraction_still_runs_while_merely_listening():
    """Continuous mode keeps his LISTENING between turns. Gating extraction on IDLE
    meant it never ran again once he stopped idling."""
    orch, _t, store, extractor = _build_mem([])
    orch._enter(ConversationState.LISTENING)  # waiting to hear, not mid-turn
    asyncio.run(orch._extract_facts("The user said: hi\nYou replied: hey"))
    assert extractor.calls == 1


# -- Phase 2 step 3: retrieval into the turn context -----------------------


def test_recalled_fact_reaches_the_llm_context():
    orch, _t, store, _e = _build_mem([WAKE, SPEECH, END])
    store.recall_facts = [Fact(text="the user's sister is named Anya", confidence=0.9,
                               subject="sister's name")]
    rec = RecordingLLM()
    orch.llm = rec
    asyncio.run(orch.run())
    # the recalled fact was injected into the messages handed to the LLM
    assert any(m.role == "system" and "Anya" in m.content for m in rec.last_messages)
    # and the current user message is still the last thing in the list
    assert rec.last_messages[-1].role == "user"


def test_asks_about_past_detects_history_questions():
    from jesse.orchestrator import _asks_about_past
    assert _asks_about_past("how were you before?")
    assert _asks_about_past("what did you used to be like")
    assert _asks_about_past("you've come a long way, huh")
    assert not _asks_about_past("what's my sister's name?")
    assert not _asks_about_past("how are you today")


def test_alert_during_listening_waits_until_after_reply():
    orch, transport = _build([])

    async def scenario():
        await orch._handle_frame(WAKE)      # -> LISTENING
        orch.notify("gym at 5pm")           # fires while user is talking
        await orch._handle_frame(SPEECH)
        await orch._handle_frame(END)       # -> starts the turn
        assert orch._turn_task is not None
        await orch._turn_task

    asyncio.run(scenario())
    # reply first, alert second — never cut the user off
    assert transport.spoken == ["I heard you say: hello world", "gym at 5pm"]


# -- extraction survives interruption (turn marked processed only on completion) ----


def _mem_orch(frames, extractor_out="[]"):
    """An orchestrator wired to fake memory, like the other extraction tests."""
    transport = FakeTransport(frames)
    store = FakeStore()
    extractor = FakeExtractor(extractor_out)
    orch = Orchestrator(
        transport=transport, wake=FakeWake(WAKE), stopword=FakeWake(STOP),
        vad=FakeVad(), transcriber=FakeTranscriber(), llm=EchoLLM(), synthesizer=TextSynth(),
        store=store, extractor=extractor,
    )
    return orch, transport, store, extractor


FACT_JSON = '[{"subject":"dog","text":"the user has a dog named Rex","confidence":0.9}]'


def test_completed_extraction_marks_its_turns_processed():
    orch, _t, store, _e = _mem_orch([WAKE, SPEECH, END], FACT_JSON)
    asyncio.run(orch.run())
    # both turns of the exchange are flagged, so it is never re-extracted
    assert store.processed == {1, 2}
    assert len(store.facts) == 1


def test_cancelled_extraction_leaves_turns_unprocessed():
    orch, _t, store, _e = _mem_orch([], FACT_JSON)

    async def scenario():
        started = asyncio.Event()

        def slow_extract(exchange):
            started.set()
            raise AssertionError("should be cancelled before running")

        orch.extractor.extract = slow_extract
        task = asyncio.create_task(orch._extract_facts("x", turn_ids=(1, 2)))
        await asyncio.sleep(0)      # let it reach the await point
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
    # nothing marked -> the exchange stays queued for the next catch-up
    assert store.processed == set()
    assert store.facts == []


def test_catch_up_extracts_a_pending_exchange_at_startup():
    orch, _t, store, extractor = _mem_orch([], FACT_JSON)
    store.pending = [(1, 2, "my dog is Rex", "Rex, nice name")]
    asyncio.run(orch.run())          # no frames; catch-up still runs
    assert extractor.calls == 1
    assert len(store.facts) == 1 and "Rex" in store.facts[0].text
    assert store.processed == {1, 2}  # now flagged, so it won't repeat


def test_catch_up_is_a_noop_when_nothing_is_pending():
    orch, _t, _s, extractor = _mem_orch([], FACT_JSON)
    asyncio.run(orch.run())
    assert extractor.calls == 0


# -- Phase 3: scheduling from speech, respecting preemption -----------------


class FakeScheduler:
    def __init__(self):
        self.added = []

    def add(self, task, fire_at, *, is_timer=False, label=""):
        self.added.append((task, fire_at, is_timer, label))
        return len(self.added)

    def pending(self):
        return []

    async def run(self):
        await asyncio.Event().wait()   # idle like the real tick loop; cancelled on exit


class SchedulingTranscriber:
    """Feeds a fixed utterance so we can drive the scheduling path."""

    def __init__(self, text):
        self.text = text

    def transcribe(self, pcm):
        return self.text


def _sched_orch(utterance):
    transport = FakeTransport([WAKE, SPEECH, END])
    sched = FakeScheduler()
    orch = Orchestrator(
        transport=transport, wake=FakeWake(WAKE), stopword=FakeWake(STOP),
        vad=FakeVad(), transcriber=SchedulingTranscriber(utterance),
        llm=EchoLLM(), synthesizer=TextSynth(), scheduler=sched,
    )
    return orch, transport, sched


def test_speaking_a_timer_request_schedules_it():
    orch, _t, sched = _sched_orch("set a timer for 10 minutes")
    asyncio.run(orch.run())
    assert len(sched.added) == 1
    task, fire_at, is_timer, label = sched.added[0]
    assert is_timer is True
    assert label == "10 minutes"   # recorded so "cancel the 10 minute one" can find it


def test_speaking_a_reminder_request_keeps_the_task():
    orch, _t, sched = _sched_orch("remind me to stretch in 20 minutes")
    asyncio.run(orch.run())
    assert len(sched.added) == 1
    assert "stretch" in sched.added[0][0]


def test_ordinary_talk_schedules_nothing():
    orch, _t, sched = _sched_orch("I had a rough day today")
    asyncio.run(orch.run())
    assert sched.added == []


def test_a_fired_reminder_never_cuts_the_user_off_mid_sentence():
    """Preemption rule: notify() during LISTENING is held until he's back at IDLE."""
    orch, transport, _s = _sched_orch("hello there")

    async def scenario():
        await orch._handle_frame(WAKE)        # -> LISTENING (he starts talking)
        orch.notify("your timer is up")       # reminder fires mid-utterance
        await orch._handle_frame(SPEECH)
        assert transport.spoken == []         # did NOT talk over his
        await orch._handle_frame(END)
        await orch._turn_task

    asyncio.run(scenario())
    # his reply came first, the reminder second — interrupting, but politely
    assert len(transport.spoken) == 2
    assert "your timer is up" == transport.spoken[-1]


def test_asking_about_timers_reports_them_and_creates_nothing():
    """The query path must never schedule — and must tell him the real list."""
    from datetime import datetime, timedelta

    class QueryScheduler(FakeScheduler):
        def __init__(self, items):
            super().__init__()
            self._items = items

        def pending(self):
            return self._items

    class Item:
        def __init__(self, task, mins):
            self.task, self.fire_at = task, datetime.now() + timedelta(minutes=mins)

    transport = FakeTransport([WAKE, SPEECH, END])
    sched = QueryScheduler([Item("go to the gym", 60)])
    llm = RecordingLLM()
    orch = Orchestrator(
        transport=transport, wake=FakeWake(WAKE), stopword=FakeWake(STOP),
        vad=FakeVad(), transcriber=SchedulingTranscriber("do I have any timers running?"),
        llm=llm, synthesizer=TextSynth(), scheduler=sched,
    )
    asyncio.run(orch.run())

    assert sched.added == []                       # a question creates nothing
    hint = " ".join(m.content for m in llm.last_messages if m.role == "system")
    assert "go to the gym" in hint                 # he was handed the real pending item


def test_asking_with_nothing_pending_says_none():
    transport = FakeTransport([WAKE, SPEECH, END])
    llm = RecordingLLM()
    orch = Orchestrator(
        transport=transport, wake=FakeWake(WAKE), stopword=FakeWake(STOP),
        vad=FakeVad(), transcriber=SchedulingTranscriber("any reminders?"),
        llm=llm, synthesizer=TextSynth(), scheduler=FakeScheduler(),
    )
    asyncio.run(orch.run())
    hint = " ".join(m.content for m in llm.last_messages if m.role == "system")
    assert "NONE" in hint and "don't invent" in hint
