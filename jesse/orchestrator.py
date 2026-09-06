"""The orchestrator — Jesse's always-on event loop and preemption state machine.

This is the signature engineering of the project (the reason we chose a custom
asyncio loop over a framework): the mic is consumed by ONE ingest loop, and a turn
(transcribe -> think -> speak) runs as a concurrent task so the ingest loop keeps
running DURING speech. That is what lets a stop-word barge in while Jesse is talking.

    ┌──── wake ────► LISTENING ──── endpoint ────► THINKING ──── reply ────┐
   IDLE                (buffer STT)                 (LLM)                    │
    ▲                                                                       ▼
    └────────── reply done / stop-word ──────────────────────────────── SPEAKING
                                                     (Piper/stub + stop-word live)

The orchestrator is COMPONENT-AGNOSTIC: it drives injected WakeWord / Vad /
Transcriber / LLM / Synthesizer / AudioTransport objects. That is what makes the
whole machine unit-testable today with fakes, and what makes swapping the TTS stub
for Piper (or Echo for Ollama) a factory change, not a rewrite.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from datetime import datetime
from collections.abc import Callable, Iterator

from jesse.core.interfaces import (
    AudioTransport,
    LLM,
    MemoryStore,
    Message,
    Synthesizer,
    Transcriber,
    WakeWord,
)
from jesse.config import CONFIG
from jesse.context import (build_messages, digest_context, episode_context,
                          knowledge_context, next_step_nudge, now_context,
                          self_state_context, shared_history_context)
from jesse.core.state import ConversationState, disposition_for
from jesse.audio.frames import SAMPLE_RATE, ms_to_chunks
from jesse.audio.vad import Vad
from jesse.memory.episodes import EpisodeStore, Summariser
from jesse.digest.feeds import FeedError, fetch_feed
from jesse.digest.parse import asks_whats_new
from jesse.memory.corpus import subjects_mentioned
from jesse.memory.extraction import FactExtractor, parse_extracted_facts
from jesse.actions.parse import (MediaCommand, OpenCommand, UnknownTarget,
                                looks_like_an_action, parse_action_command)
from jesse.actions.run import ActionError, find_files, media_key, open_target
from jesse.memory.temporal import parse_temporal_query
from jesse.persona import recall_prompt
from jesse.memory.forget_parse import parse_forget_command
from jesse.schedule.parse import (CancelCommand, IncompleteCommand, QueryCommand,
                                 RescheduleCommand, _phrase_delay,
                                 parse_schedule_command)
from jesse.reply_style import trim_reflexive_question
from jesse.stt.cleanup import strip_vocative, strip_wake_prefix
from jesse.tts.sentences import split_complete_sentences
from jesse.tts.speech_text import clean_for_speech

# Phrases that mean "tell me about your PAST self" — only then do we surface his
# self_history facts (so old-version details don't leak into normal conversation).
_PAST_PATTERNS = (
    "used to", "how were you", "how you were", "before", "previous version", "previously",
    "your past", "earlier version", "older version", "back then", "in the beginning",
    "when you started", "how you started", "how far you", "come a long way",
)


# Queue sentinel: generation has ended (distinct from any real sentence).
_GENERATION_DONE = object()

# A deterministic handler has already spoken and finished the turn; there is
# nothing left for the model to say.
_ANSWERED = object()


def _asks_about_past(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _PAST_PATTERNS)


# "How are you / what can you do / what version are you" — questions about HIMSELF.
# Only then do we spend context on the self-state block.
_SELF_PATTERNS = (
    "how are you", "how do you feel", "how're you", "how you doing", "how are things",
    "what can you do", "your abilities", "what do you do", "who are you", "introduce yourself",
    "your version", "your current version", "your state", "your build", "what are you",
    "your tech", "built you", "current version", "version are you", "your capabilities",
    "how you're built", "how you are built", "feeling",
)


# "tell me about us", "what do you remember about our time together". These retrieve
# nothing in particular, so without an anchor he invents a shared past.
_SHARED_HISTORY_PATTERNS = (
    "about us", "between us", "we have together", "our time together", "our history",
    "our relationship", "memories of us", "memory of us", "remember about us",
    "inside joke", "inside jokes", "how did we meet", "how we met", "what have we",
    "things we", "we been through", "our moments", "our memories", "you and i",
    "you and me", "us together",
)


# Explicit "stand down" — back to wake-word-required.
_QUIET_PATTERNS = (
    "go to sleep", "go quiet", "stop listening", "stand down", "that's all for now",
    "thats all for now", "stop for now", "be quiet", "sleep now", "goodnight jesse",
    "good night jesse", "night jesse", "leave me alone", "we're done", "were done",
    "that will be all",
    "stop the conversation", "end the conversation",
)


def _asks_to_go_quiet(text: str) -> bool:
    return any(p in text.lower() for p in _QUIET_PATTERNS)


def _asks_about_shared_history(text: str) -> bool:
    low = text.lower()
    if any(p in low for p in _SHARED_HISTORY_PATTERNS):
        return True
    # "what memories do you have" with no specific subject attached.
    return ("memories" in low or "memory" in low) and any(
        w in low for w in ("do you have", "do we", "your favourite", "your favorite",
                           "what are", "tell me"))


def _asks_about_self(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _SELF_PATTERNS) or _asks_about_past(text)


# "What should I do next?" — he's asking to be told the next step.
_NEXT_STEP_PATTERNS = (
    "what should i do", "what do i do", "what next", "what's next", "whats next",
    "what should we do", "what do we do", "where do i start", "what should i build",
    "what should i work on", "any ideas what", "what would you suggest",
)


def _asks_what_next(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _NEXT_STEP_PATTERNS)


# A short "yes" — the shape of an answer to his own clarifying question. Only these
# get the previous user turn attached to the retrieval query. Without this condition
# the attachment is a leak: after "my starter motor died" -> his ask -> "no, my car",
# the words "starter motor" rode along in the query and retrieved sourdough anyway.
# A declined ask must decay, not chase him.
_AFFIRMATIONS = frozenset((
    "yes", "yeah", "yep", "yup", "sure", "right", "exactly", "correct", "please",
    "ok", "okay", "aye", "definitely", "obviously", "that",
))


def _is_short_affirmation(text: str) -> bool:
    words = [w.strip(".,!?'") for w in text.lower().split()]
    return 0 < len(words) <= 6 and any(w in _AFFIRMATIONS for w in words)


class Orchestrator:
    def __init__(
        self,
        *,
        transport: AudioTransport,
        wake: WakeWord,
        stopword: WakeWord,
        vad: Vad,
        transcriber: Transcriber,
        llm: LLM,
        synthesizer: Synthesizer,
        system_prompt: str = "",
        preroll_frames: int = 0,
        store: MemoryStore | None = None,
        extractor: FactExtractor | None = None,
        episodes: EpisodeStore | None = None,
        corpus=None,
        digest=None,
        auto_read_sources: bool | None = None,
        text_channel=None,
        summariser: Summariser | None = None,
        scheduler=None,
        on_state_change: Callable[[ConversationState], None] | None = None,
    ) -> None:
        self.transport = transport
        self.wake = wake
        self.stopword = stopword
        self.vad = vad
        self.transcriber = transcriber
        self.llm = llm
        self.synth = synthesizer
        self.store = store              # None => memory disabled (e.g. Echo brain)
        self.extractor = extractor      # None => no fact extraction
        self.episodes = episodes        # None => no episodic memory
        self.corpus = corpus            # None => nothing learned from documents
        self.digest = digest            # None => he reads no sources
        # Whether the background fetch loop runs. An explicit parameter rather
        # than reading global config at start time: the smoke harness injects a
        # digest store of its own and must never touch the network, and when
        # this was implicit, enabling digests globally made the harness fetch
        # live news mid-scenario and answer from it.
        self.auto_read_sources = (CONFIG.digest.enabled
                                  if auto_read_sources is None else auto_read_sources)
        self.text_channel = text_channel  # None => no UI attached
        self.summariser = summariser
        self.scheduler = scheduler      # None => timers/reminders disabled
        self._on_state_change = on_state_change

        self.state = ConversationState.IDLE
        self.states_visited: list[ConversationState] = [ConversationState.IDLE]
        self._buffer = bytearray()
        # Rolling recent audio so the wake-word's detection latency doesn't eat the
        # start of the sentence — prepended to the turn buffer when we start listening.
        self._preroll: deque[bytes] = deque(maxlen=preroll_frames)
        self._interrupt = asyncio.Event()
        # Set when the stop-word cuts a reply short. He just said the wake word, so the
        # turn ends by LISTENING for what he actually wants rather than going idle and
        # making his say it twice.
        self._barge_in = False
        # True once woken: he keeps listening between turns until told to go quiet.
        self._engaged = False
        self._go_quiet = False
        self._listen_frames = 0
        self._listen_timeout_frames = ms_to_chunks(CONFIG.audio.listen_timeout_ms)
        self._continuous_timeout_frames = ms_to_chunks(CONFIG.audio.continuous_timeout_ms)
        self._turn_task: asyncio.Task[None] | None = None
        # Overlap gating: ONE lock guards every Ollama call (reply AND extraction), so
        # they can never hit the model/CPU at the same time. Extraction runs in the idle
        # gap as a background task and is cancelled the instant a new turn begins.
        self._llm_lock = asyncio.Lock()
        self._extract_task: asyncio.Task[None] | None = None
        self._catchup_task: asyncio.Task[None] | None = None
        self._summary_task: asyncio.Task[None] | None = None
        self._scheduler_task: asyncio.Task[None] | None = None
        self._digest_task: asyncio.Task[None] | None = None
        self._nudged = False        # the new-arrivals mention is once per session
        self._pending_action = None  # a side effect awaiting his yes, from the phone
        self._alerts: list[str] = []
        self._system_prompt = system_prompt
        self._history: list[Message] = []  # conversation turns only; persona + facts added per turn

    # -- public ------------------------------------------------------------

    async def run(self, *, max_frames: int | None = None) -> None:
        """Consume mic frames until the transport ends (or max_frames, for tests)."""
        n = 0
        # Catch up on any exchange whose extraction was cut short last time (you spoke
        # again, or quit, inside the extraction window). Backgrounded so it never
        # delays the mic loop.
        if self.store is not None and self.extractor is not None:
            self._catchup_task = asyncio.create_task(self._catch_up_extractions())
        # Fold anything left unsummarised by a previous session into an episode. Same
        # resume-after-a-crash pattern as extraction: turns are persisted first, so a
        # session that ended abruptly still becomes a memory.
        if self.episodes is not None and self.summariser is not None:
            self._summary_task = asyncio.create_task(self._summarise_pending())
        # Timers/reminders. The loop's first pass IS the startup reconcile: anything
        # that came due while he was closed (or the laptop slept) fires now.
        if self.scheduler is not None:
            self._scheduler_task = asyncio.create_task(self.scheduler.run())
        # Reading his sources. Backgrounded and SILENT: unlike the scheduler, this
        # loop can never speak. Its first pass is the reconcile — a machine that was
        # closed through the interval fetches once on waking, not once per missed one.
        if self.digest is not None and self.auto_read_sources:
            self._digest_task = asyncio.create_task(self._read_sources_loop())
        async for frame in self.transport.capture():
            await self._handle_frame(frame)
            n += 1
            if max_frames is not None and n >= max_frames:
                break
        if self._turn_task is not None:
            await self._turn_task
        # Close the session out as a memory before shutting down.
        if self.episodes is not None and self.summariser is not None:
            try:
                await self._summarise_pending()
            except Exception as e:               # noqa: BLE001 - never block shutdown
                print(f"  [memory] could not summarise this session: {e}")
        for loop in (self._scheduler_task, self._digest_task):
            if loop is not None:
                loop.cancel()                    # infinite loops; stop them on exit
        for task in (self._catchup_task, self._extract_task):
            if task is not None and not task.done():
                try:
                    await task               # let a final idle-gap extraction finish
                except asyncio.CancelledError:
                    pass

    def notify(self, text: str) -> None:
        """A fired timer/reminder. disposition_for() governs WHEN it's spoken;
        the queue is drained whenever we're safely back at IDLE (never cutting the
        user off mid-utterance)."""
        _ = disposition_for(self.state)  # documents intent; drain point enforces it
        self._alerts.append(text)

    # -- state transitions -------------------------------------------------

    def _enter(self, state: ConversationState) -> None:
        self.state = state
        self.states_visited.append(state)
        if self._on_state_change is not None:
            self._on_state_change(state)

    async def _handle_frame(self, frame: bytes) -> None:
        self._preroll.append(frame)  # always keep the most recent audio for pre-roll

        # Typed input rides the SAME loop as audio, so a message can arrive while the
        # voice loop is live. Ignored mid-turn: the reply in flight finishes first.
        if self.text_channel is not None and self.state in (ConversationState.IDLE,
                                                            ConversationState.LISTENING):
            typed = self.text_channel.take()
            if typed:
                print(f'  [ui] typed: "{typed}"')
                self._start_text_turn(typed)
                return

        # BOTH detectors see EVERY frame, whatever the state, and only the relevant
        # result is acted on. openWakeWord is a streaming model: it builds mel and
        # embedding buffers over roughly a second of CONTINUOUS audio. Feeding the wake
        # detector nothing during a ten-second reply left it stale, so the "hey jarvis"
        # right after a barge-in landed in its dead zone and was missed — which looked
        # exactly like he had died. Fakes can't show this; only a real model can.
        woke = self.wake.process(frame)
        stopped = woke if self.stopword is self.wake else self.stopword.process(frame)

        st = self.state
        if st is ConversationState.IDLE:
            if self._alerts:
                await self._speak(self._alerts.pop(0))
                return
            if woke:
                print("  [wake] heard the wake word"
                      + (" — staying awake until you tell me to stop"
                         if CONFIG.audio.continuous_mode and not self._engaged else ""))
                self._engaged = CONFIG.audio.continuous_mode
                self._begin_listening()
        elif st is ConversationState.LISTENING:
            self._buffer += frame
            self._listen_frames += 1
            if self.vad.is_endpoint(frame):
                self._start_turn()
            elif self._listen_frames > (self._continuous_timeout_frames if self._engaged
                                        else self._listen_timeout_frames):
                # Nothing said. Without this he waits forever, because the VAD cannot
                # end a turn that never started.
                print("  [listening] quiet for a while — going back to sleep"
                      if self._engaged else "  [listening] nothing said — going back to sleep")
                self._buffer = bytearray()
                self._engaged = False
                self._enter(ConversationState.IDLE)
        elif st is ConversationState.SPEAKING:
            # Half-duplex: full STT is gated, but the stop-word stays live.
            if stopped and not self._interrupt.is_set():
                print("  [interrupt] stop-word heard — cutting the reply short")
                self._barge_in = True
                self._interrupt.set()
        # THINKING: transient; frames are ignored while the LLM runs.

    def _begin_listening(self, *, interrupt_background: bool = True) -> None:
        """interrupt_background=False for the automatic post-reply transition in
        continuous mode. Cancelling there would kill the fact extraction that the turn
        just kicked off — every single turn — because he is now always listening
        rather than idling between turns."""
        # A new interaction takes priority: cancel any pending idle-gap extraction so it
        # can't compete with the coming reply. Best-effort — a lost extraction is fine.
        if interrupt_background:
            for task in (self._extract_task, self._catchup_task):
                if task is not None and not task.done():
                    task.cancel()   # safe: the turn stays unprocessed, retried later
        # Seed the turn with the pre-roll so the start of the sentence (spoken during
        # the wake word's detection latency) is included. Do NOT flush here — flushing
        # would drop exactly those queued start-of-speech frames.
        self._buffer = bytearray(b"".join(self._preroll))
        self._listen_frames = 0
        self.vad.reset()
        self._enter(ConversationState.LISTENING)

    def _start_turn(self) -> None:
        audio = bytes(self._buffer)
        self._buffer = bytearray()
        self._enter(ConversationState.THINKING)
        self._turn_task = asyncio.create_task(self._run_turn(audio))

    async def _run_turn(self, audio: bytes) -> None:
        """A SPOKEN turn: transcribe, then hand to the shared core."""
        secs = len(audio) / 2 / SAMPLE_RATE  # int16 mono @ 16k
        print(f"  [captured {secs:.1f}s of audio]")
        text = (await asyncio.to_thread(self.transcriber.transcribe, audio)).strip()
        # The pre-roll means the wake word itself is usually in the transcript, and
        # that prefix measurably breaks fact extraction on 3b. Strip it once, here.
        text = strip_wake_prefix(text, CONFIG.wake.model)
        # And a name-shaped word left in front of a comma, which the wake stripper
        # cannot touch because there is no wake token beside it. Live, "hey jarvis"
        # arrived as "Heeshak," and silently disabled every parser anchored to the
        # start of the utterance.
        text = strip_vocative(text)
        await self._handle_utterance(text, via="voice")

    def _start_text_turn(self, text: str) -> None:
        """A TYPED turn. Same core as speech, so the UI cannot drift into a second
        Jesse with his own memory."""
        self._enter(ConversationState.THINKING)
        self._turn_task = asyncio.create_task(self._handle_utterance(text, via="text"))

    async def _handle_utterance(self, text: str, *, via: str = "voice") -> None:
        appended_user = False
        try:
            if not text:
                print("  [transcript empty — heard no clear speech]")
                self._enter(ConversationState.IDLE)
                return
            print(f'  you: "{text}"')
            if self.text_channel is not None:
                self.text_channel.log("you", text, via=via)
            self._history.append(Message("user", text))
            appended_user = True
            facts = (
                self.store.recall(text, k=CONFIG.memory.recall_k,
                                  include_history=_asks_about_past(text))
                if self.store else []
            )
            if facts:
                print("  [memory] recalled " + "; ".join(f.subject or f.text[:30] for f in facts))
            if _asks_to_go_quiet(text):
                self._engaged = False
                self._go_quiet = True
                print("  [listening] told to stand down — wake word needed again")
            extra: list[Message] = [now_context()]     # he always knows the real time
            recall_mode = False
            if _asks_about_self(text):
                from jesse.memory.progress import latest, previous
                block = self_state_context(latest(), previous())
                if block is not None:
                    extra.append(block)
                    print(f"  [self] injected current state ({latest().version})")
            if self._go_quiet:
                extra.append(Message(
                    "system",
                    "He just asked you to stop listening. Say a short, warm goodbye in "
                    "ONE sentence and nothing else — no questions, no offers.",
                ))
            if _asks_what_next(text):
                extra.append(next_step_nudge())
                print("  [self] next-step question — deflecting to him")
            # Facts say what is TRUE; episodes say what HAPPENED and when. A
            # question about past conversation is routed to the record, not to
            # semantic fact recall, and is anchored to what genuinely exists.
            if self.episodes is not None:
                window = parse_temporal_query(text, now=datetime.now())
                if window is not None:
                    found = self.episodes.in_window(window)
                    if not found and window.start is None:
                        found = self.episodes.search(text, k=3)
                    extra.append(episode_context(found, window.label))
                    recall_mode = True
                    print(f"  [memory] temporal question ({window.label}) — "
                          f"{len(found)} episode(s) on record")
            # Anything he has had his read. The trigger is NAMING the subject, not the
            # distance — a pure distance gate inverted at six passages once a second
            # corpus existed (see subjects_mentioned). Recent turns count too, so
            # follow-ups work without repeating the word every time. Skipped in recall
            # mode: a question about last Tuesday wants the record, not a document.
            if self.corpus is not None and not recall_mode:
                names = self.corpus.names()
                named_now = subjects_mentioned(text, names)
                # _history already holds the current utterance; scan the turns before it.
                prior = self._history[-(CONFIG.knowledge.topic_turns + 1):-1]
                named_recent = subjects_mentioned(
                    " ".join(m.content for m in prior), names)
                named = sorted(set(named_now) | set(named_recent))
                query = text
                if named and not named_now and _is_short_affirmation(text):
                    # A bare "yes" — his answer to his own clarifying question. Those
                    # words alone cannot reach the passage, so the previous user turn
                    # (the original question) rides along as the query. ONLY on a short
                    # affirmation: on anything else the attachment leaks the old phrase
                    # into queries it has no business in.
                    prev = next((m.content for m in reversed(prior) if m.role == "user"), "")
                    query = f"{prev} {text}".strip()
                passages = self.corpus.search(
                    query, k=CONFIG.knowledge.top_k,
                    max_distance=CONFIG.knowledge.max_distance,
                    corpora=named) if named else []
                block = knowledge_context(
                    passages, char_budget=CONFIG.knowledge.char_budget)
                if block is not None:
                    extra.append(block)
                    # Same remedy as memory questions, and it was needed for the same
                    # reason: with the few-shot examples in play he answered a question
                    # the document did NOT cover by inventing string gauges, once
                    # attributing the invention to the file by name. Dropping them took
                    # that from 0/3 honest to 3/3.
                    recall_mode = True
                    print(f"  [knowledge] {len(passages)} passage(s) from "
                          f"{passages[0].corpus!r} (closest {passages[0].distance:.3f})")
                elif not named:
                    # No subject named, but his words overlap a document's own
                    # vocabulary ("strings", "starter"). Weaker evidence, so he asks
                    # instead of answering — cold-start recall was 3/12 on the name
                    # alone, and injecting on keywords is unsafe (collision phrases and
                    # real questions measure 0.005 apart). Asking puts the corpus name
                    # into the transcript, so a yes flows into normal retrieval above.
                    maybe = self.corpus.keyword_subjects(text)
                    hits = self.corpus.search(
                        text, k=1, max_distance=CONFIG.knowledge.max_distance,
                        corpora=maybe) if maybe else []
                    if hits:
                        # The ask is DETERMINISTIC — fixed words, no LLM turn. Probed
                        # the prompted version live first: a soft note answered from
                        # pretraining 3/3 ("every 3-4 months", invented), and a
                        # hardened note asked but only said the topic word 2/3 — and
                        # the resolution NEEDS that word in the transcript, because a
                        # bare "yes" resolves off his own mention of it. An ask whose
                        # output feeds a deterministic trigger is structural, and
                        # structural things in this project are not delegated to a 3B.
                        # Bonus: no round-trip, so the question lands in under a second.
                        topic = hits[0].corpus
                        print(f"  [knowledge] his words brush {topic!r} — asking, "
                              f"not answering (closest {hits[0].distance:.3f})")
                        ask = f"Are you asking about your {topic}?"
                        await self._speak(ask)
                        if self.text_channel is not None:
                            self.text_channel.log("jesse", ask)
                        self._history.append(Message("assistant", ask))
                        self._remember_turn(text, ask)
                        return
            # What he has read from his sources. Checked before the knowledge block
            # so "anything new?" is a digest question rather than a corpus search, and
            # recall mode for the usual reason: reciting a real list needs accuracy.
            if self.digest is not None:
                digest_note = await self._handle_digest_query(text)
                if digest_note is _ANSWERED:
                    return          # he has already read the headlines out
                if digest_note is not None:
                    extra.append(digest_note)
                    recall_mode = True
            if _asks_about_shared_history(text) and self.store is not None:
                extra.append(shared_history_context(self.store.all_facts()))
                recall_mode = True
                print("  [memory] broad 'about us' question — anchored to stored facts only")
            # Timers/reminders: parsed deterministically (no extra LLM round-trip),
            # scheduled immediately, then he confirms it in his own words.
            # Forget FIRST: the reminder parser treats "forget" + "that" as cancelling
            # a reminder, which would swallow "forget that my favourite colour is blue".
            # parse_forget_command bows out whenever an explicit timer/reminder word is
            # present, so the two never fight over the same sentence.
            forget_note = self._handle_forget_command(text)
            if forget_note is not None:
                extra.append(Message("system", forget_note))
            else:
                note = (self._handle_schedule_command(text)
                        if self.scheduler is not None else None)
                # Doing things on the computer comes LAST of the deterministic parsers:
                # "remind me to open the report at five" is a reminder, and the action
                # parser bows out on reminder words anyway, so neither can steal the
                # other's sentence.
                if note is None and CONFIG.actions.enabled:
                    note = await self._handle_action_command(text)
                    if note == "":     # already answered deterministically, in full
                        return
                if note is not None:
                    extra.append(Message("system", note))
            # Opt-in, and last, so it never displaces what he actually asked about.
            # Skipped in recall mode: an accurate recitation is not the place to bolt
            # a "by the way" onto.
            if self.digest is not None and not recall_mode:
                nudge = self._digest_nudge()
                if nudge is not None:
                    extra.append(nudge)
            # Recall questions drop the few-shot examples: they are vivid stories and
            # the model recites them back as history. Accuracy over register here.
            prompt = recall_prompt() if recall_mode else self._system_prompt
            messages = build_messages(
                prompt, facts, self._history,
                recent_limit=CONFIG.memory.recent_turns,
                char_budget=CONFIG.memory.context_char_budget,
                extra_system=extra,
            )
            reply = await self._think_and_speak(messages)
            if reply:                      # empty if Mithilesh cut him off before a word landed
                self._history.append(Message("assistant", reply))
                self._remember_turn(text, reply)
        except Exception as e:  # noqa: BLE001 - a failed turn must never hang "thinking"
            # LLMError, TTS failure, transcription error — surface it, don't stall.
            print(f"  [turn failed] {type(e).__name__}: {e}")
            if appended_user and self._history and self._history[-1].role == "user":
                self._history.pop()  # don't leave a dangling half-exchange in context
            try:
                await self._speak("Sorry, something went wrong on my end. Let's try again.")
            except Exception:  # noqa: BLE001 - even the apology's audio can fail
                pass
        finally:
            if self.state is not ConversationState.IDLE:
                self._enter(ConversationState.IDLE)
            await self._drain_alerts()
            if self._barge_in:
                # Mithilesh said the wake word to cut him off, so that word is already spent.
                # Going idle here would make his say it a second time before he'd
                # hear the thing he actually interrupted his to say.
                self._barge_in = False
                print("  [interrupt] listening for what you wanted instead")
                self._begin_listening()
            elif self._go_quiet:
                self._go_quiet = False       # he has said goodbye; now actually stop
                self._enter(ConversationState.IDLE)
            elif self._engaged and self.state is ConversationState.IDLE:
                # Continuous conversation: no wake word between turns. Background work
                # started by THIS turn must survive the transition.
                self._begin_listening(interrupt_background=False)

    async def _think(self, messages: list[Message]) -> str:
        """Generate the whole reply before returning. Kept for callers that want text
        rather than speech; the live turn uses _think_and_speak instead."""
        def collect() -> str:
            return "".join(self.llm.chat(messages, stream=True)).strip()

        async with self._llm_lock:
            return await asyncio.to_thread(collect)

    async def _think_and_speak(self, messages: list[Message]) -> str:
        """Generate and speak at the same time, and return what was actually said.

        The model is a BLOCKING generator, so it runs in a worker thread that pushes
        finished sentences onto an asyncio.Queue; the loop here drains that queue and
        plays each one. Sentence two is being written while sentence one is in the air,
        which is the whole point — the wait drops from "the entire reply" to "the first
        sentence".

        The last sentence is HELD back. trim_reflexive_question can only judge a
        trailing question once it knows the reply is finished, and by then a streamed
        sentence would already have been spoken. So we keep a one-sentence lookahead:
        everything before the end plays immediately, and the final sentence waits for
        generation to end before it's judged and then spoken or dropped.

        The LLM lock is held across the whole thing, including waiting for the producer
        to stop. That keeps the guarantee memory extraction relies on: it can never hit
        Ollama while a reply is still being generated.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def put(item) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                pass                       # loop already closed (aborted run)

        failure: BaseException | None = None

        def produce() -> None:
            nonlocal failure
            buffer = ""
            try:
                for token in self.llm.chat(messages, stream=True):
                    if self._interrupt.is_set():
                        break              # stop generating the moment he cuts in
                    buffer += token
                    ready, buffer = split_complete_sentences(buffer)
                    for sentence in ready:
                        put(sentence)
                tail = buffer.strip()
                if tail and not self._interrupt.is_set():
                    put(tail)
            except BaseException as e:      # noqa: BLE001 - re-raised on the loop below
                # A brain failure happens in this worker thread now, so it has to be
                # carried back out or the turn would fail SILENTLY instead of apologising.
                failure = e
            finally:
                put(_GENERATION_DONE)

        spoken: list[str] = []
        held: str | None = None
        self._interrupt.clear()
        async with self._llm_lock:
            producer = asyncio.create_task(asyncio.to_thread(produce))
            try:
                while True:
                    item = await queue.get()
                    if item is _GENERATION_DONE:
                        break
                    if held is not None:               # held is definitely not the last
                        await self._speak_sentence(held)
                        spoken.append(held)
                        if self._interrupt.is_set():
                            held = None
                            break
                    held = item

                if held is not None and not self._interrupt.is_set():
                    full = " ".join([*spoken, held])
                    kept = trim_reflexive_question(
                        full, keep_rate=CONFIG.reasoning.question_keep_rate)
                    if kept.rstrip().endswith(held.strip()):
                        await self._speak_sentence(held)
                        spoken.append(held)
                    else:
                        print("  [reply] trimmed a reflexive trailing question")
            finally:
                with contextlib.suppress(Exception):
                    await producer         # hold the lock until generation really stops
                self._finish_speaking()
        if failure is not None:
            raise failure                  # let _run_turn surface it and apologise
        return " ".join(spoken).strip()

    async def _speak_sentence(self, text: str) -> None:
        """Play one sentence, entering SPEAKING on the first one only."""
        text = clean_for_speech(text)
        if not text:
            return
        if self.state is not ConversationState.SPEAKING:
            self._enter(ConversationState.SPEAKING)
            self.transport.mute_input()    # half-duplex holds for the whole reply
        print(f'  jesse: "{text}"')         # printed as each sentence starts playing
        if self.text_channel is not None:
            self.text_channel.log("jesse", text)
            self.text_channel.set_speaking(True)
        await self.transport.play(
            self._interruptible(self.synth.synthesize(text)),
            sample_rate=self.synth.sample_rate,
        )

    def _finish_speaking(self) -> None:
        """Always run, interrupted or not: unmute, flush the echo tail, back to idle."""
        if self.text_channel is not None:
            self.text_channel.set_speaking(False)
        self.transport.unmute_input()
        if self.state is not ConversationState.IDLE:
            self._enter(ConversationState.IDLE)

    # -- memory ------------------------------------------------------------

    def _remember_turn(self, user_text: str, reply: str) -> None:
        """Persist the exchange and kick idle-gap fact extraction (both no-ops if
        memory isn't wired). Called after a reply is spoken, so we're back at IDLE."""
        if self.store is None:
            return
        uid = self.store.append_turn(Message("user", user_text))
        aid = self.store.append_turn(Message("assistant", reply))
        if self.extractor is not None:
            exchange = f"The user said: {user_text}\nYou replied: {reply}"
            self._extract_task = asyncio.create_task(
                self._extract_facts(exchange, turn_ids=(uid, aid)))

    @property
    def remote_live(self) -> bool:
        """True while his phone has the floor. False for anything at the desk."""
        return bool(getattr(self.transport, "remote_live", False))

    async def _read_sources_loop(self) -> None:
        """Read his sources on a wall-clock interval. Silent, forever, best-effort.

        Never speaks and never interrupts — a headline is not time-critical, and the
        one thing the scheduler is allowed to do (cut into his day) is exactly what
        this must not. That is also why it is not built on the Scheduler: reusing the
        class would mean reusing an alert path whose whole purpose is to interrupt.

        Network work goes to a thread so a slow or hanging feed cannot stall the mic
        loop, and every failure is caught: a source that is down is a quiet log line,
        never a dead session and never a turn where he claims to have read something.
        """
        while True:
            try:
                if self.digest.due(interval_hours=CONFIG.digest.interval_hours):
                    await self._read_sources_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:                 # noqa: BLE001 - never kill the loop
                print(f"  [sources] check failed: {type(e).__name__}: {e}")
            await asyncio.sleep(max(60.0, CONFIG.digest.interval_hours * 3600 / 12))

    async def _read_sources_once(self) -> int:
        """One pass over every configured source. Returns how many items were new."""
        added = 0
        for name, url in CONFIG.digest.sources:
            try:
                items = await asyncio.to_thread(
                    fetch_feed, url, name,
                    timeout=CONFIG.digest.fetch_timeout,
                    max_bytes=CONFIG.digest.max_bytes,
                    limit=CONFIG.digest.items_per_source)
            except FeedError as e:
                print(f"  [sources] {name}: {e}")
                continue
            new = self.digest.add(items)
            added += new
            print(f"  [sources] {name}: {len(items)} item(s), {new} new")
        # Stamped even when everything failed, so a source that is down does not mean
        # retrying on every tick for as long as it stays down.
        self.digest.set_last_fetch(datetime.now())
        return added

    @staticmethod
    def _phrase_digest(items) -> str:
        """Read the headlines out, deterministically.

        His own words were tried first and lost on the one thing that matters: asked
        "anything new?" with items waiting he said "nothing new" in roughly 1 run in
        6-12 — a false statement about what he has, and the same class as the
        unknown-app refusal that dropped its negation. Strengthening the note did not
        remove it. Headlines are a record being recited, and this project has decided
        twice already that reciting a record wants accuracy over register.

        The empty case stays in his voice: "nothing came in" measured 6/6 honest,
        because agreeing there is nothing is the easy direction.
        """
        lead = "One thing came in." if len(items) == 1 else f"{len(items)} things came in."
        parts = []
        for item in items:
            line = item.title.rstrip(".")
            parts.append(f"From {item.source}, {line}." if item.source else f"{line}.")
        return " ".join([lead, *parts])

    async def _handle_digest_query(self, text: str) -> Message | None:
        """"Anything new?" — answered from the table, or honestly not at all.

        Deterministic trigger for the usual reason, plus one specific to this: the
        answer is a list of things from outside the machine, and a model asked to
        decide *whether* to volunteer them would sometimes volunteer them into a
        conversation about something else entirely.

        Returns a system note for the EMPTY case (his voice, measured honest), and
        speaks the items directly when there are any — see `_phrase_digest`.
        """
        query = asks_whats_new(text, [name for name, _ in CONFIG.digest.sources])
        if query is None:
            return None
        items = self.digest.untold(limit=CONFIG.digest.max_items_told)
        if query.source is not None:
            items = [i for i in items if i.source == query.source]
        print(f"  [sources] asked what's new — {len(items)} item(s) to tell him"
              + (f" (from {query.source})" if query.source else ""))
        self.digest.mark_told(i.id for i in items)
        self._nudged = True          # he has just been told; no nudge on top of it
        if not items:
            return digest_context(items, source_label=query.source)
        spoken = self._phrase_digest(items)
        await self._speak(spoken)
        if self.text_channel is not None:
            self.text_channel.log("jesse", spoken)
        self._history.append(Message("assistant", spoken))
        self._remember_turn(text, spoken)
        return _ANSWERED

    def _digest_nudge(self) -> Message | None:
        """Opt-in, once a session, and only ever riding on a reply he asked for.

        Never an unprompted announcement. The rule from the beginning of this project
        is that he interrupts only for things that are time-critical, and news is the
        definition of what is not; a thing that talks at you unbidden is a thing you
        turn off. So the strongest version of "remind me every now and then" that
        survives that rule is a single line appended to a reply already happening.
        """
        if self._nudged or not CONFIG.digest.nudge or self.digest is None:
            return None
        waiting = self.digest.untold_count()
        if not waiting:
            return None
        self._nudged = True
        return Message("system", (
            f"Separately: {waiting} thing(s) you read from your sources have not been "
            "mentioned to him yet. After answering him, add ONE short clause saying "
            "you came across something and can tell him if he wants. Do NOT say what "
            "it was, do not list anything, and do not let it take over the reply."))

    async def _handle_action_command(self, text: str) -> str | None:
        """Open something, press a media key, or look for a file — then tell him what
        actually happened so he reports it instead of claiming it.

        The note always says whether it worked. Nothing here is allowed to have his
        cheerfully confirm an action that failed: an assistant that says "opened it"
        when nothing opened is worse than one that can't open anything, because he
        stops looking.

        Blocking work runs off the loop — a file search walks real directories, and a
        turn that stalls mid-sentence looks like a crash.
        """
        cmd = parse_action_command(text, CONFIG.actions.apps)

        # A confirmation he asked for a moment ago, answered. Resolved before
        # anything else, because "yes" parses as no command at all.
        confirmed = False
        if self._pending_action is not None:
            pending, self._pending_action = self._pending_action, None
            if cmd is None and _is_short_affirmation(text):
                cmd, confirmed = pending, True
            # Anything else — a new command, or his moving on — lets it lapse. A
            # request he did not confirm must decay, not wait around to be triggered
            # by an unrelated "sure" three turns later.
            elif cmd is None:
                print(f"  [action] the pending {type(pending).__name__} was not "
                      f"confirmed — dropped")

        if cmd is None:
            # Nothing parsed. If he clearly named something he can act on, that is a
            # MISS, not small talk — and a miss he must not paper over. Live, he
            # answered "I'll open Spotify." to a request that never reached the
            # parser: a promise about the world, which is the same class as the
            # unknown-app refusal that dropped its negation. Spoken deterministically
            # for the same reason.
            missed = looks_like_an_action(text, CONFIG.actions.apps)
            if missed is not None:
                print(f"  [action] NOT RUN — heard something about {missed!r} but could "
                      f"not read it as a command: {text!r}")
                line = f"I didn't catch that — did you want me to open {missed}?"
                self._pending_action = None
                await self._speak(line)
                if self.text_channel is not None:
                    self.text_channel.log("jesse", line)
                self._history.append(Message("assistant", line))
                self._remember_turn(text, line)
                return ""
            return None

        if isinstance(cmd, UnknownTarget):
            # Deterministic, like the knowledge ask, and for the same measured reason:
            # this sentence is honesty-critical and hangs on a negation, and probed
            # live the 3B dropped the negation — it said "I can open Photoshop." about
            # an app it cannot open. A claim of ability is not delegable to a model
            # that loses the word "not" one time in three.
            print(f"  [action] REFUSED — {cmd.name!r} is not in the registry")
            line = f"I don't have {cmd.name} — that's not something I can open."
            await self._speak(line)
            if self.text_channel is not None:
                self.text_channel.log("jesse", line)
            self._history.append(Message("assistant", line))
            self._remember_turn(text, line)
            return ""      # sentinel: the turn is already answered, nothing for the LLM

        # Over the phone, a side effect gets one confirmation first. Not nerves: the
        # same deterministic parsers are reading a transcript from a phone mic, and
        # whisper already mishears at the desk on clean audio ("Stay Jarvis", "Skit"
        # for "skip", "Re-soon" for "resume"). Opening things on a machine he is not
        # sitting at is the least reversible thing in this project. Reading,
        # remembering, timers and file SEARCH are untouched — they are reversible or
        # read-only, so they keep full parity.
        if (not confirmed and self.remote_live and CONFIG.remote.confirm_actions
                and isinstance(cmd, (OpenCommand, MediaCommand))):
            what = (f"open {cmd.name}" if isinstance(cmd, OpenCommand)
                    else f"press {cmd.action.replace('_', '/')}")
            self._pending_action = cmd
            line = f"You're away — do you want me to {what}?"
            print(f"  [action] remote request to {what} — confirming first")
            await self._speak(line)
            if self.text_channel is not None:
                self.text_channel.log("jesse", line)
            self._history.append(Message("assistant", line))
            self._remember_turn(text, line)
            return ""

        if isinstance(cmd, OpenCommand):
            try:
                await asyncio.to_thread(open_target, cmd.target)
            except ActionError as e:
                print(f"  [action] FAILED to open {cmd.name!r}: {e}")
                return (f"You tried to open {cmd.name} for him and it did not work. Tell him it "
                        "failed, in one short sentence. Do not claim it opened.")
            print(f"  [action] DONE — opened {cmd.name!r} -> {cmd.target}")
            return (f"You just opened {cmd.name} on his computer, and it worked. Say so in a few "
                    "words — this is a small thing, not an announcement.")

        if isinstance(cmd, MediaCommand):
            try:
                await asyncio.to_thread(media_key, cmd.action)
            except ActionError as e:
                print(f"  [action] FAILED to press {cmd.action}: {e}")
                return ("You tried to control whatever is playing and could not. Tell him it "
                        "didn't work, briefly.")
            print(f"  [action] DONE — pressed media key {cmd.action}")
            return (f"You just pressed {cmd.action.replace('_', '/')} for whatever is playing. "
                    "Acknowledge it in a couple of words at most — he is listening to something, "
                    "so do not talk over it.")

        hits = await asyncio.to_thread(
            find_files, cmd.needle, CONFIG.actions.search_roots,
            limit=CONFIG.actions.search_limit, max_depth=CONFIG.actions.search_max_depth,
        )
        print(f"  [action] DONE — searched for {cmd.needle!r}, {len(hits)} hit(s)"
              + (f": {', '.join(h.name for h in hits)}" if hits else ""))
        if not hits:
            return (f"He asked you to find {cmd.needle!r}. You looked through his documents, "
                    "desktop and downloads and found NOTHING matching. Tell him you couldn't "
                    "find it. Do NOT invent a filename or a folder.")
        listing = "; ".join(f"{p.name} in {p.parent.name}" for p in hits)
        return (f"He asked you to find {cmd.needle!r}. These are the complete results and the "
                f"only ones that exist: {listing}. Read them back naturally in one or two "
                "spoken sentences — no lists, no full paths — and do NOT invent any others.")

    def _handle_forget_command(self, text: str) -> str | None:
        """Actually delete what Mithilesh asked him to forget, and tell him what happened so
        he confirms truthfully. Returns None when this wasn't a forget request.

        He used to agree out loud while the fact stayed in the database. Agreeing
        without acting is worse than refusing, because he stops checking.
        """
        if self.store is None:
            return None
        cmd = parse_forget_command(text)
        if cmd is None:
            return None

        if not cmd.target:
            print("  [memory] forget request with no subject — asking which")
            return ("He asked you to forget something but didn't say what. Ask him which "
                    "thing he means, in one short sentence. Do NOT claim you've forgotten "
                    "anything yet — nothing has been deleted.")

        matches = self.store.find_facts(cmd.target)
        if not matches:
            print(f"  [memory] nothing stored matching {cmd.target!r}")
            return (f"He asked you to forget {cmd.target!r}, but you have nothing stored "
                    "matching that. Say so plainly and briefly — don't pretend to delete "
                    "something that was never there.")

        if len(matches) > 1:
            listing = "; ".join(f.text for f in matches[:4])
            print(f"  [memory] {len(matches)} facts match {cmd.target!r} — asking which")
            return (f"He asked you to forget {cmd.target!r}, but several things match: "
                    f"{listing}. Ask him which one — briefly. Nothing has been deleted yet.")

        gone = self.store.forget(cmd.target)
        removed = "; ".join(f.text for f in gone)
        print(f"  [memory] FORGOT: {removed}")
        return (f"You just permanently deleted this from your memory: {removed}. It is "
                "really gone. Confirm in one short sentence.")

    def _handle_schedule_command(self, text: str) -> str | None:
        """Create / reschedule / cancel a reminder. Returns a system note telling him
        what just happened so he confirms it in his own words, or None if this
        wasn't a scheduling request at all.

        Order is enforced in the parser: cancel and reschedule are recognised BEFORE
        creation, so "stop the timer set for 10 minutes" cancels instead of quietly
        setting a second timer.
        """
        cmd = parse_schedule_command(text, now=datetime.now())
        if cmd is None:
            return None

        if isinstance(cmd, QueryCommand):
            pending = self.scheduler.pending()
            print(f"  [reminder] asked what's pending: {len(pending)}")
            if not pending:
                return ("He asked what timers or reminders he has. There are NONE. Tell him "
                        "that plainly in one short sentence — don't invent any.")
            now = datetime.now()
            listing = "; ".join(
                f"{p.task or 'a timer'} in {_phrase_delay(max(0, (p.fire_at - now).total_seconds()))}"
                for p in pending
            )
            return (f"He asked what's pending. He has exactly these, and nothing else: {listing}. "
                    "Say them back naturally in one or two short spoken sentences — no lists, "
                    "no bullet points, and do NOT invent any others.")

        if isinstance(cmd, IncompleteCommand):
            print("  [reminder] change requested with no new time — asking him for it")
            return ("He wants to change a reminder but didn't say what to change it to. Ask "
                    "him what time he wants, in one short sentence, and mention he can say it "
                    "all at once like 'change the timer to 45 seconds'.")

        if isinstance(cmd, CancelCommand):
            count, reason = self.scheduler.cancel(cmd.hint, all_of_them=cmd.all_of_them)
            if reason == "none":
                return ("He asked you to cancel a reminder, but nothing is pending. "
                        "Tell his there's nothing to cancel, warmly and briefly.")
            if reason == "ambiguous":
                pending = self.scheduler.pending()
                listing = "; ".join(f"{p.task or 'a timer'} at {p.fire_at:%H:%M}" for p in pending)
                return (f"He asked to cancel a reminder but has several pending: {listing}. "
                        "Ask him which one he means — briefly, one sentence.")
            what = "all of them" if cmd.all_of_them else "it"
            return (f"You just cancelled {what} ({count} reminder(s)) for him. Confirm in one "
                    "short sentence.")

        if isinstance(cmd, RescheduleCommand):
            item, reason = self.scheduler.reschedule(cmd.fire_at, cmd.hint,
                                                     label=cmd.spoken_delay)
            if reason == "none":
                return ("He tried to change a reminder, but nothing is pending. Tell him "
                        "there's nothing set yet, and offer to set one.")
            if reason == "ambiguous":
                pending = self.scheduler.pending()
                listing = "; ".join(f"{p.task or 'a timer'} at {p.fire_at:%H:%M}" for p in pending)
                return (f"He asked to move a reminder but has several pending: {listing}. "
                        "Ask him which one he means — briefly.")
            return (f"You just MOVED his existing reminder (not made a new one) — it now goes "
                    f"off in {cmd.spoken_delay}. Confirm in one short sentence.")

        # otherwise: a brand-new timer/reminder
        self.scheduler.add(cmd.task, cmd.fire_at, is_timer=cmd.is_timer,
                           label=cmd.spoken_delay)
        what = "timer" if cmd.is_timer else f"reminder to {cmd.task}"
        print(f"  [reminder] set: {what} in {cmd.spoken_delay} (fires {cmd.fire_at:%H:%M:%S})")
        return (f"You just set a {what} for him, going off in {cmd.spoken_delay}. Confirm it "
                "warmly in one short sentence — no lists, no repeating the time twice.")

    async def _summarise_pending(self) -> None:
        """Fold un-summarised turns into one episode. Cheap no-op when there is
        nothing new, so it is safe to call at startup and at shutdown."""
        assert self.episodes is not None and self.summariser is not None
        turns = self.episodes.unsummarised_turns()
        if len(turns) < 2:
            return                               # not a conversation yet
        try:
            async with self._llm_lock:           # shares the reply/extraction gate
                summary = await asyncio.to_thread(self.summariser.summarise, turns)
            summary = summary.strip()
            if not summary:
                return
            started = datetime.fromisoformat(turns[0][3])
            ended = datetime.fromisoformat(turns[-1][3])
            self.episodes.add(summary, started, ended, [t[0] for t in turns])
            print(f"  [memory] saved this conversation: {summary[:70]}…")
        except asyncio.CancelledError:
            raise                                # retried next start; turns stay unmarked
        except Exception as e:                   # noqa: BLE001 - best effort
            print(f"  [memory] summarising failed: {type(e).__name__}: {e}")

    async def _catch_up_extractions(self) -> None:
        """Re-run extraction for exchanges that never finished, so a fact taught late
        at night is captured next time he starts instead of being lost."""
        assert self.store is not None
        pending = self.store.unprocessed_exchanges(limit=CONFIG.memory.catch_up_limit)
        if not pending:
            return
        print(f"  [memory] catching up on {len(pending)} unfinished extraction(s) "
              "from earlier…")
        for uid, aid, user_text, reply in pending:
            exchange = f"The user said: {user_text}\nYou replied: {reply}"
            await self._extract_facts(exchange, turn_ids=(uid, aid), catchup=True)

    async def _extract_facts(self, exchange: str, *, turn_ids=(),
                             catchup: bool = False) -> None:
        assert self.store is not None and self.extractor is not None
        debug = CONFIG.memory.debug_extraction
        try:
            print("  [memory] catching up on an earlier exchange…" if catchup
                  else "  [memory] extracting…")
            async with self._llm_lock:          # never overlaps a live reply on Ollama
                # Waiting-to-hear (LISTENING) is a fine time to extract; only an
                # ACTIVE turn must not be competed with. Checking for IDLE here broke
                # extraction entirely once continuous mode kept his listening between
                # turns — he was never idle again.
                if self.state in (ConversationState.THINKING, ConversationState.SPEAKING):
                    print("  [memory] skipped — a new turn started before extraction could run")
                    return
                raw = await asyncio.to_thread(self.extractor.extract, exchange)
            if debug:
                print(f"  [memory:debug] asked: {exchange!r}")
                print(f"  [memory:debug] model returned: {raw!r}")
            facts = parse_extracted_facts(raw, min_confidence=CONFIG.memory.min_fact_confidence)
            if debug:
                print(f"  [memory:debug] parsed {len(facts)} fact(s): "
                      + "; ".join(f"{f.subject}={f.text!r}" for f in facts))
            for fact in facts:
                self.store.add_fact(fact)        # sync CPU embed; on the loop thread
            # Extraction COMPLETED (facts or not) -> never redo this exchange.
            self.store.mark_processed(turn_ids)
            if facts:
                print("  [memory] stored " + "; ".join(f.subject or f.text[:30] for f in facts))
            else:
                print("  [memory] nothing to store — no durable facts found in that exchange")
        except asyncio.CancelledError:
            # Left unprocessed ON PURPOSE: picked up by catch-up next time he starts.
            print("  [memory] extraction interrupted — saved for later; he'll pick it up "
                  "next time you open him.")
            raise
        except Exception as e:                   # noqa: BLE001 - extraction is best-effort
            print(f"  [extraction failed] {type(e).__name__}: {e}")

    async def _speak(self, text: str) -> None:
        text = clean_for_speech(text)  # single choke-point: everything spoken is voice-shaped
        if not text:
            self._enter(ConversationState.IDLE)
            return
        self._enter(ConversationState.SPEAKING)
        self._interrupt.clear()
        self.transport.mute_input()
        print(f'  jesse: "{text}"')
        try:
            await self.transport.play(
                self._interruptible(self.synth.synthesize(text)),
                sample_rate=self.synth.sample_rate,
            )
        finally:
            self.transport.unmute_input()  # flush self-echo tail
            self._enter(ConversationState.IDLE)

    def _interruptible(self, frames: Iterator[bytes]) -> Iterator[bytes]:
        """Wrap the synth stream so a stop-word (which sets _interrupt from the
        ingest loop) cuts playback at the next frame boundary."""
        for chunk in frames:
            if self._interrupt.is_set():
                return
            yield chunk

    async def _drain_alerts(self) -> None:
        # Same reasoning as extraction: LISTENING is a safe moment to speak up, and
        # requiring IDLE would mean reminders never fire in continuous mode.
        while self._alerts and self.state in (ConversationState.IDLE,
                                              ConversationState.LISTENING):
            await self._speak(self._alerts.pop(0))
