"""Builds a live Orchestrator from CONFIG. THIS is the swap point.

Change one line here (or just install the Piper binary / flip use_ollama) and the
whole pipeline upgrades — no orchestrator or interface changes. That is the payoff
of coding to the contracts in jesse/core/interfaces.py.
"""

from __future__ import annotations

from jesse.core.state import ConversationState
from jesse.persona import SYSTEM_PROMPT


def _print_state(state: ConversationState) -> None:
    print(f"  [state] -> {state.value}")


def build_orchestrator(*, use_ollama: bool = False, use_experiential: bool = False,
                       input_device: int | None = None, text_channel=None):
    """Returns (orchestrator, voice_label, brain_label). input_device overrides
    CONFIG.audio.input_device (from `run --device N`).

    Brain selection: use_experiential wins over use_ollama (both are "a real brain",
    which is what unlocks memory + the scheduler below); neither = the Echo stub."""
    from jesse.audio.transport import LocalAudioTransport
    from jesse.audio.vad import EnergyVad
    from jesse.audio.wakeword import OpenWakeWordDetector
    from jesse.config import CONFIG
    from jesse.llm.echo import EchoLLM
    from jesse.orchestrator import Orchestrator
    from jesse.stt.whisper import WhisperTranscriber
    from jesse.tts.piper import PiperSynthesizer
    from jesse.tts.stub import StubSynthesizer

    # Either real brain satisfies the LLM contract, so everything downstream of the
    # brain (memory, extraction, episodes, scheduler) is wired identically for both.
    real_brain = use_ollama or use_experiential

    in_dev = input_device if input_device is not None else CONFIG.audio.input_device
    transport = LocalAudioTransport(
        input_device=in_dev, output_device=CONFIG.audio.output_device,
        gain=CONFIG.audio.capture_gain,
    )
    wake = OpenWakeWordDetector(CONFIG.wake.model)
    stopword = OpenWakeWordDetector(CONFIG.wake.stop_word)
    vad = EnergyVad(
        threshold=CONFIG.audio.vad_threshold,
        silence_ms=CONFIG.audio.vad_silence_ms,
        min_speech_ms=CONFIG.audio.vad_min_speech_ms,
    )
    transcriber = WhisperTranscriber()

    if PiperSynthesizer.is_available():
        synthesizer = PiperSynthesizer()
        voice_label = f"Piper ({CONFIG.speech.piper_voice})"
    else:
        synthesizer = StubSynthesizer()
        voice_label = "STUB (install the Piper binary to swap in real speech)"

    # Memory needs a real brain to extract facts, so it's wired only for those.
    store = None
    extractor = None
    episodes = None
    summariser = None
    corpus = None
    digest = None
    if real_brain:
        from jesse.memory.embedder import FastEmbedEmbedder
        from jesse.memory.extraction import FactExtractor
        from jesse.memory.store import SqliteMemoryStore
        if use_experiential:
            from jesse.llm.experiential import ExperientialLLM
            llm = ExperientialLLM()
            brain_label = f"Experiential/{CONFIG.reasoning.experiential_model} (NOT local)"
        else:
            from jesse.llm.ollama import OllamaLLM
            llm = OllamaLLM()
            brain_label = f"Ollama/{CONFIG.reasoning.model}"
        CONFIG.memory.db_path.parent.mkdir(parents=True, exist_ok=True)
        store = SqliteMemoryStore(
            CONFIG.memory.db_path, FastEmbedEmbedder(),
            log_path=CONFIG.memory.db_path.parent / "memory-log.txt",
        )
        from jesse.schedule.scheduler import Scheduler
        from jesse.schedule.store import SqliteScheduleStore
        from jesse.memory.episodes import EpisodeStore, Summariser
        from jesse.memory.seed import seed_if_needed
        n = seed_if_needed(store)      # first run: plant core + self facts
        if n:
            print(f"  [memory] seeded {n} core/self facts (first run)")
        extractor = FactExtractor(llm)
        episodes = EpisodeStore(CONFIG.memory.db_path, FastEmbedEmbedder())
        if CONFIG.digest.enabled:
            from jesse.digest.store import DigestStore
            digest = DigestStore(CONFIG.memory.db_path)
        if CONFIG.knowledge.enabled:
            from jesse.memory.corpus import CorpusStore
            corpus = CorpusStore(CONFIG.memory.db_path, FastEmbedEmbedder())
        summariser = Summariser(llm)
        schedule_store = SqliteScheduleStore(CONFIG.memory.db_path)
    else:
        llm = EchoLLM()
        brain_label = "Echo (Phase 0 stub brain)"

    from jesse.audio.frames import ms_to_chunks
    orch = Orchestrator(
        transport=transport, wake=wake, stopword=stopword, vad=vad,
        transcriber=transcriber, llm=llm, synthesizer=synthesizer,
        system_prompt=SYSTEM_PROMPT, preroll_frames=ms_to_chunks(CONFIG.audio.preroll_ms),
        store=store, extractor=extractor, episodes=episodes, summariser=summariser,
        corpus=corpus, digest=digest,
        text_channel=text_channel,
        on_state_change=_print_state,
    )
    # Scheduler needs the orchestrator's notify(), so it's attached after construction.
    if real_brain:
        orch.scheduler = Scheduler(
            schedule_store, orch.notify,
            tick_seconds=CONFIG.schedule.tick_seconds,
            stale_after_s=CONFIG.schedule.stale_after_minutes * 60,
            overdue_note_after_s=CONFIG.schedule.overdue_note_after_seconds,
        )
    return orch, voice_label, brain_label
