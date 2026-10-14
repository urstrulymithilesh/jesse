"""Builds a live Orchestrator from CONFIG. THIS is the swap point.

Change one line here (or just install the Piper binary / flip use_ollama) and the
whole pipeline upgrades — no orchestrator or interface changes. That is the payoff
of coding to the contracts in pinkman/core/interfaces.py.
"""

from __future__ import annotations

from pinkman.config import DATA_DIR
from pinkman.core.state import ConversationState
from pinkman.persona import SYSTEM_PROMPT

DEFAULT_VOICE_PROFILE = DATA_DIR / "voice-clone/default.json"


def _print_state(state: ConversationState) -> None:
    print(f"  [state] -> {state.value}")


def build_orchestrator(*, use_ollama: bool = False, input_device: int | None = None,
                       text_channel=None, wake_model: str | None = None,
                       wake_phrase: str | None = None, wake_threshold: float = 0.5,
                       voice: str | None = None, omnivoice_profile: str | None = None):
    """Returns (orchestrator, voice_label, brain_label). input_device overrides
    CONFIG.audio.input_device (from `run --device N`)."""
    from pinkman.audio.transport import LocalAudioTransport
    from pinkman.audio.vad import EnergyVad
    from pinkman.audio.wakeword import ManualWakeDetector, OpenWakeWordDetector
    from pinkman.config import CONFIG
    from pinkman.llm.echo import EchoLLM
    from pinkman.orchestrator import Orchestrator
    from pinkman.stt.whisper import WhisperTranscriber
    from pinkman.tts.piper import PiperSynthesizer
    from pinkman.tts.stub import StubSynthesizer

    if voice is None and omnivoice_profile is None and DEFAULT_VOICE_PROFILE.is_file():
        omnivoice_profile = str(DEFAULT_VOICE_PROFILE)
    in_dev = input_device if input_device is not None else CONFIG.audio.input_device
    transport = LocalAudioTransport(
        input_device=in_dev, output_device=CONFIG.audio.output_device,
        gain=CONFIG.audio.capture_gain,
    )
    selected_wake = wake_model or CONFIG.wake.model
    wake = (OpenWakeWordDetector(selected_wake, threshold=wake_threshold)
            if selected_wake else ManualWakeDetector())
    stopword = (OpenWakeWordDetector(selected_wake, threshold=wake_threshold)
                if selected_wake else ManualWakeDetector())
    if wake_model is not None:
        wake._ensure()  # reject incompatible exports before capture or memory writes
        stopword._ensure()
    vad = EnergyVad(
        threshold=CONFIG.audio.vad_threshold,
        silence_ms=CONFIG.audio.vad_silence_ms,
        min_speech_ms=CONFIG.audio.vad_min_speech_ms,
    )
    transcriber = WhisperTranscriber()

    if voice is not None and omnivoice_profile is not None:
        raise ValueError("Choose a Piper voice or an OmniVoice profile, not both")
    if voice is not None and not PiperSynthesizer.is_available(voice):
        raise ValueError(f"Requested Piper voice {voice!r} is unavailable")
    if omnivoice_profile is not None:
        from pinkman.tts.omnivoice import OmniVoiceSynthesizer
        synthesizer = OmniVoiceSynthesizer(omnivoice_profile)
        voice_label = f"Pinkman voice · {synthesizer.steps} steps"
    elif PiperSynthesizer.is_available(voice):
        synthesizer = PiperSynthesizer(voice=voice)
        if voice is not None:
            _ = synthesizer.sample_rate  # validate custom export/config before startup
        voice_label = f"Piper ({voice or CONFIG.speech.piper_voice})"
    else:
        synthesizer = StubSynthesizer()
        voice_label = "STUB (install the Piper binary to swap in real speech)"

    # Memory needs a real brain to extract facts, so it's wired only with Ollama.
    store = None
    extractor = None
    episodes = None
    summariser = None
    corpus = None
    digest = None
    if use_ollama:
        from pinkman.llm.ollama import OllamaLLM
        from pinkman.memory.embedder import FastEmbedEmbedder
        from pinkman.memory.extraction import FACT_SCHEMA, FactExtractor
        from pinkman.memory.store import SqliteMemoryStore
        llm = OllamaLLM()
        brain_label = f"Ollama/{CONFIG.reasoning.model}"
        CONFIG.memory.db_path.parent.mkdir(parents=True, exist_ok=True)
        store = SqliteMemoryStore(
            CONFIG.memory.db_path, FastEmbedEmbedder(),
            log_path=CONFIG.memory.db_path.parent / "memory-log.txt",
        )
        from pinkman.memory.episodes import EpisodeStore, Summariser
        from pinkman.memory.seed import seed_if_needed
        from pinkman.schedule.scheduler import Scheduler
        from pinkman.schedule.store import SqliteScheduleStore
        n = seed_if_needed(store)      # first run: plant core + self facts
        if n:
            print(f"  [memory] seeded {n} core/self facts (first run)")
        # Same local model and orchestrator lock; only the extraction request's
        # output format and sampling differ from ordinary conversation.
        extractor = FactExtractor(OllamaLLM(response_schema=FACT_SCHEMA, temperature=0))
        episodes = EpisodeStore(CONFIG.memory.db_path, FastEmbedEmbedder())
        if CONFIG.digest.enabled:
            from pinkman.digest.store import DigestStore
            digest = DigestStore(CONFIG.memory.db_path)
        if CONFIG.knowledge.enabled:
            from pinkman.memory.corpus import CorpusStore
            corpus = CorpusStore(CONFIG.memory.db_path, FastEmbedEmbedder())
        summariser = Summariser(llm)
        schedule_store = SqliteScheduleStore(CONFIG.memory.db_path)
    else:
        llm = EchoLLM()
        brain_label = "Echo (Phase 0 stub brain)"

    from pinkman.audio.frames import ms_to_chunks
    if text_channel is not None:
        text_channel.configure(voice=voice_label, brain=brain_label,
                               wake=wake_phrase or "Wake Pinkman button")
    orch = Orchestrator(
        transport=transport, wake=wake, stopword=stopword, vad=vad,
        transcriber=transcriber, llm=llm, synthesizer=synthesizer,
        system_prompt=SYSTEM_PROMPT, preroll_frames=ms_to_chunks(CONFIG.audio.preroll_ms),
        wake_phrase=wake_phrase,
        store=store, extractor=extractor, episodes=episodes, summariser=summariser,
        corpus=corpus, digest=digest,
        text_channel=text_channel,
        on_state_change=_print_state,
    )
    # Scheduler needs the orchestrator's notify(), so it's attached after construction.
    if use_ollama:
        orch.scheduler = Scheduler(
            schedule_store, orch.notify,
            tick_seconds=CONFIG.schedule.tick_seconds,
            stale_after_s=CONFIG.schedule.stale_after_minutes * 60,
            overdue_note_after_s=CONFIG.schedule.overdue_note_after_seconds,
        )
    return orch, voice_label, brain_label
