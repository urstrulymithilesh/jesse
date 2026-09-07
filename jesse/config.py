"""Central config. Swapping a model or engine later is a change HERE, not in code.

Every value is chosen to honor the fully-local, CPU/GPU-split design: his brain,
memory and processing stay on this machine. The network is a connection (RSS,
remote access), never a place any part of his runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# All runtime data stays local and private (gitignored).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class ReasoningConfig:
    # GPU = reason. Qwen pinned resident so first-token latency stays low.
    ollama_host: str = "http://localhost:11434"
    # qwen2.5:3b is the design doc's PRIMARY pick (stronger instruction-following +
    # tool-calling) and ~1.9GB Q4 fits the 4GB GPU by size. NOTE: as of this machine's
    # Ollama build it still runs ~12 tok/s on CPU (both qwen2.5:3b and llama3.2) because
    # Ollama's Vulkan GPU discovery watchdog times out ("context deadline exceeded") and
    # falls back to CPU — see server.log. GPU enablement is a separate task; ~12 tok/s
    # CPU is usable for turn-based voice. llama3.2 stays the A/B baseline (persona eval T6).
    # Back on qwen2.5:3b for fast day-to-day iteration (~3-5s replies, ~3-5s extraction).
    # STORAGE/extraction is reliable on 3b; RECALL grounding is occasionally flaky (3b can
    # confabulate around an injected fact). 7b grounds far better but ~15s/reply + ~15s
    # extraction is too much dead time to iterate through. Swap back to "qwen2.5:7b" (or a
    # better model / GPU) later — this is exactly the swap the LLM interface was built for.
    model: str = "llama3.2"
    keep_alive: int = -1                # keep the model resident in VRAM (Ollama: int -1 = forever; "-1" string is rejected)
    num_ctx: int = 4096                 # cap KV/context so Windows-reserved VRAM doesn't OOM the 4GB
    temperature: float = 0.6            # 0.8 made his over-improvise (question every turn);
                                        # 0.6 follows the persona's "don't always ask" rule better
    # Fraction of the time a reflexive trailing question is KEPT (rest are trimmed by
    # reply_style). 0 = always trim, 1 = never trim. The 3B asks too much on its own.
    question_keep_rate: float = 0.15
    request_timeout: int = 90           # seconds; CPU generation is slow, but this bounds a hang

    # --- Experiential gateway (hosted brain; opt-in via `run --experiential`) ---
    # Speaks the OpenAI Chat Completions API, so pointing at it is a base-URL + key
    # swap, nothing more. This breaks the fully-local line above by design: turning it
    # on sends conversation text off this machine and bills an Experiential account.
    # The key is NEVER stored here — it is read from the environment at call time.
    experiential_base_url: str = "https://api.experientiallabs.ai/v1"
    experiential_model: str = "claude-fable-5.1"
    experiential_api_key_env: str = "EXPLABS_API_KEY"
    experiential_max_tokens: int = 512   # replies are spoken aloud, so keep them short


@dataclass(frozen=True)
class SpeechConfig:
    # CPU = hear + speak.
    whisper_model: str = "base.en"      # faster-whisper, int8, on CPU
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    sample_rate: int = 16_000           # pipeline INPUT convention: 16 kHz mono
    # Piper via the piper-tts Python API (no PATH binary). The voice .onnx (+ .json)
    # lives in models/; download with `python -m piper.download_voices <voice> --download-dir models`.
    # 22050 Hz — playback uses the voice's own rate, input stays 16k.
    # PROVISIONAL: ryan-high is the best-quality male voice in the catalogue and
    # stands in until he picks from the four that were A/B-d (ryan, joe, bryce,
    # sam). Swapping is this one line — the samples are say_en_US-*.wav.
    piper_voice: str = "en_US-ryan-high"


@dataclass(frozen=True)
class AudioConfig:
    # Device INDICES from `python diagnose.py`. None = OS default (often the wrong
    # one — the built-in mic array, not your headset). Set input_device to your
    # headset mic's index once diagnose.py confirms the VU meter moves.
    input_device: int | None = None
    output_device: int | None = None
    # Software capture gain — boosts quiet laptop mics so speech is loud enough for
    # both VAD and STT, without touching Windows mic settings. 1.0 = off.
    # Auto-calibration overrides this at startup unless auto_calibrate is False.
    capture_gain: float = 1.0
    # RMS level (POST-gain) a frame must exceed to count as speech. 500 was too high
    # for laptop mics; auto-calibration sets a real value from your room + voice.
    vad_threshold: float = 150.0
    vad_silence_ms: int = 950           # trailing silence that ends a turn. Single-number
                                        # trade-off: 700 cut people off mid-sentence, 1100 felt
                                        # laggy after finishing; ~950 is the middle. Tune by feel.
    vad_min_speech_ms: int = 300        # need this much speech before a silence can end a turn
    preroll_ms: int = 500               # audio kept BEFORE the wake fires, prepended to the
                                        # turn so the start of your sentence isn't lost
    # If a wake fires but no speech follows (a false trigger, or a barge-in where he
    # changed his mind), give up after this and go back to sleep instead of listening
    # forever — a silent LISTENING state looks exactly like a crash from outside.
    listen_timeout_ms: int = 8000
    # Once engaged, he keeps listening between turns with no wake word. The window is
    # much longer than the post-wake one (a pause mid-conversation is normal), but it is
    # NOT infinite on purpose: a false VAD trigger on room noise would otherwise start
    # junk turns forever, and a mic that never closes is a privacy regression.
    continuous_timeout_ms: int = 45000
    continuous_mode: bool = True
    # Measure room + a test phrase on `jesse run` startup and set gain + threshold.
    auto_calibrate: bool = True


@dataclass(frozen=True)
class WakeConfig:
    # Stock openWakeWord model until "yo Jesse" is trained (a separate future task).
    model: str = "hey_jarvis"           # a stock model; "yo Jesse" gets trained later
    stop_word: str = "hey_jarvis"       # kept live during SPEAKING to allow barge-in


@dataclass(frozen=True)
class MemoryConfig:
    db_path: Path = DATA_DIR / "jesse.db"
    embedder_model: str = "BAAI/bge-small-en-v1.5"  # CPU (fastembed) — must NOT be a GPU model
    recall_k: int = 3                   # strict read budget: top-3 facts per turn
    recent_turns: int = 12              # rolling history kept in context
    context_char_budget: int = 2400     # cap on the recent-turns tail (~600 tokens); keeps
                                        # persona + facts + history well under num_ctx (4096)
    min_fact_confidence: float = 0.6    # gate out low-confidence extracted facts
    catch_up_limit: int = 5             # max unfinished exchanges re-extracted at startup
    # Two facts whose SUBJECTS are this similar are treated as the same slot, so the
    # newer one supersedes instead of both persisting ("birthday_month" vs "birthday
    # month"). Measured cosines on bge-small: birthday_month/birthday month 0.953,
    # dog's name/pet_name 0.903 — but sister's name/brother's name is 0.822 and MUST
    # NOT merge, so the bar sits above that. Deliberately conservative: a missed merge
    # leaves a harmless duplicate, a wrong merge destroys a real fact.
    dedupe_subject_similarity: float = 0.88
    debug_extraction: bool = False       # print the exchange + raw LLM output + parse result
                                        # each extraction (temporary, for trust/debugging)


@dataclass(frozen=True)
class ScheduleConfig:
    tick_seconds: float = 2.0            # how often due reminders are checked
    stale_after_minutes: int = 120       # older than this overdue -> dropped, not announced
    overdue_note_after_seconds: int = 60  # later than this -> he admits how late it is


@dataclass(frozen=True)
class RemoteConfig:
    """Reaching him from his phone, over his own tailnet. `jesse run --remote`.

    Off unless asked for on the command line: this is the one server that binds wider
    than localhost, so it should never start by accident.
    """
    port: int = 8766
    # 0.0.0.0 so the phone can reach it over Tailscale. That is precisely why the
    # token is mandatory rather than optional — see remote/auth.py.
    host: str = "0.0.0.0"
    token_path: Path = DATA_DIR / "remote-token.txt"
    # HTTPS with a certificate Jesse generates himself, because the browser will not
    # give the page a microphone otherwise. `tailscale serve` would give a trusted
    # cert instead, but it publishes this machine's hostname to public Certificate
    # Transparency logs — declined 2026-09-01 for a project whose whole claim is that
    # nothing about him is discoverable. The phone warns once; verify the fingerprint
    # Jesse prints at startup and it becomes a check rather than a shrug.
    tls: bool = True
    cert_dir: Path = DATA_DIR
    # A phone that stops sending for this long has hung up; the desk mic takes over.
    # Long enough to survive a tunnel hiccup, short enough that a pocketed phone does
    # not hold the floor.
    idle_timeout_seconds: float = 12.0
    # Side-effect actions (opening things, media keys) asked for from the phone get a
    # confirmation first. Reading, remembering and timers are unaffected. Reasoning in
    # HANDOFF: a remote channel is the least-tested one, and the same deterministic
    # parsers are reading a transcript from a phone mic in a moving car.
    confirm_actions: bool = True


@dataclass(frozen=True)
class DigestConfig:
    """Sources he reads on a schedule. See jesse/digest/feeds.py for the safety notes.

    **ON — his decision, 2026-08-27**, after shipping it off by default so the choice
    was his to make. The distinction that makes this consistent with the rest of the
    project: his brain, his memory and every piece of processing stay on this machine,
    always. What this adds is an outbound GET for public feeds — the internet as a
    connection, not as a place he lives. Nothing of his is sent, and nothing about his runs anywhere else.
    """
    enabled: bool = True
    # name -> feed url. RSS or Atom only; see feeds.py for why not web pages.
    sources: tuple[tuple[str, str], ...] = (
        ("hacker news", "https://news.ycombinator.com/rss"),
        ("bbc", "https://feeds.bbci.co.uk/news/rss.xml"),
    )
    interval_hours: float = 6.0          # wall-clock, reconciled on start like reminders
    items_per_source: int = 5            # newest N kept per fetch
    max_items_told: int = 3              # most he reads out in one answer
    fetch_timeout: int = 20              # measured: hnrss.org needs >15s, ycombinator ~2s
    max_bytes: int = 2_000_000           # hard cap while reading a response
    # When on, he may MENTION (once, in a line) that something new arrived — but only
    # riding on a reply he was already giving. He never breaks a silence for it; see
    # the reasoning in HANDOFF. Off by default: unprompted is how a feature gets muted.
    nudge: bool = False


@dataclass(frozen=True)
class KnowledgeConfig:
    """What he has read. `python -m jesse learn <name> <path>` fills it."""
    enabled: bool = True
    top_k: int = 2                       # passages considered per turn
    topic_turns: int = 4                 # recent turns also scanned for a subject name,
                                         # so a follow-up need not repeat the word
    chunk_chars: int = 800               # ~200 tokens; two of them still fit num_ctx
    char_budget: int = 1200              # hard cap on what gets injected in one turn
    # Cosine DISTANCE gate. Above this, the closest passage is not actually about what
    # he said, and injecting it would drag a document into a conversation about his day.
    # MEASURED on bge-small against a real ingested document: eight genuine questions
    # about it landed 0.182-0.446, eight ordinary utterances 0.478-0.586. The gate sits
    # just inside that gap. The margin is thin (0.032), and on a wider corpus the two
    # clusters will overlap — when they do, move this DOWN. A missed retrieval is a
    # question he answers without the document; a false one puts a paragraph about
    # guitar strings into a conversation about his day.
    max_distance: float = 0.46


def _default_apps() -> dict[str, str]:
    """What "open X" is allowed to reach. Add a line here to teach his a new one — a
    protocol URL, an exe on PATH, a full path, a folder, or a website.

    A registry rather than "whatever he named": an open list would mean guessing at an
    executable name from speech, and a wrong guess either does nothing or starts
    something he didn't ask for. A miss here is recoverable — he says he doesn't have
    that one and he adds it."""
    home = Path.home()
    return {
        "spotify": "spotify:",
        "chrome": "chrome",
        "edge": "msedge",
        "firefox": "firefox",
        "notepad": "notepad",
        "calculator": "calc",
        "paint": "mspaint",
        "explorer": str(home),
        "files": str(home),
        "file explorer": str(home),
        "downloads": str(home / "Downloads"),
        "documents": str(home / "Documents"),
        "desktop": str(home / "Desktop"),
        "code": "code",
        "vs code": "code",
        "vscode": "code",
        "terminal": "wt",
        "task manager": "taskmgr",
        "settings": "ms-settings:",
        "youtube": "https://www.youtube.com",
        "github": "https://github.com",
        "gmail": "https://mail.google.com",
        "maps": "https://maps.google.com",
        "google maps": "https://maps.google.com",
        "google": "https://www.google.com",
        "drive": "https://drive.google.com",
        "whatsapp": "https://web.whatsapp.com",
    }


@dataclass(frozen=True)
class ActionsConfig:
    enabled: bool = True
    apps: dict[str, str] = field(default_factory=_default_apps)
    # Where "find my ..." looks. Deliberately a short list of his own folders, not the
    # whole drive: a full walk stalls the turn and turns up program files, not his work.
    search_roots: tuple[Path, ...] = field(default_factory=lambda: (
        Path.home() / "Documents", Path.home() / "Desktop", Path.home() / "Downloads",
    ))
    search_limit: int = 5                # most results he will read out
    search_max_depth: int = 4            # folders deep from each root


@dataclass(frozen=True)
class Config:
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    digest: DigestConfig = field(default_factory=DigestConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)


CONFIG = Config()
