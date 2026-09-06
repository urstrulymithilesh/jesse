"""Core capability contracts for Jesse.

Design rule (locked in /plan-eng-review): every interface is defined by a MINIMAL
capability contract, never a vendor's wire format. That is what makes the
"upgrade = swap the impl, not rewrite" guarantee real. When you upgrade hardware
later, you write a new class that satisfies one of these Protocols and change a
config line — nothing else moves.

Nothing in this file has an implementation. These are the contracts Phase 1+ fills.

Audio format convention used across the pipeline: 16 kHz, mono, 16-bit signed PCM,
little-endian, carried as `bytes`. AudioTransport is responsible for capturing at
the device's native rate and resampling to this convention (and back for playback).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Shared value types
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """The reasoning backend failed (unreachable, HTTP error, timeout). Callers
    must handle this so a bad LLM call never hangs the conversation silently."""


@dataclass(frozen=True)
class Message:
    """One turn in a chat exchange. Role is 'system' | 'user' | 'assistant'."""

    role: str
    content: str


@dataclass(frozen=True)
class Fact:
    """A durable thing Jesse has learned about the user."""

    text: str
    confidence: float  # 0..1 — the extractor's confidence; low-confidence facts are gated out
    source_turn_id: int | None = None
    # A short stable key for what this fact is ABOUT (e.g. "sister's name"). Used for
    # last-write-wins conflict resolution — a new fact with the same subject replaces
    # the old one. None means dedupe by exact text instead.
    subject: str | None = None
    # Where the fact came from, which also sets its protection level:
    #   "conversation" — learned from chat; can be overwritten by later extraction (default)
    #   "core"         — seeded identity/relationship facts; extraction may NOT overwrite them
    #   "self"         — seeded facts about Jesse's own current build; extraction-protected
    #   "self_history" — seeded past-version facts; protected AND hidden from normal recall
    origin: str = "conversation"


PROTECTED_ORIGINS = frozenset({"core", "self", "self_history"})


# ---------------------------------------------------------------------------
# Voice I/O
# ---------------------------------------------------------------------------


@runtime_checkable
class AudioTransport(Protocol):
    """The mic/speaker boundary. The v1 impl is local WASAPI in/out.

    This is the seam where the rejected VoIP idea would slot back in as a drop-in
    adapter (a Twilio/WebRTC transport that yields/consumes the same PCM frames),
    if remote access ever justifies relaxing the fully-local privacy line.
    """

    async def capture(self) -> AsyncIterator[bytes]:
        """Yield 16 kHz mono PCM frames from the mic until stopped."""
        ...

    async def play(self, frames: Iterator[bytes], *, sample_rate: int) -> None:
        """Play a stream of PCM frames through the speaker/headset at sample_rate.
        (Input stays 16 kHz; TTS may play at its own native rate, e.g. 22050.)"""
        ...

    def mute_input(self) -> None:
        """Gate the mic (half-duplex invariant: called on entering SPEAKING)."""
        ...

    def unmute_input(self) -> None:
        """Re-open the mic AND flush any buffered frames (kills self-trigger echo)."""
        ...


@runtime_checkable
class WakeWord(Protocol):
    """Always-on, lightweight. Stays live even during Jesse's own speech so a
    stop-word can barge in (headset prevents self-trigger)."""

    def process(self, frame: bytes) -> bool:
        """Return True when the wake/stop word is detected in this frame."""
        ...


@runtime_checkable
class Transcriber(Protocol):
    """Bytes in, text out. v1 impl: faster-whisper int8 on CPU."""

    def transcribe(self, pcm: bytes) -> str:
        ...


@runtime_checkable
class Synthesizer(Protocol):
    """Text in, audio frames out — STREAMING, so playback can start before the
    whole reply is synthesized. v1 impl: Piper (piper-tts) on CPU."""

    @property
    def sample_rate(self) -> int:
        """Native output rate of this voice (e.g. 16000 stub, 22050 Piper medium)."""
        ...

    def synthesize(self, text: str) -> Iterator[bytes]:
        ...


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


@runtime_checkable
class LLM(Protocol):
    """Capability contract only — NOT Ollama's HTTP shape. A raw llama.cpp server
    or a bigger model after a GPU upgrade must satisfy this same signature."""

    @property
    def supports_tools(self) -> bool:
        ...

    def chat(self, messages: Sequence[Message], *, stream: bool = True) -> Iterator[str]:
        """Yield reply tokens. When stream=False, yields once with the full reply."""
        ...


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Text -> vector. v1 impl: a CPU sentence-embedder (fastembed/bge-small).
    MUST run on CPU so it never contends with the resident Qwen for the 4GB."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Structured facts (SQLite) + semantic recall (sqlite-vec). Reads are budgeted:
    callers take the top-K only. Writes happen in the idle gap after a reply, never
    concurrent with the next turn."""

    def add_fact(self, fact: Fact) -> None:
        """Upsert a fact. Conflict policy: last-write-wins on the same subject."""
        ...

    def recall(self, query: str, *, k: int = 3) -> list[Fact]:
        """Top-K facts semantically relevant to the query (the read budget)."""
        ...

    def recent(self, *, limit: int = 20) -> list[Message]:
        """Recent conversation turns, newest last."""
        ...

    def append_turn(self, message: Message) -> int:
        """Persist a raw conversation turn; return its id."""
        ...


# ---------------------------------------------------------------------------
# Skills / tools
# ---------------------------------------------------------------------------


@runtime_checkable
class Tool(Protocol):
    """A single capability Jesse can invoke (timer, reminder, note, ...)."""

    name: str
    description: str

    def run(self, **kwargs: object) -> str:
        """Execute and return a short result string for the LLM to narrate."""
        ...


@runtime_checkable
class ToolRegistry(Protocol):
    def register(self, tool: Tool) -> None:
        ...

    def get(self, name: str) -> Tool | None:
        ...

    def all(self) -> list[Tool]:
        ...
