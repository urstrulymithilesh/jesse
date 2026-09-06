"""The preemption state machine — Jesse's signature concurrency model.

This is the hardest and most interview-worthy part of the project, so it lives in
its own module and is specified explicitly rather than implied by the event loop.

    ┌──── wake word ────► LISTENING ──── VAD endpoint ────► THINKING ────┐
    │                     (full STT)                       (Qwen on GPU) │
   IDLE ◄─────────────────────────────────────────────────── reply done │
    ▲                                                                    ▼
    └──────────── stop-word / reply done ──────────────────────────── SPEAKING
                                                    (Piper + stop-word detector live)

PendingAlert (a fired timer/reminder) is a priority overlay on ANY state:
  • fires while IDLE or SPEAKING  -> speak immediately (Jesse owns the speaker)
  • fires while LISTENING         -> WAIT for the VAD endpoint, then speak
                                     (don't cut the user off mid-word; a
                                      timer/reminder counts as time-critical, so
                                      it interjects at the next silence)
  • fires while THINKING          -> queue, speak right after the reply

Half-duplex invariant: entering SPEAKING mutes full STT; exiting SPEAKING unmutes
AND flushes the mic buffer, so Jesse never hears the tail of his own voice.
"""

from __future__ import annotations

import enum


class ConversationState(enum.Enum):
    IDLE = "idle"            # waiting for the wake word
    LISTENING = "listening"  # full STT open, user is speaking
    THINKING = "thinking"    # LLM generating a reply on the GPU
    SPEAKING = "speaking"    # TTS playing; full STT gated, stop-word still live


class AlertDisposition(enum.Enum):
    """What to do with a fired alert given the current state."""

    SPEAK_NOW = "speak_now"           # IDLE / SPEAKING
    WAIT_FOR_SILENCE = "wait"         # LISTENING
    QUEUE_AFTER_REPLY = "queue"       # THINKING


def disposition_for(state: ConversationState) -> AlertDisposition:
    """Pure decision function — trivially unit-testable, no side effects."""
    if state is ConversationState.LISTENING:
        return AlertDisposition.WAIT_FOR_SILENCE
    if state is ConversationState.THINKING:
        return AlertDisposition.QUEUE_AFTER_REPLY
    return AlertDisposition.SPEAK_NOW  # IDLE or SPEAKING
