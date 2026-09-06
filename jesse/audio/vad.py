"""Voice-activity detection = turn endpointing.

The VAD decides when the user has STOPPED talking so we can transcribe. Getting
the silence threshold wrong means either cutting the user off mid-sentence or
waiting awkwardly, so it lives behind its own interface and is unit-tested.

v1 impl: a simple RMS-energy detector — zero extra dependencies, fully
deterministic, easy to explain. Silero VAD is the documented drop-in upgrade
(better in noise); it satisfies the same `Vad` protocol, so swapping it in is a
factory change, not a rewrite.
"""

from __future__ import annotations

import array
from typing import Protocol, runtime_checkable

from jesse.audio.frames import ms_to_chunks


@runtime_checkable
class Vad(Protocol):
    def is_speech(self, frame: bytes) -> bool:
        ...

    def is_endpoint(self, frame: bytes) -> bool:
        """True once we've heard speech and then enough trailing silence."""
        ...

    def reset(self) -> None:
        ...


class EnergyVad:
    """Endpoint after `silence_ms` of trailing quiet, once speech has started.

    State machine (per utterance):
        [no speech yet] --speech--> [in speech] --silence x N--> ENDPOINT
    """

    def __init__(self, *, threshold: float = 500.0, silence_ms: int = 1100,
                 min_speech_ms: int = 300) -> None:
        self._threshold = threshold
        self._silence_needed = ms_to_chunks(silence_ms)
        self._min_speech_needed = ms_to_chunks(min_speech_ms)
        self.reset()

    def reset(self) -> None:
        self._heard_speech = False
        self._speech_frames = 0
        self._trailing_silence = 0

    def set_threshold(self, value: float) -> None:
        """Update the speech/silence RMS boundary (used by auto-calibration)."""
        self._threshold = float(value)

    @property
    def threshold(self) -> float:
        return self._threshold

    @staticmethod
    def _rms(frame: bytes) -> float:
        if not frame:
            return 0.0
        samples = array.array("h")
        samples.frombytes(frame)
        if not samples:
            return 0.0
        return (sum(s * s for s in samples) / len(samples)) ** 0.5

    def is_speech(self, frame: bytes) -> bool:
        return self._rms(frame) >= self._threshold

    def is_endpoint(self, frame: bytes) -> bool:
        if self.is_speech(frame):
            self._heard_speech = True
            self._speech_frames += 1
            self._trailing_silence = 0
            return False
        # silent frame — only end after ENOUGH speech AND ENOUGH trailing silence,
        # so a brief blip can't end a turn and a natural mid-sentence pause is tolerated.
        if self._heard_speech and self._speech_frames >= self._min_speech_needed:
            self._trailing_silence += 1
            if self._trailing_silence >= self._silence_needed:
                return True
        return False
