"""PiperSynthesizer — the real v1 TTS, a clean drop-in for StubSynthesizer.

Uses the maintained `piper-tts` package (OHF-Voice/piper1-gpl) via its PYTHON API
— PiperVoice.load() / voice.synthesize() — NOT a piper.exe on PATH. That means a
pip install into the venv "just works": no PATH juggling, no separate espeak-ng DLL
(the package bundles espeak-ng data).

Detection: available iff `piper` imports AND the voice .onnx exists in models/.
The factory picks this over the stub automatically once both are true.

Setup (already done once):
    pip install piper-tts
    python -m piper.download_voices en_US-lessac-medium --download-dir models
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from jesse.config import CONFIG, MODELS_DIR

# Playback interrupt granularity: re-slice Piper's per-sentence chunks into ~90ms
# pieces so a stop-word cuts speech promptly (2048 samples * 2 bytes).
_PLAYBACK_CHUNK_BYTES = 2048 * 2


def _voice_model_path(voice: str | None = None) -> Path:
    return MODELS_DIR / f"{voice or CONFIG.speech.piper_voice}.onnx"


class PiperSynthesizer:
    def __init__(self, *, voice: str | None = None) -> None:
        self._model_path = _voice_model_path(voice)
        self._voice = None       # lazy load (the .onnx is ~60MB)
        self._sample_rate: int | None = None

    @staticmethod
    def is_available(voice: str | None = None) -> bool:
        try:
            import piper  # noqa: F401
        except ImportError:
            return False
        return _voice_model_path(voice).is_file()

    def _ensure(self) -> None:
        if self._voice is not None:
            return
        from piper import PiperVoice

        self._voice = PiperVoice.load(str(self._model_path))
        self._sample_rate = int(self._voice.config.sample_rate)

    @property
    def sample_rate(self) -> int:
        self._ensure()
        assert self._sample_rate is not None
        return self._sample_rate

    def synthesize(self, text: str) -> Iterator[bytes]:
        self._ensure()
        assert self._voice is not None
        for chunk in self._voice.synthesize(text):
            data = chunk.audio_int16_bytes
            for i in range(0, len(data), _PLAYBACK_CHUNK_BYTES):
                yield data[i:i + _PLAYBACK_CHUNK_BYTES]
