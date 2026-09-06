"""OpenWakeWordDetector — the v1 WakeWord impl.

Stays live even during Jesse's speech so a stop-word can barge in (Finding #3 of
the eng review). A headset prevents self-trigger. The same class serves both the
wake role (IDLE) and the stop role (SPEAKING); config decides which model each uses.

First run downloads the pretrained melspectrogram + embedding models (one-time,
online); after that it is fully offline. Until "yo Jesse" is trained — a separate
future task — a stock word stands in (e.g. "hey_jarvis").
"""

from __future__ import annotations

import numpy as np

from jesse.audio.frames import CHUNK_SAMPLES


class OpenWakeWordDetector:
    def __init__(self, model_name: str, *, threshold: float = 0.5) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._model = None  # lazy: don't pay load cost until first frame

    def _ensure(self) -> None:
        if self._model is not None:
            return
        try:
            from openwakeword.model import Model
        except ImportError as e:  # pragma: no cover - environment guard
            raise RuntimeError("openwakeword not installed (pip install openwakeword)") from e
        try:
            # Pin onnx: the tflite runtime isn't installed on Windows, and letting
            # openWakeWord discover that at load time emits a noisy fallback warning.
            self._model = Model(wakeword_models=[self._model_name], inference_framework="onnx")
        except Exception as e:  # noqa: BLE001 - surface a clear setup hint
            raise RuntimeError(
                f"could not load wake model {self._model_name!r}. First-time setup may need:\n"
                "    python -c \"import openwakeword.utils as u; u.download_models()\""
            ) from e

    def process(self, frame: bytes) -> bool:
        self._ensure()
        assert self._model is not None
        samples = np.frombuffer(frame, dtype=np.int16)
        if len(samples) != CHUNK_SAMPLES:
            return False
        scores = self._model.predict(samples)
        return any(score >= self._threshold for score in scores.values())
