"""WhisperTranscriber — the v1 Transcriber impl. faster-whisper int8 on CPU.

Bytes in, text out. Runs on CPU so the GPU stays dedicated to the LLM.
"""

from __future__ import annotations

import numpy as np

from jesse.config import CONFIG


class WhisperTranscriber:
    def __init__(
        self,
        *,
        model: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        cfg = CONFIG.speech
        self._model_name = model or cfg.whisper_model
        self._device = device or cfg.whisper_device
        self._compute_type = compute_type or cfg.whisper_compute_type
        self._model = None  # lazy load

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(self._model_name, device=self._device, compute_type=self._compute_type)

    def transcribe(self, pcm: bytes) -> str:
        self._ensure()
        assert self._model is not None
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return ""
        segments, _info = self._model.transcribe(audio, language="en", vad_filter=False)
        return " ".join(seg.text for seg in segments).strip()
