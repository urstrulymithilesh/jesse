"""FastEmbedEmbedder — the v1 Embedder impl. CPU sentence embeddings via fastembed.

MUST run on CPU so it never contends with the resident Qwen for the 4GB (fastembed
uses onnxruntime + a small bge model; no torch, no GPU). Lazy-loads the model so
tests and startup don't pay for it until the first embed.
"""

from __future__ import annotations

from collections.abc import Sequence

from jesse.config import CONFIG


class FastEmbedEmbedder:
    def __init__(self, *, model_name: str | None = None) -> None:
        self._model_name = model_name or CONFIG.memory.embedder_model
        self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=self._model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self._ensure()
        assert self._model is not None
        return [vec.tolist() for vec in self._model.embed(list(texts))]
