"""OllamaLLM — the real reasoning drop-in. Satisfies the LLM contract.

Talks to the local Ollama server over HTTP (stdlib only) and streams reply tokens.

Robustness (learned from a real HTTP 500): the FIRST request after a cold start can
fail transiently while Ollama loads the model (on a 4GB GPU it may fail the GPU fit,
500, then fall back to CPU). So we retry once on 5xx / connection errors, and raise a
clear LLMError otherwise — never hang. keep_alive is an int (-1 = resident); the
string "-1" is rejected with a 400.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence

from jesse.config import CONFIG
from jesse.core.interfaces import LLMError, Message


class OllamaLLM:
    def __init__(self, *, host: str | None = None, model: str | None = None,
                 response_schema: dict | None = None, temperature: float | None = None) -> None:
        cfg = CONFIG.reasoning
        self._host = host or cfg.ollama_host
        self._model = model or cfg.model
        self._keep_alive = cfg.keep_alive          # int -1
        self._num_ctx = cfg.num_ctx
        self._temperature = cfg.temperature if temperature is None else temperature
        self._response_schema = response_schema
        self._timeout = cfg.request_timeout

    @property
    def supports_tools(self) -> bool:
        return True

    def _open(self, body: bytes):
        req = urllib.request.Request(
            f"{self._host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        return urllib.request.urlopen(req, timeout=self._timeout)

    def check_available(self, *, timeout: float = 2.0) -> None:
        """Cheap startup probe; no generation, model loading or retry delay."""
        try:
            with urllib.request.urlopen(f"{self._host}/api/tags", timeout=timeout) as response:
                data = json.load(response)
            if not isinstance(data, dict) or not isinstance(data.get("models"), list):
                raise ValueError("unexpected model-list response")
        except (OSError, ValueError) as exc:
            raise LLMError(
                f"Ollama is not available at {self._host}. Start Ollama first "
                "(open the Ollama app or run 'ollama serve'), then start Jesse again. "
                f"Details: {exc}"
            ) from exc

    def chat(self, messages: Sequence[Message], *, stream: bool = True) -> Iterator[str]:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "keep_alive": self._keep_alive,
            "options": {"num_ctx": self._num_ctx, "temperature": self._temperature},
        }
        if self._response_schema is not None:
            payload["format"] = self._response_schema
        body = json.dumps(payload).encode()

        resp = None
        last: LLMError | None = None
        for attempt in range(2):  # one retry for transient cold-load 5xx / connection drops
            try:
                resp = self._open(body)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                last = LLMError(f"Ollama returned HTTP {e.code}: {detail}")
                if e.code < 500:
                    raise last  # client error (bad model/tag/payload) — retry won't help
            except (urllib.error.URLError, TimeoutError) as e:
                last = LLMError(f"Ollama not reachable at {self._host} ({e}). Is it running?")
            if attempt == 0:
                time.sleep(0.6)
        if resp is None:
            raise last or LLMError("Ollama call failed")

        with resp:
            if not stream:
                yield json.load(resp)["message"]["content"]
                return
            try:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("done"):
                        break
                    yield obj.get("message", {}).get("content", "")
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                raise LLMError(f"Ollama stream failed mid-reply: {e}") from e
