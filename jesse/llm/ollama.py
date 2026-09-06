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
    def __init__(self, *, host: str | None = None, model: str | None = None) -> None:
        cfg = CONFIG.reasoning
        self._host = host or cfg.ollama_host
        self._model = model or cfg.model
        self._keep_alive = cfg.keep_alive          # int -1
        self._num_ctx = cfg.num_ctx
        self._temperature = cfg.temperature
        self._timeout = cfg.request_timeout

    @property
    def supports_tools(self) -> bool:
        return True

    def _open(self, body: bytes):
        req = urllib.request.Request(
            f"{self._host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        return urllib.request.urlopen(req, timeout=self._timeout)

    def chat(self, messages: Sequence[Message], *, stream: bool = True) -> Iterator[str]:
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "keep_alive": self._keep_alive,
            "options": {"num_ctx": self._num_ctx, "temperature": self._temperature},
        }).encode()

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
