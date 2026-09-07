"""ExperientialLLM — hosted reasoning drop-in. Satisfies the LLM contract.

Talks to the Experiential Labs gateway, which speaks the OpenAI Chat Completions
API: POST {base_url}/chat/completions with a Bearer key, SSE frames back when
streaming. Stdlib only, same as OllamaLLM — no new dependency.

NOTE ON THE DESIGN LINE: every other backend in this package runs on this machine.
This one does not. Prompts — and therefore conversation content — leave the box and
are billed to an Experiential account. So it is strictly opt-in (`run --experiential`)
and Ollama stays the default; the fully-local path is never silently traded away.

The key is read from the environment ($EXPLABS_API_KEY) at construction and is never
written to config or to disk.

Robustness mirrors OllamaLLM: retry once on 5xx / connection blips, raise a clear
LLMError otherwise — never hang.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence

from jesse.config import CONFIG
from jesse.core.interfaces import LLMError, Message


class ExperientialLLM:
    def __init__(self, *, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None) -> None:
        cfg = CONFIG.reasoning
        self._env_var = cfg.experiential_api_key_env
        self._base_url = (base_url or cfg.experiential_base_url).rstrip("/")
        self._model = model or cfg.experiential_model
        self._key = (api_key or os.environ.get(self._env_var, "")).strip()
        if not self._key:
            raise LLMError(
                f"${self._env_var} is not set, so the Experiential brain can't authenticate. "
                f"Create a key under Settings -> API keys at Experiential, then export it:\n"
                f'    setx {self._env_var} "xpl_..."     (then open a new shell)'
            )
        self._temperature = cfg.temperature
        self._max_tokens = cfg.experiential_max_tokens
        self._timeout = cfg.request_timeout

    @property
    def supports_tools(self) -> bool:
        return True

    def _open(self, body: bytes):
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions", data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
        )
        return urllib.request.urlopen(req, timeout=self._timeout)

    def chat(self, messages: Sequence[Message], *, stream: bool = True) -> Iterator[str]:
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }).encode()

        resp = None
        last: LLMError | None = None
        for attempt in range(2):  # one retry for transient 5xx / connection drops
            try:
                resp = self._open(body)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code in (401, 403):
                    raise LLMError(
                        f"Experiential rejected ${self._env_var} (HTTP {e.code}). Check the key "
                        f"is current under Settings -> API keys: {detail}"
                    ) from e
                last = LLMError(f"Experiential returned HTTP {e.code}: {detail}")
                if e.code < 500:
                    raise last  # bad model id / payload — a retry won't help
            except (urllib.error.URLError, TimeoutError) as e:
                last = LLMError(f"Experiential not reachable at {self._base_url} ({e}).")
            if attempt == 0:
                time.sleep(0.6)
        if resp is None:
            raise last or LLMError("Experiential call failed")

        with resp:
            if not stream:
                choices = json.load(resp).get("choices") or [{}]
                yield (choices[0].get("message") or {}).get("content") or ""
                return
            try:
                for raw in resp:
                    line = raw.strip()
                    if not line.startswith(b"data:"):
                        continue           # SSE comments / keep-alives
                    payload = line[len(b"data:"):].strip()
                    if payload == b"[DONE]":
                        break
                    choices = json.loads(payload).get("choices") or [{}]
                    chunk = (choices[0].get("delta") or {}).get("content")
                    if chunk:
                        yield chunk
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                raise LLMError(f"Experiential stream failed mid-reply: {e}") from e
