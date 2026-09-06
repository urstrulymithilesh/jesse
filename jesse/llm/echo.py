"""EchoLLM — the Phase 0 "stub brain". Proves the pipeline glue before a real model.

It satisfies the SAME `LLM` contract as OllamaLLM, so Phase 1 swaps the real brain
in with a one-line factory change. It just reflects the user's words back warmly —
enough to confirm wake -> STT -> brain -> TTS is wired correctly end to end.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from jesse.core.interfaces import Message


class EchoLLM:
    @property
    def supports_tools(self) -> bool:
        return False

    def chat(self, messages: Sequence[Message], *, stream: bool = True) -> Iterator[str]:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        reply = f"I heard you say: {last_user}" if last_user else "I'm listening."
        if stream:
            for word in reply.split(" "):
                yield word + " "
        else:
            yield reply
