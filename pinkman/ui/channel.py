"""The seam between the web UI and the voice loop.

Typed text and spoken audio have to end up in the SAME turn pipeline — same persona,
same memory, same reminders — or there are two Pinkmans with different memories. So the
UI does not get its own path: it drops text into this queue, the orchestrator drains
it in the very loop that handles microphone frames, and from there a typed turn is
indistinguishable from a spoken one.

Thread-safe because the HTTP server runs in its own thread while the orchestrator
lives on the asyncio loop.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Line:
    role: str          # "you" | "pinkman"
    text: str
    via: str           # "voice" | "text"
    at: float = field(default_factory=time.time)

    def as_json(self) -> dict:
        return {"role": self.role, "text": self.text, "via": self.via, "at": self.at}


class TextChannel:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[str] = []
        self._control: str | None = None
        self._transcript: list[Line] = []
        self._speaking = False
        self._state = "starting"
        self._session_id = uuid.uuid4().hex
        self._profile: dict[str, str] = {}

    # -- called from the HTTP thread ---------------------------------------

    def submit(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._pending.append(text)

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            lines = self._transcript[since:]
            return {
                "lines": [l.as_json() for l in lines],
                "total": len(self._transcript),
                "speaking": self._speaking,
                "state": self._state,
                "session_id": self._session_id,
                "profile": dict(self._profile),
            }

    def control(self, action: str) -> bool:
        """Coalesce clicks; never queue a wake that could fire after a later turn."""
        with self._lock:
            allowed = {"wake": {"idle"}, "interrupt": {"speaking", "thinking"}}
            if self._state not in allowed.get(action, set()):
                return False
            self._control = action
            return True

    def take_control(self) -> str | None:
        with self._lock:
            action, self._control = self._control, None
            return action

    # -- called from the asyncio loop --------------------------------------

    def take(self) -> str | None:
        """Next typed message, or None. Non-blocking — polled with audio frames."""
        with self._lock:
            return self._pending.pop(0) if self._pending else None

    def log(self, role: str, text: str, *, via: str = "voice") -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._transcript.append(Line(role, text, via))

    def set_speaking(self, speaking: bool) -> None:
        with self._lock:
            self._speaking = speaking

    def set_state(self, state: str) -> None:
        with self._lock:
            self._state = state

    def configure(self, *, voice: str, brain: str, wake: str) -> None:
        with self._lock:
            self._profile = {"voice": voice, "brain": brain, "wake": wake}

    @property
    def transcript(self) -> list[Line]:
        with self._lock:
            return list(self._transcript)
