"""The second lock on the remote door.

Tailscale is the first: a device that is not on his tailnet cannot open a socket to
this machine at all, and that is genuinely strong. This exists because relying on it
alone means one misconfiguration — an accidental Funnel, a node shared with someone —
is total access with nothing behind it. The same defence-in-depth reasoning the rest
of this project uses: a wrong guess on a destructive action fails silently, and
"someone can open apps on my computer" is about as destructive as this project gets.

The token is generated once, stored beside the database with owner-only permissions,
and typed into the phone one time.
"""

from __future__ import annotations

import os
import secrets
import stat
import threading
import time
from pathlib import Path

TOKEN_BYTES = 32          # 256 bits; typed once, then kept in the phone's localStorage
_LOCKOUT_AFTER = 5        # consecutive failures from one address
_LOCKOUT_SECONDS = 300


class TokenError(RuntimeError):
    """The token file could not be read or created."""


def load_or_create(path: str | Path) -> str:
    """Read the token, creating one on first use. Returns the token string."""
    path = Path(path)
    try:
        if path.is_file():
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token
        path.parent.mkdir(parents=True, exist_ok=True)
        token = secrets.token_urlsafe(TOKEN_BYTES)
        path.write_text(token, encoding="utf-8")
        # Owner-only. Best effort: on Windows this is close to a no-op, but the file
        # lives beside the database, which is already gitignored and private.
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        return token
    except OSError as e:
        raise TokenError(f"could not read or create {path}: {e}") from e


class RemoteAuth:
    """Constant-time token checks, with a lockout per client address.

    Thread-safe: the HTTP server runs its own threads while the orchestrator lives on
    the asyncio loop.
    """

    def __init__(self, token: str, *, lockout_after: int = _LOCKOUT_AFTER,
                 lockout_seconds: int = _LOCKOUT_SECONDS) -> None:
        self._token = token
        self._lockout_after = lockout_after
        self._lockout_seconds = lockout_seconds
        self._lock = threading.Lock()
        self._failures: dict[str, tuple[int, float]] = {}   # address -> (count, until)
        self.rejections = 0

    def locked_out(self, address: str, *, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            count, until = self._failures.get(address, (0, 0.0))
            return count >= self._lockout_after and now < until

    def verdict(self, presented: str | None, address: str = "?",
                *, now: float | None = None) -> str:
        """"ok" | "missing" | "wrong" | "locked".

        Separate from `check` because "you sent no token" and "you sent the wrong
        token" need different words on the phone. Told only that it was bad, he
        re-typed a 43-character string that was never the problem — the URL had simply
        lost its `?t=` tail.
        """
        if self.locked_out(address, now=now):
            return "locked"
        if not presented:
            return "missing"
        return "ok" if self.check(presented, address, now=now) else "wrong"

    def check(self, presented: str | None, address: str = "?",
              *, now: float | None = None) -> bool:
        """True when the token matches and the address is not locked out.

        `secrets.compare_digest` rather than `==`: a plain comparison returns early on
        the first wrong byte, and over a network that difference is measurable.
        """
        now = time.time() if now is None else now
        if self.locked_out(address, now=now):
            return False
        ok = bool(presented) and secrets.compare_digest(str(presented), self._token)
        with self._lock:
            if ok:
                self._failures.pop(address, None)
            else:
                self.rejections += 1
                count = self._failures.get(address, (0, 0.0))[0] + 1
                self._failures[address] = (count, now + self._lockout_seconds)
                if count == self._lockout_after:
                    print(f"  [remote] {count} bad tokens from {address} — "
                          f"locked out for {self._lockout_seconds // 60} minutes")
        return ok


def token_from_request(headers, path: str) -> str | None:
    """Pull the token from a header, or from the query string on the first page load.

    The header is how the page talks once it is running. The query string exists only
    so the very first visit can carry it (`?t=…`) from a QR code or a pasted link —
    the page stores it and strips it from the address bar immediately.
    """
    header = headers.get("X-Jesse-Token")
    if header:
        return header.strip()
    if "t=" in path:
        return path.split("t=", 1)[1].split("&")[0].strip() or None
    return None
