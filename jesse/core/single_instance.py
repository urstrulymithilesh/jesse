"""One Jesse at a time, or say why not.

Two copies running is not a crash, which is the problem: the second one competes for
the same microphone and simply hears less. It presented as "mic calibration failed
twice and fell back to defaults" — a symptom that looks exactly like a quiet room or a
broken threshold, and sent us looking at the calibration code, which was fine.

A PID file is enough. The point is not to enforce anything; it is to name the cause
instead of leaving a confusing symptom.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _alive(pid: int) -> bool:
    """Is a process with this id still running? Best effort, no dependency."""
    if pid <= 0 or pid == os.getpid():
        return False
    if sys.platform == "win32":
        import ctypes

        SYNCHRONIZE = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True            # exists, owned by someone else
    return True


def claim(path: str | Path) -> str | None:
    """Record this process as the running Jesse.

    Returns None when the coast is clear, or a message naming the other instance. The
    caller decides what to do with it — this never exits on anyone's behalf.
    """
    path = Path(path)
    try:
        if path.is_file():
            existing = path.read_text(encoding="utf-8").strip()
            if existing.isdigit() and _alive(int(existing)):
                return (f"Another Jesse is already running (pid {existing}). He is "
                        f"holding the microphone, so this one will hear almost nothing "
                        f"— calibration will fail and the wake word will not fire. "
                        f"Close the other one first, or run this with --no-mic.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        return None            # an unwritable pid file is not worth blocking a session
    return None


def release(path: str | Path) -> None:
    path = Path(path)
    try:
        if path.is_file() and path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass
