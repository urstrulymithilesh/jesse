"""Actually doing it: open a thing, press a media key, search for a file.

Everything here is non-destructive on purpose. Opening an app, pressing pause and
listing filenames cannot lose anything, so none of it needs a confirmation step.
Deleting, moving and running arbitrary scripts are deliberately NOT here — those are
the actions where a wrong deterministic match would do real damage, and they need the
ask-first treatment the reminder canceller got before they are worth having.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

# Windows virtual key codes for the keys on a media keyboard.
_VK = {
    "play_pause": 0xB3,
    "next": 0xB0,
    "previous": 0xB1,
    "volume_up": 0xAF,
    "volume_down": 0xAE,
    "mute": 0xAD,
}
_KEYEVENTF_KEYUP = 0x0002


class ActionError(RuntimeError):
    """Raised when the action could not be carried out, so he can say so rather than
    claim it worked."""


def open_target(target: str) -> None:
    """Hand `target` to the OS: an exe, a path, a folder, a URL, or a protocol handler
    like `spotify:`. os.startfile is the one call that covers all of them on Windows."""
    try:
        if hasattr(os, "startfile"):
            os.startfile(target)            # noqa: S606 - Windows-only, target from config
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
    except OSError as e:
        raise ActionError(str(e)) from e


def media_key(action: str) -> None:
    """Press a media key system-wide, so it reaches whatever is playing."""
    code = _VK.get(action)
    if code is None:
        raise ActionError(f"no key for {action!r}")
    if not hasattr(ctypes, "windll"):
        raise ActionError("media keys are Windows-only")
    ctypes.windll.user32.keybd_event(code, 0, 0, 0)
    ctypes.windll.user32.keybd_event(code, 0, _KEYEVENTF_KEYUP, 0)


def find_files(needle: str, roots, *, limit: int = 5, max_depth: int = 4) -> list[Path]:
    """Files under `roots` whose name contains every word in `needle`.

    All words rather than any: "notes car" should not return every file with "car" in
    it. Depth-capped and hidden directories skipped, because a full recursive walk of a
    home directory takes long enough to stall a turn.
    """
    words = [w for w in needle.lower().split() if w]
    if not words:
        return []
    hits: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in _walk(root, max_depth):
            name = path.name.lower()
            if all(w in name for w in words):
                hits.append(path)
                if len(hits) >= limit:
                    return hits
    return hits


def _walk(root: Path, max_depth: int):
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:                      # permission denied, vanished, unreadable
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir(follow_symlinks=False):
                if depth < max_depth:
                    stack.append((Path(entry.path), depth + 1))
            else:
                yield Path(entry.path)
