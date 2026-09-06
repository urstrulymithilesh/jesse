"""Recognise a spoken request to forget something, deterministically.

Same approach as timers and reminders, and for the same reasons: no extra model
round-trip, fully testable, and no match simply means ordinary conversation.

This closes a trust hole. He would verbally agree — "sure, I'll forget that" — while
the fact stayed in the database, which is worse than refusing, because you believe it
is gone. Now the words and the database agree, or he says he is unsure and asks.

Deletion is destructive, so the rule matches cancellation of a reminder: act on one
clear match, ask when several fit, say so when nothing does. Never guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# An instruction aimed at HIS memory.
_FORGET_VERBS = ("forget", "delete", "erase", "remove", "unlearn", "scrub")
# "stop remembering X" / "no longer remember X" mean the same as "forget X".
_STOP_REMEMBERING = ("stop remembering", "quit remembering",
                     "no longer remember", "stop keeping")
# "I forgot my keys" / "don't forget to eat" are not instructions to delete anything —
# and "don't forget" is closer to a reminder request than a deletion.
_NOT_FORGET = (
    "i forgot", "i forget", "i've forgotten", "don't forget", "do not forget",
    "dont forget", "didn't forget", "never forget", "forget it",
)
# Handled by the reminder parser instead; "forget the timer" is a cancellation.
_SCHEDULE_NOUNS = ("timer", "timers", "reminder", "reminders", "alarm", "alarms", "countdown")

_LEAD = re.compile(
    r"^.*?\b(?:can you |could you |please |you can |i want you to |i'd like you to )?"
    r"(?:" + "|".join(_FORGET_VERBS + _STOP_REMEMBERING) + r")\b",
    re.I,
)
_FILLER_AFTER = re.compile(
    r"^\s*(?:that|about|the fact that|the fact|it that|what i said about|"
    r"everything about|anything about|my|the|i|that i|when i said)\b",
    re.I,
)


@dataclass(frozen=True)
class ForgetCommand:
    """`target` is what he wants gone; empty means he didn't say which."""

    target: str = ""


def parse_forget_command(text: str):
    """-> ForgetCommand | None."""
    if not text or not text.strip():
        return None
    low = text.lower()
    if any(p in low for p in _NOT_FORGET):
        return None
    if any(n in low for n in _SCHEDULE_NOUNS):
        return None                      # a reminder/timer cancellation, not memory
    if not any(re.search(rf"\b{v}\b", low)
               for v in _FORGET_VERBS + _STOP_REMEMBERING):
        return None

    match = _LEAD.match(text)
    if not match:
        return None
    rest = text[match.end():]
    # Strip the connective words between the verb and the actual subject.
    previous = None
    while previous != rest:
        previous = rest
        rest = _FILLER_AFTER.sub("", rest.strip(), count=1)
    target = re.sub(r"[\s,.!?]+", " ", rest).strip(" ,.!?")
    return ForgetCommand(target=target)
