"""Deterministic parsing of "when" questions.

Same reasoning as the schedule parser: this decides which slice of real history gets
shown to the model, and a wrong guess is a confabulation risk rather than a slightly
odd reply. "Which day did he mean" is not a judgement a 3B should be making.

Pure functions — a fake `now` is all the tests need.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta


@dataclass(frozen=True)
class TimeWindow:
    start: datetime | None      # None = no lower bound (all of history)
    end: datetime | None        # None = up to now
    label: str                  # how to say it back: "yesterday", "earlier today"

    def contains(self, when: datetime) -> bool:
        if self.start is not None and when < self.start:
            return False
        if self.end is not None and when >= self.end:
            return False
        return True


# "what did we talk about ..." — asking about past CONVERSATION, not a stored fact.
_DISCUSS = (
    "talk about", "talked about", "discuss", "discussed", "speak about", "spoke about",
    "chat about", "chatted about", "conversation", "we say", "we said", "tell me about our",
    "catch me up", "recap", "go over what",
)
# "when did I tell you ..." — asking for the TIME of something.
_WHEN = ("when did i", "when did we", "when was it", "how long ago", "what day did")

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _day_bounds(day: datetime) -> tuple[datetime, datetime]:
    start = datetime.combine(day.date(), time.min)
    return start, start + timedelta(days=1)


def parse_time_window(text: str, *, now: datetime) -> TimeWindow | None:
    """Extract the time range a question refers to, or None if it names no time."""
    low = text.lower()

    if "yesterday" in low:
        start, end = _day_bounds(now - timedelta(days=1))
        return TimeWindow(start, end, "yesterday")
    if "this morning" in low:
        start = datetime.combine(now.date(), time.min)
        return TimeWindow(start, datetime.combine(now.date(), time(12, 0)), "this morning")
    if "last night" in low:
        start = datetime.combine((now - timedelta(days=1)).date(), time(18, 0))
        return TimeWindow(start, datetime.combine(now.date(), time(4, 0)), "last night")
    if "today" in low:
        start, end = _day_bounds(now)
        return TimeWindow(start, end, "today")
    if "last week" in low:
        return TimeWindow(now - timedelta(days=14), now - timedelta(days=7), "last week")
    if "this week" in low:
        return TimeWindow(now - timedelta(days=7), None, "this week")

    match = re.search(r"(\d+)\s*(day|week|hour)s?\s*ago", low)
    if match:
        n, unit = int(match.group(1)), match.group(2)
        delta = {"day": timedelta(days=n), "week": timedelta(weeks=n),
                 "hour": timedelta(hours=n)}[unit]
        point = now - delta
        if unit == "hour":
            return TimeWindow(point - timedelta(hours=1), point + timedelta(hours=1),
                              f"{n} hour{'s' if n > 1 else ''} ago")
        start, end = _day_bounds(point)
        return TimeWindow(start, end, f"{n} {unit}{'s' if n > 1 else ''} ago")

    for i, day in enumerate(_WEEKDAYS):
        if day in low:
            # The most recent past occurrence of that weekday.
            delta = (now.weekday() - i) % 7 or 7
            start, end = _day_bounds(now - timedelta(days=delta))
            return TimeWindow(start, end, f"last {day.capitalize()}")

    if any(w in low for w in ("earlier", "before", "last time", "previously", "a while ago")):
        return TimeWindow(None, None, "earlier")
    return None


def is_conversation_question(text: str) -> bool:
    """True for "what did we talk about ..." — about past TALK, not a stored fact."""
    low = text.lower()
    return any(p in low for p in _DISCUSS)


def is_when_question(text: str) -> bool:
    """True for "when did I tell you my favourite colour" — asking for a time."""
    low = text.lower()
    return any(p in low for p in _WHEN)


def parse_temporal_query(text: str, *, now: datetime) -> TimeWindow | None:
    """The single entry point: does this question want the conversation record?

    Returns the window to look in (an unbounded one means "search all history"), or
    None when the question isn't about past conversation at all.
    """
    if not text or not text.strip():
        return None
    window = parse_time_window(text, now=now)
    if is_conversation_question(text) or is_when_question(text):
        return window or TimeWindow(None, None, "at some point")
    # A bare time reference only counts with a recall verb, so "I went to the gym
    # yesterday" stays an ordinary statement rather than a query about the past.
    if window is not None and any(
            v in text.lower() for v in ("remember", "recall", "did we", "did i", "what was")):
        return window
    return None
