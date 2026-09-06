"""Turn "remind me to stretch in 20 minutes" into a concrete fire time.

Deliberately deterministic (regex + datetime), NOT an LLM call. Three reasons:
every extra LLM round-trip costs 3-7s on this CPU-bound setup; a 3B's structured
output is exactly what proved unreliable for fact grounding; and timer phrasings
are a small closed set. No match simply means "this was ordinary conversation",
so nothing breaks when someone says something unexpected.

Pure functions only — fully testable with a fake `now`, no clock, no db, no model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

# "in 10 minutes", "in an hour", "in half an hour", "for 30 seconds"
_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45, "fifty": 50, "sixty": 60,
}
_UNIT_SECONDS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}

_RELATIVE = re.compile(
    r"\b(?:in|after|for)\s+"
    r"(?P<half>half\s+(?:an?\s+)?)?"
    r"(?P<qty>\d+|" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")?\s*"
    r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)\b",
    re.I,
)

# "at 5pm", "at 5:30 pm", "at 17:00"
_ABSOLUTE = re.compile(
    r"\bat\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>am|pm|a\.m\.|p\.m\.)?\b",
    re.I,
)

# Phrases that mean "this is a scheduling request", stripped to leave the task.
_TRIGGERS = (
    "could you remind me to", "can you remind me to", "please remind me to",
    "remind me to", "remind me", "set a timer for", "set a timer", "start a timer for",
    "start a timer", "set an alarm for", "set an alarm", "wake me up", "wake me",
    "give me a nudge to", "nudge me to", "ping me to", "tell me to", "let me know to",
)
_TIMER_WORDS = ("timer", "alarm")


# Cancelling / rescheduling an EXISTING reminder. These must be recognised BEFORE a
# creation request, because "stop the timer set for 10 minutes" contains a perfectly
# good "10 minutes" and would otherwise create a second timer instead of removing one.
_CANCEL_VERBS = (
    "cancel", "stop", "never mind", "nevermind", "forget", "delete", "remove",
    "clear", "call off", "scrap", "drop",
)
_RESCHEDULE_VERBS = (
    "change", "make it", "move", "reschedule", "push", "shift", "update", "set it",
    "adjust", "instead",
)
# A reminder-ish noun keeps "stop working at 5" from looking like a cancellation.
_REMINDER_NOUNS = ("timer", "reminder", "alarm", "countdown", "it", "that", "them", "all of them")


# "timer"/"reminder" said outright. The loose pronouns in _REMINDER_NOUNS are fine
# once a TIME anchors the sentence, but for asking or for an incomplete request there
# is no anchor — "I might change it later" must not look like a reminder command.
_EXPLICIT_NOUNS = ("timer", "timers", "reminder", "reminders", "alarm", "alarms", "countdown")

# "do I have any timers", "what's my timer set for", "when does it go off"
_QUERY_PATTERNS = (
    "do i have any", "do i have a", "any timers", "any reminders", "what timers",
    "what reminders", "is there a timer", "are there any", "what's my timer",
    "whats my timer", "what is my timer", "when does my timer", "when does the timer",
    "how long is left", "how long left", "how much time is left", "what am i waiting for",
    "list my", "show my", "check my", "still running", "still going", "what's pending",
    "whats pending", "what's set", "whats set",
)


@dataclass(frozen=True)
class QueryCommand:
    """Ask what's currently pending. Pure lookup — no model needed to find the facts."""


@dataclass(frozen=True)
class IncompleteCommand:
    """A reschedule with no new time ("change the timer"). We don't slot-fill across
    turns; he just asks for the missing piece so the request isn't silently dropped."""

    kind: str = "reschedule"


@dataclass(frozen=True)
class CancelCommand:
    """Cancel a pending reminder. `hint` may name which one ("the gym one")."""

    hint: str = ""
    all_of_them: bool = False


@dataclass(frozen=True)
class RescheduleCommand:
    """Move an existing pending reminder to a new time."""

    fire_at: datetime
    spoken_delay: str
    hint: str = ""


@dataclass(frozen=True)
class ScheduleRequest:
    task: str            # what to say when it fires ("" for a bare timer)
    fire_at: datetime    # absolute wall-clock target
    is_timer: bool       # a countdown ("timer for 10 min") vs a reminder ("at 5pm")
    spoken_delay: str    # human phrasing of the wait, for his confirmation


def _quantity(m: re.Match) -> float | None:
    if m.group("half"):
        return 0.5
    raw = m.group("qty")
    if raw is None:
        return None
    if raw.isdigit():
        return float(raw)
    return float(_WORD_NUMBERS.get(raw.lower(), 0)) or None


def _unit_seconds(unit: str) -> int:
    u = unit.lower().rstrip("s")
    return _UNIT_SECONDS.get(u, 60)


def _clean_task(text: str, *, spans: list[tuple[int, int]]) -> str:
    """Remove the time expression, then the trigger phrase, leaving the task."""
    out = text
    for start, end in sorted(spans, reverse=True):
        out = out[:start] + " " + out[end:]
    low = out.lower()
    for trig in _TRIGGERS:                      # longest-first via _TRIGGERS ordering
        idx = low.find(trig)
        if idx != -1:
            out = out[:idx] + " " + out[idx + len(trig):]
            low = out.lower()
            break
    out = re.sub(r"^\s*(?:to|for|that|about)\b", " ", out.strip(), flags=re.I)
    out = re.sub(r"\b(?:please|hey|isha|ok|okay)\b", " ", out, flags=re.I)
    out = re.sub(r"[\s,.!?]+", " ", out).strip(" ,.!?")
    return out


def _phrase_delay(seconds: float) -> str:
    if seconds == 60:
        return "a minute"          # he SAYS this; "in 60 seconds" sounds robotic
    if seconds < 90:
        return f"{int(round(seconds))} seconds"
    minutes = seconds / 60
    if minutes < 90:
        n = int(round(minutes))
        return "a minute" if n == 1 else f"{n} minutes"
    hours = minutes / 60
    n = round(hours * 2) / 2
    return "an hour" if n == 1 else f"{n:g} hours"


def parse_schedule_request(text: str, *, now: datetime) -> ScheduleRequest | None:
    """Return a ScheduleRequest if `text` asks for a timer/reminder, else None."""
    if not text or not text.strip():
        return None
    low = text.lower()

    rel = _RELATIVE.search(text)
    qty = _quantity(rel) if rel else None
    if rel is not None and qty:
        seconds = qty * _unit_seconds(rel.group("unit"))
        fire_at = now + timedelta(seconds=seconds)
        task = _clean_task(text, spans=[rel.span()])
        is_timer = any(w in low for w in _TIMER_WORDS) or not task
        return ScheduleRequest(task, fire_at, is_timer, _phrase_delay(seconds))

    absol = _ABSOLUTE.search(text)
    if absol is not None:
        # Only treat "at 5" as scheduling if the sentence actually asks for one.
        if not any(t in low for t in _TRIGGERS):
            return None
        hour = int(absol.group("hour"))
        minute = int(absol.group("minute") or 0)
        mer = (absol.group("meridiem") or "").replace(".", "").lower()
        if mer == "pm" and hour < 12:
            hour += 12
        elif mer == "am" and hour == 12:
            hour = 0
        elif not mer and hour < 8:
            hour += 12          # bare "at 5" almost always means this afternoon/evening
        if hour > 23 or minute > 59:
            return None
        fire_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if fire_at <= now:
            fire_at += timedelta(days=1)        # already passed today -> tomorrow
        task = _clean_task(text, spans=[absol.span()])
        delay = _phrase_delay((fire_at - now).total_seconds())
        return ScheduleRequest(task, fire_at, False, delay)

    return None


def announcement(task: str, *, is_timer: bool, overdue_seconds: float = 0.0) -> str:
    """What he should say when it fires — with an honest late note if overdue."""
    if is_timer or not task:
        body = "your timer is up"
    else:
        body = f"time to {task}" if not task.lower().startswith("time") else task
    if overdue_seconds > 0:
        body += f" — this was due {_phrase_delay(overdue_seconds)} ago, sorry"
    return body


def _mentions(text: str, phrases) -> str | None:
    low = text.lower()
    for p in phrases:
        if p in low:
            return p
    return None


def _hint_from(text: str, *, drop_spans=()) -> str:
    """Whatever's left after removing command words and the time expression — used to
    pick WHICH reminder was meant ("cancel the gym one" -> "gym")."""
    out = text
    for start, end in sorted(drop_spans, reverse=True):
        out = out[:start] + " " + out[end:]
    noise = (list(_CANCEL_VERBS) + list(_RESCHEDULE_VERBS) + list(_REMINDER_NOUNS)
             + ["the", "my", "a", "an", "to", "for", "please", "can you", "could you",
                "set", "that was", "which", "one", "jesse", "hey", "about", "instead", "of"])
    low = out.lower()
    for word in sorted(noise, key=len, reverse=True):
        low = re.sub(rf"\b{re.escape(word)}\b", " ", low)
    return re.sub(r"[\s,.!?]+", " ", low).strip(" ,.!?")


# A reschedule states the new time loosely: "change it TO 1 minute", or bare
# "make it 5 minutes instead". No "in/for" prefix required — safe here because a
# change verb and a reminder noun have already been matched.
_RELATIVE_LOOSE = re.compile(
    r"\b(?:to|in|after|for|by)?\s*"
    r"(?P<half>half\s+(?:an?\s+)?)?"
    r"(?P<qty>\d+|" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")?\s*"
    r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)\b",
    re.I,
)


# Same idea for clock times: a reschedule says "move it TO 6pm", not "at 6pm".
_ABSOLUTE_LOOSE = re.compile(
    r"\b(?:to|at|by|for)?\s*(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<meridiem>am|pm|a\.m\.|p\.m\.)\b",
    re.I,
)


def _new_time_from(text: str, *, now: datetime):
    """Reuse the creation-time matchers to read the NEW time in a reschedule."""
    rel = _RELATIVE_LOOSE.search(text)
    qty = _quantity(rel) if rel else None
    if rel is not None and qty:
        seconds = qty * _unit_seconds(rel.group("unit"))
        return now + timedelta(seconds=seconds), _phrase_delay(seconds), rel.span()
    absol = _ABSOLUTE_LOOSE.search(text)
    if absol is not None:
        hour = int(absol.group("hour"))
        minute = int(absol.group("minute") or 0)
        mer = (absol.group("meridiem") or "").replace(".", "").lower()
        if mer == "pm" and hour < 12:
            hour += 12
        elif mer == "am" and hour == 12:
            hour = 0
        elif not mer and hour < 8:
            hour += 12
        if hour > 23 or minute > 59:
            return None
        fire_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if fire_at <= now:
            fire_at += timedelta(days=1)
        return fire_at, _phrase_delay((fire_at - now).total_seconds()), absol.span()
    return None


def parse_schedule_command(text: str, *, now: datetime):
    """The single entry point. Returns CancelCommand | RescheduleCommand |
    ScheduleRequest | None.

    Order matters and is the whole point: a cancellation often names a duration
    ("stop the timer set for 10 minutes"), so checking "create" first would answer a
    cancel request by creating another timer — the exact bug this replaces.
    """
    if not text or not text.strip():
        return None
    low = text.lower()
    has_noun = _mentions(text, _REMINDER_NOUNS) is not None
    has_explicit = _mentions(text, _EXPLICIT_NOUNS) is not None

    # 0. asking what's pending — checked first so "do I have a timer for 10 minutes?"
    #    is read as a question, not as a request to set one.
    if has_explicit and _mentions(text, _QUERY_PATTERNS):
        return QueryCommand()

    # 1. cancel
    if has_noun and _mentions(text, _CANCEL_VERBS):
        every = any(w in low for w in ("all of them", "all the", "everything", "them all"))
        return CancelCommand(hint=_hint_from(text), all_of_them=every)

    # 2. reschedule — needs a change verb AND a new time to move to
    if has_noun and _mentions(text, _RESCHEDULE_VERBS):
        found = _new_time_from(text, now=now)
        if found is not None:
            fire_at, delay, span = found
            return RescheduleCommand(fire_at=fire_at, spoken_delay=delay,
                                     hint=_hint_from(text, drop_spans=[span]))
        # "change the timer" with no time. Only when the noun is EXPLICIT, so an
        # ordinary "I might change it later" stays ordinary conversation.
        if has_explicit:
            return IncompleteCommand("reschedule")

    # 3. otherwise it may be a brand-new timer/reminder
    return parse_schedule_request(text, now=now)
