"""Turn "open Spotify" / "find my tax notes" / "next track" into a concrete action.

Deterministic (regex + a registry), NOT LLM tool-calling. That was the open decision
when this step was reached, and the reasons are the same ones that made the scheduler
deterministic: an LLM round-trip costs 3-7s on this CPU-bound setup, a 3B's structured
output is the exact thing that has proved unreliable here over and over, and picking
the wrong action fails the way a wrongly-cancelled reminder fails — silently, and only
noticed later. A closed set of phrasings that always works beats an open set that
mostly works when the failure mode is invisible.

The cost is honest and stated: he only understands the phrasings written here. When
he asks for something outside them, nothing matches, and it is treated as ordinary
conversation rather than guessed at.

Pure functions only — no os calls, no registry lookups beyond the dict passed in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenCommand:
    """Open something in the registry. `name` is what he called it, `target` is what
    the machine gets."""
    name: str
    target: str


@dataclass(frozen=True)
class UnknownTarget:
    """He asked to open something that isn't in the registry. Deliberately a command
    and not a None: silently falling through to chat would have his cheerfully agree
    he opened it. He has to say he doesn't have that one."""
    name: str


@dataclass(frozen=True)
class MediaCommand:
    action: str          # play_pause | next | previous | volume_up | volume_down | mute


@dataclass(frozen=True)
class FindCommand:
    needle: str          # the words a filename must contain, space-separated


ActionCommand = OpenCommand | UnknownTarget | MediaCommand | FindCommand

# Anything with an explicit reminder word belongs to the scheduler, which runs first.
# Kept here too so the parsers can never fight over one sentence if that order changes:
# "remind me to open the report at five" is a reminder, not an open.
_SCHEDULE_WORDS = ("timer", "alarm", "remind", "reminder")

_POLITE = re.compile(r"^(?:hey\s+|ok\s+|okay\s+)?(?:could you|can you|would you|will you|"
                     r"please|i want you to|i'd like you to)\s+", re.I)
_TRAILING_POLITE = re.compile(r"[,\s]+please\b[.!?]*$", re.I)

# "run" is deliberately absent: "run me through it" would become a request to open
# something called "me through it". The verbs kept here are ones that in the imperative
# almost only ever mean "start this program".
_OPEN_VERBS = r"open(?:\s+up)?|launch|start(?:\s+up)?|fire\s+up|pull\s+up|bring\s+up"
_OPEN = re.compile(rf"^(?:{_OPEN_VERBS})\s+(?P<name>.+)$", re.I)

# Weaker verbs, added after a live probe where "put on Spotify", "show me my downloads"
# and "get me YouTube" all fell through. They only ever mean "open this" when what
# follows is something he can actually open, so unlike the verbs above they may NOT
# produce an UnknownTarget — "get me a coffee" has to stay ordinary talk.
_SOFT_OPEN_VERBS = r"show\s+me|put\s+on|get\s+me|give\s+me|find\s+me"
_SOFT_OPEN = re.compile(rf"^(?:{_SOFT_OPEN_VERBS})\s+(?P<name>.+)$", re.I)

# "find my notes on the car" -> notes car. `find out` is a question, not a file search.
_FIND = re.compile(r"^(?:find|search\s+for|look\s+for|dig\s+up|where\s+(?:is|are)|where's)\s+"
                   r"(?P<needle>.+)$", re.I)
_FIND_NOISE = ("my", "the", "a", "an", "some", "any", "file", "files", "document",
               "documents", "folder", "on", "about", "for", "of", "called", "named",
               "titled", "with", "regarding", "somewhere", "me", "us", "please")
# Words that are never a filename. "can you find some time for me" reduced to "time"
# and became a file search in a live probe — harmless in effect, but it is the wrong
# branch, and the wrong branch is how a feature stops being trusted. A needle made of
# nothing but these is not a search.
_NOT_A_FILENAME = frozenset({
    "time", "someone", "somebody", "something", "anything", "everything", "nothing",
    "way", "ways", "reason", "excuse", "room", "space", "peace", "help", "work",
    "energy", "courage", "one", "it", "them", "him", "his", "you", "yourself",
})

# Media control matches only when the whole utterance IS the command. A substring rule
# would fire on "we should play chess later"; there is no ambiguity to resolve and no
# way to notice the mistake, so the match is anchored at both ends instead.
_MEDIA: tuple[tuple[str, str], ...] = (
    (r"(?:play|resume|unpause)(?:\s+(?:it|the\s+)?(?:music|song|track|audio|video))?", "play_pause"),
    (r"(?:pause|hold)(?:\s+(?:it|the\s+)?(?:music|song|track|audio|video))?", "play_pause"),
    (r"stop\s+(?:it|the\s+)?(?:music|song|track|audio|video|playback)", "play_pause"),
    # "play the next one" reads as play, means skip. Checked before the bare next/skip
    # rule would matter because the play_pause patterns above cannot match it.
    (r"(?:play|skip\s+to)\s+the\s+next(?:\s+(?:song|track|one))?", "next"),
    (r"(?:next|skip)(?:\s+(?:this|the))?(?:\s+(?:song|track|one))?", "next"),
    (r"(?:previous|last|go\s+back(?:\s+a)?)(?:\s+(?:song|track|one))?", "previous"),
    (r"(?:turn\s+(?:it|the\s+volume)\s+up|volume\s+up|louder|turn\s+up(?:\s+the\s+volume)?)",
     "volume_up"),
    (r"(?:turn\s+(?:it|the\s+volume)\s+down|volume\s+down|quieter|turn\s+down"
     r"(?:\s+the\s+volume)?)", "volume_down"),
    (r"(?:mute|unmute|silence\s+(?:it|the\s+\w+))", "mute"),
)
_MEDIA_COMPILED = tuple((re.compile(rf"^{pat}$", re.I), action) for pat, action in _MEDIA)


def _strip(text: str) -> str:
    s = text.strip()
    s = _TRAILING_POLITE.sub("", s)
    s = _POLITE.sub("", s)
    return s.strip().strip(".!,").strip()


def _normalise(name: str) -> str:
    """Registry lookup key: lowercase, no punctuation, no filler. Whisper likes to add
    a full stop, and he might say "the Spotify app"."""
    s = name.lower().strip().strip(".!?,")
    s = re.sub(r"^(?:my|the)\s+", "", s)
    # Trailing filler, each one from a live probe miss: "open my documents folder" and
    # "start Spotify for me" both became unknown targets purely because of the tail.
    # Looped, because they stack — "the Spotify app now" has two.
    while True:
        trimmed = re.sub(r"\s+(?:for\s+(?:me|us))$", "", s)
        trimmed = re.sub(r"\s+(?:app|application|program|window|folder|please|now)$",
                         "", trimmed)
        if trimmed == s:
            return s.strip()
        s = trimmed


def _find_needle(raw: str) -> str:
    words = [w for w in re.split(r"[\s,]+", raw.lower().strip(".!?,"))
             if w and w not in _FIND_NOISE]
    return " ".join(words)


def looks_like_an_action(text: str, apps: dict[str, str]) -> str | None:
    """Names something he can act on, WITHOUT parsing as a command.

    This is the safety net under a parser that anchors at the start of the utterance.
    Live, "hey jarvis" was transcribed as "Heeshak," and the whole utterance —
    "Heeshak, can you open Spotify?" — matched nothing, so no action ran and he
    improvised "I'll open Spotify." about something that never happened.

    Deliberately narrow: an open verb AND a name from the registry, both present. A
    registry name after an open verb is a request in almost every sentence anyone
    actually says, and the cost of a false positive here is one clarifying question,
    not an app opening.
    """
    low = text.lower()
    if not re.search(rf"\b(?:{_OPEN_VERBS}|{_SOFT_OPEN_VERBS})\b", low):
        return None
    for name in sorted(apps, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name.lower())}\b", low):
            return name
    return None


def parse_action_command(text: str, apps: dict[str, str]) -> ActionCommand | None:
    """Return the action he asked for, or None when this was ordinary conversation."""
    if not text or not text.strip():
        return None
    low = text.lower()
    if any(w in low for w in _SCHEDULE_WORDS):
        return None

    stripped = _strip(text)

    for pattern, action in _MEDIA_COMPILED:
        if pattern.match(stripped):
            return MediaCommand(action)

    m = _OPEN.match(stripped)
    if m:
        name = _normalise(m.group("name"))
        if not name:
            return None
        target = apps.get(name)
        if target is not None:
            return OpenCommand(name=name, target=target)
        # An unrecognised name only counts as "I can't open that" when it is short
        # enough to BE a name. "start over with what you were saying" opens nothing
        # and is not worth answering as though he asked for a program.
        if len(name.split()) <= 3:
            return UnknownTarget(name=name)
        return None

    m = _SOFT_OPEN.match(stripped)
    if m:
        target = apps.get(_normalise(m.group("name")))
        if target is not None:
            return OpenCommand(name=_normalise(m.group("name")), target=target)
        # Deliberately falls through rather than returning: "find me the invoice" is a
        # soft-open miss and a perfectly good file search.

    m = _FIND.match(stripped)
    if m:
        raw = m.group("needle")
        if re.match(r"^out\b", raw, re.I):     # "find out whether..." is a question
            return None
        needle = _find_needle(raw)
        if not needle:
            return None
        if all(w in _NOT_A_FILENAME for w in needle.split()):
            return None
        return FindCommand(needle)

    return None
