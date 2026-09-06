"""Clean up a raw transcript before anything downstream reads it.

The pre-roll deliberately keeps audio from BEFORE the wake word fired, so the start of
a sentence spoken over the wake word isn't lost. The cost is that the wake word itself
usually lands in the transcript: "Jarvis. Set a timer for four seconds."

That prefix is not harmless. Measured on qwen2.5:3b, it silently breaks fact
extraction:

    "Jarvis. Remember that my favorite color is turquoise."  -> []          (nothing)
    "Remember that my favorite color is turquoise."          -> 1 fact      (correct)

So every turn was quietly handing the extractor a poisoned string. Stripping it is a
pure text operation, done once, before the transcript is used for anything.
"""

from __future__ import annotations

import re

# Filler that often precedes or surrounds the wake word in a transcript. The odd ones
# ("a", "they", "eight") are what whisper actually makes of a half-heard "hey": a live
# smoke run transcribed "hey jarvis" as "8 Jarvis", the junk prefix survived, and the
# action parser missed — whereupon he claimed "Photoshop opens." about nothing.
# Digits are handled in the loop for the same reason. All of these strip ONLY when a
# real wake token follows, so "they said hi" and "8 times 8" are untouched.
_FILLER = {"hey", "hi", "hello", "ok", "okay", "um", "uh", "so", "a", "hay",
           "they", "eight", "hate"}
_LEADING_PUNCT = re.compile(r"^[\s,.!?;:\-—]+")


def _is_filler(word: str) -> bool:
    return word in _FILLER or word.isdigit()


# Words that legitimately open a sentence before a comma and carry meaning. Anything
# NOT in here, sitting alone in front of a comma, is almost certainly his name coming
# back mangled — and every deterministic parser downstream is anchored to the start of
# the utterance, so leaving it there silently disables all of them.
_REAL_OPENERS = frozenset("""
no yes yeah nope sure right wait sorry look listen well actually honestly please
anyway besides however though still also first second finally meanwhile otherwise
""".split())
_VOCATIVE = re.compile(r"^([A-Za-z']{2,})\s*,\s*(?=\S)")


def strip_vocative(text: str) -> str:
    """Drop a leading name-like word before a comma.

    Found live: "hey jarvis" reached the transcript as **"Heeshak, can you open
    Spotify?"**. The wake-prefix stripper could not help — it only acts when a real
    wake token follows — so the action parser, which anchors at the start of the
    utterance, matched nothing. No action ran, and he cheerfully said "I'll open
    Spotify" about something that never happened.

    Only ONE word, only before a comma, and only when it is not a word that opens a
    sentence meaningfully — so "No, open Chrome" and "Sorry, what?" keep their first
    word while "Heeshak," and "Ayesha," are dropped.
    """
    match = _VOCATIVE.match(text or "")
    if not match:
        return text
    word = match.group(1).lower()
    if word in _REAL_OPENERS:
        return text
    return text[match.end():]


def strip_wake_prefix(text: str, wake_model: str) -> str:
    """Remove a leading wake-word utterance ("hey_jarvis" -> "hey", "jarvis").

    Only strips from the FRONT, only wake tokens and filler, and never more than a
    handful of words — so "Jarvis, remind me to call Jarvis back" keeps the second one.
    """
    if not text or not text.strip():
        return text
    wake_tokens = {t.lower() for t in re.split(r"[_\s-]+", wake_model) if t}
    strippable = wake_tokens | _FILLER

    cursor = _LEADING_PUNCT.sub("", text)
    saw_wake_token = False
    consumed = 0
    unknown_before_wake = 0
    limit = len(wake_tokens) + 2          # the wake words themselves, plus a filler or two
    while consumed < limit:
        match = re.match(r"([A-Za-z']+|\d+)", cursor)
        if not match:
            break
        word = match.group(1).lower()
        if word in wake_tokens:
            saw_wake_token = True
        elif (not saw_wake_token and unknown_before_wake == 0
              and len(word) <= 6 and not _is_filler(word)):
            # ONE unrecognised short word is allowed ahead of the wake token, because
            # the list of things whisper makes of "hey" has no end: "8 Jarvis", "A
            # Jarvis", "They Jarvis", "Stay Jarvis", "Meet Jarvis" have all been seen
            # live, each one silently breaking downstream parsing. The wake DETECTOR
            # has already fired on this audio, so the wake word really was spoken —
            # whatever whisper wrote in front of it is that word, mangled. Nothing is
            # stripped unless a genuine wake token follows, which is what keeps
            # ordinary sentences safe.
            unknown_before_wake += 1
        elif not saw_wake_token and _is_filler(word):
            pass                          # leading filler BEFORE the wake word ("hey ...")
        else:
            # Filler only counts ahead of the wake word. Past it, a word like "hello"
            # is what he actually said ("Hey Mycroft, hello").
            break
        cursor = _LEADING_PUNCT.sub("", cursor[match.end():])
        consumed += 1

    # Commit only if an actual WAKE token was there AND real content survives.
    # Without the first condition "hello world" would lose its "hello"; without the
    # second, a bare "Hey Jarvis." would be stripped to nothing and the turn dropped.
    if saw_wake_token and cursor.strip():
        return cursor
    return text
