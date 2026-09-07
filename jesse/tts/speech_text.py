"""clean_for_speech — make model output safe to speak aloud.

Even with a good persona prompt, a 3B model sometimes emits a bullet list, markdown
emphasis, a heading, or an *action* aside. Read literally by Piper those sound wrong
("asterisk smiles asterisk"). This is a pure, formatting-only pass: it flattens
structure into a natural spoken line. It deliberately does NOT touch word choice or
truncate content — brevity is the persona's job, not the cleaner's.
"""

from __future__ import annotations

import re

# List markers at the start of a line: "- ", "* ", "+ ", "• ", "1. ", "2) "
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_HEADING = re.compile(r"^\s*#{1,6}\s*")
_EMPHASIS = re.compile(r"[*_`#]+")            # **bold**, _italics_, `code`, # heads
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF\U0001F1E6-\U0001F1FF]"
)
_WS = re.compile(r"\s+")


# The persona's examples are labelled "Jesse: ...", and the model sometimes copies the
# label into the reply. Spoken aloud it sounds like he is reading a script.
_SELF_LABEL = re.compile(r"^\s*jesse\s*[:\-]\s*", re.I)


def clean_for_speech(text: str) -> str:
    if not text:
        return ""
    text = _SELF_LABEL.sub("", text)
    lines = []
    for raw in text.splitlines():
        s = _HEADING.sub("", raw)
        s = _LIST_MARKER.sub("", s)
        lines.append(s.strip())
    text = " ".join(s for s in lines if s)      # paragraphs/lists -> one spoken line
    text = _EMPHASIS.sub("", text)              # drop markdown/asterisk characters
    text = _EMOJI.sub("", text)                 # Piper mispronounces emoji
    return _WS.sub(" ", text).strip()
