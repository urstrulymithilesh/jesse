"""Reply-style post-processing — the deterministic complement to the persona prompt.

qwen2.5:3b reflexively ends almost every reply with an engaging follow-up question,
and no amount of prompt wording reliably stops it on a 3B. So we trim it here, but
CAREFULLY: only when the reply already has a real statement and then tacks a question
on the end. A reply that is ONLY a question (e.g. a genuine "what's your sister's
name?", or a recall answer he's checking) is left untouched.

`question_keep_rate` keeps the trailing question some of the time so he still asks
now and then — someone who NEVER asks is as odd as one who always does.
"""

from __future__ import annotations

import random
import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def _is_question(sentence: str) -> bool:
    return sentence.rstrip().endswith("?")


def trim_reflexive_question(text: str, *, keep_rate: float = 0.4, rng: random.Random | None = None) -> str:
    """Drop a trailing question when it follows a real statement (keep it keep_rate of
    the time). Leave single-sentence replies and all-question replies alone."""
    sentences = _sentences(text)
    if len(sentences) < 2:
        return text  # single sentence: a bare statement or a genuine question — leave it

    # Walk back over any trailing run of questions.
    cut = len(sentences)
    while cut > 0 and _is_question(sentences[cut - 1]):
        cut -= 1

    if cut == len(sentences):
        return text          # no trailing question
    if cut == 0:
        return text          # the whole reply is questions — genuine, don't gut it

    r = rng.random() if rng is not None else random.random()
    if r < keep_rate:
        return text          # occasionally let the question stand
    return " ".join(sentences[:cut]).strip()
