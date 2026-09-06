"""Parse the LLM's fact-extraction output into Facts.

Used in Phase 2 step 2 (the async idle-gap extractor), but the PARSING is a pure
function so it's fully unit-testable now. A 3B model produces messy output — this is
the confidence gate + malformed-output guard the eng review called for: it NEVER
raises and NEVER lets junk into memory. Bad JSON, wrong shape, missing text, or
low-confidence items are dropped; the caller gets a clean list[Fact].

Expected shape from the model: a JSON array of objects, e.g.
    [{"subject": "sister's name", "text": "the user's sister is named Anya",
      "confidence": 0.9}]
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from jesse.core.interfaces import LLM, Fact, Message

# Tune this freely — it's the extractor's instruction, isolated like the persona.
EXTRACTION_PROMPT = """\
You extract durable facts about the user from a short conversation snippet.
Output ONLY a JSON array (no prose, no markdown, no code fence) of objects with keys
"subject", "text", "confidence":
- subject: a short stable key for what the fact is about (e.g. "sister's name", "job",
  "coffee preference").
- text: the fact as a short third-person statement (e.g. "the user's sister is named Anya").
- confidence: 0.0 to 1.0 — how sure you are this is a real, durable fact the user stated
  about themselves.
Only include DURABLE personal facts the USER revealed about THEMSELVES: names, relationships,
preferences, routines, plans, where they live or work, and the like. Do NOT include chit-chat,
your own replies, momentary feelings, or anything you guessed but weren't told. If there are
no such facts, output exactly [].
"""


class FactExtractor:
    """Turns a conversation snippet into raw extraction JSON via the LLM. Kept separate
    from parsing so the (network) call and the (pure) parse are testable in isolation."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def extract(self, exchange: str) -> str:
        messages = [Message("system", EXTRACTION_PROMPT), Message("user", exchange)]
        return "".join(self._llm.chat(messages, stream=False))


def _strip_code_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s[:4].lower() == "json":
            s = s[4:].strip()
    return s


_FIRST_PERSON = re.compile(r"^(i|i'm|i'll|i've|i'd|my|me|we|we're|our)\b", re.IGNORECASE)


def _is_junk(text: str) -> bool:
    """Reject shapes that cannot be a durable third-person fact about the user.

    The prompt already asks for third person and forbids the model's own replies, and the
    model ignores it: the live db had "I'll start practicing the Indian accent." and
    "What would you like to practice?" filed as facts about him. Both are Jesse's own
    speech, and once stored they get recalled back at Mithilesh as his own words. Same
    everywhere else in this project — a small model is unreliable at a rule, so enforce
    the rule in code instead of asking louder.
    """
    return text.endswith("?") or bool(_FIRST_PERSON.match(text))


def parse_extracted_facts(raw: str, *, min_confidence: float = 0.6) -> list[Fact]:
    """Best-effort parse. Returns [] on any malformed input; filters out items with
    no usable text or confidence below the gate."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        data = json.loads(_strip_code_fence(raw))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        return []

    facts: list[Fact] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        if _is_junk(text):
            continue
        conf = item.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            continue
        if conf < min_confidence:
            continue
        subject = item.get("subject")
        subject = subject.strip() if isinstance(subject, str) and subject.strip() else None
        facts.append(Fact(text=text, confidence=float(conf), subject=subject))
    return facts
