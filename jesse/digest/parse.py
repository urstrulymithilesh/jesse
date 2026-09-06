"""Is he asking what he has read lately?

Deterministic, like every other trigger that decides whether something structural
happens. The bar is deliberately high: this question has exactly one honest answer
drawn from a table, so a miss costs his rephrasing once, while a false fire has his
volunteering headlines in the middle of a conversation about something else.

Pure function — no store, no clock, no model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Belongs to the scheduler, not here: "anything new on my reminders" is a pending-query.
_NOT_OURS = ("timer", "alarm", "remind", "reminder")

# Each needs a "new / learn / read / news" signal AND to be aimed at his reading.
_PATTERNS = (
    r"what(?:'?s| is| has)?\s+new",
    r"anything\s+new",
    r"any(?:thing)?\s+news\b",
    r"what(?:'?s| is)?\s+(?:the\s+)?news\b",
    r"(?:what|anything)\s+(?:did|have)\s+you\s+learn(?:ed|t)?",
    r"(?:did|have)\s+you\s+learn(?:ed|t)?\s+anything",
    r"learn(?:ed|t)?\s+anything\s+(?:new|today|lately|interesting)",
    r"(?:what|anything)\s+(?:did|have)\s+you\s+read",
    r"(?:did|have)\s+you\s+read\s+anything",
    r"been\s+reading",
    r"anything\s+interesting\s+(?:today|lately|going\s+on)",
    r"catch\s+me\s+up",
    r"what(?:'?s| is)?\s+going\s+on\s+in\s+the\s+world",
)
_COMPILED = tuple(re.compile(p, re.I) for p in _PATTERNS)

# "anything new AT WORK", "what's new with my sister" — his life, not his sources.
# Jesse could only answer these from what Mithilesh tells him, so they are talk.
# Measured: without these two guards the trigger fired on 2 of 12 held-out his-life
# utterances ("anything new at work?", "what's new in my calendar"). Both are things
# he could only ever answer from what he told his, so they belong to ordinary talk.
_HIS_POSSESSIVE = re.compile(r"\b(?:my|our)\b", re.I)
_HIS_NOUNS = re.compile(
    r"\b(?:work|job|jobs|calendar|inbox|email|emails|phone|office|team|class|classes|"
    r"course|shift|shifts|rota|meeting|meetings|flat|house|car)\b", re.I)


@dataclass(frozen=True)
class DigestQuery:
    source: str | None = None      # "anything new from the BBC" -> just that one


def asks_whats_new(text: str, sources=()) -> DigestQuery | None:
    """Return a query when he is asking what he has read, else None."""
    if not text or not text.strip():
        return None
    low = text.lower()
    if any(w in low for w in _NOT_OURS):
        return None
    if not any(p.search(low) for p in _COMPILED):
        return None
    # Naming a source settles it — "anything new from the BBC" is ours even though it
    # would otherwise trip nothing, and it must win over the his-life guards below.
    for name in sources:
        if re.search(rf"\b{re.escape(name.lower())}\b", low):
            return DigestQuery(source=name)
    if _HIS_POSSESSIVE.search(low) or _HIS_NOUNS.search(low):
        return None
    return DigestQuery()
