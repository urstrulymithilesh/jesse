"""Small, deterministic routing and lexical anchors for personal recall."""

import re
from dataclasses import replace

_QUESTION = re.compile(
    r"^\s*(?:(?:what|when|where|who|how)\b.*\b(?:my|our)\b|"
    r"(?:do you remember|can you remember|remind me|tell me about)\b.*\b(?:my|our|i)\b)",
    re.IGNORECASE,
)
_STOP = frozenset(
    "what when where who how is are was were do does did can could would should "
    "my our your the a an i me you we to of for about that this it in on and "
    "remember remind tell know have has had again please user users".split()
)


def asks_personal_memory(text: str) -> bool:
    return bool(_QUESTION.search(text)) and not re.search(
        r"\b(?:should|could|would)\b|\b(?:current|local) (?:time|date)\b", text, re.I,
    )


def topic_words(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in words if len(w) > 2 and w not in _STOP}


def matching_facts(query, facts):
    words = topic_words(query)
    return [f for f in facts if f.origin in ("conversation", "core")
            and words & topic_words(f"{f.subject or ''} {f.text}")]


_CORRECTION = re.compile(
    r"^(?:that(?:'s| is)|it(?:'s| is)) not ([\w '-]{1,60}?),? "
    r"(?:it(?:'s| is)|i mean|i meant) ([\w '-]{1,60})[.!]?$", re.IGNORECASE,
)


def apply_explicit_corrections(facts, evidence):
    """Apply only named X -> Y corrections to a temporary copy of recalled facts.

    No semantic merging, no updates to the database, and no inferred antecedent for
    a bare 'I meant Y'. Protected identity facts are never rewritten this way.
    """
    result = list(facts)
    for statement in evidence:
        match = _CORRECTION.fullmatch(statement.content.strip())
        if not match:
            continue
        old, new = (part.strip() for part in match.groups())
        pattern = re.compile(r"(?<!\w)" + re.escape(old) + r"(?!\w)", re.I)
        result = [replace(f, text=pattern.sub(lambda _: new, f.text))
                  if f.origin == "conversation" else f for f in result]
    return result
