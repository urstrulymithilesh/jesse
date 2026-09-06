"""Cut a stream of LLM tokens into speakable sentences.

Pure and incremental: feed it the buffer so far, get back whatever sentences are
definitely complete plus the leftover partial text. That's what lets Jesse start
speaking sentence one while the model is still writing sentence two.

A sentence only counts as complete when its terminator is FOLLOWED by whitespace —
mid-stream, "3." might still become "3.5", so waiting for the space avoids splitting
inside a number. At end of generation the caller flushes whatever remains.
"""

from __future__ import annotations

# Words that end in a period without ending the sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie",
    "approx", "dept", "est", "fig", "no", "vol",
}
_TERMINATORS = ".!?"


def _ends_with_abbreviation(chunk: str) -> bool:
    """True if the candidate sentence ends on something like "Dr." or an initial."""
    word = chunk.rstrip(_TERMINATORS).rsplit(None, 1)
    if not word:
        return False
    last = word[-1].strip().lower().replace(".", "")
    return last in _ABBREVIATIONS or len(last) == 1


def split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """-> (complete sentences, remaining partial text)."""
    sentences: list[str] = []
    start = 0
    i = 0
    while i < len(buffer):
        if buffer[i] not in _TERMINATORS:
            i += 1
            continue
        end = i
        while end + 1 < len(buffer) and buffer[end + 1] in _TERMINATORS:
            end += 1                                  # keep "?!" together
        if end + 1 < len(buffer) and buffer[end + 1].isspace():
            candidate = buffer[start:end + 1].strip()
            if candidate and not _ends_with_abbreviation(candidate):
                sentences.append(candidate)
                start = end + 1
                while start < len(buffer) and buffer[start].isspace():
                    start += 1
                i = start
                continue
        i = end + 1
    return sentences, buffer[start:]
