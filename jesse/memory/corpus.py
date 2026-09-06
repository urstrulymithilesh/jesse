"""Learned knowledge — documents Mithilesh hands him, retrievable in conversation.

Roadmap step 8. Third lifecycle on the same database file, alongside facts (slots) and
episodes (events): a corpus is a BODY OF TEXT he was given. Named, append-only,
deleted whole. "Learning a skill" is ingesting the sources; "keeping it permanently" is
the corpus persisting; "forgetting it" is dropping the collection.

The honest ceiling, stated up front: this makes his able to answer *from his documents*.
It does not make him an expert. Fine-tuning to expert level was ruled infeasible on 4GB
VRAM in DESIGN.md and reframed as retrieval, and retrieval over a 3B with a 4096-token
context is a lookup, not mastery. He quotes the source or says he has nothing.

No new dependencies: the same sqlite-vec + fastembed the fact store already uses.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

READABLE = (".txt", ".md", ".markdown", ".rst")


@dataclass(frozen=True)
class Passage:
    corpus: str
    source: str          # filename it came from, so he can say where he read it
    text: str
    distance: float      # cosine distance; lower is closer


def subjects_mentioned(text: str, names) -> list[str]:
    """Which corpus names he actually said. Word-boundary match, case-insensitive.

    This is the TRIGGER for retrieval, and it replaced a pure distance threshold that
    did not survive contact with a second document. Measured: with one corpus, real
    questions scored 0.182-0.446 and ordinary talk 0.478-0.586 — a clean 0.032 gap. Add
    a second, unrelated corpus and the gap INVERTS at six passages ("I think I'll cook
    something tonight" hit the sourdough corpus at 0.432, closer than a genuine guitar
    question at 0.446). More passages means a better nearest match for everything,
    small talk included, so no fixed threshold survives growth, and a contrast/ratio
    test did not separate them either.

    So the trigger is his own words, which do not drift as the corpus grows. The cost is
    stated plainly: he has to name the subject to raise it, after which follow-ups work
    because recent turns are searched too.
    """
    low = text.lower()
    return [n for n in names
            if re.search(rf"\b{re.escape(n.lower())}\b", low)]


# Common English words that can never be a trigger, however often a document uses
# them. Without this list, sourdough.md would claim "water" and any mention of a glass
# of water would raise baking. Function words plus the everyday nouns and verbs that
# appear in ordinary talk about nothing in particular.
_COMMON = frozenset("""
about above across after again against all almost alone along already also although
always among another answer any anyone anything around as ask asked at away back bad
be became because become been before began begin behind being below best better
between big both bring but by call called came can cannot care case cause certain
change close cold come comes common could course cut day days did different do does
doing done down during each early easy either else end enough even ever every
everything face fact fall famous far fast feel feet fell felt few final find fine
first five for found four free from front full gave get gets give given go goes going
gone good got great group grow had half hand happen hard has have having he head hear
heard held help his here high his his hold home hot hour hours house how however idea
if important in inside instead into is it its itself just keep keeps kept kind knew
know known large last late later learn least leave left less let letter life light
like likely line list little live long look looked looking lot low made main make
making man many matter may me mean means men might mind mine minute minutes moment
month months more morning most move much must my myself name near nearly need never
new next night no none not note nothing now number of off often old on once one only
open or order other others our out outside over own part past people perhaps person
place plan play point possible probably problem put question quite rather reach read
ready real really reason rest right room round said same saw say saying says second
seconds see seem seemed seen set seven several he short should show shown side simple
since six small so some someone something sometimes soon sort sound start state stay
still stop story such sure take taken talk tell ten than that the their them
themselves then there these they thing things think third this those though thought
three through time times to today together told too took top toward tried try turn
turned two under understand until up upon us use used using usually very want wanted
warm was watch water way ways we week weeks well went were what whatever when where
whether which while white who whole whose why will wish with within without word words
work world would write wrong year years yes yet you young your yourself
""".split())

_TRIGGER_MIN_LEN = 4       # "peg", "nut", "jam" are too short to be safe triggers
_TRIGGER_MIN_FREQ = 2      # a word used once is incidental, not what the document is about


def _stem_s(word: str) -> str:
    """Naive plural fold so "strings" said aloud matches the trigger "string"."""
    return word[:-1] if len(word) > _TRIGGER_MIN_LEN and word.endswith("s") else word


def corpus_keywords(chunk_texts, *, max_words: int = 20) -> list[str]:
    """The distinctive recurring words of a document — its deterministic triggers.

    This exists because requiring the corpus NAME cost too much cold-start recall
    (measured 3/12: "how often should I change the strings" retrieves nothing unless
    the word "guitar" is said). The document itself knows its own vocabulary, so a
    word qualifies by using only the document: long enough to be distinctive, not
    common English, and used more than once. No embeddings anywhere — this cannot
    drift as the corpus grows, which is what killed the distance trigger.
    """
    from collections import Counter

    freq: Counter[str] = Counter()
    for chunk in chunk_texts:
        freq.update(_stem_s(w) for w in re.findall(r"[a-z]+", chunk.lower()))
    words = [w for w, n in freq.items()
             if len(w) >= _TRIGGER_MIN_LEN and w not in _COMMON
             and n >= _TRIGGER_MIN_FREQ]
    words.sort(key=lambda w: (-freq[w], w))
    return words[:max_words]


def chunk_text(text: str, *, chunk_chars: int = 800) -> list[str]:
    """Split on blank lines, then pack paragraphs up to the budget.

    Paragraph boundaries rather than a fixed window: a chunk cut mid-sentence embeds
    badly and reads worse when he says it out loud. A single paragraph longer than the
    budget is left whole — splitting it would do the damage the packing avoids.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


class CorpusStore:
    def __init__(self, db_path: str | Path, embedder) -> None:
        self._embedder = embedder
        self._conn = sqlite3.connect(str(db_path))
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS corpus_chunks(
                id INTEGER PRIMARY KEY,
                corpus TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS corpus_vectors(
                chunk_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS corpus_by_name ON corpus_chunks(corpus);
            -- Deterministic trigger words derived from each corpus's own text.
            CREATE TABLE IF NOT EXISTS corpus_keywords(
                corpus TEXT NOT NULL, word TEXT NOT NULL,
                PRIMARY KEY(corpus, word));
            """
        )
        self._conn.commit()
        # A db from before trigger words existed has corpora but no keywords; derive
        # them on open, same pattern as the fact store's vector backfill.
        for name in self.names():
            if not self._conn.execute(
                    "SELECT 1 FROM corpus_keywords WHERE corpus=? LIMIT 1",
                    (name,)).fetchone():
                self._rebuild_keywords(name)

    # -- writing -----------------------------------------------------------

    def ingest(self, corpus: str, path: str | Path, *, chunk_chars: int = 800) -> int:
        """Read a file (or every readable file in a folder) into `corpus`.

        Returns the number of chunks stored. Re-ingesting the same source replaces it,
        so fixing a typo in a document and running again does not double every answer.
        """
        path = Path(path)
        files = ([p for p in sorted(path.rglob("*")) if p.suffix.lower() in READABLE]
                 if path.is_dir() else [path])
        if not files:
            return 0
        stored = 0
        for file in files:
            text = file.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_text(text, chunk_chars=chunk_chars)
            if not chunks:
                continue
            self._drop_source(corpus, file.name)
            vectors = self._embedder.embed(chunks)
            for chunk, vector in zip(chunks, vectors):
                cur = self._conn.execute(
                    "INSERT INTO corpus_chunks(corpus, source, text) VALUES(?, ?, ?)",
                    (corpus, file.name, chunk))
                self._conn.execute(
                    "INSERT INTO corpus_vectors(chunk_id, embedding) VALUES(?, ?)",
                    (int(cur.lastrowid), sqlite_vec.serialize_float32(vector)))
                stored += 1
        self._conn.commit()
        if stored:
            self._rebuild_keywords(corpus)
        return stored

    def _rebuild_keywords(self, corpus: str) -> None:
        texts = [r[0] for r in self._conn.execute(
            "SELECT text FROM corpus_chunks WHERE corpus=?", (corpus,))]
        self._conn.execute("DELETE FROM corpus_keywords WHERE corpus=?", (corpus,))
        for word in corpus_keywords(texts):
            self._conn.execute(
                "INSERT OR IGNORE INTO corpus_keywords(corpus, word) VALUES(?, ?)",
                (corpus, word))
        self._conn.commit()

    def keyword_subjects(self, text: str) -> list[str]:
        """Corpora whose OWN vocabulary appears in what he said, without the name.

        Weaker evidence than the name, and treated as weaker: the orchestrator only
        ever ASKS on this signal, it never answers from the corpus. Measured why: real
        cold questions and collision phrases ("my starter motor died" against a
        sourdough corpus) overlap in embedding distance by 0.005, so any rule that
        injected on keywords would inject on collisions too. A question is the failure
        mode that costs one word to recover from.
        """
        tokens = {_stem_s(w) for w in re.findall(r"[a-z]+", text.lower())}
        rows = self._conn.execute("SELECT corpus, word FROM corpus_keywords").fetchall()
        return sorted({c for c, w in rows if w in tokens})

    def _drop_source(self, corpus: str, source: str) -> None:
        ids = [r[0] for r in self._conn.execute(
            "SELECT id FROM corpus_chunks WHERE corpus=? AND source=?", (corpus, source))]
        self._delete(ids)

    def _delete(self, ids: list[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(f"DELETE FROM corpus_vectors WHERE chunk_id IN ({placeholders})", ids)
        self._conn.execute(f"DELETE FROM corpus_chunks WHERE id IN ({placeholders})", ids)

    def forget(self, corpus: str) -> int:
        ids = [r[0] for r in self._conn.execute(
            "SELECT id FROM corpus_chunks WHERE corpus=?", (corpus,))]
        self._delete(ids)
        self._conn.execute("DELETE FROM corpus_keywords WHERE corpus=?", (corpus,))
        self._conn.commit()
        return len(ids)

    # -- reading -----------------------------------------------------------

    def corpora(self) -> list[tuple[str, int, int]]:
        """(name, chunks, sources) for everything he has learned."""
        return [(r[0], r[1], r[2]) for r in self._conn.execute(
            "SELECT corpus, COUNT(*), COUNT(DISTINCT source) FROM corpus_chunks "
            "GROUP BY corpus ORDER BY corpus")]

    def names(self) -> list[str]:
        return [r[0] for r in self._conn.execute(
            "SELECT DISTINCT corpus FROM corpus_chunks ORDER BY corpus")]

    def search(self, query: str, *, k: int = 2, max_distance: float = 1.0,
               corpora=None) -> list[Passage]:
        """Closest passages, nearest first, dropping anything past `max_distance`.

        `corpora` restricts the search to named subjects — see `subjects_mentioned`,
        which is the actual trigger. The distance gate survives as a SECOND filter
        inside a subject he raised himself: it drops a passage that is about the right
        subject but nowhere near his question. It is no longer load-bearing on its own,
        which is exactly what the measurements said it could not be.
        """
        if not query.strip():
            return []
        where, params = "", []
        if corpora is not None:
            if not corpora:
                return []
            where = f"WHERE c.corpus IN ({','.join('?' * len(corpora))})"
            params = list(corpora)
        if not self._conn.execute("SELECT 1 FROM corpus_chunks LIMIT 1").fetchone():
            return []
        blob = sqlite_vec.serialize_float32(self._embedder.embed([query])[0])
        rows = self._conn.execute(
            "SELECT c.corpus, c.source, c.text, vec_distance_cosine(v.embedding, ?) AS d "
            f"FROM corpus_vectors v JOIN corpus_chunks c ON c.id = v.chunk_id {where} "
            "ORDER BY d LIMIT ?", (blob, *params, k)).fetchall()
        return [Passage(corpus=r[0], source=r[1], text=r[2], distance=r[3])
                for r in rows if r[3] <= max_distance]

    def close(self) -> None:
        self._conn.close()
