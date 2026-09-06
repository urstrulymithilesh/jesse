"""Episodic memory — what was actually talked about, and when.

Separate from the fact store on purpose. Facts are SLOTS: one row per subject,
last-write-wins, deduped by meaning. Episodes are EVENTS: append-only, time-ordered,
never merged. Two conversations about the gym are two things that happened, not a
duplicate slot — and the subject dedupe that keeps the fact store tidy would happily
destroy that history. Same database file, different table, different lifecycle.

A turn carries `episode_id` once it has been folded into a summary, so summarising can
resume after a crash exactly the way fact extraction does.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import sqlite_vec

from jesse.core.interfaces import LLM, Message
from jesse.memory.temporal import TimeWindow

# Tune freely — isolated like the persona and the extraction prompt.
SUMMARY_PROMPT = """\
You summarise a conversation between Jesse and Mithilesh into ONE short paragraph.
Write 1-3 sentences, past tense, third person, naming what was actually discussed and
anything notable that happened between them (a joke, a decision, a mood, something he
asked his). Be concrete and specific. Do NOT invent anything that is not in the
transcript. Output only the summary, no preamble, no bullet points.
"""


@dataclass(frozen=True)
class Episode:
    id: int
    summary: str
    started_at: datetime
    ended_at: datetime
    turn_count: int

    def when(self, *, now: datetime | None = None) -> str:
        """A human way to say when this happened."""
        now = now or datetime.now()
        delta = now - self.ended_at
        if self.ended_at.date() == now.date():
            return "earlier today"
        if delta.days <= 1:
            return "yesterday"
        if delta.days < 7:
            return f"{delta.days} days ago"
        if delta.days < 14:
            return "last week"
        return self.ended_at.strftime("on %d %B")


class EpisodeStore:
    def __init__(self, db_path: str | Path, embedder) -> None:
        self._embedder = embedder
        self._conn = sqlite3.connect(str(db_path))
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes(
                id INTEGER PRIMARY KEY,
                summary TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                turn_count INTEGER NOT NULL DEFAULT 0);
            -- Embeddings of the summaries, for "did we ever talk about X".
            CREATE TABLE IF NOT EXISTS episode_vectors(
                episode_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL);
            """
        )
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(turns)").fetchall()}
        if cols and "episode_id" not in cols:
            self._conn.execute("ALTER TABLE turns ADD COLUMN episode_id INTEGER")
        self._conn.commit()

    # -- writing -----------------------------------------------------------

    def unsummarised_turns(self, *, limit: int = 40) -> list[tuple[int, str, str, str]]:
        """(id, role, content, ts) for turns not yet folded into an episode."""
        return self._conn.execute(
            "SELECT id, role, content, ts FROM turns WHERE episode_id IS NULL "
            "ORDER BY id LIMIT ?", (limit,)
        ).fetchall()

    def add(self, summary: str, started_at: datetime, ended_at: datetime,
            turn_ids: list[int]) -> int:
        cur = self._conn.execute(
            "INSERT INTO episodes(summary, started_at, ended_at, turn_count) "
            "VALUES(?, ?, ?, ?)",
            (summary, started_at.isoformat(timespec="seconds"),
             ended_at.isoformat(timespec="seconds"), len(turn_ids)),
        )
        episode_id = int(cur.lastrowid)
        vector = sqlite_vec.serialize_float32(self._embedder.embed([summary])[0])
        self._conn.execute(
            "INSERT INTO episode_vectors(episode_id, embedding) VALUES(?, ?)",
            (episode_id, vector))
        if turn_ids:
            placeholders = ",".join("?" * len(turn_ids))
            self._conn.execute(
                f"UPDATE turns SET episode_id = ? WHERE id IN ({placeholders})",
                [episode_id, *turn_ids])
        self._conn.commit()
        return episode_id

    # -- reading -----------------------------------------------------------

    def _rows_to_episodes(self, rows) -> list[Episode]:
        return [Episode(id=r[0], summary=r[1],
                        started_at=datetime.fromisoformat(r[2]),
                        ended_at=datetime.fromisoformat(r[3]), turn_count=r[4])
                for r in rows]

    def all_episodes(self) -> list[Episode]:
        return self._rows_to_episodes(self._conn.execute(
            "SELECT id, summary, started_at, ended_at, turn_count FROM episodes "
            "ORDER BY started_at").fetchall())

    def in_window(self, window: TimeWindow) -> list[Episode]:
        """Episodes that overlap the window, oldest first. An unbounded window means
        all of history, which is what "did we ever talk about ..." wants."""
        return [e for e in self.all_episodes() if window.contains(e.ended_at)]

    def recent(self, *, limit: int = 3) -> list[Episode]:
        return self._rows_to_episodes(self._conn.execute(
            "SELECT id, summary, started_at, ended_at, turn_count FROM episodes "
            "ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall())[::-1]

    def search(self, query: str, *, k: int = 3) -> list[Episode]:
        """Semantic search over summaries — "did we ever talk about the gym"."""
        if not query.strip() or not self.all_episodes():
            return []
        blob = sqlite_vec.serialize_float32(self._embedder.embed([query])[0])
        ids = [r[0] for r in self._conn.execute(
            "SELECT episode_id, vec_distance_cosine(embedding, ?) AS d "
            "FROM episode_vectors ORDER BY d LIMIT ?", (blob, k)).fetchall()]
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT id, summary, started_at, ended_at, turn_count FROM episodes "
            f"WHERE id IN ({placeholders})", ids).fetchall()
        by_id = {e.id: e for e in self._rows_to_episodes(rows)}
        return [by_id[i] for i in ids if i in by_id]

    def close(self) -> None:
        self._conn.close()


class Summariser:
    """Turns a run of raw turns into one episode. Kept separate from the store so the
    LLM call and the persistence are testable apart."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def summarise(self, turns: list[tuple[int, str, str, str]]) -> str:
        transcript = "\n".join(
            f"{'Mithilesh' if role == 'user' else 'Jesse'}: {content}"
            for _id, role, content, _ts in turns)
        messages = [Message("system", SUMMARY_PROMPT), Message("user", transcript)]
        return "".join(self._llm.chat(messages, stream=False)).strip()
