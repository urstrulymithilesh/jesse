"""What he has read from his sources, and what he has not heard about yet.

A fourth lifecycle on the same database, and a separate one on purpose — the same
call that split episodes out of facts. Facts are slots, episodes are events, a corpus
is a body of text Mithilesh handed him. Digest items are **dated arrivals**:
deduped by url so a feed that lists the same article for a week is one item, ordered by
when they arrived, and each one carries whether it has been mentioned to him yet.

Deliberately NOT in the corpus store, even though ingesting is superficially similar.
Two reasons, one measured (see `corpus_keywords` and §6 of HANDOFF): a corpus
contributes trigger keywords, and a news source's vocabulary is *everything* — money,
family, school, weather — so folding digests into that pool would degrade the
false-fire numbers the keyword trigger was tuned to. And "what's new" is a question
about time, not meaning, which is exactly what made episodes deterministic rather than
semantic. No embeddings are computed here at all.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jesse.digest.feeds import Item, looks_like_instruction


@dataclass(frozen=True)
class StoredItem:
    id: int
    source: str
    url: str
    title: str
    summary: str
    published: str
    fetched_at: datetime
    told: bool


class DigestStore:
    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS digest_items(
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                published TEXT NOT NULL DEFAULT '',
                fetched_at TEXT NOT NULL,
                told INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS digest_meta(key TEXT PRIMARY KEY, value TEXT);
            """
        )
        self._conn.commit()

    # -- writing -----------------------------------------------------------

    def add(self, items, *, now: datetime | None = None) -> int:
        """Store items not seen before. Returns how many were actually new.

        The url is the identity. A feed republishing yesterday's article is not news,
        and counting it as new is how "what's new" would end up saying the same thing
        every morning until he stopped asking.
        """
        stamp = (now or datetime.now()).isoformat(timespec="seconds")
        added = 0
        for item in items:
            if not item.url:
                continue           # no identity, so it cannot be deduped — skip it
            if looks_like_instruction(item.title, item.summary):
                # Never stored, so it can never reach his context. See the note on
                # _INSTRUCTION_SHAPED: the risk it removes is not that he obeys it
                # (he does not), but that he cannot repeat it and invents instead.
                print(f"  [sources] dropped an instruction-shaped item from "
                      f"{item.source}: {item.title[:60]!r}")
                continue
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO digest_items"
                "(source, url, title, summary, published, fetched_at) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (item.source, item.url, item.title, item.summary, item.published, stamp))
            added += cur.rowcount
        self._conn.commit()
        return added

    def mark_told(self, ids) -> None:
        ids = list(ids)
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(
            f"UPDATE digest_items SET told = 1 WHERE id IN ({placeholders})", ids)
        self._conn.commit()

    def forget_source(self, source: str) -> int:
        cur = self._conn.execute("DELETE FROM digest_items WHERE source=?", (source,))
        self._conn.commit()
        return cur.rowcount

    # -- reading -----------------------------------------------------------

    def _rows(self, sql: str, params=()) -> list[StoredItem]:
        return [StoredItem(id=r[0], source=r[1], url=r[2], title=r[3], summary=r[4],
                           published=r[5], fetched_at=datetime.fromisoformat(r[6]),
                           told=bool(r[7]))
                for r in self._conn.execute(sql, params)]

    def untold(self, *, limit: int = 5) -> list[StoredItem]:
        """What he has not been told about yet, newest arrivals first."""
        return self._rows(
            "SELECT id, source, url, title, summary, published, fetched_at, told "
            "FROM digest_items WHERE told = 0 ORDER BY id DESC LIMIT ?", (limit,))

    def recent(self, *, limit: int = 5, source: str | None = None) -> list[StoredItem]:
        if source is None:
            return self._rows(
                "SELECT id, source, url, title, summary, published, fetched_at, told "
                "FROM digest_items ORDER BY id DESC LIMIT ?", (limit,))
        return self._rows(
            "SELECT id, source, url, title, summary, published, fetched_at, told "
            "FROM digest_items WHERE source=? ORDER BY id DESC LIMIT ?", (source, limit))

    def sources(self) -> list[tuple[str, int, int]]:
        """(source, items, untold) for everything he has ever read."""
        return [(r[0], r[1], r[2]) for r in self._conn.execute(
            "SELECT source, COUNT(*), SUM(CASE WHEN told = 0 THEN 1 ELSE 0 END) "
            "FROM digest_items GROUP BY source ORDER BY source")]

    def untold_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM digest_items WHERE told = 0").fetchone()
        return int(row[0]) if row else 0

    # -- the fetch clock ---------------------------------------------------

    def last_fetch(self) -> datetime | None:
        """Absolute wall-clock, persisted — the same reconcile-on-resume rule the
        scheduler uses. A laptop that slept through the interval fetches once on wake,
        not once per missed interval."""
        row = self._conn.execute(
            "SELECT value FROM digest_meta WHERE key='last_fetch'").fetchone()
        return datetime.fromisoformat(row[0]) if row else None

    def set_last_fetch(self, when: datetime) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO digest_meta(key, value) VALUES('last_fetch', ?)",
            (when.isoformat(timespec="seconds"),))
        self._conn.commit()

    def due(self, *, interval_hours: float, now: datetime | None = None) -> bool:
        last = self.last_fetch()
        if last is None:
            return True
        elapsed = ((now or datetime.now()) - last).total_seconds()
        return elapsed >= interval_hours * 3600

    def close(self) -> None:
        self._conn.close()
