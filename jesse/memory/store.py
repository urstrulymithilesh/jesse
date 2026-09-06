"""SqliteMemoryStore — the v1 MemoryStore impl.

One SQLite file holds everything (portable, no server, very explainable):
  * turns  — raw conversation, for recent-history context
  * facts  — durable things learned about the user (subject, text, confidence)
  * vec_facts — sqlite-vec KNN index over the fact embeddings, for semantic recall

Trust/debug: every write is appended to a human-readable memory-log so you can SEE
exactly what Jesse stored, updated, or skipped, and why.

Conflict policy (eng review): a fact WITH a subject is upserted last-write-wins on
that subject; a fact WITHOUT a subject is deduped by exact text. Recall is a
semantic top-K — the caller passes its read budget as k.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import sqlite_vec

from jesse.config import CONFIG
from jesse.core.interfaces import PROTECTED_ORIGINS, Embedder, Fact, Message


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _may_merge(origin_a: str, origin_b: str) -> bool:
    """Two SEEDED facts are never merged into each other.

    Similarity alone cannot decide this. Measured on bge-small, "job" vs "job title"
    (should merge) is 0.843 while "isha's creator" vs "isha's name" (must NOT) is
    0.885 — they overlap, so no threshold separates them. But seeded facts are
    hand-authored as deliberately separate entries, which is authoritative in a way a
    cosine isn't. Without this rule, `jesse seed` silently dropped "isha's name" by
    merging it into "isha's creator". With it, the highest colliding pair that still
    depends on the threshold is sister's/brother's name at 0.822 — safely under 0.88.
    """
    return not (origin_a in PROTECTED_ORIGINS and origin_b in PROTECTED_ORIGINS)


class SqliteMemoryStore:
    def __init__(self, db_path: str | Path, embedder: Embedder, *,
                 log_path: str | Path | None = None) -> None:
        self._embedder = embedder
        self._log_path = Path(log_path) if log_path else None
        self._conn = sqlite3.connect(str(db_path))
        try:
            self._conn.enable_load_extension(True)
        except AttributeError as e:  # pragma: no cover - platform guard
            raise RuntimeError(
                "This Python's sqlite3 can't load extensions, so sqlite-vec won't load. "
                "Install pysqlite3-binary or use a Python built with loadable extensions."
            ) from e
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._dim: int | None = None
        self._init_schema()

    # -- schema ------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS turns(
                id INTEGER PRIMARY KEY, role TEXT, content TEXT, ts TEXT,
                processed INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS facts(
                id INTEGER PRIMARY KEY, subject TEXT, text TEXT NOT NULL,
                confidence REAL, source_turn_id INTEGER, ts TEXT,
                origin TEXT NOT NULL DEFAULT 'conversation');
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            -- Subject embeddings, for catching near-duplicate slots. A plain table,
            -- not a vec0 index: there are tens of facts, so a scan with
            -- vec_distance_cosine is simpler and exact.
            CREATE TABLE IF NOT EXISTS subject_vectors(
                fact_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL);
            """
        )
        # Migrate a db created before the `processed` column existed.
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(turns)").fetchall()}
        if "processed" not in cols:
            self._conn.execute(
                "ALTER TABLE turns ADD COLUMN processed INTEGER NOT NULL DEFAULT 0")
            # Turns that predate this feature are treated as already handled: we can't
            # know whether they were extracted, and re-chewing an old backlog would burn
            # ~7s per exchange at every startup for facts that are almost certainly
            # already stored. Only turns recorded from now on can go unprocessed.
            self._conn.execute("UPDATE turns SET processed = 1")
        self._conn.commit()
        self._backfill_subject_vectors()
        # On reopen, recover the embedding dim so the vec table is recreated to match.
        row = self._conn.execute("SELECT value FROM meta WHERE key='dim'").fetchone()
        if row is not None:
            self._dim = int(row[0])
            self._ensure_vec_table()

    def _ensure_vec_table(self) -> None:
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts "
            f"USING vec0(fact_id INTEGER PRIMARY KEY, embedding float[{self._dim}])"
        )
        self._conn.commit()

    def _set_dim(self, dim: int) -> None:
        if self._dim is None:
            self._dim = dim
            self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('dim', ?)", (str(dim),))
            self._ensure_vec_table()
            self._conn.commit()

    def _backfill_subject_vectors(self) -> int:
        """Give subject vectors to facts stored before this feature existed.

        Without this, every pre-existing fact is invisible to subject dedupe — both the
        retroactive scan AND the forward check, which would silently let a new fact
        duplicate an old one. Cheap guard: one COUNT, and after the first run there is
        nothing left to do.
        """
        missing = self._conn.execute(
            "SELECT id, subject FROM facts f WHERE subject IS NOT NULL AND subject != '' "
            "AND NOT EXISTS (SELECT 1 FROM subject_vectors sv WHERE sv.fact_id = f.id)"
        ).fetchall()
        if not missing:
            return 0
        vectors = self._embedder.embed([r[1] for r in missing])
        self._conn.executemany(
            "INSERT OR REPLACE INTO subject_vectors(fact_id, embedding) VALUES(?, ?)",
            [(r[0], sqlite_vec.serialize_float32(v)) for r, v in zip(missing, vectors)],
        )
        self._conn.commit()
        return len(missing)

    def duplicate_groups(self) -> list[tuple[Fact, list[tuple[Fact, float]]]]:
        """Scan what's already stored for near-duplicate subjects. Read-only.

        -> [(keeper, [(duplicate, similarity), ...]), ...]

        The keeper is chosen so a merge can never destroy something important: a seeded
        core/self fact always wins, then the most recent, then the most confident. A
        candidate is only compared against the KEEPER, never against another duplicate,
        so a chain of merely-similar subjects can't transitively collapse into one.
        """
        self._backfill_subject_vectors()
        protected = ",".join(f"'{o}'" for o in sorted(PROTECTED_ORIGINS))
        rows = self._conn.execute(
            f"SELECT f.id, f.subject, f.text, f.confidence, f.source_turn_id, f.origin, "
            f"       sv.embedding "
            f"FROM facts f JOIN subject_vectors sv ON sv.fact_id = f.id "
            f"ORDER BY CASE WHEN f.origin IN ({protected}) THEN 0 ELSE 1 END, "
            f"         f.ts DESC, f.confidence DESC, f.id DESC"
        ).fetchall()

        def as_fact(r):
            return Fact(text=r[2], confidence=r[3], source_turn_id=r[4], subject=r[1], origin=r[5])

        groups: list[tuple[Fact, list[tuple[Fact, float]]]] = []
        self._group_ids: list[tuple[int, list[int]]] = []      # parallel ids, for apply
        claimed: set[int] = set()
        for i, keeper in enumerate(rows):
            if keeper[0] in claimed:
                continue
            dups, dup_ids = [], []
            for other in rows[i + 1:]:
                if other[0] in claimed:
                    continue
                (distance,) = self._conn.execute(
                    "SELECT vec_distance_cosine(?, ?)", (keeper[6], other[6])
                ).fetchone()
                similarity = 1.0 - float(distance)
                if not _may_merge(keeper[5], other[5]):
                    continue                       # both seeded: distinct by construction
                if similarity >= CONFIG.memory.dedupe_subject_similarity:
                    dups.append((as_fact(other), similarity))
                    dup_ids.append(other[0])
                    claimed.add(other[0])
            if dups:
                claimed.add(keeper[0])
                groups.append((as_fact(keeper), dups))
                self._group_ids.append((keeper[0], dup_ids))
        return groups

    def merge_duplicates(self) -> int:
        """Actually apply what duplicate_groups() found. Returns facts removed."""
        groups = self.duplicate_groups()
        removed = 0
        for (keeper, dups), (_keeper_id, dup_ids) in zip(groups, self._group_ids):
            for (dup, similarity), dup_id in zip(dups, dup_ids):
                self._conn.execute("DELETE FROM facts WHERE id = ?", (dup_id,))
                self._drop_vectors([dup_id])
                self._log(f"MERGED-RETRO({similarity:.2f}->{keeper.subject!r})", dup)
                removed += 1
        self._conn.commit()
        return removed

    # -- facts -------------------------------------------------------------

    def _near_duplicate_subject(self, subject_blob: bytes) -> tuple[int, str, float] | None:
        """The closest existing fact whose SUBJECT means the same thing.

        Matching on the subject, not the fact text, is deliberate: subjects are slot
        names ("birthday month"), so similarity means "same slot, spelled differently".
        Fact text would be far more dangerous — "the user likes coffee" and "the user
        hates coffee" score 0.844, and merging those would keep the wrong one.
        """
        row = self._conn.execute(
            "SELECT f.id, f.origin, vec_distance_cosine(sv.embedding, ?) AS d "
            "FROM subject_vectors sv JOIN facts f ON f.id = sv.fact_id "
            "ORDER BY d LIMIT 1",
            (subject_blob,),
        ).fetchone()
        if row is None:
            return None
        similarity = 1.0 - float(row[2])
        if similarity < CONFIG.memory.dedupe_subject_similarity:
            return None
        return int(row[0]), str(row[1]), similarity

    def _drop_vectors(self, fact_ids) -> None:
        """Delete embedding rows for these facts.

        vec_facts is only created once a dimension is known (i.e. after something has
        been embedded), so on a db holding facts but no embeddings it simply isn't
        there yet. A missing table means there is nothing to delete, not an error.
        """
        ids = list(fact_ids)
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        for table in ("vec_facts", "subject_vectors"):
            try:
                self._conn.execute(
                    f"DELETE FROM {table} WHERE fact_id IN ({placeholders})", ids)
            except sqlite3.OperationalError:
                pass

    def _write_vectors(self, fact_id: int, text_blob: bytes, subject_blob: bytes | None) -> None:
        cur = self._conn.cursor()
        self._drop_vectors([fact_id])
        cur.execute("INSERT INTO vec_facts(fact_id, embedding) VALUES(?, ?)", (fact_id, text_blob))
        if subject_blob is not None:
            cur.execute("INSERT INTO subject_vectors(fact_id, embedding) VALUES(?, ?)",
                        (fact_id, subject_blob))

    def _supersede(self, fact_id: int, fact: Fact, text_blob: bytes,
                   subject_blob: bytes | None) -> None:
        self._conn.execute(
            "UPDATE facts SET subject=?, text=?, confidence=?, source_turn_id=?, ts=?, origin=? "
            "WHERE id=?",
            (fact.subject, fact.text, fact.confidence, fact.source_turn_id, _now(),
             fact.origin, fact_id),
        )
        self._write_vectors(fact_id, text_blob, subject_blob)
        self._conn.commit()

    def add_fact(self, fact: Fact) -> None:
        emb = self._embedder.embed([fact.text])[0]
        self._set_dim(len(emb))
        blob = sqlite_vec.serialize_float32(emb)
        subject_blob = (sqlite_vec.serialize_float32(self._embedder.embed([fact.subject])[0])
                        if fact.subject else None)
        cur = self._conn.cursor()

        if fact.subject:
            existing = cur.execute(
                "SELECT id, origin FROM facts WHERE subject = ?", (fact.subject,)
            ).fetchone()
            if existing is not None:
                fid, existing_origin = existing
                # Protection: conversational extraction may NOT overwrite a seeded core/
                # self fact. An offhand remark can't rewrite Jesse's identity or our history.
                if existing_origin in PROTECTED_ORIGINS and fact.origin == "conversation":
                    self._log("PROTECTED", fact)
                    return
                self._supersede(fid, fact, blob, subject_blob)   # last-write-wins
                self._log("UPDATED", fact)
                return

            # No exact subject match — is there a subject that MEANS the same thing?
            near = self._near_duplicate_subject(subject_blob)
            if near is not None:
                fid, existing_origin, similarity = near
                if not _may_merge(existing_origin, fact.origin):
                    pass          # distinct by construction — fall through and INSERT
                elif existing_origin in PROTECTED_ORIGINS and fact.origin == "conversation":
                    self._log("PROTECTED", fact)
                    return
                else:
                    self._supersede(fid, fact, blob, subject_blob)
                    self._log(f"MERGED({similarity:.2f})", fact)
                    return
        else:
            dup = cur.execute(
                "SELECT id FROM facts WHERE subject IS NULL AND text = ?", (fact.text,)
            ).fetchone()
            if dup is not None:  # exact-text dedupe
                self._log("DEDUP", fact)
                return

        cur.execute(
            "INSERT INTO facts(subject, text, confidence, source_turn_id, ts, origin) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (fact.subject, fact.text, fact.confidence, fact.source_turn_id, _now(), fact.origin),
        )
        fid = cur.lastrowid
        self._write_vectors(fid, blob, subject_blob)
        self._conn.commit()
        self._log("STORED", fact)

    def find_facts(self, needle: str) -> list[Fact]:
        """Facts matching `needle`, WITHOUT deleting. Deletion is destructive, so the
        caller previews first and only removes when exactly one thing fits."""
        n = needle.strip().lower()
        if not n:
            return []
        rows = self._conn.execute(
            "SELECT subject, text, confidence, source_turn_id, origin FROM facts "
            "WHERE lower(COALESCE(subject,'')) LIKE ? OR lower(text) LIKE ?",
            (f"%{n}%", f"%{n}%"),
        ).fetchall()
        return [Fact(text=r[1], confidence=r[2], source_turn_id=r[3], subject=r[0],
                     origin=r[4]) for r in rows]

    def forget(self, needle: str) -> list[Fact]:
        """Delete facts whose subject or text matches `needle` (case-insensitive
        substring). Returns what was removed, so the caller can report it.

        This is the correction path: extraction sometimes stores something wrong or
        something that was never true, and being able to take it back matters more
        than being able to add. Seeded core/self facts can be forgotten too — they
        come straight back with `python -m jesse seed`.
        """
        n = needle.strip().lower()
        if not n:
            return []
        rows = self._conn.execute(
            "SELECT id, subject, text, confidence, source_turn_id, origin FROM facts "
            "WHERE lower(COALESCE(subject,'')) LIKE ? OR lower(text) LIKE ?",
            (f"%{n}%", f"%{n}%"),
        ).fetchall()
        if not rows:
            return []
        gone = [Fact(text=r[2], confidence=r[3], source_turn_id=r[4], subject=r[1], origin=r[5])
                for r in rows]
        ids = [r[0] for r in rows]
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(f"DELETE FROM facts WHERE id IN ({placeholders})", ids)
        self._drop_vectors(ids)
        self._conn.commit()
        for fact in gone:
            self._log("FORGOT", fact)
        return gone

    def recall(self, query: str, *, k: int = 3, include_history: bool = False) -> list[Fact]:
        """Semantic top-K. Past-version ('self_history') facts are hidden unless
        include_history=True, so they only surface when the user asks about his past."""
        if self._dim is None or not query.strip():
            return []
        qblob = sqlite_vec.serialize_float32(self._embedder.embed([query])[0])
        # Over-fetch so we can drop history facts and still return k real hits.
        ids = [
            row[0]
            for row in self._conn.execute(
                "SELECT fact_id FROM vec_facts WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (qblob, k * 4),
            ).fetchall()
        ]
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        by_id = {
            r[0]: Fact(text=r[2], confidence=r[3], source_turn_id=r[4], subject=r[1], origin=r[5])
            for r in self._conn.execute(
                f"SELECT id, subject, text, confidence, source_turn_id, origin "
                f"FROM facts WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        }
        history: list[Fact] = []
        other: list[Fact] = []
        for i in ids:  # nearest-first within each bucket
            f = by_id.get(i)
            if f is None:
                continue
            if f.origin == "self_history":
                if include_history:
                    history.append(f)      # only when the user asked about his past
            else:
                other.append(f)
        # When asked about the past, lead with history so the old-version facts actually
        # surface (a current 'self' fact would otherwise out-rank them by similarity).
        ordered = (history + other) if include_history else other
        return ordered[:k]

    # -- turns -------------------------------------------------------------

    def append_turn(self, message: Message) -> int:
        cur = self._conn.execute(
            "INSERT INTO turns(role, content, ts) VALUES(?, ?, ?)",
            (message.role, message.content, _now()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def unprocessed_exchanges(self, *, limit: int = 5) -> list[tuple[int, int, str, str]]:
        """Turn pairs whose fact-extraction never completed (cancelled, or app quit).

        Returns (user_turn_id, assistant_turn_id, user_text, assistant_text), oldest
        first. This is what makes a taught fact survive: turns are persisted BEFORE
        extraction runs, so anything the extractor missed can be picked up later.
        Turns are always written as a user+assistant pair; an unpaired straggler is
        skipped (it can never form an exchange, and skipping costs only a SELECT).
        """
        rows = self._conn.execute(
            "SELECT id, role, content FROM turns WHERE processed = 0 ORDER BY id"
        ).fetchall()
        out: list[tuple[int, int, str, str]] = []
        i = 0
        while i < len(rows) - 1 and len(out) < limit:
            a, b = rows[i], rows[i + 1]
            if a[1] == "user" and b[1] == "assistant":
                out.append((a[0], b[0], a[2], b[2]))
                i += 2
            else:
                i += 1
        return out

    def mark_processed(self, turn_ids) -> None:
        """Flag turns as extracted so the same exchange is never extracted twice."""
        ids = list(turn_ids)
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(
            f"UPDATE turns SET processed = 1 WHERE id IN ({placeholders})", ids
        )
        self._conn.commit()

    def all_facts(self) -> list[Fact]:
        """Every stored fact, oldest first — for inspection/debugging (jesse memory)."""
        rows = self._conn.execute(
            "SELECT subject, text, confidence, source_turn_id, origin FROM facts ORDER BY id"
        ).fetchall()
        return [Fact(text=r[1], confidence=r[2], source_turn_id=r[3], subject=r[0], origin=r[4])
                for r in rows]

    def recent(self, *, limit: int = 20) -> list[Message]:
        rows = self._conn.execute(
            "SELECT role, content FROM turns ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [Message(role=r[0], content=r[1]) for r in reversed(rows)]

    # -- misc --------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else row[0]

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))
        self._conn.commit()

    def _log(self, action: str, fact: Fact) -> None:
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        line = (f"[{_now()}] {action:7} subject={fact.subject!r} "
                f"conf={fact.confidence} text={fact.text!r}\n")
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def close(self) -> None:
        self._conn.close()
