"""Scheduler — decides what's due, what's too late to matter, and fires it.

The DECISION is a pure function (`triage`) so every rule is unit-testable with a
fake clock and no db, no audio, no model. The async loop around it is a thin tick
that hands announcements to a callback — in practice `Orchestrator.notify`, which
owns *when* it's actually spoken (never mid-utterance).

Reconcile-on-wake: because fire times are absolute and stored, startup and every
tick use the same code path. A reminder that came due while the laptop slept or
the app was closed simply looks "overdue" on the next tick and fires with an
honest late note — unless it's so old that announcing it would be noise.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime

from jesse.schedule.parse import announcement
from jesse.schedule.store import (CANCELLED, DROPPED, FIRED, ScheduledItem,
                                 SqliteScheduleStore)


def triage(
    items: list[ScheduledItem],
    *,
    now: datetime,
    stale_after_s: float,
    overdue_note_after_s: float,
) -> tuple[list[tuple[ScheduledItem, str]], list[ScheduledItem]]:
    """Pure. -> (to_fire as (item, spoken_text), to_drop).

    Anything at or past its fire time fires. If it's more than
    `overdue_note_after_s` late, the announcement admits how late it is. If it's
    more than `stale_after_s` late, it's dropped instead — waking someone at
    midnight for a 5pm gym reminder is worse than staying quiet.
    """
    to_fire: list[tuple[ScheduledItem, str]] = []
    to_drop: list[ScheduledItem] = []
    for item in items:
        late = (now - item.fire_at).total_seconds()
        if late < 0:
            continue                                   # not due yet
        if late > stale_after_s:
            to_drop.append(item)
            continue
        note_late = late if late > overdue_note_after_s else 0.0
        to_fire.append((item, announcement(item.task, is_timer=item.is_timer,
                                           overdue_seconds=note_late)))
    return to_fire, to_drop


def resolve_target(items: list[ScheduledItem], hint: str = "") -> tuple[ScheduledItem | None, str]:
    """Pure. Which pending reminder did he mean? -> (item, reason).

    reason is "" on success, else why we can't act: "none" (nothing pending) or
    "ambiguous" (several, and the hint doesn't single one out). We ASK rather than
    guess — cancelling the wrong reminder fails silently, and he'd only find out
    when the one he wanted never fires.
    """
    if not items:
        return None, "none"
    # Keep digits (so "10" survives) and words long enough to be meaningful.
    words = [w for w in hint.lower().split() if w.isdigit() or len(w) > 2]
    if words:
        # Score against the task AND how it was described ("the 10 minute one"), because
        # a bare timer has no task text. Scoring beats any-match: "10 minutes" hits both
        # "10 minutes" and "5 minutes" on the word "minutes", but only one scores twice.
        scored = [(sum(1 for w in words if w in f"{i.task} {i.label}".lower()), i)
                  for i in items]
        best = max((n for n, _ in scored), default=0)
        if best > 0:
            winners = [i for n, i in scored if n == best]
            if len(winners) == 1:
                return winners[0], ""
            return None, "ambiguous"
    if len(items) == 1:
        return items[0], ""
    return None, "ambiguous"


class Scheduler:
    def __init__(
        self,
        store: SqliteScheduleStore,
        notify: Callable[[str], None],
        *,
        tick_seconds: float = 2.0,
        stale_after_s: float = 2 * 3600,
        overdue_note_after_s: float = 60.0,
    ) -> None:
        self._store = store
        self._notify = notify
        self._tick = tick_seconds
        self._stale_after_s = stale_after_s
        self._overdue_note_after_s = overdue_note_after_s

    def add(self, task: str, fire_at: datetime, *, is_timer: bool = False,
            label: str = "") -> int:
        return self._store.add(task, fire_at, is_timer=is_timer, label=label)

    def pending(self) -> list[ScheduledItem]:
        return self._store.pending()

    def cancel(self, hint: str = "", *, all_of_them: bool = False) -> tuple[int, str]:
        """-> (count cancelled, reason-if-none). Never creates anything."""
        items = self._store.pending()
        if all_of_them:
            for item in items:
                self._store.mark(item.id, CANCELLED)
            return len(items), "" if items else "none"
        target, reason = resolve_target(items, hint)
        if target is None:
            return 0, reason
        self._store.mark(target.id, CANCELLED)
        print(f"  [reminder] cancelled: {target.task or 'timer'}")
        return 1, ""

    def reschedule(self, fire_at: datetime, hint: str = "",
                   *, label: str | None = None) -> tuple[ScheduledItem | None, str]:
        """Move an EXISTING reminder. -> (item, reason-if-none)."""
        items = self._store.pending()
        target, reason = resolve_target(items, hint)
        if target is None:
            return None, reason
        self._store.update_fire_at(target.id, fire_at, label=label)
        print(f"  [reminder] rescheduled '{target.task or 'timer'}' -> {fire_at:%H:%M:%S}")
        return target, ""

    def check(self, *, now: datetime | None = None) -> int:
        """One reconcile pass: fire what's due, drop what's stale. Returns fired count.
        Used identically at startup and on every tick — that IS the reconcile."""
        now = now or datetime.now()
        to_fire, to_drop = triage(
            self._store.pending(), now=now,
            stale_after_s=self._stale_after_s,
            overdue_note_after_s=self._overdue_note_after_s,
        )
        for item in to_drop:
            self._store.mark(item.id, DROPPED)
            print(f"  [reminder] dropped as too old to be useful: {item.task or 'timer'}")
        for item, spoken in to_fire:
            self._store.mark(item.id, FIRED)   # mark BEFORE notifying: never fire twice
            print(f"  [reminder] firing: {spoken}")
            self._notify(spoken)
        return len(to_fire)

    async def run(self) -> None:
        """Tick forever. The first pass is the startup/resume reconcile."""
        while True:
            try:
                self.check()
            except Exception as e:  # noqa: BLE001 - a bad reminder must not kill the loop
                print(f"  [reminder] check failed: {type(e).__name__}: {e}")
            await asyncio.sleep(self._tick)
