"""Timers and reminders: parsing, persistence, and firing rules.

Everything here runs on a fake clock and a temp SQLite file — no real waiting, no
audio, no model. The scheduler's decisions live in the pure `triage` function, so
"fires on time", "fires late with an apology", and "too stale, stay quiet" are all
ordinary assertions.
"""

from datetime import datetime, timedelta

from jesse.schedule.parse import announcement, parse_schedule_request
from jesse.schedule.scheduler import Scheduler, triage
from jesse.schedule.store import PENDING, SqliteScheduleStore

NOW = datetime(2026, 8, 22, 14, 0, 0)          # a fixed 2:00pm for every test


# -- parsing ----------------------------------------------------------------


def test_parses_relative_timer():
    r = parse_schedule_request("set a timer for 10 minutes", now=NOW)
    assert r is not None
    assert r.fire_at == NOW + timedelta(minutes=10)
    assert r.is_timer is True


def test_parses_relative_reminder_and_keeps_the_task():
    r = parse_schedule_request("remind me to stretch in 20 minutes", now=NOW)
    assert r is not None
    assert r.fire_at == NOW + timedelta(minutes=20)
    assert r.is_timer is False
    assert "stretch" in r.task


def test_parses_seconds_and_hours():
    assert parse_schedule_request("set a timer for 30 seconds", now=NOW).fire_at == \
        NOW + timedelta(seconds=30)
    assert parse_schedule_request("remind me to eat in an hour", now=NOW).fire_at == \
        NOW + timedelta(hours=1)
    assert parse_schedule_request("wake me in half an hour", now=NOW).fire_at == \
        NOW + timedelta(minutes=30)


def test_parses_absolute_time_later_today():
    r = parse_schedule_request("remind me to go to the gym at 5pm", now=NOW)
    assert r is not None
    assert r.fire_at == NOW.replace(hour=17, minute=0)
    assert "gym" in r.task
    assert r.is_timer is False


def test_absolute_time_already_past_rolls_to_tomorrow():
    r = parse_schedule_request("remind me to call mum at 9am", now=NOW)   # 2pm now
    assert r is not None
    assert r.fire_at == (NOW + timedelta(days=1)).replace(hour=9, minute=0)


def test_ordinary_conversation_is_not_a_schedule_request():
    for text in ("what's my dog's name?", "I had a rough day", "how are you feeling",
                 "we met at 5pm yesterday and it was nice"):
        assert parse_schedule_request(text, now=NOW) is None


def test_announcement_admits_lateness_only_when_late():
    assert "due" not in announcement("stretch", is_timer=False)
    late = announcement("stretch", is_timer=False, overdue_seconds=12 * 60)
    assert "12 minutes ago" in late and "stretch" in late


# -- persistence ------------------------------------------------------------


def _store(tmp_path):
    return SqliteScheduleStore(str(tmp_path / "sched.db"))


def test_reminder_survives_a_restart(tmp_path):
    s1 = _store(tmp_path)
    s1.add("go to the gym", NOW + timedelta(hours=3))
    s1.close()                                   # <- app quits

    s2 = _store(tmp_path)                        # <- fresh process, same file
    pending = s2.pending()
    assert len(pending) == 1
    assert pending[0].task == "go to the gym"
    assert pending[0].fire_at == NOW + timedelta(hours=3)


def test_fired_items_are_not_returned_again(tmp_path):
    s = _store(tmp_path)
    rid = s.add("stretch", NOW)
    s.mark(rid, "fired")
    assert s.pending() == []


# -- triage rules (pure) ----------------------------------------------------


def _items(tmp_path, *offsets_s):
    s = _store(tmp_path)
    for i, off in enumerate(offsets_s):
        s.add(f"task {i}", NOW + timedelta(seconds=off))
    return s


def test_not_yet_due_does_not_fire(tmp_path):
    s = _items(tmp_path, 300)                    # due in 5 min
    fire, drop = triage(s.pending(), now=NOW, stale_after_s=7200, overdue_note_after_s=60)
    assert fire == [] and drop == []


def test_due_now_fires_without_an_apology(tmp_path):
    s = _items(tmp_path, 0)
    fire, drop = triage(s.pending(), now=NOW, stale_after_s=7200, overdue_note_after_s=60)
    assert len(fire) == 1 and drop == []
    assert "due" not in fire[0][1]               # on time -> no late note


def test_overdue_fires_with_an_honest_late_note(tmp_path):
    s = _items(tmp_path, -15 * 60)               # came due 15 min ago (slept/closed)
    fire, drop = triage(s.pending(), now=NOW, stale_after_s=7200, overdue_note_after_s=60)
    assert len(fire) == 1 and drop == []
    assert "15 minutes ago" in fire[0][1]


def test_stale_reminder_is_dropped_not_announced(tmp_path):
    s = _items(tmp_path, -5 * 3600)              # 5 hours late, staleness is 2 hours
    fire, drop = triage(s.pending(), now=NOW, stale_after_s=7200, overdue_note_after_s=60)
    assert fire == [] and len(drop) == 1


# -- scheduler end to end (fake clock, fake notify) -------------------------


def test_check_notifies_and_never_fires_the_same_item_twice(tmp_path):
    store = _store(tmp_path)
    spoken: list[str] = []
    sched = Scheduler(store, spoken.append, stale_after_s=7200, overdue_note_after_s=60)
    sched.add("stretch", NOW)

    assert sched.check(now=NOW) == 1
    assert len(spoken) == 1 and "stretch" in spoken[0]
    assert sched.check(now=NOW + timedelta(minutes=1)) == 0   # already fired
    assert len(spoken) == 1


def test_startup_reconcile_fires_what_came_due_while_closed(tmp_path):
    store = _store(tmp_path)
    store.add("go to the gym", NOW)              # set, then the app closes
    store.close()

    reopened = _store(tmp_path)                  # restart, 20 minutes later
    spoken: list[str] = []
    sched = Scheduler(reopened, spoken.append, stale_after_s=7200, overdue_note_after_s=60)
    fired = sched.check(now=NOW + timedelta(minutes=20))

    assert fired == 1
    assert "gym" in spoken[0] and "20 minutes ago" in spoken[0]
    assert reopened.pending() == []              # closed out, won't repeat


# -- cancel / reschedule ----------------------------------------------------


from jesse.schedule.parse import (CancelCommand, RescheduleCommand,   # noqa: E402
                                 parse_schedule_command)
from jesse.schedule.scheduler import resolve_target                    # noqa: E402


def test_cancel_phrasing_is_not_mistaken_for_a_new_timer():
    """The reported bug: this used to CREATE a second 10-minute timer."""
    cmd = parse_schedule_command("can you stop the timer that was set for 10 minutes?", now=NOW)
    assert isinstance(cmd, CancelCommand)


def test_reschedule_phrasing_is_not_mistaken_for_a_new_timer():
    """The other reported bug: this used to ADD a 1-minute timer alongside the old one."""
    cmd = parse_schedule_command("change the timer to 1 minute", now=NOW)
    assert isinstance(cmd, RescheduleCommand)
    assert cmd.fire_at == NOW + timedelta(minutes=1)


def test_various_cancel_and_reschedule_phrasings():
    for text in ("cancel the timer", "never mind the reminder", "forget the alarm",
                 "stop the timer"):
        assert isinstance(parse_schedule_command(text, now=NOW), CancelCommand), text
    for text in ("make it 5 minutes instead", "move the reminder to 6pm",
                 "change the timer to 1 minute"):
        assert isinstance(parse_schedule_command(text, now=NOW), RescheduleCommand), text


def test_ordinary_sentences_with_stop_or_change_are_left_alone():
    for text in ("I need to stop working at 5pm", "I might change jobs in 3 months"):
        assert not isinstance(parse_schedule_command(text, now=NOW), CancelCommand), text


def test_reschedule_moves_the_existing_one_instead_of_adding(tmp_path):
    store = _store(tmp_path)
    sched = Scheduler(store, lambda _t: None)
    sched.add("", NOW + timedelta(minutes=10), is_timer=True)

    item, reason = sched.reschedule(NOW + timedelta(minutes=1))
    assert reason == "" and item is not None
    pending = store.pending()
    assert len(pending) == 1                                   # still exactly one
    assert pending[0].fire_at == NOW + timedelta(minutes=1)     # at the NEW time


def test_cancel_removes_it_and_it_never_fires(tmp_path):
    store = _store(tmp_path)
    spoken = []
    sched = Scheduler(store, spoken.append)
    sched.add("", NOW, is_timer=True)

    count, reason = sched.cancel()
    assert count == 1 and reason == ""
    assert store.pending() == []
    assert sched.check(now=NOW + timedelta(minutes=5)) == 0     # due, but cancelled
    assert spoken == []                                        # never announced


def test_cancel_with_nothing_pending_says_so(tmp_path):
    sched = Scheduler(_store(tmp_path), lambda _t: None)
    count, reason = sched.cancel()
    assert count == 0 and reason == "none"


def test_ambiguous_cancel_refuses_to_guess(tmp_path):
    store = _store(tmp_path)
    sched = Scheduler(store, lambda _t: None)
    sched.add("go to the gym", NOW + timedelta(hours=1))
    sched.add("call mum", NOW + timedelta(hours=2))

    count, reason = sched.cancel()                 # no hint, two candidates
    assert count == 0 and reason == "ambiguous"
    assert len(store.pending()) == 2               # nothing destroyed on a guess


def test_a_hint_disambiguates_when_it_matches_one(tmp_path):
    store = _store(tmp_path)
    sched = Scheduler(store, lambda _t: None)
    sched.add("go to the gym", NOW + timedelta(hours=1))
    sched.add("call mum", NOW + timedelta(hours=2))

    count, reason = sched.cancel("gym")
    assert count == 1 and reason == ""
    assert [p.task for p in store.pending()] == ["call mum"]


def test_cancel_all_clears_everything(tmp_path):
    store = _store(tmp_path)
    sched = Scheduler(store, lambda _t: None)
    sched.add("go to the gym", NOW + timedelta(hours=1))
    sched.add("call mum", NOW + timedelta(hours=2))

    count, reason = sched.cancel(all_of_them=True)
    assert count == 2 and reason == ""
    assert store.pending() == []


def test_ambiguous_reschedule_also_refuses_to_guess(tmp_path):
    store = _store(tmp_path)
    sched = Scheduler(store, lambda _t: None)
    sched.add("go to the gym", NOW + timedelta(hours=1))
    sched.add("call mum", NOW + timedelta(hours=2))

    item, reason = sched.reschedule(NOW + timedelta(minutes=5))
    assert item is None and reason == "ambiguous"
    assert [p.fire_at for p in store.pending()] == [NOW + timedelta(hours=1),
                                                    NOW + timedelta(hours=2)]


def test_resolve_target_rules():
    from jesse.schedule.store import ScheduledItem
    a = ScheduledItem(1, "go to the gym", NOW, False)
    b = ScheduledItem(2, "call mum", NOW, False)
    assert resolve_target([], "") == (None, "none")
    assert resolve_target([a], "")[0] is a           # only one -> unambiguous
    assert resolve_target([a, b], "")[1] == "ambiguous"
    assert resolve_target([a, b], "gym")[0] is a     # hint picks it out


def test_a_bare_timer_can_be_named_by_its_duration(tmp_path):
    """He says "stop the timer set for 10 minutes" — a timer has no task text, so the
    label it was created with is the only thing that can identify it."""
    store = _store(tmp_path)
    sched = Scheduler(store, lambda _t: None)
    sched.add("", NOW + timedelta(minutes=10), is_timer=True, label="10 minutes")
    sched.add("", NOW + timedelta(minutes=5), is_timer=True, label="5 minutes")

    count, reason = sched.cancel("10 minutes")
    assert count == 1 and reason == ""
    assert [p.label for p in store.pending()] == ["5 minutes"]


def test_a_hint_matching_several_equally_stays_ambiguous(tmp_path):
    store = _store(tmp_path)
    sched = Scheduler(store, lambda _t: None)
    sched.add("", NOW + timedelta(minutes=10), is_timer=True, label="10 minutes")
    sched.add("", NOW + timedelta(minutes=5), is_timer=True, label="5 minutes")

    count, reason = sched.cancel("minutes")     # matches both equally
    assert count == 0 and reason == "ambiguous"
    assert len(store.pending()) == 2


# -- asking what's pending --------------------------------------------------


from jesse.schedule.parse import IncompleteCommand, QueryCommand   # noqa: E402


def test_query_phrasings_are_recognised():
    for text in ("do I have any timers running?", "any reminders?",
                 "what is my timer set for", "when does my timer go off",
                 "what timers do I have", "is there a timer still running"):
        assert isinstance(parse_schedule_command(text, now=NOW), QueryCommand), text


def test_a_question_about_a_timer_does_not_create_one():
    """"do I have a timer for 10 minutes?" contains a perfectly good duration."""
    cmd = parse_schedule_command("do I have a timer for 10 minutes?", now=NOW)
    assert isinstance(cmd, QueryCommand)


def test_query_does_not_fire_on_ordinary_questions():
    for text in ("do I have time for a coffee", "what did I have for lunch",
                 "how are you feeling"):
        assert not isinstance(parse_schedule_command(text, now=NOW), QueryCommand), text


def test_query_is_a_pure_lookup(tmp_path):
    """No model involved in finding the facts — just the table."""
    store = _store(tmp_path)
    sched = Scheduler(store, lambda _t: None)
    assert sched.pending() == []
    sched.add("go to the gym", NOW + timedelta(hours=1), label="an hour")
    sched.add("", NOW + timedelta(minutes=5), is_timer=True, label="5 minutes")
    pending = sched.pending()
    assert len(pending) == 2
    assert {p.task or p.label for p in pending} == {"go to the gym", "5 minutes"}


def test_cancelled_and_fired_items_drop_out_of_the_query(tmp_path):
    store = _store(tmp_path)
    sched = Scheduler(store, lambda _t: None)
    sched.add("go to the gym", NOW + timedelta(hours=1))
    sched.add("call mum", NOW + timedelta(hours=2))
    sched.cancel("gym")
    assert [p.task for p in sched.pending()] == ["call mum"]


# -- a change with no time asks instead of silently doing nothing -----------


def test_reschedule_without_a_time_is_flagged_incomplete():
    cmd = parse_schedule_command("please change the timer", now=NOW)
    assert isinstance(cmd, IncompleteCommand) and cmd.kind == "reschedule"


def test_incomplete_needs_an_explicit_reminder_word():
    """Guard against ordinary speech: no bare pronouns here, or "I might change it
    later" would turn into a reminder prompt."""
    assert parse_schedule_command("I might change it later", now=NOW) is None
    assert parse_schedule_command("can you change that", now=NOW) is None


def test_a_bare_time_alone_is_not_a_command():
    """We deliberately do NOT slot-fill across turns, so "45 seconds" on its own is
    just conversation — the reschedule has to be said in one sentence."""
    assert parse_schedule_command("45 seconds", now=NOW) is None
