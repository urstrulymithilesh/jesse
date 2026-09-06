"""Tests for the progress log + the self-state context it drives.

Pure logic — no db, no LLM, no mic. The key behavior: he only sounds "more alive"
when the newest entry was a REAL capability change (significant=True); otherwise he
reports being about the same.
"""

from jesse.context import self_state_context
from jesse.memory.progress import (
    PROGRESS_LOG,
    ProgressEntry,
    latest,
    previous,
    significant_count,
)

SIG = ProgressEntry("v9 — big", "2026-01-02", "he gained a whole new sense", True)
MINOR = ProgressEntry("v9.1 — tweak", "2026-01-03", "a small wording fix", False)
OLD = ProgressEntry("v8 — before", "2026-01-01", "he was simpler then", True)


# -- log shape --------------------------------------------------------------


def test_log_is_ordered_newest_last_and_nonempty():
    assert PROGRESS_LOG, "progress log should be backfilled with his real history"
    assert latest() is PROGRESS_LOG[-1]
    assert previous() is PROGRESS_LOG[-2]
    dates = [e.date for e in PROGRESS_LOG]
    assert dates == sorted(dates), "entries must be time-ordered, newest last"


def test_significant_count_counts_only_real_changes():
    assert significant_count() == sum(1 for e in PROGRESS_LOG if e.significant)


# -- mood: the significant flag drives how he feels -------------------------


def _mood(msg) -> str:
    """The 'right now you feel ...' clause only — not the trailing instructions."""
    return msg.content.split("right now you feel", 1)[1].split(".")[0]


def test_significant_change_makes_her_feel_more_alive():
    msg = self_state_context(SIG, OLD)
    assert msg is not None and msg.role == "system"
    assert "alive" in _mood(msg)
    assert "same as before" not in _mood(msg)


def test_minor_tweak_means_same_as_before():
    msg = self_state_context(MINOR, SIG)
    assert msg is not None
    assert "about the same" in _mood(msg)
    assert "more alive" not in _mood(msg)


def test_self_state_includes_version_summary_and_previous():
    msg = self_state_context(SIG, OLD)
    assert "v9 — big" in msg.content
    assert "whole new sense" in msg.content
    assert "v8 — before" in msg.content        # he can contrast with his past


def test_self_state_handles_no_previous_entry():
    msg = self_state_context(SIG, None)
    assert msg is not None and "v9 — big" in msg.content


def test_no_progress_yet_means_no_self_state_block():
    assert self_state_context(None, None) is None


def test_self_state_invites_offering_the_reason():
    # behavior 4: he should be told he can offer to explain WHY he feels this way
    msg = self_state_context(SIG, OLD)
    assert "WHY" in msg.content or "why" in msg.content


# -- routing: when do we spend context on self-state? -----------------------


def test_asks_about_self_detects_state_questions():
    from jesse.orchestrator import _asks_about_self
    for q in ["how are you feeling?", "how are you?", "what can you do",
              "what's your current version", "who are you", "how were you before"]:
        assert _asks_about_self(q), q
    for q in ["what's my dog's name?", "it's raining today"]:
        assert not _asks_about_self(q), q


def test_asks_what_next_detects_next_step_questions():
    from jesse.orchestrator import _asks_what_next
    for q in ["what should I do next?", "what's next", "what should I work on",
              "where do I start"]:
        assert _asks_what_next(q), q
    for q in ["how are you?", "what's my dog's name?"]:
        assert not _asks_what_next(q), q


def test_next_step_nudge_forbids_listing_options():
    from jesse.context import next_step_nudge
    c = next_step_nudge().content
    assert "boss" in c and "not list" in c.lower()


def test_self_state_defers_the_reason_until_asked():
    # behavior 4: offer the WHY, don't dump it in the same breath
    c = self_state_context(SIG, OLD).content
    assert "want to know why" in c.lower()
    assert "do NOT explain" in c
