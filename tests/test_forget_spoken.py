"""Spoken "forget that" must actually delete.

He used to agree out loud — "sure, I'll forget that" — while the fact stayed in the
database. Agreeing without acting is worse than refusing, because you stop checking.
Deletion is destructive, so this mirrors reminder cancellation: act on one clear
match, ask when several fit, say so when nothing does, never guess.
"""

import asyncio

from jesse.core.interfaces import Fact
from jesse.memory.forget_parse import ForgetCommand, parse_forget_command
from jesse.memory.store import SqliteMemoryStore

from tests.test_memory import FakeEmbedder


# -- parsing ----------------------------------------------------------------


def test_recognises_forget_phrasings():
    for text in ("forget that my favourite colour is turquoise",
                 "you can forget about my dog",
                 "delete what I said about my job",
                 "stop remembering my birthday",
                 "no longer remember my job"):
        assert isinstance(parse_forget_command(text), ForgetCommand), text


def test_extracts_the_subject():
    assert parse_forget_command("forget that my dog is called Rex").target == "dog is called Rex"
    assert parse_forget_command("you can forget about my job").target == "job"


def test_does_not_fire_on_ordinary_speech():
    for text in ("I forgot my keys", "don't forget to call mum", "I had a rough day",
                 "I never forget a face"):
        assert parse_forget_command(text) is None, text


def test_leaves_reminder_cancellation_to_the_scheduler():
    """"forget the timer" is a cancellation, not a memory deletion — the two parsers
    must not fight over the same sentence."""
    for text in ("forget the timer", "forget the reminder", "delete the alarm"):
        assert parse_forget_command(text) is None, text


def test_a_bare_forget_has_no_target():
    cmd = parse_forget_command("forget")
    assert isinstance(cmd, ForgetCommand) and cmd.target == ""


# -- the store side ---------------------------------------------------------


def _store():
    return SqliteMemoryStore(":memory:", FakeEmbedder())


def test_find_facts_previews_without_deleting():
    s = _store()
    s.add_fact(Fact(text="the user has a dog named Rex", confidence=0.9, subject="pet"))
    assert len(s.find_facts("dog")) == 1
    assert len(s.all_facts()) == 1          # preview must not remove anything


def test_forget_actually_removes_it():
    s = _store()
    s.add_fact(Fact(text="the user has a dog named Rex", confidence=0.9, subject="pet"))
    assert len(s.forget("dog")) == 1
    assert s.all_facts() == []


# -- the orchestrator decision ----------------------------------------------


class _Orch:
    """Just enough orchestrator to exercise _handle_forget_command."""

    def __init__(self, store):
        from jesse.orchestrator import Orchestrator
        self.store = store
        self._handle = Orchestrator._handle_forget_command.__get__(self)


def test_one_clear_match_is_deleted_and_confirmed():
    s = _store()
    s.add_fact(Fact(text="the user's favourite colour is turquoise", confidence=1.0,
                    subject="colour"))
    note = _Orch(s)._handle("forget that my favourite colour is turquoise")
    assert note and "deleted" in note.lower()
    assert s.all_facts() == []              # really gone, not just agreed to


def test_several_matches_asks_and_deletes_nothing():
    s = _store()
    s.add_fact(Fact(text="the user has a dog named Rex", confidence=0.9, subject="dog name"))
    s.add_fact(Fact(text="the user's dog is a spaniel", confidence=0.9, subject="dog breed"))
    note = _Orch(s)._handle("forget about my dog")
    assert note and "which" in note.lower()
    assert len(s.all_facts()) == 2          # nothing destroyed on an ambiguous request


def test_no_match_says_so_instead_of_pretending():
    s = _store()
    s.add_fact(Fact(text="the user likes coffee", confidence=0.9, subject="drink"))
    note = _Orch(s)._handle("forget about my motorbike")
    assert note and "nothing" in note.lower()
    assert len(s.all_facts()) == 1


def test_a_bare_forget_asks_which_and_deletes_nothing():
    s = _store()
    s.add_fact(Fact(text="the user likes coffee", confidence=0.9, subject="drink"))
    note = _Orch(s)._handle("forget")
    assert note and "which" in note.lower()
    assert len(s.all_facts()) == 1


def test_ordinary_speech_produces_no_note():
    s = _store()
    assert _Orch(s)._handle("I had a rough day") is None
