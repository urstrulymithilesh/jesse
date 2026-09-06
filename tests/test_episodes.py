"""Episodic memory: time windows, the append-only record, and the anchor.

Episodes are EVENTS, not slots. The hard requirement carried over from the
shared-history fix: asked about a time with nothing stored, he must say so rather
than invent a conversation for it.
"""

from datetime import datetime, timedelta

from jesse.context import episode_context
from jesse.memory.episodes import Episode, EpisodeStore, Summariser
from jesse.memory.temporal import (TimeWindow, is_conversation_question,
                                  parse_temporal_query, parse_time_window)

NOW = datetime(2026, 8, 22, 14, 0, 0)          # a Saturday, 2pm


class FakeEmbedder:
    def embed(self, texts):
        return [[float(len(t) % 7), 1.0, 0.5, 0.25] for t in texts]


class FakeLLM:
    supports_tools = False

    def __init__(self, reply="They talked about his car and he teased him."):
        self.reply = reply
        self.seen = None

    def chat(self, messages, *, stream=True):
        self.seen = messages
        yield self.reply


# -- time parsing (pure) ----------------------------------------------------


def test_parses_common_time_references():
    for text, label in [("what did we talk about yesterday", "yesterday"),
                        ("what did we discuss today", "today"),
                        ("what did we talk about this morning", "this morning"),
                        ("what did we discuss last week", "last week")]:
        w = parse_time_window(text, now=NOW)
        assert w is not None and w.label == label, text


def test_yesterday_is_the_whole_of_yesterday():
    w = parse_time_window("what did we talk about yesterday", now=NOW)
    assert w.contains(datetime(2026, 8, 21, 9, 0))
    assert w.contains(datetime(2026, 8, 21, 23, 59))
    assert not w.contains(datetime(2026, 8, 22, 0, 30))    # that is today
    assert not w.contains(datetime(2026, 8, 20, 23, 0))    # that is the day before


def test_n_days_ago():
    w = parse_time_window("what did we talk about 3 days ago", now=NOW)
    assert w.contains(datetime(2026, 8, 19, 12, 0))
    assert not w.contains(datetime(2026, 8, 20, 12, 0))


def test_a_weekday_means_the_most_recent_past_one():
    w = parse_time_window("what did we talk about on tuesday", now=NOW)   # Sat 22nd
    assert w.contains(datetime(2026, 8, 18, 15, 0))        # the Tuesday before
    assert not w.contains(datetime(2026, 8, 22, 15, 0))


def test_vague_recall_searches_all_of_history():
    w = parse_time_window("what did we talk about earlier", now=NOW)
    assert w.start is None and w.end is None


def test_conversation_questions_are_recognised():
    for t in ("what did we talk about yesterday", "what did we discuss",
              "recap our conversation", "what did we chat about last night"):
        assert is_conversation_question(t), t


def test_a_statement_that_merely_mentions_a_day_is_not_a_query():
    """"I went to the gym yesterday" is his telling his something, not asking."""
    assert parse_temporal_query("I went to the gym yesterday", now=NOW) is None
    assert parse_temporal_query("I have a meeting on tuesday", now=NOW) is None


def test_a_fact_question_is_not_routed_to_episodes():
    assert parse_temporal_query("what is my favourite colour", now=NOW) is None
    assert parse_temporal_query("set a timer for ten minutes", now=NOW) is None


def test_when_did_i_tell_you_is_a_temporal_query():
    assert parse_temporal_query("when did I tell you my favourite colour", now=NOW) is not None


# -- the store --------------------------------------------------------------


def _store(tmp_path):
    db = tmp_path / "ep.db"
    import sqlite3
    conn = sqlite3.connect(db)          # the turns table normally already exists
    conn.execute("CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, role TEXT, "
                 "content TEXT, ts TEXT, processed INTEGER DEFAULT 0)")
    conn.commit()
    conn.close()
    return EpisodeStore(db, FakeEmbedder())


def test_episodes_are_append_only_and_never_merged(tmp_path):
    """Two conversations about the same topic are two EVENTS. The fact store would
    dedupe them by subject; the episode store must not."""
    s = _store(tmp_path)
    s.add("They talked about the gym.", NOW - timedelta(days=2), NOW - timedelta(days=2), [])
    s.add("They talked about the gym again.", NOW, NOW, [])
    assert len(s.all_episodes()) == 2


def test_in_window_filters_by_time(tmp_path):
    s = _store(tmp_path)
    s.add("Yesterday's chat.", NOW - timedelta(days=1), NOW - timedelta(days=1), [])
    s.add("Today's chat.", NOW, NOW, [])
    yesterday = parse_time_window("what did we talk about yesterday", now=NOW)
    found = s.in_window(yesterday)
    assert len(found) == 1 and "Yesterday" in found[0].summary


def test_turns_are_marked_so_they_are_summarised_once(tmp_path):
    s = _store(tmp_path)
    s._conn.execute("INSERT INTO turns(id, role, content, ts) VALUES(1,'user','hi',?)",
                    (NOW.isoformat(),))
    s._conn.execute("INSERT INTO turns(id, role, content, ts) VALUES(2,'assistant','hey',?)",
                    (NOW.isoformat(),))
    s._conn.commit()
    assert len(s.unsummarised_turns()) == 2
    s.add("They said hello.", NOW, NOW, [1, 2])
    assert s.unsummarised_turns() == []          # never summarised twice


def test_summariser_builds_a_transcript_and_returns_prose():
    llm = FakeLLM()
    turns = [(1, "user", "my car is light", NOW.isoformat()),
             (2, "assistant", "sounds nippy", NOW.isoformat())]
    out = Summariser(llm).summarise(turns)
    assert out == llm.reply
    transcript = llm.seen[-1].content
    assert "Mithilesh: my car is light" in transcript
    assert "Jesse: sounds nippy" in transcript


# -- the anchor (confabulation guard) ---------------------------------------


def test_nothing_stored_for_that_time_means_say_so():
    """The whole risk of this feature: inventing a Tuesday that never happened."""
    msg = episode_context([], "last Tuesday")
    assert "NO record" in msg.content
    assert "do not invent" in msg.content.lower()
    assert "last Tuesday" in msg.content


def test_real_episodes_are_listed_and_nothing_may_be_added():
    ep = Episode(1, "They talked about his car and he teased him about the upgrade.",
                 NOW, NOW, 4)
    msg = episode_context([ep], "yesterday", now=NOW + timedelta(days=1))
    assert "teased him about the upgrade" in msg.content
    assert "complete" in msg.content.lower()
    assert "not written there" in msg.content.lower()
