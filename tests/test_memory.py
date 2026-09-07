"""Memory tests: store / dedupe / recall / read-budget / conflict / persistence,
plus extraction-output rejection. All with a fake deterministic embedder and an
in-memory (or tmp-file) SQLite — no fastembed model, no mic, no LLM.
"""

import zlib

from jesse.core.interfaces import Fact, Message
from jesse.memory.extraction import parse_extracted_facts
from jesse.memory.store import SqliteMemoryStore


IDENTITY_DIMS = 16
IDENTITY_WEIGHT = 0.7


class FakeEmbedder:
    """Deterministic keyword embedding so recall is predictable in tests.

    Topic dims [sister, drink, gym, bias] drive recall: sharing a topic keeps two
    vectors close. The trailing identity dims give each DISTINCT string its own small
    orthogonal component, so two unrelated subjects ("pet_name" vs "dog's name") don't
    come out perfectly identical and get merged by subject dedupe. crc32, not hash(),
    because hash() is salted per process and would make these tests flaky.
    """

    def _vec(self, t: str) -> list[float]:
        t = t.lower()
        topic = [
            1.0 if "sister" in t else 0.0,
            1.0 if ("coffee" in t or "tea" in t or "drink" in t) else 0.0,
            1.0 if ("gym" in t or "workout" in t) else 0.0,
            1.0,  # bias keeps every vector non-zero
        ]
        identity = [0.0] * IDENTITY_DIMS
        identity[zlib.crc32(t.encode()) % IDENTITY_DIMS] = IDENTITY_WEIGHT
        return topic + identity

    def embed(self, texts):
        return [self._vec(t) for t in texts]


class SubjectEmbedder(FakeEmbedder):
    """Treats listed strings as the SAME slot, for exercising near-duplicate merging
    without depending on a real model."""

    def __init__(self, synonyms=()):
        self._canon = {}
        for group in synonyms:
            for word in group:
                self._canon[word.lower()] = group[0].lower()

    def _vec(self, t: str) -> list[float]:
        return super()._vec(self._canon.get(t.lower(), t))


def _store(tmp_path=None, log=None):
    db = str(tmp_path / "mem.db") if tmp_path else ":memory:"
    return SqliteMemoryStore(db, FakeEmbedder(), log_path=log)


# -- store + recall ---------------------------------------------------------


def test_store_and_semantic_recall():
    s = _store()
    s.add_fact(Fact(text="the user's sister is named Anya", confidence=0.9, subject="sister's name"))
    s.add_fact(Fact(text="the user likes coffee in the morning", confidence=0.8, subject="drink"))
    hits = s.recall("tell me about my sister", k=3)
    assert hits and "Anya" in hits[0].text          # nearest is the sister fact
    assert hits[0].subject == "sister's name"


def test_recall_empty_store_returns_nothing():
    assert _store().recall("anything", k=3) == []


def test_recall_respects_k_read_budget():
    s = _store()
    for i in range(5):
        s.add_fact(Fact(text=f"the user has hobby number {i}", confidence=0.9, subject=f"hobby-{i}"))
    assert len(s.recall("hobbies", k=3)) == 3       # budget cap honored


# -- dedupe + conflict ------------------------------------------------------


def _count(s) -> int:
    return s._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]


def test_exact_text_dedupe_when_no_subject():
    s = _store()
    s.add_fact(Fact(text="the user is learning guitar", confidence=0.8))
    s.add_fact(Fact(text="the user is learning guitar", confidence=0.8))  # identical
    assert _count(s) == 1


def test_conflict_last_write_wins_on_subject():
    s = _store()
    s.add_fact(Fact(text="the user's sister is named Anya", confidence=0.9, subject="sister's name"))
    s.add_fact(Fact(text="the user's sister is named Anaya", confidence=0.95, subject="sister's name"))
    assert _count(s) == 1                            # replaced, not duplicated
    hits = s.recall("my sister", k=3)
    assert "Anaya" in hits[0].text                   # the newer value wins


# -- turns ------------------------------------------------------------------


def test_recent_turns_in_chronological_order():
    s = _store()
    s.append_turn(Message("user", "one"))
    s.append_turn(Message("assistant", "two"))
    s.append_turn(Message("user", "three"))
    recent = s.recent(limit=2)
    assert [m.content for m in recent] == ["two", "three"]  # last 2, oldest-first


# -- memory log -------------------------------------------------------------


def test_memory_log_records_writes(tmp_path):
    log = tmp_path / "memory-log.txt"
    s = _store(log=log)
    s.add_fact(Fact(text="the user's sister is named Anya", confidence=0.9, subject="sister's name"))
    s.add_fact(Fact(text="the user's sister is named Anaya", confidence=0.95, subject="sister's name"))
    text = log.read_text(encoding="utf-8")
    assert "STORED" in text and "UPDATED" in text
    assert "Anya" in text and "Anaya" in text


# -- persistence (the real-proof at unit level) -----------------------------


def test_facts_persist_across_reopen(tmp_path):
    db = str(tmp_path / "mem.db")
    s1 = SqliteMemoryStore(db, FakeEmbedder())
    s1.add_fact(Fact(text="the user's sister is named Anya", confidence=0.9, subject="sister's name"))
    s1.close()

    s2 = SqliteMemoryStore(db, FakeEmbedder())          # fresh connection, like an app restart
    hits = s2.recall("tell me about my sister", k=3)
    assert hits and "Anya" in hits[0].text


# -- extraction rejection ---------------------------------------------------


def test_parse_valid_facts():
    raw = '[{"subject":"sister","text":"the user has a sister named Anya","confidence":0.9}]'
    facts = parse_extracted_facts(raw)
    assert len(facts) == 1 and facts[0].subject == "sister" and facts[0].confidence == 0.9


def test_parse_malformed_json_returns_empty():
    assert parse_extracted_facts("not json at all") == []
    assert parse_extracted_facts("") == []
    assert parse_extracted_facts('{"not": "a list"}') == []


def test_parse_gates_low_confidence():
    raw = '[{"text":"maybe true","confidence":0.3}]'
    assert parse_extracted_facts(raw, min_confidence=0.6) == []


def test_parse_drops_items_missing_text_or_bad_confidence():
    raw = '[{"confidence":0.9},{"text":"","confidence":0.9},{"text":"ok","confidence":"high"}]'
    assert parse_extracted_facts(raw) == []


def test_parse_tolerates_code_fence():
    raw = '```json\n[{"text":"the user codes in python","confidence":0.8}]\n```'
    facts = parse_extracted_facts(raw)
    assert len(facts) == 1 and "python" in facts[0].text


def test_parse_drops_her_own_speech_and_questions():
    """Live db junk: three of nine learned facts were Jesse's own lines, filed as facts
    about him. The prompt already forbids it; the model does it anyway."""
    raw = ("[{\"subject\":\"practicing\",\"text\":\"I'll start practicing the Indian accent.\",\"confidence\":0.9},"
           "{\"subject\":\"mood\",\"text\":\"I'm energized, though.\",\"confidence\":0.6},"
           '{"subject":"practicing2","text":"I will start practicing the Indian accent.","confidence":0.9},'
           '{"subject":"energy","text":"I am energized, though.","confidence":0.6},'
           '{"subject":"accent","text":"What would you like to practice with that accent?","confidence":0.7},'
           '{"subject":"gym","text":"the user goes to the gym on Tuesdays","confidence":0.9}]')
    facts = parse_extracted_facts(raw)
    assert [f.subject for f in facts] == ["gym"]


def test_parse_keeps_facts_that_merely_contain_i_or_a_question_mark():
    """Only the SHAPE is rejected — first word, or a trailing '?'. A real fact that
    happens to quote his must survive."""
    raw = ('[{"subject":"job","text":"the user said I should ask about his job","confidence":0.9},'
           "{\"subject\":\"band\",\"text\":\"the user's favourite band is Wire\",\"confidence\":0.9}]")
    assert len(parse_extracted_facts(raw)) == 2


# -- protection + history gating + seeding ----------------------------------


def test_conversational_extraction_cannot_overwrite_a_core_fact():
    s = _store()
    s.add_fact(Fact(text="the user's name is Mithilesh", confidence=1.0,
                    subject="user's name", origin="core"))
    # an offhand conversational fact on the same subject must NOT win
    s.add_fact(Fact(text="the user's name is Bob", confidence=0.9, subject="user's name"))
    facts = s.all_facts()
    assert len(facts) == 1
    assert "Mithilesh" in facts[0].text and facts[0].origin == "core"


def test_seeding_can_update_a_protected_fact():
    s = _store()
    s.add_fact(Fact(text="Jesse is at build v1", confidence=1.0, subject="self: version", origin="self"))
    s.add_fact(Fact(text="Jesse is at build v2", confidence=1.0, subject="self: version", origin="self"))
    facts = s.all_facts()
    assert len(facts) == 1 and "v2" in facts[0].text  # seed origin updates itself


def test_self_history_hidden_unless_explicitly_requested():
    s = _store()
    s.add_fact(Fact(text="Jesse used to sound robotic", confidence=1.0,
                    subject="self-history: v0", origin="self_history"))
    assert s.recall("tell me about your old voice", k=3) == []          # hidden by default
    hits = s.recall("tell me about your old voice", k=3, include_history=True)
    assert hits and "robotic" in hits[0].text                            # shown when asked


def test_seed_plants_protected_facts_and_is_idempotent():
    from jesse.memory.seed import seed, seed_if_needed
    s = _store()
    assert seed(s) > 0
    assert any(f.origin == "core" and "Jesse" in f.text for f in s.all_facts())
    assert seed_if_needed(s) == 0  # unchanged content -> no-op


def test_edited_seed_content_reaches_an_already_seeded_db():
    """The old gate was "any core facts?", so editing seed.py did nothing to a live db.
    He kept calling himself a companion and naming a model he no longer ran on."""
    from jesse.memory import seed as seed_mod
    s = _store()
    seed_mod.seed(s)
    original = seed_mod.CORE_FACTS[0]
    seed_mod.CORE_FACTS[0] = Fact(subject=original.subject, text="the AI partner's name is Jesse, v2",
                                  confidence=1.0, origin="core")
    try:
        assert seed_mod.seed_if_needed(s) > 0
        assert any("v2" in f.text for f in s.all_facts())
    finally:
        seed_mod.CORE_FACTS[0] = original


# -- unprocessed-exchange tracking (extraction retry) -----------------------


def test_unprocessed_exchanges_pairs_user_and_assistant():
    s = _store()
    s.append_turn(Message("user", "my dog is Rex"))
    s.append_turn(Message("assistant", "Rex, nice"))
    pending = s.unprocessed_exchanges()
    assert len(pending) == 1
    uid, aid, utext, atext = pending[0]
    assert utext == "my dog is Rex" and atext == "Rex, nice" and uid < aid


def test_mark_processed_means_never_extracted_twice():
    s = _store()
    uid = s.append_turn(Message("user", "my dog is Rex"))
    aid = s.append_turn(Message("assistant", "Rex, nice"))
    assert len(s.unprocessed_exchanges()) == 1
    s.mark_processed((uid, aid))
    assert s.unprocessed_exchanges() == []      # idempotent: gone for good


def test_unprocessed_respects_limit_and_ordering():
    s = _store()
    for i in range(4):
        s.append_turn(Message("user", f"fact {i}"))
        s.append_turn(Message("assistant", f"reply {i}"))
    pending = s.unprocessed_exchanges(limit=2)
    assert len(pending) == 2
    assert pending[0][2] == "fact 0" and pending[1][2] == "fact 1"   # oldest first


def test_unprocessed_skips_an_unpaired_straggler():
    s = _store()
    s.append_turn(Message("assistant", "orphan reply"))   # no user turn before it
    s.append_turn(Message("user", "my dog is Rex"))
    s.append_turn(Message("assistant", "Rex, nice"))
    pending = s.unprocessed_exchanges()
    assert len(pending) == 1 and pending[0][2] == "my dog is Rex"


# -- forgetting (the correction path) ---------------------------------------


def test_forget_removes_a_fact_and_it_stops_being_recalled():
    s = _store()
    s.add_fact(Fact(text="the user lives in Pune", confidence=1.0, subject="location"))
    s.add_fact(Fact(text="the user's sister is named Anya", confidence=0.9, subject="sister's name"))

    gone = s.forget("Pune")
    assert len(gone) == 1 and "Pune" in gone[0].text
    assert [f.subject for f in s.all_facts()] == ["sister's name"]
    assert all("Pune" not in f.text for f in s.recall("where do I live", k=3))


def test_forget_matches_subject_as_well_as_text():
    s = _store()
    s.add_fact(Fact(text="the user drinks coffee", confidence=0.9, subject="drink"))
    assert len(s.forget("drink")) == 1
    assert s.all_facts() == []


def test_forget_is_case_insensitive_and_can_remove_several():
    s = _store()
    s.add_fact(Fact(text="the user has a dog named Rex", confidence=0.9, subject="pet_name"))
    s.add_fact(Fact(text="the user's dog is named Rex", confidence=1.0, subject="dog's name"))
    assert len(s.forget("rex")) == 2
    assert s.all_facts() == []


def test_forget_with_no_match_changes_nothing():
    s = _store()
    s.add_fact(Fact(text="the user drinks coffee", confidence=0.9, subject="drink"))
    assert s.forget("kangaroo") == []
    assert len(s.all_facts()) == 1


def test_forget_can_remove_a_seeded_fact_too(tmp_path):
    log = tmp_path / "memory-log.txt"
    s = _store(log=log)
    s.add_fact(Fact(text="the user's name is Mithilesh", confidence=1.0,
                    subject="user's name", origin="core"))
    gone = s.forget("Mithilesh")
    assert len(gone) == 1 and gone[0].origin == "core"
    assert "FORGOT" in log.read_text(encoding="utf-8")   # the deletion is auditable


# -- near-duplicate subjects (semantic dedupe) ------------------------------


def _dedupe_store(synonyms):
    return SqliteMemoryStore(":memory:", SubjectEmbedder(synonyms))


BIRTHDAY = ("birthday month", "birthday_month")


def test_near_duplicate_subjects_merge_instead_of_both_persisting():
    """The reported bug: birthday_month and birthday month were two separate facts."""
    s = _dedupe_store([BIRTHDAY])
    s.add_fact(Fact(text="the user is born in November", confidence=1.0,
                    subject="birthday_month"))
    s.add_fact(Fact(text="the user's birthday month is November", confidence=1.0,
                    subject="birthday month"))

    facts = s.all_facts()
    assert len(facts) == 1, [f.subject for f in facts]
    assert facts[0].text == "the user's birthday month is November"   # newer wins
    assert facts[0].subject == "birthday month"


def test_merged_fact_is_recalled_once_not_twice():
    s = _dedupe_store([BIRTHDAY])
    s.add_fact(Fact(text="the user is born in November", confidence=1.0, subject="birthday_month"))
    s.add_fact(Fact(text="the user's birthday month is November", confidence=1.0,
                    subject="birthday month"))
    hits = s.recall("when is my birthday", k=5)
    assert len(hits) == 1


def test_genuinely_different_subjects_stay_separate():
    """sister's name vs brother's name score 0.822 on the real model — close, but they
    must never merge, or a real fact is destroyed."""
    s = _dedupe_store([])            # no synonyms: these are distinct slots
    s.add_fact(Fact(text="the user's sister is named Anya", confidence=0.9,
                    subject="sister's name"))
    s.add_fact(Fact(text="the user's brother is named Bob", confidence=0.9,
                    subject="brother's name"))
    assert len(s.all_facts()) == 2


def test_a_conversational_near_duplicate_cannot_overwrite_a_core_fact():
    s = _dedupe_store([("user's name", "the user's name")])
    s.add_fact(Fact(text="the user's name is Mithilesh", confidence=1.0,
                    subject="user's name", origin="core"))
    s.add_fact(Fact(text="the user's name is Bob", confidence=0.9,
                    subject="the user's name"))          # near-dup subject, conversational
    facts = s.all_facts()
    assert len(facts) == 1
    assert "Mithilesh" in facts[0].text and facts[0].origin == "core"


def test_a_merge_is_written_to_the_memory_log(tmp_path):
    log = tmp_path / "memory-log.txt"
    s = SqliteMemoryStore(str(tmp_path / "m.db"), SubjectEmbedder([BIRTHDAY]), log_path=log)
    s.add_fact(Fact(text="the user is born in November", confidence=1.0, subject="birthday_month"))
    s.add_fact(Fact(text="the user's birthday month is November", confidence=1.0,
                    subject="birthday month"))
    assert "MERGED" in log.read_text(encoding="utf-8")


def test_forget_clears_the_subject_vector_too():
    """A stale subject vector would keep matching after its fact was deleted."""
    s = _dedupe_store([])
    s.add_fact(Fact(text="the user drinks coffee", confidence=0.9, subject="drink"))
    s.forget("coffee")
    assert s.all_facts() == []
    left = s._conn.execute("SELECT COUNT(*) FROM subject_vectors").fetchone()[0]
    assert left == 0


# -- retroactive dedupe (jesse memory --dedupe) ------------------------------


def test_dedupe_preview_finds_a_real_duplicate_and_changes_nothing():
    s = _dedupe_store([BIRTHDAY])
    # inserted directly so both survive, mimicking facts stored before dedupe existed
    for subject, text in [("birthday_month", "the user is born in November"),
                          ("birthday month", "the user's birthday month is November")]:
        s._conn.execute(
            "INSERT INTO facts(subject, text, confidence, source_turn_id, ts, origin) "
            "VALUES(?, ?, 1.0, NULL, '2026-01-01', 'conversation')", (subject, text))
    s._conn.commit()

    groups = s.duplicate_groups()               # preview only
    assert len(groups) == 1
    keeper, dups = groups[0]
    assert len(dups) == 1 and dups[0][1] >= 0.88
    assert len(s.all_facts()) == 2              # DRY RUN: nothing removed


def test_dedupe_apply_merges_and_keeps_one():
    s = _dedupe_store([BIRTHDAY])
    for subject, text in [("birthday_month", "the user is born in November"),
                          ("birthday month", "the user's birthday month is November")]:
        s._conn.execute(
            "INSERT INTO facts(subject, text, confidence, source_turn_id, ts, origin) "
            "VALUES(?, ?, 1.0, NULL, '2026-01-01', 'conversation')", (subject, text))
    s._conn.commit()

    removed = s.merge_duplicates()
    assert removed == 1
    assert len(s.all_facts()) == 1


def test_dedupe_never_merges_a_false_positive_pair():
    """sister's/brother's name score 0.822 on the real model — under the 0.88 bar."""
    s = _dedupe_store([])
    s.add_fact(Fact(text="the user's sister is named Anya", confidence=0.9,
                    subject="sister's name"))
    s.add_fact(Fact(text="the user's brother is named Bob", confidence=0.9,
                    subject="brother's name"))
    assert s.duplicate_groups() == []
    assert s.merge_duplicates() == 0
    assert len(s.all_facts()) == 2


def test_two_seeded_facts_are_never_merged_into_each_other():
    """The bug this rule exists for: "jesse's creator" and "jesse's name" score 0.885,
    over the threshold, and seeding silently lost "jesse's name"."""
    s = _dedupe_store([("jesse's creator", "jesse's name")])   # forced to look identical
    s.add_fact(Fact(text="Jesse was created by Mithilesh", confidence=1.0,
                    subject="jesse's creator", origin="core"))
    s.add_fact(Fact(text="the AI partner's name is Jesse", confidence=1.0,
                    subject="jesse's name", origin="core"))

    assert len(s.all_facts()) == 2, "two seeded facts must never collapse"
    assert s.duplicate_groups() == []
    assert s.merge_duplicates() == 0


def test_seeding_twice_keeps_every_seeded_fact():
    from jesse.memory.seed import all_seed_facts, seed
    s = _dedupe_store([])
    seed(s)
    seed(s)
    assert len(s.all_facts()) == len(all_seed_facts())


def test_dedupe_backfills_subject_vectors_for_legacy_facts():
    """Facts stored before subject vectors existed were invisible to dedupe."""
    s = _dedupe_store([])
    s._conn.execute(
        "INSERT INTO facts(subject, text, confidence, source_turn_id, ts, origin) "
        "VALUES('drink', 'the user drinks coffee', 1.0, NULL, '2026-01-01', 'conversation')")
    s._conn.commit()
    assert s._conn.execute("SELECT COUNT(*) FROM subject_vectors").fetchone()[0] == 0

    s.duplicate_groups()
    assert s._conn.execute("SELECT COUNT(*) FROM subject_vectors").fetchone()[0] == 1


def test_a_preference_jesse_states_can_never_become_a_stored_fact():
    """The leak that matters is not him saying a taste out loud — it is that taste
    surviving as memory and being repeated back as true.

    Probed live: asked whether he likes pineapple on pizza he answers with a taste he
    does not have, 8 times in 9, and no prompt wording fixed it. But "Jesse does not
    like pineapple on pizza" is third person and well formed, so the old filter would
    have stored it forever. Shutting that door makes the spoken slip harmless.
    """
    raw = ('[{"subject":"pizza","text":"Jesse does not like pineapple on pizza","confidence":0.9},'
           '{"subject":"food","text":"Jesse\'s favourite food is pizza","confidence":0.9},'
           '{"subject":"food","text":"Nah.","confidence":0.9},'
           '{"subject":"birthday","text":"the user\'s birthday month is November","confidence":1.0}]')
    kept = parse_extracted_facts(raw)
    assert [f.text for f in kept] == ["the user's birthday month is November"]


def test_real_facts_are_not_caught_by_the_new_filters():
    """A filter that eats real facts is worse than the leak it prevents."""
    from jesse.memory.extraction import _is_junk

    for text in ("the user's sister is named Anya",
                 "the user works as a data analyst at a hospital",
                 "Mithilesh's favourite colour is black",
                 "the user goes to the gym on Tuesday evenings"):
        assert not _is_junk(text), text
