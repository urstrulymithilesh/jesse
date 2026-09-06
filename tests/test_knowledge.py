"""Learned-document memory: chunking, ingest, the distance gate, and the injected block.

Fake embedder, tmp db — no fastembed model, no LLM. The gate VALUE itself was chosen
from real bge-small numbers (see config.knowledge.max_distance); what is tested here is
that a gate is applied at all, and that the block never lets his go beyond the text.
"""

import zlib

from jesse.context import knowledge_context
from jesse.memory.corpus import CorpusStore, chunk_text


class FakeEmbedder:
    """Deterministic bag-of-words vector so distances are predictable. Words shared
    between query and chunk pull them together; nothing else does."""

    VOCAB = ("guitar", "string", "tune", "humidity", "dinner", "work", "sister")

    def embed(self, texts):
        out = []
        for t in texts:
            low = t.lower()
            vec = [1.0 if w in low else 0.0 for w in self.VOCAB]
            # A tiny identity component so two unrelated texts are never identical.
            vec.append((zlib.crc32(low.encode()) % 1000) / 100000.0)
            out.append(vec)
        return out


def _store(tmp_path):
    return CorpusStore(tmp_path / "k.db", FakeEmbedder())


# -- chunking ---------------------------------------------------------------


def test_chunks_break_on_paragraphs_not_mid_sentence():
    text = "one two three.\n\nfour five six.\n\nseven eight nine."
    chunks = chunk_text(text, chunk_chars=20)
    assert chunks == ["one two three.", "four five six.", "seven eight nine."]
    assert all(not c.startswith(" ") for c in chunks)


def test_short_paragraphs_are_packed_together():
    text = "aaa\n\nbbb\n\nccc"
    assert chunk_text(text, chunk_chars=1000) == ["aaa\n\nbbb\n\nccc"]


def test_a_paragraph_longer_than_the_budget_is_kept_whole():
    """Splitting it would do exactly the mid-sentence damage the packing avoids."""
    long = "x" * 500
    assert chunk_text(long, chunk_chars=100) == [long]


# -- ingest -----------------------------------------------------------------


def test_ingest_stores_and_lists(tmp_path):
    doc = tmp_path / "guitar.md"
    doc.write_text("tune the guitar\n\nchange the string", encoding="utf-8")
    s = _store(tmp_path)
    assert s.ingest("guitar", doc, chunk_chars=20) == 2
    assert s.corpora() == [("guitar", 2, 1)]


def test_reingesting_a_source_replaces_it(tmp_path):
    """Fixing a typo and running again must not double every answer."""
    doc = tmp_path / "guitar.md"
    doc.write_text("tune the guitar", encoding="utf-8")
    s = _store(tmp_path)
    s.ingest("guitar", doc)
    doc.write_text("tune the guitar properly", encoding="utf-8")
    s.ingest("guitar", doc)
    assert s.corpora() == [("guitar", 1, 1)]


def test_ingest_reads_a_whole_folder_and_skips_unreadable_types(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("guitar", encoding="utf-8")
    (folder / "b.txt").write_text("string", encoding="utf-8")
    (folder / "c.pdf").write_bytes(b"%PDF-not-read")
    s = _store(tmp_path)
    assert s.ingest("guitar", folder) == 2


def test_forget_drops_the_whole_corpus(tmp_path):
    doc = tmp_path / "guitar.md"
    doc.write_text("tune the guitar\n\nchange the string", encoding="utf-8")
    s = _store(tmp_path)
    s.ingest("guitar", doc, chunk_chars=20)
    assert s.forget("guitar") == 2
    assert s.corpora() == []
    assert s.search("guitar", max_distance=9) == []


# -- the gate ---------------------------------------------------------------


def test_the_subject_name_is_the_trigger():
    """A pure distance gate did not survive a second corpus — measured, the margin
    INVERTED at six passages. His own words don't drift as the corpus grows."""
    from jesse.memory.corpus import subjects_mentioned
    names = ["guitar", "sourdough"]
    assert subjects_mentioned("how do I tune my guitar", names) == ["guitar"]
    assert subjects_mentioned("I think I'll cook something tonight", names) == []
    assert subjects_mentioned("how was your day", names) == []


def test_subject_matching_is_word_boundary_not_substring():
    """"guitarist" is not the guitar corpus, and "a sour taste" is not sourdough."""
    from jesse.memory.corpus import subjects_mentioned
    names = ["guitar", "sour"]
    assert subjects_mentioned("he's a brilliant guitarist", names) == []
    assert subjects_mentioned("that left a sour taste", names) == ["sour"]


def test_search_can_be_restricted_to_named_corpora(tmp_path):
    guitar = tmp_path / "guitar.md"
    guitar.write_text("tune the guitar string", encoding="utf-8")
    other = tmp_path / "dinner.md"
    other.write_text("guitar string dinner", encoding="utf-8")
    s = _store(tmp_path)
    s.ingest("guitar", guitar)
    s.ingest("dinner", other)
    assert {p.corpus for p in s.search("guitar", max_distance=9)} == {"guitar", "dinner"}
    assert {p.corpus for p in s.search("guitar", max_distance=9, corpora=["guitar"])} \
        == {"guitar"}
    # No subject named means nothing searched at all — not "search everything".
    assert s.search("guitar", max_distance=9, corpora=[]) == []


def test_corpus_keywords_are_the_documents_recurring_distinctive_words():
    from jesse.memory.corpus import corpus_keywords
    chunks = [
        "Tune the guitar string by the fifth fret. A string settles after tuning.",
        "Change strings when dull. The fret edges poke out when the guitar is dry.",
    ]
    words = corpus_keywords(chunks)
    assert "string" in words          # recurring, distinctive ("strings" folds in)
    assert "fret" in words
    assert "the" not in words         # common English never triggers
    assert "settles" not in words     # used once = incidental, not the subject
    assert "dry" not in words         # under four letters


def test_keywords_persist_and_survive_forget(tmp_path):
    doc = tmp_path / "g.md"
    doc.write_text("guitar string tuning\n\nstring tuning fret fret", encoding="utf-8")
    s = _store(tmp_path)
    s.ingest("guitar", doc, chunk_chars=25)
    assert s.keyword_subjects("my string snapped") == ["guitar"]
    s.forget("guitar")
    assert s.keyword_subjects("my string snapped") == []


def test_keywords_are_backfilled_for_a_pre_keyword_db(tmp_path):
    """The live db was ingested before trigger words existed."""
    doc = tmp_path / "g.md"
    doc.write_text("string tuning\n\nstring tuning", encoding="utf-8")
    s = _store(tmp_path)
    s.ingest("guitar", doc, chunk_chars=15)
    s._conn.execute("DELETE FROM corpus_keywords")
    s._conn.commit()
    s.close()
    s2 = _store(tmp_path)
    assert s2.keyword_subjects("the tuning is off") == ["guitar"]


def test_keyword_match_folds_plurals_and_respects_word_boundaries(tmp_path):
    doc = tmp_path / "g.md"
    doc.write_text("string tuning\n\nstring tuning", encoding="utf-8")
    s = _store(tmp_path)
    s.ingest("guitar", doc, chunk_chars=15)
    assert s.keyword_subjects("my strings snapped") == ["guitar"]   # strings -> string
    assert s.keyword_subjects("he strung me along") == []           # not a variant
    assert s.keyword_subjects("restring it") == []                  # substring, no fire


def test_short_affirmation_shapes():
    """Only these attach the previous turn to the retrieval query. "no, my car" must
    NOT — that attachment is how a declined ask chased him with the old phrase."""
    from jesse.orchestrator import _is_short_affirmation
    assert _is_short_affirmation("yes")
    assert _is_short_affirmation("Yeah, that.")
    assert _is_short_affirmation("okay sure")
    assert not _is_short_affirmation("no, my car")
    assert not _is_short_affirmation("anyway, how was your day")
    assert not _is_short_affirmation("yes but let me tell you about my whole morning first")
    assert not _is_short_affirmation("")


def test_an_unrelated_question_retrieves_nothing(tmp_path):
    """The gate IS the trigger — there is no keyword parser in front of this, so a
    loose gate is the difference between a useful passage and a document barging into
    small talk."""
    doc = tmp_path / "guitar.md"
    doc.write_text("tune the guitar string", encoding="utf-8")
    s = _store(tmp_path)
    s.ingest("guitar", doc)
    assert s.search("how was work", max_distance=0.46) == []
    assert s.search("how do I tune the guitar", max_distance=0.46)


def test_search_on_an_empty_corpus_is_empty(tmp_path):
    assert _store(tmp_path).search("anything", max_distance=9) == []


# -- what he is told -------------------------------------------------------


class _P:
    def __init__(self, text, source="guitar.md"):
        self.text, self.source, self.corpus, self.distance = text, source, "guitar", 0.1


def test_no_passages_means_no_block():
    assert knowledge_context([]) is None


def test_the_block_names_the_source_and_forbids_going_beyond_it():
    block = knowledge_context([_P("Standard tuning is E A D G B E.")])
    assert "guitar.md" in block.content
    # The two clauses that measurably changed behaviour: the passages are the COMPLETE
    # extent of what he knows, and a question about the same subject that the text does
    # not answer is still a question he cannot answer.
    assert "COMPLETE extent" in block.content
    assert "even if it is about the same subject" in block.content
    assert "not yours to give" in block.content


def test_the_block_respects_the_char_budget():
    """Two passages plus persona plus history has to stay inside num_ctx."""
    block = knowledge_context([_P("a" * 900), _P("b" * 900)], char_budget=1200)
    assert "a" * 900 in block.content and "b" * 900 not in block.content
