"""Voice-shape tests for clean_for_speech.

These check PACING and FORMAT only — that whatever Jesse says comes out as a natural
spoken line, not a wall of text or a bulleted list. They deliberately do NOT assert
any particular wording, so the persona stays free to say what it wants.
"""

from jesse.tts.speech_text import clean_for_speech


def test_plain_sentence_is_untouched():
    s = "Hey, how's your day going so far?"
    assert clean_for_speech(s) == s


def test_bulleted_list_becomes_one_spoken_line():
    out = clean_for_speech("Here's the plan:\n- boil water\n- add pasta\n- wait ten minutes")
    assert "\n" not in out                       # no line breaks in speech
    assert not out.split()[0].startswith("-")    # no leading bullet marker
    for word in ("boil water", "add pasta", "wait ten minutes"):
        assert word in out                       # content preserved, not dropped


def test_numbered_steps_flatten():
    out = clean_for_speech("1. Open it\n2. Press go")
    assert not out.startswith("1.")
    assert "\n" not in out
    assert "Open it" in out and "Press go" in out


def test_markdown_emphasis_and_headings_stripped():
    out = clean_for_speech("## Morning\nThat sounds **wonderful** and _relaxing_")
    assert "#" not in out and "*" not in out and "_" not in out
    assert "wonderful" in out and "relaxing" in out


def test_paragraphs_collapse_to_a_single_line():
    out = clean_for_speech("First thought.\n\nSecond thought.")
    assert out == "First thought. Second thought."


def test_stage_direction_asterisks_do_not_get_spoken_as_symbols():
    out = clean_for_speech("*smiles* I'm glad you're back")
    assert "*" not in out
    assert "glad you're back" in out


def test_emoji_are_removed():
    out = clean_for_speech("That's great 😊 really")
    assert "😊" not in out
    assert "great" in out and "really" in out


def test_empty_stays_empty():
    assert clean_for_speech("") == ""
    assert clean_for_speech("   \n  ") == ""
