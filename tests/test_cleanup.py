"""Wake-word prefix stripping.

Found by the live smoke harness, not by unit tests: the pre-roll puts the wake word
into the transcript, and on qwen2.5:3b that prefix makes fact extraction return []
where the same sentence without it extracts correctly. Every turn was handing the
extractor a poisoned string.
"""

import pytest

from jesse.stt.cleanup import strip_wake_prefix

WAKE = "hey_jarvis"


def test_strips_the_wake_word_whisper_actually_produces():
    assert strip_wake_prefix("Jarvis. Remember that my favorite color is turquoise.", WAKE) \
        == "Remember that my favorite color is turquoise."


def test_strips_common_whisper_renderings():
    assert strip_wake_prefix("Hey Jarvis, set a timer", WAKE) == "set a timer"
    assert strip_wake_prefix("Hey, Jarvis. What is my name?", WAKE) == "What is my name?"
    assert strip_wake_prefix("hey jarvis tell me a joke", WAKE) == "tell me a joke"


def test_leaves_the_wake_word_alone_when_it_is_the_whole_utterance():
    """He said only the wake word — stripping to empty would silently drop the turn."""
    assert strip_wake_prefix("Jarvis", WAKE) == "Jarvis"
    assert strip_wake_prefix("Hey Jarvis.", WAKE) == "Hey Jarvis."


def test_only_strips_from_the_front():
    assert strip_wake_prefix("Remind me to call Jarvis back", WAKE) \
        == "Remind me to call Jarvis back"


def test_ordinary_speech_is_untouched():
    for text in ("What is the weather", "I had a rough day", ""):
        assert strip_wake_prefix(text, WAKE) == text


def test_adapts_to_a_different_wake_model():
    assert strip_wake_prefix("Alexa, what time is it", "alexa") == "what time is it"
    assert strip_wake_prefix("Hey Mycroft, hello", "hey_mycroft") == "hello"


def test_whisper_mishearings_of_hey_before_the_wake_token_are_stripped():
    """Live smoke run: "hey jarvis" came back as "8 Jarvis", the junk survived, the
    action parser missed, and he claimed "Photoshop opens." about nothing."""
    assert strip_wake_prefix("8 Jarvis Open Photoshop", "hey_jarvis") == "Open Photoshop"
    assert strip_wake_prefix("A Jarvis, set a timer", "hey_jarvis") == "set a timer"
    assert strip_wake_prefix("They Jarvis, hello", "hey_jarvis") == "hello"


def test_digits_and_odd_filler_without_a_wake_token_are_untouched():
    assert strip_wake_prefix("8 times 8 is 64", "hey_jarvis") == "8 times 8 is 64"
    assert strip_wake_prefix("They said hi to me", "hey_jarvis") == "They said hi to me"
    assert strip_wake_prefix("A dog barked", "hey_jarvis") == "A dog barked"


@pytest.mark.parametrize("heard", [
    "Stay Jarvis Open Photoshop",
    "Meet Jarvis.  Open Photoshop.",
    "8 Jarvis Open Photoshop",
    "Grey Jarvis Open Photoshop",
])
def test_any_single_mangled_word_before_the_wake_token_is_stripped(heard):
    """The list of things whisper makes of "hey" has no end — 8 / A / They / Stay /
    Meet all seen live, each silently breaking downstream parsing. One unrecognised
    short word is allowed ahead of a GENUINE wake token, which is what keeps ordinary
    sentences safe."""
    assert strip_wake_prefix(heard, "hey_jarvis").lower().startswith("open photoshop")


@pytest.mark.parametrize("said", [
    "A dog barked",
    "They said hi to me",
    "8 times 8 is 64",
    "Stay where you are",
    "hello world",
])
def test_without_a_wake_token_nothing_is_stripped(said):
    assert strip_wake_prefix(said, "hey_jarvis") == said


def test_two_unknown_words_are_not_swallowed():
    """One mangled 'hey' is plausible; two is a sentence."""
    assert strip_wake_prefix("Tell Sarah Jarvis is here", "hey_jarvis") == \
        "Tell Sarah Jarvis is here"
