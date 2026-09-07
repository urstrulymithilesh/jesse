"""Absolute honesty about the present world, and talking TO him rather than about him.

Asked the time he used to answer "about 3:47 PM" at 09:51, and "it's a Wednesday" on a
Sunday. Asked the weather: "Grey and pouring." A model with no clock and no window does
not refuse — it guesses — so the fix is to give him the clock and forbid the rest.
"""

from datetime import datetime

from jesse.context import now_context
from jesse.persona import SYSTEM_PROMPT
from jesse.tts.speech_text import clean_for_speech

WHEN = datetime(2026, 8, 23, 9, 53)


# -- the clock he now has --------------------------------------------------


def test_the_real_date_and_time_are_supplied():
    content = now_context(now=WHEN).content
    assert "Sunday" in content and "23 August 2026" in content and "09:53" in content


def test_she_is_told_it_is_the_only_thing_she_can_perceive():
    content = now_context(now=WHEN).content.lower()
    for blind in ("weather", "news", "location"):
        assert blind in content, blind
    assert "never state a different one" in content


def test_the_clock_moves():
    a = now_context(now=datetime(2026, 8, 23, 9, 0)).content
    b = now_context(now=datetime(2026, 8, 23, 17, 30)).content
    assert "09:00" in a and "17:30" in b


# -- the persona rules that back it up --------------------------------------


def test_the_honesty_rule_is_stated_in_the_persona():
    """Checks the substance, not one phrasing — the rule gets reworded, and a test that
    pins wording just breaks on every improvement while proving nothing."""
    low = SYSTEM_PROMPT.lower()
    # It names the three things he may draw on, and forbids filling gaps.
    assert "you know exactly three things" in low
    assert "never make things up" in low
    assert "invented answer" in low
    # And it explicitly covers each leak class that has actually happened.
    for leak in ("shared history", "preferences", "weather"):
        assert leak in low, f"the no-fabrication rule does not cover {leak}"


def test_her_tastes_no_longer_mention_weather():
    """"You love rain and grey afternoons" came back out as "Grey and pouring" when he
    was asked what it was like outside. A taste that names a perceivable world-state is
    a false claim waiting to happen."""
    low = SYSTEM_PROMPT.lower()
    for leak in ("love rain", "grey afternoon", "gray afternoon"):
        assert leak not in low, leak


def test_the_persona_does_not_hand_her_a_quotable_refusal():
    """A capitalised instruction came back verbatim: he literally said
    "I CANNOT KNOW IT." Directives must describe behaviour, not supply a line."""
    assert "SAY YOU CANNOT KNOW IT" not in SYSTEM_PROMPT


def test_no_rule_quotes_the_bad_output_it_forbids():
    """Twice now a rule has caused the thing it was written to prevent.

    First: an anti-tic rule quoted "you're the one who made me, Mithilesh" as the
    behaviour to avoid, and the model copied the example (0/5 -> 2/5).

    Then, writing Jesse's persona, a rule against empty reactions quoted one — and
    "Burnt rice, dude" came back as a reply to an unrelated turn about work. Caught by
    probing, in a file whose own docstring warns against exactly this.

    So: no rule may contain a quoted reply. Describe the shape, never supply the line.
    """
    import re

    # Quoted fragments that appear OUTSIDE the labelled few-shot block, where quoting
    # is the whole point. Everything before that marker is rules.
    rules = SYSTEM_PROMPT.split("Here is the register.")[0]
    quoted = re.findall(r'"([^"\n]{6,})"', rules)
    # Banned PHRASES are legitimate quotes — they are what he must not say, and the
    # measured risk is the opposite one: a quoted example of a bad REPLY he then makes.
    # Phrases he must NOT SAY are legitimately quoted — that list is the opposite risk,
    # and it has to name them exactly to be useful. Found by the paragraph that forbids
    # them rather than by an exact heading, so rewording the heading does not silently
    # turn the whole banned list into offenders.
    banned_section = ""
    for para in rules.split("\n\n"):
        if "never" in para.lower() and ("helpdesk" in para.lower()
                                        or "assistant" in para.lower()):
            banned_section += para
    offenders = [q for q in quoted
                 if q not in banned_section and len(q.split()) >= 3]
    assert not offenders, f"rules quote a repeatable line: {offenders}"


# -- the spoken-line cleanup ------------------------------------------------


def test_a_leaked_speaker_label_is_stripped():
    """The few-shot exchanges are labelled "Jesse: ...", and he copied the label."""
    assert clean_for_speech("Jesse: I have no way of seeing outside") == \
        "I have no way of seeing outside"
    assert clean_for_speech("jesse - hello") == "hello"


def test_her_name_is_untouched_when_it_is_part_of_the_sentence():
    assert clean_for_speech("Jesse is the name you gave me") == "Jesse is the name you gave me"
