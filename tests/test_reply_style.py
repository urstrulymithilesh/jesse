"""Tests for trim_reflexive_question — trims a reflexive trailing question that
follows a real statement, while protecting genuine questions."""

from jesse.reply_style import trim_reflexive_question as trim


def test_statement_plus_trailing_question_is_trimmed_when_rate_zero():
    out = trim("Oh, pasta. That's a satisfying meal. How did your day go?", keep_rate=0.0)
    assert out == "Oh, pasta. That's a satisfying meal."


def test_trailing_question_kept_when_rate_one():
    text = "Oh, pasta. That's satisfying. How did your day go?"
    assert trim(text, keep_rate=1.0) == text


def test_multiple_trailing_questions_all_trimmed():
    out = trim("That sounds tough. How does it feel? Do you need advice?", keep_rate=0.0)
    assert out == "That sounds tough."


def test_reply_that_is_only_a_question_is_left_alone():
    # genuine question / recall check — never gut it
    q = "What's your sister's name?"
    assert trim(q, keep_rate=0.0) == q


def test_all_question_reply_left_alone():
    text = "Really? What happened?"
    assert trim(text, keep_rate=0.0) == text  # no statement to fall back on


def test_statement_only_reply_untouched():
    text = "That's a good feeling. You've earned a break tonight."
    assert trim(text, keep_rate=0.0) == text


def test_midtext_question_not_at_end_is_untouched():
    text = "Wait, is that right? Yeah, that's wild."  # ends on a statement
    assert trim(text, keep_rate=0.0) == text
