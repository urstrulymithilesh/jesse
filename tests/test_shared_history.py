"""Anti-confabulation guard for broad "tell me about us" questions.

Measured before this existed: 5 of 5 broad questions invented a shared past — lazy
Sundays, walks, a yellow sweater, spilled coffee. None of it happened. The invented
details were his own persona tastes (rain, grey afternoons, pineapple, teasing)
replayed as autobiography, which is what made it structural rather than a model
ceiling: specific questions ground correctly because they retrieve a matching fact,
broad ones retrieve nothing and leave the model free-associating from the persona.

Inventing intimacy is worse than getting a fact wrong, so the guard is deterministic.
"""

from jesse.context import shared_history_context
from jesse.core.interfaces import Fact
from jesse.orchestrator import _asks_about_shared_history


def test_detects_broad_questions_about_the_relationship():
    for text in ("what memories do you have between us",
                 "tell me about us",
                 "what do you remember about our time together",
                 "what's your favourite memory of us",
                 "do we have any inside jokes",
                 "how did we meet",
                 "what have we been through"):
        assert _asks_about_shared_history(text), text


def test_does_not_fire_on_specific_questions_that_already_ground():
    for text in ("what is my favourite colour",
                 "what do I do for work",
                 "do you remember what I told you about my sister",
                 "what's my dog's name"):
        assert not _asks_about_shared_history(text), text


def test_does_not_fire_on_ordinary_conversation():
    for text in ("I had a rough day", "set a timer for ten minutes",
                 "I have a light car", "how are you feeling"):
        assert not _asks_about_shared_history(text), text


def test_the_anchor_lists_only_real_facts_and_forbids_inventing():
    facts = [
        Fact(text="the user's favourite colour is black", confidence=1.0, subject="colour"),
        Fact(text="the user works as a software engineer", confidence=1.0, subject="job"),
    ]
    msg = shared_history_context(facts)
    assert msg.role == "system"
    assert "favourite colour is black" in msg.content
    assert "software engineer" in msg.content
    assert "no other shared history" in msg.content.lower()
    assert "do not invent" in msg.content.lower()


def test_the_anchor_excludes_facts_about_her_own_build():
    """self / self_history describe HER, not their history together."""
    facts = [
        Fact(text="the user's favourite colour is black", confidence=1.0,
             subject="colour", origin="conversation"),
        Fact(text="Jesse runs on a local language model", confidence=1.0,
             subject="self: tech", origin="self"),
        Fact(text="Jesse used to sound robotic", confidence=1.0,
             subject="self-history: v0", origin="self_history"),
    ]
    content = shared_history_context(facts).content
    assert "favourite colour is black" in content
    assert "local language model" not in content
    assert "robotic" not in content


def test_with_nothing_stored_she_is_told_to_say_so():
    content = shared_history_context([]).content
    assert "nothing" in content.lower()
    assert "do not invent" in content.lower()


def test_the_anchor_names_the_persona_leak_explicitly():
    """The fabrications were his own tastes replayed as shared history, so the block
    has to forbid exactly that, not just 'don't lie'."""
    content = shared_history_context(
        [Fact(text="the user likes tea", confidence=1.0, subject="drink")]).content
    assert "your own tastes" in content.lower()
