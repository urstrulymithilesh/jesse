"""Unit tests for the preemption decision logic.

This is the point of the design: the concurrency CORE is a pure function, so the
hardest, scariest part of the system is 100% testable with zero mocks, zero
hardware, zero models. These tests run today.
"""

from jesse.core.state import AlertDisposition, ConversationState, disposition_for


def test_alert_while_user_speaking_waits_for_silence():
    # Concept rule: don't cut the user off mid-word.
    assert disposition_for(ConversationState.LISTENING) is AlertDisposition.WAIT_FOR_SILENCE


def test_alert_while_thinking_queues_until_reply_done():
    assert disposition_for(ConversationState.THINKING) is AlertDisposition.QUEUE_AFTER_REPLY


def test_alert_while_idle_speaks_now():
    assert disposition_for(ConversationState.IDLE) is AlertDisposition.SPEAK_NOW


def test_alert_while_speaking_speaks_now():
    # Jesse already owns the speaker; a fired timer can interject immediately.
    assert disposition_for(ConversationState.SPEAKING) is AlertDisposition.SPEAK_NOW


def test_every_state_has_a_disposition():
    for state in ConversationState:
        assert isinstance(disposition_for(state), AlertDisposition)
