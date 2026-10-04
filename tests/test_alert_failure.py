"""Reminder delivery must not take down the conversation on audio failure."""

import asyncio

import pytest

from jesse.core.state import ConversationState
from jesse.ui.channel import TextChannel
from tests.test_orchestrator import END, SPEECH, WAKE, _build


def test_idle_alert_failure_does_not_stop_capture_or_the_next_spoken_turn():
    orch, transport = _build([b"quiet", WAKE, SPEECH, END])
    orch.text_channel = TextChannel()
    original_play = transport.play
    attempts = []

    async def play(frames, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("output device unavailable")
        await original_play(frames, **kwargs)

    transport.play = play
    orch.notify("your timer is up")
    asyncio.run(orch.run())

    assert transport.spoken == ["I heard you say: hello world"]
    assert orch.state is ConversationState.LISTENING
    lines = [line.text for line in orch.text_channel.transcript]
    assert lines[:2] == ["your timer is up", "I couldn't play that reminder aloud."]
    assert orch._alerts == [] and len(attempts) == 2  # no retry loop


@pytest.mark.parametrize("failure", ["synthesis", "playback"])
def test_failed_queued_alert_preserves_later_alerts_and_continuous_text_input(failure):
    orch, transport = _build([])
    orch.text_channel = TextChannel()
    orch._engaged = True
    original_synthesize, original_play = orch.synth.synthesize, transport.play

    def synthesize(text):
        if failure == "synthesis" and text == "go to the gym":
            raise RuntimeError("Piper failed")
        yield from original_synthesize(text)

    async def play(frames, **kwargs):
        frames = list(frames)
        if failure == "playback" and b"".join(frames) == b"go to the gym":
            raise OSError("output device unavailable")
        await original_play(frames, **kwargs)

    orch.synth.synthesize, transport.play = synthesize, play

    async def scenario():
        orch._enter(ConversationState.THINKING)
        orch.notify("go to the gym")
        orch.notify("call Sam")
        await orch._handle_utterance("hello")
        assert orch.state is ConversationState.LISTENING
        assert orch._alerts == []
        orch.text_channel.submit("hello again")
        await orch._handle_frame(SPEECH)
        await orch._turn_task

    asyncio.run(scenario())
    assert transport.spoken == ["I heard you say: hello", "call Sam",
                                "I heard you say: hello again"]
    lines = [line.text for line in orch.text_channel.transcript]
    start = lines.index("go to the gym")
    assert lines[start:start + 3] == ["go to the gym",
                                    "I couldn't play that reminder aloud.", "call Sam"]
    assert transport.mute_calls.count("mute") == transport.mute_calls.count("unmute")
    assert orch.state is ConversationState.LISTENING


def test_successful_alert_is_visible_in_ui_without_becoming_conversation_memory():
    orch, transport = _build([b"quiet"])
    orch.text_channel = TextChannel()
    orch.notify("your timer is up")
    asyncio.run(orch.run())
    assert transport.spoken == ["your timer is up"]
    assert [line.text for line in orch.text_channel.transcript] == ["your timer is up"]
    assert orch._history == []


def test_alert_cancellation_propagates_without_reporting_audio_failure():
    orch, transport = _build([])
    orch.text_channel = TextChannel()

    async def play(frames, **kwargs):
        raise asyncio.CancelledError()

    transport.play = play
    orch.notify("your timer is up")
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(orch._drain_alerts())
    assert not any("couldn't play" in line.text for line in orch.text_channel.transcript)
    assert transport.mute_calls == ["mute", "unmute"]
