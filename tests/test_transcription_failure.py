"""A failed STT worker must not strand the live loop in THINKING."""

import asyncio
import threading

import pytest

from jesse.core.state import ConversationState
from jesse.ui.channel import TextChannel
from tests.test_orchestrator import END, SPEECH, WAKE, _build_mem


class FailOnce:
    def __init__(self):
        self.calls = 0

    def transcribe(self, pcm):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("simulated Whisper failure")
        return "hello world"


@pytest.mark.parametrize("engaged", [False, True])
@pytest.mark.parametrize("broken_playback", [False, True])
def test_transcription_failure_recovers_even_when_apology_cannot_play(engaged, broken_playback):
    orch, transport, store, extractor = _build_mem([])
    orch.transcriber = FailOnce()
    orch.text_channel = TextChannel()
    orch._engaged = engaged
    orch._enter(ConversationState.THINKING)
    if broken_playback:
        async def fail_play(*args, **kwargs):
            raise RuntimeError("simulated audio failure")
        transport.play = fail_play

    asyncio.run(orch._run_turn(SPEECH))

    expected = ConversationState.LISTENING if engaged else ConversationState.IDLE
    assert orch.state is expected
    assert orch._history == [] and store.turns == [] and extractor.calls == 0
    assert "transcribe" in orch.text_channel.transcript[-1].text.lower()
    assert transport.mute_calls == ["mute", "unmute"]
    if not broken_playback:
        assert "transcribe" in transport.spoken[-1].lower()


@pytest.mark.parametrize("typed_followup", [False, True])
def test_next_turn_and_pending_alert_work_after_transcription_failure(typed_followup):
    orch, transport, store, _ = _build_mem([])
    orch.transcriber = FailOnce()
    orch.text_channel = TextChannel()

    async def scenario():
        await orch._handle_frame(WAKE)
        orch.notify("your timer is up")
        await orch._handle_frame(SPEECH)
        await orch._handle_frame(END)
        await orch._turn_task
        assert orch.state is ConversationState.LISTENING
        assert transport.spoken[-1] == "your timer is up"
        if typed_followup:
            orch.text_channel.submit("hello world")
            await orch._handle_frame(SPEECH)
        else:
            await orch._handle_frame(SPEECH)
            await orch._handle_frame(END)
        await orch._turn_task
        await orch._extract_task

    asyncio.run(scenario())
    assert transport.spoken[-1] == "I heard you say: hello world"
    assert [m.content for m in store.turns] == ["hello world", "I heard you say: hello world"]
    assert orch.state is ConversationState.LISTENING


def test_cancelling_transcription_is_not_reported_as_a_failure():
    orch, transport, store, _ = _build_mem([])
    started, release = threading.Event(), threading.Event()

    def transcribe(pcm):
        started.set()
        assert release.wait(5)
        return "late transcript"

    orch.transcriber.transcribe = transcribe

    async def scenario():
        orch._enter(ConversationState.THINKING)
        task = asyncio.create_task(orch._run_turn(SPEECH))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

    asyncio.run(scenario())
    assert orch.state is ConversationState.IDLE
    assert transport.spoken == [] and store.turns == [] and orch._history == []
