"""Playback audio must never become pre-roll for the next user turn."""

import asyncio

import pytest

from jesse.core.state import ConversationState
from jesse.orchestrator import Orchestrator
from tests.test_orchestrator import (
    END, SPEECH, STOP, WAKE, FakeTransport, FakeVad, FakeWake,
    RecordingTranscriber, TextSynth,
)
from tests.test_streaming import ScriptedLLM


def build():
    transport = FakeTransport([])
    recorder = RecordingTranscriber()
    orch = Orchestrator(
        transport=transport, wake=FakeWake(WAKE), stopword=FakeWake(STOP),
        vad=FakeVad(), transcriber=recorder, llm=ScriptedLLM("All right."),
        synthesizer=TextSynth(), preroll_frames=8,
    )
    return orch, transport, recorder


@pytest.mark.parametrize("streamed", [True, False])
@pytest.mark.parametrize("playback_fails", [False, True])
def test_reply_audio_does_not_reach_the_followup_transcript(streamed, playback_fails):
    orch, transport, recorder = build()
    normal_play = transport.play

    async def echoing_play(frames, *, sample_rate):
        for chunk in frames:
            await orch._handle_frame(b"JESSE_ECHO:" + chunk)
        if playback_fails:
            raise RuntimeError("speaker disconnected")

    transport.play = echoing_play

    async def scenario():
        await orch._handle_frame(WAKE)
        await orch._handle_frame(b"OLD_USER_AUDIO")
        if playback_fails:
            with pytest.raises(RuntimeError, match="speaker disconnected"):
                await speak()
        else:
            await speak()
        transport.play = normal_play
        orch._begin_listening(interrupt_background=False)
        await orch._handle_frame(SPEECH)
        await orch._handle_frame(END)
        await orch._turn_task

    async def speak():
        if streamed:
            await orch._think_and_speak([])
        else:
            await orch._speak("Your timer is done.")

    asyncio.run(scenario())
    assert recorder.last == SPEECH + END


def test_barge_in_keeps_words_after_the_stop_word_but_not_earlier_echo():
    orch, _transport, recorder = build()

    async def scenario():
        orch._enter(ConversationState.SPEAKING)
        await orch._handle_frame(b"JESSE_ECHO")
        await orch._handle_frame(STOP)
        await orch._handle_frame(b"USER_START")
        orch._finish_speaking()
        orch._barge_in = False
        orch._begin_listening()
        await orch._handle_frame(SPEECH)
        await orch._handle_frame(END)
        await orch._turn_task

    asyncio.run(scenario())
    assert recorder.last == STOP + b"USER_START" + SPEECH + END


def test_thinking_audio_is_not_reused_when_no_reply_is_spoken():
    orch, _transport, _recorder = build()

    async def scenario():
        await orch._handle_frame(WAKE)
        orch._enter(ConversationState.THINKING)
        await orch._handle_frame(b"IGNORED_WHILE_THINKING")
        orch._finish_speaking()
        orch._begin_listening(interrupt_background=False)

    asyncio.run(scenario())
    assert not orch._buffer
