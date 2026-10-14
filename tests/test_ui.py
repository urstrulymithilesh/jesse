"""Text UI backend: the channel, the HTTP endpoints, and the shared turn pipeline.

The page itself is not testable here, but the thing that matters is: typed input must
travel the SAME path as speech. Two paths would mean two Pinkmans with different memories.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

import pytest

from pinkman.core.state import ConversationState
from pinkman.orchestrator import Orchestrator
from pinkman.ui.channel import TextChannel
from pinkman.ui.server import start
from tests.test_continuous import Says
from tests.test_orchestrator import (
    END,
    SPEECH,
    STOP,
    WAKE,
    FakeTransport,
    FakeVad,
    FakeWake,
    TextSynth,
)
from tests.test_streaming import ScriptedLLM

# -- the channel ------------------------------------------------------------


def test_submitted_text_is_taken_once():
    c = TextChannel()
    c.submit("hello")
    assert c.take() == "hello"
    assert c.take() is None


def test_blank_input_is_ignored():
    c = TextChannel()
    c.submit("   ")
    c.submit("")
    assert c.take() is None


def test_transcript_records_both_sides_with_their_source():
    c = TextChannel()
    c.log("you", "typed this", via="text")
    c.log("pinkman", "said this")
    snap = c.snapshot()
    assert [l["role"] for l in snap["lines"]] == ["you", "pinkman"]
    assert snap["lines"][0]["via"] == "text"
    assert snap["lines"][1]["via"] == "voice"


def test_snapshot_since_returns_only_new_lines():
    c = TextChannel()
    c.log("you", "one")
    first = c.snapshot()
    c.log("pinkman", "two")
    later = c.snapshot(first["total"])
    assert [l["text"] for l in later["lines"]] == ["two"]


def test_speaking_flag_drives_the_animation():
    c = TextChannel()
    assert c.snapshot()["speaking"] is False
    c.set_speaking(True)
    assert c.snapshot()["speaking"] is True


# -- the HTTP surface -------------------------------------------------------


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.load(r)


def test_the_server_serves_the_page_and_round_trips_a_message():
    c = TextChannel()
    url = start(c, port=8791)

    with urllib.request.urlopen(url, timeout=5) as r:
        page = r.read().decode()
    assert "<title>Pinkman</title>" in page
    assert 'aria-label="Conversation transcript"' in page
    assert "prefers-reduced-motion" in page

    req = urllib.request.Request(
        url + "/send", data=json.dumps({"text": "typed hello"}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert json.load(r)["ok"] is True
    assert c.take() == "typed hello"

    c.log("pinkman", "spoken reply")
    assert _get(url + "/events?since=0")["lines"][0]["text"] == "spoken reply"


# -- the shared pipeline ----------------------------------------------------


def _orch(channel):
    transport = FakeTransport([])
    orch = Orchestrator(
        transport=transport, wake=FakeWake(WAKE), stopword=FakeWake(STOP),
        vad=FakeVad(), transcriber=Says(), llm=ScriptedLLM("Typed reply."),
        synthesizer=TextSynth(), text_channel=channel)
    return orch, transport


def test_typed_input_runs_a_full_turn_through_the_same_pipeline():
    c = TextChannel()
    orch, transport = _orch(c)
    c.submit("what is my name")

    async def scenario():
        await orch._handle_frame(b"quiet")     # typed input is drained on the audio loop
        assert orch._turn_task is not None
        await orch._turn_task

    asyncio.run(scenario())
    assert transport.spoken == ["Typed reply."]          # he actually replied
    assert [m.content for m in orch._history if m.role == "user"] == ["what is my name"]


def test_a_typed_turn_appears_in_the_transcript_with_his_reply():
    c = TextChannel()
    orch, _t = _orch(c)
    c.submit("hello there")

    async def scenario():
        await orch._handle_frame(b"quiet")
        await orch._turn_task

    asyncio.run(scenario())
    lines = c.snapshot()["lines"]
    assert lines[0]["role"] == "you" and lines[0]["via"] == "text"
    assert lines[0]["text"] == "hello there"
    assert any(l["role"] == "pinkman" and l["text"] == "Typed reply." for l in lines)


def test_a_spoken_turn_lands_in_the_same_transcript():
    """One unified view regardless of how the turn started."""
    c = TextChannel()
    orch, _t = _orch(c)

    async def scenario():
        await orch._handle_frame(WAKE)
        await orch._handle_frame(SPEECH)
        await orch._handle_frame(END)
        await orch._turn_task

    asyncio.run(scenario())
    lines = c.snapshot()["lines"]
    assert lines[0]["via"] == "voice", "a spoken turn must be marked as voice"
    assert any(l["role"] == "pinkman" for l in lines)


def test_typing_is_ignored_mid_turn_so_a_live_reply_is_not_cut_off():
    c = TextChannel()
    orch, _t = _orch(c)
    orch._enter(ConversationState.SPEAKING)
    c.submit("typed while he talks")
    asyncio.run(orch._handle_frame(b"quiet"))
    assert orch._turn_task is None               # not started mid-reply
    assert c.take() == "typed while he talks"   # still queued, not lost


def test_real_turn_states_reach_ui_and_shutdown_is_offline():
    c = TextChannel()
    orch, _ = _orch(c)
    visited = []
    orch._on_state_change = lambda state: visited.append(c.snapshot()["state"])

    async def scenario():
        c.submit("hello")
        await orch._handle_frame(b"quiet")
        await orch._turn_task
        assert "thinking" in visited and "speaking" in visited
        await orch.run()
        assert c.snapshot()["state"] == "offline"

    asyncio.run(scenario())


def test_profile_snapshot_is_detached_and_session_changes_on_restart():
    c = TextChannel()
    c.configure(voice="Selected voice", brain="Local model", wake="yo pinkman")
    snap = c.snapshot()
    assert snap["state"] == "starting"
    snap["profile"]["voice"] = "changed by reader"
    assert c.snapshot()["profile"]["voice"] == "Selected voice"
    assert c.snapshot()["session_id"] == snap["session_id"]
    assert TextChannel().snapshot()["session_id"] != snap["session_id"]


def test_manual_wake_starts_voice_and_discards_audio_before_the_click():
    c = TextChannel()
    orch, transport = _orch(c)
    orch._enter(ConversationState.IDLE)
    orch._preroll.append(b"private audio before wake")

    async def scenario():
        assert c.control("wake")
        await orch._handle_frame(b"quiet")
        assert orch.state is ConversationState.LISTENING
        assert b"private audio" not in orch._buffer
        assert orch._engaged
        assert not c.control("wake")
        await orch._handle_frame(SPEECH)
        await orch._handle_frame(END)
        await orch._turn_task

    asyncio.run(scenario())
    assert transport.spoken == ["Typed reply."]


@pytest.mark.parametrize("state", [ConversationState.THINKING, ConversationState.SPEAKING])
def test_manual_interrupt_reaches_existing_cancellation_path(state):
    c = TextChannel()
    orch, _ = _orch(c)
    orch._enter(state)
    assert c.control("interrupt")
    asyncio.run(orch._handle_frame(b"quiet"))
    assert orch._interrupt.is_set() and orch._barge_in
    assert c.take_control() is None


def test_manual_controls_reject_cross_origin_and_wrong_state():
    c = TextChannel()
    url = start(c, port=8792)

    def post(origin, action="wake", header=True):
        headers = {"Content-Type": "application/json", "Origin": origin}
        if header:
            headers["X-Pinkman-Control"] = "1"
        req = urllib.request.Request(url + "/control", headers=headers,
                                     data=json.dumps({"action": action}).encode())
        return urllib.request.urlopen(req, timeout=5)

    with pytest.raises(urllib.error.HTTPError) as exc:
        post(url)
    assert exc.value.code == 409
    c.set_state("idle")
    for origin, header in [("https://outside.example", True), (url, False)]:
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(origin, header=header)
        assert exc.value.code == 403
    assert c.take_control() is None
    with post(url) as response:
        assert json.load(response) == {"ok": True}
    assert c.take_control() == "wake"
