"""Text UI backend: the channel, the HTTP endpoints, and the shared turn pipeline.

The page itself is not testable here, but the thing that matters is: typed input must
travel the SAME path as speech. Two paths would mean two Ishas with different memories.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request

from jesse.core.state import ConversationState
from jesse.orchestrator import Orchestrator
from jesse.ui.channel import TextChannel
from jesse.ui.server import start

from tests.test_continuous import Says
from tests.test_orchestrator import (END, SPEECH, WAKE, STOP, FakeTransport, FakeVad,
                                     FakeWake, TextSynth)
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
    c.log("jesse", "said this")
    snap = c.snapshot()
    assert [l["role"] for l in snap["lines"]] == ["you", "jesse"]
    assert snap["lines"][0]["via"] == "text"
    assert snap["lines"][1]["via"] == "voice"


def test_snapshot_since_returns_only_new_lines():
    c = TextChannel()
    c.log("you", "one")
    first = c.snapshot()
    c.log("jesse", "two")
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
    assert "<title>Jesse</title>" in page
    # The page is chat bubbles on charcoal with one accent, replacing the original
    # black-and-white terminal look. These pin the decisions rather than the palette:
    # a single accent variable, bubbles rather than lines, and motion that a reader
    # can switch off.
    assert "--accent" in page and "--bg:#0e0f12" in page
    assert "bubble" in page and "prefers-reduced-motion" in page

    req = urllib.request.Request(
        url + "/send", data=json.dumps({"text": "typed hello"}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert json.load(r)["ok"] is True
    assert c.take() == "typed hello"

    c.log("jesse", "spoken reply")
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


def test_a_typed_turn_appears_in_the_transcript_with_her_reply():
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
    assert any(l["role"] == "jesse" and l["text"] == "Typed reply." for l in lines)


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
    assert any(l["role"] == "jesse" for l in lines)


def test_typing_is_ignored_mid_turn_so_a_live_reply_is_not_cut_off():
    c = TextChannel()
    orch, _t = _orch(c)
    orch._enter(ConversationState.SPEAKING)
    c.submit("typed while he talks")
    asyncio.run(orch._handle_frame(b"quiet"))
    assert orch._turn_task is None               # not started mid-reply
    assert c.take() == "typed while he talks"   # still queued, not lost
