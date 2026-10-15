"""Real subprocess lifecycle tests with a tiny worker, no model or GPU needed."""

import asyncio
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from pinkman.tts.omnivoice import OmniVoiceSynthesizer


@pytest.fixture
def voice(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").touch()
    prompt = tmp_path / "prompt.pt"
    prompt.touch()
    profile = tmp_path / "voice.json"
    profile.write_text(json.dumps({"python": sys.executable, "model": "model",
                                   "prompt": "prompt.pt", "steps": 32, "mode": "cuda-fp32",
                                   "keep_pauses": True, "seed": 42}))
    worker = tmp_path / "worker.py"
    worker.write_text('''import json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.getcwd())
from pinkman.tts.omnivoice_worker import GenerationCancelled, serve_requests
print(json.dumps({"ready": True, "sample_rate": 24000}), flush=True)
def render(text, cancelled):
    Path(__file__).with_suffix(".started").write_text(text)
    if text == "hang":
        time.sleep(60)
    if text == "slow":
        cancelled.wait(60)
        raise GenerationCancelled
    if text == "crash":
        sys.exit(1)
    if text == "error":
        print(json.dumps({"error": "CUDA out of memory"}), flush=True)
        sys.exit(1)
    return text.encode() * 2
serve_requests(render, sys.stdin, lambda reply: print(json.dumps(reply), flush=True))
''')
    popen = subprocess.Popen
    def launch(command, **kwargs):
        if "pinkman.tts.omnivoice_worker" not in command:
            return popen(command, **kwargs)
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert "pinkman.tts.omnivoice_worker" in command
        assert command[command.index("--mode") + 1] == "cuda-fp32"
        assert command[command.index("--steps") + 1] == "32"
        assert "--keep-pauses" in command
        assert command[command.index("--seed") + 1] == "42"
        return popen([sys.executable, "-u", str(worker)], **kwargs)
    monkeypatch.setattr("pinkman.tts.omnivoice.subprocess.Popen", launch)
    synth = OmniVoiceSynthesizer(profile)
    synth.started = worker.with_suffix(".started")
    yield synth
    synth.close()


def test_prompt_worker_reused_and_closed(voice):
    voice.start()
    worker = voice._process
    assert b"".join(voice.synthesize("one")) == b"oneone"
    assert b"".join(voice.synthesize("two")) == b"twotwo"
    assert voice._process is worker
    voice.close()
    assert worker.poll() is not None
    with pytest.raises(RuntimeError, match="closed"):
        list(voice.synthesize("three"))


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "Worker did not reach expected state"
        time.sleep(0.01)


def test_interrupt_before_first_audio_keeps_worker_and_next_turn_is_fresh(voice):
    voice.start()
    worker = voice._process
    interrupt = threading.Event()
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(list, voice.synthesize_interruptible("slow", interrupt))
        wait_until(voice.started.exists)
        interrupt.set()
        assert pending.result(timeout=2) == []
    assert b"".join(voice.synthesize("new")) == b"newnew"
    assert voice._process is worker and worker.poll() is None


def test_uncooperative_cancel_reaps_worker_without_another_turn(voice):
    voice.start()
    voice.cancel_timeout = 0.15
    worker = voice._process
    interrupt = threading.Event()
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(list, voice.synthesize_interruptible("hang", interrupt))
        wait_until(voice.started.exists)
        interrupt.set()
        assert pending.result(timeout=1) == []
        wait_until(lambda: worker.poll() is not None)
    assert b"".join(voice.synthesize("new")) == b"newnew"
    assert voice._process is not worker


def test_cancel_after_reply_is_queued_discards_old_audio(voice):
    voice.start()
    worker = voice._process
    interrupt = threading.Event()
    # Force the terminal audio response to arrive before the caller sees cancel.
    voice._send({"id": 1, "text": "old"})
    voice._request_id = 1
    wait_until(lambda: not voice._responses.empty())
    interrupt.set()
    with voice._lock:
        assert voice._read(interrupt, 1) is None
    assert b"".join(voice.synthesize("new")) == b"newnew"
    assert voice._process is worker


def test_wrong_response_id_fails_closed(voice):
    voice.start()
    worker = voice._process
    voice._responses.put({"id": -1, "pcm": "b2xkb2xk"})
    with pytest.raises(RuntimeError, match="request ID"):
        list(voice.synthesize("new"))
    assert worker.poll() is not None


def test_close_during_cancel_reaps_worker(voice):
    voice.start()
    worker = voice._process
    interrupt = threading.Event()
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(list, voice.synthesize_interruptible("hang", interrupt))
        wait_until(voice.started.exists)
        interrupt.set()
        assert pending.result(timeout=1) == []
        voice.close()
    assert worker.poll() is not None
    assert voice._process is None


@pytest.mark.parametrize("text,match", [("crash", "exited"), ("error", "out of memory"),
                                        ("hang", "timed out")])
def test_failed_request_reaps_worker_and_recovers(voice, text, match):
    voice.start()
    worker = voice._process
    voice.timeout = 0.2
    with pytest.raises(RuntimeError, match=match):
        list(voice.synthesize(text))
    assert worker.poll() is not None
    voice.timeout = 5
    assert b"".join(voice.synthesize("fresh")) == b"freshfresh"


@pytest.mark.parametrize("route", ["local", "remote"])
@pytest.mark.parametrize("cancel_task", [False, True])
def test_microphone_loop_can_interrupt_slow_generation(voice, monkeypatch, route, cancel_task):
    from pinkman.audio.transport import LocalAudioTransport
    from pinkman.remote.transport import RemoteSource, SwitchingTransport
    from tests.test_orchestrator import STOP, _build

    writes = []
    class Output:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
        def stop(self):
            pass
        def close(self):
            pass
        def write(self, chunk):
            writes.append(chunk)
    monkeypatch.setattr("pinkman.audio.transport.sd.RawOutputStream", Output)

    async def scenario():
        voice.start()
        worker = voice._process
        orch, _ = _build([])
        orch.synth = voice
        orch.transport = LocalAudioTransport()
        remote = RemoteSource()
        if route == "remote":
            orch.transport = SwitchingTransport(orch.transport, remote)
            orch.transport._remote_live = True
        orch.llm.chat = lambda *args, **kwargs: iter(["slow"])
        speaking = asyncio.create_task(orch._think_and_speak([]))
        try:
            # This timer cannot run if next(synthesis) blocks microphone ingestion.
            await asyncio.sleep(0.1)
            if cancel_task:
                speaking.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(speaking, 2)
            else:
                await orch._handle_frame(STOP)
                assert await asyncio.wait_for(speaking, 2) == ""
            assert worker.poll() is None
            assert writes == []
            assert remote.take_reply() is None
            assert await asyncio.to_thread(lambda: b"".join(voice.synthesize("new"))) == b"newnew"
            assert voice._process is worker
        finally:
            voice.close()
    asyncio.run(scenario())


def test_session_closes_voice_on_capture_error(voice):
    from tests.test_orchestrator import _build
    orch, _ = _build([])
    orch.synth = voice
    async def broken_capture():
        raise RuntimeError("microphone disconnected")
        yield
    orch.transport.capture = broken_capture
    with pytest.raises(RuntimeError, match="microphone"):
        asyncio.run(orch.run())
    assert voice._closed
    assert voice._process is None
