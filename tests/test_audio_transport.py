"""Capture callbacks delayed by playback must not cross the unmute boundary."""

import asyncio

import numpy as np
import pytest

from jesse.audio.transport import LocalAudioTransport


@pytest.fixture
def microphone(monkeypatch):
    callbacks = []

    class InputStream:
        def __init__(self, **kwargs):
            callbacks.append(kwargs["callback"])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("jesse.audio.transport.validate_input_device", lambda _: None)
    monkeypatch.setattr("jesse.audio.transport.sd.RawInputStream", InputStream)
    return callbacks


@pytest.mark.parametrize("delayed", [False, True])
@pytest.mark.parametrize("speaking_again", [False, True])
def test_unmute_discards_playback_audio_even_if_callback_delivery_is_pending(
    microphone, monkeypatch, delayed, speaking_again,
):
    async def scenario():
        transport = LocalAudioTransport()
        capture = transport.capture()
        next_frame = asyncio.create_task(anext(capture))
        await asyncio.sleep(0)  # open the mocked microphone
        transport.mute_input()
        loop = asyncio.get_running_loop()
        pending = []
        with monkeypatch.context() as patch:
            patch.setattr(loop, "call_soon_threadsafe", lambda fn, *args: pending.append((fn, args)))
            microphone[0](b"echo", 2, None, None)
        if not delayed:
            for fn, args in pending:
                fn(*args)  # already in the queue when unmute flushes it
        transport.unmute_input()
        if speaking_again:
            transport.mute_input()  # an old callback must not reappear in the next reply
        if delayed:
            for fn, args in pending:
                fn(*args)  # callback was captured while muted, delivered after unmute
        microphone[0](b"user", 2, None, None)
        try:
            assert await asyncio.wait_for(next_frame, 1) == b"user"
        finally:
            await capture.aclose()

    asyncio.run(scenario())


def test_live_muted_audio_still_reaches_detectors_with_gain_and_clipping(microphone):
    async def scenario():
        transport = LocalAudioTransport(gain=2)
        capture = transport.capture()
        next_frame = asyncio.create_task(anext(capture))
        await asyncio.sleep(0)
        transport.mute_input()
        microphone[0](np.array([100, 20000, -20000], dtype=np.int16).tobytes(), 3, None, None)
        try:
            pcm = await asyncio.wait_for(next_frame, 1)
            assert np.frombuffer(pcm, dtype=np.int16).tolist() == [200, 32767, -32768]
            assert transport.muted
        finally:
            await capture.aclose()

    asyncio.run(scenario())


def test_output_failure_closes_synthesis_iterator(monkeypatch):
    closed = []
    class BrokenOutput:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
        def stop(self):
            pass
        def close(self):
            pass
        def write(self, data):
            raise OSError("output disconnected")
    def frames():
        try:
            yield b"\x00\x00"
            yield b"\x01\x01"
        finally:
            closed.append(True)
    monkeypatch.setattr("jesse.audio.transport.sd.RawOutputStream", BrokenOutput)
    async def scenario():
        with pytest.raises(OSError, match="disconnected"):
            await LocalAudioTransport().play(frames())
        assert closed == [True]
    asyncio.run(scenario())
