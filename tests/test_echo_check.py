import asyncio
import struct

import pytest

from jesse.audio.echo_check import Measurement, measure_echo, report
from jesse.audio.frames import CHUNK_SAMPLES
from jesse.audio.vad import EnergyVad


def pcm(level):
    return struct.pack("<h", level) * CHUNK_SAMPLES


@pytest.mark.parametrize("speech_frames, endpoint", [(2, False), (4, True)])
def test_tail_blips_are_distinguished_from_a_complete_false_turn(speech_frames, endpoint):
    measurement = Measurement("tail 1")
    vad = EnergyVad(threshold=150, silence_ms=950, min_speech_ms=300)
    for frame in [pcm(200)] * speech_frames + [pcm(0)] * 12:
        measurement.observe(frame, vad)
    assert measurement.endpoint == endpoint
    assert measurement.speech_frames == speech_frames
    assert measurement.frames == speech_frames + 12
    assert measurement.peak_rms == 200
    assert ("RISK:" in report([measurement])) == endpoint


@pytest.mark.parametrize("measurements", [[], [Measurement("ambient")],
                                             [Measurement("ambient", frames=20)]])
def test_missing_or_all_zero_capture_is_inconclusive(measurements):
    assert "INCONCLUSIVE:" in report(measurements)
    assert "No post-playback" not in report(measurements)


def test_tail_with_enough_speech_is_risky_even_before_trailing_silence():
    measurement = Measurement("tail 1")
    vad = EnergyVad(threshold=150, min_speech_ms=300)
    for _ in range(4):
        measurement.observe(pcm(200), vad)
    assert not measurement.endpoint
    assert "enough later silence" in report([measurement])
    assert "RISK:" not in report([measurement], min_speech_ms=800)


class Transport:
    def __init__(self, *, fail_capture=False, fail_play=False):
        self.queue = asyncio.Queue()
        self.closed = False
        self.muted = False
        self.fail_capture = fail_capture
        self.fail_play = fail_play

    async def capture(self):
        try:
            if self.fail_capture:
                raise RuntimeError("mic disconnected")
            while True:
                yield await self.queue.get()
        finally:
            self.closed = True

    async def play(self, chunks, *, sample_rate):
        assert self.muted
        assert sample_rate == 22050
        if self.fail_play:
            raise RuntimeError("speaker disconnected")
        self.queue.put_nowait(pcm(1000))
        await asyncio.sleep(0)  # capture during playback
        # Tail frames arrive only after play returns and the input reopens.
        for frame in [pcm(200)] * 4 + [pcm(0)] * 12:
            asyncio.get_running_loop().call_soon(self.queue.put_nowait, frame)

    def mute_input(self):
        self.muted = True

    def unmute_input(self):
        self.muted = False


def run(transport):
    return asyncio.run(measure_echo(transport, [pcm(1000)], 22050, threshold=150,
                                   ambient_s=0.001, tail_s=0.01, trials=2))


def test_measurement_resets_vad_at_reopen_and_each_trial_and_closes_capture():
    transport = Transport()
    measurements = run(transport)
    assert [m.phase for m in measurements] == [
        "ambient", "playback 1", "tail 1", "playback 2", "tail 2",
    ]
    tails = [m for m in measurements if m.phase.startswith("tail")]
    assert all(m.peak_rms == 200 and m.endpoint and m.speech_frames == 4 for m in tails)
    assert transport.closed and not transport.muted


@pytest.mark.parametrize("failure", ["capture", "play"])
def test_device_errors_close_capture_and_reopen_input(failure):
    transport = Transport(fail_capture=failure == "capture", fail_play=failure == "play")
    with pytest.raises(RuntimeError, match="disconnected"):
        run(transport)
    assert transport.closed and not transport.muted


def test_cancellation_during_playback_closes_capture_and_unmutes():
    async def scenario():
        started = asyncio.Event()

        class BlockingTransport(Transport):
            async def play(self, chunks, *, sample_rate):
                started.set()
                await asyncio.Event().wait()

        transport = BlockingTransport()
        task = asyncio.create_task(measure_echo(transport, [], 22050, threshold=150,
                                               ambient_s=0.001))
        await asyncio.wait_for(started.wait(), 2)
        assert transport.muted
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert transport.closed and not transport.muted

    asyncio.run(scenario())


def test_echo_command_rejects_a_running_jesse_before_opening_audio(monkeypatch, capsys):
    import diagnose

    monkeypatch.setattr("jesse.core.single_instance.claim", lambda _: "Another Jesse is running")
    monkeypatch.setattr(diagnose.sd, "query_devices", lambda *args: pytest.fail("opened audio"))
    assert diagnose._echo([]) == 1
    assert "Another Jesse" in capsys.readouterr().out
