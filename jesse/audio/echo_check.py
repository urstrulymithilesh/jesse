"""Bounded, local speaker-to-mic measurements. No recorded PCM is retained."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from jesse.audio.frames import ms_to_chunks
from jesse.audio.vad import EnergyVad

TEST_PHRASE = "This is a short microphone echo check. The test is ending now."


@dataclass
class Measurement:
    phase: str
    frames: int = 0
    peak_rms: float = 0.0
    speech_frames: int = 0
    endpoint: bool = False

    def observe(self, pcm: bytes, vad: EnergyVad) -> None:
        self.frames += 1
        self.peak_rms = max(self.peak_rms, vad._rms(pcm))
        self.speech_frames += int(vad.is_speech(pcm))
        # Keep feeding after the first endpoint so level statistics cover the window.
        ended = vad.is_endpoint(pcm)
        self.endpoint = self.endpoint or ended


async def measure_echo(
    transport, chunks: list[bytes], sample_rate: int, *, threshold: float,
    silence_ms: int = 950, min_speech_ms: int = 300,
    ambient_s: float = 2.0, tail_s: float = 3.0, trials: int = 3,
) -> list[Measurement]:
    """Exercise the real transport's mute/play/unmute boundary, without STT or memory.

    The caller bounds the overall run. Capture errors terminate the exercise;
    playback errors and cancellation always close capture and reopen the input.
    """
    vad = EnergyVad(threshold=threshold, silence_ms=silence_ms,
                    min_speech_ms=min_speech_ms)
    measurements = []
    current = None

    def begin(phase):
        nonlocal current
        vad.reset()
        current = Measurement(phase)
        measurements.append(current)

    async def capture():
        async for pcm in transport.capture():
            if current is not None:
                current.observe(pcm, vad)
        raise RuntimeError("Microphone capture ended before the echo check completed")

    async def exercise():
        await asyncio.sleep(0.5)  # discard stream startup transients
        begin("ambient")
        await asyncio.sleep(ambient_s)
        for trial in range(1, trials + 1):
            begin(f"playback {trial}")
            transport.mute_input()
            try:
                await transport.play(iter(chunks), sample_rate=sample_rate)
            finally:
                transport.unmute_input()
            begin(f"tail {trial}")
            await asyncio.sleep(tail_s)

    reader = asyncio.create_task(capture())
    player = asyncio.create_task(exercise())
    try:
        done, _ = await asyncio.wait((reader, player), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            await task
        return measurements
    finally:
        reader.cancel()
        player.cancel()
        await asyncio.gather(reader, player, return_exceptions=True)


def report(measurements: list[Measurement], *, min_speech_ms: int = 300) -> str:
    lines = ["Phase          frames   peak RMS   speech frames   VAD endpoint"]
    for m in measurements:
        lines.append(f"{m.phase:14} {m.frames:6} {m.peak_rms:10.1f} "
                     f"{m.speech_frames:15}   {'yes' if m.endpoint else 'no'}")
    if not measurements or any(m.frames == 0 for m in measurements):
        lines.append("INCONCLUSIVE: missing microphone frames in a measurement window.")
    elif all(m.peak_rms == 0 for m in measurements):
        lines.append("INCONCLUSIVE: microphone samples were entirely zero.")
    elif any(m.endpoint for m in measurements if m.phase.startswith("tail ")):
        lines.append("RISK: post-playback audio formed a VAD turn. Check echo and room noise.")
    elif any(m.speech_frames >= ms_to_chunks(min_speech_ms)
             for m in measurements if m.phase.startswith("tail ")):
        lines.append("RISK: post-playback audio reached the speech minimum; enough later "
                     "silence can complete a false turn. Check echo and room noise.")
    else:
        lines.append("No post-playback VAD turn observed in these samples.")
    lines.append("Levels are post-gain. Keep the room quiet and confirm the phrase is audible. "
                 "This does not prove echo cancellation or phone readiness.")
    return "\n".join(lines)
