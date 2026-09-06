"""StubSynthesizer — a placeholder Synthesizer used until the Piper binary is set up.

It satisfies the SAME `Synthesizer` contract as the real Piper impl (text in,
streaming PCM frames out), so swapping to Piper later is a one-line factory change
— NOT a rewrite. This is the whole point of coding to the interface.

What it does: prints what Jesse "would say" so you can read the conversation, and
emits a short, quiet tone whose length is proportional to the text. The tone is
real audio in the pipeline's format, so the interruptible-playback path (stop-word
barge-in) is exercised for real — the stub streams multiple frames, and a stop-word
cuts it off just like it will cut off Piper.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

from jesse.audio.frames import CHUNK_SAMPLES, SAMPLE_RATE, ms_to_chunks


class StubSynthesizer:
    def __init__(self, *, freq: float = 330.0, ms_per_char: int = 35, min_ms: int = 400) -> None:
        self._freq = freq
        self._ms_per_char = ms_per_char
        self._min_ms = min_ms

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def synthesize(self, text: str) -> Iterator[bytes]:
        print(f'  Jesse (stub voice): "{text}"')
        total_ms = max(self._min_ms, len(text) * self._ms_per_char)
        n_chunks = ms_to_chunks(total_ms)
        phase = 0.0
        step = 2 * math.pi * self._freq / SAMPLE_RATE
        for _ in range(n_chunks):
            buf = bytearray()
            for _ in range(CHUNK_SAMPLES):
                # low amplitude so it's a soft placeholder blip, not a scream
                val = int(3000 * math.sin(phase))
                phase += step
                buf += int(val).to_bytes(2, "little", signed=True)
            yield bytes(buf)
