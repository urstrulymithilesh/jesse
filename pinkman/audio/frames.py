"""Pipeline audio format — one convention, enforced everywhere.

16 kHz, mono, 16-bit signed PCM, little-endian, carried as `bytes`. openWakeWord
wants 1280-sample (80 ms) chunks at 16 kHz, so that is the frame size the whole
pipeline speaks.
"""

from __future__ import annotations

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 1280            # 80 ms — openWakeWord's expected chunk size
BYTES_PER_SAMPLE = 2           # int16
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE  # 2560
CHUNK_MS = CHUNK_SAMPLES * 1000 // SAMPLE_RATE  # 80


def ms_to_chunks(ms: int) -> int:
    """How many 80 ms frames span `ms` milliseconds (rounded up)."""
    return max(1, (ms + CHUNK_MS - 1) // CHUNK_MS)
