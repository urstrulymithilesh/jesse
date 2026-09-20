"""G.711 mu-law <-> PCM, and the 8k/16k rate change either side of it.

Twilio Media Streams speak 8 kHz mu-law; the pipeline speaks 16 kHz signed 16-bit
PCM (jesse.audio.frames). Something has to sit between them.

That job used to belong to the standard library -- `audioop.ulaw2lin` and
`audioop.ratecv` did both halves in two calls. **`audioop` was removed in Python
3.13**, which this project runs, so it is written out here instead. It is ~40 lines
of table lookup; the alternative was the `audioop-lts` backport, i.e. a dependency
carrying the whole removed module for two functions.

The codec is the canonical Sun/ITU implementation, vectorised with numpy. The
strong correctness property is in the tests: encode(decode(u)) == u for all 256
code points, which pins both directions against each other rather than against my
memory of the spec.

Resampling uses scipy's resample_poly, already a dependency and already used by the
smoke harness for exactly this.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from jesse.audio.frames import SAMPLE_RATE

PHONE_RATE = 8_000          # what Twilio sends and expects
_BIAS = 0x84
_CLIP = 8159                # max magnitude in the codec's 14-bit domain
_SEG_ENDS = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])


def _build_decode_table() -> np.ndarray:
    """The 256 PCM values mu-law can represent. Built from the spec, once."""
    out = np.empty(256, dtype=np.int16)
    for byte in range(256):
        u = ~byte & 0xFF
        t = (((u & 0x0F) << 3) + _BIAS) << ((u & 0x70) >> 4)
        out[byte] = (_BIAS - t) if (u & 0x80) else (t - _BIAS)
    return out


_DECODE = _build_decode_table()


def ulaw_to_pcm(payload: bytes) -> bytes:
    """8 kHz mu-law bytes -> 8 kHz signed 16-bit PCM."""
    codes = np.frombuffer(payload, dtype=np.uint8)
    return _DECODE[codes].tobytes()


def pcm_to_ulaw(pcm: bytes) -> bytes:
    """8 kHz signed 16-bit PCM -> 8 kHz mu-law bytes."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.int32) >> 2  # to 14 bits
    sign = np.where(samples < 0, 0x7F, 0xFF).astype(np.uint8)           # mask, not a bit
    mag = np.minimum(np.abs(samples), _CLIP) + (_BIAS >> 2)
    seg = np.searchsorted(_SEG_ENDS, mag)                               # 0..8
    # seg 8 means the value saturated past the last segment: the codec's loudest code.
    code = np.where(
        seg >= 8, 0x7F,
        ((seg << 4) | ((mag >> (seg + 1)) & 0x0F)),
    ).astype(np.uint8)
    return (code ^ sign).astype(np.uint8).tobytes()


def _resample(pcm: bytes, up: int, down: int) -> bytes:
    if not pcm:
        return b""
    samples = np.frombuffer(pcm, dtype=np.int16)
    out = resample_poly(samples.astype(np.float64), up, down)
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()


def phone_to_pipeline(payload: bytes) -> bytes:
    """A Twilio media payload -> a 16 kHz PCM frame the pipeline can consume."""
    return _resample(ulaw_to_pcm(payload), SAMPLE_RATE, PHONE_RATE)


def pipeline_to_phone(pcm16k: bytes) -> bytes:
    """16 kHz PCM from Piper -> a mu-law payload Twilio will play down the line."""
    return pcm_to_ulaw(_resample(pcm16k, PHONE_RATE, SAMPLE_RATE))
