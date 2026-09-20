"""The mu-law codec that replaces the removed `audioop`.

The codec is pinned against ITSELF rather than against a remembered spec: every
mu-law code must survive a decode/encode round trip. That caught the one real
subtlety immediately -- mu-law has TWO zero codes (0x7F and 0xFF), so a naive
"all 256 round-trip exactly" assertion fails on a correct implementation.
"""

import numpy as np

from jesse.audio.frames import SAMPLE_RATE
from jesse.remote.g711 import (
    PHONE_RATE,
    phone_to_pipeline,
    pipeline_to_phone,
    pcm_to_ulaw,
    ulaw_to_pcm,
)

ALL_CODES = bytes(range(256))


def test_every_code_round_trips_except_the_redundant_zero():
    back = pcm_to_ulaw(ulaw_to_pcm(ALL_CODES))
    off = [i for i in range(256) if back[i] != i]
    assert off == [0x7F], f"unexpected round-trip losses: {off}"


def test_both_zero_codes_decode_to_zero():
    """0x7F and 0xFF are mu-law's negative and positive zero. Encoding can only
    return one of them, which is why 0x7F is the single round-trip exception."""
    decoded = np.frombuffer(ulaw_to_pcm(bytes([0x7F, 0xFF])), dtype=np.int16)
    assert list(decoded) == [0, 0]


def test_the_codec_is_idempotent_on_decoded_audio():
    """The fully general property: once audio has been through mu-law, further
    trips change nothing. This holds for all 256 codes, zeros included."""
    once = ulaw_to_pcm(ALL_CODES)
    assert ulaw_to_pcm(pcm_to_ulaw(once)) == once


def test_decoded_values_stay_in_mu_law_range():
    decoded = np.frombuffer(ulaw_to_pcm(ALL_CODES), dtype=np.int16)
    assert decoded.min() == -32124 and decoded.max() == 32124


def test_loud_input_saturates_rather_than_wrapping():
    """Clipping must clamp. A wrap turns a loud sample into a loud sample of the
    OPPOSITE sign, which is audible as a crack rather than as distortion."""
    loud = np.array([32767, -32768, 32000, -32000], dtype=np.int16).tobytes()
    out = np.frombuffer(ulaw_to_pcm(pcm_to_ulaw(loud)), dtype=np.int16)
    assert list(np.sign(out)) == [1, -1, 1, -1]
    assert np.all(np.abs(out) <= 32124)


def test_silence_survives_both_directions():
    quiet = b"\x00\x00" * 160
    assert set(pipeline_to_phone(quiet)) <= {0xFF, 0x7F}
    assert np.abs(np.frombuffer(phone_to_pipeline(b"\xff" * 160),
                                dtype=np.int16)).max() == 0


def test_rate_conversion_halves_and_doubles_the_sample_count():
    """20ms at each end: 160 mu-law bytes on the line, 320 PCM samples inside."""
    assert len(phone_to_pipeline(b"\xff" * 160)) == 320 * 2      # 16k int16
    assert len(pipeline_to_phone(b"\x00\x00" * 320)) == 160      # 8k mu-law
    assert SAMPLE_RATE == 2 * PHONE_RATE                          # the assumption above


def test_empty_payloads_do_not_crash():
    assert phone_to_pipeline(b"") == b""
    assert pipeline_to_phone(b"") == b""


def test_a_tone_keeps_its_frequency_through_the_phone_leg():
    """Guards the resampler wiring: an up/down mix-up would halve or double pitch."""
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    tone = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16).tobytes()
    out = np.frombuffer(phone_to_pipeline(pipeline_to_phone(tone)), dtype=np.int16)
    peak = np.argmax(np.abs(np.fft.rfft(out.astype(float)))) * SAMPLE_RATE / out.size
    assert 435 < peak < 445, f"440 Hz came back as {peak:.0f} Hz"
