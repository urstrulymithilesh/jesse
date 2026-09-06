"""Endpoint-logic tests for EnergyVad — the capture-splitting fixes.

These pin the two guards that stop sentences from being cut apart:
  * a brief blip must NOT end a turn (min-speech guard),
  * a natural mid-sentence pause shorter than the silence window must be tolerated.
"""

import array

from jesse.audio.vad import EnergyVad


def _frame(value: int, n: int = 160) -> bytes:
    return array.array("h", [value] * n).tobytes()


LOUD = _frame(3000)   # RMS 3000, well above threshold
QUIET = _frame(0)     # silence


def test_brief_blip_then_silence_never_endpoints():
    # min_speech_ms=320 -> needs 4 speech frames; one blip is not enough, ever.
    v = EnergyVad(threshold=500, silence_ms=240, min_speech_ms=320)
    assert not v.is_endpoint(LOUD)
    for _ in range(20):
        assert not v.is_endpoint(QUIET)


def test_endpoints_after_enough_speech_then_enough_silence():
    v = EnergyVad(threshold=500, silence_ms=160, min_speech_ms=160)  # 2 speech, 2 silence
    for _ in range(3):
        assert not v.is_endpoint(LOUD)
    assert not v.is_endpoint(QUIET)   # 1st trailing silence
    assert v.is_endpoint(QUIET)       # 2nd -> endpoint


def test_natural_pause_shorter_than_window_is_tolerated():
    v = EnergyVad(threshold=500, silence_ms=400, min_speech_ms=160)  # silence window = 5 frames
    for _ in range(3):
        v.is_endpoint(LOUD)                       # enough speech
    for _ in range(4):
        assert not v.is_endpoint(QUIET)           # a 4-frame pause (< 5) does NOT end the turn
    assert not v.is_endpoint(LOUD)                # speech resumes -> silence counter resets
    for _ in range(4):
        assert not v.is_endpoint(QUIET)           # another sub-window pause, still fine


def test_reset_clears_speech_and_silence_counters():
    v = EnergyVad(threshold=500, silence_ms=160, min_speech_ms=160)
    for _ in range(3):
        v.is_endpoint(LOUD)
    v.reset()
    # after reset a single blip is again insufficient to endpoint
    assert not v.is_endpoint(LOUD)
    assert not v.is_endpoint(QUIET)
