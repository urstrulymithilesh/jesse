import numpy as np
import pytest

from training.clone_omnivoice import listening_preview, validate_reference


def sample(seconds=6):
    t = np.arange(int(seconds * 24000)) / 24000
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_complete_short_reference_is_accepted():
    assert validate_reference(sample(), 24000, "Good to hear from you.") == 6


@pytest.mark.parametrize("seconds", [2.9, 10.1])
def test_long_recording_requires_an_excerpt(seconds):
    with pytest.raises(ValueError, match="excerpt"):
        validate_reference(sample(seconds), 24000, "A reference.")


def test_reference_needs_a_transcript():
    with pytest.raises(ValueError, match="exact words"):
        validate_reference(sample(), 24000, " ")


@pytest.mark.parametrize("kind", ["silence", "nan", "clipping"])
def test_unusable_audio_is_rejected(kind):
    audio = sample()
    if kind == "silence":
        audio[:] = 0
    elif kind == "nan":
        audio[100] = np.nan
    else:
        audio[:len(audio) // 50] = 1
    with pytest.raises(ValueError):
        validate_reference(audio, 24000, "A reference.")


def test_listening_preview_preserves_waveform_and_raw_input():
    raw = sample()
    original = raw.copy()
    preview, gain = listening_preview(raw)
    assert float(np.max(np.abs(preview))) == pytest.approx(0.85)
    np.testing.assert_allclose(preview, raw * gain)
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("audio", [np.zeros(10), np.array([np.nan]), np.array([])])
def test_silent_or_invalid_generated_audio_cannot_be_a_successful_audition(audio):
    with pytest.raises(ValueError):
        listening_preview(audio)
