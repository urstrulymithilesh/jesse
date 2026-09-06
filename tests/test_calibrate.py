"""Unit tests for the auto-calibration decision (pure math, no mic)."""

from jesse.audio.calibrate import MAX_GAIN, recommend_audio_settings


def test_quiet_laptop_mic_gets_boosted_and_threshold_between_room_and_voice():
    # This machine's real numbers: room ~40, speech ~180 (raw).
    c = recommend_audio_settings(noise_rms=40, speech_rms=180)
    assert c.ok
    assert c.gain > 1.0  # quiet mic -> boost applied
    noise_g, speech_g = 40 * c.gain, 180 * c.gain
    assert noise_g < c.threshold < speech_g  # sits between room and voice


def test_speech_not_above_noise_is_not_confident():
    c = recommend_audio_settings(noise_rms=100, speech_rms=110)  # 110 < 100*1.3
    assert not c.ok
    assert c.gain == 1.0
    assert "mic level" in c.message.lower() or "recalibrate" in c.message.lower()


def test_near_silence_speech_is_rejected():
    c = recommend_audio_settings(noise_rms=5, speech_rms=12)  # below the 20 floor
    assert not c.ok


def test_gain_is_clamped_to_max():
    c = recommend_audio_settings(noise_rms=1, speech_rms=40)  # 2500/40 = 62.5 -> clamp
    assert c.ok
    assert c.gain == MAX_GAIN


def test_loud_mic_needs_no_gain():
    c = recommend_audio_settings(noise_rms=200, speech_rms=3000)
    assert c.ok
    assert c.gain == 1.0  # already at/above target, clamp floor is 1.0
