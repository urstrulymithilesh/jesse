"""Audio auto-calibration: pick a capture gain + VAD threshold from the real mic.

Guessing a single RMS threshold across every laptop mic is hopeless (this machine's
speech peaked ~178 where the old default demanded 500). So we measure: a moment of
ambient noise, then a test phrase, and derive both a software gain (to lift quiet
mics for better STT) and a threshold that sits between the room and the voice.

The DECISION is a pure function (`recommend_audio_settings`) so it's unit-tested
with no hardware; only the measuring step touches the mic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from jesse.audio.frames import CHUNK_SAMPLES, SAMPLE_RATE

TARGET_SPEECH_RMS = 2500.0   # post-gain speech level we aim for (safe: ~8% of int16 FS)
MAX_GAIN = 30.0


@dataclass(frozen=True)
class Calibration:
    noise_rms: float          # raw (pre-gain) ambient level
    speech_rms: float         # raw (pre-gain) speech level
    gain: float               # software multiplier to apply in the transport
    threshold: float          # VAD threshold in POST-gain RMS terms
    ok: bool
    message: str


def recommend_audio_settings(
    noise_rms: float, speech_rms: float, *, target_speech: float = TARGET_SPEECH_RMS
) -> Calibration:
    """Pure: given measured ambient + speech RMS (pre-gain), pick gain + threshold."""
    if speech_rms < noise_rms * 1.3 or speech_rms < 20:
        return Calibration(
            noise_rms, speech_rms, 1.0, 150.0, False,
            "Speech barely rose above the room noise. Raise your Windows mic level "
            "(Settings > Sound > Input) or move the mic closer, then recalibrate.",
        )
    gain = float(np.clip(target_speech / max(speech_rms, 1.0), 1.0, MAX_GAIN))
    noise_g = noise_rms * gain
    speech_g = speech_rms * gain
    # Sit the threshold 40% of the way from room to voice, but always clearly above room.
    threshold = noise_g + 0.40 * (speech_g - noise_g)
    threshold = max(threshold, noise_g * 1.5, 120.0)
    return Calibration(
        noise_rms, speech_rms, round(gain, 1), round(threshold), True,
        f"gain x{gain:.1f}, VAD threshold {round(threshold)} "
        f"(room~{noise_g:.0f}, voice~{speech_g:.0f} post-gain)",
    )


# ---------------------------------------------------------------------------
# Mic measurement (the only part that needs hardware)
# ---------------------------------------------------------------------------


def _measure_rms(device: int | None, seconds: float, *, warmup_s: float = 0.3,
                 live: bool = False) -> list[float]:
    import sounddevice as sd

    rms: list[float] = []
    n_frames = max(1, int(seconds * SAMPLE_RATE / CHUNK_SAMPLES))
    warm_frames = max(0, int(warmup_s * SAMPLE_RATE / CHUNK_SAMPLES))
    with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SAMPLES,
                        channels=1, dtype="int16", device=device) as stream:
        for _ in range(warm_frames):
            stream.read(CHUNK_SAMPLES)  # discard warmup frames (stream settling)
        for _ in range(n_frames):
            data, _over = stream.read(CHUNK_SAMPLES)
            s = np.asarray(data, dtype=np.int16).reshape(-1).astype(np.float64)
            r = float(np.sqrt(np.mean(s ** 2))) if s.size else 0.0
            rms.append(r)
            if live:
                bar = "#" * min(30, int(30 * r / 2000))
                print(f"\r     hearing: {r:6.0f} |{bar:<30}|", end="", flush=True)
    if live:
        print()
    return rms


def _countdown(prompt: str) -> None:
    for k in (3, 2, 1):
        print(f"\r  {prompt} in {k}...  ", end="", flush=True)
        time.sleep(0.8)
    print(f"\r  {prompt} -> GO! speak now:            ")


def calibrate(device: int | None, *, ambient_s: float = 1.5, speech_s: float = 3.0,
              max_attempts: int = 2) -> Calibration:
    """Interactive: measure ambient, then a test phrase (with a countdown + live
    level so you don't miss the window); retry once if it hears nothing."""
    dev = device if device is not None else "default"
    print(f"\n  Calibrating mic (device {dev}).")
    print("  Stay SILENT for a moment — measuring the room...", flush=True)
    time.sleep(0.5)
    noise = _measure_rms(device, ambient_s)
    noise_rms = float(np.percentile(noise, 90)) if noise else 0.0

    result: Calibration | None = None
    for attempt in range(1, max_attempts + 1):
        _countdown("Say a full sentence (e.g. 'good morning, how are you doing today')")
        speech = _measure_rms(device, speech_s, warmup_s=0.1, live=True)
        speech_rms = float(np.percentile(speech, 75)) if speech else 0.0
        result = recommend_audio_settings(noise_rms, speech_rms)
        print(f"  -> room~{noise_rms:.0f}, voice~{speech_rms:.0f} (raw); {result.message}")
        if result.ok or attempt == max_attempts:
            return result
        print("  I barely heard you that time — let's try once more.\n")
    assert result is not None
    return result
