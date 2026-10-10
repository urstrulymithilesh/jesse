import shutil
import subprocess
import sys
import threading
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from jesse.tts.cleanup import RATE, clean_pcm, load_noise_reference


def test_cleanup_preserves_speech_from_first_sample_and_reduces_pause_noise():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg integration requires the optional voice cleanup dependency")
    rng = np.random.default_rng(42)
    noise = (rng.normal(0, .002, RATE) * 32768).astype("<i2").tobytes()
    audio = rng.normal(0, .002, 3 * RATE)
    audio[:RATE] += .3 * np.cos(2 * np.pi * 220 * np.arange(RATE) / RATE)
    pcm = (audio * 32768).astype("<i2").tobytes()
    cleaned = np.frombuffer(clean_pcm(pcm, noise, ffmpeg), "<i2") / 32768
    assert len(cleaned) == len(audio)
    assert np.corrcoef(cleaned[:RATE], audio[:RATE])[0, 1] > .995
    assert np.sqrt(np.mean(cleaned[2*RATE:]**2)) < np.sqrt(np.mean(audio[2*RATE:]**2)) * .4


def test_wrong_noise_format_rejected(tmp_path):
    path = tmp_path / "noise.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 16000)
    with pytest.raises(ValueError, match="one second"):
        load_noise_reference(path)


@pytest.mark.parametrize("action", ["cancel", "timeout", "failure"])
def test_cleanup_reaps_failed_or_cancelled_subprocess(monkeypatch, action):
    popen = subprocess.Popen
    processes = []
    launched = threading.Event()
    cancel = threading.Event()
    def start(*args, **kwargs):
        code = "raise SystemExit(1)" if action == "failure" else "import time; time.sleep(60)"
        proc = popen([sys.executable, "-c", code], **kwargs)
        processes.append(proc)
        launched.set()
        return proc
    monkeypatch.setattr("jesse.tts.cleanup.subprocess.Popen", start)
    with ThreadPoolExecutor() as pool:
        job = pool.submit(clean_pcm, b"\0\0" * RATE, b"\0\0" * RATE, "ffmpeg",
                          cancelled=cancel.is_set, timeout=.2 if action == "timeout" else 10)
        assert launched.wait(2)
        if action == "cancel":
            cancel.set()
            assert job.result(timeout=2) == b""
        else:
            with pytest.raises(RuntimeError, match="timed out|failed"):
                job.result(timeout=2)
    assert processes[0].poll() is not None
