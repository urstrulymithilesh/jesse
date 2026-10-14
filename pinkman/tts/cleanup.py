"""Noise cleanup calibrated from an approved take, never from new speech."""

import subprocess
import threading
import time
import wave

RATE = 24000


def load_noise_reference(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) != (
                1, 2, RATE, RATE):
            raise ValueError("Noise reference must be exactly one second of mono 24kHz PCM16")
        return wav.readframes(RATE)


def clean_pcm(pcm, noise, ffmpeg, *, cancelled=lambda: False, timeout=10):
    """Prepend known noise for calibration, then discard it and the filter delay.

    New speech may start at sample zero. No part of it is treated as a noise sample.
    Processing stays on the playback worker thread; cancellation kills/reaps FFmpeg.
    """
    if not pcm or len(pcm) % 2 or len(noise) != RATE * 2:
        raise ValueError("Cleanup requires PCM16 and a one-second noise reference")
    if cancelled():
        return b""
    prefix = (noise * 2)[:RATE * 3]  # 1.5s gives profiling time to settle
    filters = ("apad=pad_len=600,asendcmd=c='0.2 afftdn sn start;1.2 afftdn sn stop',"
               "afftdn=nr=12:nf=-40:gs=8,atrim=start_sample=36600")
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
               "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0",
               "-af", filters, "-f", "s16le", "-c:a", "pcm_s16le", "pipe:1"]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    deadline = time.monotonic() + timeout
    finished = threading.Event()
    result = []
    errors = []

    def communicate():
        # On Windows, communicate(timeout=...) can still block writing stdin.
        # Keep ALL pipe I/O off the cancellation-monitoring thread.
        try:
            result.append(proc.communicate(input=prefix + pcm))
        except Exception as exc:  # noqa: BLE001 - propagate on the caller thread
            errors.append(exc)
        finally:
            finished.set()

    io_thread = threading.Thread(target=communicate, daemon=True, name="voice-cleanup-io")
    io_thread.start()
    try:
        while not finished.wait(0.05):
            if cancelled():
                return b""
            if time.monotonic() >= deadline:
                raise RuntimeError("Voice cleanup timed out")
        if cancelled():
            return b""
        if errors:
            raise errors[0]
        output, error = result[0]
        if proc.returncode:
            raise RuntimeError("Voice cleanup failed: " + error.decode(errors="replace")[-500:])
        if len(output) != len(pcm):
            raise RuntimeError("Voice cleanup changed the audio length")
        return b"" if cancelled() else output
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        io_thread.join(timeout=5)  # killed child closes the pipe, releasing any blocked write
        if io_thread.is_alive():
            raise RuntimeError("Voice cleanup pipe did not close")
