"""Optional local OmniVoice worker, isolated from Pinkman's dependency environment."""

from __future__ import annotations

import base64
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class OmniVoiceSynthesizer:
    sample_rate = 24000

    def __init__(self, profile: str | Path):
        profile = Path(profile).resolve()
        config = json.loads(profile.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise TypeError("OmniVoice profile must be a JSON object")
        self.paths = {}
        for name in ("python", "model", "prompt"):
            value = config.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"OmniVoice profile requires {name}")
            path = (profile.parent / value).resolve()
            if not (path.is_dir() if name == "model" else path.is_file()):
                raise ValueError(f"OmniVoice {name} does not exist: {path}")
            self.paths[name] = path
        if not (self.paths["model"] / "model.safetensors").is_file():
            raise ValueError("OmniVoice model.safetensors is missing")
        self.steps = config.get("steps", 16)
        if type(self.steps) is not int or not 1 <= self.steps <= 64:
            raise ValueError("OmniVoice steps must be an integer between 1 and 64")
        self.mode = config.get("mode", "cuda-fp16")
        if self.mode not in ("cuda-fp16", "cuda-fp32", "cpu"):
            raise ValueError("OmniVoice mode must be cuda-fp16, cuda-fp32 or cpu")
        self.keep_pauses = config.get("keep_pauses", False)
        if type(self.keep_pauses) is not bool:
            raise ValueError("OmniVoice keep_pauses must be a boolean")
        self.seed = config.get("seed")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**32):
            raise ValueError("OmniVoice seed must be an integer between 0 and 4294967295")
        self.noise = None
        self.ffmpeg = None
        if "noise_reference" in config:
            from pinkman.tts.cleanup import load_noise_reference
            value = config["noise_reference"]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("OmniVoice noise_reference must name a local WAV")
            self.noise = load_noise_reference((profile.parent / value).resolve())
            self.ffmpeg = shutil.which("ffmpeg")
            if self.ffmpeg is None:
                raise ValueError("This voice requires FFmpeg for its approved noise cleanup")
        self.timeout = 180.0
        self._lock = threading.Lock()
        self._lifecycle = threading.Lock()
        self._process = None
        self._responses = None
        self._closed = False
        self._request_id = 0
        self._pending_cancel = None
        self.cancel_timeout = 2.0

    def _stop(self):
        with self._lifecycle:
            proc, self._process = self._process, None
            self._pending_cancel = None
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            proc.stdin.close()
            # stdout belongs to the reader, which closes it after EOF.

    def close(self):
        with self._lifecycle:
            self._closed = True
        self._stop()

    def _send(self, request):
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()

    def _read(self, interrupt, request_id=None):
        deadline = time.monotonic() + self.timeout
        while True:
            if self._closed:
                self._stop()
                return None
            if interrupt.is_set():
                if request_id is None:  # loading has no safe cancellation point
                    self._stop()
                else:
                    self._send({"id": request_id, "cancel": True})
                    self._pending_cancel = (request_id, time.monotonic() + self.cancel_timeout)
                    threading.Thread(target=self._finish_cancel, args=(request_id,),
                                     daemon=True, name="omnivoice-cancel").start()
                return None
            if time.monotonic() >= deadline:
                self._stop()
                raise RuntimeError("OmniVoice timed out; its worker was stopped")
            try:
                reply = self._responses.get(timeout=0.05)
            except queue.Empty:
                continue
            if reply is None:
                self._stop()
                raise RuntimeError("OmniVoice worker exited unexpectedly")
            if "error" in reply:
                self._stop()
                raise RuntimeError(f"OmniVoice: {reply['error']}")
            if request_id is not None and reply.get("id") != request_id:
                self._stop()
                raise RuntimeError("OmniVoice returned an unexpected request ID")
            return reply

    def _finish_cancel(self, request_id):
        # Release the audio caller immediately; reap a stalled worker even if the
        # user never speaks again. The generation lock also serializes next use.
        with self._lock:
            pending = self._pending_cancel
            if pending is not None and pending[0] == request_id:
                self._drain_cancelled(threading.Event())

    def _drain_cancelled(self, interrupt):
        """Discard the old terminal reply before reuse, or reap a stuck worker."""
        pending = self._pending_cancel
        if pending is None:  # close() may have already reaped it
            return not self._closed
        request_id, deadline = pending
        while not self._closed and not interrupt.is_set():
            try:
                reply = self._responses.get_nowait()
            except queue.Empty:
                if time.monotonic() >= deadline:
                    self._stop()
                    return True
                interrupt.wait(0.05)
                continue
            if (not isinstance(reply, dict) or reply.get("id") != request_id
                    or "error" in reply
                    or not (reply.get("cancelled") is True or "pcm" in reply)):
                self._stop()
            else:
                self._pending_cancel = None
            return True
        return False

    def _start(self, interrupt):
        if self._pending_cancel is not None and not self._drain_cancelled(interrupt):
            return False
        with self._lifecycle:
            if self._closed:
                raise RuntimeError("OmniVoice is closed")
            if self._process is not None:
                return True
            env = os.environ.copy()
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
            command = [str(self.paths["python"]), "-u", "-m", "pinkman.tts.omnivoice_worker",
                       "--model", str(self.paths["model"]), "--prompt", str(self.paths["prompt"]),
                       "--mode", self.mode, "--steps", str(self.steps)]
            if self.keep_pauses:
                command.append("--keep-pauses")
            if self.seed is not None:
                command.extend(["--seed", str(self.seed)])
            self._process = proc = subprocess.Popen(
                command, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            responses = self._responses = queue.Queue()

        def read():
            try:
                for line in proc.stdout:
                    responses.put(json.loads(line))
            except (ValueError, OSError):
                responses.put({"error": "Invalid response from voice worker"})
            finally:
                proc.stdout.close()
                responses.put(None)

        threading.Thread(target=read, daemon=True, name="omnivoice-output").start()
        ready = self._read(interrupt)
        if ready is None:
            return False
        if ready.get("sample_rate") != self.sample_rate or not ready.get("ready"):
            self._stop()
            raise RuntimeError("OmniVoice worker returned an incompatible audio format")
        return True

    def start(self):
        with self._lock:
            self._start(threading.Event())

    def synthesize(self, text):
        return self.synthesize_interruptible(text, threading.Event())

    def synthesize_interruptible(self, text, interrupt):
        with self._lock:
            if interrupt.is_set() or not text.strip():
                return
            try:
                if not self._start(interrupt):
                    return
                self._request_id += 1
                self._send({"id": self._request_id, "text": text})
                reply = self._read(interrupt, self._request_id)
                if reply is None or interrupt.is_set():
                    return
                pcm = base64.b64decode(reply["pcm"], validate=True)
                if not pcm or len(pcm) % 2:
                    raise ValueError("Invalid PCM from OmniVoice")
                if self.noise is not None:
                    from pinkman.tts.cleanup import clean_pcm
                    pcm = clean_pcm(pcm, self.noise, self.ffmpeg,
                                    cancelled=lambda: self._closed or interrupt.is_set())
                for offset in range(0, len(pcm), 4096):
                    if interrupt.is_set() or self._closed:
                        return
                    yield pcm[offset:offset + 4096]
            except Exception:
                self._stop()
                raise
