"""Jesse Phase 0 spike — hardware + plumbing, in one sitting.

Answers two questions from the design doc's Assignment:
  1. HARDWARE: is the wake -> STT -> LLM -> TTS round-trip in the ~2-5s range?
  2. PLUMBING: do the Windows install landmines actually clear on this machine?

Runs on the standard library alone. Every probe degrades gracefully: a missing
dependency is reported, not a crash. Re-run it as you install things — it gets
greener over time. A green latency number is worthless if a probe below is RED.

    python spike.py                  # probe everything installed
    python spike.py path/to/clip.wav  # also time STT on a real 16kHz mono wav

Nothing here touches the network except the local Ollama server (localhost).
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from jesse.config import CONFIG

OK, WARN, BAD, DASH = "PASS", "WARN", "FAIL", " -- "
_rows: list[tuple[str, str, str]] = []


def row(name: str, status: str, detail: str = "") -> None:
    _rows.append((name, status, detail))
    mark = {OK: "[PASS]", WARN: "[WARN]", BAD: "[FAIL]", DASH: "[ -- ]"}[status]
    print(f"  {mark}  {name:<28} {detail}")


def probe(name: str):
    """Decorator: run a probe, catch anything, report a clean FAIL."""

    def wrap(fn):
        print(f"\n> {name}")
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — a spike WANTS to catch everything
            row(name, BAD, f"{type(e).__name__}: {e}")

    return wrap


# ---------------------------------------------------------------------------


def check_python() -> None:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}"
    if v.minor in (11, 12, 13):
        # 3.13 verified: ctranslate2/onnxruntime/sqlite-vec/fastembed all have wheels.
        row("Python version", OK, detail)
    else:
        row("Python version", WARN, f"{detail} — use 3.11/3.12/3.13")


def check_sqlite_vec() -> None:
    # The whole point: stdlib sqlite3 on Windows usually can't load extensions.
    try:
        import pysqlite3 as sqlite  # type: ignore
        driver = "pysqlite3"
    except ImportError:
        import sqlite3 as sqlite  # type: ignore
        driver = f"stdlib sqlite3 {sqlite.sqlite_version}"
    try:
        import sqlite_vec  # type: ignore
    except ImportError:
        row("sqlite-vec", BAD, "sqlite_vec not installed (pip install sqlite-vec)")
        return
    conn = sqlite.connect(":memory:")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        (ver,) = conn.execute("SELECT vec_version()").fetchone()
        row("sqlite-vec load", OK, f"vec_version={ver} via {driver}")
    finally:
        conn.close()


def check_faster_whisper(wav: Path | None) -> None:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        row("faster-whisper import", BAD, "not installed (pip install faster-whisper)")
        return
    cfg = CONFIG.speech
    t0 = time.perf_counter()
    model = WhisperModel(cfg.whisper_model, device=cfg.whisper_device, compute_type=cfg.whisper_compute_type)
    row("faster-whisper load", OK, f"{cfg.whisper_model} {cfg.whisper_compute_type}/CPU in {time.perf_counter()-t0:.1f}s")
    if wav and wav.exists():
        t0 = time.perf_counter()
        segments, info = model.transcribe(str(wav))
        text = " ".join(s.text for s in segments).strip()
        dt = time.perf_counter() - t0
        rtf = dt / info.duration if info.duration else 0
        row("STT transcribe", OK if rtf < 1 else WARN,
            f"{dt:.1f}s for {info.duration:.1f}s audio (RTF {rtf:.2f}) -> {text[:40]!r}")
    else:
        row("STT transcribe", DASH, "pass a 16kHz mono wav path to time this (python spike.py clip.wav)")


def check_wakeword() -> None:
    try:
        import openwakeword  # type: ignore
        import onnxruntime  # type: ignore
    except ImportError as e:
        row("openWakeWord + onnx", BAD, f"missing: {e.name}")
        return
    row("openWakeWord + onnx", OK, f"onnxruntime {onnxruntime.__version__}")
    # The models are a RUNTIME download into site-packages, not a pip dependency —
    # so a fresh clone (or a rebuilt venv) has the package but no models, and the
    # app crashes at the first wake check. Catch that here instead.
    models = Path(openwakeword.__file__).parent / "resources" / "models"
    wanted = f"{CONFIG.wake.model}_v0.1.onnx"
    if (models / wanted).is_file():
        row("wake-word models", OK, f"{CONFIG.wake.model} present")
    else:
        row("wake-word models", BAD,
            "not downloaded — run: python -c \"import openwakeword.utils as u; "
            "u.download_models()\"")


def check_audio() -> None:
    try:
        import sounddevice as sd  # type: ignore
    except ImportError:
        row("audio devices (WASAPI)", BAD, "sounddevice not installed")
        return
    ins = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    outs = [d["name"] for d in sd.query_devices() if d["max_output_channels"] > 0]
    status = OK if (ins and outs) else WARN
    row("audio devices (WASAPI)", status, f"{len(ins)} in / {len(outs)} out; default in: {ins[0] if ins else 'none'}")


def check_embedder() -> None:
    try:
        from fastembed import TextEmbedding  # type: ignore
    except ImportError:
        row("embedder (fastembed/CPU)", BAD, "fastembed not installed")
        return
    t0 = time.perf_counter()
    emb = TextEmbedding(model_name=CONFIG.memory.embedder_model)
    vecs = list(emb.embed(["hello, this is a memory test"]))
    row("embedder (fastembed/CPU)", OK, f"dim={len(vecs[0])} in {time.perf_counter()-t0:.1f}s (CPU)")


def check_piper() -> None:
    try:
        import piper  # noqa: F401
    except ImportError:
        row("Piper (piper-tts)", BAD, "not installed (pip install piper-tts)")
        return
    from jesse.tts.piper import PiperSynthesizer
    if PiperSynthesizer.is_available():
        row("Piper (piper-tts)", OK, f"voice '{CONFIG.speech.piper_voice}' ready")
    else:
        row("Piper (piper-tts)", WARN,
            f"pkg ok, voice missing — python -m piper.download_voices "
            f"{CONFIG.speech.piper_voice} --download-dir models")


def check_ollama() -> None:
    host = CONFIG.reasoning.ollama_host
    want = CONFIG.reasoning.model
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as r:
            tags = json.load(r)
    except (urllib.error.URLError, TimeoutError) as e:
        row("Ollama server", BAD, f"not reachable at {host} ({e}) — run `ollama serve`")
        return
    models = [m["name"] for m in tags.get("models", [])]
    has = any(want.split(":")[0] in m for m in models)
    row("Ollama server", OK, f"up; models: {', '.join(models) or 'none'}")
    if not has:
        row("Ollama model", WARN, f"{want} not pulled — run `ollama pull {want}`")
        return
    # Time a tiny non-streaming generation for tok/s + first-token feel.
    body = json.dumps({
        "model": want, "prompt": "Say hello in one short sentence.",
        "stream": False, "keep_alive": CONFIG.reasoning.keep_alive,
        "options": {"num_ctx": CONFIG.reasoning.num_ctx},
    }).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:120]
        row("Ollama generate", WARN, f"HTTP {e.code}: {msg}")
        return
    wall = time.perf_counter() - t0
    eval_count = resp.get("eval_count", 0)
    eval_ns = resp.get("eval_duration", 0) or 1
    toks = eval_count / (eval_ns / 1e9)
    ttft = resp.get("prompt_eval_duration", 0) / 1e9
    row("Ollama generate", OK if toks > 8 else WARN,
        f"{toks:.1f} tok/s, ~{ttft:.2f}s prompt-eval, {wall:.1f}s wall")


# ---------------------------------------------------------------------------


def main() -> int:
    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    print("=" * 68)
    print(" Jesse Phase 0 spike — hardware + plumbing")
    print("=" * 68)

    probe("Python")(check_python)
    probe("Reasoning: Ollama")(check_ollama)
    probe("STT: faster-whisper (CPU)")(lambda: check_faster_whisper(wav))
    probe("TTS: Piper")(check_piper)
    probe("Wake word: openWakeWord")(check_wakeword)
    probe("Memory: sqlite-vec")(check_sqlite_vec)
    probe("Embeddings: fastembed (CPU)")(check_embedder)
    probe("Audio I/O: WASAPI")(check_audio)

    fails = [n for n, s, _ in _rows if s == BAD]
    warns = [n for n, s, _ in _rows if s == WARN]
    print("\n" + "=" * 68)
    if fails:
        print(f" VERDICT: {len(fails)} blocker(s) — clear these before app code:")
        for n in fails:
            print(f"   - {n}")
    elif warns:
        print(f" VERDICT: usable, {len(warns)} warning(s) to tighten:")
        for n in warns:
            print(f"   - {n}")
    else:
        print(" VERDICT: all green — Phase 0 plumbing clear. Green-light the walking skeleton.")
    print("=" * 68)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
