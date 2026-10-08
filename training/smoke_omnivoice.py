"""Headless real-worker check: reuse, interrupt before audio, then recover."""

import argparse
import json
import threading
import time
import wave
from pathlib import Path

from jesse.tts.omnivoice import OmniVoiceSynthesizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local output directory")
    args = parser.parse_args()
    synth = OmniVoiceSynthesizer(args.profile)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"steps": synth.steps, "mode": synth.mode, "runs": [],
              "user_approved_similarity": False}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def render(text, name):
        started = time.perf_counter()
        pcm = b"".join(synth.synthesize(text))
        elapsed = time.perf_counter() - started
        assert pcm, "No speech generated"
        with wave.open(str(args.output / f"{name}.wav"), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(synth.sample_rate)
            out.writeframes(pcm)
        result = {"text": text, "generation_seconds": round(elapsed, 2),
                  "audio_seconds": round(len(pcm) / 2 / synth.sample_rate, 2)}
        report["runs"].append(result)
        print(json.dumps(result), flush=True)
        save()

    try:
        started = time.perf_counter()
        synth.start()
        report["startup_seconds"] = round(time.perf_counter() - started, 2)
        worker = synth._process
        render("Hey, it's Jesse. Good to hear from you.", "first")
        render("That sounds exciting. Tell me a little more about what you have in mind.", "second")
        assert synth._process is worker, "Worker was not reused"
        report["worker_reused"] = True
        interrupt = threading.Event()
        timer = threading.Timer(0.5, interrupt.set)
        timer.start()
        started = time.perf_counter()
        try:
            pcm = b"".join(synth.synthesize_interruptible(
                "This sentence should be interrupted before any audio is returned.", interrupt))
        finally:
            timer.cancel()
        report["interrupt_total_seconds"] = round(time.perf_counter() - started, 2)
        assert not pcm and worker.poll() is not None, "Interrupted worker leaked audio or survived"
        assert report["interrupt_total_seconds"] < 2, "Interruption took too long"
        report["interruption_passed"] = True
        save()
        render("I'm back. Let's try the next thing together.", "after-interruption")
        report["recovered"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        worker = synth._process
        synth.close()
        report["closed"] = worker is None or worker.poll() is not None
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
