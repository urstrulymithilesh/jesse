"""Evaluate local wake models on labeled WAVs without microphone or memory access."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import wave

from jesse.audio.frames import CHUNK_SAMPLES, SAMPLE_RATE
from jesse.audio.wakeword import OpenWakeWordDetector


def read_clip(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (
            1, 2, SAMPLE_RATE, "NONE",
        ):
            raise ValueError(f"{path}: expected mono, 16-bit PCM WAV at {SAMPLE_RATE} Hz")
        pcm = wav.readframes(wav.getnframes())
    if not pcm:
        raise ValueError(f"{path}: empty recording")
    return pcm


def evaluate_clip(detector, pcm: bytes, threshold: float) -> dict:
    detector.reset()
    silence = bytes(CHUNK_SAMPLES * 2)
    for _ in range(16):  # warm the streaming feature extractor before each clip
        detector.score(silence)
    peak, events, active, last = 0.0, 0, False, -100
    padded = pcm + silence * 16
    for i, start in enumerate(range(0, len(padded), len(silence))):
        score = detector.score(padded[start:start + len(silence)].ljust(len(silence), b"\0"))
        peak = max(peak, score)
        above = score >= threshold
        if above and not active and i - last >= 13:  # count rising edges, >=1 second apart
            events += 1
            last = i
        active = above
    return {"seconds": len(pcm) / (SAMPLE_RATE * 2), "peak": peak, "events": events}


def summarize(positive: list[dict], negative: list[dict]) -> dict:
    hits = sum(row["events"] > 0 for row in positive)
    hours = sum(row["seconds"] for row in negative) / 3600
    events = sum(row["events"] for row in negative)
    recall = hits / len(positive) if positive else 0.0
    rate = events / hours if hours else None
    # Sampled release gates; these are not statistical guarantees or proof of
    # generalization. Human/environment coverage still needs manual verification.
    enough = len(positive) >= 20 and hours >= 1
    passed = enough and recall >= 0.95 and rate is not None and rate <= 0.5
    return {"positive_clips": len(positive), "detected": hits, "recall": recall,
            "negative_hours": hours, "false_activations": events,
            "false_activations_per_hour": rate, "enough_samples": enough,
            "sampled_gate_passed": passed}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True, help="WAV folder, one intended wake per clip")
    parser.add_argument("--negative", type=Path, required=True, help="WAV folder without the wake phrase")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--report", type=Path, help="write a new JSON report (never overwrites)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold) or not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    try:
        if not args.model.is_file() or args.model.suffix.lower() != ".onnx":
            raise ValueError("--model must be an existing ONNX file")
        if args.report and args.report.exists():
            raise ValueError(f"Report already exists: {args.report}")
        files = [sorted(folder.rglob("*.wav")) for folder in (args.positive, args.negative)]
        if not all(files):
            raise ValueError("Both labeled folders must contain WAV recordings")
        if set(p.resolve() for p in files[0]) & set(p.resolve() for p in files[1]):
            raise ValueError("Positive and negative recordings must be separate")
        detector = OpenWakeWordDetector(str(args.model.resolve()), threshold=args.threshold)
        groups = []
        for label, paths in zip(("positive", "negative"), files):
            rows = []
            for path in paths:
                row = evaluate_clip(detector, read_clip(path), args.threshold)
                row["file"] = str(path.resolve())
                rows.append(row)
                print(f"{label}: {path.name}: peak={row['peak']:.3f}, events={row['events']}", flush=True)
            groups.append(rows)
        summary = summarize(*groups)
        report = {"model": str(args.model.resolve()),
                  "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  "threshold": args.threshold, "summary": summary,
                  "positive": groups[0], "negative": groups[1]}
        print(json.dumps(summary, indent=2))
        print("Sampled gates require 20 positive clips, 1 hour negative audio, >=95% recall,")
        print("and <=0.5 false activations/hour. Human coverage and live echo checks remain required.")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
        return 0 if summary["sampled_gate_passed"] else 1
    except Exception as exc:
        print(f"Wake evaluation failed: {exc}")
        return 2
