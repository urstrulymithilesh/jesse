"""Local OmniVoice audition. Run in its separate environment, not Jesse's venv."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def validate_reference(audio, rate: int, transcript: str) -> float:
    import numpy as np
    duration = len(audio) / rate
    if not transcript.strip():
        raise ValueError("Supply the exact words spoken in the reference.")
    if not 3 <= duration <= 10:
        raise ValueError(f"Select a complete 3–10 second excerpt; got {duration:.2f}s.")
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 0.003:
        raise ValueError("Reference is non-finite or almost silent.")
    if np.mean(np.abs(audio) >= 0.999) > 0.005:
        raise ValueError("Reference is clipped; use a quieter recording.")
    return duration


def listening_preview(audio):
    """Apply only constant gain; preserve the raw model output separately."""
    import numpy as np
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("Model returned empty or non-finite audio.")
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-8:
        raise ValueError("Model returned silent audio.")
    gain = 0.85 / peak
    return audio * gain, gain


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New audition directory")
    parser.add_argument("--model", type=Path, default=ROOT / "data/omnivoice-model")
    parser.add_argument("--steps", type=int, choices=range(1, 65), default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not args.text.strip() or args.threads < 1:
        parser.error("Text must be nonempty and threads positive.")
    # Model downloads are a separate setup step; inference never needs the network.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    import soundfile as sf
    audio, rate = sf.read(args.reference, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    transcript = args.transcript.read_text(encoding="utf-8").strip()
    duration = validate_reference(audio, rate, transcript)
    if not (args.model / "model.safetensors").is_file():
        parser.error("Download the OmniVoice model locally before inference.")
    if not (args.model / "audio_tokenizer/config.json").is_file():
        parser.error("The local model is missing its audio tokenizer.")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from omnivoice import OmniVoice
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    started = time.perf_counter()
    print("Loading local OmniVoice on CPU...", flush=True)
    model = OmniVoice.from_pretrained(
        str(args.model.resolve()), device_map="cpu", dtype=torch.float32,
        local_files_only=True, load_asr=False,
    )
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with torch.inference_mode():
        prompt = model.create_voice_clone_prompt(ref_audio=(audio, rate), ref_text=transcript)
        prompt.save(str(args.output / "voice-prompt.pt"))
        print("Voice prompt saved. Generating audition...", flush=True)
        output = model.generate(text=args.text, language="en",
                                voice_clone_prompt=prompt, num_step=args.steps)[0]
    elapsed = time.perf_counter() - started
    preview, preview_gain = listening_preview(output)
    sf.write(args.output / "audition.wav", output, model.sampling_rate, subtype="PCM_16")
    sf.write(args.output / "audition-listen.wav", preview, model.sampling_rate, subtype="PCM_16")
    audio_seconds = len(output) / model.sampling_rate
    report = {
        "reference": str(args.reference.resolve()), "reference_seconds": duration,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "reference_transcript": transcript, "text": args.text, "steps": args.steps,
        "device": "cpu", "threads": args.threads, "sample_rate": model.sampling_rate,
        "load_seconds": round(load_seconds, 2), "prompt_and_generation_seconds": round(elapsed, 2),
        "audio_seconds": round(audio_seconds, 2),
        "real_time_factor_including_prompt": round(elapsed / audio_seconds, 2),
        "listening_preview_gain": round(preview_gain, 4),
        "user_approved_similarity": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
