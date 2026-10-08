"""Compare local OmniVoice inference with a cached voice prompt; never change Jesse."""

import argparse
import json
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New benchmark directory")
    parser.add_argument("--mode", choices=("cpu", "cpu-int8", "cuda-fp16", "cuda-fp32"),
                        default="cpu")
    parser.add_argument("--steps", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--speed", type=float, help="Model duration factor (below 1 is slower)")
    parser.add_argument("--keep-pauses", action="store_true",
                        help="Disable generated-audio silence removal for a pacing comparison")
    parser.add_argument("--model", type=Path, default=ROOT / "data/omnivoice-model")
    parser.add_argument("--text", default="Hey, it's Jesse. Good to hear from you. "
                        "What are we working on today?")
    args = parser.parse_args(argv)
    if args.threads < 1 or any(not 1 <= step <= 64 for step in args.steps):
        parser.error("Use positive threads and steps between 1 and 64.")
    if len(args.steps) != len(set(args.steps)) or not args.text.strip():
        parser.error("Step counts must be unique and text nonempty.")
    if not args.prompt.is_file() or not (args.model / "model.safetensors").is_file():
        parser.error("A saved local prompt and complete local model are required.")
    generation_options = {}
    if args.speed is not None:
        if not math.isfinite(args.speed) or not 0.5 <= args.speed <= 2:
            parser.error("Audition speed must be finite and between 0.5 and 2.")
        generation_options["speed"] = args.speed
    if args.keep_pauses:
        generation_options["postprocess_output"] = False
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
                "DO_NOT_TRACK"):
        os.environ[key] = "1"
    import soundfile as sf
    import torch
    from omnivoice import OmniVoice, VoiceClonePrompt

    if __package__:
        from .clone_omnivoice import listening_preview
    else:
        from clone_omnivoice import listening_preview
    torch.set_num_threads(args.threads)
    cuda = args.mode.startswith("cuda")
    if cuda and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; no silent fallback to CPU.")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "mode": args.mode, "threads": args.threads, "torch": torch.__version__,
        "text": args.text, "prompt": str(args.prompt.resolve()), "runs": [],
        "tokenizer_device": "cpu", "user_approved_similarity": False,
        "seed": 42, "generation_options": generation_options,
    }

    def save():
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")

    try:
        started = time.perf_counter()
        print(f"Loading {args.mode}; audio tokenizer stays on CPU...", flush=True)
        model = OmniVoice.from_pretrained(str(args.model.resolve()), device_map="cpu",
                                         dtype=torch.float32, local_files_only=True,
                                         load_asr=False)
        if cuda:
            # Keep the codec off the small GPU; only the diffusion model is moved.
            tokenizer = model.audio_tokenizer
            model.audio_tokenizer = None
            model.to(device="cuda", dtype=torch.float16 if args.mode == "cuda-fp16"
                     else torch.float32)
            model.audio_tokenizer = tokenizer
            torch.cuda.synchronize()
            report["gpu"] = torch.cuda.get_device_name()
        elif args.mode == "cpu-int8":
            # Experimental dynamic quantization: only the language model's linear layers.
            # Embeddings, output head and audio codec remain float32.
            torch.ao.quantization.quantize_dynamic(
                model.llm, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        report["load_and_setup_seconds"] = round(time.perf_counter() - started, 2)
        prompt = VoiceClonePrompt.load(str(args.prompt))
        save()
        with torch.inference_mode():
            for steps in args.steps:
                torch.manual_seed(42)
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                print(f"Generating {steps} steps...", flush=True)
                started = time.perf_counter()
                audio = model.generate(text=args.text, language="en",
                                       voice_clone_prompt=prompt, num_step=steps,
                                       **generation_options)[0]
                if cuda:
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                preview, gain = listening_preview(audio)
                sf.write(args.output / f"steps-{steps}.wav", audio, model.sampling_rate,
                         subtype="PCM_16")
                sf.write(args.output / f"steps-{steps}-listen.wav", preview, model.sampling_rate,
                         subtype="PCM_16")
                duration = len(audio) / model.sampling_rate
                result = {"steps": steps, "generation_seconds": round(elapsed, 2),
                          "audio_seconds": round(duration, 2), "rtf": round(elapsed / duration, 2),
                          "preview_gain": round(gain, 4)}
                if cuda:
                    result["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
                report["runs"].append(result)
                save()
                print(json.dumps(result), flush=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
