"""Private JSON-lines worker; run with the separate OmniVoice Python environment."""

import argparse
import base64
import contextlib
import json
import os
import queue
import sys
import threading


class GenerationCancelled(Exception):
    """Unwind a generation between model steps without unloading its weights."""


@contextlib.contextmanager
def cancellation_guard(model, cancelled):
    def check(*_):
        if cancelled.is_set():
            raise GenerationCancelled

    check()
    # The pinned OmniVoice decoder calls model(...) once per diffusion step.
    # A hook leaves inputs, weights, RNG and completed audio unchanged.
    hook = model.register_forward_pre_hook(check)
    try:
        yield
        check()  # also discard cancellation during final CPU audio decoding
    finally:
        hook.remove()


def serve_requests(render, source, send):
    """Read cancellation concurrently; generate serially with one terminal reply/id."""
    jobs = queue.Queue()
    active = {}
    lock = threading.Lock()

    def read():
        try:
            for line in source:
                request = json.loads(line)
                request_id = request["id"]
                with lock:
                    if request.get("cancel") is True:
                        event = active.get(request_id)
                        if event is not None:
                            event.set()
                    else:
                        if request_id in active:
                            raise ValueError("Duplicate voice request ID")
                        event = active[request_id] = threading.Event()
                        jobs.put((request_id, request["text"], event))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            jobs.put(exc)
        finally:
            with lock:
                for event in active.values():
                    event.set()
            jobs.put(None)

    threading.Thread(target=read, daemon=True, name="voice-requests").start()
    while True:
        job = jobs.get()
        if job is None:
            return
        if isinstance(job, Exception):
            raise job
        request_id, text, cancelled = job
        try:
            if cancelled.is_set():
                raise GenerationCancelled
            pcm = render(text, cancelled)
            reply = {"pcm": base64.b64encode(pcm).decode("ascii")}
        except GenerationCancelled:
            reply = {"cancelled": True}
        finally:
            with lock:
                active.pop(request_id, None)
        if cancelled.is_set():
            reply = {"cancelled": True}
        send({"id": request_id, **reply})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", choices=("cuda-fp16", "cuda-fp32", "cpu"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--keep-pauses", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    output = sys.stdout

    def send(value):
        output.write(json.dumps(value) + "\n")
        output.flush()

    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
                "DO_NOT_TRACK"):
        os.environ[key] = "1"
    try:
        # Libraries may print progress; stdout is exclusively the IPC protocol.
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            import torch
            from omnivoice import OmniVoice, VoiceClonePrompt

            torch.set_num_threads(4)
            if args.mode.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; no automatic CPU fallback")
            model = OmniVoice.from_pretrained(args.model, device_map="cpu",
                dtype=torch.float32, local_files_only=True, load_asr=False)
            if args.mode.startswith("cuda"):
                tokenizer = model.audio_tokenizer
                model.audio_tokenizer = None
                model.to(device="cuda", dtype=torch.float16 if args.mode == "cuda-fp16"
                         else torch.float32)
                model.audio_tokenizer = tokenizer
            prompt = VoiceClonePrompt.load(args.prompt)
        send({"ready": True, "sample_rate": model.sampling_rate})
        def render(text, cancelled):
            with (contextlib.redirect_stdout(sys.stderr), torch.inference_mode(),
                  cancellation_guard(model, cancelled)):
                if args.seed is not None:
                    torch.manual_seed(args.seed)
                audio = np.asarray(model.generate(text=text, language="en",
                    voice_clone_prompt=prompt, num_step=args.steps,
                    postprocess_output=not args.keep_pauses)[0], dtype=np.float32)
                if audio.size == 0 or not np.isfinite(audio).all():
                    raise ValueError("Voice generation returned empty or nonfinite audio")
                peak = float(np.max(np.abs(audio)))
                if peak < 1e-6:
                    raise ValueError("Voice generation returned silence")
                # Match the audible auditions with constant peak gain, no pitch change.
                pcm = (np.clip(audio * (0.85 / peak), -1, 1) * 32767).astype("<i2").tobytes()
            return pcm

        serve_requests(render, sys.stdin, send)
    except Exception as exc:  # noqa: BLE001 - report any model failure over IPC
        send({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
