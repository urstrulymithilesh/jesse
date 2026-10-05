"""Interactive local collection of reviewed, single-speaker Piper training clips."""

import argparse
import csv
from pathlib import Path
import time
import uuid
import wave

import numpy as np

from jesse.config import CONFIG, DATA_DIR

RATE = 22050
PROMPTS = Path(__file__).resolve().parent.parent / "training" / "voice-prompts.txt"


def quality(pcm: bytes) -> tuple[bool, str]:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if not samples.size:
        return False, "No audio captured."
    rms = float(np.sqrt(np.mean(samples ** 2)))
    clipping = float(np.mean(np.abs(samples) >= 32760))
    if rms < 10 or np.max(np.abs(samples)) < 100:
        return False, "Almost silent: check the microphone and distance."
    if clipping > 0.005:
        return False, "Clipping: lower the microphone level or move farther away."
    return True, f"RMS {rms:.0f}, clipped samples {clipping:.2%}. Listen for noise or missing words."


def save_clip(root: Path, text: str, pcm: bytes) -> Path:
    if not text.strip() or any(c in text for c in "|\r\n"):
        raise ValueError("A prompt must be one nonempty line without a pipe character")
    ok, reason = quality(pcm)
    if not ok:
        raise ValueError(reason)
    audio = root / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    path = audio / f"{uuid.uuid4().hex}.wav"
    with path.open("xb") as stream, wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm)
    # The audio is complete before it is referenced in the training manifest.
    with (root / "metadata.csv").open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="|").writerow([path.name, text.strip()])
    return path


def pending_prompts(path: Path, root: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#")]
    if any("|" in line for line in prompts):
        raise ValueError("Prompts cannot contain pipe characters")
    existing = set()
    metadata = root / "metadata.csv"
    if metadata.exists():
        with metadata.open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream, delimiter="|"):
                if len(row) != 2 or not (root / "audio" / row[0]).is_file():
                    raise ValueError("Existing metadata has a malformed row or missing audio; repair it first")
                existing.add(row[1])
    return [text for text in dict.fromkeys(prompts) if text not in existing]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=CONFIG.audio.input_device)
    parser.add_argument("--output-device", type=int, default=CONFIG.audio.output_device)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "voice-training")
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--seconds", type=int, choices=range(3, 21), default=8)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    if args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        prompts = pending_prompts(args.prompts, args.output)[:args.limit]
        if not prompts:
            print("All prompts already have accepted recordings. Add more with --prompts FILE.")
            return 0
        import sounddevice as sd
        sd.check_input_settings(device=args.device, samplerate=RATE, channels=1, dtype="int16")
        sd.check_output_settings(device=args.output_device, samplerate=RATE, channels=1, dtype="int16")
        print(f"Local voice dataset: {args.output.resolve()}")
        print("Use your own voice, one speaker, a quiet room and a consistent mic distance.")
        print("Read naturally and confidently. Accepted clips and exact prompts are saved locally.")
        print("No upload or training is performed. Ctrl-C or q quits; saved clips can be resumed.")
        for i, text in enumerate(prompts, 1):
            while True:
                print(f"\n[{i}/{len(prompts)}] {text}")
                if input("Enter when ready, or q to quit: ").strip().lower() == "q":
                    return 0
                for count in (3, 2, 1):
                    print(count, flush=True)
                    time.sleep(1)
                print(f"RECORDING for {args.seconds} seconds...", flush=True)
                audio = sd.rec(args.seconds * RATE, samplerate=RATE, channels=1,
                               dtype="int16", device=args.device)
                sd.wait()
                pcm = audio.tobytes()
                ok, reason = quality(pcm)
                print(reason)
                if not ok:
                    continue
                print("Playing your take back. Check that every word matches the prompt.")
                sd.play(audio, RATE, device=args.output_device)
                sd.wait()
                answer = input("Keep [k], retry [Enter], skip [s], quit [q]: ").strip().lower()
                if answer == "q":
                    return 0
                if answer == "k":
                    print(f"Saved {save_clip(args.output, text, pcm).name}")
                    break
                if answer == "s":
                    break
        print("Session complete. Re-run to continue; accepted prompts are skipped.")
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\nRecording stopped. Accepted clips are preserved.")
        return 0
    except Exception as exc:
        print(f"Voice recording failed: {exc}")
        return 1
