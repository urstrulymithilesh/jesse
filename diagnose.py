"""Jesse audio diagnostics — find where the wake->STT chain breaks.

Two commands:

    python diagnose.py                 # list every audio device (find your headset mic)
    python diagnose.py listen          # live monitor on the DEFAULT input device
    python diagnose.py listen 5        # live monitor on device index 5
    python diagnose.py listen 5 --threshold 400

The live monitor shows, per 80ms frame, all in one line:
  * a VU meter of the mic level (RMS)   -> is audio arriving at all?
  * which device it opened               -> is it your headset or the built-in mic?
  * the live 'hey_jarvis' wake score     -> is openWakeWord hearing you, just under threshold?
  * [SPEECH] when level clears the VAD threshold, and <<< WAKE when the word fires

If the VU meter stays flat while you talk, it's the wrong device -> pick the right
index and pass it to `python -m jesse run --device N`.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import sounddevice as sd

from jesse.audio.frames import CHUNK_SAMPLES, SAMPLE_RATE

WAKE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Command 1: list devices
# ---------------------------------------------------------------------------


def list_devices(inputs_only: bool = False) -> None:
    from jesse.audio.devices import format_device_table

    title = "AUDIO INPUT DEVICES" if inputs_only else "AUDIO DEVICES  (cap = IN / OUT / IN+OUT)"
    print("=" * 94)
    print(f" {title}  (note your headset mic's index)")
    print("=" * 94)
    print(format_device_table(inputs_only=inputs_only))
    print(" " + "-" * 92)
    print(" Only rows with cap 'IN' or 'IN+OUT' can be recorded from.")
    print(" Prefer an 'MME' or 'Windows WASAPI' entry over 'WDM-KS' (WDM-KS Bluetooth")
    print(" mics often refuse to open). Then:  python diagnose.py listen <idx>")


# ---------------------------------------------------------------------------
# Command 2: live monitor
# ---------------------------------------------------------------------------


def _wake_model_info() -> object:
    import openwakeword
    from openwakeword.model import Model

    res_dir = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
    onnx = os.path.join(res_dir, "hey_jarvis_v0.1.onnx")
    print(f"  wake model : hey_jarvis")
    print(f"  model file : {onnx}")
    print(f"  file found : {os.path.isfile(onnx)}")
    # inference_framework pinned to onnx (tflite runtime isn't installed on Windows)
    model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    return model


def _vu(rms: float, width: int = 40, full_scale: float = 3000.0) -> str:
    n = min(width, int(width * rms / full_scale))
    return "#" * n + "-" * (width - n)


def listen(device: int | None, threshold: float, gain: float = 1.0) -> None:
    from jesse.audio.devices import DeviceError, format_device_table, validate_input_device

    print("=" * 78)
    print(" LIVE MONITOR — talk into your mic; Ctrl-C to stop")
    print("=" * 78)

    try:
        validate_input_device(device)
    except DeviceError as e:
        print(f"\n{e}")
        return

    dev_info = sd.query_devices(device, "input") if device is not None else sd.query_devices(kind="input")
    print(f"  device     : [{device if device is not None else 'default'}] {dev_info['name']}")
    print(f"  samplerate : {SAMPLE_RATE} Hz, frame {CHUNK_SAMPLES} samples ({CHUNK_SAMPLES*1000//SAMPLE_RATE} ms)")
    print(f"  capture gain: x{gain:.1f}  (levels + wake below are POST-gain, as the loop sees them)")
    print(f"  VAD thresh : {threshold:.0f} RMS  (level must exceed this to count as speech)")

    model = _wake_model_info()
    print("  " + "-" * 74)
    print("  Say 'hey jarvis'. Watch the wake score spike toward 1.0.\n")

    peak_rms = 0.0
    peak_score = 0.0
    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SAMPLES,
                                channels=1, dtype="int16", device=device)
    except sd.PortAudioError as e:
        print(f"\nCould not open device {device} for 16 kHz mono capture ({e}).")
        print("WDM-KS Bluetooth mics often refuse to open — try the mic's MME/WASAPI")
        print(f"entry instead.\n\n{format_device_table(inputs_only=True)}")
        return
    with stream:
        while True:
            data, overflowed = stream.read(CHUNK_SAMPLES)
            samples = np.asarray(data, dtype=np.int16).reshape(-1)
            if gain != 1.0 and samples.size:
                boosted = np.clip(samples.astype(np.float32) * gain, -32768, 32767)
                samples = boosted.astype(np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.0
            scores = model.predict(samples)
            score = float(scores.get("hey_jarvis", 0.0))
            peak_rms = max(peak_rms, rms)
            peak_score = max(peak_score, score)

            speech = "SPEECH" if rms >= threshold else "  ..  "
            wake = "  <<< WAKE!" if score >= WAKE_THRESHOLD else ""
            over = " OVERFLOW" if overflowed else ""
            print(f"\r level {rms:6.0f} |{_vu(rms)}| {speech}  wake={score:0.3f}"
                  f"  (peak lvl {peak_rms:5.0f}, peak wake {peak_score:0.3f}){wake}{over}",
                  end="", flush=True)


def _parse_listen_args(args: list[str]) -> tuple[int | None, float, float]:
    from jesse.config import CONFIG

    device: int | None = None
    threshold = CONFIG.audio.vad_threshold
    gain = CONFIG.audio.capture_gain
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--threshold":
            threshold = float(args[i + 1]); i += 2; continue
        if a == "--gain":
            gain = float(args[i + 1]); i += 2; continue
        if a.lstrip("-").isdigit():
            device = int(a)
        i += 1
    return device, threshold, gain


def _calibrate(args: list[str]) -> None:
    from jesse.audio.calibrate import calibrate
    from jesse.audio.devices import DeviceError, validate_input_device

    device = int(args[0]) if args and args[0].lstrip("-").isdigit() else None
    try:
        validate_input_device(device)
        result = calibrate(device)
    except DeviceError as e:
        print(f"\n{e}")
        return
    print(f"\n  Confirm it works together:  python diagnose.py listen "
          f"{device if device is not None else ''} --gain {result.gain} --threshold {result.threshold}")
    print("  Make permanent in jesse/config.py -> AudioConfig:")
    print(f"      capture_gain: float = {result.gain}")
    print(f"      vad_threshold: float = {result.threshold}")


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "listen":
        device, threshold, gain = _parse_listen_args(args[1:])
        try:
            listen(device, threshold, gain)
        except KeyboardInterrupt:
            print("\n\nStopped.")
        return 0
    if args and args[0] == "calibrate":
        _calibrate(args[1:])
        return 0
    if args and args[0] == "inputs":
        list_devices(inputs_only=True)
        return 0
    list_devices()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
