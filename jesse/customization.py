"""Explicit, per-launch selection of trained local wake and voice assets."""

import math
import re
from pathlib import Path

from jesse.config import MODELS_DIR


def run_asset_options(argv: list[str]) -> dict:
    values = {}
    for flag in ("--wake-model", "--wake-phrase", "--wake-threshold", "--voice",
                 "--omnivoice-profile"):
        if flag not in argv:
            continue
        i = argv.index(flag)
        if argv.count(flag) != 1 or i + 1 == len(argv) or argv[i + 1].startswith("--"):
            raise ValueError(f"{flag} requires one value")
        values[flag] = argv[i + 1].strip()
    if not values:
        return {}
    out = {}
    if "--omnivoice-profile" in values:
        if "--voice" in values:
            raise ValueError("Choose either --voice or --omnivoice-profile")
        path = Path(values["--omnivoice-profile"]).resolve()
        if not values["--omnivoice-profile"] or not path.is_file():
            raise ValueError(f"OmniVoice profile does not exist: {path}")
        out["omnivoice_profile"] = str(path)
    if "--wake-model" in values or "--wake-phrase" in values:
        if not values.get("--wake-model") or not values.get("--wake-phrase"):
            raise ValueError("Use --wake-model FILE.onnx together with --wake-phrase 'yo Jesse'")
        path = Path(values["--wake-model"]).resolve()
        if path.suffix.lower() != ".onnx" or not path.is_file():
            raise ValueError(f"Custom wake model is not an existing ONNX file: {path}")
        out.update(wake_model=str(path), wake_phrase=values["--wake-phrase"])
    if "--wake-threshold" in values:
        threshold = float(values["--wake-threshold"])
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("--wake-threshold must be between 0 and 1, exclusive")
        out["wake_threshold"] = threshold
    if "--voice" in values:
        voice = values["--voice"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", voice):
            raise ValueError("--voice must be a model name in models/, without a path or extension")
        for suffix in (".onnx", ".onnx.json"):
            if not (MODELS_DIR / f"{voice}{suffix}").is_file():
                raise ValueError(f"Missing voice file: {MODELS_DIR / (voice + suffix)}")
        out["voice"] = voice
    return out
