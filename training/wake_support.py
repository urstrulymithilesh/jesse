"""Small, testable helpers for the separate Colab training environment."""

import hashlib
import json
from pathlib import Path

OPENWAKEWORD_REV = "368c03716d1e92591906a84949bc477f3a834455"
PIPER_REV = "f1988a4d54eddb23d99e86f0adfef6226a85acc7"


def fix_boolean_defaults(source: str) -> str:
    """Pinned trainer uses truthy strings, accidentally enabling TFLite export."""
    before, after = 'default="False",', 'default=False,'
    if source.count(before) != 5:
        raise ValueError("Unexpected upstream flags; review trainer before patching")
    return source.replace(before, after)


def write_once(path: Path, content: str) -> None:
    """Allow identical reruns, refuse to mix runs with different settings."""
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing file differs: {path}. Use a fresh run directory.")
    else:
        path.write_text(content, encoding="utf-8")


def make_config(defaults: dict, overlay: dict, root: Path, features: Path,
                validation: Path) -> dict:
    root = root.resolve()
    config = dict(defaults)
    config.update(overlay)
    config.update({
        "piper_sample_generator_path": str(root / "piper-sample-generator"),
        "output_dir": str(root / "output"),
        "rir_paths": [str(root / "mit_rirs")],
        "background_paths": [str(root / "background")],
        "background_paths_duplication_rate": [1],
        "false_positive_validation_data_path": str(validation.resolve()),
        "feature_data_files": {"ACAV100M_sample": str(features.resolve())},
        # Conservative generation batches for a free T4. These are starting settings.
        "tts_batch_size": 16,
        "augmentation_batch_size": 16,
    })
    if features.resolve() == validation.resolve():
        raise ValueError("Training and validation features must be separate")
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_features(path: Path, *, validation: bool) -> tuple:
    import numpy as np
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    # Upstream training examples are windows; false-positive validation is a stream.
    ndim = 2 if validation else 3
    if data.ndim != ndim or data.shape[-1] != 96 or data.shape[0] == 0:
        raise ValueError(f"Invalid feature array: {path}: {data.shape}")
    return data.shape


def package_model(root: Path, session_factory=None) -> Path:
    """Check exported ONNX is runnable and retain provenance, not a quality claim."""
    import numpy as np

    if session_factory is None:
        from onnxruntime import InferenceSession
        session_factory = InferenceSession
    model = root / "output" / "yo_jesse.onnx"
    session = session_factory(str(model), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError("Expected one wake embedding input")
    shape = inputs[0].shape
    if len(shape) != 3 or shape[-1] != 96 or not isinstance(shape[1], int):
        raise ValueError(f"Unexpected wake model input: {shape}")
    score = np.asarray(session.run(None, {
        inputs[0].name: np.zeros((1, shape[1], 96), dtype=np.float32),
    })[0])
    if score.size != 1 or not np.isfinite(score).all() or not (0 <= score.item() <= 1):
        raise ValueError("Export does not produce a finite binary wake score")
    report = {
        "model": model.name, "sha256": sha256(model),
        "openwakeword_revision": OPENWAKEWORD_REV, "piper_revision": PIPER_REV,
        "trainer_patch": "five argparse string False defaults replaced by boolean False",
        "input_shape": shape, "silence_embedding_score": float(score.item()),
        "config_sha256": sha256(root / "yo_jesse.yaml"),
        "human_acceptance_passed": False,
        "note": "Export/inference check only. Run Jesse wake-check on held-out human audio.",
    }
    report_path = root / "export-report.json"
    # A later successful training attempt can produce a different model; keep each bundle.
    import uuid
    import zipfile
    bundle = root / f"yo_jesse-{uuid.uuid4().hex[:8]}.zip"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (model, report_path, root / "yo_jesse.yaml", root / "environment.txt"):
            archive.write(path, path.name)
        for path in sorted(root.glob("*.log")):
            archive.write(path, path.name)
    return bundle
