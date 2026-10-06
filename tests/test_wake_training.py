import argparse
import ast
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training.build_wake_notebook import build_notebook
from training.wake_support import (
    check_features, fix_boolean_defaults, make_config, package_model, sha256, write_once,
)


def test_notebook_matches_source_and_all_cells_and_embedded_scripts_compile():
    saved = json.loads(Path("training/train_yo_jesse.ipynb").read_text(encoding="utf-8"))
    assert saved == build_notebook(), "Regenerate the notebook after changing its inputs"
    scripts = []
    for cell in saved["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and cell["outputs"] == []
        tree = ast.parse("".join(cell["source"]), feature_version=(3, 10))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "script":
                    scripts.append(node.args[0].value)
                    ast.parse(node.args[1].value, feature_version=(3, 10))
    assert scripts == ["probe.py", "prepare_data.py", "export_candidate.py"]


def test_upstream_flag_fix_keeps_unselected_stages_and_tflite_disabled():
    source = "\n".join(
        f'parser.add_argument("--{flag}", action="store_true", default="False",)'
        for flag in ("generate", "augment", "overwrite", "train", "tflite")
    )
    parser = argparse.ArgumentParser()
    exec(fix_boolean_defaults(source), {"parser": parser})
    args = vars(parser.parse_args(["--train"]))
    assert args == dict(generate=False, augment=False, overwrite=False, train=True, tflite=False)


def test_patch_refuses_changed_upstream_instead_of_silently_skipping():
    with pytest.raises(ValueError, match="Unexpected upstream"):
        fix_boolean_defaults('default="False",')


def test_configuration_uses_distinct_absolute_data_paths_and_jesse_phrase(tmp_path):
    overlay = json.loads(Path("training/yo_jesse.json").read_text())
    config = make_config({"max_negative_weight": 1500}, overlay, tmp_path,
                         tmp_path / "train.npy", tmp_path / "validation.npy")
    assert config["target_phrase"] == ["yo Jesse"]
    assert config["max_negative_weight"] == 1500
    assert config["n_samples"] == 20000
    assert config["feature_data_files"]["ACAV100M_sample"] == str(tmp_path / "train.npy")
    assert config["false_positive_validation_data_path"] == str(tmp_path / "validation.npy")
    assert len(config["background_paths"]) == len(config["background_paths_duplication_rate"])
    with pytest.raises(ValueError, match="separate"):
        make_config({}, overlay, tmp_path, tmp_path / "train.npy", tmp_path / "train.npy")


def test_identical_rerun_preserves_config_but_changed_settings_require_fresh_run(tmp_path):
    path = tmp_path / "config.yaml"
    write_once(path, "steps: 50000")
    write_once(path, "steps: 50000")
    with pytest.raises(ValueError, match="fresh run"):
        write_once(path, "steps: 100")
    assert path.read_text() == "steps: 50000"


def test_validation_is_a_continuous_stream_and_training_uses_windows(tmp_path):
    train, validation = tmp_path / "train.npy", tmp_path / "validation.npy"
    np.save(train, np.zeros((2, 16, 96), dtype=np.float16))
    np.save(validation, np.zeros((32, 96), dtype=np.float32))
    assert check_features(train, validation=False) == (2, 16, 96)
    assert check_features(validation, validation=True) == (32, 96)
    with pytest.raises(ValueError, match="Invalid feature"):
        check_features(validation, validation=False)


def session_factory(score):
    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 16, 96])]

        def run(self, _, inputs):
            assert inputs["input"].shape == (1, 16, 96)
            return [np.array([[score]], dtype=np.float32)]
    return Session


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_export_rejects_invalid_scores_without_packaging(tmp_path, score):
    with pytest.raises(ValueError, match="finite binary"):
        package_model(tmp_path, session_factory(score))
    assert not list(tmp_path.glob("*.zip"))


def test_export_bundle_records_model_hash_and_does_not_claim_acceptance(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    model = output / "yo_jesse.onnx"
    model.write_bytes(b"fake model checked by fake session")
    (tmp_path / "yo_jesse.yaml").write_text("target_phrase: [yo Jesse]")
    (tmp_path / "environment.txt").write_text("torch==2.5.1")
    (tmp_path / "train.log").write_text("training log")
    bundle = package_model(tmp_path, session_factory(0.1))
    with zipfile.ZipFile(bundle) as archive:
        report = json.loads(archive.read("export-report.json"))
        assert report["sha256"] == sha256(model)
        assert report["human_acceptance_passed"] is False
        assert archive.read("train.log") == b"training log"
    assert package_model(tmp_path, session_factory(0.1)) != bundle
