import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from training.benchmark_omnivoice import main


@pytest.fixture
def bench(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "model.safetensors").touch()
    prompt_path = tmp_path / "voice-prompt.pt"
    prompt_path.touch()
    output = tmp_path / "result"
    prompt = object()
    calls = []

    class Model:
        sampling_rate = 24000
        audio_tokenizer = object()

        def generate(self, **kwargs):
            calls.append(kwargs)
            assert kwargs["voice_clone_prompt"] is prompt
            if kwargs["num_step"] == 32:
                raise RuntimeError("injected inference failure")
            return [np.ones(2400, dtype=np.float32) * 0.1]

    model = Model()
    torch = SimpleNamespace(
        __version__="test", float32="float32", set_num_threads=lambda _: None,
        manual_seed=lambda _: None, inference_mode=nullcontext,
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "omnivoice", SimpleNamespace(
        OmniVoice=SimpleNamespace(from_pretrained=lambda *a, **kw: model),
        VoiceClonePrompt=SimpleNamespace(load=lambda _: prompt),
    ))
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(
        write=lambda path, *a, **kw: path.write_bytes(b"test audio"),
    ))
    argv = ["--prompt", str(prompt_path), "--model", str(model_path), "--output", str(output)]
    return argv, output, calls


def test_benchmark_retains_completed_audio_and_error_when_later_run_fails(bench):
    argv, output, calls = bench
    with pytest.raises(RuntimeError, match="injected"):
        main(argv + ["--steps", "16", "32"])
    report = json.loads((output / "report.json").read_text())
    assert report["runs"][0]["steps"] == 16
    assert report["error"] == "RuntimeError: injected inference failure"
    assert report["user_approved_similarity"] is False
    assert (output / "steps-16-listen.wav").exists()
    assert not (output / "steps-32.wav").exists()
    assert len(calls) == 2


def test_requested_cuda_never_silently_benchmarks_cpu(bench):
    argv, output, calls = bench
    with pytest.raises(SystemExit) as error:
        main(argv + ["--mode", "cuda-fp16"])
    assert error.value.code == 2
    assert not output.exists()
    assert not calls


def test_existing_benchmark_is_not_overwritten(bench):
    argv, output, calls = bench
    assert main(argv + ["--steps", "16"]) == 0
    original = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        main(argv + ["--steps", "16"])
    assert (output / "report.json").read_bytes() == original
    assert len(calls) == 1


@pytest.mark.parametrize("flags,options", [
    ([], {}),
    (["--speed", "0.95", "--keep-pauses"],
     {"speed": 0.95, "postprocess_output": False}),
])
def test_audition_records_effective_delivery_options_without_changing_defaults(bench, flags, options):
    argv, output, calls = bench
    assert main(argv + ["--steps", "16", *flags]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["seed"] == 42
    assert report["generation_options"] == options
    actual = {key: value for key, value in calls[0].items()
              if key in ("speed", "postprocess_output")}
    assert actual == options
    assert report["user_approved_similarity"] is False


@pytest.mark.parametrize("speed", ["nan", "inf", "0", "-1", "2.1"])
def test_invalid_audition_speed_fails_before_model_load(bench, speed):
    argv, output, calls = bench
    with pytest.raises(SystemExit):
        main(argv + ["--speed", speed])
    assert calls == []
    assert not output.exists()
