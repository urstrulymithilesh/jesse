"""A measured gain/threshold pair must reach the live loop unchanged."""

from types import SimpleNamespace

import pytest

from jesse.__main__ import _run
from jesse.audio.calibrate import Calibration
from jesse.audio.vad import EnergyVad
from jesse.config import CONFIG


@pytest.fixture
def startup(monkeypatch):
    seen = []
    orch = SimpleNamespace(
        transport=SimpleNamespace(gain=CONFIG.audio.capture_gain),
        vad=EnergyVad(threshold=CONFIG.audio.vad_threshold),
    )

    async def run():
        seen.append((orch.transport.gain, orch.vad.threshold))

    def calibrate(device):
        seen.append("calibration")
        return Calibration(1, 169, 14.8, 1004, True, "measured")

    orch.run = run
    monkeypatch.setattr("jesse.factory.build_orchestrator", lambda **kwargs: (orch, "test", "test"))
    monkeypatch.setattr("jesse.audio.calibrate.calibrate", calibrate)
    monkeypatch.setattr("jesse.core.single_instance.claim", lambda _: None)
    monkeypatch.setattr("jesse.core.single_instance.release", lambda _: None)
    return seen


@pytest.mark.parametrize("args, expected", [
    (["--gain", "14.8", "--threshold", "1004"], (14.8, 1004)),
    (["--gain", "2"], (2, CONFIG.audio.vad_threshold)),
    (["--threshold", "500"], (CONFIG.audio.capture_gain, 500)),
])
def test_manual_audio_settings_reach_loop_without_recalibration(startup, args, expected):
    assert _run(args) == 0
    assert startup == [expected]


def test_no_overrides_still_auto_calibrates(startup):
    assert _run([]) == 0
    assert startup == ["calibration", (14.8, 1004)]


def test_no_calibrate_keeps_configured_pair(startup):
    assert _run(["--no-calibrate"]) == 0
    assert startup == [(CONFIG.audio.capture_gain, CONFIG.audio.vad_threshold)]


@pytest.mark.parametrize("args", [[], ["--ui", "--remote"]])
def test_running_instance_stops_startup_before_ui_models_or_audio(monkeypatch, capsys, args):
    def unexpected(*args, **kwargs):
        pytest.fail("startup continued despite another Jesse holding the microphone")

    monkeypatch.setattr("jesse.core.single_instance.claim", lambda _: "Another Jesse is running")
    monkeypatch.setattr("jesse.core.single_instance.release", unexpected)
    monkeypatch.setattr("jesse.factory.build_orchestrator", unexpected)
    monkeypatch.setattr("jesse.ui.server.start", unexpected)
    assert _run(args) == 1
    assert "Another Jesse is running" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--gain", "--threshold"])
@pytest.mark.parametrize("value", [None, "0", "-1", "nan", "inf", "quiet"])
def test_invalid_audio_settings_fail_before_services_or_devices(monkeypatch, capsys, flag, value):
    def unexpected(*args, **kwargs):
        pytest.fail("startup continued with invalid audio options")

    monkeypatch.setattr("jesse.llm.ollama.OllamaLLM.check_available", unexpected)
    monkeypatch.setattr("jesse.core.single_instance.claim", unexpected)
    monkeypatch.setattr("jesse.factory.build_orchestrator", unexpected)
    args = ["--ollama", "--ui", flag] + ([] if value is None else [value])
    assert _run(args) == 2
    assert flag in capsys.readouterr().out
