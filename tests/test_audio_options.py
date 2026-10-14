"""A measured gain/threshold pair must reach the live loop unchanged."""

from types import SimpleNamespace

import pytest

from pinkman.__main__ import _run
from pinkman.audio.calibrate import Calibration
from pinkman.audio.vad import EnergyVad
from pinkman.config import CONFIG


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
    monkeypatch.setattr("pinkman.factory.build_orchestrator", lambda **kwargs: (orch, "test", "test"))
    monkeypatch.setattr("pinkman.audio.calibrate.calibrate", calibrate)
    monkeypatch.setattr("pinkman.core.single_instance.claim", lambda _: None)
    monkeypatch.setattr("pinkman.core.single_instance.release", lambda _: None)
    monkeypatch.setattr("pinkman.ui.server.start", lambda *a, **kw: "http://127.0.0.1:8765")
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
        pytest.fail("startup continued despite another Pinkman holding the microphone")

    monkeypatch.setattr("pinkman.core.single_instance.claim", lambda _: "Another Pinkman is running")
    monkeypatch.setattr("pinkman.core.single_instance.release", unexpected)
    monkeypatch.setattr("pinkman.factory.build_orchestrator", unexpected)
    monkeypatch.setattr("pinkman.ui.server.start", unexpected)
    assert _run(args) == 1
    assert "Another Pinkman is running" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--gain", "--threshold"])
@pytest.mark.parametrize("value", [None, "0", "-1", "nan", "inf", "quiet"])
def test_invalid_audio_settings_fail_before_services_or_devices(monkeypatch, capsys, flag, value):
    def unexpected(*args, **kwargs):
        pytest.fail("startup continued with invalid audio options")

    monkeypatch.setattr("pinkman.llm.ollama.OllamaLLM.check_available", unexpected)
    monkeypatch.setattr("pinkman.core.single_instance.claim", unexpected)
    monkeypatch.setattr("pinkman.factory.build_orchestrator", unexpected)
    args = ["--ollama", "--ui", flag] + ([] if value is None else [value])
    assert _run(args) == 2
    assert flag in capsys.readouterr().out
