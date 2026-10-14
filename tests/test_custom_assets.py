import asyncio
from types import SimpleNamespace

import pytest

from pinkman.customization import run_asset_options
from tests.test_orchestrator import _build


@pytest.mark.parametrize("args", [
    ["--wake-model"], ["--wake-phrase", "yo Pinkman"],
    ["--wake-threshold", "nan"], ["--wake-threshold", "0"],
    ["--wake-threshold", "1"], ["--voice", "../wrong"],
])
def test_bad_asset_options_fail_before_startup(args):
    with pytest.raises(ValueError):
        run_asset_options(args)


def test_asset_names_are_independent_of_spoken_phrase_and_require_voice_config(tmp_path, monkeypatch):
    monkeypatch.setattr("pinkman.customization.MODELS_DIR", tmp_path)
    model = tmp_path / "checkpoint-123.onnx"
    model.write_bytes(b"fixture")
    voice = tmp_path / "en_US-pinkman-medium.onnx"
    voice.write_bytes(b"fixture")
    args = ["--wake-model", str(model), "--wake-phrase", "yo Pinkman",
            "--voice", "en_US-pinkman-medium", "--wake-threshold", "0.7"]
    with pytest.raises(ValueError, match="onnx.json"):
        run_asset_options(args)
    voice.with_suffix(".onnx.json").write_text("{}")
    assert run_asset_options(args) == {"wake_model": str(model.resolve()),
        "wake_phrase": "yo Pinkman", "wake_threshold": 0.7, "voice": "en_US-pinkman-medium"}
    assert run_asset_options([]) == {}


def test_factory_routes_selected_assets_and_validates_before_use(monkeypatch):
    from pinkman.factory import build_orchestrator
    ensured = []
    monkeypatch.setattr("pinkman.audio.transport.LocalAudioTransport", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr("pinkman.audio.wakeword.OpenWakeWordDetector._ensure",
                        lambda self: ensured.append(self._model_name))
    monkeypatch.setattr("pinkman.tts.piper.PiperSynthesizer.is_available", lambda voice=None: True)
    monkeypatch.setattr("pinkman.tts.piper.PiperSynthesizer.sample_rate", property(lambda _: 22050))
    orch, voice_label, _ = build_orchestrator(wake_model="candidate.onnx", wake_phrase="yo Pinkman",
                                            wake_threshold=0.7, voice="en_US-pinkman-medium")
    assert ensured == ["candidate.onnx", "candidate.onnx"]
    assert orch.wake_phrase == "yo Pinkman"
    assert orch.wake._threshold == orch.stopword._threshold == 0.7
    assert "en_US-pinkman-medium" in voice_label
    assert orch.synth._model_path.name == "en_US-pinkman-medium.onnx"


def test_custom_wake_phrase_is_removed_before_shared_turn_processing():
    orch, _ = _build([])
    orch.wake_phrase = "yo Pinkman"
    orch.transcriber.transcribe = lambda _: "Yo Pinkman, I like tea."
    asyncio.run(orch._run_turn(b"audio"))
    assert orch._history[0].content == "I like tea."


def test_invalid_export_fails_cleanly_and_releases_instance(monkeypatch, capsys):
    from pinkman.__main__ import _run
    released = []
    monkeypatch.setattr("pinkman.core.single_instance.claim", lambda _: None)
    monkeypatch.setattr("pinkman.core.single_instance.release", released.append)
    def fail(**kwargs):
        raise ValueError("invalid ONNX export")
    monkeypatch.setattr("pinkman.factory.build_orchestrator", fail)
    assert _run([]) == 1
    assert len(released) == 1
    assert "invalid ONNX export" in capsys.readouterr().out


def test_omnivoice_is_explicit_and_conflicts_with_piper(tmp_path):
    profile = tmp_path / "voice.json"
    profile.write_text("{}")
    assert run_asset_options(["--omnivoice-profile", str(profile)]) == {
        "omnivoice_profile": str(profile.resolve())}
    with pytest.raises(ValueError, match="either"):
        run_asset_options(["--omnivoice-profile", str(profile), "--voice", "fallback"])
    with pytest.raises(ValueError, match="does not exist"):
        run_asset_options(["--omnivoice-profile", str(tmp_path / "missing.json")])


def test_factory_selects_clone_without_starting_a_worker(monkeypatch):
    from pinkman.factory import build_orchestrator
    selected = []
    def clone(profile):
        selected.append(profile)
        return SimpleNamespace(steps=8)
    monkeypatch.setattr("pinkman.tts.omnivoice.OmniVoiceSynthesizer", clone)
    monkeypatch.setattr("pinkman.tts.piper.PiperSynthesizer.is_available",
                        lambda *args: pytest.fail("Piper should not be selected"))
    orch, label, _ = build_orchestrator(omnivoice_profile="candidate.json")
    assert selected == ["candidate.json"]
    assert orch.synth.steps == 8
    assert "Pinkman voice" in label


def test_local_default_profile_is_used_but_explicit_piper_wins(tmp_path, monkeypatch):
    from pinkman.factory import build_orchestrator
    profile = tmp_path / "default.json"
    profile.write_text("{}")
    monkeypatch.setattr("pinkman.factory.DEFAULT_VOICE_PROFILE", profile)
    chosen = []
    def clone(path):
        chosen.append(path)
        return SimpleNamespace(steps=32)
    monkeypatch.setattr("pinkman.tts.omnivoice.OmniVoiceSynthesizer", clone)
    monkeypatch.setattr("pinkman.tts.piper.PiperSynthesizer.is_available", lambda *args: True)
    monkeypatch.setattr("pinkman.tts.piper.PiperSynthesizer.sample_rate", property(lambda _: 22050))
    _, label, _ = build_orchestrator()
    assert "Pinkman voice" in label and chosen == [str(profile)]
    _, label, _ = build_orchestrator(voice="en_US-pinkman-fallback")
    assert "Piper" in label and len(chosen) == 1


def test_broken_local_default_never_silently_switches_to_piper(tmp_path, monkeypatch):
    from pinkman.factory import build_orchestrator
    profile = tmp_path / "default.json"
    profile.write_text("{}")
    monkeypatch.setattr("pinkman.factory.DEFAULT_VOICE_PROFILE", profile)
    monkeypatch.setattr("pinkman.tts.piper.PiperSynthesizer.is_available",
                        lambda *args: pytest.fail("Selected clone must fail clearly"))
    with pytest.raises(ValueError, match="requires python"):
        build_orchestrator()
