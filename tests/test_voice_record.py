import csv
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from jesse.voice_record import RATE, main, pending_prompts, quality, save_clip


def audio():
    return (np.sin(np.arange(RATE) * 0.1) * 2000).astype(np.int16)


def test_quality_rejects_silence_and_clipping():
    assert not quality(bytes(RATE * 2))[0]
    assert not quality(np.full(RATE, 32767, dtype=np.int16).tobytes())[0]
    assert quality(audio().tobytes())[0]


def test_accepted_clips_have_piper_metadata_and_never_overwrite(tmp_path):
    first = save_clip(tmp_path, "Hello there.", audio().tobytes())
    second = save_clip(tmp_path, "How are you?", audio().tobytes())
    assert first != second and first.is_file() and second.is_file()
    with wave.open(str(first), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, RATE)
    with (tmp_path / "metadata.csv").open(newline="") as stream:
        assert list(csv.reader(stream, delimiter="|")) == [
            [first.name, "Hello there."], [second.name, "How are you?"]]
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("Hello there.\nNext sentence.\n")
    assert pending_prompts(prompts, tmp_path) == ["Next sentence."]


def test_bad_take_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError):
        save_clip(tmp_path, "hello", bytes(RATE * 2))
    with pytest.raises(ValueError):
        save_clip(tmp_path, "bad|prompt", audio().tobytes())
    assert not (tmp_path / "metadata.csv").exists()


def test_recording_requires_review_and_resumes_without_opening_audio(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("Good morning.\n")
    events = []
    answers = iter(["", "k"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr("jesse.voice_record.time.sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(
        check_input_settings=lambda **_: None, check_output_settings=lambda **_: None,
        rec=lambda *a, **kw: audio(), wait=lambda: None,
        play=lambda *a, **kw: events.append("review")))
    args = ["--prompts", str(prompts), "--output", str(tmp_path / "dataset")]
    assert main(args) == 0 and events == ["review"]
    assert main(args) == 0 and events == ["review"]


def test_missing_audio_in_existing_dataset_is_not_silently_skipped(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("Good morning.\n")
    (tmp_path / "metadata.csv").write_text("missing.wav|Good morning.\n")
    with pytest.raises(ValueError, match="missing audio"):
        pending_prompts(prompts, tmp_path)
