import wave

import pytest

from jesse.wake_check import evaluate_clip, read_clip, summarize


def test_evaluation_warms_and_resets_and_counts_events_instead_of_frames():
    class Detector:
        def reset(self):
            self.scores = iter([0] * 16 + [0.8] * 15 + [0] + [0.9] * 4)
        def score(self, frame):
            return next(self.scores, 0)
    result = evaluate_clip(Detector(), bytes(32000), 0.5)
    assert result == {"seconds": 1.0, "peak": 0.9, "events": 2}


@pytest.mark.parametrize("n,hits,seconds,events,passed", [
    (1, 1, 1, 0, False), (20, 19, 3600, 0, True),
    (20, 18, 3600, 0, False), (20, 20, 3600, 1, False),
])
def test_sampled_gates_require_coverage_and_both_error_metrics(n, hits, seconds, events, passed):
    pos = [{"events": int(i < hits)} for i in range(n)]
    report = summarize(pos, [{"seconds": seconds, "events": events}])
    assert report["sampled_gate_passed"] is passed


def test_wav_reader_rejects_wrong_sample_rate(tmp_path):
    path = tmp_path / "wrong.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(bytes(16000))
    with pytest.raises(ValueError, match="16000"):
        read_clip(path)
