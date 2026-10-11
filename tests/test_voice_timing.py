from contextlib import nullcontext

import pytest

from training.voice_timing import measure_stages


@pytest.mark.parametrize("fail", [False, True])
def test_timing_synchronizes_accumulates_and_restores_methods(monkeypatch, fail):
    class Model:
        def _generate_iterative(self, value):
            return value * 2

        def _decode_and_post_process(self, value):
            if fail:
                raise ValueError("decode failed")
            return value + 1

    model = Model()
    # Preserve an existing instance override as well as the inherited method.
    model._generate_iterative = model._generate_iterative
    override = model._generate_iterative
    clock = iter([0, 2, 3, 6, 7, 11])
    monkeypatch.setattr("training.voice_timing.perf_counter", lambda: next(clock))
    sync = []
    with (
        pytest.raises(ValueError) if fail else nullcontext(),
        measure_stages(model, lambda: sync.append(True)) as timings,
    ):
        assert model._generate_iterative(2) == 4
        assert model._generate_iterative(3) == 6
        assert model._decode_and_post_process(6) == 7
    assert timings == {"_generate_iterative": 5, "_decode_and_post_process": 4}
    assert len(sync) == 6
    assert model._generate_iterative is override
    assert "_decode_and_post_process" not in vars(model)
