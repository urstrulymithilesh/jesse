"""Cancellation must unwind generation without changing completed voice output."""

import threading

import pytest

from pinkman.tts.omnivoice_worker import GenerationCancelled, cancellation_guard


class Model:
    def __init__(self):
        self.hooks = []
        self.steps = 0

    def register_forward_pre_hook(self, callback):
        self.hooks.append(callback)
        class Handle:
            def remove(inner):
                self.hooks.remove(callback)
        return Handle()

    def __call__(self, value):
        for hook in self.hooks:
            hook(self, (value,))
        self.steps += 1
        return value * 2


def test_cancel_between_steps_unwinds_and_next_generation_works():
    model, cancel = Model(), threading.Event()
    with pytest.raises(GenerationCancelled), cancellation_guard(model, cancel):
        assert model(3) == 6
        cancel.set()
        model(4)
    assert model.steps == 1
    assert model.hooks == []
    with cancellation_guard(model, threading.Event()):
        assert model(4) == 8
    assert model.hooks == []


def test_completed_generation_has_unchanged_results():
    model = Model()
    expected = [model(i) for i in range(16)]
    with cancellation_guard(model, threading.Event()):
        assert [model(i) for i in range(16)] == expected
    assert model.hooks == []


def test_cancel_during_decode_discards_result():
    model, cancel = Model(), threading.Event()
    with pytest.raises(GenerationCancelled), cancellation_guard(model, cancel):
        model(3)
        cancel.set()  # final CPU decoding can finish after the last model call
    assert model.hooks == []


def test_model_failure_is_not_swallowed_and_hook_removed():
    model = Model()
    with pytest.raises(ValueError, match="failed"), cancellation_guard(model, threading.Event()):
        raise ValueError("generation failed")
    assert model.hooks == []


def test_already_cancelled_request_does_not_touch_model():
    model, cancel = Model(), threading.Event()
    cancel.set()
    with pytest.raises(GenerationCancelled), cancellation_guard(model, cancel):
        pytest.fail("Cancelled request entered generation")
    assert model.steps == 0 and model.hooks == []
