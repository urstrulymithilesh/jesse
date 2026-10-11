"""Optional synchronized stage timings for local OmniVoice experiments."""

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def measure_stages(model, synchronize=lambda: None):
    """Measure generation and decode separately, restoring methods even on failure."""
    timings = {}
    originals = []
    try:
        for name in ("_generate_iterative", "_decode_and_post_process"):
            original = getattr(model, name)
            had_override = name in vars(model)

            def measured(*args, _original=original, _name=name, **kwargs):
                synchronize()
                started = perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    synchronize()
                    timings[_name] = timings.get(_name, 0.0) + perf_counter() - started

            originals.append((name, original, had_override))
            setattr(model, name, measured)
        yield timings
    finally:
        for name, original, had_override in reversed(originals):
            if had_override:
                setattr(model, name, original)
            else:
                delattr(model, name)
