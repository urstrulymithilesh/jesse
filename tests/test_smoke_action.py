"""The smoke assertion checks the parsed target, including STT word spacing."""

import asyncio
from types import SimpleNamespace

import pytest

from jesse.core.interfaces import Message
from jesse import smoke


@pytest.mark.parametrize("target", ["photoshop", "photo shop"])
@pytest.mark.parametrize("refuses", [True, False])
def test_action_smoke_checks_exact_refusal_and_closes_store(monkeypatch, tmp_path, target, refuses):
    reply = (f"I don't have {target} — that's not something I can open."
             if refuses else f"I can open {target}.")
    async def run():
        pass
    orch = SimpleNamespace(run=run, _history=[
        Message("user", f"open {target}"), Message("assistant", reply),
    ])
    closed = []
    store = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(smoke, "_build", lambda *args: (orch, store, None))
    mouth = SimpleNamespace(frames=lambda text: [])
    result = asyncio.run(smoke.scenario_action(mouth, tmp_path / "smoke.db"))
    assert result.passed is refuses
    assert closed == [True]
