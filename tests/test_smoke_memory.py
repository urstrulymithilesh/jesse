"""Storing a fact must not hide a wrong or dishonest spoken acknowledgement."""

import asyncio
from types import SimpleNamespace

import pytest

from jesse import smoke
from jesse.core.interfaces import Fact, Message


@pytest.mark.parametrize("reply", ["I don't have a favorite color.", "I've saved that.", ""])
def test_memory_smoke_rejects_wrong_acknowledgement_even_when_fact_was_saved(
    monkeypatch, tmp_path, reply,
):
    async def run():
        pass

    orch = SimpleNamespace(run=run, _extract_task=None,
                           _history=[Message("assistant", reply)] if reply else [])
    closed = []
    store = SimpleNamespace(
        _conn=SimpleNamespace(execute=lambda sql: SimpleNamespace(fetchone=lambda: (2,))),
        all_facts=lambda: [Fact(text="the user likes turquoise best", confidence=0.9)],
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(smoke, "_build", lambda *args: (orch, store, None))
    mouth = SimpleNamespace(frames=lambda text: [])
    result = asyncio.run(smoke.scenario_memory(mouth, tmp_path / "memory.db"))
    assert not result.passed
    assert "acknowledged honestly" in result.detail
    assert closed == [True]
