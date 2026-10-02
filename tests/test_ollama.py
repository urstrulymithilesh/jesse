"""Extraction constraints belong to that request, not to ordinary replies."""

import io
import json
import urllib.error
import pytest

from jesse.config import CONFIG
from jesse.core.interfaces import Message
from jesse.core.interfaces import LLMError
from jesse.llm.ollama import OllamaLLM
from jesse.memory.extraction import FACT_SCHEMA, FactExtractor


def test_extraction_sends_schema_and_zero_temperature_without_affecting_chat(monkeypatch):
    requests = []

    def open_request(self, body):
        requests.append(json.loads(body))
        return io.BytesIO(b'{"message": {"content": "[]"}}')

    monkeypatch.setattr(OllamaLLM, "_open", open_request)
    extractor = FactExtractor(OllamaLLM(response_schema=FACT_SCHEMA, temperature=0))
    assert extractor.extract("My favorite color is turquoise.") == "[]"
    list(OllamaLLM().chat([Message("user", "hello")], stream=False))
    structured, ordinary = requests
    assert structured["format"] == FACT_SCHEMA
    assert structured["options"]["temperature"] == 0
    assert structured["stream"] is False
    assert "format" not in ordinary
    assert ordinary["options"]["temperature"] == CONFIG.reasoning.temperature


def test_streaming_chat_still_emits_tokens(monkeypatch):
    def open_request(self, body):
        assert json.loads(body)["stream"] is True
        return io.BytesIO(
            b'{"message":{"content":"Hi"},"done":false}\n'
            b'{"message":{"content":" there"},"done":false}\n'
            b'{"done":true}\n'
        )

    monkeypatch.setattr(OllamaLLM, "_open", open_request)
    assert list(OllamaLLM().chat([Message("user", "hello")])) == ["Hi", " there"]


def test_startup_probe_is_a_bounded_get_without_generation(monkeypatch):
    def open_request(url, *, timeout):
        assert url == "http://localhost:11434/api/tags"
        assert timeout == 2.0
        return io.BytesIO(b'{"models": []}')
    monkeypatch.setattr("urllib.request.urlopen", open_request)
    OllamaLLM(host="http://localhost:11434").check_available()


@pytest.mark.parametrize("failure", [urllib.error.URLError("offline"), TimeoutError("slow")])
def test_probe_reports_start_ollama_first(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(LLMError, match="Start Ollama first"):
        OllamaLLM().check_available()


def test_run_fails_before_claiming_mic_or_starting_ui(monkeypatch, capsys):
    from jesse.__main__ import _run
    def fail(self):
        raise LLMError("Start Ollama first")
    def unexpected(*args, **kwargs):
        pytest.fail("startup continued after the failed preflight")
    monkeypatch.setattr(OllamaLLM, "check_available", fail)
    monkeypatch.setattr("jesse.core.single_instance.claim", unexpected)
    monkeypatch.setattr("jesse.factory.build_orchestrator", unexpected)
    assert _run(["--ollama", "--ui"]) == 1
    assert "Start Ollama first" in capsys.readouterr().out
