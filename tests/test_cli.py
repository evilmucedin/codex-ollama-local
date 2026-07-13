"""Tests for the argparse CLI.

The CLI builds its own :class:`OllamaClient` from config, so these tests patch
``codex_ollama.cli.OllamaClient`` with a fake to stay offline while still
exercising ``main`` end-to-end (argument parsing, dispatch, output, exit codes).
"""

from __future__ import annotations

import io

import pytest

import codex_ollama.cli as cli
from codex_ollama.ollama_client import OllamaError
from codex_ollama.types import ChatChunk, ModelInfo


class FakeClient:
    """Stand-in for OllamaClient; records calls and returns canned data."""

    def __init__(self, *, models=None, chunks=None, error=None) -> None:
        self._models = models or []
        self._chunks = chunks or []
        self._error = error
        self.chat_calls: list[dict] = []

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def list_models(self):
        if self._error:
            raise self._error
        return self._models

    def chat(self, messages, *, model=None):
        if self._error:
            raise self._error
        self.chat_calls.append({"messages": list(messages), "model": model})
        return iter(self._chunks)


def patch_client(monkeypatch: pytest.MonkeyPatch, fake: FakeClient) -> None:
    monkeypatch.setattr(cli, "OllamaClient", lambda config: fake)


def test_no_command_prints_help_and_returns_1(capsys: pytest.CaptureFixture) -> None:
    assert cli.main([]) == 1
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_version(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "col" in capsys.readouterr().out


def test_models_lists_names(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    fake = FakeClient(models=[ModelInfo("a:1"), ModelInfo("b:2")])
    patch_client(monkeypatch, fake)
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "a:1" in out
    assert "b:2" in out


def test_models_empty_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    patch_client(monkeypatch, FakeClient(models=[]))
    assert cli.main(["models"]) == 0
    assert "ollama pull" in capsys.readouterr().out


def test_chat_streams_reply(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    fake = FakeClient(
        chunks=[ChatChunk("Hel", False), ChatChunk("lo", False), ChatChunk("", True)]
    )
    patch_client(monkeypatch, fake)
    assert cli.main(["chat", "hi there"]) == 0
    assert capsys.readouterr().out == "Hello\n"
    assert fake.chat_calls[0]["messages"][0].content == "hi there"


def test_chat_reads_stdin_when_no_prompt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    fake = FakeClient(chunks=[ChatChunk("ok", True)])
    patch_client(monkeypatch, fake)
    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin\n"))
    assert cli.main(["chat"]) == 0
    assert fake.chat_calls[0]["messages"][0].content == "from stdin"


def test_chat_empty_prompt_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    patch_client(monkeypatch, FakeClient())
    monkeypatch.setattr("sys.stdin", io.StringIO("   \n"))
    assert cli.main(["chat"]) == 2
    assert "empty prompt" in capsys.readouterr().err


def test_chat_model_override_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient(chunks=[ChatChunk("x", True)])
    patch_client(monkeypatch, fake)
    assert cli.main(["chat", "-m", "custom:9", "hi"]) == 0
    assert fake.chat_calls[0]["model"] == "custom:9"


def test_ollama_error_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    fake = FakeClient(error=OllamaError("is it running?"))
    patch_client(monkeypatch, fake)
    assert cli.main(["models"]) == 2
    assert "error:" in capsys.readouterr().err
