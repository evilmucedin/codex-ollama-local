"""Tests for the Ollama HTTP client, using httpx.MockTransport."""

from __future__ import annotations

import httpx
import pytest

from codex_ollama.ollama_client import OllamaClient, OllamaError
from codex_ollama.types import Message

from .conftest import make_client, ndjson


def test_list_models_parses_payload(tags_client: OllamaClient) -> None:
    with tags_client:
        models = tags_client.list_models()
    names = [m.name for m in models]
    assert names == ["qwen2.5-coder:7b", "llama3.2:3b"]
    assert models[0].size == 42
    assert models[0].modified_at == "t0"


def test_list_models_empty() -> None:
    client = make_client(lambda request: httpx.Response(200, json={}))
    with client:
        assert client.list_models() == []


def test_chat_assembles_stream(chat_client: OllamaClient) -> None:
    with chat_client:
        chunks = list(chat_client.chat([Message("user", "hi")]))
    text = "".join(c.content for c in chunks)
    assert text == "Hello world"
    assert chunks[-1].done is True
    assert chunks[0].done is False


def test_chat_sends_expected_body() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            content=ndjson({"message": {"content": "ok"}, "done": True}),
        )

    client = make_client(handler)
    with client:
        list(client.chat([Message("user", "hi")], model="custom:1"))

    assert captured["model"] == "custom:1"
    assert captured["stream"] is True
    assert captured["messages"] == [{"role": "user", "content": "hi"}]


def test_list_models_raises_on_http_error() -> None:
    client = make_client(lambda request: httpx.Response(500, text="boom"))
    with client:
        with pytest.raises(OllamaError) as exc:
            client.list_models()
    assert "500" in str(exc.value)


def test_list_models_raises_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = make_client(handler)
    with client:
        with pytest.raises(OllamaError) as exc:
            client.list_models()
    assert "ollama serve" in str(exc.value)


def test_chat_raises_on_http_error() -> None:
    client = make_client(lambda request: httpx.Response(404, text="no model"))
    with client:
        with pytest.raises(OllamaError) as exc:
            list(client.chat([Message("user", "hi")]))
    assert "404" in str(exc.value)


def test_chat_raises_on_malformed_line() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json\n")

    client = make_client(handler)
    with client:
        with pytest.raises(OllamaError) as exc:
            list(client.chat([Message("user", "hi")]))
    assert "Malformed" in str(exc.value)
