"""Shared pytest fixtures.

All fixtures build an :class:`OllamaClient` on top of an ``httpx.MockTransport``
so the entire suite runs offline and deterministically on every platform.
"""

from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from codex_ollama.config import Config
from codex_ollama.ollama_client import OllamaClient


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaClient:
    """Build an :class:`OllamaClient` whose transport is driven by ``handler``."""

    transport = httpx.MockTransport(handler)
    config = Config(host="http://ollama.test")
    return OllamaClient(config, transport=transport)


def ndjson(*objects: dict) -> bytes:
    """Encode ``objects`` as newline-delimited JSON, as Ollama streams them."""

    return "".join(json.dumps(obj) + "\n" for obj in objects).encode("utf-8")


@pytest.fixture
def tags_client() -> OllamaClient:
    """A client whose ``/api/tags`` endpoint returns two models."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen2.5-coder:7b", "size": 42, "modified_at": "t0"},
                    {"name": "llama3.2:3b", "size": 7, "modified_at": "t1"},
                ]
            },
        )

    return make_client(handler)


@pytest.fixture
def chat_client() -> OllamaClient:
    """A client whose ``/api/chat`` endpoint streams a two-token reply."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        stream = ndjson(
            {"message": {"role": "assistant", "content": "Hello"}, "done": False},
            {"message": {"role": "assistant", "content": " world"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True},
        )
        return httpx.Response(200, content=stream)

    return make_client(handler)
