"""Shared pytest fixtures.

All fixtures build an :class:`OllamaClient` on top of an ``httpx.MockTransport``
so the entire suite runs offline and deterministically on every platform.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Callable

import httpx
import pytest

from codex_ollama.config import Config
from codex_ollama.ollama_client import OllamaClient

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str) -> ModuleType:
    """Import a top-level script (``install.py`` / ``run.py``) by file path.

    These live at the repo root rather than in the installed package, so they are
    loaded explicitly instead of via a normal import.
    """

    path = _PROJECT_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def install_mod() -> ModuleType:
    return load_script("install")


@pytest.fixture(scope="session")
def run_mod() -> ModuleType:
    return load_script("run")


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
