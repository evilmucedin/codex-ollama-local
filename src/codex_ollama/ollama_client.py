"""HTTP client for a local Ollama server.

Wraps :class:`httpx.Client` so the transport can be swapped for an
``httpx.MockTransport`` in tests, keeping the whole suite offline. All network
failures are normalised into :class:`OllamaError` with actionable messages.
"""

from __future__ import annotations

import json
from typing import Iterable, Iterator, Optional

import httpx

from .config import Config
from .types import ChatChunk, Message, ModelInfo


class OllamaError(RuntimeError):
    """Raised when the Ollama server cannot be reached or returns an error."""


class OllamaClient:
    """A thin synchronous client for the subset of the Ollama API we use."""

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        client: Optional[httpx.Client] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.config = config or Config()
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            timeout = httpx.Timeout(
                self.config.request_timeout,
                connect=self.config.connect_timeout,
            )
            self._client = httpx.Client(
                base_url=self.config.host,
                timeout=timeout,
                transport=transport,
            )
            self._owns_client = True

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- API ---------------------------------------------------------------
    def list_models(self) -> list[ModelInfo]:
        """Return the models available locally (``GET /api/tags``)."""

        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OllamaError(_status_message(exc)) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(_connect_message(self.config.host, exc)) from exc

        payload = response.json()
        models: list[ModelInfo] = []
        for item in payload.get("models", []):
            models.append(
                ModelInfo(
                    name=item.get("name", ""),
                    size=item.get("size"),
                    modified_at=item.get("modified_at"),
                )
            )
        return models

    def chat(
        self,
        messages: Iterable[Message],
        *,
        model: Optional[str] = None,
    ) -> Iterator[ChatChunk]:
        """Stream a chat completion (``POST /api/chat``).

        Yields one :class:`ChatChunk` per newline-delimited JSON object returned
        by Ollama. The generator opens the streaming response lazily; consume it
        to completion (or close it) to release the connection.
        """

        body = {
            "model": model or self.config.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
        }
        return self._stream_chat(body)

    def _stream_chat(self, body: dict) -> Iterator[ChatChunk]:
        try:
            with self._client.stream("POST", "/api/chat", json=body) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    response.read()
                    raise OllamaError(_status_message(exc)) from exc
                for line in response.iter_lines():
                    if not line:
                        continue
                    yield _parse_chunk(line)
        except httpx.HTTPError as exc:
            raise OllamaError(_connect_message(self.config.host, exc)) from exc


def _parse_chunk(line: str) -> ChatChunk:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Malformed response from Ollama: {line!r}") from exc
    message = data.get("message") or {}
    return ChatChunk(content=message.get("content", ""), done=bool(data.get("done")))


def _connect_message(host: str, exc: Exception) -> str:
    return (
        f"Could not reach Ollama at {host} ({exc}). "
        "Is it running? Start it with `ollama serve`."
    )


def _status_message(exc: httpx.HTTPStatusError) -> str:
    return (
        f"Ollama returned HTTP {exc.response.status_code} for "
        f"{exc.request.url.path}: {exc.response.text.strip()}"
    )
