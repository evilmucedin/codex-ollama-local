"""Lightweight data structures shared across the package.

These are plain :mod:`dataclasses` with no third-party dependencies so they are
trivial to construct in tests and cheap to import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Message:
    """A single chat message exchanged with the model.

    ``role`` is one of ``"system"``, ``"user"`` or ``"assistant"`` following the
    Ollama / OpenAI chat conventions.
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ChatChunk:
    """A single streamed fragment of an assistant reply.

    Ollama streams a chat response as newline-delimited JSON objects; each is
    decoded into one :class:`ChatChunk`. ``content`` is the incremental text (may
    be empty) and ``done`` marks the final chunk of the stream.
    """

    content: str
    done: bool


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about a model available locally in Ollama."""

    name: str
    size: Optional[int] = None
    modified_at: Optional[str] = None
