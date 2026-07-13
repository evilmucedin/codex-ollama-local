"""codex-ollama-local: a standalone coding CLI agent backed by a local Ollama.

The package exposes a small CLI (``col``) that connects to a locally running
Ollama server, lists available models, and holds streaming chat conversations.
It is designed to grow into a full agentic coding assistant; see
``docs/ARCHITECTURE.md`` for the target design.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
