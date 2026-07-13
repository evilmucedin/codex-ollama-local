"""Configuration loading with cross-platform config-directory resolution.

Precedence (highest wins): explicit keyword arguments (from CLI flags) > environment
variables > values in the config file > built-in defaults. Paths are handled with
:mod:`pathlib` so the code behaves identically on Linux, Windows and macOS.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

# The config file lives inside this directory, if present.
APP_DIR_NAME = "codex-ollama"
CONFIG_FILE_NAME = "config.toml"

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_REQUEST_TIMEOUT = 120.0

ENV_HOST = "OLLAMA_HOST"
ENV_MODEL = "CODEX_OLLAMA_MODEL"


def config_dir() -> Path:
    """Return the per-user configuration directory for the current platform.

    - Windows: ``%APPDATA%\\codex-ollama``
    - macOS: ``~/Library/Application Support/codex-ollama``
    - Linux/other: ``$XDG_CONFIG_HOME/codex-ollama`` (falls back to ``~/.config``)
    """

    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME")
        root = Path(base) if base else Path.home() / ".config"
    return root / APP_DIR_NAME


def config_file() -> Path:
    """Return the full path to the config file (which may not exist)."""

    return config_dir() / CONFIG_FILE_NAME


def _read_config_file(path: Path) -> dict[str, Any]:
    """Read a TOML config file, returning an empty mapping if it is absent.

    Uses the stdlib :mod:`tomllib` (Python 3.11+). Malformed files raise, so a
    broken config is surfaced loudly rather than silently ignored.
    """

    if not path.is_file():
        return {}
    import tomllib

    with path.open("rb") as fh:
        return tomllib.load(fh)


@dataclass(frozen=True)
class Config:
    """Resolved runtime configuration."""

    host: str = DEFAULT_HOST
    model: str = DEFAULT_MODEL
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT

    @classmethod
    def load(
        cls,
        *,
        host: Optional[str] = None,
        model: Optional[str] = None,
        path: Optional[Path] = None,
        env: Optional[dict[str, str]] = None,
    ) -> "Config":
        """Build a :class:`Config` by merging the available sources.

        ``host``/``model`` are the highest-priority overrides (typically parsed
        from CLI flags). ``env`` defaults to :data:`os.environ` and ``path`` to
        :func:`config_file`; both are injectable to keep tests hermetic.
        """

        env = os.environ if env is None else env
        file_values = _read_config_file(config_file() if path is None else path)

        resolved_host = (
            host
            if host is not None
            else env.get(ENV_HOST) or file_values.get("host") or DEFAULT_HOST
        )
        resolved_model = (
            model
            if model is not None
            else env.get(ENV_MODEL) or file_values.get("model") or DEFAULT_MODEL
        )

        cfg = cls(host=resolved_host, model=resolved_model)

        # Optional numeric tunables only come from the config file.
        overrides: dict[str, float] = {}
        for key in ("connect_timeout", "request_timeout"):
            value = file_values.get(key)
            if value is not None:
                overrides[key] = float(value)
        if overrides:
            cfg = replace(cfg, **overrides)
        return cfg
