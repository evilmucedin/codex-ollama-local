"""The package version is importable and matches pyproject.toml."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import codex_ollama

if sys.version_info >= (3, 11):
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_version_is_a_string() -> None:
    assert isinstance(codex_ollama.__version__, str)
    assert codex_ollama.__version__


def test_version_matches_pyproject() -> None:
    pyproject = _PROJECT_ROOT / "pyproject.toml"
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    assert data["project"]["version"] == codex_ollama.__version__
