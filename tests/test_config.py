"""Tests for configuration loading and cross-platform path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from codex_ollama import config as config_mod
from codex_ollama.config import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    Config,
    config_dir,
    config_file,
)


def test_defaults_with_empty_env(tmp_path: Path) -> None:
    cfg = Config.load(env={}, path=tmp_path / "missing.toml")
    assert cfg.host == DEFAULT_HOST
    assert cfg.model == DEFAULT_MODEL


def test_env_overrides_defaults(tmp_path: Path) -> None:
    env = {"OLLAMA_HOST": "http://host:1", "CODEX_OLLAMA_MODEL": "m:1"}
    cfg = Config.load(env=env, path=tmp_path / "missing.toml")
    assert cfg.host == "http://host:1"
    assert cfg.model == "m:1"


def test_explicit_args_override_env(tmp_path: Path) -> None:
    env = {"OLLAMA_HOST": "http://host:1", "CODEX_OLLAMA_MODEL": "m:1"}
    cfg = Config.load(
        host="http://cli:2", model="m:2", env=env, path=tmp_path / "x.toml"
    )
    assert cfg.host == "http://cli:2"
    assert cfg.model == "m:2"


def test_config_file_values_used_when_env_absent(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'host = "http://file:3"\n'
        'model = "m:3"\n'
        "connect_timeout = 1.5\n"
        "request_timeout = 30\n",
        encoding="utf-8",
    )
    cfg = Config.load(env={}, path=path)
    assert cfg.host == "http://file:3"
    assert cfg.model == "m:3"
    assert cfg.connect_timeout == 1.5
    assert cfg.request_timeout == 30.0


def test_env_beats_config_file(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('host = "http://file:3"\n', encoding="utf-8")
    cfg = Config.load(env={"OLLAMA_HOST": "http://env:4"}, path=path)
    assert cfg.host == "http://env:4"


@pytest.mark.parametrize(
    "platform, env, expected_parts",
    [
        ("win32", {"APPDATA": "/appdata"}, ("appdata", "codex-ollama")),
        ("darwin", {}, ("Library", "Application Support", "codex-ollama")),
        ("linux", {"XDG_CONFIG_HOME": "/xdg"}, ("xdg", "codex-ollama")),
        ("linux", {}, (".config", "codex-ollama")),
    ],
)
def test_config_dir_per_platform(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    env: dict,
    expected_parts: tuple,
) -> None:
    monkeypatch.setattr(config_mod.sys, "platform", platform)
    for key in ("APPDATA", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    result = config_dir()
    parts = result.parts
    for expected in expected_parts:
        assert expected in parts, f"{expected!r} not in {parts!r}"


def test_config_file_lives_in_config_dir() -> None:
    assert config_file().parent == config_dir()
    assert config_file().name == "config.toml"
