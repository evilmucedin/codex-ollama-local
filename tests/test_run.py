"""Tests for the root ``run.py`` launcher script.

Network checks and subprocess launches are mocked, so no server or Codex process
is actually started.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest


def test_build_codex_command(run_mod: ModuleType) -> None:
    cmd = run_mod.build_codex_command("qwen2.5-coder:7b", ["--foo", "bar"])
    assert cmd == ["codex", "--oss", "-m", "qwen2.5-coder:7b", "--foo", "bar"]


def test_default_model_and_host_from_env(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_OLLAMA_MODEL", "m:9")
    monkeypatch.setenv("OLLAMA_HOST", "http://h:9")
    assert run_mod.default_model() == "m:9"
    assert run_mod.default_host() == "http://h:9"


def test_defaults_when_env_absent(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CODEX_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert run_mod.default_model() == run_mod.DEFAULT_MODEL
    assert run_mod.default_host() == run_mod.DEFAULT_HOST


def test_is_ollama_running_true_and_false(run_mod: ModuleType) -> None:
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def ok_opener(url, timeout):
        assert url.endswith("/api/tags")
        return _Resp()

    def bad_opener(url, timeout):
        raise OSError("refused")

    assert run_mod.is_ollama_running("http://x", opener=ok_opener) is True
    assert run_mod.is_ollama_running("http://x", opener=bad_opener) is False


def test_ensure_running_noop_when_up(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_mod, "is_ollama_running", lambda host, **kw: True)
    assert run_mod.ensure_ollama_running("http://x") is None


def test_ensure_running_no_autostart_raises(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_mod, "is_ollama_running", lambda host, **kw: False)
    with pytest.raises(run_mod.RunError):
        run_mod.ensure_ollama_running("http://x", autostart=False)


def test_ensure_running_missing_binary_raises(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_mod, "is_ollama_running", lambda host, **kw: False)
    monkeypatch.setattr(run_mod, "command_exists", lambda name: False)
    with pytest.raises(run_mod.RunError) as exc:
        run_mod.ensure_ollama_running("http://x")
    assert "install.py" in str(exc.value)


def test_ensure_running_starts_then_ready(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    states = iter([False, True])  # down at first check, up after spawn

    def fake_running(host, **kw):
        try:
            return next(states)
        except StopIteration:
            return True

    sentinel = object()
    monkeypatch.setattr(run_mod, "is_ollama_running", fake_running)
    monkeypatch.setattr(run_mod, "command_exists", lambda name: True)
    monkeypatch.setattr(run_mod, "_spawn_ollama_serve", lambda: sentinel)
    monkeypatch.setattr(run_mod.time, "sleep", lambda s: None)
    assert run_mod.ensure_ollama_running("http://x") is sentinel


def test_ensure_running_times_out(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_mod, "is_ollama_running", lambda host, **kw: False)
    monkeypatch.setattr(run_mod, "command_exists", lambda name: True)
    monkeypatch.setattr(run_mod, "_spawn_ollama_serve", lambda: object())
    monkeypatch.setattr(run_mod.time, "sleep", lambda s: None)
    with pytest.raises(run_mod.RunError):
        run_mod.ensure_ollama_running("http://x", wait_seconds=0.0)


def test_main_errors_when_codex_missing(
    run_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(run_mod, "command_exists", lambda name: name != "codex")
    assert run_mod.main([]) == 2
    assert "install.py" in capsys.readouterr().err


def test_main_launches_codex(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_mod, "command_exists", lambda name: True)
    monkeypatch.setattr(run_mod, "ensure_ollama_running", lambda host, **kw: None)
    captured: dict = {}

    def fake_subprocess_run(cmd, env=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_mod.subprocess, "run", fake_subprocess_run)
    rc = run_mod.main(["-m", "custom:1", "--host", "http://h:2", "--", "--help"])
    assert rc == 0
    assert captured["cmd"] == ["codex", "--oss", "-m", "custom:1", "--help"]
    assert captured["env"]["OLLAMA_HOST"] == "http://h:2"


def test_main_forwards_codex_exit_code(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_mod, "command_exists", lambda name: True)
    monkeypatch.setattr(run_mod, "ensure_ollama_running", lambda host, **kw: None)
    monkeypatch.setattr(
        run_mod.subprocess, "run", lambda cmd, env=None: SimpleNamespace(returncode=3)
    )
    assert run_mod.main([]) == 3


def test_main_reports_ollama_error(
    run_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(run_mod, "command_exists", lambda name: True)

    def boom(host, **kw):
        raise run_mod.RunError("no server")

    monkeypatch.setattr(run_mod, "ensure_ollama_running", boom)
    assert run_mod.main([]) == 2
    assert "error:" in capsys.readouterr().err
