"""Tests for the root ``run.py`` launcher script.

Network checks and subprocess launches are mocked, so no server or Codex process
is actually started.
"""

from __future__ import annotations

import io
import json
import pathlib
from types import ModuleType, SimpleNamespace

import pytest


# -- command construction --------------------------------------------------
def test_build_codex_command(run_mod: ModuleType) -> None:
    assert run_mod.build_codex_command("m:1", ["--sandbox", "read-only"]) == [
        "codex",
        "--oss",
        "-m",
        "m:1",
        "--sandbox",
        "read-only",
    ]


def test_build_codex_command_minimal(run_mod: ModuleType) -> None:
    assert run_mod.build_codex_command("m:1", []) == ["codex", "--oss", "-m", "m:1"]


# -- model resolution ------------------------------------------------------
def test_resolve_model_prefers_cli(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CODEX_OLLAMA_MODEL", raising=False)
    assert run_mod.resolve_model("cli:1", ["a", "b"]) == "cli:1"


def test_resolve_model_uses_env(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_OLLAMA_MODEL", "env:2")
    assert run_mod.resolve_model(None, ["a"]) == "env:2"


def test_resolve_model_picks_first_installed(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CODEX_OLLAMA_MODEL", raising=False)
    assert run_mod.resolve_model(None, ["first", "second"]) == "first"


def test_resolve_model_falls_back_to_default(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CODEX_OLLAMA_MODEL", raising=False)
    assert run_mod.resolve_model(None, []) == run_mod.DEFAULT_MODEL


# -- ollama helpers --------------------------------------------------------
def test_list_ollama_models_parses(run_mod: ModuleType) -> None:
    body = json.dumps(
        {"models": [{"name": "a:1"}, {"name": "b:2"}, {"nope": 1}]}
    ).encode()

    def opener(url, timeout):
        assert url.endswith("/api/tags")
        return io.BytesIO(body)

    assert run_mod.list_ollama_models("http://x", opener=opener) == ["a:1", "b:2"]


def test_list_ollama_models_empty_on_error(run_mod: ModuleType) -> None:
    def opener(url, timeout):
        raise OSError("down")

    assert run_mod.list_ollama_models("http://x", opener=opener) == []


# -- codex discovery -------------------------------------------------------
def test_npm_global_bin_dir(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_mod, "command_exists", lambda name: True)

    def runner(cmd, capture_output, text):
        assert cmd == ["npm", "prefix", "-g"]
        return SimpleNamespace(stdout="/home/u/.npm-global\n", stderr="")

    # Expected value mirrors the helper's own pathlib ops, so it stays correct on
    # POSIX (<prefix>/bin) and Windows (<prefix>) without patching os.name (which
    # would break pathlib). Each OS branch is exercised natively by the CI matrix.
    prefix = pathlib.Path("/home/u/.npm-global")
    expected = prefix if run_mod.os.name == "nt" else prefix / "bin"
    assert run_mod.npm_global_bin_dir(runner=runner) == str(expected)


def test_npm_global_bin_dir_none_without_npm(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_mod, "command_exists", lambda name: False)
    assert run_mod.npm_global_bin_dir(runner=lambda *a, **k: None) is None


def test_find_codex_dir_returns_none_when_on_path(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_mod, "command_exists", lambda name: True)
    assert run_mod.find_codex_dir(bin_dir="/whatever") is None


def test_find_codex_dir_finds_off_path_binary(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(run_mod, "command_exists", lambda name: False)
    (tmp_path / "codex").write_text("#!/bin/sh\n")
    assert run_mod.find_codex_dir(bin_dir=str(tmp_path)) == str(tmp_path)


def test_find_codex_dir_none_when_absent(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(run_mod, "command_exists", lambda name: False)
    assert run_mod.find_codex_dir(bin_dir=str(tmp_path)) is None


def test_is_ollama_running_true_and_false(run_mod: ModuleType) -> None:
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    assert run_mod.is_ollama_running("http://x", opener=lambda u, timeout: _Resp())
    assert not run_mod.is_ollama_running(
        "http://x", opener=lambda u, timeout: (_ for _ in ()).throw(OSError())
    )


# -- ensure_ollama_running -------------------------------------------------
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


def test_ensure_running_starts_then_ready(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    states = iter([False, True])
    monkeypatch.setattr(
        run_mod, "is_ollama_running", lambda host, **kw: next(states, True)
    )
    monkeypatch.setattr(run_mod, "command_exists", lambda name: True)
    sentinel = object()
    monkeypatch.setattr(run_mod, "_spawn_ollama_serve", lambda: sentinel)
    monkeypatch.setattr(run_mod.time, "sleep", lambda s: None)
    assert run_mod.ensure_ollama_running("http://x") is sentinel


# -- main ------------------------------------------------------------------
def _patch_common(run_mod, monkeypatch, *, models=("m:1",)):
    monkeypatch.setattr(run_mod, "command_exists", lambda name: True)
    monkeypatch.setattr(run_mod, "ensure_ollama_running", lambda host, **kw: None)
    monkeypatch.setattr(run_mod, "list_ollama_models", lambda host: list(models))


def test_main_errors_when_ollama_missing(
    run_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(run_mod, "command_exists", lambda name: name != "ollama")
    assert run_mod.main([]) == 2
    assert "install.py" in capsys.readouterr().err


def test_main_errors_when_codex_missing(
    run_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(run_mod, "command_exists", lambda name: name != "codex")
    # Not discoverable in npm's global bin either.
    monkeypatch.setattr(run_mod, "find_codex_dir", lambda *a, **k: None)
    assert run_mod.main([]) == 2
    assert "Codex" in capsys.readouterr().err


def test_main_launches_codex_oss(
    run_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.delenv("CODEX_OLLAMA_MODEL", raising=False)
    _patch_common(run_mod, monkeypatch, models=("a:1", "b:2"))
    captured: dict = {}

    def fake_run(cmd, env=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_mod.subprocess, "run", fake_run)
    rc = run_mod.main(["--host", "http://h:2", "--", "--sandbox", "read-only"])
    assert rc == 0
    # First installed model becomes the default; extra args forwarded to Codex.
    assert captured["cmd"] == [
        "codex",
        "--oss",
        "-m",
        "a:1",
        "--sandbox",
        "read-only",
    ]
    assert captured["env"]["OLLAMA_HOST"] == "http://h:2"
    out = capsys.readouterr().out
    assert "2 local model(s)" in out
    assert "a:1" in out and "b:2" in out


def test_main_uses_default_model_when_none_installed(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CODEX_OLLAMA_MODEL", raising=False)
    _patch_common(run_mod, monkeypatch, models=())
    captured: dict = {}
    monkeypatch.setattr(
        run_mod.subprocess,
        "run",
        lambda cmd, env=None: captured.update(cmd=cmd) or SimpleNamespace(returncode=0),
    )
    assert run_mod.main([]) == 0
    assert captured["cmd"] == ["codex", "--oss", "-m", run_mod.DEFAULT_MODEL]


def test_main_respects_explicit_model(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_common(run_mod, monkeypatch, models=("a:1",))
    captured: dict = {}
    monkeypatch.setattr(
        run_mod.subprocess,
        "run",
        lambda cmd, env=None: captured.update(cmd=cmd) or SimpleNamespace(returncode=0),
    )
    assert run_mod.main(["-m", "chosen:1"]) == 0
    assert captured["cmd"] == ["codex", "--oss", "-m", "chosen:1"]


def test_main_adds_npm_bin_to_path_when_codex_off_path(
    run_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.delenv("CODEX_OLLAMA_MODEL", raising=False)
    # codex is NOT on PATH, but discoverable in npm's global bin.
    monkeypatch.setattr(run_mod, "command_exists", lambda name: name == "ollama")
    monkeypatch.setattr(run_mod, "ensure_ollama_running", lambda host, **kw: None)
    monkeypatch.setattr(run_mod, "list_ollama_models", lambda host: ["a:1"])
    monkeypatch.setattr(run_mod, "find_codex_dir", lambda *a, **k: "/opt/npm/bin")
    captured: dict = {}
    monkeypatch.setattr(
        run_mod.subprocess,
        "run",
        lambda cmd, env=None: captured.update(cmd=cmd, env=env)
        or SimpleNamespace(returncode=0),
    )
    assert run_mod.main([]) == 0
    # The discovered dir is prepended to PATH for the launched process.
    assert captured["env"]["PATH"].split(run_mod.os.pathsep)[0] == "/opt/npm/bin"
    assert "not on PATH" in capsys.readouterr().err


def test_main_dry_run_does_not_launch(
    run_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    _patch_common(run_mod, monkeypatch, models=("a:1",))

    def fail(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("subprocess.run must not run on --dry-run")

    monkeypatch.setattr(run_mod.subprocess, "run", fail)
    # ensure_ollama_running must also be skipped on dry-run
    monkeypatch.setattr(
        run_mod,
        "ensure_ollama_running",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no serve on dry-run")),
    )
    assert run_mod.main(["--dry-run", "-m", "a:1"]) == 0
    assert "codex --oss -m a:1" in capsys.readouterr().out


def test_main_forwards_exit_code(
    run_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_common(run_mod, monkeypatch)
    monkeypatch.setattr(
        run_mod.subprocess, "run", lambda cmd, env=None: SimpleNamespace(returncode=7)
    )
    assert run_mod.main([]) == 7


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
