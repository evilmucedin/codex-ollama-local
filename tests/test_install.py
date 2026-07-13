"""Tests for the root ``install.py`` bootstrap script.

All subprocess calls are mocked, so nothing is actually installed.
"""

from __future__ import annotations

import pathlib
from types import ModuleType, SimpleNamespace

import pytest


def test_detect_os_maps_platforms(
    install_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    for system, expected in [
        ("Linux", "linux"),
        ("Darwin", "macos"),
        ("Windows", "windows"),
    ]:
        monkeypatch.setattr(install_mod.platform, "system", lambda s=system: s)
        assert install_mod.detect_os() == expected


def test_detect_os_rejects_unknown(
    install_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Plan9")
    with pytest.raises(install_mod.InstallError):
        install_mod.detect_os()


def test_ollama_install_command_linux(install_mod: ModuleType) -> None:
    cmd, shell = install_mod.ollama_install_command("linux")
    assert shell is True
    assert "install.sh" in cmd and "| sh" in cmd


def test_ollama_install_command_macos_uses_brew(
    install_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_mod, "command_exists", lambda name: name == "brew")
    cmd, shell = install_mod.ollama_install_command("macos")
    assert cmd == ["brew", "install", "ollama"]
    assert shell is False


def test_ollama_install_command_macos_without_brew_errors(
    install_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_mod, "command_exists", lambda name: False)
    with pytest.raises(install_mod.InstallError):
        install_mod.ollama_install_command("macos")


def test_ollama_install_command_windows_uses_winget(
    install_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_mod, "command_exists", lambda name: name == "winget")
    cmd, shell = install_mod.ollama_install_command("windows")
    assert cmd[:2] == ["winget", "install"]
    assert "Ollama.Ollama" in cmd


def test_codex_install_command_prefers_npm(
    install_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_mod, "command_exists", lambda name: name == "npm")
    cmd, shell = install_mod.codex_install_command("linux")
    assert cmd == ["npm", "install", "-g", "@openai/codex"]
    assert shell is False


def test_codex_install_command_without_npm_errors(
    install_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_mod, "command_exists", lambda name: False)
    with pytest.raises(install_mod.InstallError):
        install_mod.codex_install_command("linux")


def test_install_skips_when_present(
    install_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(install_mod, "command_exists", lambda name: True)
    ran = install_mod.install_ollama("linux")
    assert ran is False
    assert "already installed" in capsys.readouterr().out


def test_install_runs_when_missing(
    install_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list = []
    monkeypatch.setattr(install_mod, "command_exists", lambda name: name == "npm")
    monkeypatch.setattr(
        install_mod, "run", lambda cmd, **kw: calls.append((cmd, kw)) or 0
    )
    ran = install_mod.install_codex("linux")
    assert ran is True
    assert calls and calls[0][0] == ["npm", "install", "-g", "@openai/codex"]


def test_install_raises_on_nonzero_exit(
    install_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_mod, "command_exists", lambda name: name == "npm")
    monkeypatch.setattr(install_mod, "run", lambda cmd, **kw: 1)
    with pytest.raises(install_mod.InstallError):
        install_mod.install_codex("linux")


def test_run_dry_run_does_not_execute(
    install_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("subprocess.run should not be called on dry-run")

    monkeypatch.setattr(install_mod.subprocess, "run", fail)
    assert install_mod.run(["echo", "hi"], dry_run=True) == 0


def test_main_dry_run_installs_both(
    install_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Linux")
    # Nothing installed yet, npm available for codex.
    monkeypatch.setattr(install_mod, "command_exists", lambda name: name == "npm")
    executed: list = []
    monkeypatch.setattr(install_mod, "run", lambda cmd, **kw: executed.append(cmd) or 0)
    assert install_mod.main(["--dry-run"]) == 0
    # Both codex and ollama install commands were issued.
    joined = " ".join(str(c) for c in executed)
    assert "@openai/codex" in joined
    assert "install.sh" in joined


def test_main_only_ollama(
    install_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install_mod, "command_exists", lambda name: False)
    executed: list = []
    monkeypatch.setattr(install_mod, "run", lambda cmd, **kw: executed.append(cmd) or 0)
    assert install_mod.main(["--only", "ollama"]) == 0
    joined = " ".join(str(c) for c in executed)
    assert "install.sh" in joined
    assert "@openai/codex" not in joined


def test_main_reports_unsupported_os(
    install_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Plan9")
    assert install_mod.main([]) == 2


# -- codex PATH verification -----------------------------------------------
def test_npm_global_bin_dir(
    install_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_mod, "command_exists", lambda name: True)
    monkeypatch.setattr(
        install_mod.subprocess,
        "run",
        lambda cmd, capture_output, text: SimpleNamespace(stdout="/usr/local\n"),
    )
    # Mirror the helper's pathlib ops so the assertion holds on every OS without
    # patching os.name (which would make pathlib build a PosixPath on Windows).
    prefix = pathlib.Path("/usr/local")
    expected = prefix if install_mod.os.name == "nt" else prefix / "bin"
    assert install_mod.npm_global_bin_dir() == str(expected)


def test_warn_if_codex_unreachable_warns(
    install_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(install_mod, "command_exists", lambda name: False)
    monkeypatch.setattr(install_mod, "npm_global_bin_dir", lambda: "/x/bin")
    install_mod.warn_if_codex_unreachable(dry_run=False)
    err = capsys.readouterr().err
    assert "not on your PATH" in err
    assert "/x/bin" in err


def test_warn_if_codex_unreachable_noop_when_present(
    install_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(install_mod, "command_exists", lambda name: True)
    install_mod.warn_if_codex_unreachable(dry_run=False)
    assert capsys.readouterr().err == ""


def test_warn_if_codex_unreachable_noop_on_dry_run(
    install_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(install_mod, "command_exists", lambda name: False)
    install_mod.warn_if_codex_unreachable(dry_run=True)
    assert capsys.readouterr().err == ""


def test_main_warns_when_codex_off_path(
    install_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Linux")
    # npm present so Codex "installs", but codex never lands on PATH.
    monkeypatch.setattr(install_mod, "command_exists", lambda name: name == "npm")
    monkeypatch.setattr(install_mod, "run", lambda cmd, **kw: 0)
    monkeypatch.setattr(install_mod, "npm_global_bin_dir", lambda: "/x/bin")
    assert install_mod.main(["--only", "codex"]) == 0
    assert "not on your PATH" in capsys.readouterr().err
