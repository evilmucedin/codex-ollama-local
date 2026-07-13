"""Tests for the root ``install.py`` bootstrap script.

All subprocess calls are mocked, so nothing is actually installed.
"""

from __future__ import annotations

from types import ModuleType

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
