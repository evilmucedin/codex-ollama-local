#!/usr/bin/env python3
"""Bootstrap the local toolchain for codex-ollama-local.

This installs the two external tools the project drives:

  * the **OpenAI Codex CLI** (the coding-agent front-end), via ``npm``
  * the **Ollama** server (runs the local models)

Each tool is skipped if it is already installed, so re-running is safe. The
script uses only the Python standard library so it can run on a clean machine
before anything else (including this project's package) is installed.

Cross-platform: Ubuntu (Linux), macOS, and Windows.

Usage::

    python install.py                # install whatever is missing
    python install.py --dry-run      # print the commands without running them
    python install.py --force        # reinstall even if already present
    python install.py --only ollama  # install just one tool (repeatable)
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple, Union

Command = Union[List[str], str]
DOWNLOAD_OLLAMA = "https://ollama.com/download"
CODEX_REPO = "https://github.com/openai/codex"


class InstallError(RuntimeError):
    """Raised when a prerequisite is missing or an install step fails."""


def detect_os() -> str:
    """Return ``"linux"``, ``"macos"`` or ``"windows"`` for the current host."""

    system = platform.system()
    mapping = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}
    try:
        return mapping[system]
    except KeyError:
        raise InstallError(f"Unsupported operating system: {system!r}")


def command_exists(name: str) -> bool:
    """True if ``name`` is found on ``PATH``."""

    return shutil.which(name) is not None


def run(cmd: Command, *, dry_run: bool = False, shell: bool = False) -> int:
    """Echo and execute ``cmd``, returning its exit code (0 when ``dry_run``)."""

    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"$ {printable}")
    if dry_run:
        return 0
    return subprocess.run(cmd, shell=shell).returncode


def ollama_install_command(os_name: str) -> Tuple[Command, bool]:
    """Return ``(command, shell)`` used to install Ollama on ``os_name``."""

    if os_name == "linux":
        # Official convenience installer; supported on Ubuntu.
        return "curl -fsSL https://ollama.com/install.sh | sh", True
    if os_name == "macos":
        if command_exists("brew"):
            return ["brew", "install", "ollama"], False
        raise InstallError(
            "Homebrew not found. Install it from https://brew.sh, or download "
            f"Ollama from {DOWNLOAD_OLLAMA}."
        )
    if os_name == "windows":
        if command_exists("winget"):
            return (
                ["winget", "install", "--id", "Ollama.Ollama", "-e"],
                False,
            )
        raise InstallError(f"winget not found. Download Ollama from {DOWNLOAD_OLLAMA}.")
    raise InstallError(f"Unsupported OS: {os_name!r}")


def codex_install_command(os_name: str) -> Tuple[Command, bool]:
    """Return ``(command, shell)`` used to install the Codex CLI on ``os_name``.

    ``npm`` is the documented install path and works on all three platforms; on
    macOS we fall back to Homebrew if Node is not available.
    """

    if command_exists("npm"):
        return ["npm", "install", "-g", "@openai/codex"], False
    if os_name == "macos" and command_exists("brew"):
        return ["brew", "install", "codex"], False
    raise InstallError(
        "npm (Node.js) was not found. Install Node.js from https://nodejs.org "
        f"and re-run, or see {CODEX_REPO} for alternative install methods."
    )


def _install_tool(
    label: str,
    binary: str,
    command_factory,
    os_name: str,
    *,
    force: bool,
    dry_run: bool,
) -> bool:
    """Install one tool; returns True if an install ran, False if skipped."""

    if command_exists(binary) and not force:
        print(f"{label} is already installed; skipping (use --force to reinstall).")
        return False

    print(f"Installing {label}...")
    cmd, shell = command_factory(os_name)
    code = run(cmd, dry_run=dry_run, shell=shell)
    if code != 0:
        raise InstallError(f"{label} installation failed (exit code {code}).")
    return True


def install_codex(os_name: str, *, force: bool = False, dry_run: bool = False) -> bool:
    return _install_tool(
        "Codex CLI",
        "codex",
        codex_install_command,
        os_name,
        force=force,
        dry_run=dry_run,
    )


def install_ollama(os_name: str, *, force: bool = False, dry_run: bool = False) -> bool:
    return _install_tool(
        "Ollama",
        "ollama",
        ollama_install_command,
        os_name,
        force=force,
        dry_run=dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="install.py",
        description="Install the Codex CLI and Ollama server locally.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall even if the tool is already present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    parser.add_argument(
        "--only",
        choices=["codex", "ollama"],
        action="append",
        help="Install only this tool (may be given more than once).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        os_name = detect_os()
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    targets = args.only or ["codex", "ollama"]
    installers = {"codex": install_codex, "ollama": install_ollama}

    try:
        for target in targets:
            installers[target](os_name, force=args.force, dry_run=args.dry_run)
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        "\nDone. Start the agent with `python run.py` "
        "(it will launch Codex against your local Ollama)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
