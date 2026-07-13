#!/usr/bin/env python3
"""Start the Codex coding agent against the local Ollama server.

This ensures Ollama is reachable — starting ``ollama serve`` in the background
if needed — and then launches the Codex CLI in its local open-source mode
(``codex --oss``) pointed at a local model. Any extra command-line arguments are
forwarded to Codex.

Uses only the Python standard library. Cross-platform: Ubuntu, macOS, Windows.

Usage::

    python run.py                       # launch codex --oss with the default model
    python run.py -m qwen2.5-coder:7b   # pick a local model
    python run.py --no-serve            # do not auto-start `ollama serve`
    python run.py -- --help             # forward `--help` to codex
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import List, Optional, Sequence

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gpt-oss:20b"
ENV_HOST = "OLLAMA_HOST"
ENV_MODEL = "CODEX_OLLAMA_MODEL"


class RunError(RuntimeError):
    """Raised when the agent cannot be started."""


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def default_host() -> str:
    return os.environ.get(ENV_HOST) or DEFAULT_HOST


def default_model() -> str:
    return os.environ.get(ENV_MODEL) or DEFAULT_MODEL


def is_ollama_running(host: str, *, timeout: float = 1.0, opener=None) -> bool:
    """Return True if an Ollama server answers at ``host``.

    ``opener`` defaults to :func:`urllib.request.urlopen` and is injectable so
    tests need no real network.
    """

    url = host.rstrip("/") + "/api/tags"
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(url, timeout=timeout):
            return True
    except Exception:
        return False


def _spawn_ollama_serve() -> "subprocess.Popen":
    """Start ``ollama serve`` detached, discarding its output."""

    devnull = subprocess.DEVNULL
    return subprocess.Popen(["ollama", "serve"], stdout=devnull, stderr=devnull)


def ensure_ollama_running(
    host: str,
    *,
    autostart: bool = True,
    wait_seconds: float = 15.0,
    poll_interval: float = 0.3,
) -> Optional["subprocess.Popen"]:
    """Make sure Ollama is up at ``host``, optionally starting it.

    Returns the spawned process if one was started, else ``None``. Raises
    :class:`RunError` if the server is unavailable and cannot be started.
    """

    if is_ollama_running(host):
        return None
    if not autostart:
        raise RunError(
            f"Ollama is not running at {host}. Start it with `ollama serve` "
            "(or drop --no-serve to start it automatically)."
        )
    if not command_exists("ollama"):
        raise RunError("Ollama is not installed. Run `python install.py` first.")

    print("Starting `ollama serve`...")
    proc = _spawn_ollama_serve()
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if is_ollama_running(host):
            return proc
        time.sleep(poll_interval)
    raise RunError(
        f"Started `ollama serve` but it was not reachable at {host} "
        f"within {wait_seconds:.0f}s."
    )


def build_codex_command(model: str, extra_args: Sequence[str]) -> List[str]:
    """Build the ``codex --oss`` command line for ``model`` plus pass-through args."""

    return ["codex", "--oss", "-m", model, *extra_args]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Launch the Codex agent against local Ollama models.",
    )
    parser.add_argument(
        "-m",
        "--model",
        help=f"Local model to use (default: ${ENV_MODEL} or {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--host",
        help=f"Ollama base URL (default: ${ENV_HOST} or {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--no-serve",
        dest="autostart",
        action="store_false",
        help="Do not auto-start `ollama serve` if it is not already running.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, extra = build_parser().parse_known_args(argv)
    # Drop a leading "--" separator used to pass flags through to codex.
    if extra and extra[0] == "--":
        extra = extra[1:]

    host = args.host or default_host()
    model = args.model or default_model()

    if not command_exists("codex"):
        print(
            "error: Codex CLI is not installed. Run `python install.py` first.",
            file=sys.stderr,
        )
        return 2

    try:
        ensure_ollama_running(host, autostart=args.autostart)
    except RunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    cmd = build_codex_command(model, extra)
    print(f"$ {' '.join(cmd)}")
    env = dict(os.environ)
    env[ENV_HOST] = host
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
