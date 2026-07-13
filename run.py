#!/usr/bin/env python3
"""Start the OpenAI Codex CLI against the local Ollama server.

This launches the real Codex CLI in its local open-source mode
(``codex --oss -m <model>``), which points Codex straight at a local
[Ollama](https://ollama.com) server (``http://localhost:11434/v1``) and uses a
locally installed model -- no cloud API key required. This script wraps that
command with the ergonomics we want:

  * make sure the Ollama server is actually running (starting it if needed);
  * report which local models are available;
  * pick a sensible default model to hand to Codex;
  * forward any extra arguments straight through to Codex.

``-m`` sets Codex's default model; Codex can still switch models in-session with
``/model``.

Uses only the Python standard library. Cross-platform: Ubuntu, macOS, Windows.

Usage::

    python run.py                       # launch Codex against a local model
    python run.py -m qwen2.5-coder:7b   # pick the default model
    python run.py --no-serve            # do not auto-start `ollama serve`
    python run.py --dry-run             # show what would run, change nothing
    python run.py -- --sandbox workspace-write   # forward flags to Codex
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import List, Optional, Sequence

DEFAULT_HOST = "http://localhost:11434"
# Codex's own default OSS model; used when nothing else is specified and no
# local models are installed yet.
DEFAULT_MODEL = "gpt-oss:20b"
ENV_HOST = "OLLAMA_HOST"
ENV_MODEL = "CODEX_OLLAMA_MODEL"


class RunError(RuntimeError):
    """Raised when the agent cannot be started."""


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def npm_global_bin_dir(runner=None) -> Optional[str]:
    """Best-effort path to the directory npm installs global binaries into.

    ``npm install -g`` frequently lands binaries in a directory that is not on
    ``PATH`` (a common Ubuntu setup), which is why a "successful" install can
    still leave ``codex`` unreachable. We query ``npm prefix -g`` to recover that
    location. Returns ``None`` on any failure. ``runner`` is injectable for tests.
    """

    runner = runner or subprocess.run
    if not command_exists("npm"):
        return None
    try:
        result = runner(["npm", "prefix", "-g"], capture_output=True, text=True)
    except OSError:
        return None
    prefix = (getattr(result, "stdout", "") or "").strip()
    if not prefix:
        return None
    # POSIX npm puts executables in <prefix>/bin; on Windows they sit in <prefix>.
    base = pathlib.Path(prefix)
    return str(base if os.name == "nt" else base / "bin")


def find_codex_dir(bin_dir=None) -> Optional[str]:
    """Return a directory to add to ``PATH`` so ``codex`` becomes reachable.

    ``None`` means nothing needs to change: either ``codex`` is already on ``PATH``
    or it could not be located. When ``codex`` is missing from ``PATH`` but present
    in npm's global bin directory, that directory is returned so the caller can
    prepend it to the launched process's ``PATH``. ``bin_dir`` is injectable for
    tests; by default it is discovered via :func:`npm_global_bin_dir`.
    """

    if command_exists("codex"):
        return None
    bin_dir = bin_dir if bin_dir is not None else npm_global_bin_dir()
    if not bin_dir:
        return None
    directory = pathlib.Path(bin_dir)
    for name in ("codex", "codex.cmd", "codex.exe"):
        if (directory / name).exists():
            return str(directory)
    return None


def default_host() -> str:
    return os.environ.get(ENV_HOST) or DEFAULT_HOST


def env_model() -> Optional[str]:
    """The model requested via the environment, if any."""

    return os.environ.get(ENV_MODEL) or None


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


def list_ollama_models(host: str, *, timeout: float = 2.0, opener=None) -> List[str]:
    """Return the names of all models installed in the local Ollama server.

    Best-effort and read-only: any error (server down, bad payload) yields an
    empty list rather than raising. ``opener`` is injectable for tests.
    """

    url = host.rstrip("/") + "/api/tags"
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    models = payload.get("models", []) if isinstance(payload, dict) else []
    names = [m.get("name", "") for m in models if isinstance(m, dict)]
    return [name for name in names if name]


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
    """Build the ``codex --oss`` command that launches the Codex CLI.

    ``--oss`` puts Codex in its local open-source mode, talking to the Ollama
    server (via ``OLLAMA_HOST``); ``-m`` sets ``model`` as the default. Extra
    args are forwarded to Codex unchanged.
    """

    return ["codex", "--oss", "-m", model, *list(extra_args)]


def resolve_model(cli_model: Optional[str], available: Sequence[str]) -> str:
    """Pick the default model to hand to Codex.

    Precedence: explicit ``--model`` > ``$CODEX_OLLAMA_MODEL`` > the first
    installed model > :data:`DEFAULT_MODEL`. Codex can still switch to any other
    installed model in-session.
    """

    return cli_model or env_model() or (available[0] if available else DEFAULT_MODEL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Launch the OpenAI Codex CLI against local Ollama models.",
    )
    parser.add_argument(
        "-m",
        "--model",
        help=f"Default model (also reads ${ENV_MODEL}). Codex can still switch "
        "between installed models in-session with /model.",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without changing anything or launching Codex.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, extra = build_parser().parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    host = args.host or default_host()

    if not command_exists("ollama"):
        print(
            "error: Ollama is not installed. Run `python install.py` first.",
            file=sys.stderr,
        )
        return 2

    # Locate Codex. `npm install -g @openai/codex` often lands the binary in a
    # directory that is not on PATH (common on Ubuntu), so a "successful" install
    # can still leave `codex` unreachable. Look in npm's global bin too, and if we
    # find it there, remember the dir to prepend to the launched process's PATH.
    codex_dir = None if args.dry_run else find_codex_dir()
    if not args.dry_run and not (command_exists("codex") or codex_dir):
        print(
            "error: Codex CLI not found. `python install.py` may have installed it "
            "into a directory that is not on your PATH. Install it with "
            "`npm install -g @openai/codex`, then ensure npm's global bin directory "
            "(see `npm prefix -g`) is on PATH.",
            file=sys.stderr,
        )
        return 2
    if codex_dir:
        print(
            f"note: found Codex in {codex_dir} (not on PATH); "
            "adding it to PATH for this run.",
            file=sys.stderr,
        )

    # Make sure the server is up (skipped on --dry-run to avoid side effects).
    if not args.dry_run:
        try:
            ensure_ollama_running(host, autostart=args.autostart)
        except RunError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    models = list_ollama_models(host)
    if models:
        print(f"Codex can use {len(models)} local model(s):")
        for name in models:
            print(f"  - {name}")
    else:
        print(
            "warning: no local models found. Pull one with "
            "`ollama pull qwen2.5-coder:7b` so Codex has something to use.",
            file=sys.stderr,
        )

    model = resolve_model(args.model, models)
    cmd = build_codex_command(model, extra)

    print(f"$ {' '.join(cmd)}")
    if args.dry_run:
        return 0

    env = dict(os.environ)
    env[ENV_HOST] = host
    if codex_dir:
        env["PATH"] = codex_dir + os.pathsep + env.get("PATH", "")
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
