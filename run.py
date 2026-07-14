#!/usr/bin/env python3
"""Start the OpenAI Codex CLI against the local Ollama server.

This launches the real Codex CLI in its local open-source mode
(``codex --oss -m <model>``), which points Codex straight at a local
[Ollama](https://ollama.com) server (``http://localhost:11434/v1``) and uses a
locally installed model -- no cloud API key required. This script wraps that
command with the ergonomics we want:

  * make sure the Ollama server is actually running (starting it if needed);
  * report which local models are available;
  * expose **all** local models in Codex's ``/model`` picker (see below);
  * pick a sensible default model to hand to Codex;
  * forward any extra arguments straight through to Codex.

``-m`` sets Codex's default model; Codex can still switch models in-session with
``/model``. In ``--oss`` mode Codex normally shows only its built-in model catalog
in ``/model`` (it skips the remote refresh), so your local Ollama models never
appear. To fix that we generate a *model catalog* -- Codex's own bundled catalog
(read via ``codex debug models --bundled``, so it stays schema-correct across Codex
versions) plus one cloned entry per local Ollama model -- and point Codex at it for
the run via ``-c model_catalog_json="..."``. Cloned entries are given a
locally-compatible capability profile (e.g. web search disabled) so Codex does not
emit request items the Ollama endpoint rejects. Pass ``--no-catalog`` to skip this
and launch plain ``codex --oss``.

Uses only the Python standard library. Cross-platform: Ubuntu, macOS, Windows.

Usage::

    python run.py                       # launch Codex; all local models in /model
    python run.py --gui                 # launch the Codex desktop app instead
    python run.py -m qwen2.5-coder:7b   # pick the default model
    python run.py --no-serve            # do not auto-start `ollama serve`
    python run.py --no-catalog          # do not customize Codex's /model catalog
    python run.py --dry-run             # show what would run, change nothing
    python run.py -- --sandbox workspace-write   # forward flags to Codex
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Sequence

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


def build_codex_command(
    model: str,
    extra_args: Sequence[str],
    *,
    catalog_path: Optional[str] = None,
) -> List[str]:
    """Build the ``codex --oss`` command that launches the Codex CLI.

    ``--oss`` puts Codex in its local open-source mode, talking to the Ollama
    server (via ``OLLAMA_HOST``); ``-m`` sets ``model`` as the default. When
    ``catalog_path`` is given, ``-c model_catalog_json="..."`` points Codex at our
    generated catalog so ``/model`` lists every local model. Extra args are
    forwarded to Codex unchanged.
    """

    cmd = ["codex", "--oss", "-m", model]
    if catalog_path:
        # The value is parsed as TOML, so a path must be a quoted string.
        cmd += ["-c", f'model_catalog_json="{catalog_path}"']
    cmd += list(extra_args)
    return cmd


# Platforms for which Codex ships the desktop app (`codex app` is compiled in only
# for macOS and Windows).
GUI_PLATFORMS = ("darwin", "win32")


def build_gui_command(extra_args: Sequence[str]) -> List[str]:
    """Build the ``codex app`` command that launches the Codex desktop GUI.

    ``codex app`` opens Codex Desktop for the current workspace (installing it
    first if it is missing). Unlike the CLI it accepts none of the ``--oss`` /
    ``-m`` / ``-c`` flags -- the desktop app has its own provider and model
    picker -- so we forward only the user's extra args (e.g. a workspace path or
    ``--download-url``) and nothing else.
    """

    return ["codex", "app"] + list(extra_args)


def resolve_model(cli_model: Optional[str], available: Sequence[str]) -> str:
    """Pick the default model to hand to Codex.

    Precedence: explicit ``--model`` > ``$CODEX_OLLAMA_MODEL`` > the first
    installed model > :data:`DEFAULT_MODEL`. Codex can still switch to any other
    installed model in-session.
    """

    return cli_model or env_model() or (available[0] if available else DEFAULT_MODEL)


def codex_home() -> pathlib.Path:
    """Codex's config directory (``$CODEX_HOME`` or ``~/.codex``)."""

    return pathlib.Path(
        os.environ.get("CODEX_HOME") or (pathlib.Path.home() / ".codex")
    )


def catalog_path() -> pathlib.Path:
    """Where we write the generated model catalog (inside Codex's home)."""

    return codex_home() / "col-ollama-catalog.json"


def bundled_model_catalog(runner=None, env=None) -> Optional[Dict]:
    """Return Codex's bundled model catalog as ``{"models": [...]}``.

    Read via ``codex debug models --bundled``, whose entries match the schema of
    the installed Codex version -- so cloning them stays valid across releases.
    Returns ``None`` on any failure (old Codex without the subcommand, non-zero
    exit, or unparseable output). ``runner``/``env`` are injectable for tests.
    """

    runner = runner or subprocess.run
    try:
        result = runner(
            ["codex", "debug", "models", "--bundled"],
            capture_output=True,
            text=True,
            env=env,
        )
    except OSError:
        return None
    if getattr(result, "returncode", 0):
        return None
    out = (getattr(result, "stdout", "") or "").strip()
    if not out:
        return None
    try:
        data = json.loads(out)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, list):
        return {"models": data}
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return data
    return None


# Capability flags that must be toned down when we retarget a bundled entry
# (typically a cloud model -- Codex's bundled catalog ships no gpt-oss entry) to
# a local Ollama model. Left as-is, a cloud template advertises features whose
# Responses-API request items the Ollama endpoint rejects with "unknown input
# item type". Two such items have bitten us:
#   * ``supports_search_tool=true`` makes Codex emit a ``web_search_call`` item.
#   * ``use_responses_lite=true`` makes Codex prepend an ``additional_tools``
#     item (the "Responses Lite" tool-passing shape) on every turn.
# Ollama understands neither, so it 400s each query. We force a locally-compatible
# profile: no web search, no Responses Lite, text-only input, no verbosity control.
# Only keys already present in the template are overwritten, and only with values
# that are safe across Codex versions: booleans, or a subset of a value already in
# the template (``input_modalities``). We deliberately do NOT rewrite enum-valued
# fields such as ``apply_patch_tool_type`` -- substituting a variant the installed
# Codex doesn't accept makes it reject the whole catalog (e.g. ``unknown variant
# `function`, expected `freeform```). :func:`catalog_accepted` validates that the
# file still parses, but note it cannot catch request-time item rejections like the
# above -- those are why we neutralize the capabilities here in the first place.
_LOCAL_MODEL_OVERRIDES: Dict[str, object] = {
    "supports_search_tool": False,
    "support_verbosity": False,
    "use_responses_lite": False,
    "input_modalities": ["text"],
}


def _catalog_template(entries: Sequence[Dict]) -> Optional[Dict]:
    """Pick a bundled entry to clone for local models.

    Prefer an open-source ("oss") entry -- its reasoning levels and capability
    flags suit locally served models -- otherwise fall back to the first entry.
    """

    for entry in entries:
        if isinstance(entry, dict) and "oss" in str(entry.get("slug", "")).lower():
            return entry
    return entries[0] if entries and isinstance(entries[0], dict) else None


def _localize_entry(entry: Dict, name: str) -> Dict:
    """Retarget a cloned template entry to the local Ollama model ``name``.

    Sets identity fields and forces a locally-compatible capability profile (see
    :data:`_LOCAL_MODEL_OVERRIDES`), mutating and returning ``entry``.
    """

    entry["slug"] = name
    entry["display_name"] = name
    entry["description"] = f"Local model served by Ollama ({name})."
    entry["visibility"] = "list"
    for key, value in _LOCAL_MODEL_OVERRIDES.items():
        if key in entry:
            entry[key] = copy.deepcopy(value)
    return entry


def build_model_catalog(models: Sequence[str], bundled: Dict) -> Dict:
    """Return a catalog of Codex's bundled models plus the local Ollama models.

    Each local model missing from the bundled catalog gets a deep copy of a
    template entry, retargeted via :func:`_localize_entry` so every
    (schema-required) field is present and valid *and* the capability profile
    won't make Codex emit request items the Ollama endpoint rejects. The catalog
    *replaces* Codex's model list, so we keep the bundled entries too -- cloud
    models stay available alongside the local ones.
    """

    entries = [e for e in bundled.get("models", []) if isinstance(e, dict)]
    template = _catalog_template(entries)
    if template is None:
        return {"models": entries}
    slugs = {e.get("slug") for e in entries}
    for name in models:
        if name in slugs:
            continue
        entry = _localize_entry(copy.deepcopy(template), name)
        entries.append(entry)
        slugs.add(name)
    return {"models": entries}


def write_model_catalog(catalog: Dict, path: pathlib.Path) -> None:
    """Write ``catalog`` as JSON to ``path``, creating parent dirs as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")


def catalog_accepted(path: str, *, runner=None, env=None) -> bool:
    """Return True if Codex loads the catalog at ``path`` without error.

    Codex's catalog schema is strict and drifts across versions; a value it does
    not accept makes it reject the *entire* catalog at startup (e.g. ``unknown
    variant `function```). We verify with ``codex debug models -c
    model_catalog_json="<path>"`` -- which loads config the same way a launch does
    -- and treat a non-zero exit (or the command being unavailable) as rejection,
    so the caller can fall back to launching without the catalog rather than
    handing Codex a config that makes it refuse to start. Injectable for tests.
    """

    runner = runner or subprocess.run
    try:
        result = runner(
            ["codex", "debug", "models", "-c", f'model_catalog_json="{path}"'],
            capture_output=True,
            text=True,
            env=env,
        )
    except OSError:
        return False
    return getattr(result, "returncode", 0) == 0


def prepare_model_catalog(models: Sequence[str], *, env=None) -> Optional[str]:
    """Generate the model catalog and return its path, or ``None`` on failure.

    Best-effort: prints a warning and returns ``None`` if Codex's bundled catalog
    cannot be read, the file cannot be written, or Codex rejects the result, so
    the caller can still launch plain ``codex --oss`` instead of failing.
    """

    bundled = bundled_model_catalog(env=env)
    if not (bundled and bundled.get("models")):
        print(
            "warning: could not read Codex's bundled catalog "
            "(`codex debug models --bundled`); /model will show only Codex's "
            "built-in models, not your local Ollama models.",
            file=sys.stderr,
        )
        return None
    catalog = build_model_catalog(models, bundled)
    path = catalog_path()
    try:
        write_model_catalog(catalog, path)
    except OSError as exc:
        print(
            f"warning: could not write model catalog to {path}: {exc}", file=sys.stderr
        )
        return None
    if not catalog_accepted(str(path), env=env):
        print(
            f"warning: Codex rejected the generated model catalog ({path}); "
            "launching without it, so your local models won't appear in /model. "
            "This usually means Codex's catalog schema changed -- please report it.",
            file=sys.stderr,
        )
        return None
    print(f"Exposing {len(models)} local model(s) to Codex's /model via {path}.")
    return str(path)


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
        "--gui",
        action="store_true",
        help="Launch the Codex desktop app (`codex app`) instead of the CLI. The "
        "GUI manages its own model selection, so -m/--no-catalog are ignored; "
        "macOS/Windows only.",
    )
    parser.add_argument(
        "--no-serve",
        dest="autostart",
        action="store_false",
        help="Do not auto-start `ollama serve` if it is not already running.",
    )
    parser.add_argument(
        "--no-catalog",
        dest="catalog",
        action="store_false",
        help="Do not customize Codex's /model catalog; launch plain `codex --oss`.",
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

    env = dict(os.environ)
    env[ENV_HOST] = host
    if codex_dir:
        env["PATH"] = codex_dir + os.pathsep + env.get("PATH", "")

    # The desktop GUI is a distinct Codex surface. `codex app` opens Codex Desktop
    # and takes none of the CLI's --oss/-m/-c flags (it has its own model picker),
    # so we skip catalog generation and default-model selection and just launch it.
    # Codex ships the app for macOS and Windows only; warn elsewhere but still try,
    # so Codex itself is the source of truth on availability. The local server is
    # already up (above), which is what the app needs to talk to Ollama.
    if args.gui:
        if sys.platform not in GUI_PLATFORMS:
            print(
                "warning: Codex Desktop (`codex app`) is only available on macOS "
                "and Windows; it may not launch on this platform.",
                file=sys.stderr,
            )
        cmd = build_gui_command(extra)
        print(f"$ {' '.join(cmd)}")
        if args.dry_run:
            return 0
        return subprocess.run(cmd, env=env).returncode

    model = resolve_model(args.model, models)

    # Expose every local model in Codex's /model picker by generating a catalog
    # (skipped when opted out, on --dry-run, or when there are no local models).
    catalog_arg: Optional[str] = None
    if args.catalog and models:
        if args.dry_run:
            catalog_arg = str(catalog_path())
        else:
            catalog_arg = prepare_model_catalog(models, env=env)

    cmd = build_codex_command(model, extra, catalog_path=catalog_arg)

    print(f"$ {' '.join(cmd)}")
    if args.dry_run:
        if catalog_arg:
            print(
                f"note: a model catalog exposing all {len(models)} local model(s) "
                f"would be generated at {catalog_arg}.",
                file=sys.stderr,
            )
        return 0

    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
