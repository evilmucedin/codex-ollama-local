# AGENTS.md

Guidance for LLM CLI agents (Codex, Claude Code, and similar) working in this
repository. This is the canonical source of truth; `CLAUDE.md` points here.

## What this project is

`codex-ollama-local` is a **standalone coding CLI agent** that talks directly to a
**local [Ollama](https://ollama.com) server**. It runs an interactive/one-shot
chat today and is designed to grow into a full agentic coding assistant (tool use,
file edits, shell execution) — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

The console command is `col`.

## Repository layout

```
install.py             Bootstrap script: installs Codex CLI + Ollama (stdlib only)
run.py                 Launcher: starts `codex --oss` against local Ollama (stdlib only)
src/codex_ollama/      Package source (src layout)
  cli.py               argparse CLI, entry point `main`
  config.py            Config dataclass + cross-platform config dir
  ollama_client.py     httpx-based Ollama client (list_models, chat)
  types.py             Message / ChatChunk / ModelInfo dataclasses
tests/                 pytest suite (offline; httpx.MockTransport)
docs/ARCHITECTURE.md   Design and roadmap
.github/workflows/     CI (lint + tests on Ubuntu/Windows/macOS)
```

### Root scripts (`install.py`, `run.py`)

These are **standard-library only** on purpose: `install.py` must run on a clean
machine before the package (or its deps) exist, so do not import `codex_ollama` or
third-party libraries from them. Keep the logic in small, mockable functions (see
how `subprocess`/network calls are injected) — tests load these scripts by path via
the `install_mod` / `run_mod` fixtures in `tests/conftest.py` and mock every
subprocess and network call, so the suite never installs anything or hits the
network.

## Environment setup

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
pip install -e .[dev]
```

Requires **Python 3.11+**. Runtime dependency: `httpx`. Running the agent (not the
tests) additionally needs a local Ollama (`ollama serve`).

## Build / lint / test — run before every commit

```bash
black --check .      # formatting (line length 88)
flake8               # lint
pytest               # tests (no network / no Ollama needed)
```

`black .` (without `--check`) auto-formats. All three must pass; CI enforces them
on all three operating systems.

## Coding conventions

- **Formatting/lint:** black + flake8, 88-column lines. Keep them green.
- **Type hints** on all public functions; `from __future__ import annotations` at
  the top of every module.
- **Cross-platform is a hard requirement.** Use `pathlib.Path`, never hard-coded
  `/` or `\` separators. Branch on `sys.platform` only in one place
  (`config.config_dir`). Do not assume a POSIX shell or Unix-only APIs.
- **Testability:** keep side effects (network, stdout, filesystem) injectable.
  `OllamaClient` accepts a `transport`/`client`; `cli.main` takes `argv` and
  returns an exit code instead of calling `sys.exit`.
- **Errors:** wrap network failures in `OllamaError` with an actionable message.

## Testing expectations

- Every new module or behavior ships with tests. The suite must stay **offline** —
  mock HTTP with `httpx.MockTransport` (see `tests/conftest.py`).
- Cover success paths, error paths, and platform branches (parametrize over
  `sys.platform` rather than skipping on the host OS).

## Commit & PR conventions

- Small, focused commits; imperative subject lines (e.g. "Add chat streaming").
- Ensure `black --check . && flake8 && pytest` pass before pushing.
- PRs should describe the change, note test coverage, and confirm cross-platform
  safety. CI must be green on Ubuntu, Windows, and macOS.
