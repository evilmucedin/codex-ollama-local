# Contributing

Thanks for your interest in `codex-ollama-local`! This guide covers local setup and
the checks your change needs to pass.

## Prerequisites

- Python **3.11+**
- (Optional, only to run the agent — not the tests) a local
  [Ollama](https://ollama.com) install: `ollama serve`

## Setup

```bash
git clone https://github.com/evilmucedin/codex-ollama-local.git
cd codex-ollama-local
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Checks

Run all three before pushing — CI runs them on Linux, Windows, and macOS:

```bash
black .            # format (use --check in CI)
flake8             # lint
pytest             # tests (offline, no Ollama required)
```

## Guidelines

- Keep changes focused and well-tested; the test suite must remain **offline**
  (mock HTTP with `httpx.MockTransport`).
- Preserve cross-platform behavior: use `pathlib`, avoid POSIX-only assumptions.
- Follow the conventions in [`AGENTS.md`](AGENTS.md) and the design in
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Pull requests

Describe what changed and why, note the tests you added, and confirm the three
checks pass. Please make sure CI is green on all supported operating systems.
