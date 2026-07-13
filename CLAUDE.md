# CLAUDE.md

This file orients Claude Code (and other LLM CLI agents) when working in this
repository.

**The canonical agent guide is [`AGENTS.md`](AGENTS.md).** Read it first — it
covers the project's purpose, layout, setup, the lint/format/test commands, coding
conventions, and PR expectations. This file intentionally stays short to avoid
drift between the two.

## Quick reference

- Purpose: a standalone coding CLI agent (`col`) backed by a local Ollama server.
- Setup: `pip install -e .[dev]` (Python 3.11+).
- Before committing, always run: `black --check . && flake8 && pytest`.
- Tests are offline — never require a live Ollama or network access.
- Everything must work on Ubuntu, Windows, and macOS; use `pathlib`, not raw paths.

For anything beyond this, defer to [`AGENTS.md`](AGENTS.md) and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
