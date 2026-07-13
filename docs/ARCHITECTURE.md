# Architecture

This document describes the design of `codex-ollama-local` — both what exists today
and the target shape it is being built toward. It is written for contributors and
for LLM CLI agents extending the codebase.

## Goals

- A **local-first** coding CLI agent: no cloud API keys, everything runs against a
  local [Ollama](https://ollama.com) server.
- **Cross-platform**: identical behavior on Ubuntu (Linux), Windows, and macOS.
- **Small, dependency-light core** (`httpx` at runtime) that is easy to test
  fully offline.
- A clear path from "chat with a model" to "agent that edits code and runs tools".

## Layered design

```
┌─────────────────────────────────────────────┐
│ cli.py            argparse CLI (`col`)        │  presentation / I/O
│                   commands: models, chat      │
├─────────────────────────────────────────────┤
│ agent core        (planned) tool-use loop     │  orchestration
│                   conversation + tool calls    │
├─────────────────────────────────────────────┤
│ ollama_client.py  list_models(), chat()        │  Ollama API access
│                   OllamaError on failure        │
├─────────────────────────────────────────────┤
│ config.py         Config, config_dir()          │  configuration
│ types.py          Message, ChatChunk, ModelInfo │  shared data
└─────────────────────────────────────────────┘
                        │
                        ▼  HTTP (httpx)
              Local Ollama server (:11434)
```

Each layer depends only on the layers below it. The CLI never speaks HTTP directly;
it goes through `OllamaClient`. The agent core (planned) will sit between the CLI
and the client.

## Components (current)

### `types.py`
Immutable dataclasses with no third-party imports:
- `Message(role, content)` — a chat turn.
- `ChatChunk(content, done)` — one streamed fragment of a reply.
- `ModelInfo(name, size, modified_at)` — a locally available model.

### `config.py`
- `Config` — resolved settings: `host`, `model`, `connect_timeout`,
  `request_timeout`.
- `Config.load()` — merges sources by precedence:
  **CLI flags > env vars (`OLLAMA_HOST`, `CODEX_OLLAMA_MODEL`) > config file >
  defaults**. `env` and `path` are injectable for hermetic tests.
- `config_dir()` — the one place that branches on `sys.platform`
  (`%APPDATA%` / `~/Library/Application Support` / `$XDG_CONFIG_HOME`). All paths
  use `pathlib.Path`.

### `ollama_client.py`
- `OllamaClient` wraps an `httpx.Client`; the transport is injectable so tests use
  `httpx.MockTransport` and never touch the network.
- `list_models()` → `GET /api/tags`.
- `chat()` → `POST /api/chat`, streaming newline-delimited JSON into `ChatChunk`s.
- Every network failure is normalized to `OllamaError` with an actionable message.

### `cli.py`
- `argparse` with subcommands `models` and `chat`, plus global `--host`/`--model`.
- `main(argv=None) -> int` returns an exit code (no `sys.exit` inside) so it is
  directly testable; `__main__.py` and the `col` console script wrap it.

### Root scripts: `install.py` / `run.py`
Standalone bootstrap/launch scripts that sit outside the package and use **only the
standard library** (so `install.py` runs on a clean machine before anything is
installed):
- `install.py` — installs the **Codex CLI** (`npm i -g @openai/codex`) and the
  **Ollama** server, choosing the right command per OS (Ubuntu convenience script /
  Homebrew / winget). Idempotent: each tool is skipped if already on `PATH`. After
  installing Codex it verifies `codex` is reachable; `npm install -g` often writes to
  a directory that is not on `PATH`, so when that happens it prints npm's global bin
  directory and the `export PATH=...` line to fix it instead of reporting a hollow
  success.
- `run.py` — ensures Ollama is reachable (auto-starting `ollama serve` if needed),
  enumerates the installed models, and hands off to **`ollama launch codex`**,
  forwarding extra args to Codex. This is how the project realizes "Codex, but
  against local models".

Both keep side effects in small injectable functions; tests mock every subprocess
and HTTP call.

#### Why `run.py` delegates to `ollama launch codex`

The requirement is that Codex can use **every** model in the local Ollama, not just
one. Codex discovers non-default models through a *model catalog* (`model_catalog_json`
in `~/.codex`), whose JSON schema is strict, largely undocumented, and changes across
Codex releases (wrong enum values cause hard parse failures). Rather than generate and
maintain that file ourselves, `run.py` delegates to Ollama's first-party
`ollama launch codex` integration, which refreshes the catalog from `/api/tags` and
writes a Codex profile. Ollama keeps this integration current with both tools, so the
"all models available" behavior stays correct without us tracking Codex's internal
schema. `run.py`'s own responsibilities are the surrounding orchestration: server
readiness, model enumeration/reporting, default-model resolution, a safe `--dry-run`,
and clean error messages when a prerequisite is missing.

`run.py` flow:
1. Verify `ollama` is installed. Locate `codex`: check `PATH`, and — because
   `npm install -g` frequently installs outside `PATH` — also npm's global bin
   directory (`npm prefix -g`). If found only there, that directory is prepended to
   the launched process's `PATH`. If found nowhere (and launching), point at
   `install.py`.
2. Detect whether the Ollama build supports `ollama launch codex`.
3. Ensure the server is running (auto-start unless `--no-serve`; skipped on `--dry-run`).
4. List local models via `GET /api/tags` and report them.
5. Resolve a default model (`--model` > `$CODEX_OLLAMA_MODEL` > first installed when a
   model is required, e.g. headless `-y`).
6. Build and run `ollama launch codex [--model M] [-y] [--config] [-- <codex args>]`
   with `OLLAMA_HOST` set (and the discovered Codex dir on `PATH`, so both Ollama's
   own lookup and Codex itself can find the binary); `--dry-run` prints the command.

If the `ollama launch codex` integration is absent (older Ollama), step 6 falls back
to `codex --oss -m <model>` (default model `--model` > `$CODEX_OLLAMA_MODEL` > first
installed > `gpt-oss:20b`) — the pre-integration launch path — so `run.py` still starts
Codex instead of erroring. `--config-only` has no fallback and requires the integration.

## Data flow: `col chat "..."`

1. `main` parses args, builds a `Config` via `Config.load`.
2. An `OllamaClient` is opened as a context manager.
3. The prompt (arg or stdin) becomes a `Message`.
4. `client.chat()` streams `ChatChunk`s from `POST /api/chat`.
5. The CLI writes each chunk's `content` to stdout, flushing as it goes.
6. On any `OllamaError`, a friendly message is printed to stderr and exit code 2.

## Roadmap: the agentic loop (planned)

The next milestones turn the chat front-end into a coding agent:

1. **Conversation state** — a `Conversation` holding message history and a system
   prompt describing the agent's role.
2. **Tool registry** — declarative tools (read file, write/patch file, list dir,
   run shell command) exposed to the model. Tools return structured results fed
   back into the loop.
3. **Agent loop** — send history → parse tool calls from the model → execute →
   append results → repeat until the model produces a final answer.
4. **Approval / safety** — an approval policy gating file writes and shell
   execution (auto / prompt / read-only), with an audit log. Shell commands run
   cross-platform (no assumption of a POSIX shell).
5. **Sessions** — persist conversations under `config_dir()` for resuming work.

These layers slot in above `ollama_client.py` without changing it. Ollama's
tool-calling support (`tools` field on `/api/chat`) is the intended mechanism.

## Testing strategy

- Unit tests per module; the suite is fully offline (`httpx.MockTransport`).
- Platform-specific logic is parametrized over `sys.platform` so all three OS
  branches are covered on any host, in addition to the real 3-OS CI matrix.
