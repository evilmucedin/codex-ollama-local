# Architecture

This document describes the design of `codex-ollama-local` — both what exists today
and the target shape it is being built toward. It is written for contributors and
for LLM CLI agents extending the codebase.

## Goals

- A **local-first** coding CLI agent: no cloud API keys, everything runs against a
  local [Ollama](https://ollama.com) server.
- **Cross-platform**: identical behavior on Linux, Windows, and macOS.
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
