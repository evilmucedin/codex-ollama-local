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
  enumerates the installed models, exposes them all in Codex's `/model` picker via a
  generated model catalog (see below), and launches the real **OpenAI Codex CLI** in
  its local open-source mode (`codex --oss -m <model>`), forwarding extra args to
  Codex. This is how the project realizes "Codex, but against local models".

Both keep side effects in small injectable functions; tests mock every subprocess
and HTTP call.

#### Why `run.py` launches `codex --oss`

The whole point of the project is to run the genuine OpenAI Codex CLI without a cloud
API key. Codex ships a first-party local mode for exactly this: `codex --oss` points
Codex at a local Ollama server (`http://localhost:11434/v1`) and uses a locally
installed model, and `-m` sets the default (Codex can still switch models in-session
with `/model`). `run.py` invokes that directly rather than going through any Ollama-side
integration, so behavior does not depend on which Ollama build is installed. `run.py`'s
responsibilities are the surrounding orchestration: server readiness, model
enumeration/reporting, default-model resolution, a safe `--dry-run`, and clean error
messages when a prerequisite is missing.

`run.py` flow:
1. Verify `ollama` is installed. Locate `codex`: check `PATH`, and — because
   `npm install -g` frequently installs outside `PATH` — also npm's global bin
   directory (`npm prefix -g`). If found only there, that directory is prepended to
   the launched process's `PATH`. If found nowhere, point at `install.py`.
2. Ensure the server is running (auto-start unless `--no-serve`; skipped on `--dry-run`).
3. List local models via `GET /api/tags` and report them.
4. Resolve a default model (`--model` > `$CODEX_OLLAMA_MODEL` > first installed >
   `gpt-oss:20b`).
5. Generate a *model catalog* so every local model is selectable in Codex's `/model`
   (unless `--no-catalog`, `--dry-run`, or there are no local models). See below.
6. Build and run `codex --oss -m <model> [-c model_catalog_json="…"] [-- <codex args>]`
   with `OLLAMA_HOST` set (and the discovered Codex dir on `PATH` so Codex can be
   found); `--dry-run` prints the command instead of running it.

`--gui` branches at step 6: instead of the CLI it launches the Codex desktop app with
`codex app [-- <args>]` (still with `OLLAMA_HOST` set and the server ensured). The
desktop app is a distinct Codex surface — it has its own model picker and accepts none
of the `--oss`/`-m`/`-c` flags — so steps 4–5 (default-model resolution and catalog
generation) are skipped in GUI mode. Codex compiles `codex app` in for macOS and
Windows only (`#[cfg(any(target_os = "macos", target_os = "windows"))]`); on other
platforms `run.py` warns but still invokes it, letting Codex report availability.

#### Exposing all local models in `/model`

In `--oss` mode Codex skips its remote catalog refresh and resolves models from just
two layers: the catalog *bundled* into the binary, and an optional *local override*
file named by `model_catalog_json`. The bundled catalog is cloud models only, so a
plain `codex --oss` shows none of your Ollama models in `/model`.

`run.py` builds the override. It reads Codex's bundled catalog with `codex debug models
--bundled` (native JSON), then — because the catalog schema is strict and enum-checked
and drifts across Codex releases — *clones* a real bundled entry (preferring an "oss"
one) for each installed Ollama model, retargeting `slug`/`display_name`/`description`/
`visibility`. That keeps every schema-required field valid for the installed Codex
version without us hard-coding the schema. The override *replaces* the model list, so
the bundled entries are carried through too (cloud models stay selectable alongside
local ones). The result is written to `$CODEX_HOME/col-ollama-catalog.json` and passed
for the single run via `-c model_catalog_json="…"`, leaving the user's `config.toml`
untouched.

Codex's bundled catalog currently ships **only cloud models** (no `gpt-oss` entry), so
the template we clone is typically a cloud entry that advertises capabilities — web
search, Responses Lite, image input, verbosity — whose Responses-API request items a
local Ollama endpoint rejects with `unknown input item type`. Two have bitten us in
practice: `supports_search_tool=true` makes Codex emit a `web_search_call` item (see
Codex issue #24612), and `use_responses_lite=true` makes it prepend an
`additional_tools` item on every turn. So when retargeting a clone to a local model,
`run.py` forces a locally-compatible capability profile (`supports_search_tool=false`,
`use_responses_lite=false`, `input_modalities=["text"]`, `support_verbosity=false`).

Two rules keep the override schema-valid across Codex releases. First, `run.py` only
overwrites keys that already exist in the template — introducing an unknown key would
make Codex discard the entire catalog. Second, it only substitutes *safe* values:
booleans, or a subset of an existing list. It deliberately does **not** rewrite
enum-valued fields such as `apply_patch_tool_type`, because a guessed variant that the
installed Codex doesn't accept makes it reject the whole file (`unknown variant
"function", expected "freeform"`); the cloned template already carries a valid value.

This is best-effort and never fatal. As a backstop against any remaining schema drift,
`run.py` validates the generated catalog by asking Codex to parse it (`catalog_accepted`
runs `codex debug models -c model_catalog_json="…"` and checks the exit code); if Codex
rejects it — or if `codex debug models` is unavailable (older Codex) or the file can't
be written — `run.py` warns and launches plain `codex --oss`.

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
