# codex-ollama-local

A standalone coding CLI agent backed by a **local [Ollama](https://ollama.com)
server**. No cloud API keys — the models run on your machine. Today it lists local
models and holds streaming chats; it is being built toward a full agentic coding
assistant (file edits, tool use, shell execution). See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design and roadmap.

Works on **Ubuntu (Linux), Windows, and macOS**.

## Quickstart

Two helper scripts at the repo root bootstrap and launch the toolchain — they use
only the Python standard library, so you can run them on a clean machine:

```bash
# 1. Install the Codex CLI and the Ollama server (skips whatever is already present)
python install.py

# 2. Pull one or more local models
ollama pull qwen2.5-coder:7b
ollama pull gpt-oss:20b

# 3. Launch Codex against your local models
python run.py
```

`install.py` is idempotent (re-running is safe); use `--dry-run` to preview the
commands or `--force` to reinstall. Codex is installed with `npm install -g`, whose
global bin directory is often not on `PATH` (common on Ubuntu); `install.py` detects
that and prints the exact `export PATH=...` line to fix it, and `run.py` locates Codex
there automatically so it still launches.

`run.py` starts `ollama serve` automatically if it is not already running, lists the
local models available, and then launches the real **OpenAI Codex CLI** in its local
open-source mode — `codex --oss -m <model>` — pointed at your Ollama server. `-m` sets
Codex's default model; you can still switch between installed models in-session with
`/model`. Useful flags:

```bash
python run.py                 # launch Codex against a local model
python run.py -m gpt-oss:20b  # set the default model (still switchable in-session)
python run.py --no-serve      # do not auto-start `ollama serve`
python run.py --dry-run       # show what would run, without changing anything
python run.py -- --sandbox workspace-write   # forward flags to Codex
```

## Requirements

- Python **3.11+**
- A running Ollama server with at least one model pulled:
  ```bash
  ollama serve
  ollama pull qwen2.5-coder:7b
  ```

## Install

```bash
pip install -e .
```

This installs the `col` command.

## Usage

```bash
# List models available locally in Ollama
col models

# One-shot chat (prompt as an argument)
col chat "write a hello world in python"

# Or pipe a prompt via stdin
echo "explain this error: NameError" | col chat

# Pick a model / server for a single run
col chat -m llama3.2:3b "hi"
col --host http://localhost:11434 models
```

You can also run it as a module: `python -m codex_ollama ...`.

## Configuration

Settings are resolved with this precedence (highest first):

1. CLI flags (`--host`, `--model` / `-m`)
2. Environment variables: `OLLAMA_HOST`, `CODEX_OLLAMA_MODEL`
3. A TOML config file in the per-user config directory
   (`%APPDATA%` on Windows, `~/Library/Application Support` on macOS,
   `$XDG_CONFIG_HOME` / `~/.config` on Linux), under `codex-ollama/config.toml`:
   ```toml
   host = "http://localhost:11434"
   model = "qwen2.5-coder:7b"
   connect_timeout = 5.0
   request_timeout = 120.0
   ```
4. Built-in defaults.

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md). In short:

```bash
pip install -e .[dev]
black --check . && flake8 && pytest
```

Guidance for LLM CLI agents lives in [`AGENTS.md`](AGENTS.md) and
[`CLAUDE.md`](CLAUDE.md).

## License

See [`LICENSE`](LICENSE).
