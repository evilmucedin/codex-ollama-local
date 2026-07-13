"""Command-line interface for codex-ollama-local.

Built on :mod:`argparse` (stdlib) so there are no runtime dependencies beyond
``httpx``. The entry point is :func:`main`, which returns an integer exit code
and never calls :func:`sys.exit` directly, making it straightforward to test.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from . import __version__
from .config import Config
from .ollama_client import OllamaClient, OllamaError
from .types import Message

PROG = "col"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="A local coding CLI agent backed by Ollama.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--host",
        help="Ollama base URL (overrides OLLAMA_HOST and the config file).",
    )
    parser.add_argument(
        "--model",
        help="Model name (overrides CODEX_OLLAMA_MODEL and the config file).",
    )

    sub = parser.add_subparsers(dest="command", metavar="command")

    sub.add_parser("models", help="List models available locally in Ollama.")

    chat = sub.add_parser("chat", help="Send a prompt and stream the reply.")
    chat.add_argument(
        "prompt",
        nargs="?",
        help="Prompt text. If omitted, it is read from standard input.",
    )
    chat.add_argument(
        "-m",
        "--model",
        dest="chat_model",
        help="Model to use for this chat (overrides the global --model).",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entry point. Returns a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    config = Config.load(host=args.host, model=args.model)

    try:
        with OllamaClient(config) as client:
            if args.command == "models":
                return _cmd_models(client)
            if args.command == "chat":
                return _cmd_chat(client, args)
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Unreachable given the subparser set, but keeps type checkers happy.
    parser.print_help()
    return 1


def _cmd_models(client: OllamaClient) -> int:
    models = client.list_models()
    if not models:
        print("No models found. Pull one with `ollama pull <model>`.")
        return 0
    for model in models:
        print(model.name)
    return 0


def _cmd_chat(client: OllamaClient, args: argparse.Namespace) -> int:
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    prompt = prompt.strip()
    if not prompt:
        print("error: empty prompt", file=sys.stderr)
        return 2

    model = getattr(args, "chat_model", None)
    messages = [Message(role="user", content=prompt)]
    for chunk in client.chat(messages, model=model):
        if chunk.content:
            sys.stdout.write(chunk.content)
            sys.stdout.flush()
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
