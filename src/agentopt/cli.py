"""``agentopt`` CLI — currently dispatches a single subcommand: ``serve``.

Wired to ``pyproject.toml``'s ``[project.scripts]`` table as
``agentopt = "agentopt.cli:main"``.  Run ``agentopt serve --help`` for
daemon options.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="agentopt",
        description="agentopt — LLM model selection + per-call tracking",
    )
    subs = parser.add_subparsers(dest="cmd", required=True)
    subs.add_parser(
        "serve",
        help="Run the long-lived gateway daemon. See `agentopt serve --help`.",
        add_help=False,
    )
    # Parse only the first positional so subcommands can own their own flags.
    args, rest = parser.parse_known_args(argv)

    if args.cmd == "serve":
        from .proxy.daemon import cli as serve_cli

        serve_cli(rest)
        return

    parser.error(f"unknown subcommand: {args.cmd!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
