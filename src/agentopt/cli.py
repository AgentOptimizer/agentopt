"""``agentopt`` CLI — top-level argparse dispatcher.

Wired to ``pyproject.toml``'s ``[project.scripts]`` as
``agentopt = "agentopt.cli:main"``.

Each subcommand registers itself by adding a subparser and calling
``set_defaults(func=<handler>)``; :func:`main` dispatches via
``args.func(args)``.  This pattern keeps ``--help`` ergonomics
predictable at every level (``agentopt --help``,
``agentopt serve --help``) without the ``parse_known_args`` /
``add_help=False`` workarounds those break.
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
    subparsers = parser.add_subparsers(
        dest="cmd", required=True, metavar="COMMAND",
    )

    # Subcommands register themselves here.  Each must call
    # `set_defaults(func=<handler>)` so dispatch below stays generic.
    from .proxy.daemon import register_serve_subparser

    register_serve_subparser(subparsers)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
