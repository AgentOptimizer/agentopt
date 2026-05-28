"""
A user-supplied Router that the daemon imports via ``--policy-module``
and that the Python client imports for its ``with`` block.

Contract (see docs/api/router.md):
    1. Define ``Router`` subclasses at module level.
    2. Implement ``_config_kwargs`` returning JSON-serializable __init__ kwargs.
    3. ``__init__`` must accept the same kwargs back.

Avoid stdlib stem collisions (``random.py``, ``json.py``, …) — the
daemon imports this file by stem.
"""

from __future__ import annotations

from typing import Optional

from agentopt import Router
from agentopt.routing import RouteContext


class LengthBasedRouter(Router):
    """Long prompts → big model, short prompts → small model."""

    def __init__(self, big: str, small: str, threshold_chars: int = 120) -> None:
        self.big = big
        self.small = small
        self.threshold_chars = threshold_chars

    def route(self, ctx: RouteContext) -> Optional[str]:
        messages = ctx.request_body.get("messages") or []
        total = sum(len(str(m.get("content", ""))) for m in messages)
        return self.big if total > self.threshold_chars else self.small

    def _config_kwargs(self) -> dict:
        return {
            "big": self.big,
            "small": self.small,
            "threshold_chars": self.threshold_chars,
        }
