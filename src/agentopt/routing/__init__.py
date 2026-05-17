"""agentopt.routing — per-call model routing policies."""

from .base import (
    RouteContext,
    RouteDecision,
    Router,
    get_active_router,
)
from .policies import RandomRouter

__all__ = [
    "Router",
    "RouteContext",
    "RouteDecision",
    "RandomRouter",
    "get_active_router",
]
