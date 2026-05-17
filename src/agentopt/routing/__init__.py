"""agentopt.routing — per-call model routing policies.

Top-level convenience imports re-export ``Router`` and ``RandomRouter``
on the ``agentopt`` package itself.  ``RouteContext`` (the type of the
``ctx`` parameter on :meth:`Router.route`) lives here for users who
want type annotations on their custom routers but isn't exported at
the top level.
"""

from .base import (
    RouteContext,
    Router,
    get_active_router,
)
from .config import BUILTIN_POLICIES, resolve_policy
from .random_policy import RandomRouter

__all__ = [
    "Router",
    "RouteContext",
    "RandomRouter",
    "BUILTIN_POLICIES",
    "get_active_router",
    "resolve_policy",
]
