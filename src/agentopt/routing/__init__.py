"""agentopt.routing — per-call model routing policies.

Top-level convenience imports re-export ``Router`` and ``RandomRouter``
on the ``agentopt`` package itself.  ``RouteContext`` (the type of the
``ctx`` parameter on :meth:`Router.route`) lives here for users who
want type annotations on their custom routers but isn't exported at
the top level.
"""

from .base import RouteContext, Router
from .config import BUILTIN_POLICIES, load_policy_module, resolve_policy
from .random_policy import RandomRouter
from .summary import format_routing_summary, print_routing_summary

__all__ = [
    "Router",
    "RouteContext",
    "RandomRouter",
    "BUILTIN_POLICIES",
    "format_routing_summary",
    "load_policy_module",
    "print_routing_summary",
    "resolve_policy",
]
