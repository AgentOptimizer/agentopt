"""Routing abstractions: ``Router`` protocol plus ``RouteContext`` /
``RouteDecision`` dataclasses.

A *router* is a per-call policy that decides which model to actually send
a request to.  It sees the parsed request body, session metadata, and
prior LLM calls in the session; it returns either a
:class:`RouteDecision` (swap the model) or ``None`` (keep the client's
choice).

Scope note (v1): same-provider only.  ``RouteDecision`` carries
``provider`` / ``api_key`` fields for future cross-provider routing, but
the proxy raises ``NotImplementedError`` if a router actually sets them.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence

from agentopt.proxy.models import CallRecord


@dataclass(frozen=True)
class RouteContext:
    """Per-call context passed to :meth:`Router.route`.

    ``request_body`` is the parsed inbound JSON; treat it as read-only.
    ``history`` is a snapshot of prior ``CallRecord``\\ s in the current
    session (chronological).  The three ``session_*`` fields come from
    ``tracker.track(data_id=..., combo_id=..., agent_id=...)``.
    """

    request_body: Dict[str, Any]
    provider: str
    requested_model: Optional[str]
    session_data_id: Optional[str]
    session_combo_id: Optional[str]
    session_agent_id: Optional[str]
    history: Sequence[CallRecord]


@dataclass(frozen=True)
class RouteDecision:
    """Return value from :meth:`Router.route`.

    v1 only honours ``model``.  Setting ``provider`` or ``api_key`` will
    raise ``NotImplementedError`` in the proxy — those fields are reserved
    for future cross-provider routing so the API can grow without breaks.
    """

    model: str
    provider: Optional[str] = None
    api_key: Optional[str] = None


class Router(Protocol):
    """Per-call model routing policy."""

    def route(self, ctx: RouteContext) -> Optional[RouteDecision]:
        """Return a :class:`RouteDecision` to swap the model, or ``None``."""
        ...
