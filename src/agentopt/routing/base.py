"""Routing abstractions: :class:`Router` base class plus ``RouteContext``
and ``RouteDecision`` dataclasses.

A *router* is a per-call policy that decides which model to actually send
a request to.  It sees the parsed request body, session metadata, and
prior LLM calls in the session, and returns either a
:class:`RouteDecision` (swap the model) or ``None`` (keep the client's
choice).

Public API shape::

    router = RandomRouter(model_candidates=["gpt-4o", "gpt-4o-mini"])
    with router:
        answer = agent.run(question)

Scope note (v1): same-provider only.  :class:`RouteDecision` carries
``provider`` / ``api_key`` fields for future cross-provider routing; the
proxy raises ``NotImplementedError`` if a router actually sets them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from agentopt.proxy.models import CallRecord


@dataclass(frozen=True)
class RouteContext:
    """Per-call context passed to :meth:`Router.route`."""

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
    raise ``NotImplementedError`` at dispatch — those fields are reserved
    for future cross-provider routing so the API can grow without breaks.
    """

    model: str
    provider: Optional[str] = None
    api_key: Optional[str] = None


class Router:
    """Base class for per-call model routing policies.

    Subclasses implement :meth:`route`.  Use the instance as a context
    manager to scope routing to a block of agent code::

        router = MyRouter(...)
        with router:
            answer = agent.run(question)

    Any LLM HTTP call made inside the ``with`` block is transparently
    run through the policy — no framework integration required.

    **Not re-entrant on a single instance.**  If you need nested or
    concurrent scopes, instantiate a separate router per scope.
    """

    def route(self, ctx: RouteContext) -> Optional[RouteDecision]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Context manager — the public "activate this router" API.
    # ------------------------------------------------------------------

    def __enter__(self) -> "Router":
        from ._runtime import _activate

        if getattr(self, "_routing_handle", None) is not None:
            raise RuntimeError(
                "Router is already active — `with router:` is not "
                "re-entrant on the same instance"
            )
        self._routing_handle = _activate(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        from ._runtime import _deactivate

        handle = getattr(self, "_routing_handle", None)
        if handle is None:
            return
        self._routing_handle = None
        _deactivate(handle)
