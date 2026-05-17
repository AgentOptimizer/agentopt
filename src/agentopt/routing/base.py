"""Routing abstractions: :class:`Router`, :class:`RouteContext`, :class:`RouteDecision`.

A *router* is a per-call policy that decides which model to actually
send a request to.  It sees the parsed request body, session metadata,
and prior LLM calls in the session, and returns either a
:class:`RouteDecision` (swap the model) or ``None`` (keep the client's
choice).

Public API shape::

    from agentopt import RandomRouter

    router = RandomRouter(candidates=["gpt-4o", "gpt-4o-mini"])
    with tracker.track(data_id="dp", combo_id="c", router=router):
        agent.run(question)

    # Or, equivalently, scope the router around a track() block:
    with router:
        with tracker.track(data_id="dp", combo_id="c"):
            agent.run(question)

The swap happens transparently at the HTTP layer (in-process httpx
patch + per-session mitmproxy addon), so any framework or subprocess
agent works without integration code.

**Scope (v1):** same-provider routing only.  ``RouteDecision`` carries
``provider`` / ``api_key`` fields for future cross-provider routing;
setting either today raises ``NotImplementedError`` at dispatch so the
API can grow without breaks.  Routing is library-only: passing a router
through ``LLMTracker`` in remote (daemon) mode raises a clear error.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

if TYPE_CHECKING:
    from agentopt.proxy.models import CallRecord
    from agentopt.proxy.session import SessionInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Route data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteContext:
    """Per-call context passed to :meth:`Router.route`.

    All fields are read-only by contract.  The router may inspect them
    but must not mutate the request body — return a
    :class:`RouteDecision` to express any desired change.
    """

    request_body: Dict[str, Any]
    provider: str  # "openai" | "anthropic" | "google" | ... | "unknown"
    requested_model: Optional[str]
    session: "SessionInfo"
    history: Sequence["CallRecord"]


@dataclass(frozen=True)
class RouteDecision:
    """Return value from :meth:`Router.route`.

    v1 only honours ``model``.  Setting ``provider`` or ``api_key``
    raises ``NotImplementedError`` at dispatch — reserved for future
    cross-provider routing so the API can grow without breaks.
    """

    model: str
    provider: Optional[str] = None
    api_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Activation — ContextVar for `with router:` sugar
# ---------------------------------------------------------------------------


_active_router_var: ContextVar[Optional["Router"]] = ContextVar(
    "agentopt_active_router", default=None,
)


def get_active_router() -> Optional["Router"]:
    """Return the router activated by the enclosing ``with router:`` block.

    Read by ``LLMTracker.track()`` when no explicit ``router=`` argument
    is supplied.  Returns ``None`` when no router is active.
    """
    return _active_router_var.get()


# ---------------------------------------------------------------------------
# Router base class
# ---------------------------------------------------------------------------


class Router:
    """Base class for per-call model routing policies.

    Subclasses implement :meth:`route`.  Use the instance as a context
    manager to scope routing to a block of code::

        router = MyRouter(...)
        with router:
            with tracker.track(...):
                agent.run(question)

    Inside the ``with`` block, any ``tracker.track()`` that does *not*
    pass an explicit ``router=`` argument picks this router up via a
    ``ContextVar``.

    **Not re-entrant on a single instance.**  If you need nested or
    concurrent scopes, instantiate a separate router per scope.
    """

    def route(self, ctx: RouteContext) -> Optional[RouteDecision]:
        """Decide which model to use for *ctx*.

        Return a :class:`RouteDecision` to swap the model, or ``None``
        to keep the client's requested model unchanged.

        Exceptions raised here are caught by the dispatcher and logged;
        the request proceeds unrouted.  A router should never break an
        agent.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Context manager — the "activate this router" API
    # ------------------------------------------------------------------

    def __enter__(self) -> "Router":
        if getattr(self, "_routing_token", None) is not None:
            raise RuntimeError(
                "Router is already active — `with router:` is not "
                "re-entrant on the same instance.  Instantiate a fresh "
                "Router for nested scopes."
            )
        self._routing_token = _active_router_var.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        token = getattr(self, "_routing_token", None)
        if token is None:
            return
        self._routing_token = None
        _active_router_var.reset(token)


# ---------------------------------------------------------------------------
# Dispatcher — called by both interception sites
# ---------------------------------------------------------------------------


def apply_router(
    router: "Router",
    request_body: Dict[str, Any],
    request_path: str,
    session: "SessionInfo",
) -> bool:
    """Run *router* on this request; mutate *request_body* if a decision was made.

    Called from both interception sites:

    * :class:`agentopt.proxy.interceptor.LocalHandler` — in-process httpx
    * :class:`agentopt.proxy.mitm_addon.AgentoptAddon` — subprocess via mitmproxy

    Returns ``True`` iff the body was mutated.  Errors from
    ``router.route()`` are caught and logged; the request proceeds
    unrouted.  A router must never break an agent.
    """
    from agentopt.proxy.providers import detect_provider

    try:
        provider = detect_provider(request_path)
        provider_name = provider.name if provider is not None else "unknown"
        requested_model = request_body.get("model")
        ctx = RouteContext(
            request_body=request_body,
            provider=provider_name,
            requested_model=requested_model,
            session=session,
            history=tuple(session.records),
        )
        decision = router.route(ctx)
    except Exception:
        logger.exception(
            "router.route() raised — passing request unrouted (the router "
            "must never break an agent)"
        )
        return False

    if decision is None:
        return False

    if decision.provider is not None or decision.api_key is not None:
        # v1: cross-provider routing is reserved; loud error so this
        # never silently passes a misconfigured decision.
        raise NotImplementedError(
            "RouteDecision.provider / RouteDecision.api_key are reserved "
            "for future cross-provider routing.  v1 supports same-provider "
            "model swaps only — set RouteDecision.model and leave "
            "provider / api_key as None."
        )

    if requested_model is not None and decision.model == requested_model:
        return False  # no-op decision

    if "model" not in request_body:
        # Gemini-style: model lives in the URL path, not the body.
        # Path-rewrite routing is a v1.x follow-up; skip for now.
        logger.debug(
            "router decision dropped: request body has no 'model' key "
            "(URL-encoded model routing, e.g. Gemini, is not yet supported). "
            "Provider=%s path=%s decision.model=%s",
            provider_name,
            request_path,
            decision.model,
        )
        return False

    request_body["model"] = decision.model
    return True
