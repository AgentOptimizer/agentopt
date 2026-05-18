"""Routing abstractions: :class:`Router`, :class:`RouteContext`, dispatcher.

A *router* is a per-call policy that decides which model to actually
send a request to.  It sees the parsed request body, session metadata,
and prior LLM calls in the session, and returns either a model name
(swap to that model) or ``None`` (keep the client's choice).

Public API shape::

    from agentopt import LLMTracker, RandomRouter

    router = RandomRouter(candidates=["gpt-4o", "gpt-4o-mini"])
    with LLMTracker(combo_id="X", router=router) as tracker:
        agent.run(question)
    tracker.print_summary()

The swap happens transparently at the HTTP layer (in-process httpx
patch + per-session mitmproxy addon), so any framework or subprocess
agent works without integration code.

**Scope (v1):** same-provider routing only.  ``Router.route`` returns
a model name string; cross-provider routing (rewriting host + auth +
schema) is a v2 feature.  Routing also works in daemon mode via router
serialization in ``RemoteBackend.track()`` and daemon-side resolution;
custom routers must implement ``_config_kwargs()`` so they can be
reconstructed remotely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from agentopt.proxy.models import CallRecord
    from agentopt.proxy.session import SessionInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RouteContext — the call's read-only view passed to Router.route
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteContext:
    """Per-call context passed to :meth:`Router.route`.

    Not exported at the top-level (``agentopt.RouteContext``) — import
    from :mod:`agentopt.routing` if you want a type annotation on
    custom routers.  Most user code accesses the fields by attribute
    on the ``ctx`` parameter without ever naming the type.

    ``request_body`` is wrapped in :class:`types.MappingProxyType` so
    ``ctx.request_body["foo"] = bar`` raises :class:`TypeError`.  This
    is a shallow guarantee — nested lists/dicts (``messages``,
    ``tools``) are still mutable through the proxy, but the cheap
    top-level guard catches the common contract violations without
    paying for a deep copy on every call.
    """

    request_body: Mapping[str, Any]
    provider: str  # "openai" | "anthropic" | "google" | ... | "unknown"
    requested_model: Optional[str]
    session: "SessionInfo"
    history: Sequence["CallRecord"]


# ---------------------------------------------------------------------------
# Router base class
# ---------------------------------------------------------------------------


class Router:
    """Base class for per-call model routing policies.

    Subclasses implement :meth:`route`, which returns the model name to
    use (or ``None`` to keep the client's requested model).

    Activation is always via :class:`agentopt.LLMTracker`::

        router = RandomRouter(candidates=["gpt-4o", "gpt-4o-mini"])
        with LLMTracker(combo_id="X", router=router) as tracker:
            agent.run(question)
        tracker.print_summary()

    The same code works against a local proxy or a long-lived
    ``agentopt serve`` daemon — setting ``AGENTOPT_GATEWAY_URL`` is the
    entire deployment switch.

    **Daemon-mode serialization.**  When ``AGENTOPT_GATEWAY_URL`` is
    set, ``RemoteBackend`` describes the router to the daemon via
    :meth:`config`.  Subclasses must override :meth:`_config_kwargs`
    to return a JSON-serializable dict of ``__init__`` kwargs; the
    base class's default raises :class:`NotImplementedError` so the
    failure is loud at the wire boundary, not on a quiet HTTP 500.
    """

    #: Short alias for built-in policies.  ``RemoteBackend`` sends this
    #: as ``router.policy`` so the daemon can look up the class in
    #: ``BUILTIN_POLICIES``.  Custom routers leave this empty;
    #: :meth:`config` falls back to a ``module:Class`` import path.
    POLICY_NAME: str = ""

    def route(self, ctx: RouteContext) -> Optional[str]:
        """Decide which model to use for *ctx*.

        Return the model name as a string to swap the model, or ``None``
        to keep the client's requested model unchanged.

        Exceptions raised here are caught by the dispatcher and logged;
        the request proceeds unrouted.  A router should never break an
        agent.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Daemon-mode serialization
    # ------------------------------------------------------------------

    def config(self) -> Dict[str, Any]:
        """Return a JSON-serializable ``{policy, kwargs}`` for daemon transport.

        ``RemoteBackend`` calls this and POSTs the result as the
        ``router`` field on ``/sessions``.  Default implementation
        uses :attr:`POLICY_NAME` if set, else falls back to
        ``"module:Class"``; subclasses override :meth:`_config_kwargs`
        to supply the constructor arguments.
        """
        policy = self.POLICY_NAME or (
            f"{type(self).__module__}:{type(self).__qualname__}"
        )
        return {"policy": policy, "kwargs": self._config_kwargs()}

    def _config_kwargs(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of ``__init__`` kwargs.

        Override to enable daemon-mode routing.  The default raises so
        custom routers fail loudly at the wire boundary instead of
        silently bypassing the daemon.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support daemon-mode routing. "
            "Override `_config_kwargs()` to return a JSON-serializable dict "
            "of __init__ kwargs, or use a built-in router (e.g. RandomRouter), "
            "or unset AGENTOPT_GATEWAY_URL to run in library mode."
        )


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
            request_body=MappingProxyType(request_body),
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
    if not isinstance(decision, str):
        logger.error(
            "router.route() returned type %s — expected Optional[str] "
            "(a model name) for v1 same-provider routing; passing request "
            "unrouted",
            type(decision).__name__,
        )
        return False
    if requested_model is not None and decision == requested_model:
        return False  # no-op decision

    if "model" not in request_body:
        # Gemini-style: model lives in the URL path, not the body.
        # Path-rewrite routing is a v1.x follow-up; skip for now.
        logger.debug(
            "router decision dropped: request body has no 'model' key "
            "(URL-encoded model routing, e.g. Gemini, is not yet supported). "
            "Provider=%s path=%s decision=%s",
            provider_name,
            request_path,
            decision,
        )
        return False

    request_body["model"] = decision
    return True
