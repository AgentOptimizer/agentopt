"""Routing abstractions: :class:`Router`, :class:`RouteContext`, dispatcher.

A *router* is a per-call policy that decides which model to actually
send a request to.  It sees the parsed request body, session metadata,
and prior LLM calls in the session, and returns either a model name
(swap to that model) or ``None`` (keep the client's choice).

Public API shape::

    from agentopt import RandomRouter

    # Standalone — one line, no tracker boilerplate
    with RandomRouter(candidates=["gpt-4o", "gpt-4o-mini"]):
        agent.run(question)

    # Or, when you already have an LLMTracker, pass `router=` explicitly
    with tracker.track(data_id="dp", combo_id="c", router=router):
        agent.run(question)

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
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

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
    """

    request_body: Dict[str, Any]
    provider: str  # "openai" | "anthropic" | "google" | ... | "unknown"
    requested_model: Optional[str]
    session: "SessionInfo"
    history: Sequence["CallRecord"]


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

    Subclasses implement :meth:`route`, which returns the model name to
    use (or ``None`` to keep the client's requested model).

    Use the instance as a context manager — ``with router:`` spins up
    an ephemeral :class:`LLMTracker` and a tracking session for you, so
    you never have to write the lifecycle boilerplate manually::

        with RandomRouter(candidates=["gpt-4o", "gpt-4o-mini"]):
            agent.run(question)

    If you already have an ``LLMTracker`` and want to attach a router
    to a specific scope, pass ``router=`` to ``track()`` instead — and
    do *not* nest ``with router:`` inside that ``track()`` (the
    auto-spawn would clobber attribution; we raise a clear error).

    **Not re-entrant on a single instance.**  Use distinct instances
    for nested or concurrent scopes.

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

    @classmethod
    def from_config(cls, **kwargs: Any) -> "Router":
        """Reconstruct an instance from ``_config_kwargs`` output.

        Default just calls ``cls(**kwargs)``; override if your
        ``__init__`` needs argument processing the kwargs don't capture
        (e.g. resolving model aliases).
        """
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Context manager — auto-spins an ephemeral tracker when needed
    # ------------------------------------------------------------------

    def __enter__(self) -> "Router":
        if getattr(self, "_routing_token", None) is not None:
            raise RuntimeError(
                "Router is already active — `with router:` is not "
                "re-entrant on the same instance.  Instantiate a fresh "
                "Router for nested or concurrent scopes."
            )

        # If the user is already inside an LLMTracker.track() scope, we'd
        # silently fail to route (the handler was built without a router).
        # Raise with a clear pointer to the correct pattern.
        from agentopt.proxy.interceptor import _active_session_var

        if _active_session_var.get() is not None:
            raise RuntimeError(
                "Cannot enter `with router:` inside an active "
                "tracker.track() scope — the surrounding session was "
                "constructed without this router and wouldn't see it.  "
                "Either move `with router:` to wrap the track() block, "
                "or pass `router=...` to track() directly."
            )

        self._routing_token = _active_router_var.set(self)

        # Spin an ephemeral LLMTracker so this router does something on
        # its own.  Users with attribution requirements should construct
        # their own tracker and use `tracker.track(router=...)` instead.
        from agentopt.proxy.tracker import LLMTracker

        self._auto_tracker = LLMTracker()
        try:
            self._auto_tracker.start()
            self._auto_track_ctx = self._auto_tracker.track(
                data_id="__router__", combo_id="__router__", router=self,
            )
            self._auto_track_ctx.__enter__()
        except Exception:
            # Ensure ContextVar + tracker state are cleaned up on a
            # failed enter so the next attempt isn't blocked.
            try:
                self._auto_tracker.close()
            except Exception:
                logger.debug("ignored error closing auto-tracker", exc_info=True)
            _active_router_var.reset(self._routing_token)
            self._routing_token = None
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        token = getattr(self, "_routing_token", None)
        if token is None:
            return
        self._routing_token = None
        try:
            track_ctx = getattr(self, "_auto_track_ctx", None)
            if track_ctx is not None:
                track_ctx.__exit__(exc_type, exc, tb)
            tracker = getattr(self, "_auto_tracker", None)
            if tracker is not None:
                tracker.close()
        finally:
            _active_router_var.reset(token)
            for attr in ("_auto_tracker", "_auto_track_ctx"):
                if hasattr(self, attr):
                    delattr(self, attr)


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
