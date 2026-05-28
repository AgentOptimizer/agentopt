"""In-process httpx + subprocess interception — record LLM calls directly.

The active tracking session lives in a ``ContextVar`` set by
``LLMTracker.track()``.  Two patches consult it:

* ``httpx.Client.send`` / ``AsyncClient.send`` — in-process Python
  agents.  When httpx is about to send a POST whose URL matches a known
  LLM endpoint, the patched ``send`` hands the request off to the
  session's :class:`CallHandler`.
* ``subprocess.Popen.__init__`` — subprocess / CLI agents.  When a
  subprocess is spawned inside an active ``track()`` scope, the patched
  init injects ``HTTPS_PROXY`` + the merged CA bundle path into the
  child's env so its LLM calls route through this session's mitmproxy
  port.  The child inherits exactly enough to be tracked; the user's
  ``run()`` method stays free of any agentopt imports.

Two handlers ship with the proxy:

* :class:`LocalHandler` — the original in-process behaviour (cache
  lookup, forward to upstream, record).  Used by :class:`LocalBackend`.
* :class:`RemoteHandler` (see :mod:`._remote_backend`) — forwards the
  request through a long-lived daemon's per-session proxy port; the
  daemon performs cache and recording.  Used by ``RemoteBackend``.

Both the httpx and subprocess install/uninstall helpers are refcounted
so coexisting ``LLMTracker`` instances don't tear down each other's
patches.
"""

from __future__ import annotations

import abc
import json
import os
import subprocess
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, Optional

import httpx

from .cache import CacheEntry, ResponseCache, _make_cache_key
from .recording import Recorder
from .session import SessionInfo
from .usage import decode_json_body, is_streaming_request

if TYPE_CHECKING:
    from agentopt.routing.base import Router


# ---------------------------------------------------------------------------
# Path-pattern detection
# ---------------------------------------------------------------------------


def _is_llm_request(request: httpx.Request, patterns: FrozenSet[str]) -> bool:
    """True if *request* targets an LLM endpoint listed in *patterns* via POST."""
    if request.method != "POST":
        return False
    path = request.url.raw_path.decode("ascii", errors="ignore")
    return any(pattern in path for pattern in patterns)


# ---------------------------------------------------------------------------
# Body / response helpers (shared by all handlers)
# ---------------------------------------------------------------------------


def _make_cached_response(request: httpx.Request, entry: CacheEntry,) -> httpx.Response:
    """Build an ``httpx.Response`` from a cache entry, attached to *request*."""
    headers = {
        k: v
        for k, v in entry.response_headers.items()
        if k.lower() not in ("transfer-encoding", "content-encoding")
    }
    return httpx.Response(
        status_code=200, headers=headers, content=entry.response_bytes, request=request,
    )


# ---------------------------------------------------------------------------
# CallHandler — the work the patched send dispatches to.
# ---------------------------------------------------------------------------


class CallHandler(abc.ABC):
    """Per-session strategy for handling an intercepted LLM HTTP call."""

    @abc.abstractmethod
    def handle_sync(
        self,
        client: httpx.Client,
        request: httpx.Request,
        *,
        stream: bool,
        **kwargs: Any,
    ) -> httpx.Response:
        """Synchronously process *request*; return the response to the caller."""

    @abc.abstractmethod
    async def handle_async(
        self,
        client: httpx.AsyncClient,
        request: httpx.Request,
        *,
        stream: bool,
        **kwargs: Any,
    ) -> httpx.Response:
        """Asynchronously process *request*; return the response."""


class LocalHandler(CallHandler):
    """In-process handler: route, cache lookup, forward to upstream, record.

    This is the body of the original ``_patched_sync_send`` /
    ``_patched_async_send`` extracted into a class so a remote-mode
    handler can plug into the same seam.

    When a router is attached, it runs **before** the cache lookup so
    cache keys reflect the model that will actually be sent upstream,
    not the model the client originally requested.
    """

    def __init__(
        self,
        session: SessionInfo,
        recorder: Recorder,
        cache: Optional[ResponseCache],
        router: Optional["Router"] = None,
    ) -> None:
        self._session = session
        self._recorder = recorder
        self._cache = cache
        self._router = router

    # -- sync ---------------------------------------------------------

    def handle_sync(
        self,
        client: httpx.Client,
        request: httpx.Request,
        *,
        stream: bool,
        **kwargs: Any,
    ) -> httpx.Response:
        json_body = decode_json_body(request.content)
        request = self._maybe_route(request, json_body)

        # Cache lookup (post-routing — key reflects the actual model)
        if self._cache is not None and json_body:
            entry = self._cache.get(_make_cache_key(json_body, str(request.url.path)))
            if entry is not None:
                resp = _make_cached_response(request, entry)
                self._record(
                    request, resp, None, entry.latency_seconds, cached=True,
                )
                return resp

        assert _original_sync_send is not None  # set by install_redirect
        t0 = time.monotonic()
        try:
            response = _original_sync_send(client, request, stream=stream, **kwargs)
        except Exception as exc:
            latency = time.monotonic() - t0
            self._record(request, None, str(exc), latency, cached=False)
            raise
        latency = time.monotonic() - t0
        self._record(request, response, None, latency, cached=False)
        return response

    # -- async --------------------------------------------------------

    async def handle_async(
        self,
        client: httpx.AsyncClient,
        request: httpx.Request,
        *,
        stream: bool,
        **kwargs: Any,
    ) -> httpx.Response:
        json_body = decode_json_body(request.content)
        request = self._maybe_route(request, json_body)

        if self._cache is not None and json_body:
            entry = self._cache.get(_make_cache_key(json_body, str(request.url.path)))
            if entry is not None:
                resp = _make_cached_response(request, entry)
                self._record(
                    request, resp, None, entry.latency_seconds, cached=True,
                )
                return resp

        assert _original_async_send is not None
        t0 = time.monotonic()
        try:
            response = await _original_async_send(
                client, request, stream=stream, **kwargs,
            )
        except Exception as exc:
            latency = time.monotonic() - t0
            self._record(request, None, str(exc), latency, cached=False)
            raise
        latency = time.monotonic() - t0
        self._record(request, response, None, latency, cached=False)
        return response

    # -- routing ------------------------------------------------------

    def _maybe_route(
        self, request: httpx.Request, json_body: Dict[str, Any],
    ) -> httpx.Request:
        """Apply the active router; return a fresh ``Request`` if it mutated.

        Returns the original *request* unchanged when there's nothing to
        do.  When the router rewrites the body, we build a new
        ``httpx.Request`` so its ``.stream`` (what the transport actually
        iterates over the wire) and ``Content-Length`` stay coherent —
        mutating ``request._content`` alone leaves a stale stream and
        ``h11`` rejects the mismatch.
        """
        if self._router is None or not json_body:
            return request
        from agentopt.routing.base import apply_router

        mutated = apply_router(
            self._router,
            json_body,
            request.url.raw_path.decode("ascii", errors="ignore"),
            self._session,
        )
        if not mutated:
            return request

        new_bytes = json.dumps(json_body, separators=(",", ":")).encode("utf-8")
        new_headers = httpx.Headers(request.headers)
        # Let httpx recompute Content-Length from the new body.
        new_headers.pop("content-length", None)
        return httpx.Request(
            method=request.method,
            url=request.url,
            headers=new_headers,
            content=new_bytes,
            extensions=dict(request.extensions),
        )

    # -- recording (shared by sync + async) --------------------------

    def _record(
        self,
        request: httpx.Request,
        response: Optional[httpx.Response],
        error: Optional[str],
        latency: float,
        *,
        cached: bool,
    ) -> None:
        """Convert a completed (or failed) httpx exchange into a CallRecord."""
        json_body = decode_json_body(request.content)
        request_url = str(request.url)
        is_streaming = is_streaming_request(json_body, request_url)

        if response is None:
            self._recorder.record(
                self._session,
                json_body,
                None,
                request_url,
                latency,
                cached=False,
                is_streaming=is_streaming,
                status_code=0,
                error=error or "upstream error",
            )
            return

        status = response.status_code
        body_bytes = response.content  # forces buffering for streamed bodies
        record_error = None if status == 200 else f"HTTP {status}"

        self._recorder.record(
            self._session,
            json_body,
            body_bytes,
            request_url,
            latency,
            cached=cached,
            is_streaming=is_streaming,
            status_code=status,
            error=record_error,
        )

        # Cache successful 200s on the miss path only.
        if not cached and status == 200 and self._cache is not None and json_body:
            self._cache.put(
                _make_cache_key(json_body, str(request.url.path)),
                CacheEntry(
                    response_bytes=body_bytes,
                    response_headers=dict(response.headers),
                    latency_seconds=latency,
                ),
            )


# ---------------------------------------------------------------------------
# Active-session context — populated by ``LLMTracker.track()``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveSession:
    """Per-track() bundle: what the patched ``send`` needs to dispatch a call.

    ``path_patterns`` is a snapshot of the owning backend's
    :class:`ProviderRegistry` patterns at ``track()`` entry — consulting
    it (rather than a process-global set) keeps registrations on one
    tracker from leaking into other tracker instances.

    ``handler`` is the per-session strategy (cache+forward+record locally,
    or forward through a daemon).  See :class:`CallHandler`.

    ``proxy_host`` and ``ca_bundle_path`` carry the subprocess-routing
    bits.  ``LocalBackend`` leaves them ``None`` (the subprocess patch
    falls back to ``127.0.0.1`` + ``_ensure_ca_bundle()`` from the local
    mitmproxy install).  ``RemoteBackend`` sets them so children of a
    daemon-mode session route to the daemon's host with its CA bundle.
    """

    session: SessionInfo
    port: int  # session's proxy port (local SessionMaster or daemon-allocated)
    path_patterns: FrozenSet[str]
    handler: CallHandler
    proxy_host: Optional[str] = None
    ca_bundle_path: Optional[str] = None


_active_session_var: ContextVar[Optional[ActiveSession]] = ContextVar(
    "agentopt_active_session", default=None,
)


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------

_original_sync_send: Optional[Callable] = None
_original_async_send: Optional[Callable] = None
_installed = False
# Refcount so multiple coexisting ``LLMTracker`` instances don't tear
# down each other's httpx patch.  Only the first install actually
# patches; only the matching last uninstall actually restores.
_install_refcount = 0


def install_redirect() -> None:
    """Monkey-patch ``httpx.Client.send`` and ``AsyncClient.send``.

    Idempotent across multiple callers — uses a refcount so each
    ``install_redirect`` is paired with one ``uninstall_redirect``;
    the patch is restored only when the count drops back to zero.
    """
    global _original_sync_send, _original_async_send, _installed
    global _install_refcount
    _install_refcount += 1
    if _installed:
        return

    _original_sync_send = httpx.Client.send
    _original_async_send = httpx.AsyncClient.send

    def _patched_sync_send(
        self: Any, request: httpx.Request, *, stream: bool = False, **kwargs: Any,
    ) -> httpx.Response:
        active = _active_session_var.get()
        if active is None or not _is_llm_request(request, active.path_patterns):
            return _original_sync_send(self, request, stream=stream, **kwargs)  # type: ignore[misc]
        return active.handler.handle_sync(self, request, stream=stream, **kwargs)

    async def _patched_async_send(
        self: Any, request: httpx.Request, *, stream: bool = False, **kwargs: Any,
    ) -> httpx.Response:
        active = _active_session_var.get()
        if active is None or not _is_llm_request(request, active.path_patterns):
            return await _original_async_send(self, request, stream=stream, **kwargs)  # type: ignore[misc]
        return await active.handler.handle_async(self, request, stream=stream, **kwargs)

    httpx.Client.send = _patched_sync_send  # type: ignore[assignment]
    httpx.AsyncClient.send = _patched_async_send  # type: ignore[assignment]
    _installed = True


def forward_sync(
    client: httpx.Client,
    request: httpx.Request,
    *,
    stream: bool = False,
    **kwargs: Any,
) -> httpx.Response:
    """Forward a request via the saved-original ``httpx.Client.send``.

    Lets a handler in a *different* module (e.g. ``RemoteHandler``)
    bypass the patch without importing the private module global at
    import time (the value is ``None`` until :func:`install_redirect`
    runs).
    """
    assert _original_sync_send is not None, "install_redirect() not called"
    return _original_sync_send(client, request, stream=stream, **kwargs)


async def forward_async(
    client: httpx.AsyncClient,
    request: httpx.Request,
    *,
    stream: bool = False,
    **kwargs: Any,
) -> httpx.Response:
    """Async counterpart of :func:`forward_sync`."""
    assert _original_async_send is not None, "install_redirect() not called"
    return await _original_async_send(client, request, stream=stream, **kwargs)


def uninstall_redirect() -> None:
    """Restore the original ``httpx`` send methods.

    Decrements the refcount; only the matching last call actually
    restores the original ``httpx`` methods.  Earlier calls are no-ops
    so coexisting ``LLMTracker`` instances don't tear down each
    other's patch on exit.
    """
    global _original_sync_send, _original_async_send, _installed
    global _install_refcount
    if _install_refcount > 0:
        _install_refcount -= 1
    if _install_refcount > 0 or not _installed:
        return

    if _original_sync_send is not None:
        httpx.Client.send = _original_sync_send  # type: ignore[assignment]
    if _original_async_send is not None:
        httpx.AsyncClient.send = _original_async_send  # type: ignore[assignment]

    _original_sync_send = None
    _original_async_send = None
    _installed = False


# ---------------------------------------------------------------------------
# Subprocess interception — inject HTTPS_PROXY + CA bundle into spawned
# children so subprocess agents are tracked without per-agent boilerplate.
# ---------------------------------------------------------------------------

_original_popen_init: Optional[Callable] = None
_subprocess_installed = False
_subprocess_install_refcount = 0


def _session_env_for(active: "ActiveSession") -> Dict[str, str]:
    """Env vars routing a child process through *active*'s proxy port.

    Local mode: host is ``127.0.0.1`` and the CA bundle is the merged
    mitmproxy + certifi file under ``~/.mitmproxy``.  Remote mode: host
    and bundle are carried on the ``ActiveSession`` by ``RemoteBackend``.

    The ``_ensure_ca_bundle`` import is lazy so callers that never spawn
    a subprocess inside a ``track()`` scope never touch the filesystem.
    """
    if active.ca_bundle_path is not None:
        bundle = active.ca_bundle_path
    else:
        from ._backend import _ensure_ca_bundle

        bundle = _ensure_ca_bundle()
    host = active.proxy_host or "127.0.0.1"
    return {
        "HTTPS_PROXY": f"http://{host}:{active.port}",
        "SSL_CERT_FILE": bundle,
        "REQUESTS_CA_BUNDLE": bundle,
        "NODE_EXTRA_CA_CERTS": bundle,
    }


def install_subprocess_redirect() -> None:
    """Monkey-patch ``subprocess.Popen.__init__`` to inject session env.

    While an ``ActiveSession`` is set in the calling context, every new
    ``Popen`` (and therefore every ``subprocess.run`` / ``check_output`` /
    ``call`` / ``check_call``, which all go through ``Popen``) gets the
    session's ``HTTPS_PROXY`` + CA bundle keys merged into its env.

    Merge policy — *explicit beats implicit*:

    * ``env=None`` (the default — inherit ``os.environ``): build
      ``{**os.environ, **session_env}``.  The session wins over any
      ``HTTPS_PROXY`` the parent shell happened to set, because the
      user said nothing about env in this call.
    * ``env=<dict>`` (caller wrote an explicit env): build
      ``{**session_env, **user_env}``.  ``user_env`` wins on conflicts
      — a key the user *explicitly* set in this very call (e.g.
      ``HTTPS_PROXY`` pointed at a custom upstream for a test) must
      not be silently overridden.  The common case (``env={"PATH":
      ...}`` for non-LLM reasons) still gets tracking because the
      caller didn't write the session keys, so session_env fills them.

    Idempotent and refcounted, identical to :func:`install_redirect`.
    """
    global _original_popen_init, _subprocess_installed
    global _subprocess_install_refcount
    _subprocess_install_refcount += 1
    if _subprocess_installed:
        return

    _original_popen_init = subprocess.Popen.__init__

    def _patched_popen_init(self: Any, *args: Any, **kwargs: Any) -> None:
        active = _active_session_var.get()
        if active is not None:
            session_env = _session_env_for(active)
            user_env = kwargs.get("env")
            if user_env is None:
                # Implicit inherit — session wins over the parent shell.
                kwargs["env"] = {**os.environ, **session_env}
            else:
                # Explicit env — caller's keys win over our injection.
                kwargs["env"] = {**session_env, **user_env}
        return _original_popen_init(self, *args, **kwargs)  # type: ignore[misc]

    subprocess.Popen.__init__ = _patched_popen_init  # type: ignore[assignment]
    _subprocess_installed = True


def uninstall_subprocess_redirect() -> None:
    """Restore the original ``subprocess.Popen.__init__``.

    Refcounted; the patch is removed only when the count drops to zero.
    """
    global _original_popen_init, _subprocess_installed
    global _subprocess_install_refcount
    if _subprocess_install_refcount > 0:
        _subprocess_install_refcount -= 1
    if _subprocess_install_refcount > 0 or not _subprocess_installed:
        return

    if _original_popen_init is not None:
        subprocess.Popen.__init__ = _original_popen_init  # type: ignore[assignment]

    _original_popen_init = None
    _subprocess_installed = False
