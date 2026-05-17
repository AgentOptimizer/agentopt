"""In-process httpx interception — record LLM calls directly.

The active tracking session lives in a ``ContextVar`` set by
``LLMTracker.track()``.  When httpx is about to send a POST whose URL
matches a known LLM endpoint, the patched ``send`` hands the request off
to the session's :class:`CallHandler` to do the actual work.

Two handlers ship with the proxy:

* :class:`LocalHandler` — the original in-process behaviour (cache
  lookup, forward to upstream, record).  Used by :class:`LocalBackend`.
* :class:`RemoteHandler` (see :mod:`._remote_backend`) — forwards the
  request through a long-lived daemon's per-session proxy port; the
  daemon performs cache and recording.  Used by ``RemoteBackend``.

The handler seam is the smallest possible split: the patched
``send`` is a dispatcher; everything else (cache, forward, record)
lives in the handler.  Both modes share the ContextVar activation, the
fast-path check (``_is_llm_request``), and the install/uninstall plumbing.
"""

from __future__ import annotations

import abc
import json
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
    """

    session: SessionInfo
    port: int  # session's proxy port (local SessionMaster or daemon-allocated)
    path_patterns: FrozenSet[str]
    handler: CallHandler


_active_session_var: ContextVar[Optional[ActiveSession]] = ContextVar(
    "agentopt_active_session", default=None,
)


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------

_original_sync_send: Optional[Callable] = None
_original_async_send: Optional[Callable] = None
_installed = False
# Refcount so multiple coexisting ``LLMTracker`` instances (or the
# ephemeral one ``Router.__enter__`` spins up) don't tear down each
# other's httpx patch.  Only the first install actually patches;
# only the matching last uninstall actually restores.
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
    so a nested auto-tracker exit can't tear down a still-active
    user tracker's patch.
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
