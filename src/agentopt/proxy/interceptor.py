"""In-process httpx interception — record LLM calls directly.

The active tracking session lives in a ``ContextVar`` set by
``LLMTracker.track()``.  When httpx is about to send a POST whose URL
matches a known LLM endpoint, we cache-look-up first and short-circuit on
hit, otherwise we forward the call to the real upstream, then record the
result and (for 200s) populate the cache.

No localhost round-trip — the in-process path doesn't need a proxy
listener.  Only subprocess agents do; they go through the mitmproxy-based
``SessionMaster`` (see ``mitm_runner.py``).

The httpx wrapper is a process-wide monkey-patch installed once by
``LLMTracker.start()``; the ContextVar makes it safe under concurrent
``track()`` blocks in different async tasks or threads.
"""

from __future__ import annotations

import json
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterable, Optional

import httpx

from .cache import CacheEntry, ResponseCache, _make_cache_key
from .providers import DEFAULT_PROVIDERS
from .recording import Recorder
from .session import SessionInfo
from .usage import is_streaming_request


# ---------------------------------------------------------------------------
# Active-session context — populated by ``LLMTracker.track()``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveSession:
    """Per-track() bundle: what the wrapper needs to record a call."""

    session: SessionInfo
    recorder: Recorder
    cache: Optional[ResponseCache]
    port: int  # ``SessionMaster`` port — for ``get_current_session_proxy``


_active_session_var: ContextVar[Optional[ActiveSession]] = ContextVar(
    "agentopt_active_session", default=None,
)


# ---------------------------------------------------------------------------
# Path-pattern detection
# ---------------------------------------------------------------------------
# Mutable so ``LLMTracker.register_provider`` can extend it.  We use a
# ``frozenset`` rebuilt-on-write under a lock — set mutation during
# iteration would be a real race in the wrapper hot path, and the
# rebuild-then-rebind pattern is atomic under the GIL so reads need no lock.

_path_patterns_lock = threading.Lock()
_LLM_PATH_PATTERNS: FrozenSet[str] = frozenset(
    pattern
    for provider in DEFAULT_PROVIDERS.values()
    for pattern in provider.path_patterns
)


def register_extra_llm_paths(patterns: Iterable[str]) -> None:
    """Add path patterns the wrapper should treat as LLM endpoints.

    Called by :meth:`LLMTracker.register_provider` so in-process httpx
    calls to runtime-registered providers are recorded like the built-ins.
    """
    global _LLM_PATH_PATTERNS
    with _path_patterns_lock:
        _LLM_PATH_PATTERNS = _LLM_PATH_PATTERNS | frozenset(patterns)


def _is_llm_request(request: httpx.Request) -> bool:
    """True if *request* targets a known LLM API endpoint via POST."""
    if request.method != "POST":
        return False
    path = request.url.raw_path.decode("ascii", errors="ignore")
    # Single read — name binding is atomic, so we get a consistent snapshot.
    patterns = _LLM_PATH_PATTERNS
    return any(pattern in path for pattern in patterns)


# ---------------------------------------------------------------------------
# Recording — shared by sync and async paths
# ---------------------------------------------------------------------------


def _decode_json_body(body: Optional[bytes]) -> Dict[str, Any]:
    """Best-effort JSON decode; non-dict / invalid → ``{}``."""
    if not body:
        return {}
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _record_after_response(
    active: ActiveSession,
    request: httpx.Request,
    response: Optional[httpx.Response],
    error: Optional[str],
    latency: float,
    cached: bool,
) -> None:
    """Convert a completed (or failed) httpx exchange into a CallRecord."""
    json_body = _decode_json_body(request.content)
    request_url = str(request.url)
    is_streaming = is_streaming_request(json_body, request_url)

    if response is None:
        active.recorder.record(
            active.session,
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

    active.recorder.record(
        active.session,
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
    if not cached and status == 200 and active.cache is not None and json_body:
        active.cache.put(
            _make_cache_key(json_body),
            CacheEntry(
                response_bytes=body_bytes,
                response_headers=dict(response.headers),
                latency_seconds=latency,
            ),
        )


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------

_original_sync_send: Optional[Callable] = None
_original_async_send: Optional[Callable] = None
_installed = False


def install_redirect() -> None:
    """Monkey-patch ``httpx.Client.send`` and ``AsyncClient.send``.

    Idempotent.  Requests outside an active tracking session, or whose
    URL doesn't look like an LLM call, are passed through unmodified.
    """
    global _original_sync_send, _original_async_send, _installed
    if _installed:
        return

    _original_sync_send = httpx.Client.send
    _original_async_send = httpx.AsyncClient.send

    def _patched_sync_send(
        self: Any, request: httpx.Request, *, stream: bool = False, **kwargs: Any,
    ) -> httpx.Response:
        if not _is_llm_request(request):
            return _original_sync_send(self, request, stream=stream, **kwargs)  # type: ignore[misc]
        active = _active_session_var.get()
        if active is None:
            return _original_sync_send(self, request, stream=stream, **kwargs)  # type: ignore[misc]

        json_body = _decode_json_body(request.content)
        # Cache lookup
        if active.cache is not None and json_body:
            entry = active.cache.get(_make_cache_key(json_body))
            if entry is not None:
                resp = _make_cached_response(request, entry)
                _record_after_response(
                    active, request, resp, None, entry.latency_seconds, cached=True,
                )
                return resp

        # Forward to real upstream
        t0 = time.monotonic()
        try:
            response = _original_sync_send(self, request, stream=stream, **kwargs)  # type: ignore[misc]
        except Exception as exc:
            latency = time.monotonic() - t0
            _record_after_response(
                active, request, None, str(exc), latency, cached=False
            )
            raise
        latency = time.monotonic() - t0
        _record_after_response(active, request, response, None, latency, cached=False)
        return response

    async def _patched_async_send(
        self: Any, request: httpx.Request, *, stream: bool = False, **kwargs: Any,
    ) -> httpx.Response:
        if not _is_llm_request(request):
            return await _original_async_send(self, request, stream=stream, **kwargs)  # type: ignore[misc]
        active = _active_session_var.get()
        if active is None:
            return await _original_async_send(self, request, stream=stream, **kwargs)  # type: ignore[misc]

        json_body = _decode_json_body(request.content)
        if active.cache is not None and json_body:
            entry = active.cache.get(_make_cache_key(json_body))
            if entry is not None:
                resp = _make_cached_response(request, entry)
                _record_after_response(
                    active, request, resp, None, entry.latency_seconds, cached=True,
                )
                return resp

        t0 = time.monotonic()
        try:
            response = await _original_async_send(self, request, stream=stream, **kwargs)  # type: ignore[misc]
        except Exception as exc:
            latency = time.monotonic() - t0
            _record_after_response(
                active, request, None, str(exc), latency, cached=False
            )
            raise
        latency = time.monotonic() - t0
        _record_after_response(active, request, response, None, latency, cached=False)
        return response

    httpx.Client.send = _patched_sync_send  # type: ignore[assignment]
    httpx.AsyncClient.send = _patched_async_send  # type: ignore[assignment]
    _installed = True


def uninstall_redirect() -> None:
    """Restore the original ``httpx`` send methods."""
    global _original_sync_send, _original_async_send, _installed
    if not _installed:
        return

    if _original_sync_send is not None:
        httpx.Client.send = _original_sync_send  # type: ignore[assignment]
    if _original_async_send is not None:
        httpx.AsyncClient.send = _original_async_send  # type: ignore[assignment]

    _original_sync_send = None
    _original_async_send = None
    _installed = False
