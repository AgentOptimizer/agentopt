"""httpx monkey-patch — redirect in-process LLM calls to the session proxy.

The active session's ephemeral proxy port lives in a ``ContextVar`` set by
``LLMTracker.track()``.  When httpx is about to send a POST whose URL
matches a known LLM endpoint, we rewrite it to
``http://127.0.0.1:{port}/...`` and forward the original base URL through
the ``X-AgentOpt-Target`` header.  Everything else is passed through
untouched.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable, Optional, Tuple

import httpx

from .providers import DEFAULT_PROVIDERS, TARGET_HEADER


# ---------------------------------------------------------------------------
# Active-session port — set by ``LLMTracker.track()``
# ---------------------------------------------------------------------------

_session_port_var: ContextVar[Optional[int]] = ContextVar(
    "agentopt_session_port", default=None,
)


# ---------------------------------------------------------------------------
# Path-pattern detection — derived from the provider registry so the
# patch and the proxy stay in sync as new providers are added.
# ---------------------------------------------------------------------------


def _collect_llm_path_patterns() -> Tuple[str, ...]:
    patterns: list[str] = []
    for provider in DEFAULT_PROVIDERS.values():
        patterns.extend(provider.path_patterns)
    return tuple(patterns)


_LLM_PATH_PATTERNS: Tuple[str, ...] = _collect_llm_path_patterns()


def _is_llm_request(request: httpx.Request) -> bool:
    """True if *request* targets a known LLM API endpoint via POST."""
    if request.method != "POST":
        return False
    path = request.url.raw_path.decode("ascii", errors="ignore")
    return any(pattern in path for pattern in _LLM_PATH_PATTERNS)


# ---------------------------------------------------------------------------
# URL rewrite
# ---------------------------------------------------------------------------


def _rewrite_request(request: httpx.Request, session_port: int) -> httpx.Request:
    """Build a localhost-targeted clone of *request* for the proxy."""
    original_url = request.url
    original_base = f"{original_url.scheme}://{original_url.host}"
    if original_url.port and original_url.port not in (80, 443):
        original_base += f":{original_url.port}"

    remaining = original_url.raw_path.decode("ascii", errors="ignore")
    if original_url.query:
        remaining += "?" + original_url.query.decode("ascii", errors="ignore")

    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    headers[TARGET_HEADER] = original_base

    return httpx.Request(
        method=request.method,
        url=f"http://127.0.0.1:{session_port}{remaining}",
        headers=headers,
        content=request.content,
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
        if _is_llm_request(request):
            port = _session_port_var.get()
            if port is not None:
                request = _rewrite_request(request, port)
        return _original_sync_send(self, request, stream=stream, **kwargs)  # type: ignore[misc]

    async def _patched_async_send(
        self: Any, request: httpx.Request, *, stream: bool = False, **kwargs: Any,
    ) -> httpx.Response:
        if _is_llm_request(request):
            port = _session_port_var.get()
            if port is not None:
                request = _rewrite_request(request, port)
        return await _original_async_send(self, request, stream=stream, **kwargs)  # type: ignore[misc]

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
