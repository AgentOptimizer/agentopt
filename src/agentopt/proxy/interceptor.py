"""httpx monkey-patching — redirect LLM requests through the proxy server.

This module replaces the SDK's original base URL with the local proxy URL
at *request time*, inserting the current session ID (from a ``ContextVar``)
into the URL path and forwarding the original base URL as a header so the
proxy knows where to forward.

All token parsing, caching, and ``CallRecord`` creation is handled by the
proxy server — the interceptor only rewrites URLs.
"""

from contextvars import ContextVar
from typing import Any, Callable, Optional

import httpx

from .providers import TARGET_HEADER

# ---------------------------------------------------------------------------
# ContextVar — the active session ID (set by LLMTracker.track())
# ---------------------------------------------------------------------------

_session_id_var: ContextVar[Optional[str]] = ContextVar(
    "agentopt_session_id", default=None
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_original_sync_send: Optional[Callable] = None
_original_async_send: Optional[Callable] = None
_installed = False
_proxy_base_url: Optional[str] = None

# URL path patterns that indicate LLM API endpoints.
_LLM_PATH_PATTERNS = (
    "/chat/completions",
    "/v1/messages",
    "/v1/responses",
    "/v1beta/models",
    "/v1/models",
)


def _is_llm_request(request: httpx.Request) -> bool:
    """Check if this request targets a known LLM API endpoint."""
    if request.method != "POST":
        return False
    path = request.url.raw_path.decode("ascii", errors="ignore")
    return any(pattern in path for pattern in _LLM_PATH_PATTERNS)


def _rewrite_request(
    request: httpx.Request, proxy_base_url: str, session_id: str,
) -> httpx.Request:
    """Create a new ``httpx.Request`` pointing at the proxy.

    The original base URL (scheme + host) is forwarded as the
    ``X-AgentOpt-Target`` header so the proxy can route upstream.
    """
    original_url = request.url
    original_base = f"{original_url.scheme}://{original_url.host}"
    if original_url.port and original_url.port not in (80, 443):
        original_base += f":{original_url.port}"

    # Preserve both path and query string.
    remaining = original_url.raw_path.decode("ascii", errors="ignore")
    if original_url.query:
        remaining += "?" + original_url.query.decode("ascii", errors="ignore")

    new_url = f"{proxy_base_url}/{session_id}{remaining}"

    # Merge existing headers with the target header.
    # Remove the original Host header (case-insensitive) — httpx will set
    # a new one for the proxy URL.
    headers = {
        k: v for k, v in request.headers.items() if k.lower() != "host"
    }
    headers[TARGET_HEADER] = original_base

    return httpx.Request(
        method=request.method, url=new_url, headers=headers, content=request.content,
    )


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------


def install_redirect(proxy_base_url: str) -> None:
    """Monkey-patch ``httpx`` to redirect LLM requests through the proxy.

    Non-LLM requests and requests outside an active tracking session are
    passed through unmodified.
    """
    global _original_sync_send, _original_async_send, _installed, _proxy_base_url

    if _installed:
        return

    _proxy_base_url = proxy_base_url
    _original_sync_send = httpx.Client.send
    _original_async_send = httpx.AsyncClient.send

    def _patched_sync_send(
        self: Any, request: httpx.Request, *, stream: bool = False, **kwargs: Any
    ) -> httpx.Response:
        if _is_llm_request(request):
            session_id = _session_id_var.get()
            if session_id is not None:
                request = _rewrite_request(request, _proxy_base_url, session_id)  # type: ignore[arg-type]
        return _original_sync_send(self, request, stream=stream, **kwargs)  # type: ignore[misc]

    async def _patched_async_send(
        self: Any, request: httpx.Request, *, stream: bool = False, **kwargs: Any
    ) -> httpx.Response:
        if _is_llm_request(request):
            session_id = _session_id_var.get()
            if session_id is not None:
                request = _rewrite_request(request, _proxy_base_url, session_id)  # type: ignore[arg-type]
        return await _original_async_send(self, request, stream=stream, **kwargs)  # type: ignore[misc]

    httpx.Client.send = _patched_sync_send  # type: ignore[assignment]
    httpx.AsyncClient.send = _patched_async_send  # type: ignore[assignment]
    _installed = True


def uninstall_redirect() -> None:
    """Restore the original ``httpx.Client.send`` methods."""
    global _original_sync_send, _original_async_send, _installed, _proxy_base_url

    if not _installed:
        return

    if _original_sync_send is not None:
        httpx.Client.send = _original_sync_send  # type: ignore[assignment]
    if _original_async_send is not None:
        httpx.AsyncClient.send = _original_async_send  # type: ignore[assignment]

    _original_sync_send = None
    _original_async_send = None
    _proxy_base_url = None
    _installed = False
