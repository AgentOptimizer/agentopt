"""httpx monkey-patching — redirect LLM requests through the proxy server.

Thin URL rewrite + header injection.  The session port (from a ContextVar)
determines which session-specific proxy port to target.  All logic lives
in the proxy server.
"""

from contextvars import ContextVar
from typing import Any, Callable, Optional

import httpx

from .providers import TARGET_HEADER

# ---------------------------------------------------------------------------
# ContextVar — the active session's proxy port (set by LLMTracker.track())
# ---------------------------------------------------------------------------

_session_port_var: ContextVar[Optional[int]] = ContextVar(
    "agentopt_session_port", default=None
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_original_sync_send: Optional[Callable] = None
_original_async_send: Optional[Callable] = None
_installed = False

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


def _rewrite_request(request: httpx.Request, session_port: int) -> httpx.Request:
    """Create a new ``httpx.Request`` pointing at the session proxy port.

    The original base URL is forwarded as ``X-AgentOpt-Target`` so the
    proxy knows where to route upstream.
    """
    original_url = request.url
    original_base = f"{original_url.scheme}://{original_url.host}"
    if original_url.port and original_url.port not in (80, 443):
        original_base += f":{original_url.port}"

    remaining = original_url.raw_path.decode("ascii", errors="ignore")
    if original_url.query:
        remaining += "?" + original_url.query.decode("ascii", errors="ignore")

    new_url = f"http://127.0.0.1:{session_port}{remaining}"

    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    headers[TARGET_HEADER] = original_base

    return httpx.Request(
        method=request.method, url=new_url, headers=headers, content=request.content,
    )


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------


def install_redirect() -> None:
    """Monkey-patch ``httpx`` to redirect LLM requests through the proxy.

    Non-LLM requests and requests outside an active tracking session are
    passed through unmodified.
    """
    global _original_sync_send, _original_async_send, _installed

    if _installed:
        return

    _original_sync_send = httpx.Client.send
    _original_async_send = httpx.AsyncClient.send

    def _patched_sync_send(
        self: Any, request: httpx.Request, *, stream: bool = False, **kwargs: Any
    ) -> httpx.Response:
        if _is_llm_request(request):
            port = _session_port_var.get()
            if port is not None:
                request = _rewrite_request(request, port)
        return _original_sync_send(self, request, stream=stream, **kwargs)  # type: ignore[misc]

    async def _patched_async_send(
        self: Any, request: httpx.Request, *, stream: bool = False, **kwargs: Any
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
    """Restore the original ``httpx.Client.send`` methods."""
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
