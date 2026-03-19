"""httpx monkey-patching for LLM call interception."""

import json
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

from .cache import CacheEntry, ResponseCache, _make_cache_key

# ---------------------------------------------------------------------------
# ContextVars for attribution
# ---------------------------------------------------------------------------

_data_id_var: ContextVar[Optional[str]] = ContextVar("agentproxy_data_id", default=None)
_combo_id_var: ContextVar[Optional[str]] = ContextVar(
    "agentproxy_combo_id", default=None
)
_agent_id_var: ContextVar[Optional[str]] = ContextVar(
    "agentproxy_agent_id", default=None
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_original_sync_send: Optional[Callable] = None
_original_async_send: Optional[Callable] = None
_installed = False

# Cache instance (None means caching disabled)
_cache: Optional[ResponseCache] = None

# URL path patterns that indicate LLM API endpoints
_LLM_PATH_PATTERNS = ("/chat/completions", "/v1/messages", "/v1/responses")


def _is_llm_request(request: httpx.Request) -> bool:
    """Check if this request targets a known LLM API endpoint."""
    if request.method != "POST":
        return False
    path = request.url.raw_path.decode("ascii", errors="ignore")
    return any(pattern in path for pattern in _LLM_PATH_PATTERNS)


def _parse_usage(body: dict) -> Optional[dict]:
    """Extract model and token usage from an LLM API response body.

    Handles both OpenAI format (prompt_tokens/completion_tokens)
    and Anthropic format (input_tokens/output_tokens).
    """
    model = body.get("model")
    usage = body.get("usage")
    if not model or not usage:
        return None

    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion_tokens = (
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )

    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _get_request_body(request: httpx.Request) -> Optional[dict]:
    """Parse and cache the JSON body of an httpx.Request.

    The parsed body is stored in request.extensions so that repeated calls
    for the same request instance do not re-deserialize the content.
    """
    cache_key = "_agentproxy_json_body"
    if cache_key in request.extensions:
        return request.extensions[cache_key]

    try:
        body = json.loads(request.content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = None

    request.extensions[cache_key] = body
    return body


def _try_record(
    request: httpx.Request,
    response: httpx.Response,
    latency_seconds: float,
    callback: Callable,
    cached: bool = False,
) -> None:
    """Attempt to parse the response and create a CallRecord."""
    from .models import CallRecord

    try:
        body = json.loads(response.content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    parsed = _parse_usage(body)
    if parsed is None:
        return

    request_body = _get_request_body(request) or {}

    model = request_body.get("model") or parsed["model"]

    record = CallRecord(
        data_id=_data_id_var.get(),
        combo_id=_combo_id_var.get(),
        agent_id=_agent_id_var.get(),
        model=model,
        prompt_tokens=parsed["prompt_tokens"],
        completion_tokens=parsed["completion_tokens"],
        latency_seconds=latency_seconds,
        request_url=str(request.url),
        request_body=request_body,
        response_body=body,
        timestamp=datetime.now(timezone.utc).isoformat(),
        cached=cached,
    )
    callback(record)


def _try_cache_lookup(
    request: httpx.Request,
) -> Optional[tuple[httpx.Response, float]]:
    """Check if a cached response exists for this request.

    Returns ``(response, original_latency_seconds)`` on hit, ``None`` on miss.
    """
    if _cache is None:
        return None

    request_body = _get_request_body(request)
    if request_body is None:
        return None

    key = _make_cache_key(request_body)
    entry = _cache.get(key)
    if entry is None:
        return None

    # Build a synthetic httpx.Response from cached bytes.
    # Strip content-encoding/transfer-encoding: the stored bytes are
    # already decoded, so re-applying these headers causes httpx to
    # attempt double-decompression.
    headers = {
        k: v
        for k, v in entry.response_headers.items()
        if k.lower() not in ("content-encoding", "transfer-encoding")
    }
    response = httpx.Response(
        status_code=200, content=entry.response_bytes, headers=headers,
    )
    response.request = request
    return response, entry.latency_seconds


def _try_cache_store(
    request: httpx.Request, response: httpx.Response, latency_seconds: float = 0.0,
) -> None:
    """Store a successful response in the cache."""
    if _cache is None:
        return

    request_body = _get_request_body(request)
    if request_body is None:
        return

    key = _make_cache_key(request_body)
    _cache.put(
        key,
        CacheEntry(
            response_bytes=response.content,
            response_headers=dict(response.headers),
            latency_seconds=latency_seconds,
        ),
    )


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------


def install(callback: Callable, cache: Optional[ResponseCache] = None,) -> None:
    """Monkey-patch httpx.Client.send and httpx.AsyncClient.send."""
    global _original_sync_send, _original_async_send, _installed
    global _cache

    if _installed:
        return

    _cache = cache

    _original_sync_send = httpx.Client.send
    _original_async_send = httpx.AsyncClient.send

    def _patched_sync_send(
        self: Any, request: httpx.Request, *, stream: bool = False, **kwargs: Any
    ) -> httpx.Response:
        if not _is_llm_request(request):
            return _original_sync_send(self, request, stream=stream, **kwargs)

        # Cache lookup (only for non-streaming requests)
        if not stream:
            cache_hit = _try_cache_lookup(request)
            if cache_hit is not None:
                cached_response, original_latency = cache_hit
                _try_record(
                    request, cached_response, original_latency, callback, cached=True
                )
                return cached_response

        t0 = time.monotonic()
        response = _original_sync_send(self, request, stream=stream, **kwargs)
        latency = time.monotonic() - t0

        if not stream and response.status_code == 200:
            try:
                response.read()
                _try_cache_store(request, response, latency)
                _try_record(request, response, latency, callback)
            except Exception:
                pass

        return response

    async def _patched_async_send(
        self: Any, request: httpx.Request, *, stream: bool = False, **kwargs: Any
    ) -> httpx.Response:
        if not _is_llm_request(request):
            return await _original_async_send(self, request, stream=stream, **kwargs)

        # Cache lookup (only for non-streaming requests)
        if not stream:
            cache_hit = _try_cache_lookup(request)
            if cache_hit is not None:
                cached_response, original_latency = cache_hit
                _try_record(
                    request, cached_response, original_latency, callback, cached=True
                )
                return cached_response

        t0 = time.monotonic()
        response = await _original_async_send(self, request, stream=stream, **kwargs)
        latency = time.monotonic() - t0

        if not stream and response.status_code == 200:
            try:
                await response.aread()
                _try_cache_store(request, response, latency)
                _try_record(request, response, latency, callback)
            except Exception:
                pass

        return response

    httpx.Client.send = _patched_sync_send  # type: ignore[assignment]
    httpx.AsyncClient.send = _patched_async_send  # type: ignore[assignment]
    _installed = True


def uninstall() -> None:
    """Restore original httpx send methods."""
    global _original_sync_send, _original_async_send, _installed
    global _cache

    if not _installed:
        return

    if _original_sync_send is not None:
        httpx.Client.send = _original_sync_send  # type: ignore[assignment]
    if _original_async_send is not None:
        httpx.AsyncClient.send = _original_async_send  # type: ignore[assignment]

    _original_sync_send = None
    _original_async_send = None
    _cache = None
    _installed = False
