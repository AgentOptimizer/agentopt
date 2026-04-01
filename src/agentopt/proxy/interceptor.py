"""httpx and botocore monkey-patching for LLM call interception."""

import json
import re
import time
import warnings
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

# Optional botocore import for Bedrock support
try:
    import botocore.httpsession as _botocore_http

    _HAS_BOTOCORE = True
except ImportError:
    _botocore_http = None  # type: ignore[assignment]
    _HAS_BOTOCORE = False

from .cache import CacheEntry, ResponseCache, _make_cache_key

# ---------------------------------------------------------------------------
# ContextVars for attribution
# ---------------------------------------------------------------------------

_data_id_var: ContextVar[Optional[str]] = ContextVar("agentopt_data_id", default=None)
_combo_id_var: ContextVar[Optional[str]] = ContextVar("agentopt_combo_id", default=None)
_agent_id_var: ContextVar[Optional[str]] = ContextVar("agentopt_agent_id", default=None)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_original_sync_send: Optional[Callable] = None
_original_async_send: Optional[Callable] = None
_original_botocore_send: Optional[Callable] = None
_installed = False

# Cache instance (None means caching disabled)
_cache: Optional[ResponseCache] = None

# Warn-once flags for streaming limitations
_streaming_cache_warned: bool = False
_streaming_tokens_warned: bool = False

# URL path patterns that indicate LLM API endpoints
_LLM_PATH_PATTERNS = ("/chat/completions", "/v1/messages", "/v1/responses")


def _is_llm_request(request: httpx.Request) -> bool:
    """Check if this request targets a known LLM API endpoint."""
    if request.method != "POST":
        return False
    path = request.url.raw_path.decode("ascii", errors="ignore")
    return any(pattern in path for pattern in _LLM_PATH_PATTERNS)


# Regex to extract model ID from Bedrock Converse URL path
_BEDROCK_MODEL_RE = re.compile(r"/model/([^/]+)/converse")


def _is_bedrock_request(url: str) -> bool:
    """Check if a URL targets the Bedrock Runtime Converse API."""
    return "bedrock-runtime" in url and "/model/" in url and "/converse" in url


def _extract_bedrock_model(url: str) -> Optional[str]:
    """Extract the model ID from a Bedrock Converse API URL."""
    from urllib.parse import unquote

    m = _BEDROCK_MODEL_RE.search(url)
    return unquote(m.group(1)) if m else None


def _parse_bedrock_usage(body: dict) -> Optional[dict]:
    """Extract token usage from a Bedrock Converse response.

    Bedrock uses camelCase: ``inputTokens`` / ``outputTokens``.
    """
    usage = body.get("usage")
    if not usage:
        return None

    prompt_tokens = usage.get("inputTokens", 0)
    completion_tokens = usage.get("outputTokens", 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


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
    cache_key = "_agentopt_json_body"
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


def _try_record_bedrock(
    url: str,
    request_body: dict,
    response_body: dict,
    latency_seconds: float,
    callback: Callable,
    cached: bool = False,
) -> None:
    """Attempt to parse a Bedrock response and create a CallRecord."""
    from .models import CallRecord

    parsed = _parse_bedrock_usage(response_body)
    if parsed is None:
        return

    model = _extract_bedrock_model(url) or "unknown"

    # Extract server-side latency from Bedrock metrics
    metrics = response_body.get("metrics", {})
    server_latency_ms = metrics.get("latencyMs")

    record = CallRecord(
        data_id=_data_id_var.get(),
        combo_id=_combo_id_var.get(),
        agent_id=_agent_id_var.get(),
        model=model,
        prompt_tokens=parsed["prompt_tokens"],
        completion_tokens=parsed["completion_tokens"],
        latency_seconds=latency_seconds,
        request_url=url,
        request_body=request_body,
        response_body=response_body,
        timestamp=datetime.now(timezone.utc).isoformat(),
        cached=cached,
        server_latency_ms=server_latency_ms,
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
            request_body=request_body,
        ),
    )


# ---------------------------------------------------------------------------
# Bedrock cache helpers
# ---------------------------------------------------------------------------


def _try_bedrock_cache_lookup(
    url: str,
    request_body: dict,
) -> Optional[tuple[dict, float]]:
    """Check for a cached Bedrock response.

    Returns ``(response_body_dict, original_latency)`` on hit, ``None`` on miss.
    """
    if _cache is None:
        return None

    # Include URL in key so different models don't share cache entries
    # (Bedrock puts model ID in URL, not request body)
    key_body = {**request_body, "_bedrock_url": url}
    key = _make_cache_key(key_body)
    entry = _cache.get(key)
    if entry is None:
        return None

    try:
        body = json.loads(entry.response_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    return body, entry.latency_seconds


def _try_bedrock_cache_store(
    url: str,
    request_body: dict,
    response_bytes: bytes,
    latency_seconds: float = 0.0,
) -> None:
    """Store a Bedrock response in the cache."""
    if _cache is None:
        return

    # Include URL in key so different models don't share cache entries
    key_body = {**request_body, "_bedrock_url": url}
    key = _make_cache_key(key_body)
    _cache.put(
        key,
        CacheEntry(
            response_bytes=response_bytes,
            response_headers={},
            latency_seconds=latency_seconds,
            request_body={**request_body, "_bedrock_url": url},
        ),
    )


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------


def install(callback: Callable, cache: Optional[ResponseCache] = None,) -> None:
    """Monkey-patch httpx.Client.send, httpx.AsyncClient.send, and
    optionally botocore.httpsession.URLLib3Session.send."""
    global _original_sync_send, _original_async_send, _original_botocore_send
    global _installed, _cache

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
        else:
            warnings.warn("Caching does not support streaming")

        t0 = time.monotonic()
        response = _original_sync_send(self, request, stream=stream, **kwargs)
        latency = time.monotonic() - t0

        if not stream:
            if response.status_code == 200:
                try:
                    response.read()
                    _try_cache_store(request, response, latency)
                    _try_record(request, response, latency, callback)
                except Exception:
                    pass
        else:
            warnings.warn("Token tracking does not support streaming")

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
        else:
            global _streaming_cache_warned
            if not _streaming_cache_warned:
                warnings.warn(
                    "Caching does not support streaming",
                    category=UserWarning,
                    stacklevel=2,
                )
                _streaming_cache_warned = True

        t0 = time.monotonic()
        response = await _original_async_send(self, request, stream=stream, **kwargs)
        latency = time.monotonic() - t0

        if not stream:
            if response.status_code == 200:
                try:
                    await response.aread()
                    _try_cache_store(request, response, latency)
                    _try_record(request, response, latency, callback)
                except Exception:
                    pass
        else:
            global _streaming_tokens_warned
            if not _streaming_tokens_warned:
                warnings.warn(
                    "Token tracking does not support streaming",
                    category=UserWarning,
                    stacklevel=2,
                )
                _streaming_tokens_warned = True

        return response

    httpx.Client.send = _patched_sync_send  # type: ignore[assignment]
    httpx.AsyncClient.send = _patched_async_send  # type: ignore[assignment]

    # --- Botocore patch (optional) ---
    if _HAS_BOTOCORE:
        _original_botocore_send = _botocore_http.URLLib3Session.send

        def _patched_botocore_send(self: Any, request: Any) -> Any:
            import io

            import botocore.awsrequest
            import urllib3

            url = request.url
            if not _is_bedrock_request(url):
                return _original_botocore_send(self, request)

            # Parse request body
            try:
                req_body = json.loads(request.body) if request.body else {}
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                req_body = {}

            # Cache lookup
            cache_hit = _try_bedrock_cache_lookup(url, req_body)
            if cache_hit is not None:
                cached_body, original_latency = cache_hit
                _try_record_bedrock(
                    url, req_body, cached_body, original_latency, callback, cached=True,
                )
                # Build a synthetic AWSResponse for botocore
                cached_bytes = json.dumps(cached_body).encode("utf-8")
                raw = urllib3.HTTPResponse(
                    body=io.BytesIO(cached_bytes),
                    headers={"Content-Type": "application/json"},
                    status=200,
                    preload_content=False,
                )
                aws_resp = botocore.awsrequest.AWSResponse(
                    url, 200, {"Content-Type": "application/json"}, raw,
                )
                return aws_resp

            t0 = time.monotonic()
            response = _original_botocore_send(self, request)
            latency = time.monotonic() - t0

            if response.status_code == 200:
                try:
                    resp_bytes = response.content
                    resp_body = json.loads(resp_bytes)

                    _try_bedrock_cache_store(url, req_body, resp_bytes, latency)
                    _try_record_bedrock(url, req_body, resp_body, latency, callback)
                except Exception:
                    pass

            return response

        _botocore_http.URLLib3Session.send = _patched_botocore_send  # type: ignore[assignment]

    _installed = True


def uninstall() -> None:
    """Restore original httpx and botocore send methods."""
    global _original_sync_send, _original_async_send, _original_botocore_send
    global _installed, _cache

    if not _installed:
        return

    if _original_sync_send is not None:
        httpx.Client.send = _original_sync_send  # type: ignore[assignment]
    if _original_async_send is not None:
        httpx.AsyncClient.send = _original_async_send  # type: ignore[assignment]
    if _HAS_BOTOCORE and _original_botocore_send is not None:
        _botocore_http.URLLib3Session.send = _original_botocore_send  # type: ignore[assignment]

    _original_sync_send = None
    _original_async_send = None
    _original_botocore_send = None
    _cache = None
    _installed = False
