"""mitmproxy addon — translates intercepted flows into ``CallRecord``s.

One addon instance per session.  The owning ``SessionMaster`` holds it and
binds it to a single ``SessionInfo``, so every flow this addon sees is
already attributed.

Hook responsibilities:

* ``tls_clienthello`` — let traffic to non-LLM hosts pass through encrypted
  without TLS termination.  Mirrors the current
  ``ProxyServer._connect_passthrough`` decision but at the ClientHello SNI
  layer instead of CONNECT-line parsing.
* ``request`` — JSON-decode the body, look up a cache entry, short-circuit
  with ``flow.response`` on hit so mitmproxy skips the upstream call.
* ``response`` — extract token usage from the response body, build a
  ``CallRecord``, and cache successful 200s.

The fast in-memory work in these hooks (cache dict lookup, JSON parse,
list append) all happens under existing locks in ``ResponseCache`` and
``SessionManager``, so blocking the addon's event loop briefly is safe.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from mitmproxy import http, tls

from .cache import CacheEntry, ResponseCache, _make_cache_key
from .providers import ProviderRegistry
from .recording import Recorder
from .session import SessionInfo
from .usage import has_include_usage, is_openai_compatible_url, is_streaming_request

logger = logging.getLogger(__name__)


# Headers we strip from cached responses before re-emitting them.
# Transfer-Encoding/Content-Encoding don't survive a cache round-trip
# (the cached bytes are already decoded), and Content-Length is set by
# mitmproxy from the body length we hand it.
_RESPONSE_HEADERS_STRIP = frozenset({"transfer-encoding", "content-encoding"})

# Stash key for our request-start timestamp on flow.metadata.  Populated
# in `request`, read in `response` to compute upstream latency.
_T0_KEY = "agentopt.t0"


class AgentoptAddon:
    """Per-session mitmproxy addon."""

    def __init__(
        self,
        session: SessionInfo,
        registry: ProviderRegistry,
        recorder: Recorder,
        cache: Optional[ResponseCache] = None,
    ) -> None:
        self._session = session
        self._registry = registry
        self._recorder = recorder
        self._cache = cache

    # ------------------------------------------------------------------
    # TLS — passthrough non-LLM hosts
    # ------------------------------------------------------------------

    def tls_clienthello(self, data: tls.ClientHelloData) -> None:
        """Skip TLS termination for hosts we don't intercept.

        The client's outbound HTTPS to (say) GitHub or PyPI shouldn't be
        decrypted just because the subprocess routes through our proxy.
        Setting ``ignore_connection`` makes mitmproxy tunnel the rest of
        the bytes encrypted, exactly like the old
        ``ProxyServer._connect_passthrough``.
        """
        sni = data.client_hello.sni
        if not sni:
            # No SNI — modern TLS basically always has it; if it doesn't
            # we can't make a host-based decision, so play it safe and
            # let it pass.
            data.ignore_connection = True
            return
        if not self._registry.should_intercept(sni):
            data.ignore_connection = True

    # ------------------------------------------------------------------
    # Request — cache lookup and short-circuit
    # ------------------------------------------------------------------

    def request(self, flow: http.HTTPFlow) -> None:
        json_body = _decode_json_body(flow.request.content)
        flow.metadata[_T0_KEY] = time.monotonic()

        # OpenAI-compatible streaming: force `stream_options.include_usage`
        # so the SSE response carries a usage frame. Some subprocess agents
        # (OpenHarness's `oh` does this whenever any tool is attached, as a
        # Kimi/Moonshot compat hack) strip it from the request, which
        # otherwise leaves the proxy unable to count tokens.
        if _ensure_openai_include_usage(json_body, flow.request.pretty_url):
            flow.request.content = json.dumps(json_body, separators=(",", ":")).encode(
                "utf-8"
            )

        # Only POSTs with JSON bodies are LLM calls worth caching.
        if self._cache is None or flow.request.method != "POST" or not json_body:
            return

        cache_key = _make_cache_key(json_body, flow.request.path)
        entry = self._cache.get(cache_key)
        if entry is None:
            return

        # Cache hit — synthesize a response and let mitmproxy short-circuit.
        flow.response = http.Response.make(
            200, entry.response_bytes, _safe_response_headers(entry.response_headers),
        )

        # Mark the flow as a cache hit. mitmproxy still invokes `response`
        # for synthesized responses, and that hook does the actual
        # recording using this metadata and the cached latency.
        flow.metadata["agentopt.cached"] = True
        flow.metadata["agentopt.cached_latency"] = entry.latency_seconds

    # ------------------------------------------------------------------
    # Response — record + cache
    # ------------------------------------------------------------------

    def response(self, flow: http.HTTPFlow) -> None:
        if flow.response is None:
            return  # shouldn't happen, but be defensive

        json_body = _decode_json_body(flow.request.content)
        request_url = flow.request.pretty_url
        is_streaming = is_streaming_request(json_body, request_url)

        cached = bool(flow.metadata.get("agentopt.cached"))
        if cached:
            latency = float(flow.metadata.get("agentopt.cached_latency", 0.0))
        else:
            t0 = flow.metadata.get(_T0_KEY)
            latency = (time.monotonic() - t0) if t0 is not None else 0.0

        status = flow.response.status_code
        body_bytes = flow.response.content if flow.response.content else b""
        record_error: Optional[str] = None
        if status != 200:
            record_error = f"HTTP {status}"

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

        # Cache successful 200s on the miss path.
        if not cached and status == 200 and self._cache is not None and json_body:
            self._cache.put(
                _make_cache_key(json_body, flow.request.path),
                CacheEntry(
                    response_bytes=body_bytes,
                    response_headers=dict(flow.response.headers),
                    latency_seconds=latency,
                ),
            )

    # ------------------------------------------------------------------
    # Errors — record connection failures
    # ------------------------------------------------------------------

    def error(self, flow: http.HTTPFlow) -> None:
        """Upstream couldn't be reached / errored mid-flight.

        mitmproxy fires this in place of `response` when the server side
        fails before producing a complete HTTP response.  We still want
        the failed attempt in the ledger.
        """
        if flow.response is not None:
            # An HTTP error is already a complete response — `response`
            # handled it.  This hook fires for transport-level errors.
            return
        json_body = _decode_json_body(flow.request.content)
        request_url = flow.request.pretty_url
        t0 = flow.metadata.get(_T0_KEY)
        latency = (time.monotonic() - t0) if t0 is not None else 0.0
        err = flow.error.msg if flow.error else "upstream error"

        self._recorder.record(
            self._session,
            json_body,
            None,
            request_url,
            latency,
            cached=False,
            is_streaming=is_streaming_request(json_body, request_url),
            status_code=0,
            error=err,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_json_body(body: Optional[bytes]) -> Dict[str, Any]:
    """Best-effort JSON decode of a request body.

    Mirrors the lenient handling in the old ``_read_request`` — non-JSON
    or non-dict bodies become ``{}`` rather than raising, so the proxy
    never blocks a request just because it can't introspect the body.
    """
    if not body:
        return {}
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_response_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Strip headers that don't survive cache-replay."""
    return {
        k: v for k, v in headers.items() if k.lower() not in _RESPONSE_HEADERS_STRIP
    }


def _ensure_openai_include_usage(json_body: Dict[str, Any], request_url: str) -> bool:
    """Inject ``stream_options.include_usage=True`` for OpenAI-compatible streams.

    Returns True if the body was mutated.  No-op for non-streaming
    requests, non-OpenAI-compatible URLs, or bodies that already opted in.
    """
    if not is_openai_compatible_url(request_url):
        return False
    if json_body.get("stream") is not True:
        return False
    if has_include_usage(json_body):
        return False
    opts = json_body.get("stream_options")
    if isinstance(opts, dict):
        opts["include_usage"] = True
    else:
        json_body["stream_options"] = {"include_usage": True}
    return True
