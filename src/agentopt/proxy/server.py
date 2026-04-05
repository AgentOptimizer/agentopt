"""HTTP proxy server for transparent LLM call interception.

Runs as a background daemon thread.  All LLM API traffic — from in-process
agents (via the httpx monkey-patch) and subprocess agents (via env-var
base-URL override) — is routed through this server.

The server handles:
* Request forwarding to the real LLM provider
* Token / usage tracking per session
* Response caching (reuses ``ResponseCache`` from ``agentopt.proxy.cache``)
* Streaming (SSE) passthrough with deferred usage parsing
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import aiohttp
from aiohttp import web

from .cache import CacheEntry, ResponseCache, _make_cache_key
from .models import CallRecord
from .providers import (
    DEFAULT_PROVIDERS,
    TARGET_HEADER,
    ProviderConfig,
    resolve_target,
)
from .session import SessionManager

logger = logging.getLogger(__name__)

# Headers that must not be forwarded to the upstream provider.
_HOP_BY_HOP = frozenset(
    {
        "host",
        "transfer-encoding",
        "content-length",
        "connection",
        "keep-alive",
        TARGET_HEADER,
    }
)


def _parse_usage(body: dict) -> Optional[dict]:
    """Extract model and token usage from an LLM API response body.

    Handles OpenAI (prompt_tokens/completion_tokens) and Anthropic
    (input_tokens/output_tokens) formats.
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


class ProxyServer:
    """Localhost HTTP proxy for LLM API calls.

    Parameters
    ----------
    host : str
        Bind address (default ``"127.0.0.1"`` — localhost only).
    port : int
        TCP port.  ``0`` lets the OS pick a free port (recommended).
    cache : ResponseCache, optional
        Shared response cache instance.  ``None`` disables caching.
    providers : dict, optional
        Provider registry for subprocess auto-detection.  Defaults to
        ``DEFAULT_PROVIDERS`` (OpenAI, Anthropic, Google).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        cache: Optional[ResponseCache] = None,
        providers: Optional[Dict[str, ProviderConfig]] = None,
    ) -> None:
        self.session_manager = SessionManager()
        self._host = host
        self._port = port
        self._cache = cache
        self._providers: Dict[str, ProviderConfig] = dict(
            providers or DEFAULT_PROVIDERS
        )

        # Lifecycle
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        self._started = threading.Event()
        self._actual_port: Optional[int] = None

        # aiohttp client session for upstream requests (created in event loop)
        self._client: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        """``http://127.0.0.1:{port}`` — available after ``start()``."""
        if self._actual_port is None:
            raise RuntimeError("Server not started")
        return f"http://{self._host}:{self._actual_port}"

    def session_url(self, session_id: str) -> str:
        """Base URL that agents should use for a given session."""
        return f"{self.base_url}/{session_id}"

    def register_provider(
        self, name: str, base_url: str, path_patterns: tuple,
    ) -> None:
        """Add or replace a provider in the registry."""
        self._providers[name] = ProviderConfig(
            name=name, base_url=base_url, path_patterns=path_patterns,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the proxy in a background daemon thread.

        Blocks until the server is listening and ready to accept requests.
        """
        if self._thread is not None:
            return

        self._started.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="agentopt-proxy"
        )
        self._thread.start()
        if not self._started.wait(timeout=10):
            raise RuntimeError("Proxy server failed to start within 10 s")

    def stop(self) -> None:
        """Shut down the proxy server and collect remaining sessions."""
        if self._loop is None or self._shutdown_event is None:
            return

        self.session_manager.force_end_all()
        self._loop.call_soon_threadsafe(self._shutdown_event.set)
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._actual_port = None

    # ------------------------------------------------------------------
    # Event-loop thread
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        """Create the aiohttp app, bind, and run until shutdown."""
        self._shutdown_event = asyncio.Event()

        app = web.Application()
        app.router.add_route("*", "/_agentopt/{action}", self._handle_management)
        app.router.add_route("*", "/{session_id}/{path:.*}", self._handle_proxy)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, self._host, self._port)
        await site.start()

        # Capture the actual port (important when port=0).
        sockets = site._server.sockets  # type: ignore[union-attr]
        self._actual_port = sockets[0].getsockname()[1]

        self._client = aiohttp.ClientSession()

        self._started.set()
        logger.info("Proxy server listening on %s", self.base_url)

        await self._shutdown_event.wait()

        await self._client.close()
        self._client = None
        await runner.cleanup()

    # ------------------------------------------------------------------
    # Management endpoints
    # ------------------------------------------------------------------

    async def _handle_management(self, request: web.Request) -> web.Response:
        """``POST /_agentopt/{start_session|end_session}``."""
        action = request.match_info["action"]

        if action == "start_session":
            body = await request.json()
            session = self.session_manager.create_session(
                data_id=body.get("data_id"),
                combo_id=body.get("combo_id"),
                agent_id=body.get("agent_id"),
            )
            return web.json_response(
                {
                    "session_id": session.session_id,
                    "base_url": self.session_url(session.session_id),
                }
            )

        if action == "end_session":
            body = await request.json()
            session = self.session_manager.end_session(body["session_id"])
            if session is None:
                return web.json_response({"error": "session not found"}, status=404)
            usage: Dict[str, Any] = {}
            total_tokens = 0
            for rec in session.records:
                total_tokens += rec.prompt_tokens + rec.completion_tokens
            usage["tokens"] = total_tokens
            usage["calls"] = len(session.records)
            return web.json_response({"usage": usage})

        return web.json_response({"error": f"unknown action {action}"}, status=400)

    # ------------------------------------------------------------------
    # Proxy handler
    # ------------------------------------------------------------------

    async def _handle_proxy(self, request: web.Request) -> web.StreamResponse:
        """Catch-all: ``ANY /{session_id}/{path:.*}``."""
        session_id = request.match_info["session_id"]
        remaining_path = "/" + request.match_info.get("path", "")
        # Preserve query string for providers that use it (e.g. API keys).
        if request.query_string:
            remaining_path += "?" + request.query_string

        session = self.session_manager.get_session(session_id)
        if session is None:
            return web.json_response(
                {"error": f"unknown session {session_id}"}, status=404
            )

        # Read the raw request body.
        raw_body = await request.read()

        # Parse JSON body for cache key / usage detection.
        try:
            request_body = json.loads(raw_body) if raw_body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            request_body = {}

        is_streaming = request_body.get("stream", False)

        # Resolve upstream target.
        headers_dict = {k.lower(): v for k, v in request.headers.items()}
        try:
            target_base, upstream_path = resolve_target(
                remaining_path, headers_dict, self._providers
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=502)

        upstream_url = target_base.rstrip("/") + upstream_path

        # --- Cache check (non-streaming only) ---
        if not is_streaming and self._cache is not None and request_body:
            cache_key = _make_cache_key(request_body)
            entry = self._cache.get(cache_key)
            if entry is not None:
                self._record_call(
                    session=session,
                    request_body=request_body,
                    response_body_bytes=entry.response_bytes,
                    request_url=upstream_url,
                    latency=entry.latency_seconds,
                    cached=True,
                )
                resp_headers = {
                    k: v
                    for k, v in entry.response_headers.items()
                    if k.lower() not in ("content-encoding", "transfer-encoding")
                }
                return web.Response(
                    body=entry.response_bytes, status=200, headers=resp_headers,
                )

        # --- Build upstream headers ---
        fwd_headers: Dict[str, str] = {}
        for k, v in request.headers.items():
            if k.lower() not in _HOP_BY_HOP:
                fwd_headers[k] = v

        # --- Forward request ---
        assert self._client is not None
        t0 = time.monotonic()

        try:
            upstream_resp = await self._client.request(
                method=request.method,
                url=upstream_url,
                headers=fwd_headers,
                data=raw_body,
            )
        except Exception as exc:
            logger.warning("Upstream request failed: %s", exc)
            return web.json_response({"error": f"upstream error: {exc}"}, status=502)

        latency = time.monotonic() - t0

        # --- Streaming response ---
        if is_streaming:
            return await self._proxy_streaming(
                request, upstream_resp, session, request_body, upstream_url, t0
            )

        # --- Non-streaming response ---
        resp_bytes = await upstream_resp.read()
        upstream_resp.release()

        if upstream_resp.status == 200:
            self._record_call(
                session=session,
                request_body=request_body,
                response_body_bytes=resp_bytes,
                request_url=upstream_url,
                latency=latency,
                cached=False,
            )
            # Cache the response.
            if self._cache is not None and request_body:
                cache_key = _make_cache_key(request_body)
                resp_hdrs = dict(upstream_resp.headers)
                self._cache.put(
                    cache_key,
                    CacheEntry(
                        response_bytes=resp_bytes,
                        response_headers=resp_hdrs,
                        latency_seconds=latency,
                    ),
                )

        # Build downstream response — forward status and safe headers.
        resp_headers = {}
        for k, v in upstream_resp.headers.items():
            if k.lower() not in ("transfer-encoding", "content-encoding"):
                resp_headers[k] = v

        return web.Response(
            body=resp_bytes, status=upstream_resp.status, headers=resp_headers,
        )

    # ------------------------------------------------------------------
    # Streaming passthrough
    # ------------------------------------------------------------------

    async def _proxy_streaming(
        self,
        web_request: web.Request,
        upstream_resp: aiohttp.ClientResponse,
        session: Any,
        request_body: dict,
        upstream_url: str,
        t0: float,
    ) -> web.StreamResponse:
        """Forward an SSE stream chunk-by-chunk, accumulate for usage."""
        resp = web.StreamResponse(
            status=upstream_resp.status,
            headers={
                "Content-Type": upstream_resp.headers.get(
                    "Content-Type", "text/event-stream"
                ),
                "Cache-Control": "no-cache",
            },
        )
        await resp.prepare(web_request)

        accumulated: list[bytes] = []
        async for chunk in upstream_resp.content.iter_any():
            accumulated.append(chunk)
            await resp.write(chunk)
        upstream_resp.release()

        latency = time.monotonic() - t0
        full_body = b"".join(accumulated)

        # Best-effort usage extraction from the accumulated SSE data.
        self._record_streaming_call(
            session=session,
            request_body=request_body,
            raw_sse=full_body,
            request_url=upstream_url,
            latency=latency,
        )

        await resp.write_eof()
        return resp

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def _record_call(
        self,
        session: Any,
        request_body: dict,
        response_body_bytes: bytes,
        request_url: str,
        latency: float,
        cached: bool,
    ) -> None:
        """Parse a non-streaming response and record a ``CallRecord``."""
        try:
            resp_body = json.loads(response_body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        parsed = _parse_usage(resp_body)
        if parsed is None:
            return

        model = request_body.get("model") or parsed["model"]

        record = CallRecord(
            data_id=session.data_id,
            combo_id=session.combo_id,
            agent_id=session.agent_id,
            model=model,
            prompt_tokens=parsed["prompt_tokens"],
            completion_tokens=parsed["completion_tokens"],
            latency_seconds=latency,
            request_url=request_url,
            request_body=request_body,
            response_body=resp_body,
            timestamp=datetime.now(timezone.utc).isoformat(),
            cached=cached,
        )
        self.session_manager.add_record(session.session_id, record)

    def _record_streaming_call(
        self,
        session: Any,
        request_body: dict,
        raw_sse: bytes,
        request_url: str,
        latency: float,
    ) -> None:
        """Best-effort usage extraction from accumulated SSE data."""
        # OpenAI includes a final `data: {..., "usage": {...}}` chunk when
        # `stream_options.include_usage` is set.  Anthropic includes usage
        # in the `message_delta` event.  Try to find either.
        text = raw_sse.decode("utf-8", errors="replace")
        model = request_body.get("model", "unknown")
        prompt_tokens = 0
        completion_tokens = 0

        for line in reversed(text.splitlines()):
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload.strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = (
                    usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                )
                completion_tokens = (
                    usage.get("completion_tokens") or usage.get("output_tokens") or 0
                )
                model = chunk.get("model", model)
                break

        record = CallRecord(
            data_id=session.data_id,
            combo_id=session.combo_id,
            agent_id=session.agent_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=latency,
            request_url=request_url,
            request_body=request_body,
            response_body={},
            timestamp=datetime.now(timezone.utc).isoformat(),
            cached=False,
        )
        self.session_manager.add_record(session.session_id, record)
