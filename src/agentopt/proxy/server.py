"""Dual-mode HTTP proxy server for transparent LLM call interception.

Runs as a background daemon thread.  Each tracking session gets its own
TCP port — the port IS the session identity.  This eliminates session-ID
parsing from URLs and makes both in-process and subprocess agents work
through the same mechanism.

Two modes per session port:

* **Direct mode** — in-process agents.  The httpx patch rewrites URLs to
  ``http://127.0.0.1:{session_port}/...`` with an ``X-AgentOpt-Target``
  header.  Traffic arrives as plaintext HTTP.

* **CONNECT mode** — subprocess/Docker agents.  The agent's HTTP client
  honours ``HTTPS_PROXY=http://127.0.0.1:{session_port}`` and sends
  ``CONNECT hostname:443``.  The proxy terminates TLS using a per-hostname
  certificate from the local CA.
"""

import asyncio
import fnmatch
import json
import logging
import ssl
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from .cache import CacheEntry, ResponseCache, _make_cache_key
from .certs import CertificateAuthority
from .models import CallRecord
from .providers import (
    DEFAULT_PROVIDERS,
    INTERCEPT_HOSTS,
    INTERCEPT_PATTERNS,
    TARGET_HEADER,
    ProviderConfig,
    resolve_target,
)
from .session import SessionInfo, SessionManager

logger = logging.getLogger(__name__)

# Headers that must not be forwarded to the upstream provider.
_HOP_BY_HOP = frozenset(
    {
        "host",
        "transfer-encoding",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-connection",
        TARGET_HEADER,
    }
)


# OpenAI-compatible chat/completions paths. Used to detect the streaming
# `stream_options.include_usage` quirk (Anthropic uses `/v1/messages` and
# always emits usage, so it's excluded).
_OPENAI_COMPAT_PATHS = ("/v1/chat/completions", "/v1/completions", "/chat/completions")


def _is_openai_compatible_url(url: str) -> bool:
    """True when *url* targets an OpenAI-style chat/completions endpoint."""
    path = urlparse(url).path or ""
    return any(path.endswith(p) for p in _OPENAI_COMPAT_PATHS)


def _has_include_usage(request_body: dict) -> bool:
    """True when the request opts in to streaming usage frames."""
    opts = request_body.get("stream_options")
    return isinstance(opts, dict) and opts.get("include_usage") is True


def _parse_usage(body: dict) -> Optional[dict]:
    """Extract model and token usage from an LLM API response body."""
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
    """Dual-mode (Direct + CONNECT) HTTP proxy for LLM API calls.

    Each tracking session gets a dedicated TCP port via
    :meth:`open_session_port`.  All traffic on that port is attributed
    to the session — no session IDs in URLs or ContextVars needed.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        cache: Optional[ResponseCache] = None,
        ca: Optional[CertificateAuthority] = None,
        providers: Optional[Dict[str, ProviderConfig]] = None,
    ) -> None:
        self.session_manager = SessionManager()
        self._host = host
        self._cache = cache
        self._ca = ca
        self._providers: Dict[str, ProviderConfig] = dict(
            providers or DEFAULT_PROVIDERS
        )
        # Per-instance CONNECT intercept set — seeded from module defaults.
        # register_provider() extends this with hostnames from custom providers.
        self._intercept_hosts: Set[str] = set(INTERCEPT_HOSTS)
        self._intercept_patterns: List[str] = list(INTERCEPT_PATTERNS)

        # Hostnames we've already warned about for the OpenAI streaming
        # `stream_options.include_usage` quirk. Deduped to avoid log spam.
        self._openai_stream_usage_warned: Set[str] = set()

        # Lifecycle
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        self._started = threading.Event()

        # Per-session listeners: session_id -> (asyncio.Server, port)
        self._session_listeners: Dict[str, Tuple[asyncio.Server, int]] = {}
        # Per-session connection tasks for clean cancellation.
        self._session_tasks: Dict[str, List[asyncio.Task]] = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def register_provider(
        self, name: str, base_url: str, path_patterns: tuple,
    ) -> None:
        """Add or replace a provider in the registry.

        Registers the provider for both modes:
        * **Direct mode** — adds to path-pattern auto-detection registry.
        * **CONNECT mode** — adds the hostname from *base_url* to the
          CONNECT intercept set, so subprocess agents routing through
          ``HTTPS_PROXY`` get TLS-terminated and tracked.
        """
        self._providers[name] = ProviderConfig(
            name=name, base_url=base_url, path_patterns=path_patterns,
        )
        hostname = urlparse(base_url).hostname
        if hostname:
            self._intercept_hosts.add(hostname)

    def _should_intercept(self, hostname: str) -> bool:
        """Return ``True`` if CONNECT traffic to *hostname* should be intercepted."""
        if hostname in self._intercept_hosts:
            return True
        return any(fnmatch.fnmatch(hostname, pat) for pat in self._intercept_patterns)

    # ------------------------------------------------------------------
    # Session ports
    # ------------------------------------------------------------------

    def open_session_port(self, session_id: str) -> int:
        """Bind a new port for *session_id*.  Returns the port number.

        Thread-safe — can be called from the main thread while the proxy
        event loop runs in its background thread.
        """
        assert self._loop is not None, "call start() first"
        future = asyncio.run_coroutine_threadsafe(
            self._open_session_port_async(session_id), self._loop,
        )
        return future.result(timeout=5)

    def close_session_port(self, session_id: str) -> None:
        """Close the port for *session_id*.  Thread-safe."""
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(
            self._close_session_port_async(session_id), self._loop,
        )
        future.result(timeout=5)

    async def _open_session_port_async(self, session_id: str) -> int:
        session = self.session_manager.get_session(session_id)
        assert session is not None, f"session {session_id} not found"
        self._session_tasks[session_id] = []

        def _on_connect(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
            task = asyncio.ensure_future(self._handle_session_connection(session, r, w))
            self._session_tasks.get(session_id, []).append(task)
            task.add_done_callback(
                lambda t: self._session_tasks.get(session_id, []).remove(t)
                if t in self._session_tasks.get(session_id, [])
                else None
            )

        # reuse_address=True lets us bind a fresh ephemeral port immediately
        # even when a recently closed one is still in TIME_WAIT — prevents
        # port exhaustion during long sweeps with many sessions.
        server = await asyncio.start_server(
            _on_connect, self._host, 0, reuse_address=True,
        )
        port = server.sockets[0].getsockname()[1]
        self._session_listeners[session_id] = (server, port)
        logger.debug("Session %s listening on port %d", session_id, port)
        return port

    async def _close_session_port_async(self, session_id: str) -> None:
        entry = self._session_listeners.pop(session_id, None)
        if entry is not None:
            server, port = entry
            server.close()
            await server.wait_closed()
            # Cancel any pending connection tasks (e.g. keep-alive waits).
            for task in self._session_tasks.pop(session_id, []):
                task.cancel()
            logger.debug("Session %s port %d closed", session_id, port)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the proxy in a background daemon thread."""
        if self._thread is not None:
            return
        self._started.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="agentopt-proxy",
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

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        self._shutdown_event = asyncio.Event()
        self._started.set()
        logger.info("Proxy server started")
        await self._shutdown_event.wait()

        # Close all session listeners.
        for session_id in list(self._session_listeners):
            await self._close_session_port_async(session_id)

    # ------------------------------------------------------------------
    # Session connection handler
    # ------------------------------------------------------------------

    async def _handle_session_connection(
        self,
        session: SessionInfo,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a connection on a session port.

        Dispatches to Direct or CONNECT mode.  Loops for HTTP keep-alive.
        """
        try:
            while True:
                first_line = await reader.readline()
                if not first_line:
                    break
                line = first_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    break
                method = parts[0].upper()
                if method == "CONNECT":
                    await self._handle_connect(session, line, reader, writer)
                    break
                else:
                    await self._handle_direct(session, first_line, reader, writer)
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        except Exception:
            logger.exception("Error handling connection")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ==================================================================
    # CONNECT MODE
    # ==================================================================

    async def _handle_connect(
        self,
        session: SessionInfo,
        request_line: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle an HTTP CONNECT tunnel request."""
        parts = request_line.split()
        if len(parts) < 2:
            writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            await writer.drain()
            return

        target = parts[1]
        if ":" in target:
            hostname, port_str = target.rsplit(":", 1)
            remote_port = int(port_str)
        else:
            hostname = target
            remote_port = 443

        # Drain remaining headers.
        while True:
            header_line = await reader.readline()
            if header_line in (b"\r\n", b"\n", b""):
                break

        if not self._should_intercept(hostname) or self._ca is None:
            await self._connect_passthrough(hostname, remote_port, reader, writer)
            return

        # Respond 200 to establish the tunnel.
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()

        # TLS termination in a blocking thread.
        ssl_ctx = self._ca.get_server_context(hostname)
        raw_sock = writer.transport.get_extra_info("socket").dup()
        writer.transport.close()

        await asyncio.get_event_loop().run_in_executor(
            None,
            self._handle_connect_blocking,
            raw_sock,
            ssl_ctx,
            hostname,
            remote_port,
            session,
        )

    async def _connect_passthrough(
        self,
        hostname: str,
        port: int,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ) -> None:
        """Tunnel traffic through without TLS termination."""
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(
                hostname, port,
            )
        except Exception as exc:
            client_writer.write(f"HTTP/1.1 502 Bad Gateway\r\n\r\n{exc}".encode())
            await client_writer.drain()
            return

        client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await client_writer.drain()

        async def _relay(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            try:
                while True:
                    data = await src.read(65536)
                    if not data:
                        break
                    dst.write(data)
                    await dst.drain()
            except Exception:
                pass
            finally:
                try:
                    dst.close()
                except Exception:
                    pass

        await asyncio.gather(
            _relay(client_reader, upstream_writer),
            _relay(upstream_reader, client_writer),
        )

    def _handle_connect_blocking(
        self,
        raw_sock,
        ssl_ctx: ssl.SSLContext,
        hostname: str,
        remote_port: int,
        session: SessionInfo,
    ) -> None:
        """Handle a CONNECT tunnel using blocking I/O in a thread."""
        import http.client

        raw_sock.setblocking(True)
        try:
            ssl_sock = ssl_ctx.wrap_socket(raw_sock, server_side=True)
        except (ssl.SSLError, ConnectionError, OSError) as exc:
            logger.warning("TLS handshake failed for %s: %s", hostname, exc)
            raw_sock.close()
            return

        try:
            rfile = ssl_sock.makefile("rb")
            wfile = ssl_sock.makefile("wb")

            while True:
                request_line = rfile.readline()
                if not request_line:
                    break
                line = request_line.decode("utf-8", errors="replace").strip()
                if not line:
                    break

                parts = line.split(" ", 2)
                if len(parts) < 2:
                    break
                method, path = parts[0], parts[1]

                headers: Dict[str, str] = {}
                while True:
                    hdr = rfile.readline()
                    if hdr in (b"\r\n", b"\n", b""):
                        break
                    decoded = hdr.decode("utf-8", errors="replace").strip()
                    if ":" in decoded:
                        k, v = decoded.split(":", 1)
                        headers[k.strip().lower()] = v.strip()

                content_length = int(headers.get("content-length", "0"))
                body = rfile.read(content_length) if content_length > 0 else b""

                try:
                    request_body = json.loads(body) if body else {}
                except (json.JSONDecodeError, UnicodeDecodeError):
                    request_body = {}

                is_streaming = request_body.get("stream", False)
                upstream_url = f"https://{hostname}{path}"

                # Cache check.
                if self._cache is not None and request_body:
                    cache_key = _make_cache_key(request_body)
                    entry = self._cache.get(cache_key)
                    if entry is not None:
                        if is_streaming:
                            self._record_streaming_call(
                                session=session,
                                request_body=request_body,
                                raw_sse=entry.response_bytes,
                                request_url=upstream_url,
                                latency=entry.latency_seconds,
                                cached=True,
                            )
                        else:
                            self._record_call(
                                session=session,
                                request_body=request_body,
                                response_body_bytes=entry.response_bytes,
                                request_url=upstream_url,
                                latency=entry.latency_seconds,
                                cached=True,
                            )
                        wfile.write(
                            _build_http_response(
                                200, entry.response_headers, entry.response_bytes,
                            )
                        )
                        wfile.flush()
                        continue

                fwd_headers = {
                    k: v
                    for k, v in headers.items()
                    if k not in _HOP_BY_HOP and k != "accept-encoding"
                }
                fwd_headers["host"] = hostname

                t0 = time.monotonic()
                try:
                    conn = http.client.HTTPSConnection(
                        hostname, remote_port, timeout=120
                    )
                    conn.request(method, path, body=body or None, headers=fwd_headers)
                    resp = conn.getresponse()
                    resp_status = resp.status
                    resp_headers = {k.lower(): v for k, v in resp.getheaders()}
                    resp_body = resp.read()
                    conn.close()
                except Exception as exc:
                    latency = time.monotonic() - t0
                    logger.warning(
                        "Upstream request to %s failed: %s", upstream_url, exc
                    )
                    # Record the failed attempt for metrics.
                    self._record_call(
                        session=session,
                        request_body=request_body,
                        response_body_bytes=None,
                        request_url=upstream_url,
                        latency=latency,
                        cached=False,
                        status_code=0,
                        error=str(exc),
                    )
                    error_body = json.dumps({"error": str(exc)}).encode()
                    wfile.write(_build_http_response(502, {}, error_body))
                    wfile.flush()
                    continue
                latency = time.monotonic() - t0

                # Always record — 2xx, 4xx, 5xx all count toward cost/latency.
                if is_streaming and resp_status == 200:
                    self._record_streaming_call(
                        session=session,
                        request_body=request_body,
                        raw_sse=resp_body,
                        request_url=upstream_url,
                        latency=latency,
                    )
                else:
                    self._record_call(
                        session=session,
                        request_body=request_body,
                        response_body_bytes=resp_body,
                        request_url=upstream_url,
                        latency=latency,
                        cached=False,
                        status_code=resp_status,
                        error=None if resp_status == 200 else f"HTTP {resp_status}",
                    )
                if resp_status == 200 and self._cache is not None and request_body:
                    cache_key = _make_cache_key(request_body)
                    self._cache.put(
                        cache_key,
                        CacheEntry(
                            response_bytes=resp_body,
                            response_headers=dict(resp_headers),
                            latency_seconds=latency,
                        ),
                    )

                safe_headers = {
                    k: v
                    for k, v in resp_headers.items()
                    if k not in ("transfer-encoding", "content-encoding")
                }
                wfile.write(_build_http_response(resp_status, safe_headers, resp_body))
                wfile.flush()

        except (ConnectionError, OSError, BrokenPipeError):
            pass
        finally:
            try:
                ssl_sock.close()
            except Exception:
                pass

    # ==================================================================
    # DIRECT MODE
    # ==================================================================

    async def _handle_direct(
        self,
        session: SessionInfo,
        first_line: bytes,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a plaintext HTTP request (direct mode from httpx patch)."""
        line = first_line.decode("utf-8", errors="replace").strip()
        parts = line.split(" ", 2)
        if len(parts) < 2:
            writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            await writer.drain()
            return

        method = parts[0]
        raw_path = parts[1]  # e.g. /v1/chat/completions

        # Read headers.
        headers: Dict[str, str] = {}
        while True:
            header_line = await reader.readline()
            if header_line in (b"\r\n", b"\n", b""):
                break
            decoded = header_line.decode("utf-8", errors="replace").strip()
            if ":" in decoded:
                key, val = decoded.split(":", 1)
                headers[key.strip().lower()] = val.strip()

        # Read body.
        content_length = int(headers.get("content-length", "0"))
        body = b""
        if content_length > 0:
            body = await reader.readexactly(content_length)

        # Parse body.
        try:
            request_body = json.loads(body) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            request_body = {}

        is_streaming = request_body.get("stream", False)

        # Resolve upstream target.
        try:
            target_base, upstream_path = resolve_target(
                raw_path, headers, self._providers,
            )
        except ValueError as exc:
            resp = json.dumps({"error": str(exc)}).encode()
            writer.write(
                _build_http_response(502, {"content-type": "application/json"}, resp)
            )
            await writer.drain()
            return

        upstream_url = target_base.rstrip("/") + upstream_path

        # Cache check.
        if self._cache is not None and request_body:
            cache_key = _make_cache_key(request_body)
            entry = self._cache.get(cache_key)
            if entry is not None:
                if is_streaming:
                    self._record_streaming_call(
                        session=session,
                        request_body=request_body,
                        raw_sse=entry.response_bytes,
                        request_url=upstream_url,
                        latency=entry.latency_seconds,
                        cached=True,
                    )
                else:
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
                writer.write(
                    _build_http_response(200, resp_headers, entry.response_bytes)
                )
                await writer.drain()
                return

        # Build upstream headers.
        fwd_headers: Dict[str, str] = {}
        for k, v in headers.items():
            if k not in _HOP_BY_HOP:
                fwd_headers[k] = v

        # Forward request.
        t0 = time.monotonic()
        try:
            status, resp_headers, resp_body = await _forward_https(
                method, upstream_url, fwd_headers, body, is_streaming,
            )
        except Exception as exc:
            latency = time.monotonic() - t0
            logger.warning("Upstream request failed: %s", exc)
            # Record the failed attempt so latency / error counts show up in metrics.
            self._record_call(
                session=session,
                request_body=request_body,
                response_body_bytes=None,
                request_url=upstream_url,
                latency=latency,
                cached=False,
                status_code=0,
                error=str(exc),
            )
            resp = json.dumps({"error": f"upstream error: {exc}"}).encode()
            writer.write(
                _build_http_response(502, {"content-type": "application/json"}, resp)
            )
            await writer.drain()
            return
        latency = time.monotonic() - t0

        if is_streaming:
            writer.write(_build_http_response(status, resp_headers, resp_body))
            await writer.drain()
            self._record_streaming_call(
                session=session,
                request_body=request_body,
                raw_sse=resp_body,
                request_url=upstream_url,
                latency=latency,
            )
        else:
            # Always record — 2xx, 4xx, 5xx all count toward cost/latency.
            self._record_call(
                session=session,
                request_body=request_body,
                response_body_bytes=resp_body,
                request_url=upstream_url,
                latency=latency,
                cached=False,
                status_code=status,
                error=None if status == 200 else f"HTTP {status}",
            )
            if status == 200 and self._cache is not None and request_body:
                cache_key = _make_cache_key(request_body)
                self._cache.put(
                    cache_key,
                    CacheEntry(
                        response_bytes=resp_body,
                        response_headers=dict(resp_headers),
                        latency_seconds=latency,
                    ),
                )

            safe_headers = {
                k: v
                for k, v in resp_headers.items()
                if k.lower() not in ("transfer-encoding", "content-encoding")
            }
            writer.write(_build_http_response(status, safe_headers, resp_body))
            await writer.drain()

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def _record_call(
        self,
        session: Any,
        request_body: dict,
        response_body_bytes: Optional[bytes],
        request_url: str,
        latency: float,
        cached: bool,
        status_code: int = 200,
        error: Optional[str] = None,
    ) -> None:
        """Record a ``CallRecord`` for any outcome (success or failure).

        Token counts are populated if the response body parses and contains
        a ``usage`` object; otherwise they default to 0.  Failed requests
        (non-200, connection errors) are still recorded so cost/latency
        metrics reflect real work done.
        """
        prompt_tokens = 0
        completion_tokens = 0
        resp_body: Dict[str, Any] = {}
        model = request_body.get("model", "unknown")

        if response_body_bytes:
            try:
                parsed_body = json.loads(response_body_bytes)
                if isinstance(parsed_body, dict):
                    resp_body = parsed_body
                    parsed = _parse_usage(parsed_body)
                    if parsed is not None:
                        prompt_tokens = parsed["prompt_tokens"]
                        completion_tokens = parsed["completion_tokens"]
                        model = request_body.get("model") or parsed["model"]
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

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
            response_body=resp_body,
            timestamp=datetime.now(timezone.utc).isoformat(),
            cached=cached,
            status_code=status_code,
            error=error,
        )
        self.session_manager.add_record(session.session_id, record)

    def _record_streaming_call(
        self,
        session: Any,
        request_body: dict,
        raw_sse: bytes,
        request_url: str,
        latency: float,
        cached: bool = False,
    ) -> None:
        """Best-effort usage extraction from accumulated SSE data."""
        text = raw_sse.decode("utf-8", errors="replace")
        model = request_body.get("model", "unknown")
        prompt_tokens = 0
        completion_tokens = 0

        # Scan all SSE lines to find usage from both message_start (input)
        # and message_delta (output).  Anthropic splits usage across events:
        #   message_start  → message.usage (input_tokens, cache_read/creation)
        #   message_delta  → usage (output_tokens, plus input echo)
        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload.strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            # Check top-level usage (message_delta)
            usage = chunk.get("usage")
            # Also check message.usage (message_start)
            msg = chunk.get("message")
            if isinstance(msg, dict) and msg.get("usage"):
                usage = usage or msg["usage"]

            if usage:
                inp = (
                    (usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                    + (usage.get("cache_read_input_tokens") or 0)
                    + (usage.get("cache_creation_input_tokens") or 0)
                )
                out = usage.get("completion_tokens") or usage.get("output_tokens") or 0
                if inp > prompt_tokens:
                    prompt_tokens = inp
                if out > completion_tokens:
                    completion_tokens = out
                model = chunk.get("model", model)

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
            cached=cached,
        )
        self.session_manager.add_record(session.session_id, record)

        # Warn once per hostname when an OpenAI-compatible streaming call
        # comes back with 0 tokens because the request is missing
        # `stream_options.include_usage: true`. OpenAI-style endpoints omit
        # the final usage frame by default, so the proxy has nothing to
        # parse. Anthropic streams aren't affected (they always emit usage).
        if (
            prompt_tokens == 0
            and completion_tokens == 0
            and _is_openai_compatible_url(request_url)
            and request_body.get("stream") is True
            and not _has_include_usage(request_body)
        ):
            hostname = urlparse(request_url).hostname or request_url
            if hostname not in self._openai_stream_usage_warned:
                self._openai_stream_usage_warned.add(hostname)
                logger.warning(
                    "agentopt: streaming request to %s returned 0 tokens. "
                    "OpenAI-compatible streams omit usage by default — set "
                    '`stream_options={"include_usage": True}` on the request '
                    "to enable token tracking. (Anthropic streams are unaffected.)",
                    hostname,
                )


# ======================================================================
# HTTP helpers
# ======================================================================


def _build_http_response(status: int, headers: Dict[str, str], body: bytes,) -> bytes:
    """Build a raw HTTP/1.1 response as bytes."""
    from http import HTTPStatus

    try:
        status_text = HTTPStatus(status).phrase
    except ValueError:
        status_text = "Unknown"
    lines: List[str] = [f"HTTP/1.1 {status} {status_text}"]
    header_lower = {k.lower(): v for k, v in headers.items()}
    if "content-length" not in header_lower:
        headers["content-length"] = str(len(body))
    for k, v in headers.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("")
    return "\r\n".join(lines).encode("utf-8") + body


async def _forward_https(
    method: str, url: str, headers: Dict[str, str], body: bytes, is_streaming: bool,
) -> Tuple[int, Dict[str, str], bytes]:
    """Forward an HTTP request to an upstream HTTPS URL."""
    import aiohttp

    # Keep trust_env=True so aiohttp picks up user's SSL_CERT_FILE /
    # REQUESTS_CA_BUNDLE (needed behind corporate MITM).  Override proxy
    # per-request to None so our own HTTPS_PROXY doesn't loop back.
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method=method,
            url=url,
            headers=headers,
            data=body if body else None,
            proxy=None,
        ) as resp:
            resp_body = await resp.read()
            resp_headers = {k: v for k, v in resp.headers.items()}
            return resp.status, resp_headers, resp_body
