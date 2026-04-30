"""Dual-mode HTTP proxy for transparent LLM call interception.

Each tracking session gets its own ephemeral TCP port — the port IS the
session identity, so no session IDs in URLs or ContextVars are needed.

Two arrival modes per port:

* **Direct** — in-process agents whose ``httpx`` calls have been rewritten
  by :mod:`agentopt.proxy.interceptor` to ``http://127.0.0.1:{port}/...``
  with an ``X-AgentOpt-Target`` header.  Plaintext.
* **CONNECT** — subprocess agents driving traffic through
  ``HTTPS_PROXY=http://127.0.0.1:{port}``.  The proxy answers
  ``CONNECT host:443`` by TLS-terminating the tunnel using a per-hostname
  certificate from the local CA.

Once a request is parsed, both modes share a single lifecycle:
``cache → forward → record → respond``.

Implementation: blocking I/O, one thread per accepted connection.  Keeps
the code linear and lets us use stdlib ``http.client`` for forwarding —
no asyncio / aiohttp.
"""

from __future__ import annotations

import http.client
import json
import logging
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any, BinaryIO, Dict, Optional, Tuple
from urllib.parse import urlparse

from .cache import CacheEntry, ResponseCache, _make_cache_key
from .certs import CertificateAuthority
from .models import CallRecord
from .providers import (
    DEFAULT_PROVIDERS,  # re-exported for back-compat
    INTERCEPT_HOSTS,  # re-exported for back-compat
    INTERCEPT_PATTERNS,  # re-exported for back-compat
    Provider,
    ProviderConfig,  # re-exported for back-compat
    ProviderRegistry,
    TARGET_HEADER,
    resolve_target,  # re-exported for back-compat
)
from .session import SessionInfo, SessionManager
from .usage import (
    PARSE_FAILED_MODEL,
    UsageExtractionError,
    extract_usage,
    extract_usage_streaming,
    is_streaming_request,
)

logger = logging.getLogger(__name__)


# Headers we strip before forwarding to upstream — connection-management
# meta plus our own routing header.
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


# ---------------------------------------------------------------------------
# Parsed-request value type
# ---------------------------------------------------------------------------


@dataclass
class _Request:
    """One parsed HTTP/1.1 request."""

    method: str
    path: str
    headers: Dict[str, str]
    body: bytes
    json_body: Dict[str, Any]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class ProxyServer:
    """Synchronous, thread-per-connection proxy.

    One listener thread per session port; each accepted connection runs
    in its own thread and handles either CONNECT (TLS termination + LLM
    request loop) or direct (plaintext + LLM request loop).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        cache: Optional[ResponseCache] = None,
        ca: Optional[CertificateAuthority] = None,
        providers: Optional[Dict[str, Provider]] = None,
    ) -> None:
        self.session_manager = SessionManager()
        self._host = host
        self._cache = cache
        self._ca = ca

        self._registry = ProviderRegistry()
        if providers:
            for p in providers.values():
                self._registry.register(p.name, p.base_url, p.path_patterns)
        # Back-compat: tests inspect ``_providers`` for membership.
        self._providers = self._registry.providers

        # session_id -> (listen_sock, listener_thread, port)
        self._listeners: Dict[str, Tuple[socket.socket, threading.Thread, int]] = {}
        self._listeners_lock = threading.Lock()

        # Hostnames we've already warned about for the OpenAI streaming
        # `stream_options.include_usage` quirk.  Deduped to avoid log spam.
        self._stream_warned_hosts: set = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """No-op.  Listeners are bound per-session in ``open_session_port``."""

    def stop(self) -> None:
        """Close every listener socket and archive remaining sessions."""
        self.session_manager.force_end_all()
        with self._listeners_lock:
            entries = list(self._listeners.values())
            self._listeners.clear()
        for sock, _, _ in entries:
            _close_quietly(sock)
        for _, thread, _ in entries:
            thread.join(timeout=2)

    # ------------------------------------------------------------------
    # Provider registry
    # ------------------------------------------------------------------

    def register_provider(
        self, name: str, base_url: str, path_patterns: tuple,
    ) -> None:
        """Add or replace a provider in this server's registry."""
        self._registry.register(name, base_url, path_patterns)

    def _should_intercept(self, hostname: str) -> bool:
        """Whether CONNECT traffic to *hostname* should be TLS-terminated."""
        return self._registry.should_intercept(hostname)

    # ------------------------------------------------------------------
    # Session ports
    # ------------------------------------------------------------------

    def open_session_port(self, session_id: str) -> int:
        """Bind a fresh ephemeral port for *session_id* and start accepting."""
        session = self.session_manager.get_session(session_id)
        assert session is not None, f"session {session_id} not found"

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, 0))
        sock.listen(64)
        port = sock.getsockname()[1]

        thread = threading.Thread(
            target=self._listen_loop,
            args=(sock, session),
            daemon=True,
            name=f"agentopt-listener-{session_id}",
        )
        with self._listeners_lock:
            self._listeners[session_id] = (sock, thread, port)
        thread.start()
        logger.debug("Session %s listening on port %d", session_id, port)
        return port

    def close_session_port(self, session_id: str) -> None:
        """Close *session_id*'s listener.  Connection threads exit on their own."""
        with self._listeners_lock:
            entry = self._listeners.pop(session_id, None)
        if entry is None:
            return
        sock, thread, port = entry
        _close_quietly(sock)
        thread.join(timeout=2)
        logger.debug("Session %s port %d closed", session_id, port)

    def get_session_port(self, session_id: str) -> int:
        """Look up the port bound to *session_id*."""
        with self._listeners_lock:
            entry = self._listeners.get(session_id)
        assert entry is not None, f"no port for session {session_id}"
        return entry[2]

    # ------------------------------------------------------------------
    # Listener + connection dispatch
    # ------------------------------------------------------------------

    def _listen_loop(self, listen_sock: socket.socket, session: SessionInfo,) -> None:
        """Accept loop for one session port."""
        try:
            while True:
                try:
                    client_sock, _ = listen_sock.accept()
                except OSError:
                    return  # listener closed
                threading.Thread(
                    target=self._handle_connection,
                    args=(client_sock, session),
                    daemon=True,
                    name=f"agentopt-conn-{session.session_id}",
                ).start()
        except Exception:
            logger.exception("listener loop crashed")

    def _handle_connection(
        self, client_sock: socket.socket, session: SessionInfo,
    ) -> None:
        """Per-connection top-level: peek first line, dispatch to mode."""
        rfile = wfile = None
        try:
            client_sock.settimeout(120)
            rfile = client_sock.makefile("rb")
            wfile = client_sock.makefile("wb")

            first_line = rfile.readline()
            if not first_line:
                return
            line = first_line.decode("utf-8", errors="replace").strip()
            parts = line.split()
            if len(parts) < 2:
                return

            if parts[0].upper() == "CONNECT":
                self._handle_connect(client_sock, rfile, wfile, line, session)
            else:
                self._direct_request_loop(rfile, wfile, first_line, session)
        except (ConnectionError, OSError, ssl.SSLError, socket.timeout):
            pass
        except Exception:
            logger.exception("connection handler crashed")
        finally:
            for f in (rfile, wfile):
                if f is not None:
                    try:
                        f.close()
                    except Exception:
                        pass
            _close_quietly(client_sock)

    # ==================================================================
    # CONNECT mode
    # ==================================================================

    def _handle_connect(
        self,
        client_sock: socket.socket,
        rfile: BinaryIO,
        wfile: BinaryIO,
        request_line: str,
        session: SessionInfo,
    ) -> None:
        """Handle an HTTP CONNECT tunnel — passthrough or MITM."""
        target = request_line.split()[1]
        if ":" in target:
            host_part, port_part = target.rsplit(":", 1)
            try:
                remote_port = int(port_part)
            except ValueError:
                remote_port = -1
            if not host_part or not (0 < remote_port < 65536):
                _write_response(
                    wfile,
                    400,
                    {"content-type": "application/json"},
                    json.dumps(
                        {"error": f"malformed CONNECT target: {target!r}"}
                    ).encode(),
                )
                return
            hostname = host_part
        else:
            hostname, remote_port = target, 443

        # Drain CONNECT headers.
        while True:
            h = rfile.readline()
            if h in (b"\r\n", b"\n", b""):
                break

        if not self._should_intercept(hostname) or self._ca is None:
            self._connect_passthrough(client_sock, hostname, remote_port)
            return

        # Establish tunnel, then upgrade to TLS.
        wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        wfile.flush()

        ssl_ctx = self._ca.get_server_context(hostname)
        try:
            tls_sock = ssl_ctx.wrap_socket(client_sock, server_side=True)
        except (ssl.SSLError, OSError) as exc:
            logger.warning("TLS handshake failed for %s: %s", hostname, exc)
            return

        tls_rfile = tls_sock.makefile("rb")
        tls_wfile = tls_sock.makefile("wb")
        try:
            while True:
                req = _read_request(tls_rfile)
                if req is None:
                    break
                upstream_url = f"https://{hostname}{req.path}"
                self._process_and_respond(
                    session=session,
                    req=req,
                    wfile=tls_wfile,
                    upstream_url=upstream_url,
                    upstream_hostname=hostname,
                    upstream_port=remote_port,
                    upstream_path=req.path,
                    scheme="https",
                )
        finally:
            for f in (tls_rfile, tls_wfile):
                try:
                    f.close()
                except Exception:
                    pass
            _close_quietly(tls_sock)

    def _connect_passthrough(
        self, client_sock: socket.socket, hostname: str, port: int,
    ) -> None:
        """Tunnel raw bytes through to non-LLM hosts without inspection."""
        try:
            upstream = socket.create_connection((hostname, port), timeout=30)
        except Exception as exc:
            try:
                client_sock.sendall(f"HTTP/1.1 502 Bad Gateway\r\n\r\n{exc}".encode())
            except Exception:
                pass
            return

        try:
            client_sock.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            _bidirectional_relay(client_sock, upstream)
        finally:
            _close_quietly(upstream)

    # ==================================================================
    # Direct mode
    # ==================================================================

    def _direct_request_loop(
        self, rfile: BinaryIO, wfile: BinaryIO, first_line: bytes, session: SessionInfo,
    ) -> None:
        """Handle plaintext direct-mode requests, looping for keep-alive."""
        pending_first_line: Optional[bytes] = first_line
        while True:
            req = _read_request(rfile, first_line=pending_first_line)
            pending_first_line = None
            if req is None:
                return

            try:
                base, upstream_path = self._registry.resolve_target(
                    req.path, req.headers,
                )
            except ValueError as exc:
                _write_response(
                    wfile,
                    502,
                    {"content-type": "application/json"},
                    json.dumps({"error": str(exc)}).encode(),
                )
                continue

            base_url = urlparse(base)
            scheme = base_url.scheme or "https"
            hostname = base_url.hostname or ""
            port = base_url.port or (443 if scheme == "https" else 80)
            self._process_and_respond(
                session=session,
                req=req,
                wfile=wfile,
                upstream_url=base.rstrip("/") + upstream_path,
                upstream_hostname=hostname,
                upstream_port=port,
                upstream_path=upstream_path,
                scheme=scheme,
            )

    # ==================================================================
    # Shared request lifecycle
    # ==================================================================

    def _process_and_respond(
        self,
        *,
        session: SessionInfo,
        req: _Request,
        wfile: BinaryIO,
        upstream_url: str,
        upstream_hostname: str,
        upstream_port: int,
        upstream_path: str,
        scheme: str,
    ) -> None:
        """Cache check → forward → record → respond.

        Used by both direct and CONNECT modes once their request is parsed.
        """
        is_streaming = is_streaming_request(req.json_body, req.path)

        # 1. Cache.
        if self._cache is not None and req.json_body:
            cache_key = _make_cache_key(req.json_body)
            entry = self._cache.get(cache_key)
            if entry is not None:
                self._record(
                    session,
                    req,
                    entry.response_bytes,
                    upstream_url,
                    entry.latency_seconds,
                    cached=True,
                    is_streaming=is_streaming,
                )
                _write_response(
                    wfile,
                    200,
                    _safe_response_headers(entry.response_headers),
                    entry.response_bytes,
                )
                return

        # 2. Forward.
        fwd_headers = _build_forward_headers(
            req.headers, upstream_hostname, upstream_port
        )
        t0 = time.monotonic()
        try:
            status, resp_headers, resp_body = _forward(
                method=req.method,
                scheme=scheme,
                hostname=upstream_hostname,
                port=upstream_port,
                path=upstream_path,
                headers=fwd_headers,
                body=req.body,
            )
        except Exception as exc:
            latency = time.monotonic() - t0
            logger.warning("upstream %s failed: %s", upstream_url, exc)
            self._record(
                session,
                req,
                None,
                upstream_url,
                latency,
                cached=False,
                is_streaming=is_streaming,
                status_code=0,
                error=str(exc),
            )
            _write_response(
                wfile,
                502,
                {"content-type": "application/json"},
                json.dumps({"error": f"upstream error: {exc}"}).encode(),
            )
            return
        latency = time.monotonic() - t0

        # 3. Record (every status — 2xx, 4xx, 5xx all count as work).
        self._record(
            session,
            req,
            resp_body,
            upstream_url,
            latency,
            cached=False,
            is_streaming=is_streaming,
            status_code=status,
            error=None if status == 200 else f"HTTP {status}",
        )

        # 4. Cache successful responses.
        if status == 200 and self._cache is not None and req.json_body:
            self._cache.put(
                _make_cache_key(req.json_body),
                CacheEntry(
                    response_bytes=resp_body,
                    response_headers=dict(resp_headers),
                    latency_seconds=latency,
                ),
            )

        # 5. Respond.
        _write_response(
            wfile, status, _safe_response_headers(resp_headers), resp_body,
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _record(
        self,
        session: SessionInfo,
        req: _Request,
        response_body: Optional[bytes],
        request_url: str,
        latency: float,
        *,
        cached: bool,
        is_streaming: bool,
        status_code: int = 200,
        error: Optional[str] = None,
    ) -> None:
        """Build a ``CallRecord`` and hand it to the session manager."""
        usage = None
        parse_err: Optional[str] = None

        if response_body is None:
            # Failure path — caller already supplied the error message.
            pass
        else:
            try:
                if is_streaming and status_code == 200:
                    usage = extract_usage_streaming(
                        response_body, req.json_body, request_url,
                    )
                else:
                    usage = extract_usage(response_body)
            except UsageExtractionError as exc:
                parse_err = str(exc)

        model: Optional[str] = req.json_body.get("model")
        prompt_tokens = 0
        completion_tokens = 0
        resp_body_for_record: Dict[str, Any] = {}
        if usage is not None:
            model = model or usage.model
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            resp_body_for_record = usage.response_body

        record_error = error
        # Loud surfacing of token-extraction failures on otherwise-successful
        # calls — silent zeros mask real proxy gaps.
        if usage is None and status_code == 200 and not error and parse_err is not None:
            record_error = (
                f"agentopt proxy could not extract token usage: {parse_err}. "
                f"URL={request_url}"
            )
            self._maybe_warn(request_url, parse_err, model)
            if not model:
                model = PARSE_FAILED_MODEL
        elif not model:
            model = "unknown"

        self.session_manager.add_record(
            session.session_id,
            CallRecord(
                data_id=session.data_id,
                combo_id=session.combo_id,
                agent_id=session.agent_id,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_seconds=latency,
                request_url=request_url,
                request_body=req.json_body,
                response_body=resp_body_for_record,
                timestamp=datetime.now(timezone.utc).isoformat(),
                cached=cached,
                status_code=status_code,
                error=record_error,
            ),
        )

    def _maybe_warn(self, request_url: str, reason: str, model: Optional[str],) -> None:
        host = urlparse(request_url).hostname or request_url
        if host in self._stream_warned_hosts:
            return
        self._stream_warned_hosts.add(host)
        logger.warning(
            "agentopt: %s call to %s succeeded but token usage could not be "
            "extracted — %s",
            model or "<unknown>",
            request_url,
            reason,
        )


# ---------------------------------------------------------------------------
# HTTP parsing / forwarding / serialization helpers
# ---------------------------------------------------------------------------


def _read_request(
    rfile: BinaryIO, first_line: Optional[bytes] = None,
) -> Optional[_Request]:
    """Parse one HTTP/1.1 request from a binary file-like.

    Returns ``None`` on EOF or a malformed request line.
    """
    if first_line is None:
        first_line = rfile.readline()
    if not first_line:
        return None

    line = first_line.decode("utf-8", errors="replace").strip()
    parts = line.split(" ", 2)
    if len(parts) < 2:
        return None
    method, path = parts[0], parts[1]

    headers: Dict[str, str] = {}
    while True:
        h = rfile.readline()
        if h in (b"\r\n", b"\n", b""):
            break
        decoded = h.decode("utf-8", errors="replace").strip()
        if ":" in decoded:
            k, v = decoded.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    content_length = _parse_content_length(headers, method)
    body = rfile.read(content_length) if content_length > 0 else b""

    json_body: Dict[str, Any] = {}
    if body:
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                json_body = parsed
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    return _Request(
        method=method, path=path, headers=headers, body=body, json_body=json_body,
    )


def _parse_content_length(headers: Dict[str, str], method: str) -> int:
    """Strict Content-Length parse.

    HTTP framing must not be guessed.  Silently defaulting to 0 on a missing
    or malformed header would either truncate the request body (sending a
    broken request upstream) or desynchronize the keep-alive parser, since
    the unread body bytes would be misread as the next request line.

    Raises ``ValueError`` if:
    * the header is absent on a body-bearing method (POST/PUT/PATCH)
    * the value isn't a non-negative integer
    """
    raw = headers.get("content-length")
    if raw is None:
        if method.upper() in ("POST", "PUT", "PATCH"):
            raise ValueError(
                f"{method} request is missing Content-Length header "
                f"(header keys present: {sorted(headers)})"
            )
        return 0

    try:
        n = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Content-Length is not an integer: {raw!r}") from exc
    if n < 0:
        raise ValueError(f"Content-Length is negative: {n}")
    return n


def _build_forward_headers(
    incoming: Dict[str, str], hostname: str, port: int,
) -> Dict[str, str]:
    """Strip hop-by-hop + accept-encoding, set Host."""
    fwd = {k: v for k, v in incoming.items() if k not in _HOP_BY_HOP}
    # accept-encoding stripped so we don't get gzipped responses we'd have
    # to decode just to extract usage.
    fwd.pop("accept-encoding", None)
    if port in (80, 443):
        fwd["host"] = hostname
    else:
        fwd["host"] = f"{hostname}:{port}"
    return fwd


def _forward(
    *,
    method: str,
    scheme: str,
    hostname: str,
    port: int,
    path: str,
    headers: Dict[str, str],
    body: bytes,
) -> Tuple[int, Dict[str, str], bytes]:
    """Forward one HTTP/HTTPS request synchronously and return the full response."""
    if scheme == "https":
        conn = http.client.HTTPSConnection(hostname, port, timeout=120)
    else:
        conn = http.client.HTTPConnection(hostname, port, timeout=120)
    try:
        conn.request(method, path, body=body or None, headers=headers)
        resp = conn.getresponse()
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, resp_headers, resp.read()
    finally:
        conn.close()


def _build_response(status: int, headers: Dict[str, str], body: bytes,) -> bytes:
    """Serialize an HTTP/1.1 response."""
    try:
        phrase = HTTPStatus(status).phrase
    except ValueError:
        phrase = "Unknown"

    out_headers = dict(headers)
    if not any(k.lower() == "content-length" for k in out_headers):
        out_headers["content-length"] = str(len(body))

    lines = [f"HTTP/1.1 {status} {phrase}"]
    for k, v in out_headers.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("")
    return "\r\n".join(lines).encode("utf-8") + body


def _write_response(
    wfile: BinaryIO, status: int, headers: Dict[str, str], body: bytes,
) -> None:
    wfile.write(_build_response(status, headers, body))
    wfile.flush()


def _safe_response_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Strip headers that don't survive proxy retransmission."""
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in ("transfer-encoding", "content-encoding")
    }


def _bidirectional_relay(a: socket.socket, b: socket.socket) -> None:
    """Pump bytes between two sockets until either side closes."""

    def relay(src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except Exception:
            pass
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except Exception:
                pass

    t1 = threading.Thread(target=relay, args=(a, b), daemon=True)
    t2 = threading.Thread(target=relay, args=(b, a), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def _close_quietly(sock: Any) -> None:
    try:
        sock.close()
    except Exception:
        pass
