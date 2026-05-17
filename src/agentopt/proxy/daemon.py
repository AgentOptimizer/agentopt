"""HTTP control plane for the long-lived ``agentopt serve`` daemon.

The daemon owns a singleton :class:`LocalBackend` and exposes its surface
over HTTP so any number of client processes (each holding a
``RemoteBackend``) can share one set of sessions, records, and cache.

Endpoints (localhost-only, JSON in/out)::

    GET    /health                          → {"ok": true}
    POST   /sessions                        → {session_id, proxy_port, ca_pem_b64}
    DELETE /sessions/{session_id}           → 204
    GET    /records?data_id=&combo_id=…     → [CallRecord, …]
    GET    /usage?data_id=&…                → {model: [in, out]}
    GET    /cached_latency?…                → {seconds: float}
    POST   /cache/flush                     → 204
    POST   /cache/clear                     → 204
    POST   /providers                       → 204
    GET    /ca                              → {ca_pem_b64}

No auth, no TLS — bind 127.0.0.1 only.  ``--allow-remote`` is reserved
for a future revision that adds authentication; until then, attempting
to bind 0.0.0.0 fails fast.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import logging
from typing import Any, Dict, List, Optional

from aiohttp import web

from ._backend import _MITMPROXY_CA_CERT, LocalBackend
from .models import CallRecord

logger = logging.getLogger(__name__)


_BACKEND_KEY = web.AppKey("backend", LocalBackend)


# ---------------------------------------------------------------------------
# CallRecord serialisation
# ---------------------------------------------------------------------------


def record_to_dict(r: CallRecord) -> Dict[str, Any]:
    """Serialize a ``CallRecord`` to a JSON-compatible dict."""
    return dataclasses.asdict(r)


def record_from_dict(d: Dict[str, Any]) -> CallRecord:
    """Reconstruct a ``CallRecord`` from a server-side dict."""
    return CallRecord(**d)


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------


async def _health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def _create_session(request: web.Request) -> web.Response:
    body = await request.json() if request.body_exists else {}
    backend = request.app[_BACKEND_KEY]
    loop = asyncio.get_running_loop()
    session, port = await loop.run_in_executor(
        None,
        backend.open_session,
        body.get("data_id"),
        body.get("combo_id"),
        body.get("agent_id"),
    )
    ca_pem_b64 = _read_ca_pem_b64()
    return web.json_response(
        {
            "session_id": session.session_id,
            "proxy_port": port,
            "ca_pem_b64": ca_pem_b64,
        }
    )


async def _close_session(request: web.Request) -> web.Response:
    sid = request.match_info["session_id"]
    backend = request.app[_BACKEND_KEY]
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, backend.close_session, sid)
    return web.Response(status=204)


async def _get_records(request: web.Request) -> web.Response:
    backend = request.app[_BACKEND_KEY]
    records: List[CallRecord] = backend.get_records(
        data_id=request.query.get("data_id"),
        combo_id=request.query.get("combo_id"),
        agent_id=request.query.get("agent_id"),
    )
    return web.json_response([record_to_dict(r) for r in records])


async def _get_usage(request: web.Request) -> web.Response:
    backend = request.app[_BACKEND_KEY]
    usage = backend.get_usage(
        data_id=request.query.get("data_id"),
        combo_id=request.query.get("combo_id"),
        agent_id=request.query.get("agent_id"),
    )
    return web.json_response({k: list(v) for k, v in usage.items()})


async def _get_cached_latency(request: web.Request) -> web.Response:
    backend = request.app[_BACKEND_KEY]
    seconds = backend.get_cached_latency(
        data_id=request.query.get("data_id"),
        combo_id=request.query.get("combo_id"),
        agent_id=request.query.get("agent_id"),
    )
    return web.json_response({"seconds": seconds})


async def _flush_cache(request: web.Request) -> web.Response:
    request.app[_BACKEND_KEY].flush_cache()
    return web.Response(status=204)


async def _clear_cache(request: web.Request) -> web.Response:
    request.app[_BACKEND_KEY].clear_cache()
    return web.Response(status=204)


async def _register_provider(request: web.Request) -> web.Response:
    body = await request.json()
    request.app[_BACKEND_KEY].register_provider(
        name=body["name"],
        base_url=body["base_url"],
        path_patterns=tuple(body["path_patterns"]),
    )
    return web.Response(status=204)


async def _get_ca(request: web.Request) -> web.Response:
    return web.json_response({"ca_pem_b64": _read_ca_pem_b64()})


# ---------------------------------------------------------------------------
# App / runner
# ---------------------------------------------------------------------------


def make_app(backend: LocalBackend) -> web.Application:
    """Build the aiohttp application bound to *backend*."""
    app = web.Application()
    app[_BACKEND_KEY] = backend
    app.router.add_get("/health", _health)
    app.router.add_post("/sessions", _create_session)
    app.router.add_delete("/sessions/{session_id}", _close_session)
    app.router.add_get("/records", _get_records)
    app.router.add_get("/usage", _get_usage)
    app.router.add_get("/cached_latency", _get_cached_latency)
    app.router.add_post("/cache/flush", _flush_cache)
    app.router.add_post("/cache/clear", _clear_cache)
    app.router.add_post("/providers", _register_provider)
    app.router.add_get("/ca", _get_ca)
    return app


def _read_ca_pem_b64() -> str:
    """Read and base64-encode mitmproxy's CA cert.

    The CA file is generated by mitmproxy the first time a ``SessionMaster``
    starts.  Daemon startup eagerly warms a throwaway session so this is
    available before any client connects.
    """
    if not _MITMPROXY_CA_CERT.exists():
        raise RuntimeError(
            f"mitmproxy CA cert not found at {_MITMPROXY_CA_CERT}. "
            "Daemon CA warmup must have failed."
        )
    return base64.b64encode(_MITMPROXY_CA_CERT.read_bytes()).decode("ascii")


def _warmup_ca(backend: LocalBackend) -> None:
    """Ensure mitmproxy's CA is generated by spinning a throwaway session.

    Subsequent ``GET /ca`` and ``POST /sessions`` (which embed the CA)
    can then return the cert immediately rather than 503ing the first
    caller.
    """
    if _MITMPROXY_CA_CERT.exists():
        return
    logger.info("Warming up mitmproxy CA (one-time per machine)...")
    session, _port = backend.open_session(
        data_id="__warmup__", combo_id="__warmup__", agent_id=None,
    )
    backend.close_session(session.session_id)


def run(
    host: str = "127.0.0.1",
    port: int = 9000,
    cache_dir: Optional[str] = ".agentopt_cache",
    allow_remote: bool = False,
) -> None:
    """Start the daemon and block until interrupted.

    *host* defaults to ``127.0.0.1``; binding any other address requires
    ``allow_remote=True``, which is reserved for a future revision that
    ships authentication.  Until then, refusing fast is safer than
    accidentally exposing an unauthenticated proxy on the network.
    """
    if host not in ("127.0.0.1", "localhost", "::1") and not allow_remote:
        raise SystemExit(
            f"refusing to bind {host!r}: agentopt serve is localhost-only "
            "in this release (no auth).  Use --host 127.0.0.1, or pass "
            "--allow-remote once authentication is wired up."
        )

    backend = LocalBackend(cache=True, cache_dir=cache_dir)
    backend.start()
    try:
        _warmup_ca(backend)
        app = make_app(backend)
        logger.info("agentopt serve listening on http://%s:%d", host, port)
        web.run_app(app, host=host, port=port, print=None, handle_signals=True)
    finally:
        backend.stop()


def cli(argv: Optional[List[str]] = None) -> None:
    """argparse entrypoint for the ``agentopt serve`` subcommand."""
    parser = argparse.ArgumentParser(prog="agentopt serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--cache-dir", default=".agentopt_cache")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="(reserved) bind a non-localhost host. Requires auth which is not yet shipped.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    run(
        host=args.host,
        port=args.port,
        cache_dir=args.cache_dir,
        allow_remote=args.allow_remote,
    )
