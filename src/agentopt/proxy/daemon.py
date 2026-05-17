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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aiohttp import web

from ._backend import _MITMPROXY_CA_CERT
from ._local_backend import LocalBackend
from .models import CallRecord

if False:  # type-only — avoid runtime import on the proxy side
    from agentopt.routing.base import Router

logger = logging.getLogger(__name__)


_BACKEND_KEY = web.AppKey("backend", LocalBackend)
# Optional default router applied to every session that doesn't carry
# its own.  ``None`` means "no default — sessions without an explicit
# router run unrouted."
_DEFAULT_ROUTER_KEY: "web.AppKey[Optional[Any]]" = web.AppKey(
    "default_router", object,
)


# ---------------------------------------------------------------------------
# CallRecord serialisation
# ---------------------------------------------------------------------------


def record_to_dict(r: CallRecord) -> Dict[str, Any]:
    """Serialize a ``CallRecord`` to a JSON-compatible dict."""
    return dataclasses.asdict(r)


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------


async def _health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def _create_session(request: web.Request) -> web.Response:
    body = await request.json() if request.body_exists else {}
    backend = request.app[_BACKEND_KEY]

    # Router resolution.  Body field "router" is the per-session override:
    #   - {"policy": "...", "kwargs": {...}}  → resolve and use
    #   - explicit None                       → no routing for this session
    #   - field absent                        → fall back to the daemon default
    router: Optional["Router"]
    if "router" in body:
        router_cfg = body["router"]
        if router_cfg is None:
            router = None
        else:
            try:
                from agentopt.routing.config import resolve_policy

                router = resolve_policy(
                    router_cfg["policy"], router_cfg.get("kwargs", {}),
                )
            except (KeyError, ValueError) as exc:
                return web.json_response({"error": str(exc)}, status=400,)
    else:
        router = request.app[_DEFAULT_ROUTER_KEY]

    loop = asyncio.get_running_loop()
    session, port = await loop.run_in_executor(
        None,
        backend.open_session,
        body.get("data_id"),
        body.get("combo_id"),
        body.get("agent_id"),
        router,
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


def make_app(
    backend: LocalBackend, default_router: Optional[Any] = None,
) -> web.Application:
    """Build the aiohttp application bound to *backend*.

    *default_router* (if provided) is applied to every session that
    arrives via ``POST /sessions`` without its own ``router`` field.
    """
    app = web.Application()
    app[_BACKEND_KEY] = backend
    app[_DEFAULT_ROUTER_KEY] = default_router
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
    cache_dir: Union[str, Path] = ".agentopt_cache",
    allow_remote: bool = False,
    routing_policy: Optional[str] = None,
    candidate_models: Optional[List[str]] = None,
    seed: Optional[int] = None,
    policy_modules: Optional[List[str]] = None,
) -> None:
    """Start the daemon and block until interrupted.

    *host* defaults to ``127.0.0.1``; binding any other address requires
    ``allow_remote=True``, which is reserved for a future revision that
    ships authentication.  Until then, refusing fast is safer than
    accidentally exposing an unauthenticated proxy on the network.

    *cache_dir* is resolved to an absolute path before opening the cache
    so the daemon's on-disk state lands somewhere predictable when run
    under a supervisor (systemd, supervisord, docker) where the CWD
    isn't where the operator expects.  The resolved path is logged at
    startup.
    """
    if host not in ("127.0.0.1", "localhost", "::1") and not allow_remote:
        raise SystemExit(
            f"refusing to bind {host!r}: agentopt serve is localhost-only "
            "in this release (no auth).  Use --host 127.0.0.1, or pass "
            "--allow-remote once authentication is wired up."
        )

    if not str(cache_dir).strip():
        raise SystemExit("agentopt serve: --cache-dir cannot be empty.  Pass a path.")
    resolved_cache_dir = Path(cache_dir).expanduser().resolve()

    # Pre-import user policy modules so custom ``Router`` subclasses
    # resolve when clients POST {"policy": "my_mod:MyRouter", ...}.
    if policy_modules:
        from agentopt.routing.config import load_policy_module

        for mod_path in policy_modules:
            try:
                load_policy_module(mod_path)
            except ValueError as exc:
                raise SystemExit(f"agentopt serve: --policy-module {exc}")

    default_router = _build_default_router(routing_policy, candidate_models, seed,)

    backend = LocalBackend(cache=True, cache_dir=resolved_cache_dir)
    backend.start()
    try:
        _warmup_ca(backend)
        app = make_app(backend, default_router=default_router)
        router_summary = (
            f" default router: {routing_policy} candidates={candidate_models}"
            if default_router is not None
            else " no default router"
        )
        logger.info(
            "agentopt serve listening on http://%s:%d (cache dir: %s);%s",
            host,
            port,
            resolved_cache_dir,
            router_summary,
        )
        web.run_app(app, host=host, port=port, print=None, handle_signals=True)
    finally:
        backend.stop()


def _build_default_router(
    policy: Optional[str], candidate_models: Optional[List[str]], seed: Optional[int],
) -> Optional[Any]:
    """Construct a daemon default router from CLI flags, or return ``None``.

    Only the built-in ``random`` policy is wired up to CLI flags today;
    custom policies arrive per-session via ``POST /sessions``.
    """
    if policy is None:
        if candidate_models or seed is not None:
            raise SystemExit(
                "agentopt serve: --candidate-models / --seed require "
                "--routing-policy.  Either set --routing-policy random, or "
                "drop the other routing flags."
            )
        return None

    from agentopt.routing.config import resolve_policy

    if policy == "random":
        if not candidate_models:
            raise SystemExit(
                "agentopt serve: --routing-policy random requires "
                "--candidate-models gpt-4o,gpt-4o-mini,..."
            )
        kwargs: Dict[str, Any] = {"candidates": list(candidate_models)}
        if seed is not None:
            kwargs["seed"] = seed
        return resolve_policy("random", kwargs)

    raise SystemExit(
        f"agentopt serve: --routing-policy {policy!r} is not a built-in.  "
        f"v1 CLI supports only 'random'; custom policies arrive per-session "
        f"via the wire protocol."
    )


def register_serve_subparser(
    subparsers: "argparse._SubParsersAction",
) -> argparse.ArgumentParser:
    """Register ``agentopt serve`` on the top-level CLI parser.

    Returns the subparser (for callers that want to introspect it).
    Sets ``args.func = _serve_main`` so :func:`agentopt.cli.main` can
    dispatch generically via ``args.func(args)``.
    """
    p = subparsers.add_parser(
        "serve",
        help="Run the long-lived gateway daemon.",
        description="Run the agentopt gateway daemon (localhost-only).",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument(
        "--cache-dir",
        default=".agentopt_cache",
        help="Directory for the response cache (cache.db). Resolved to an "
        "absolute path relative to the daemon's CWD at startup; the "
        "resolved path is logged. Default: ./.agentopt_cache",
    )
    p.add_argument(
        "--allow-remote",
        action="store_true",
        help="(reserved) bind a non-localhost host. Requires auth which is not yet shipped.",
    )
    # Routing — only the 'random' built-in is wired to CLI flags in v1.
    # Custom policies arrive per-session via POST /sessions; clients use
    # their Python `with router:` API as usual.
    p.add_argument(
        "--routing-policy",
        default=None,
        help="Default router for every session (currently only 'random').  "
        "Sessions can still override via POST /sessions.  Omit for no "
        "default routing.",
    )
    p.add_argument(
        "--candidate-models",
        default=None,
        help="Comma-separated model names for --routing-policy random.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for --routing-policy random.",
    )
    p.add_argument(
        "--policy-module",
        action="append",
        default=None,
        metavar="PATH",
        help="Path to a Python file defining custom Router subclasses.  "
        "Pre-imports the file so clients can POST "
        '{"policy": "filename:ClassName", "kwargs": {...}}.  '
        "Repeatable.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=_serve_main)
    return p


def _serve_main(args: argparse.Namespace) -> None:
    """Dispatched by :func:`agentopt.cli.main` after argument parsing."""
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    candidates = (
        [m.strip() for m in args.candidate_models.split(",") if m.strip()]
        if args.candidate_models
        else None
    )
    run(
        host=args.host,
        port=args.port,
        cache_dir=args.cache_dir,
        allow_remote=args.allow_remote,
        routing_policy=args.routing_policy,
        candidate_models=candidates,
        seed=args.seed,
        policy_modules=args.policy_module,
    )
