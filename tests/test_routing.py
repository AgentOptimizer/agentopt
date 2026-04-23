"""Tests for per-call model routing via the proxy.

Covers:

* ``_apply_router`` unit behaviour — mutation, pass-through, exception
  containment, and cross-provider ``NotImplementedError``.
* End-to-end via ``LLMTracker`` — the routed model actually reaches the
  upstream, ``CallRecord.model`` / ``requested_model`` are set correctly.
* Cache keying — cache entries are keyed on the *routed* model, not the
  requested one.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Optional

import httpx
import pytest
from aiohttp import web

from agentopt import LLMTracker, RandomRouter, RouteContext, RouteDecision
from agentopt.proxy.server import ProxyServer
from agentopt.proxy.session import SessionInfo


# ---------------------------------------------------------------------------
# Upstream that echoes the received request body — lets tests assert on
# exactly what the proxy forwarded.
# ---------------------------------------------------------------------------


class EchoUpstream:
    """aiohttp stub that echoes the received JSON body into its response."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown: Optional[asyncio.Event] = None
        self._started = threading.Event()
        self._port: Optional[int] = None
        self.seen_bodies: list[dict] = []
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        assert self._port is not None
        return f"http://127.0.0.1:{self._port}"

    def start(self) -> None:
        self._started.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._started.wait(timeout=5)

    def stop(self) -> None:
        if self._loop and self._shutdown:
            self._loop.call_soon_threadsafe(self._shutdown.set)
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self._port = None

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        self._shutdown = asyncio.Event()
        app = web.Application()
        app.router.add_post("/{path:.*}", self._handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self._port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self._started.set()
        await self._shutdown.wait()
        await runner.cleanup()

    async def _handle(self, request: web.Request) -> web.Response:
        body = await request.json()
        with self._lock:
            self.seen_bodies.append(body)
        return web.json_response(
            {
                "id": "echo",
                "object": "chat.completion",
                "model": body.get("model", "unknown"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "_echoed_request": body,
            }
        )


@pytest.fixture()
def echo_upstream():
    s = EchoUpstream()
    s.start()
    yield s
    s.stop()


# ---------------------------------------------------------------------------
# Minimal session + request fixtures for unit tests.
# ---------------------------------------------------------------------------


def _fresh_session() -> SessionInfo:
    return SessionInfo(
        session_id="sess_test", data_id="dp_1", combo_id="combo_a", agent_id=None,
    )


def _fresh_request():
    body_dict = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
    }
    body_bytes = json.dumps(body_dict).encode("utf-8")
    return body_dict, body_bytes


# ---------------------------------------------------------------------------
# RandomRouter: determinism under a fixed seed.
# ---------------------------------------------------------------------------


def test_random_router_is_deterministic_under_seed():
    pool = ["model-a", "model-b", "model-c"]
    r1 = RandomRouter(pool, seed=0)
    r2 = RandomRouter(pool, seed=0)
    ctx = RouteContext(
        request_body={"model": "orig"},
        provider="openai",
        requested_model="orig",
        session_data_id=None,
        session_combo_id=None,
        session_agent_id=None,
        history=(),
    )
    seq1 = [r1.route(ctx).model for _ in range(20)]  # type: ignore[union-attr]
    seq2 = [r2.route(ctx).model for _ in range(20)]  # type: ignore[union-attr]
    assert seq1 == seq2
    assert set(seq1) <= set(pool)


# ---------------------------------------------------------------------------
# _apply_router unit tests — construct a ProxyServer without starting it.
# ---------------------------------------------------------------------------


def test_apply_router_mutates_body_and_reports_requested_model():
    class FixedRouter:
        def route(self, ctx):
            assert ctx.requested_model == "gpt-4o"
            assert ctx.provider == "openai"
            assert ctx.session_data_id == "dp_1"
            return RouteDecision(model="gpt-4o-mini")

    server = ProxyServer(router=FixedRouter())
    body_dict, body_bytes = _fresh_request()
    session = _fresh_session()

    rb, b, requested = server._apply_router(
        body_dict, body_bytes, "/v1/chat/completions", session,
    )
    assert requested == "gpt-4o"
    assert rb["model"] == "gpt-4o-mini"
    # Body bytes must be re-encoded to match the mutated dict.
    assert json.loads(b)["model"] == "gpt-4o-mini"


def test_apply_router_none_decision_leaves_body_unchanged():
    class NullRouter:
        def route(self, ctx):
            return None

    server = ProxyServer(router=NullRouter())
    body_dict, body_bytes = _fresh_request()
    session = _fresh_session()

    rb, b, requested = server._apply_router(
        body_dict, body_bytes, "/v1/chat/completions", session,
    )
    assert requested is None
    assert rb is body_dict
    assert b is body_bytes


def test_apply_router_same_model_decision_records_but_does_not_re_encode():
    class IdentityRouter:
        def route(self, ctx):
            return RouteDecision(model=ctx.requested_model)  # same as incoming

    server = ProxyServer(router=IdentityRouter())
    body_dict, body_bytes = _fresh_request()
    session = _fresh_session()

    rb, b, requested = server._apply_router(
        body_dict, body_bytes, "/v1/chat/completions", session,
    )
    # Router fired → requested_model is set, even though nothing swapped.
    assert requested == "gpt-4o"
    assert rb["model"] == "gpt-4o"
    assert b is body_bytes  # no re-encode


def test_apply_router_exception_is_swallowed():
    class BrokenRouter:
        def route(self, ctx):
            raise RuntimeError("policy error")

    server = ProxyServer(router=BrokenRouter())
    body_dict, body_bytes = _fresh_request()
    session = _fresh_session()

    rb, b, requested = server._apply_router(
        body_dict, body_bytes, "/v1/chat/completions", session,
    )
    # Broken router is treated as no-op — request proceeds unrouted.
    assert requested is None
    assert rb["model"] == "gpt-4o"
    assert b is body_bytes


def test_apply_router_cross_provider_raises_notimplementederror():
    class CrossProviderRouter:
        def route(self, ctx):
            return RouteDecision(model="claude-opus-4-7", provider="anthropic")

    server = ProxyServer(router=CrossProviderRouter())
    body_dict, body_bytes = _fresh_request()
    session = _fresh_session()

    with pytest.raises(NotImplementedError):
        server._apply_router(
            body_dict, body_bytes, "/v1/chat/completions", session,
        )


def test_apply_router_no_router_is_passthrough():
    server = ProxyServer(router=None)
    body_dict, body_bytes = _fresh_request()
    session = _fresh_session()

    rb, b, requested = server._apply_router(
        body_dict, body_bytes, "/v1/chat/completions", session,
    )
    assert requested is None
    assert rb is body_dict
    assert b is body_bytes


def test_apply_router_skips_bodies_without_model_field():
    class ShouldNotFireRouter:
        def route(self, ctx):  # pragma: no cover — should not be called
            raise AssertionError("router must not run for bodyless requests")

    server = ProxyServer(router=ShouldNotFireRouter())
    session = _fresh_session()

    # Empty body dict (e.g. body failed to parse) → hook no-ops.
    rb, b, requested = server._apply_router({}, b"", "/v1/chat/completions", session,)
    assert requested is None
    assert rb == {}
    assert b == b""


# ---------------------------------------------------------------------------
# End-to-end via LLMTracker — the routed model actually hits the upstream
# and lands in CallRecord.
# ---------------------------------------------------------------------------


class _StaticRouter:
    """Always swap to a fixed model."""

    def __init__(self, target: str) -> None:
        self.target = target

    def route(self, ctx):
        return RouteDecision(model=self.target)


def test_router_swap_reaches_upstream_and_is_recorded(echo_upstream):
    tracker = LLMTracker(
        cache=False, cache_dir=None, router=_StaticRouter("gpt-4o-mini"),
    )
    tracker.start()
    try:
        with tracker.track(data_id="dp_1", combo_id="combo_a"):
            client = httpx.Client(base_url=echo_upstream.base_url)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 200
            # Upstream saw the swapped model.
            assert resp.json()["_echoed_request"]["model"] == "gpt-4o-mini"

        # CallRecord reflects actually-called model + preserves original.
        records = tracker.get_records()
        assert len(records) == 1
        assert records[0].model == "gpt-4o-mini"
        assert records[0].requested_model == "gpt-4o"
    finally:
        tracker.stop()

    # What the upstream received matches what CallRecord logged.
    assert len(echo_upstream.seen_bodies) == 1
    assert echo_upstream.seen_bodies[0]["model"] == "gpt-4o-mini"


def test_token_usage_attributes_to_actually_called_model(echo_upstream):
    tracker = LLMTracker(
        cache=False, cache_dir=None, router=_StaticRouter("gpt-4o-mini"),
    )
    tracker.start()
    try:
        with tracker.track(data_id="dp_1", combo_id="combo_a"):
            httpx.Client(base_url=echo_upstream.base_url).post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        usage = tracker.get_usage(combo_id="combo_a")
        # Keyed on the routed model — what was actually paid for.
        assert "gpt-4o-mini" in usage
        assert "gpt-4o" not in usage
        assert usage["gpt-4o-mini"] == (1, 1)
    finally:
        tracker.stop()


def test_no_router_configured_leaves_request_untouched(echo_upstream):
    tracker = LLMTracker(cache=False, cache_dir=None)  # no router
    tracker.start()
    try:
        with tracker.track(data_id="dp_1", combo_id="combo_a"):
            httpx.Client(base_url=echo_upstream.base_url).post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        records = tracker.get_records()
        assert len(records) == 1
        assert records[0].model == "gpt-4o"
        assert records[0].requested_model is None
    finally:
        tracker.stop()

    assert echo_upstream.seen_bodies[0]["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Cache semantics — routed model is part of the cache key.
# ---------------------------------------------------------------------------


def test_cache_keyed_on_routed_model_not_requested(echo_upstream, tmp_path):
    # Two trackers sharing a cache dir: one routes to A, one to B.
    # Same prompt → two distinct cache entries because the routed model
    # is part of the key.
    cache_dir = tmp_path / "cache"

    t1 = LLMTracker(
        cache=True, cache_dir=str(cache_dir), router=_StaticRouter("model-a"),
    )
    t1.start()
    try:
        with t1.track(data_id="dp_1", combo_id="combo_a"):
            httpx.Client(base_url=echo_upstream.base_url).post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "cache me"}],
                },
            )
            # Second identical call — routed to same model-a → should be cached.
            httpx.Client(base_url=echo_upstream.base_url).post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "cache me"}],
                },
            )
        recs = t1.get_records()
        assert len(recs) == 2
        assert recs[0].cached is False
        assert recs[1].cached is True
        assert recs[1].model == "model-a"
    finally:
        t1.stop()

    upstream_calls_after_t1 = len(echo_upstream.seen_bodies)

    # Fresh tracker with router picking a different model — same prompt
    # must miss the cache because the routed model differs.
    t2 = LLMTracker(
        cache=True, cache_dir=str(cache_dir), router=_StaticRouter("model-b"),
    )
    t2.start()
    try:
        with t2.track(data_id="dp_1", combo_id="combo_b"):
            httpx.Client(base_url=echo_upstream.base_url).post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "cache me"}],
                },
            )
        recs = t2.get_records()
        assert len(recs) == 1
        assert recs[0].cached is False
        assert recs[0].model == "model-b"
    finally:
        t2.stop()

    # Upstream got exactly one more call for model-b (cache miss).
    assert len(echo_upstream.seen_bodies) == upstream_calls_after_t1 + 1
    assert echo_upstream.seen_bodies[-1]["model"] == "model-b"


# ---------------------------------------------------------------------------
# Public ``with router:`` API — no LLMTracker exposure.
# ---------------------------------------------------------------------------


class _StaticRouterCM(_StaticRouter):
    """Same behaviour as _StaticRouter but inherits Router so it's a ctx mgr."""

    def __init__(self, target: str) -> None:
        super().__init__(target)

    # _StaticRouter doesn't inherit Router, so wrap via a class that does.


from agentopt import Router  # noqa: E402


class _StaticRouterWithBase(Router):
    def __init__(self, target: str) -> None:
        self.target = target

    def route(self, ctx):
        return RouteDecision(model=self.target)


def test_with_router_activates_and_deactivates(echo_upstream):
    """`with router:` routes all LLM calls in the block; deactivates cleanly."""
    router = _StaticRouterWithBase("gpt-4o-mini")

    with router:
        resp = httpx.Client(base_url=echo_upstream.base_url).post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],},
        )
        assert resp.status_code == 200
        assert resp.json()["_echoed_request"]["model"] == "gpt-4o-mini"

    # Outside the block the redirect contextvar is cleared — new httpx calls
    # would not hit our singleton session.  We don't assert the upstream
    # sees new calls (nothing to send), just that the router's handle has
    # been released.
    assert getattr(router, "_routing_handle", None) is None


def test_with_router_random_policy_seeded_end_to_end(echo_upstream):
    """RandomRouter built via model_candidates= drives the upstream swap."""
    router = RandomRouter(model_candidates=["alpha", "beta"], seed=0)

    with router:
        for _ in range(4):
            httpx.Client(base_url=echo_upstream.base_url).post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

    seen_models = [b["model"] for b in echo_upstream.seen_bodies]
    assert len(seen_models) == 4
    # Every call got routed to one of the candidates.
    assert set(seen_models) <= {"alpha", "beta"}


def test_with_router_is_not_reentrant_on_same_instance():
    router = _StaticRouterWithBase("gpt-4o-mini")
    with router:
        with pytest.raises(RuntimeError, match="not re-entrant"):
            with router:
                pass
    # After the error the outer scope still exits cleanly.
    assert getattr(router, "_routing_handle", None) is None


def test_sequential_with_blocks_on_same_router_work(echo_upstream):
    """Exiting and re-entering the same router instance works fine."""
    router = _StaticRouterWithBase("gpt-4o-mini")

    for _ in range(3):
        with router:
            httpx.Client(base_url=echo_upstream.base_url).post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
    assert len(echo_upstream.seen_bodies) == 3
    for b in echo_upstream.seen_bodies:
        assert b["model"] == "gpt-4o-mini"
