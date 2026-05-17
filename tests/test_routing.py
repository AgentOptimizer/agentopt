"""Routing tests — both interception paths + the ``with router:`` sugar."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional

import httpx
import pytest

from agentopt import (
    LLMTracker,
    RandomRouter,
    RouteContext,
    RouteDecision,
    Router,
)


# ---------------------------------------------------------------------------
# Test routers — recording wrappers so we can assert what was seen
# ---------------------------------------------------------------------------


class _RecordingRouter(Router):
    """Records every ``RouteContext`` and returns a configurable model."""

    def __init__(self, target_model: str) -> None:
        self._target = target_model
        self.seen: List[RouteContext] = []

    def route(self, ctx: RouteContext) -> Optional[RouteDecision]:
        self.seen.append(ctx)
        return RouteDecision(model=self._target)


class _PassthroughRouter(Router):
    """Returns ``None`` — no swap.  Useful to assert the unrouted path."""

    def __init__(self) -> None:
        self.calls = 0

    def route(self, ctx: RouteContext) -> Optional[RouteDecision]:
        self.calls += 1
        return None


# ---------------------------------------------------------------------------
# In-process httpx path
# ---------------------------------------------------------------------------


def test_router_explicit_swaps_model_in_request_body(mock_upstream):
    """``tracker.track(router=...)`` rewrites ``body['model']`` before
    the request leaves the in-process httpx wrapper."""
    router = _RecordingRouter(target_model="gpt-4o-mini-routed")
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with tracker.track(data_id="dp", combo_id="c", router=router) as session:
            client = httpx.Client(base_url=mock_upstream.base_url)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 200

        # Router saw the *original* request body (with the client's model).
        assert len(router.seen) == 1
        ctx = router.seen[0]
        assert ctx.requested_model == "gpt-4o"
        assert ctx.provider == "openai"
        assert ctx.session.session_id == session.session_id

        # The recorded model reflects the router's decision, not the client's.
        rec = tracker.get_records(combo_id="c")[0]
        assert rec.model == "gpt-4o-mini-routed"
    finally:
        tracker.stop()


def test_router_none_decision_leaves_body_untouched(mock_upstream):
    """A router that returns ``None`` does not mutate the request body."""
    router = _PassthroughRouter()
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with tracker.track(data_id="dp", combo_id="c", router=router):
            client = httpx.Client(base_url=mock_upstream.base_url)
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert router.calls == 1
        rec = tracker.get_records(combo_id="c")[0]
        assert rec.model == "gpt-4o"  # unchanged
    finally:
        tracker.stop()


def test_with_router_context_manager_picks_up_in_track(mock_upstream):
    """``with router:`` activates a ContextVar that ``track()`` reads."""
    router = _RecordingRouter(target_model="claude-routed")
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with router:
            with tracker.track(data_id="dp", combo_id="c"):
                client = httpx.Client(base_url=mock_upstream.base_url)
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        assert len(router.seen) == 1
        assert tracker.get_records(combo_id="c")[0].model == "claude-routed"
    finally:
        tracker.stop()


def test_explicit_router_wins_over_with_router(mock_upstream):
    """An explicit ``router=`` kwarg overrides any active ``with router:``."""
    ambient = _RecordingRouter(target_model="ambient")
    explicit = _RecordingRouter(target_model="explicit")
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with ambient:
            with tracker.track(data_id="dp", combo_id="c", router=explicit):
                client = httpx.Client(base_url=mock_upstream.base_url)
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        # Only the explicit router was consulted.
        assert ambient.seen == []
        assert len(explicit.seen) == 1
        assert tracker.get_records(combo_id="c")[0].model == "explicit"
    finally:
        tracker.stop()


def test_router_raising_does_not_break_agent(mock_upstream):
    """An exception inside ``route()`` is caught; the request proceeds."""

    class _Broken(Router):
        def route(self, ctx):
            raise RuntimeError("boom")

    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with tracker.track(data_id="dp", combo_id="c", router=_Broken()):
            client = httpx.Client(base_url=mock_upstream.base_url)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 200
        # Original model survives.
        assert tracker.get_records(combo_id="c")[0].model == "gpt-4o"
    finally:
        tracker.stop()


def test_route_decision_with_provider_raises():
    """Cross-provider routing is reserved; setting ``provider`` raises."""

    class _CrossProvider(Router):
        def route(self, ctx):
            return RouteDecision(model="claude-haiku-4-5", provider="anthropic")

    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with tracker.track(data_id="dp", combo_id="c", router=_CrossProvider()):
            client = httpx.Client(base_url="http://127.0.0.1:1")
            with pytest.raises(NotImplementedError, match="cross-provider"):
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
    finally:
        tracker.stop()


def test_random_router_picks_from_candidates(mock_upstream):
    """``RandomRouter(seed=...)`` is reproducible and chooses from the pool."""
    candidates = ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"]
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with tracker.track(
            data_id="dp",
            combo_id="c",
            router=RandomRouter(candidates=candidates, seed=42),
        ):
            client = httpx.Client(base_url=mock_upstream.base_url)
            for _ in range(5):
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "ignored-by-router",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )

        recorded_models = {r.model for r in tracker.get_records(combo_id="c")}
        # Every recorded model must come from the candidate pool.
        assert recorded_models <= set(candidates)
        # And we exercised at least one swap (seed=42 isn't a fixed point).
        assert recorded_models != {"ignored-by-router"}
    finally:
        tracker.stop()


def test_router_runs_before_cache_so_keys_reflect_actual_model(mock_upstream):
    """Cache hit/miss is keyed by the routed model, not the requested one.

    Strategy: with caching on, route the *same* request body to two
    different models and verify both reach the upstream (no cross-model
    cache pollution).
    """

    class _ToggleRouter(Router):
        def __init__(self) -> None:
            self.next = "model-a"

        def route(self, ctx):
            picked = self.next
            self.next = "model-b" if picked == "model-a" else "model-a"
            return RouteDecision(model=picked)

    tracker = LLMTracker(cache=True, cache_dir=None)
    tracker.start()
    try:
        with tracker.track(data_id="dp", combo_id="c", router=_ToggleRouter()):
            client = httpx.Client(base_url=mock_upstream.base_url)
            payload = {
                "model": "client-requested",  # irrelevant — router overrides
                "messages": [{"role": "user", "content": "same content"}],
            }
            # 4 calls, all with the *same* original payload.  Router
            # alternates model-a / model-b before the cache lookup, so
            # the cache sees two distinct (model, body) pairs.
            for _ in range(4):
                resp = client.post("/v1/chat/completions", json=payload)
                assert resp.status_code == 200

        # 2 distinct routed models → 2 upstream calls, then 2 cache hits.
        # If routing ran *after* cache, all four would collapse to one
        # upstream call (single key on the original body).  This assertion
        # is the invariant.
        assert mock_upstream.request_count == 2
        records = tracker.get_records(combo_id="c")
        assert [r.model for r in records] == [
            "model-a",
            "model-b",
            "model-a",
            "model-b",
        ]
        assert [r.cached for r in records] == [False, False, True, True]
    finally:
        tracker.stop()


def test_router_reentry_on_same_instance_raises():
    """Entering the same ``with router:`` twice is a clear error."""
    router = _RecordingRouter(target_model="x")
    with router:
        with pytest.raises(RuntimeError, match="not re-entrant"):
            router.__enter__()


# ---------------------------------------------------------------------------
# Subprocess path (mitmproxy addon)
#
# Subprocess routing requires a TLS-terminated proxy session and a real
# httpx process talking through it.  Covered minimally here: a Python
# subprocess spawned with the session's env vars hits the mock upstream
# via the proxy, the addon's router rewrites body['model'], the upstream
# records the rewritten model.
# ---------------------------------------------------------------------------


_SUBPROCESS_AGENT = """
import json
import os
import sys
import httpx

base = sys.argv[1]
with httpx.Client(base_url=base, timeout=10.0) as client:
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    print(r.json()["model"])
"""


def test_subprocess_routing_via_https_proxy(mock_upstream, tmp_path):
    """A subprocess hitting the mock through ``HTTPS_PROXY`` gets routed too.

    The mock upstream is plain HTTP so we don't need the CA bundle path
    to verify any TLS handshake — only ``HTTPS_PROXY`` matters here.
    """
    router = _RecordingRouter(target_model="subprocess-routed")
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with tracker.track(data_id="dp", combo_id="c", router=router) as session:
            env = tracker.get_session_env(session)
            # Override HTTPS_PROXY to be HTTP_PROXY too — our mock is plain HTTP.
            env_full = {
                **os.environ,
                "HTTP_PROXY": env["HTTPS_PROXY"],
                "HTTPS_PROXY": env["HTTPS_PROXY"],
            }
            agent = tmp_path / "agent.py"
            agent.write_text(_SUBPROCESS_AGENT)
            result = subprocess.run(
                [sys.executable, str(agent), mock_upstream.base_url],
                env=env_full,
                capture_output=True,
                text=True,
                timeout=30,
            )
        assert result.returncode == 0, result.stderr
        # The mock returns the model field from the request body — so its
        # output is the routed model name.
        assert result.stdout.strip() == "subprocess-routed" or len(router.seen) == 1
        # And the record carries the routed model.
        rec = tracker.get_records(combo_id="c")[0]
        assert rec.model == "subprocess-routed"
    finally:
        tracker.stop()


# ---------------------------------------------------------------------------
# Remote-mode refusal
# ---------------------------------------------------------------------------


def test_remote_mode_with_router_raises_not_implemented(monkeypatch):
    """In daemon mode, passing a router is a clear error (library-only v1).

    Spawning the daemon is the e2e fixture; here we only need to verify
    the ``track(router=...)`` call raises *before* any HTTP work — so
    we point the env var at a host nothing's listening on and assert
    we never reach the ``start()`` health probe.
    """
    monkeypatch.setenv("AGENTOPT_GATEWAY_URL", "http://127.0.0.1:1")
    # __init__ reads the env var and picks RemoteBackend.  No network call
    # happens until start(), so we can poke at track() through the backend
    # without a live daemon.
    tracker = LLMTracker()
    backend = tracker._backend
    # Mark it active so track() doesn't trip the "call start() first" assert.
    backend._active = True  # type: ignore[attr-defined]
    try:
        with pytest.raises(NotImplementedError, match="library-only"):
            with backend.track(
                data_id="dp", combo_id="c", router=RandomRouter(candidates=["x"]),
            ):
                pass
    finally:
        backend._active = False  # type: ignore[attr-defined]
