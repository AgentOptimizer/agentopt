"""Routing tests — both interception paths, daemon-side refusal of custom routers."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, List, Optional

import httpx
import pytest

from agentopt import LLMTracker, RandomRouter, Router
from agentopt.routing import RouteContext  # power-user type import


# ---------------------------------------------------------------------------
# Test routers — recording wrappers so we can assert what was seen
# ---------------------------------------------------------------------------


class _RecordingRouter(Router):
    """Records every ``ctx`` and returns a configurable model name."""

    def __init__(self, target_model: str) -> None:
        self._target = target_model
        self.seen: List[Any] = []

    def route(self, ctx: RouteContext) -> Optional[str]:
        self.seen.append(ctx)
        return self._target


class _PassthroughRouter(Router):
    """Returns ``None`` — no swap.  Useful to assert the unrouted path."""

    def __init__(self) -> None:
        self.calls = 0

    def route(self, ctx: RouteContext) -> Optional[str]:
        self.calls += 1
        return None


# ---------------------------------------------------------------------------
# Single-session sugar: ``with LLMTracker(router=...)`` is the one pattern
# ---------------------------------------------------------------------------


def test_tracker_with_router_routes_in_process_call(mock_upstream):
    """``with LLMTracker(router=...)`` routes the in-process httpx call."""
    router = _RecordingRouter(target_model="routed-by-tracker")
    with LLMTracker(combo_id="c", router=router, cache=False, cache_dir=None):
        client = httpx.Client(base_url=mock_upstream.base_url)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200

    assert len(router.seen) == 1
    ctx = router.seen[0]
    assert ctx.requested_model == "gpt-4o"
    assert ctx.provider == "openai"


def test_router_returning_none_passes_through(mock_upstream):
    """A router that returns ``None`` does not mutate the request body."""
    router = _PassthroughRouter()
    with LLMTracker(combo_id="c", router=router, cache=False, cache_dir=None):
        client = httpx.Client(base_url=mock_upstream.base_url)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
    assert router.calls == 1


# ---------------------------------------------------------------------------
# Explicit ``tracker.track(router=...)`` path — multi-session host
# ---------------------------------------------------------------------------


def test_router_via_explicit_track_swaps_model_in_request_body(mock_upstream):
    """``tracker.track(router=...)`` rewrites ``body['model']`` before
    the request leaves the in-process httpx wrapper."""
    router = _RecordingRouter(target_model="gpt-4o-mini-routed")
    with LLMTracker(cache=False, cache_dir=None) as tracker:
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

        # Router saw the *original* requested_model.
        assert len(router.seen) == 1
        ctx = router.seen[0]
        assert ctx.requested_model == "gpt-4o"
        assert ctx.provider == "openai"
        assert ctx.session.session_id == session.session_id

        # The recorded model reflects the router's decision, not the client's.
        rec = tracker.get_records(combo_id="c")[0]
        assert rec.model == "gpt-4o-mini-routed"


def test_router_raising_does_not_break_agent(mock_upstream):
    """An exception inside ``route()`` is caught; the request proceeds."""

    class _Broken(Router):
        def route(self, ctx):
            raise RuntimeError("boom")

    with LLMTracker(cache=False, cache_dir=None) as tracker:
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


def test_route_returning_non_string_passes_through(mock_upstream):
    """A router that returns something weird is logged and ignored."""

    class _Weird(Router):
        def route(self, ctx):
            return {"model": "gpt-4o-mini"}  # not a string — should be Optional[str]

    with LLMTracker(cache=False, cache_dir=None) as tracker:
        with tracker.track(data_id="dp", combo_id="c", router=_Weird()):
            client = httpx.Client(base_url=mock_upstream.base_url)
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert tracker.get_records(combo_id="c")[0].model == "gpt-4o"


def test_random_router_picks_from_candidates(mock_upstream):
    """``RandomRouter(seed=...)`` is reproducible and chooses from the pool."""
    candidates = ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"]
    with LLMTracker(
        combo_id="c",
        router=RandomRouter(candidates=candidates, seed=42),
        cache=False,
        cache_dir=None,
    ) as tracker:
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
        assert recorded_models <= set(candidates)
        assert recorded_models != {"ignored-by-router"}


def test_router_runs_before_cache_so_keys_reflect_actual_model(mock_upstream):
    """Cache hit/miss is keyed by the routed model, not the requested one.

    With caching on, route the *same* request body to two different
    models and verify both reach the upstream (no cross-model cache
    pollution).
    """

    class _ToggleRouter(Router):
        def __init__(self) -> None:
            self.next = "model-a"

        def route(self, ctx):
            picked = self.next
            self.next = "model-b" if picked == "model-a" else "model-a"
            return picked

    with LLMTracker(
        combo_id="c", router=_ToggleRouter(), cache=True, cache_dir=None,
    ) as tracker:
        client = httpx.Client(base_url=mock_upstream.base_url)
        payload = {
            "model": "client-requested",
            "messages": [{"role": "user", "content": "same content"}],
        }
        for _ in range(4):
            resp = client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200

        # 2 distinct routed models → 2 upstream calls, then 2 cache hits.
        assert mock_upstream.request_count == 2
        records = tracker.get_records(combo_id="c")
        assert [r.model for r in records] == [
            "model-a",
            "model-b",
            "model-a",
            "model-b",
        ]
        assert [r.cached for r in records] == [False, False, True, True]


# ---------------------------------------------------------------------------
# Subprocess path (mitmproxy addon)
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
    """A subprocess hitting the mock through ``HTTPS_PROXY`` gets routed too."""
    router = _RecordingRouter(target_model="subprocess-routed")
    with LLMTracker(cache=False, cache_dir=None) as tracker:
        with tracker.track(data_id="dp", combo_id="c", router=router) as session:
            env = tracker.get_session_env(session)
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
        # The mock returns the model field from the request body, so its
        # stdout is the routed model name (or the recorder caught it).
        assert result.stdout.strip() == "subprocess-routed" or len(router.seen) == 1
        rec = tracker.get_records(combo_id="c")[0]
        assert rec.model == "subprocess-routed"


# ---------------------------------------------------------------------------
# Custom-router serialization (refused in daemon mode unless overridden)
# ---------------------------------------------------------------------------


def test_custom_router_without_config_kwargs_raises_in_daemon_mode():
    """A custom ``Router`` that doesn't override ``_config_kwargs`` fails
    at the wire boundary with a pointer to the fix.

    Built-ins (``RandomRouter``) implement ``_config_kwargs`` and travel
    fine — verified end-to-end in ``test_daemon_e2e.py``.
    """

    class _Custom(Router):
        def route(self, ctx):
            return "x"

    with pytest.raises(NotImplementedError, match="_config_kwargs"):
        _Custom().config()
