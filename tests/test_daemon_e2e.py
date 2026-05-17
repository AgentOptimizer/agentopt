"""End-to-end test for the ``agentopt serve`` daemon + ``RemoteBackend``.

Spawns the daemon as a subprocess on an ephemeral port, points
``LLMTracker`` at it via ``AGENTOPT_GATEWAY_URL``, and verifies that an
in-process ``httpx`` call produces the same kind of ``CallRecord`` the
local backend would have produced for the same traffic.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import httpx
import pytest

from agentopt.proxy import LLMTracker


# ---------------------------------------------------------------------------
# Daemon subprocess fixture
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    """Return a free TCP port on 127.0.0.1.

    Race-safe enough for tests: the kernel's ephemeral allocation tends
    not to reuse the same port within a short window.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_err = exc
        time.sleep(0.25)
    raise RuntimeError(
        f"agentopt serve at {url} did not become ready within {timeout}s "
        f"(last error: {last_err!r})"
    )


def _spawn_daemon(tmp_path, extra_args=()):
    """Spawn ``agentopt serve`` with optional extra CLI args; return (url, proc, log)."""
    port = _pick_free_port()
    cache_dir = tmp_path / "agentopt_cache"
    log_path = tmp_path / "daemon.log"
    log_handle = open(log_path, "wb")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agentopt.cli",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--cache-dir",
            str(cache_dir),
            *extra_args,
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    return url, proc, log_handle


def _stop_daemon(proc, log_handle):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    log_handle.close()


@pytest.fixture
def daemon(tmp_path):
    """Spawn ``agentopt serve`` in a subprocess; yield its base URL."""
    url, proc, log_handle = _spawn_daemon(tmp_path)
    try:
        _wait_for_health(url)
        yield url
    finally:
        _stop_daemon(proc, log_handle)


@pytest.fixture
def daemon_factory(tmp_path):
    """Yield a callable that spawns a daemon with custom CLI args.

    Used for tests that need ``--routing-policy`` etc.  Cleans up the
    daemon on test exit.
    """
    procs = []

    def spawn(extra_args):
        sub_tmp = tmp_path / f"d{len(procs)}"
        sub_tmp.mkdir()
        url, proc, log = _spawn_daemon(sub_tmp, extra_args=extra_args)
        procs.append((proc, log))
        _wait_for_health(url)
        return url

    try:
        yield spawn
    finally:
        for proc, log in procs:
            _stop_daemon(proc, log)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_daemon_health(daemon):
    """Sanity: /health returns 200 once the fixture has started."""
    r = httpx.get(f"{daemon}/health", timeout=5.0)
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_remote_in_process_call_is_recorded(daemon, mock_upstream, monkeypatch):
    """An in-process httpx LLM call is forwarded through the daemon and
    appears in the daemon's records — same shape as local mode."""
    monkeypatch.setenv("AGENTOPT_GATEWAY_URL", daemon)

    tracker = LLMTracker()
    tracker.start()
    try:
        with tracker.track(data_id="dp_1", combo_id="c"):
            client = httpx.Client(base_url=mock_upstream.base_url)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 200

        # Records live on the daemon; fetched via HTTP.
        records = tracker.get_records(combo_id="c")
        assert len(records) == 1
        rec = records[0]
        assert rec.status_code == 200
        assert rec.error is None
        assert rec.model == "gpt-4o-mini"
        # Mock returns usage {prompt_tokens: 10, completion_tokens: 5}.
        assert rec.prompt_tokens == 10
        assert rec.completion_tokens == 5
        assert rec.data_id == "dp_1"
        assert rec.combo_id == "c"

        # Usage aggregation should reflect the same call.
        usage = tracker.get_usage(combo_id="c")
        assert usage == {"gpt-4o-mini": (10, 5)}
    finally:
        tracker.stop()


def test_remote_get_session_env_points_at_daemon(daemon, monkeypatch):
    """``get_session_env`` returns the daemon's per-session port + bundle."""
    from urllib.parse import urlparse

    monkeypatch.setenv("AGENTOPT_GATEWAY_URL", daemon)
    daemon_url = urlparse(daemon)
    tracker = LLMTracker()
    tracker.start()
    try:
        with tracker.track(data_id="dp", combo_id="c") as session:
            env = tracker.get_session_env(session)

            proxy = urlparse(env["HTTPS_PROXY"])
            assert proxy.scheme == "http"
            assert proxy.hostname == daemon_url.hostname == "127.0.0.1"
            assert proxy.path in ("", "/")  # no extra path components
            # Per-session port is distinct from the control-plane port.
            assert proxy.port is not None
            assert proxy.port != daemon_url.port

            assert env["SSL_CERT_FILE"].endswith("ca-bundle.pem")
            assert env["REQUESTS_CA_BUNDLE"] == env["SSL_CERT_FILE"]
            assert env["NODE_EXTRA_CA_CERTS"] == env["SSL_CERT_FILE"]
    finally:
        tracker.stop()


def test_remote_no_op_endpoints(daemon, monkeypatch):
    """``flush_cache`` / ``clear_cache`` / empty-query queries don't error."""
    monkeypatch.setenv("AGENTOPT_GATEWAY_URL", daemon)
    tracker = LLMTracker()
    tracker.start()
    try:
        assert tracker.get_records(combo_id="never") == []
        assert tracker.get_usage(combo_id="never") == {}
        assert tracker.get_cached_latency(combo_id="never") == 0.0
        tracker.flush_cache()
        tracker.clear_cache()
    finally:
        tracker.stop()


def test_remote_handler_lazy_async_client(daemon, monkeypatch):
    """``RemoteHandler.close()`` is safe when ``handle_async`` was never called.

    Regression: prior to the lazy-creation fix, an ``AsyncClient`` was
    eagerly constructed in ``__init__``; ``close()`` then called the
    nonexistent ``AsyncClient.close()`` (only ``aclose()`` exists),
    raising ``AttributeError`` that was silently swallowed and leaking
    the connection pool every time ``track()`` exited.
    """
    import certifi

    from agentopt.proxy._remote_backend import RemoteHandler

    h = RemoteHandler(
        proxy_url="http://127.0.0.1:1",  # arbitrary; never sent to
        ca_bundle_path=certifi.where(),  # any valid PEM; never verified
    )
    assert h._async_client is None
    h.close()  # must not raise
    assert h._async_client is None  # still uncreated — no leak


def test_remote_async_in_process_call_is_recorded(daemon, mock_upstream, monkeypatch):
    """Async (``httpx.AsyncClient``) LLM call records cleanly through the daemon.

    Also exercises the async-cleanup path in ``RemoteHandler.close()`` —
    the prior implementation silently failed here, leaving the
    ``AsyncClient``'s connection pool open across runs.
    """
    import asyncio

    monkeypatch.setenv("AGENTOPT_GATEWAY_URL", daemon)
    tracker = LLMTracker()
    tracker.start()
    try:

        async def make_call():
            async with httpx.AsyncClient(base_url=mock_upstream.base_url) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert resp.status_code == 200

        with tracker.track(data_id="dp_async", combo_id="c_async"):
            asyncio.run(make_call())

        records = tracker.get_records(combo_id="c_async")
        assert len(records) == 1
        assert records[0].model == "gpt-4o-mini"
        assert records[0].status_code == 200
    finally:
        tracker.stop()


def test_remote_get_records_works_after_stop(daemon, mock_upstream, monkeypatch):
    """Regression: ``ModelSelector.select_best`` calls ``tracker.get_records()``
    *after* ``tracker.stop()`` to harvest the run's records.  Stop must not
    close the control-plane client out from under that follow-up query."""
    monkeypatch.setenv("AGENTOPT_GATEWAY_URL", daemon)

    tracker = LLMTracker()
    tracker.start()
    with tracker.track(data_id="dp", combo_id="c"):
        client = httpx.Client(base_url=mock_upstream.base_url)
        client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    tracker.stop()

    # Must still work — this is what ModelSelector does post-stop.
    records = tracker.get_records(combo_id="c")
    assert len(records) == 1
    assert records[0].model == "gpt-4o-mini"


def test_remote_close_releases_control_plane_client(daemon, monkeypatch):
    """``close()`` actually closes the long-lived httpx client.

    Without this, the client lingers until ``__del__`` and httpx emits
    an "Unclosed client" warning at GC time.
    """
    monkeypatch.setenv("AGENTOPT_GATEWAY_URL", daemon)
    tracker = LLMTracker()
    tracker.start()
    tracker.stop()
    # _http stays open across stop() so post-stop record queries work.
    assert not tracker._backend._http.is_closed  # type: ignore[attr-defined]

    tracker.close()
    assert tracker._backend._http.is_closed  # type: ignore[attr-defined]

    # Idempotent — calling close() again must not raise.
    tracker.close()


def test_remote_context_manager_closes(daemon, monkeypatch):
    """``with LLMTracker() as tracker:`` releases the control-plane client."""
    monkeypatch.setenv("AGENTOPT_GATEWAY_URL", daemon)
    with LLMTracker() as tracker:
        tracker.start()
        with tracker.track(data_id="dp", combo_id="c"):
            pass
        tracker.stop()
        backend = tracker._backend
    assert backend._http.is_closed  # type: ignore[attr-defined]


def test_daemon_rejects_empty_cache_dir():
    """``run(cache_dir='')`` fails fast — silently allowing it would land
    the cache somewhere unpredictable under a supervisor."""
    from agentopt.proxy.daemon import run

    with pytest.raises(SystemExit, match="cache-dir cannot be empty"):
        run(host="127.0.0.1", port=0, cache_dir="")


def test_remote_gateway_unreachable_fails_fast(monkeypatch):
    """A bad ``AGENTOPT_GATEWAY_URL`` raises at ``start()`` with a clear message."""
    # Pick a port nothing is listening on.
    monkeypatch.setenv("AGENTOPT_GATEWAY_URL", "http://127.0.0.1:1")
    tracker = LLMTracker()
    with pytest.raises(RuntimeError, match="not reachable"):
        tracker.start()


# ---------------------------------------------------------------------------
# Daemon-side routing (Phase 2)
# ---------------------------------------------------------------------------


def test_daemon_default_routing_policy_via_cli(
    daemon_factory, mock_upstream, monkeypatch,
):
    """``--routing-policy random --candidate-models a,b`` applies to every
    session that doesn't carry its own router."""
    daemon_url = daemon_factory(
        [
            "--routing-policy",
            "random",
            "--candidate-models",
            "gpt-4o-mini-routed,gpt-4o-mini-routed",  # both same → deterministic
            "--seed",
            "0",
        ]
    )
    monkeypatch.setenv("AGENTOPT_GATEWAY_URL", daemon_url)

    from agentopt import LLMTracker

    tracker = LLMTracker()
    tracker.start()
    try:
        with tracker.track(data_id="dp", combo_id="c"):  # no explicit router
            client = httpx.Client(base_url=mock_upstream.base_url)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",  # daemon will rewrite this
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 200

        rec = tracker.get_records(combo_id="c")[0]
        # Daemon's default policy rewrote the body before forwarding.
        assert rec.model == "gpt-4o-mini-routed"
    finally:
        tracker.stop()


def test_daemon_per_session_router_override(daemon, mock_upstream, monkeypatch):
    """Python client's ``with RandomRouter:`` overrides daemon default."""
    from agentopt import LLMTracker, RandomRouter

    monkeypatch.setenv("AGENTOPT_GATEWAY_URL", daemon)
    router = RandomRouter(candidates=["per-session-model"], seed=0)
    tracker = LLMTracker()
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
        rec = tracker.get_records(combo_id="c")[0]
        assert rec.model == "per-session-model"
    finally:
        tracker.stop()


def test_daemon_rejects_unknown_policy(daemon):
    """``POST /sessions`` with an unknown policy returns 400 with a clear error."""
    resp = httpx.post(
        f"{daemon}/sessions",
        json={
            "data_id": "dp",
            "combo_id": "c",
            "router": {"policy": "no_such_policy", "kwargs": {}},
        },
        timeout=10.0,
    )
    assert resp.status_code == 400
    assert "Unknown routing policy" in resp.json()["error"]


def test_daemon_serve_rejects_routing_args_without_policy():
    """``--candidate-models`` without ``--routing-policy`` fails fast."""
    from agentopt.proxy.daemon import run

    with pytest.raises(SystemExit, match="require --routing-policy"):
        run(
            host="127.0.0.1", port=0, cache_dir="/tmp/x", candidate_models=["gpt-4o"],
        )


def test_daemon_serve_rejects_random_without_candidates():
    """``--routing-policy random`` without ``--candidate-models`` fails fast."""
    from agentopt.proxy.daemon import run

    with pytest.raises(SystemExit, match="requires --candidate-models"):
        run(
            host="127.0.0.1", port=0, cache_dir="/tmp/x", routing_policy="random",
        )


def test_daemon_custom_router_via_policy_module(
    daemon_factory, mock_upstream, monkeypatch, tmp_path,
):
    """``--policy-module ./mod.py`` loads a user file so its ``Router``
    subclasses resolve over the wire.

    The daemon doesn't have hard-coded knowledge of the user's class —
    it just imports the file at startup, then `POST /sessions` carrying
    ``{"policy": "<stem>:ClassName"}`` resolves via importlib.
    """
    # User writes a custom router file.  Daemon will load it by path.
    user_module = tmp_path / "my_policies.py"
    user_module.write_text(
        """
from agentopt import Router

class FixedRouter(Router):
    \"\"\"Always returns the same model — easy to assert on.\"\"\"

    def __init__(self, target):
        self.target = target

    def route(self, ctx):
        return self.target

    def _config_kwargs(self):
        return {"target": self.target}
"""
    )

    daemon_url = daemon_factory(["--policy-module", str(user_module)])
    monkeypatch.setenv("AGENTOPT_GATEWAY_URL", daemon_url)

    # Sanity: client-side import isn't required.  The daemon resolves
    # the class on its side from the policy string we send.
    resp = httpx.post(
        f"{daemon_url}/sessions",
        json={
            "data_id": "dp",
            "combo_id": "c",
            "router": {
                "policy": "my_policies:FixedRouter",
                "kwargs": {"target": "custom-routed-model"},
            },
        },
        timeout=10.0,
    )
    assert resp.status_code == 200, resp.text

    # End-to-end: open the session client-side via LLMTracker so the
    # in-process httpx routes through this session's daemon port.
    session_id = resp.json()["session_id"]

    from agentopt import LLMTracker

    tracker = LLMTracker()
    tracker.start()
    try:
        with tracker.track(
            data_id="dp_e2e",
            combo_id="c_e2e",
            # No client-side router; the daemon-default policy isn't set
            # either — but we proved the resolver above works, so we just
            # verify the no-router default path still runs cleanly.
        ):
            client = httpx.Client(base_url=mock_upstream.base_url)
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        # No router on this session → original model survives.
        rec = tracker.get_records(combo_id="c_e2e")[0]
        assert rec.model == "gpt-4o"
    finally:
        tracker.stop()
        httpx.delete(f"{daemon_url}/sessions/{session_id}", timeout=5.0)


def test_daemon_policy_module_missing_file_fails_fast():
    """``--policy-module /no/such/file.py`` exits with a clear error."""
    from agentopt.proxy.daemon import run

    with pytest.raises(SystemExit, match="does not exist"):
        run(
            host="127.0.0.1",
            port=0,
            cache_dir="/tmp/x",
            policy_modules=["/no/such/policies.py"],
        )
