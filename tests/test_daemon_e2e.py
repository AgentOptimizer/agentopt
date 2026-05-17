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


@pytest.fixture
def daemon(tmp_path):
    """Spawn ``agentopt serve`` in a subprocess; yield its base URL."""
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
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(url)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log_handle.close()


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
    monkeypatch.setenv("AGENTOPT_GATEWAY_URL", daemon)
    tracker = LLMTracker()
    tracker.start()
    try:
        with tracker.track(data_id="dp", combo_id="c") as session:
            env = tracker.get_session_env(session)
            assert "HTTPS_PROXY" in env
            assert env["HTTPS_PROXY"].startswith("http://127.0.0.1:")
            # Per-session port is distinct from the control-plane port.
            assert env["HTTPS_PROXY"] != daemon
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


def test_remote_gateway_unreachable_fails_fast(monkeypatch):
    """A bad ``AGENTOPT_GATEWAY_URL`` raises at ``start()`` with a clear message."""
    # Pick a port nothing is listening on.
    monkeypatch.setenv("AGENTOPT_GATEWAY_URL", "http://127.0.0.1:1")
    tracker = LLMTracker()
    with pytest.raises(RuntimeError, match="not reachable"):
        tracker.start()
