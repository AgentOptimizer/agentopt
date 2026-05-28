"""Tests for CallRecord creation: every outcome records, records survive session end.

Covers two guarantees:

* **All outcomes recorded** — 2xx, 4xx, 5xx, and connection failures each
  produce a ``CallRecord`` so cost / latency metrics reflect real work done.
* **Late-arriving records preserved** — records added after ``end_session``
  (e.g. from blocking upstream threads that finish post-teardown) land in
  the archived session, not dropped silently.
"""

import asyncio
import json
import threading

import httpx
import pytest
from aiohttp import web

from agentopt.proxy import LLMTracker
from agentopt.proxy.models import CallRecord
from agentopt.proxy.session import SessionManager


# ---------------------------------------------------------------------------
# Mock upstream that can return arbitrary status codes
# ---------------------------------------------------------------------------


class ControllableUpstream:
    """Mock LLM server whose next response is controllable."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown: asyncio.Event | None = None
        self._started = threading.Event()
        self._port: int | None = None
        self.next_status = 200
        self.next_body: dict = {
            "id": "c1",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

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
        return web.json_response(self.next_body, status=self.next_status)


@pytest.fixture
def upstream():
    s = ControllableUpstream()
    s.start()
    yield s
    s.stop()


# ---------------------------------------------------------------------------
# #6: Failed requests are recorded
# ---------------------------------------------------------------------------


def test_successful_request_records_status_200(upstream):
    """A 200 response records status_code=200 and error=None."""
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with tracker.track(data_id="dp_1", combo_id="c"):
            client = httpx.Client(base_url=upstream.base_url)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 200
        records = tracker.get_records()
        assert len(records) == 1
        assert records[0].status_code == 200
        assert records[0].error is None
        assert records[0].prompt_tokens == 3
        assert records[0].completion_tokens == 2
    finally:
        tracker.stop()


def test_http_error_request_is_recorded(upstream):
    """A 500 response is recorded with status_code=500 and error set."""
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    upstream.next_status = 500
    upstream.next_body = {"error": {"message": "internal error"}}
    try:
        with tracker.track(data_id="dp_1", combo_id="c"):
            client = httpx.Client(base_url=upstream.base_url)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 500
        records = tracker.get_records()
        assert len(records) == 1
        assert records[0].status_code == 500
        assert records[0].error == "HTTP 500"
        # No usage in error body → token counts default to 0.
        assert records[0].prompt_tokens == 0
        assert records[0].completion_tokens == 0
        # Latency should still be recorded.
        assert records[0].latency_seconds > 0
    finally:
        tracker.stop()


def test_rate_limit_is_recorded(upstream):
    """A 429 rate-limit response is recorded."""
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    upstream.next_status = 429
    upstream.next_body = {"error": {"message": "rate limited"}}
    try:
        with tracker.track(data_id="dp_1", combo_id="c"):
            client = httpx.Client(base_url=upstream.base_url)
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        records = tracker.get_records()
        assert len(records) == 1
        assert records[0].status_code == 429
        assert records[0].error == "HTTP 429"
    finally:
        tracker.stop()


def test_connection_failure_is_recorded():
    """A connection failure (upstream unreachable) is recorded with status_code=0."""
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with tracker.track(data_id="dp_1", combo_id="c"):
            # Point at a port nothing is listening on.
            client = httpx.Client(base_url="http://127.0.0.1:1")
            try:
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            except Exception:
                pass
        records = tracker.get_records()
        assert len(records) == 1
        assert records[0].status_code == 0
        assert records[0].error is not None
        assert records[0].prompt_tokens == 0
    finally:
        tracker.stop()


# ---------------------------------------------------------------------------
# #10: Late records reach ended sessions
# ---------------------------------------------------------------------------


def test_late_record_reaches_ended_session():
    """Records added after end_session() are preserved in the archive."""
    mgr = SessionManager()
    session = mgr.create_session(data_id="dp_1", combo_id="c")
    mgr.end_session(session.session_id)

    # Simulate a late-arriving record from a blocking thread.
    late = CallRecord(
        data_id="dp_1",
        combo_id="c",
        agent_id=None,
        model="gpt-4o",
        prompt_tokens=5,
        completion_tokens=3,
        latency_seconds=0.1,
        request_url="https://api.openai.com/v1/chat/completions",
    )
    mgr.add_record(session.session_id, late)

    all_records = mgr.get_all_records()
    assert len(all_records) == 1
    assert all_records[0].prompt_tokens == 5


def test_record_to_unknown_session_is_dropped():
    """Records for a session that never existed are silently dropped (no crash)."""
    mgr = SessionManager()
    late = CallRecord(
        data_id="dp_1",
        combo_id="c",
        agent_id=None,
        model="gpt-4o",
        prompt_tokens=5,
        completion_tokens=3,
        latency_seconds=0.1,
        request_url="https://api.openai.com/v1/chat/completions",
    )
    mgr.add_record("sess_does_not_exist", late)
    assert mgr.get_all_records() == []
