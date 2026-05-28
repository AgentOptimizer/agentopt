"""Shared fixtures for agentopt proxy tests."""

import asyncio
import json
import threading

import pytest
from aiohttp import web

# ---------------------------------------------------------------------------
# Standard mock LLM response (OpenAI chat completions format)
# ---------------------------------------------------------------------------

MOCK_RESPONSE_BODY = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Paris"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

MOCK_REQUEST_BODY = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}


# ---------------------------------------------------------------------------
# Mock upstream LLM API server
# ---------------------------------------------------------------------------


class MockUpstream:
    """A tiny aiohttp server that pretends to be an LLM API provider."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown: asyncio.Event | None = None
        self._started = threading.Event()
        self._port: int | None = None
        self.request_count = 0
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        assert self._port is not None
        return f"http://127.0.0.1:{self._port}"

    def start(self) -> None:
        self._started.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._started.wait(timeout=5), "Mock upstream failed to start"

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
        # Accept any POST to any path — act like a universal LLM API.
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
        with self._lock:
            self.request_count += 1
        return web.json_response(MOCK_RESPONSE_BODY)


@pytest.fixture()
def mock_upstream():
    """Start a mock LLM API server for the duration of a test."""
    server = MockUpstream()
    server.start()
    yield server
    server.stop()
