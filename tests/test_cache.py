"""Tests for agentopt.proxy API-level response cache."""

import asyncio
import json
import threading

import httpx
import pytest

from agentopt.proxy import LLMTracker
from agentopt.proxy.cache import CacheEntry, ResponseCache, _make_cache_key

from .conftest import MOCK_REQUEST_BODY, MOCK_RESPONSE_BODY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESPONSE_BODY = MOCK_RESPONSE_BODY
_RESPONSE_BYTES = json.dumps(_RESPONSE_BODY).encode()
_REQUEST_BODY = MOCK_REQUEST_BODY


def _make_client(base_url: str = "https://api.openai.com") -> httpx.Client:
    """Plain httpx client — requests go through the monkey-patched send."""
    return httpx.Client(base_url=base_url)


def _post(client: httpx.Client, body: dict | None = None) -> httpx.Response:
    return client.post("/v1/chat/completions", json=body or _REQUEST_BODY)


# ---------------------------------------------------------------------------
# Unit tests: ResponseCache
# ---------------------------------------------------------------------------


class TestResponseCache:
    def test_get_miss(self):
        cache = ResponseCache()
        assert cache.get("nonexistent") is None

    def test_put_and_get_hit(self):
        cache = ResponseCache()
        entry = CacheEntry(response_bytes=b"hello")
        cache.put("key1", entry)
        result = cache.get("key1")
        assert result is not None
        assert result.response_bytes == b"hello"

    def test_clear(self):
        cache = ResponseCache()
        cache.put("k", CacheEntry(response_bytes=b"v"))
        cache.get("k")
        cache.clear()
        assert len(cache) == 0


class TestMakeCacheKey:
    def test_deterministic(self):
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        assert _make_cache_key(body) == _make_cache_key(body)

    def test_different_content_different_key(self):
        body1 = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        body2 = {"model": "gpt-4o", "messages": [{"role": "user", "content": "bye"}]}
        assert _make_cache_key(body1) != _make_cache_key(body2)

    def test_different_model_different_key(self):
        body1 = {"model": "gpt-4o", "messages": []}
        body2 = {"model": "gpt-4o-mini", "messages": []}
        assert _make_cache_key(body1) != _make_cache_key(body2)

    def test_stream_field_ignored(self):
        body1 = {"model": "gpt-4o", "messages": [], "stream": True}
        body2 = {"model": "gpt-4o", "messages": [], "stream": False}
        body3 = {"model": "gpt-4o", "messages": []}
        assert (
            _make_cache_key(body1) == _make_cache_key(body2) == _make_cache_key(body3)
        )

    def test_key_order_independent(self):
        body1 = {"model": "gpt-4o", "temperature": 0.5, "messages": []}
        body2 = {"messages": [], "temperature": 0.5, "model": "gpt-4o"}
        assert _make_cache_key(body1) == _make_cache_key(body2)


# ---------------------------------------------------------------------------
# Integration tests: cache through proxy server
# ---------------------------------------------------------------------------


class TestCacheIntegration:
    def setup_method(self, method, mock_upstream=None):
        # Called via the fixture-based tests below.
        pass

    @pytest.fixture(autouse=True)
    def _setup(self, mock_upstream):
        self.mock_upstream = mock_upstream
        self.tracker = LLMTracker(cache=True, cache_dir=None)
        self.tracker.start()
        yield
        self.tracker.stop()

    def _client(self):
        """Client pointing at mock upstream — header routing handles the rest."""
        return _make_client(base_url=self.mock_upstream.base_url)

    def test_cache_hit_returns_same_response(self):
        client = self._client()
        with self.tracker.track(data_id="dp_1", combo_id="c"):
            resp1 = _post(client)
            resp2 = _post(client)
        assert resp1.json() == resp2.json()
        assert resp1.status_code == resp2.status_code == 200

    def test_cache_hit_recorded_with_original_latency(self):
        client = self._client()
        with self.tracker.track(data_id="dp_1", combo_id="c"):
            _post(client)  # miss
            _post(client)  # hit

        records = self.tracker.get_records()
        assert len(records) == 2
        assert records[0].cached is False
        assert records[0].latency_seconds > 0
        assert records[1].cached is True
        assert records[1].latency_seconds == records[0].latency_seconds

    def test_cache_hit_records_tokens(self):
        client = self._client()
        with self.tracker.track(data_id="dp_1", combo_id="c"):
            _post(client)
            _post(client)

        records = self.tracker.get_records()
        for r in records:
            assert r.prompt_tokens == 10
            assert r.completion_tokens == 5

    def test_cache_hit_count(self):
        client = self._client()
        with self.tracker.track(data_id="dp_1", combo_id="c"):
            _post(client)  # miss
            _post(client)  # hit
            _post(client)  # hit

        records = self.tracker.get_records()
        assert sum(1 for r in records if r.cached) == 2
        assert sum(1 for r in records if not r.cached) == 1

    def test_different_requests_no_hit(self):
        client = self._client()
        body1 = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "q1"}],
        }
        body2 = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "q2"}],
        }
        with self.tracker.track(data_id="dp_1", combo_id="c"):
            _post(client, body1)
            _post(client, body2)

        records = self.tracker.get_records()
        assert all(not r.cached for r in records)

    def test_clear_cache(self):
        client = self._client()
        with self.tracker.track(data_id="dp_1", combo_id="c"):
            _post(client)
        self.tracker.clear_cache()
        with self.tracker.track(data_id="dp_2", combo_id="c"):
            _post(client)  # should miss again

        records = self.tracker.get_records()
        assert len(records) == 2
        assert all(not r.cached for r in records)

    def test_cache_with_attribution(self):
        client = self._client()
        with self.tracker.track(data_id="dp_1", combo_id="combo_a"):
            _post(client)  # miss
            _post(client)  # hit

        records = self.tracker.get_records(combo_id="combo_a")
        assert len(records) == 2
        assert all(r.combo_id == "combo_a" for r in records)
        assert all(r.data_id == "dp_1" for r in records)

    def test_non_llm_request_not_intercepted(self):
        """GET /health is not an LLM endpoint — should not be redirected."""
        handler = lambda req: httpx.Response(200, json={"status": "ok"})
        client = httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://api.openai.com",
        )
        with self.tracker.track(data_id="dp_1", combo_id="c"):
            resp = client.get("/health")
            # Verify the request was NOT rewritten to the proxy URL.
            assert resp.status_code == 200
        assert len(self.tracker.get_records()) == 0


class TestCacheDisabled:
    def test_disabled_via_constructor(self, mock_upstream):
        tracker = LLMTracker(cache=False)
        tracker.start()
        try:
            client = _make_client(base_url=mock_upstream.base_url)
            with tracker.track(data_id="dp_1", combo_id="c"):
                _post(client)
                _post(client)

            records = tracker.get_records()
            assert len(records) == 2
            assert all(r.cached is False for r in records)
        finally:
            tracker.stop()


class TestCacheThreadSafety:
    def test_concurrent_access(self):
        cache = ResponseCache()
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(100):
                    key = f"t{thread_id}_k{i}"
                    cache.put(key, CacheEntry(response_bytes=f"v{i}".encode()))
                    cache.get(key)
                    cache.get(f"nonexistent_{thread_id}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(cache) == 800  # 8 threads × 100 keys

    def test_concurrent_tracker_with_cache(self, mock_upstream):
        tracker = LLMTracker(cache=True, cache_dir=None)
        tracker.start()
        errors = []

        def worker():
            try:
                client = _make_client(base_url=mock_upstream.base_url)
                with tracker.track(data_id="dp", combo_id="c"):
                    for _ in range(5):
                        _post(client)
            except Exception as e:
                errors.append(e)

        try:
            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            records = tracker.get_records()
            assert len(records) == 20
            cached_count = sum(1 for r in records if r.cached)
            assert cached_count > 0
        finally:
            tracker.stop()


class TestCacheAsync:
    def test_async_cache_hit(self, mock_upstream):
        async def _run():
            tracker = LLMTracker(cache=True, cache_dir=None)
            tracker.start()
            try:
                client = httpx.AsyncClient(base_url=mock_upstream.base_url)

                with tracker.track(data_id="dp_1", combo_id="c"):
                    resp1 = await client.post(
                        "/v1/chat/completions", json=_REQUEST_BODY
                    )
                    resp2 = await client.post(
                        "/v1/chat/completions", json=_REQUEST_BODY
                    )

                records = tracker.get_records()
                assert len(records) == 2
                assert records[0].cached is False
                assert records[1].cached is True
                assert records[1].latency_seconds == records[0].latency_seconds
                assert resp1.json() == resp2.json()
            finally:
                tracker.stop()

        asyncio.run(_run())
