"""Tests for agentproxy API-level response cache."""

import asyncio
import json
import threading

import httpx
import pytest

from agentproxy import LLMTracker
from agentproxy.cache import CacheEntry, ResponseCache, _make_cache_key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A minimal OpenAI-style chat completion response
_RESPONSE_BODY = {
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
_RESPONSE_BYTES = json.dumps(_RESPONSE_BODY).encode()

_REQUEST_BODY = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}


def _fake_handler(request: httpx.Request) -> httpx.Response:
    """httpx mock transport handler returning a canned response."""
    return httpx.Response(200, json=_RESPONSE_BODY)


def _make_client() -> httpx.Client:
    """Create an httpx client with a mock transport targeting an LLM endpoint."""
    return httpx.Client(
        transport=httpx.MockTransport(_fake_handler),
        base_url="https://api.openai.com",
    )


def _post(client: httpx.Client, body: dict | None = None) -> httpx.Response:
    """Send a chat completion request through the client."""
    return client.post(
        "/v1/chat/completions",
        json=body or _REQUEST_BODY,
    )


# ---------------------------------------------------------------------------
# Unit tests: ResponseCache
# ---------------------------------------------------------------------------


class TestResponseCache:
    def test_get_miss(self):
        cache = ResponseCache()
        assert cache.get("nonexistent") is None
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

    def test_put_and_get_hit(self):
        cache = ResponseCache()
        entry = CacheEntry(response_bytes=b"hello")
        cache.put("key1", entry)
        result = cache.get("key1")
        assert result is not None
        assert result.response_bytes == b"hello"
        assert cache.stats.hits == 1

    def test_max_size_eviction(self):
        cache = ResponseCache(max_size=2)
        cache.put("a", CacheEntry(response_bytes=b"1"))
        cache.put("b", CacheEntry(response_bytes=b"2"))
        cache.put("c", CacheEntry(response_bytes=b"3"))  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None
        assert len(cache) == 2

    def test_clear(self):
        cache = ResponseCache()
        cache.put("k", CacheEntry(response_bytes=b"v"))
        cache.get("k")
        cache.clear()
        assert len(cache) == 0
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0

    def test_hit_rate(self):
        cache = ResponseCache()
        cache.put("k", CacheEntry(response_bytes=b"v"))
        cache.get("k")       # hit
        cache.get("missing")  # miss
        assert cache.stats.hit_rate == 0.5


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
        key1 = _make_cache_key(body1)
        key2 = _make_cache_key(body2)
        key3 = _make_cache_key(body3)
        assert key1 == key2 == key3

    def test_key_order_independent(self):
        body1 = {"model": "gpt-4o", "temperature": 0.5, "messages": []}
        body2 = {"messages": [], "temperature": 0.5, "model": "gpt-4o"}
        assert _make_cache_key(body1) == _make_cache_key(body2)


# ---------------------------------------------------------------------------
# Integration tests: cache through LLMTracker + httpx interception
# ---------------------------------------------------------------------------


class TestCacheIntegration:
    def setup_method(self):
        """Fresh tracker for each test."""
        self.tracker = LLMTracker(cache=True)
        self.tracker.start()

    def teardown_method(self):
        self.tracker.stop()

    def test_cache_hit_returns_same_response(self):
        client = _make_client()

        resp1 = _post(client)
        resp2 = _post(client)

        assert resp1.json() == resp2.json()
        assert resp1.status_code == resp2.status_code == 200

    def test_cache_hit_recorded_with_zero_latency(self):
        client = _make_client()

        _post(client)  # miss
        _post(client)  # hit

        records = self.tracker.get_records()
        assert len(records) == 2
        assert records[0].cached is False
        assert records[0].latency_seconds > 0
        assert records[1].cached is True
        assert records[1].latency_seconds == 0.0

    def test_cache_hit_records_tokens(self):
        client = _make_client()

        _post(client)  # miss
        _post(client)  # hit

        records = self.tracker.get_records()
        # Both records should have the same token counts
        for r in records:
            assert r.prompt_tokens == 10
            assert r.completion_tokens == 5

    def test_cache_stats(self):
        client = _make_client()

        _post(client)  # miss
        _post(client)  # hit
        _post(client)  # hit

        stats = self.tracker.cache_stats
        assert stats.misses == 1
        assert stats.hits == 2
        assert stats.hit_rate == pytest.approx(2 / 3)

    def test_different_requests_no_hit(self):
        client = _make_client()

        body1 = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "q1"}]}
        body2 = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "q2"}]}
        _post(client, body1)
        _post(client, body2)

        stats = self.tracker.cache_stats
        assert stats.misses == 2
        assert stats.hits == 0

    def test_clear_cache(self):
        client = _make_client()

        _post(client)
        assert self.tracker.cache_stats.misses == 1

        self.tracker.clear_cache()
        _post(client)  # should miss again
        assert self.tracker.cache_stats.misses == 1  # stats reset too
        assert self.tracker.cache_stats.hits == 0

    def test_cache_with_attribution(self):
        client = _make_client()

        with self.tracker.track(data_id="dp_1", combo_id="combo_a"):
            _post(client)  # miss
            _post(client)  # hit

        records = self.tracker.get_records(combo_id="combo_a")
        assert len(records) == 2
        assert all(r.combo_id == "combo_a" for r in records)
        assert all(r.data_id == "dp_1" for r in records)

    def test_non_llm_request_not_cached(self):
        client = _make_client()
        # GET request — not an LLM endpoint
        try:
            client.get("/health")
        except Exception:
            pass
        assert self.tracker.cache_stats.total == 0


class TestCacheDisabled:
    def test_disabled_via_constructor(self):
        tracker = LLMTracker(cache=False)
        tracker.start()
        try:
            client = _make_client()
            _post(client)
            _post(client)

            records = tracker.get_records()
            assert len(records) == 2
            assert all(r.cached is False for r in records)
            assert tracker.cache_stats.hits == 0
            assert tracker.cache_stats.misses == 0  # no cache => no tracking
        finally:
            tracker.stop()

    def test_disabled_at_runtime(self):
        tracker = LLMTracker(cache=True)
        tracker.start()
        try:
            client = _make_client()
            _post(client)  # miss, stored

            tracker.cache_enabled = False
            _post(client)  # should NOT hit cache

            records = tracker.get_records()
            assert len(records) == 2
            assert records[0].cached is False
            assert records[1].cached is False  # no cache hit despite stored entry

            tracker.cache_enabled = True
            _post(client)  # should hit cache now

            records = tracker.get_records()
            assert len(records) == 3
            assert records[2].cached is True
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
        assert cache.stats.hits > 0
        assert cache.stats.misses > 0

    def test_concurrent_tracker_with_cache(self):
        tracker = LLMTracker(cache=True)
        tracker.start()
        errors = []

        def worker():
            try:
                client = _make_client()
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
            # 4 threads × 5 calls = 20 records (some cached)
            assert len(records) == 20
            cached_count = sum(1 for r in records if r.cached)
            assert cached_count > 0  # at least some hits
        finally:
            tracker.stop()


class TestCacheAsync:
    def test_async_cache_hit(self):
        async def _run():
            tracker = LLMTracker(cache=True)
            tracker.start()
            try:
                async def _fake_async_handler(request: httpx.Request) -> httpx.Response:
                    return httpx.Response(200, json=_RESPONSE_BODY)

                client = httpx.AsyncClient(
                    transport=httpx.MockTransport(_fake_async_handler),
                    base_url="https://api.openai.com",
                )

                resp1 = await client.post("/v1/chat/completions", json=_REQUEST_BODY)
                resp2 = await client.post("/v1/chat/completions", json=_REQUEST_BODY)

                records = tracker.get_records()
                assert len(records) == 2
                assert records[0].cached is False
                assert records[1].cached is True
                assert records[1].latency_seconds == 0.0
                assert resp1.json() == resp2.json()

                assert tracker.cache_stats.hits == 1
                assert tracker.cache_stats.misses == 1
            finally:
                tracker.stop()

        asyncio.run(_run())
