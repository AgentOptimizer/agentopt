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
    return httpx.Response(200, json=_RESPONSE_BODY)


def _make_client() -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(_fake_handler),
        base_url="https://api.openai.com",
    )


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
        assert _make_cache_key(body1) == _make_cache_key(body2) == _make_cache_key(body3)

    def test_key_order_independent(self):
        body1 = {"model": "gpt-4o", "temperature": 0.5, "messages": []}
        body2 = {"messages": [], "temperature": 0.5, "model": "gpt-4o"}
        assert _make_cache_key(body1) == _make_cache_key(body2)


# ---------------------------------------------------------------------------
# Integration tests: cache through LLMTracker + httpx interception
# ---------------------------------------------------------------------------


class TestCacheIntegration:
    def setup_method(self):
        self.tracker = LLMTracker(cache=True, cache_dir=None)
        self.tracker.start()

    def teardown_method(self):
        self.tracker.stop()

    def test_cache_hit_returns_same_response(self):
        client = _make_client()
        resp1 = _post(client)
        resp2 = _post(client)
        assert resp1.json() == resp2.json()
        assert resp1.status_code == resp2.status_code == 200

    def test_cache_hit_recorded_with_original_latency(self):
        client = _make_client()
        _post(client)  # miss
        _post(client)  # hit

        records = self.tracker.get_records()
        assert len(records) == 2
        assert records[0].cached is False
        assert records[0].latency_seconds > 0
        assert records[1].cached is True
        assert records[1].latency_seconds == records[0].latency_seconds

    def test_cache_hit_records_tokens(self):
        client = _make_client()
        _post(client)
        _post(client)

        records = self.tracker.get_records()
        for r in records:
            assert r.prompt_tokens == 10
            assert r.completion_tokens == 5

    def test_cache_hit_count(self):
        client = _make_client()
        _post(client)  # miss
        _post(client)  # hit
        _post(client)  # hit

        records = self.tracker.get_records()
        assert sum(1 for r in records if r.cached) == 2
        assert sum(1 for r in records if not r.cached) == 1

    def test_different_requests_no_hit(self):
        client = _make_client()
        body1 = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "q1"}]}
        body2 = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "q2"}]}
        _post(client, body1)
        _post(client, body2)

        records = self.tracker.get_records()
        assert all(not r.cached for r in records)

    def test_clear_cache(self):
        client = _make_client()
        _post(client)
        self.tracker.clear_cache()
        _post(client)  # should miss again

        records = self.tracker.get_records()
        assert len(records) == 2
        assert all(not r.cached for r in records)

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
        try:
            client.get("/health")
        except Exception:
            pass
        assert len(self.tracker.get_records()) == 0


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

    def test_concurrent_tracker_with_cache(self):
        tracker = LLMTracker(cache=True, cache_dir=None)
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
            assert len(records) == 20
            cached_count = sum(1 for r in records if r.cached)
            assert cached_count > 0
        finally:
            tracker.stop()


class TestCacheAsync:
    def test_async_cache_hit(self):
        async def _run():
            tracker = LLMTracker(cache=True, cache_dir=None)
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
                assert records[1].latency_seconds == records[0].latency_seconds
                assert resp1.json() == resp2.json()
            finally:
                tracker.stop()

        asyncio.run(_run())
