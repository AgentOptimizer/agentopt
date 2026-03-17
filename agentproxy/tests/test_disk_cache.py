"""Tests for JSON file-based disk persistence of the response cache."""

import json
import threading

import httpx
import pytest

from agentproxy import LLMTracker
from agentproxy.cache import CacheEntry, ResponseCache

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
# Unit tests: CacheEntry serialization
# ---------------------------------------------------------------------------


class TestCacheEntrySerialization:
    def test_round_trip(self):
        entry = CacheEntry(
            response_bytes=b'{"hello": "world"}',
            response_headers={"content-type": "application/json"},
            latency_seconds=1.23,
        )
        d = entry.to_dict()
        restored = CacheEntry.from_dict(d)
        assert restored.response_bytes == entry.response_bytes
        assert restored.response_headers == entry.response_headers
        assert restored.latency_seconds == entry.latency_seconds

    def test_to_dict_is_json_serializable(self):
        entry = CacheEntry(response_bytes=b"\x00\xff binary data")
        d = entry.to_dict()
        # Must not raise
        text = json.dumps(d)
        assert isinstance(text, str)

    def test_from_dict_defaults(self):
        entry = CacheEntry.from_dict({"response_bytes_b64": "aGVsbG8="})
        assert entry.response_bytes == b"hello"
        assert entry.response_headers == {}
        assert entry.latency_seconds == 0.0


# ---------------------------------------------------------------------------
# Unit tests: ResponseCache with disk persistence
# ---------------------------------------------------------------------------


class TestDiskCache:
    def test_entries_written_to_disk(self, tmp_path):
        cache = ResponseCache(cache_dir=tmp_path / "cache")
        cache.put("key1", CacheEntry(response_bytes=b"data1", latency_seconds=0.5))

        files = list((tmp_path / "cache").glob("*.json"))
        assert len(files) == 1
        assert files[0].stem == "key1"

        data = json.loads(files[0].read_text())
        assert "response_bytes_b64" in data
        assert data["latency_seconds"] == 0.5

    def test_entries_loaded_from_disk(self, tmp_path):
        cache_dir = tmp_path / "cache"

        # Write entries with one cache instance
        cache1 = ResponseCache(cache_dir=cache_dir)
        cache1.put("k1", CacheEntry(response_bytes=b"v1", latency_seconds=1.0))
        cache1.put("k2", CacheEntry(response_bytes=b"v2", latency_seconds=2.0))

        # Create a new cache instance — should load from disk
        cache2 = ResponseCache(cache_dir=cache_dir)
        assert len(cache2) == 2

        entry = cache2.get("k1")
        assert entry is not None
        assert entry.response_bytes == b"v1"
        assert entry.latency_seconds == 1.0

        entry2 = cache2.get("k2")
        assert entry2 is not None
        assert entry2.response_bytes == b"v2"

    def test_clear_removes_disk_files(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=cache_dir)
        cache.put("k1", CacheEntry(response_bytes=b"v1"))
        cache.put("k2", CacheEntry(response_bytes=b"v2"))

        assert len(list(cache_dir.glob("*.json"))) == 2

        cache.clear()
        assert len(cache) == 0
        assert len(list(cache_dir.glob("*.json"))) == 0

    def test_eviction_removes_disk_file(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(max_size=2, cache_dir=cache_dir)
        cache.put("a", CacheEntry(response_bytes=b"1"))
        cache.put("b", CacheEntry(response_bytes=b"2"))
        cache.put("c", CacheEntry(response_bytes=b"3"))  # evicts "a"

        assert cache.get("a") is None
        assert not (cache_dir / "a.json").exists()
        assert (cache_dir / "b.json").exists()
        assert (cache_dir / "c.json").exists()

    def test_max_size_enforced_on_load(self, tmp_path):
        cache_dir = tmp_path / "cache"

        # Write 5 entries with unlimited cache
        cache1 = ResponseCache(cache_dir=cache_dir)
        for i in range(5):
            cache1.put(f"k{i}", CacheEntry(response_bytes=f"v{i}".encode()))

        # Load with max_size=3 — should keep only 3
        cache2 = ResponseCache(max_size=3, cache_dir=cache_dir)
        assert len(cache2) == 3

    def test_corrupt_file_skipped(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Write a valid entry
        valid = CacheEntry(response_bytes=b"good")
        (cache_dir / "good.json").write_text(json.dumps(valid.to_dict()))

        # Write a corrupt file
        (cache_dir / "bad.json").write_text("not valid json{{{")

        cache = ResponseCache(cache_dir=cache_dir)
        assert len(cache) == 1
        assert cache.get("good") is not None

    def test_no_cache_dir_is_memory_only(self, tmp_path):
        cache = ResponseCache()
        cache.put("k", CacheEntry(response_bytes=b"v"))
        assert cache.get("k") is not None
        # No files created anywhere
        assert len(list(tmp_path.glob("**/*.json"))) == 0

    def test_cache_dir_created_automatically(self, tmp_path):
        cache_dir = tmp_path / "deep" / "nested" / "cache"
        assert not cache_dir.exists()
        ResponseCache(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_overwrite_existing_key(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=cache_dir)
        cache.put("k", CacheEntry(response_bytes=b"old"))
        cache.put("k", CacheEntry(response_bytes=b"new"))

        # Disk should have updated value
        cache2 = ResponseCache(cache_dir=cache_dir)
        entry = cache2.get("k")
        assert entry is not None
        assert entry.response_bytes == b"new"


# ---------------------------------------------------------------------------
# Integration: LLMTracker with disk cache
# ---------------------------------------------------------------------------


class TestTrackerDiskCache:
    def test_tracker_cache_dir_param(self, tmp_path):
        cache_dir = tmp_path / "llm_cache"
        tracker = LLMTracker(cache=True, cache_dir=cache_dir)
        tracker.start()
        try:
            client = _make_client()
            _post(client)  # miss
            _post(client)  # hit

            assert tracker.cache_stats.misses == 1
            assert tracker.cache_stats.hits == 1

            # Verify files on disk
            files = list(cache_dir.glob("*.json"))
            assert len(files) == 1
        finally:
            tracker.stop()

    def test_cache_survives_restart(self, tmp_path):
        cache_dir = tmp_path / "llm_cache"

        # First tracker: populate cache
        tracker1 = LLMTracker(cache=True, cache_dir=cache_dir)
        tracker1.start()
        try:
            client = _make_client()
            _post(client)  # miss, stored to disk
            assert tracker1.cache_stats.misses == 1
        finally:
            tracker1.stop()

        # Second tracker: should load from disk
        tracker2 = LLMTracker(cache=True, cache_dir=cache_dir)
        tracker2.start()
        try:
            client = _make_client()
            _post(client)  # should be a hit from disk cache

            assert tracker2.cache_stats.hits == 1
            assert tracker2.cache_stats.misses == 0
        finally:
            tracker2.stop()

    def test_clear_cache_clears_disk(self, tmp_path):
        cache_dir = tmp_path / "llm_cache"
        tracker = LLMTracker(cache=True, cache_dir=cache_dir)
        tracker.start()
        try:
            client = _make_client()
            _post(client)
            assert len(list(cache_dir.glob("*.json"))) == 1

            tracker.clear_cache()
            assert len(list(cache_dir.glob("*.json"))) == 0
        finally:
            tracker.stop()

    def test_cache_false_ignores_cache_dir(self, tmp_path):
        cache_dir = tmp_path / "llm_cache"
        tracker = LLMTracker(cache=False, cache_dir=cache_dir)
        tracker.start()
        try:
            client = _make_client()
            _post(client)
            # No cache dir created when cache=False
            assert not cache_dir.exists()
        finally:
            tracker.stop()

    def test_disk_cache_with_attribution(self, tmp_path):
        cache_dir = tmp_path / "llm_cache"

        # First run: populate cache with attribution
        tracker1 = LLMTracker(cache=True, cache_dir=cache_dir)
        tracker1.start()
        try:
            client = _make_client()
            with tracker1.track(data_id="dp_1", combo_id="combo_a"):
                _post(client)
        finally:
            tracker1.stop()

        # Second run: hit from disk cache
        tracker2 = LLMTracker(cache=True, cache_dir=cache_dir)
        tracker2.start()
        try:
            client = _make_client()
            with tracker2.track(data_id="dp_2", combo_id="combo_b"):
                _post(client)  # hit

            records = tracker2.get_records()
            assert len(records) == 1
            assert records[0].cached is True
            assert records[0].data_id == "dp_2"
            assert records[0].combo_id == "combo_b"
        finally:
            tracker2.stop()


class TestDiskCacheThreadSafety:
    def test_concurrent_disk_writes(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=cache_dir)
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(50):
                    key = f"t{thread_id}_k{i}"
                    cache.put(key, CacheEntry(response_bytes=f"v{i}".encode()))
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All files should exist
        files = list(cache_dir.glob("*.json"))
        assert len(files) == len(cache)
