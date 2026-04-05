"""Tests for SQLite-based disk persistence of the response cache."""

import json
import sqlite3
import threading
import time

import httpx
import pytest

from agentopt.proxy import LLMTracker
from agentopt.proxy.cache import CacheEntry, ResponseCache, _DB_FILENAME

from .conftest import MOCK_REQUEST_BODY, MOCK_RESPONSE_BODY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESPONSE_BODY = MOCK_RESPONSE_BODY
_REQUEST_BODY = MOCK_REQUEST_BODY


def _make_client() -> httpx.Client:
    return httpx.Client(base_url="https://api.openai.com")


def _post(client: httpx.Client, body: dict | None = None) -> httpx.Response:
    return client.post("/v1/chat/completions", json=body or _REQUEST_BODY)


def _db_row_count(cache_dir) -> int:
    """Return number of rows in the cache table."""
    db_path = cache_dir / _DB_FILENAME
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        (count,) = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        return count
    finally:
        conn.close()


def _register_mock(tracker, mock_upstream):
    """Point the tracker's proxy at the mock upstream."""
    tracker._server.register_provider(
        "openai",
        mock_upstream.base_url,
        ("/v1/chat/completions", "/chat/completions", "/v1/responses"),
    )


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
        text = json.dumps(d)
        assert isinstance(text, str)

    def test_from_dict_defaults(self):
        entry = CacheEntry.from_dict({"response_bytes_b64": "aGVsbG8="})
        assert entry.response_bytes == b"hello"
        assert entry.response_headers == {}
        assert entry.latency_seconds == 0.0


# ---------------------------------------------------------------------------
# Unit tests: ResponseCache with SQLite persistence (lazy flush)
# ---------------------------------------------------------------------------


class TestDiskCache:
    def test_put_does_not_write_immediately(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=cache_dir, flush_interval=0)
        cache.put("key1", CacheEntry(response_bytes=b"data1"))
        assert _db_row_count(cache_dir) == 0
        cache.close()

    def test_flush_writes_dirty_entries(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=cache_dir, flush_interval=0)
        cache.put("key1", CacheEntry(response_bytes=b"data1", latency_seconds=0.5))
        cache.flush()

        assert _db_row_count(cache_dir) == 1

        # Verify stored data
        conn = sqlite3.connect(str(cache_dir / _DB_FILENAME))
        row = conn.execute(
            "SELECT data_json FROM cache WHERE key = ?", ("key1",)
        ).fetchone()
        conn.close()
        data = json.loads(row[0])
        assert "response_bytes_b64" in data
        assert data["latency_seconds"] == 0.5
        cache.close()

    def test_close_flushes(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=cache_dir, flush_interval=0)
        cache.put("k1", CacheEntry(response_bytes=b"v1"))
        cache.close()
        assert _db_row_count(cache_dir) == 1

    def test_entries_loaded_from_disk(self, tmp_path):
        cache_dir = tmp_path / "cache"

        cache1 = ResponseCache(cache_dir=cache_dir, flush_interval=0)
        cache1.put("k1", CacheEntry(response_bytes=b"v1", latency_seconds=1.0))
        cache1.put("k2", CacheEntry(response_bytes=b"v2", latency_seconds=2.0))
        cache1.close()

        cache2 = ResponseCache(cache_dir=cache_dir, flush_interval=0)
        assert len(cache2) == 2
        entry = cache2.get("k1")
        assert entry is not None
        assert entry.response_bytes == b"v1"
        assert entry.latency_seconds == 1.0
        cache2.close()

    def test_clear_removes_db_entries(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=cache_dir, flush_interval=0)
        cache.put("k1", CacheEntry(response_bytes=b"v1"))
        cache.put("k2", CacheEntry(response_bytes=b"v2"))
        cache.flush()
        assert _db_row_count(cache_dir) == 2

        cache.clear()
        assert len(cache) == 0
        assert _db_row_count(cache_dir) == 0
        cache.close()

    def test_corrupt_row_skipped(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Manually create the DB with one good and one corrupt row
        db_path = cache_dir / _DB_FILENAME
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE cache (key TEXT PRIMARY KEY, data_json TEXT NOT NULL)"
        )
        good = CacheEntry(response_bytes=b"good")
        conn.execute(
            "INSERT INTO cache VALUES (?, ?)", ("good", json.dumps(good.to_dict())),
        )
        conn.execute(
            "INSERT INTO cache VALUES (?, ?)", ("bad", "not valid json{{{"),
        )
        conn.commit()
        conn.close()

        cache = ResponseCache(cache_dir=cache_dir, flush_interval=0)
        assert len(cache) == 1
        assert cache.get("good") is not None
        cache.close()

    def test_no_cache_dir_is_memory_only(self, tmp_path):
        cache = ResponseCache()
        cache.put("k", CacheEntry(response_bytes=b"v"))
        assert cache.get("k") is not None
        # No DB files created anywhere
        assert len(list(tmp_path.glob("**/*.db"))) == 0

    def test_cache_dir_created_automatically(self, tmp_path):
        cache_dir = tmp_path / "deep" / "nested" / "cache"
        assert not cache_dir.exists()
        cache = ResponseCache(cache_dir=cache_dir, flush_interval=0)
        assert cache_dir.exists()
        cache.close()

    def test_overwrite_existing_key(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=cache_dir, flush_interval=0)
        cache.put("k", CacheEntry(response_bytes=b"old"))
        cache.put("k", CacheEntry(response_bytes=b"new"))
        cache.close()

        cache2 = ResponseCache(cache_dir=cache_dir, flush_interval=0)
        entry = cache2.get("k")
        assert entry is not None
        assert entry.response_bytes == b"new"
        cache2.close()

    def test_flush_is_idempotent(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=cache_dir, flush_interval=0)
        cache.put("k", CacheEntry(response_bytes=b"v"))
        cache.flush()
        cache.flush()  # no-op
        assert _db_row_count(cache_dir) == 1
        cache.close()

    def test_flush_without_cache_dir_is_noop(self):
        cache = ResponseCache()
        cache.put("k", CacheEntry(response_bytes=b"v"))
        cache.flush()  # should not raise


class TestBackgroundFlush:
    def test_background_thread_flushes(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=cache_dir, flush_interval=0.1)
        cache.put("k", CacheEntry(response_bytes=b"v"))
        time.sleep(0.3)
        assert _db_row_count(cache_dir) == 1
        cache.close()

    def test_close_stops_background_thread(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=cache_dir, flush_interval=0.1)
        assert cache._flush_thread is not None
        assert cache._flush_thread.is_alive()
        cache.close()
        assert cache._flush_thread is None

    def test_no_background_thread_when_interval_zero(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=cache_dir, flush_interval=0)
        assert cache._flush_thread is None
        cache.close()


# ---------------------------------------------------------------------------
# Integration: LLMTracker with disk cache
# ---------------------------------------------------------------------------


class TestTrackerDiskCache:
    def test_tracker_cache_dir_param(self, tmp_path, mock_upstream):
        cache_dir = tmp_path / "llm_cache"
        tracker = LLMTracker(cache=True, cache_dir=cache_dir)
        tracker.start()
        _register_mock(tracker, mock_upstream)
        try:
            client = _make_client()
            with tracker.track(data_id="dp_1", combo_id="c"):
                _post(client)  # miss
                _post(client)  # hit

            records = tracker.get_records()
            assert sum(1 for r in records if r.cached) == 1
            assert sum(1 for r in records if not r.cached) == 1
        finally:
            tracker.stop()

        assert _db_row_count(cache_dir) == 1

    def test_cache_survives_restart(self, tmp_path, mock_upstream):
        cache_dir = tmp_path / "llm_cache"

        tracker1 = LLMTracker(cache=True, cache_dir=cache_dir)
        tracker1.start()
        _register_mock(tracker1, mock_upstream)
        try:
            client = _make_client()
            with tracker1.track(data_id="dp_1", combo_id="c"):
                _post(client)  # miss
        finally:
            tracker1.stop()

        tracker2 = LLMTracker(cache=True, cache_dir=cache_dir)
        tracker2.start()
        _register_mock(tracker2, mock_upstream)
        try:
            client = _make_client()
            with tracker2.track(data_id="dp_2", combo_id="c"):
                _post(client)  # hit from disk

            records = tracker2.get_records()
            assert len(records) == 1
            assert records[0].cached is True
        finally:
            tracker2.stop()

    def test_flush_cache_method(self, tmp_path, mock_upstream):
        cache_dir = tmp_path / "llm_cache"
        tracker = LLMTracker(cache=True, cache_dir=cache_dir)
        tracker.start()
        _register_mock(tracker, mock_upstream)
        try:
            client = _make_client()
            with tracker.track(data_id="dp_1", combo_id="c"):
                _post(client)
            tracker.flush_cache()
            assert _db_row_count(cache_dir) == 1
        finally:
            tracker.stop()

    def test_clear_cache_clears_db(self, tmp_path, mock_upstream):
        cache_dir = tmp_path / "llm_cache"
        tracker = LLMTracker(cache=True, cache_dir=cache_dir)
        tracker.start()
        _register_mock(tracker, mock_upstream)
        try:
            client = _make_client()
            with tracker.track(data_id="dp_1", combo_id="c"):
                _post(client)
            tracker.flush_cache()
            assert _db_row_count(cache_dir) == 1

            tracker.clear_cache()
            assert _db_row_count(cache_dir) == 0
        finally:
            tracker.stop()

    def test_cache_false_ignores_cache_dir(self, tmp_path, mock_upstream):
        cache_dir = tmp_path / "llm_cache"
        tracker = LLMTracker(cache=False, cache_dir=cache_dir)
        tracker.start()
        _register_mock(tracker, mock_upstream)
        try:
            client = _make_client()
            with tracker.track(data_id="dp_1", combo_id="c"):
                _post(client)
            assert not cache_dir.exists()
        finally:
            tracker.stop()

    def test_disk_cache_with_attribution(self, tmp_path, mock_upstream):
        cache_dir = tmp_path / "llm_cache"

        tracker1 = LLMTracker(cache=True, cache_dir=cache_dir)
        tracker1.start()
        _register_mock(tracker1, mock_upstream)
        try:
            client = _make_client()
            with tracker1.track(data_id="dp_1", combo_id="combo_a"):
                _post(client)
        finally:
            tracker1.stop()

        tracker2 = LLMTracker(cache=True, cache_dir=cache_dir)
        tracker2.start()
        _register_mock(tracker2, mock_upstream)
        try:
            client = _make_client()
            with tracker2.track(data_id="dp_2", combo_id="combo_b"):
                _post(client)  # hit from disk

            records = tracker2.get_records()
            assert len(records) == 1
            assert records[0].cached is True
            assert records[0].data_id == "dp_2"
            assert records[0].combo_id == "combo_b"
        finally:
            tracker2.stop()


class TestDiskCacheThreadSafety:
    def test_concurrent_puts_then_flush(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=cache_dir, flush_interval=0)
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
        cache.flush()
        assert _db_row_count(cache_dir) == len(cache)
        cache.close()

    def test_concurrent_flush_safe(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ResponseCache(cache_dir=cache_dir, flush_interval=0)
        errors = []

        for i in range(20):
            cache.put(f"k{i}", CacheEntry(response_bytes=f"v{i}".encode()))

        def flusher():
            try:
                cache.flush()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=flusher) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        cache.close()
