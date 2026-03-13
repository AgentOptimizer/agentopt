"""Tests for EvalCache — unit tests and integration with the evaluation loop."""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentopt.cache import EvalCache, NoCache

# ---------------------------------------------------------------------------
# EvalCache unit tests
# ---------------------------------------------------------------------------


class TestEvalCacheMakeKey:
    def test_deterministic(self):
        k1 = EvalCache.make_key("gpt-4o", "hello")
        k2 = EvalCache.make_key("gpt-4o", "hello")
        assert k1 == k2

    def test_length_is_64(self):
        key = EvalCache.make_key("m", "in")
        assert len(key) == 64

    def test_different_model_different_key(self):
        k1 = EvalCache.make_key("gpt-4o", "q")
        k2 = EvalCache.make_key("gpt-4o-mini", "q")
        assert k1 != k2

    def test_different_input_different_key(self):
        k1 = EvalCache.make_key("m", "q1")
        k2 = EvalCache.make_key("m", "q2")
        assert k1 != k2

    def test_same_input_different_expected_same_key(self):
        """Expected answer is not part of the cache key."""
        k1 = EvalCache.make_key("m", "q")
        k2 = EvalCache.make_key("m", "q")
        assert k1 == k2


class TestEvalCacheGetSet:
    def test_miss_returns_none(self, tmp_path):
        cache = EvalCache(tmp_path / "c.json")
        assert cache.get("nokey") is None

    def test_set_then_get(self, tmp_path):
        cache = EvalCache(tmp_path / "c.json")
        entry = {"score": 1.0, "latency": 0.5, "response": "Paris", "tokens": {}}
        cache.set("k1", entry)
        assert cache.get("k1") == entry

    def test_contains(self, tmp_path):
        cache = EvalCache(tmp_path / "c.json")
        cache.set("k1", {"score": 1.0})
        assert "k1" in cache
        assert "k2" not in cache

    def test_len(self, tmp_path):
        cache = EvalCache(tmp_path / "c.json")
        assert len(cache) == 0
        cache.set("k1", {"score": 1.0})
        assert len(cache) == 1
        cache.set("k2", {"score": 0.5})
        assert len(cache) == 2


class TestEvalCachePersistence:
    def test_persists_to_disk(self, tmp_path):
        path = tmp_path / "c.json"
        cache = EvalCache(path, flush_interval=1)
        cache.set("k1", {"score": 1.0, "latency": 0.3})
        cache2 = EvalCache(path)
        assert cache2.get("k1") == {"score": 1.0, "latency": 0.3}

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "c.json"
        cache = EvalCache(path, flush_interval=1)
        cache.set("k", {"score": 0.5})
        assert path.exists()

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("not valid json!!!", encoding="utf-8")
        cache = EvalCache(path)
        assert len(cache) == 0
        cache.set("k", {"score": 1.0})
        assert cache.get("k") == {"score": 1.0}

    def test_empty_file_starts_fresh(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("", encoding="utf-8")
        cache = EvalCache(path)
        assert len(cache) == 0


class TestEvalCacheSaveScore:
    def test_save_score_default_false(self, tmp_path):
        cache = EvalCache(tmp_path / "c.json")
        assert cache.save_score is False

    def test_save_score_true(self, tmp_path):
        cache = EvalCache(tmp_path / "c.json", save_score=True)
        assert cache.save_score is True


class TestEvalCacheFlush:
    def test_flush_interval_batches_writes(self, tmp_path):
        path = tmp_path / "c.json"
        cache = EvalCache(path, flush_interval=3)
        cache.set("k1", {"score": 1.0})
        cache.set("k2", {"score": 2.0})
        # Not yet flushed — file should not exist yet.
        assert not path.exists()
        cache.set("k3", {"score": 3.0})
        # 3rd write triggers flush.
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 3

    def test_flush_writes_remaining(self, tmp_path):
        path = tmp_path / "c.json"
        cache = EvalCache(path, flush_interval=10)
        cache.set("k1", {"score": 1.0})
        assert not path.exists()
        cache.flush()
        assert path.exists()
        data = json.loads(path.read_text())
        assert "k1" in data

    def test_flush_noop_when_clean(self, tmp_path):
        path = tmp_path / "c.json"
        cache = EvalCache(path, flush_interval=1)
        cache.set("k1", {"score": 1.0})
        mtime1 = path.stat().st_mtime
        cache.flush()  # nothing dirty — should be a no-op
        mtime2 = path.stat().st_mtime
        assert mtime1 == mtime2

    def test_default_flush_interval_is_10(self, tmp_path):
        cache = EvalCache(tmp_path / "c.json")
        assert cache._flush_interval == 10


class TestEvalCacheClear:
    def test_clear_removes_entries_and_file(self, tmp_path):
        path = tmp_path / "c.json"
        cache = EvalCache(path, flush_interval=1)
        cache.set("k1", {"score": 1.0})
        assert path.exists()
        cache.clear()
        assert len(cache) == 0
        assert not path.exists()


class TestEvalCacheThreadSafety:
    def test_concurrent_writes(self, tmp_path):
        cache = EvalCache(tmp_path / "c.json", flush_interval=1)
        errors = []

        def write(i):
            try:
                cache.set(f"k{i}", {"score": float(i)})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(cache) == 20


# ---------------------------------------------------------------------------
# NoCache
# ---------------------------------------------------------------------------


class TestNoCache:
    def test_get_always_none(self):
        assert NoCache().get("anything") is None

    def test_set_is_noop(self):
        nc = NoCache()
        nc.set("k", {"score": 1.0})
        assert nc.get("k") is None

    def test_len_always_zero(self):
        assert len(NoCache()) == 0

    def test_contains_always_false(self):
        assert "k" not in NoCache()

    def test_make_key_returns_empty(self):
        assert NoCache.make_key("m", "i") == ""


# ---------------------------------------------------------------------------
# Integration: caching in _evaluate_sequential
# ---------------------------------------------------------------------------


def _make_selector(cache, invoke_fn, eval_fn, dataset):
    """Build a minimal BruteForceModelSelector with a mock proxy."""
    from agentopt import BruteForceModelSelector, ModelProxy

    # Create a real proxy wrapping a dummy object.
    proxy = ModelProxy("dummy")

    selector = BruteForceModelSelector.__new__(BruteForceModelSelector)
    selector._models = {proxy: ["dummy-model"]}
    selector._proxies = [proxy]
    selector._cache = cache if cache is not None else NoCache()
    selector.eval_fn = eval_fn
    selector.invoke_fn = invoke_fn
    selector.is_async = False
    selector._token_tracker = None
    selector.dataset = dataset
    selector.agent = None
    selector._adapter = None
    selector._invoke_method_name = None
    return selector


class TestCachingInEvaluation:
    """Test that _evaluate_sequential reads/writes the cache correctly."""

    def test_first_run_populates_cache(self, tmp_path):
        """On first run, every item is a cache miss → invoke_fn is called."""
        cache = EvalCache(tmp_path / "c.json")
        dataset = [("What is 1+1?", "2"), ("Capital of France?", "Paris")]
        responses = iter(["2", "Paris"])
        call_count = 0

        def invoke_fn(inp):
            nonlocal call_count
            call_count += 1
            return next(responses)

        def eval_fn(expected, actual):
            return 1.0 if expected == actual else 0.0

        selector = _make_selector(cache, invoke_fn, eval_fn, dataset)
        scores, latencies, tokens = selector._evaluate_sequential(dataset, label="m1")

        assert call_count == 2
        assert scores == [1.0, 1.0]
        assert len(latencies) == 2
        assert len(cache) == 2

    def test_second_run_uses_cache(self, tmp_path):
        """On second run with same data, invoke_fn should NOT be called."""
        cache = EvalCache(tmp_path / "c.json")
        dataset = [("q1", "a1"), ("q2", "a2")]
        call_count = 0

        def invoke_fn(inp):
            nonlocal call_count
            call_count += 1
            return "a1" if inp == "q1" else "a2"

        def eval_fn(expected, actual):
            return 1.0 if expected == actual else 0.0

        selector = _make_selector(cache, invoke_fn, eval_fn, dataset)

        # First run — populates cache.
        selector._evaluate_sequential(dataset, label="m1")
        assert call_count == 2

        # Second run — should hit cache, no new calls.
        call_count = 0
        scores, latencies, _ = selector._evaluate_sequential(dataset, label="m1")
        assert call_count == 0
        assert scores == [1.0, 1.0]
        assert len(latencies) == 2

    def test_different_model_label_causes_cache_miss(self, tmp_path):
        """Different model label → different cache keys → invoke_fn called again."""
        cache = EvalCache(tmp_path / "c.json")
        dataset = [("q", "a")]
        call_count = 0

        def invoke_fn(inp):
            nonlocal call_count
            call_count += 1
            return "a"

        def eval_fn(expected, actual):
            return 1.0

        selector = _make_selector(cache, invoke_fn, eval_fn, dataset)

        selector._evaluate_sequential(dataset, label="model-A")
        assert call_count == 1

        # Same data, different model label.
        call_count = 0
        selector._evaluate_sequential(dataset, label="model-B")
        assert call_count == 1

        assert len(cache) == 2

    def test_new_dataset_items_reuse_existing_cache(self, tmp_path):
        """Adding new items doesn't invalidate cache for existing items."""
        cache = EvalCache(tmp_path / "c.json")
        call_count = 0

        def invoke_fn(inp):
            nonlocal call_count
            call_count += 1
            return inp.upper()

        def eval_fn(expected, actual):
            return 1.0 if expected == actual else 0.0

        selector_small = _make_selector(cache, invoke_fn, eval_fn, [("hello", "HELLO")])
        selector_small._evaluate_sequential([("hello", "HELLO")], label="m1")
        assert call_count == 1

        # Now evaluate with an expanded dataset.
        call_count = 0
        bigger = [("hello", "HELLO"), ("world", "WORLD")]
        selector_big = _make_selector(cache, invoke_fn, eval_fn, bigger)
        scores, _, _ = selector_big._evaluate_sequential(bigger, label="m1")

        # Only the new item should trigger a call.
        assert call_count == 1
        assert scores == [1.0, 1.0]

    def test_cache_persists_across_instances(self, tmp_path):
        """Cache file survives selector reconstruction."""
        path = tmp_path / "c.json"
        dataset = [("q", "a")]
        call_count = 0

        def invoke_fn(inp):
            nonlocal call_count
            call_count += 1
            return "a"

        def eval_fn(expected, actual):
            return 1.0

        # First selector populates cache.
        s1 = _make_selector(EvalCache(path), invoke_fn, eval_fn, dataset)
        s1._evaluate_sequential(dataset, label="m")
        assert call_count == 1

        # New selector, new EvalCache instance from same file.
        call_count = 0
        s2 = _make_selector(EvalCache(path), invoke_fn, eval_fn, dataset)
        scores, _, _ = s2._evaluate_sequential(dataset, label="m")
        assert call_count == 0
        assert scores == [1.0]

    def test_no_cache_always_calls(self, tmp_path):
        """Without cache (None), invoke_fn is always called."""
        dataset = [("q", "a")]
        call_count = 0

        def invoke_fn(inp):
            nonlocal call_count
            call_count += 1
            return "a"

        def eval_fn(expected, actual):
            return 1.0

        selector = _make_selector(None, invoke_fn, eval_fn, dataset)
        selector._cache = NoCache()
        selector._evaluate_sequential(dataset, label="m")
        assert call_count == 1

        selector._evaluate_sequential(dataset, label="m")
        assert call_count == 2  # called again

    def test_cached_scores_and_latencies_match(self, tmp_path):
        """Cached results should return the same scores and latencies."""
        cache = EvalCache(tmp_path / "c.json")
        dataset = [("q1", "a1"), ("q2", "wrong")]

        def invoke_fn(inp):
            return "a1"

        def eval_fn(expected, actual):
            return 1.0 if expected == actual else 0.0

        selector = _make_selector(cache, invoke_fn, eval_fn, dataset)

        scores1, lat1, _ = selector._evaluate_sequential(dataset, label="m")
        scores2, lat2, _ = selector._evaluate_sequential(dataset, label="m")

        assert scores1 == scores2
        # Latencies from cache should match the stored original latencies.
        assert lat1 == lat2
