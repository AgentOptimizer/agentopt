"""LLMTracker — context management and record storage."""

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .cache import CacheStats, ResponseCache
from .interceptor import (
    _agent_id_var,
    _combo_id_var,
    _data_id_var,
    install,
    uninstall,
)
from .models import CallRecord


class LLMTracker:
    """Tracks LLM API calls via httpx interception.

    Parameters
    ----------
    cache : bool
        Enable API-level response caching (default ``True``).
        When enabled, identical requests (same model, messages, etc.)
        return cached responses instantly without hitting the API.
    cache_dir : str or Path, optional
        Directory for persisting cache entries as JSON files on disk.
        When set, the cache is loaded from disk on init and flushed
        periodically and on ``stop()``. Enables cache survival across
        process restarts. If ``None`` (default), the cache is
        in-memory only.

    Usage::

        tracker = LLMTracker()           # cache on by default
        tracker = LLMTracker(cache=False) # disable caching
        tracker = LLMTracker(cache_dir="./llm_cache")  # persist to disk
        tracker.start()

        with tracker.track(data_id="dp_1", combo_id="gpt4o+haiku"):
            result = agent(input_data)

        usage = tracker.get_usage(combo_id="gpt4o+haiku")
        print(tracker.cache_stats)       # CacheStats(hits=3, misses=2)
        tracker.stop()
    """

    def __init__(
        self,
        cache: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self._records: List[CallRecord] = []
        self._lock = threading.Lock()
        self._active = False
        self._cache_on = cache
        self._response_cache = (
            ResponseCache(cache_dir=cache_dir) if cache else None
        )

    @property
    def cache_enabled(self) -> bool:
        """Whether response caching is currently active."""
        return self._cache_on

    @cache_enabled.setter
    def cache_enabled(self, value: bool) -> None:
        """Enable or disable caching at runtime."""
        import agentproxy.interceptor as _int

        # If no cache was initialized (e.g. constructed with cache=False),
        # enabling caching would silently not work because the interceptor
        # has no cache instance to use. Fail fast instead of misrepresenting
        # the state.
        if value and self._response_cache is None:
            raise RuntimeError(
                "Cannot enable caching: LLMTracker was constructed with "
                "cache=False and no ResponseCache was initialized. "
                "Create the tracker with cache=True to use caching."
            )

        self._cache_on = value
        _int._cache_enabled = value

    @property
    def cache_stats(self) -> CacheStats:
        """Return cache hit/miss statistics."""
        if self._response_cache is not None:
            return self._response_cache.stats
        return CacheStats()

    def clear_cache(self) -> None:
        """Clear all cached responses and reset stats."""
        if self._response_cache is not None:
            self._response_cache.clear()

    def start(self) -> None:
        """Install httpx patches. Idempotent."""
        if self._active:
            return
        install(
            callback=self._on_call,
            cache=self._response_cache,
            cache_enabled=self._cache_on,
        )
        self._active = True

    def flush_cache(self) -> None:
        """Flush dirty cache entries to disk immediately."""
        if self._response_cache is not None:
            self._response_cache.flush()

    def stop(self) -> None:
        """Remove httpx patches and flush cache to disk. Idempotent."""
        if not self._active:
            return
        uninstall()
        if self._response_cache is not None:
            self._response_cache.close()
        self._active = False

    def _on_call(self, record: CallRecord) -> None:
        """Callback from interceptor — appends record thread-safely."""
        with self._lock:
            self._records.append(record)

    @contextmanager
    def track(self, data_id: str, combo_id: str, agent_id: Optional[str] = None):
        """Set context vars for attribution during this scope."""
        t1 = _data_id_var.set(data_id)
        t2 = _combo_id_var.set(combo_id)
        t3 = _agent_id_var.set(agent_id)
        try:
            yield
        finally:
            _data_id_var.reset(t1)
            _combo_id_var.reset(t2)
            _agent_id_var.reset(t3)

    @contextmanager
    def track_agent(self, agent_id: str):
        """Set only agent_id, keeping data_id and combo_id unchanged."""
        t = _agent_id_var.set(agent_id)
        try:
            yield
        finally:
            _agent_id_var.reset(t)

    def get_records(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[CallRecord]:
        """Return CallRecord list, filtered by any combination of IDs."""
        with self._lock:
            records = list(self._records)

        result = []
        for r in records:
            if data_id is not None and r.data_id != data_id:
                continue
            if combo_id is not None and r.combo_id != combo_id:
                continue
            if agent_id is not None and r.agent_id != agent_id:
                continue
            result.append(r)
        return result

    def get_usage(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Tuple[int, int]]:
        """Return aggregated ``{model: (input_tokens, output_tokens)}``.

        Filtered by any combination of IDs.
        """
        records = self.get_records(
            data_id=data_id, combo_id=combo_id, agent_id=agent_id
        )
        totals: Dict[str, List[int]] = {}
        for r in records:
            if r.model not in totals:
                totals[r.model] = [0, 0]
            totals[r.model][0] += r.prompt_tokens
            totals[r.model][1] += r.completion_tokens
        return {k: (v[0], v[1]) for k, v in totals.items()}

    def get_cached_latency(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> float:
        """Return total latency (seconds) from cached responses.

        Sums ``latency_seconds`` for all ``cached=True`` records matching
        the given filters.  This represents the time that *would* have been
        spent on real API calls but was saved by the cache.
        """
        records = self.get_records(
            data_id=data_id, combo_id=combo_id, agent_id=agent_id
        )
        return sum(r.latency_seconds for r in records if r.cached)

    def clear(self) -> None:
        """Clear all recorded data."""
        with self._lock:
            self._records.clear()
