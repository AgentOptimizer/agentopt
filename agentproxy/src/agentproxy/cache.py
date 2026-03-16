"""API-level response cache for LLM calls.

Caches responses keyed by a hash of the request body (model + messages +
other parameters). On a cache hit the original response bytes are returned
directly, skipping the network round-trip.

Enabled by default; can be disabled via ``LLMTracker(cache=False)`` or at
runtime with ``tracker.cache_enabled = False``.
"""

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional


def _make_cache_key(request_body: dict) -> str:
    """Deterministic hash of the request payload.

    We include every field that affects the response (model, messages,
    temperature, etc.) but exclude ephemeral metadata like ``stream``.
    Callers should already have ensured ``stream`` is False before
    reaching the cache.
    """
    # Copy and remove fields that should not affect cache identity
    body = {k: v for k, v in request_body.items() if k not in ("stream",)}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class CacheStats:
    """Running counters for cache performance."""

    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0


@dataclass
class CacheEntry:
    """A cached LLM API response."""

    response_bytes: bytes
    response_headers: Dict[str, str] = field(default_factory=dict)


class ResponseCache:
    """Thread-safe in-memory LRU-ish response cache.

    Parameters
    ----------
    max_size : int, optional
        Maximum number of entries. 0 means unlimited (default).
    """

    def __init__(self, max_size: int = 0) -> None:
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[CacheEntry]:
        """Look up a cached response. Returns ``None`` on miss."""
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                self.stats.hits += 1
            else:
                self.stats.misses += 1
            return entry

    def put(self, key: str, entry: CacheEntry) -> None:
        """Store a response in the cache."""
        with self._lock:
            if self._max_size > 0 and len(self._store) >= self._max_size:
                # Evict oldest entry (first inserted)
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
            self._store[key] = entry

    def clear(self) -> None:
        """Remove all cached entries and reset stats."""
        with self._lock:
            self._store.clear()
            self.stats = CacheStats()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
