"""API-level response cache for LLM calls.

Caches responses keyed by a hash of the request body (model + messages +
other parameters). On a cache hit the original response bytes are returned
directly, skipping the network round-trip.

Enabled by default; can be disabled via ``LLMTracker(cache=False)`` or at
runtime with ``tracker.cache_enabled = False``.

When ``cache_dir`` is provided, entries are persisted as JSON files on disk
so the cache survives process restarts.
"""

import base64
import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


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
    latency_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "response_bytes_b64": base64.b64encode(self.response_bytes).decode("ascii"),
            "response_headers": self.response_headers,
            "latency_seconds": self.latency_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        """Deserialize from a JSON-compatible dict."""
        return cls(
            response_bytes=base64.b64decode(data["response_bytes_b64"]),
            response_headers=data.get("response_headers", {}),
            latency_seconds=data.get("latency_seconds", 0.0),
        )


class ResponseCache:
    """Thread-safe in-memory LRU response cache with optional disk persistence.

    Parameters
    ----------
    max_size : int, optional
        Maximum number of entries. 0 means unlimited (default).
    cache_dir : str or Path, optional
        Directory for persisting cache entries as JSON files.
        When set, entries are loaded from disk on init and written
        through on every ``put``. If ``None`` (default), the cache
        is in-memory only.
    """

    def __init__(
        self,
        max_size: int = 0,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size
        self.stats = CacheStats()
        self._cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None

        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Disk persistence helpers
    # ------------------------------------------------------------------

    def _entry_path(self, key: str) -> Path:
        """Return the file path for a cache key."""
        assert self._cache_dir is not None
        return self._cache_dir / f"{key}.json"

    def _save_entry(self, key: str, entry: CacheEntry) -> None:
        """Write a single entry to disk (called under lock)."""
        if self._cache_dir is None:
            return
        try:
            path = self._entry_path(key)
            path.write_text(json.dumps(entry.to_dict(), ensure_ascii=True))
        except OSError as exc:
            logger.warning("Failed to write cache entry %s: %s", key, exc)

    def _delete_entry(self, key: str) -> None:
        """Remove a single entry file from disk (called under lock)."""
        if self._cache_dir is None:
            return
        try:
            self._entry_path(key).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to delete cache entry %s: %s", key, exc)

    def _load_from_disk(self) -> None:
        """Load all cached entries from ``cache_dir`` into memory."""
        assert self._cache_dir is not None
        loaded = 0
        for path in sorted(self._cache_dir.glob("*.json")):
            key = path.stem
            try:
                data = json.loads(path.read_text())
                entry = CacheEntry.from_dict(data)
                self._store[key] = entry
                loaded += 1
            except (json.JSONDecodeError, KeyError, OSError) as exc:
                logger.warning("Skipping corrupt cache file %s: %s", path, exc)
        # Enforce max_size: keep only the most recent entries
        if self._max_size > 0:
            while len(self._store) > self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                self._delete_entry(evicted_key)
        if loaded:
            logger.debug("Loaded %d cache entries from %s", loaded, self._cache_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[CacheEntry]:
        """Look up a cached response. Returns ``None`` on miss."""
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                # Promote to most recently used on access
                self._store.move_to_end(key)
                self.stats.hits += 1
            else:
                self.stats.misses += 1
            return entry

    def put(self, key: str, entry: CacheEntry) -> None:
        """Store a response in the cache (and on disk if configured)."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = entry
            else:
                if self._max_size > 0 and len(self._store) >= self._max_size:
                    evicted_key, _ = self._store.popitem(last=False)
                    self._delete_entry(evicted_key)
                self._store[key] = entry
            self._save_entry(key, entry)

    def clear(self) -> None:
        """Remove all cached entries and reset stats."""
        with self._lock:
            if self._cache_dir is not None:
                for key in list(self._store):
                    self._delete_entry(key)
            self._store.clear()
            self.stats = CacheStats()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
