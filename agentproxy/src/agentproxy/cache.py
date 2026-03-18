"""API-level response cache for LLM calls.

Caches responses keyed by a hash of the request body (model + messages +
other parameters). On a cache hit the original response bytes are returned
directly, skipping the network round-trip.

Enabled by default; can be disabled via ``LLMTracker(cache=False)``.

When ``cache_dir`` is provided, entries are loaded from disk on init and
flushed back periodically (and on ``close()``) so the cache survives
process restarts without blocking the hot path with synchronous I/O.
"""

import base64
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Union

logger = logging.getLogger(__name__)

# Default interval (seconds) between automatic background flushes.
_DEFAULT_FLUSH_INTERVAL = 60.0


def _make_cache_key(request_body: dict) -> str:
    """Deterministic hash of the request payload.

    Includes every field that affects the response (model, messages,
    temperature, etc.) but excludes ephemeral metadata like ``stream``.
    """
    body = {k: v for k, v in request_body.items() if k not in ("stream",)}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


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
    """Thread-safe in-memory response cache with lazy disk persistence.

    Every entry is kept in memory until explicitly cleared.
    When ``cache_dir`` is set, entries are loaded from disk on init and
    dirty entries are flushed to disk:

    * Periodically by a background daemon thread (every ``flush_interval``
      seconds, default 60).
    * Explicitly via :meth:`flush`.
    * Automatically when :meth:`close` is called (which ``LLMTracker.stop``
      calls for you).

    This keeps ``put`` and ``get`` free of disk I/O.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory for persisting cache entries as JSON files.
        If ``None`` (default), the cache is in-memory only.
    flush_interval : float, optional
        Seconds between automatic background flushes (default 60).
        Set to 0 to disable the background flush thread (manual /
        close-only flushing).
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        flush_interval: float = _DEFAULT_FLUSH_INTERVAL,
    ) -> None:
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None

        # Dirty tracking — keys that need to be written on next flush
        self._dirty: Set[str] = set()

        # Background flush thread
        self._flush_interval = flush_interval
        self._flush_stop = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None

        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
            if flush_interval > 0:
                self._start_flush_thread()

    # ------------------------------------------------------------------
    # Background flush thread
    # ------------------------------------------------------------------

    def _start_flush_thread(self) -> None:
        self._flush_stop.clear()
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="agentproxy-cache-flush",
        )
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        while not self._flush_stop.wait(timeout=self._flush_interval):
            self.flush()

    # ------------------------------------------------------------------
    # Disk persistence helpers
    # ------------------------------------------------------------------

    def _entry_path(self, key: str) -> Path:
        assert self._cache_dir is not None
        return self._cache_dir / f"{key}.json"

    def _write_entry(self, key: str, entry: CacheEntry) -> None:
        try:
            path = self._entry_path(key)
            path.write_text(json.dumps(entry.to_dict(), ensure_ascii=True))
        except OSError as exc:
            logger.warning("Failed to write cache entry %s: %s", key, exc)

    def _remove_entry_file(self, key: str) -> None:
        try:
            self._entry_path(key).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to delete cache entry %s: %s", key, exc)

    def _load_from_disk(self) -> None:
        assert self._cache_dir is not None
        loaded = 0
        for path in self._cache_dir.glob("*.json"):
            key = path.stem
            try:
                data = json.loads(path.read_text())
                self._store[key] = CacheEntry.from_dict(data)
                loaded += 1
            except (json.JSONDecodeError, KeyError, OSError) as exc:
                logger.warning("Skipping corrupt cache file %s: %s", path, exc)
        if loaded:
            logger.debug("Loaded %d cache entries from %s", loaded, self._cache_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[CacheEntry]:
        """Look up a cached response. Returns ``None`` on miss."""
        with self._lock:
            return self._store.get(key)

    def put(self, key: str, entry: CacheEntry) -> None:
        """Store a response in the cache. Disk write is deferred to flush."""
        with self._lock:
            self._store[key] = entry
            if self._cache_dir is not None:
                self._dirty.add(key)

    def flush(self) -> None:
        """Write all dirty entries to disk."""
        if self._cache_dir is None:
            return

        with self._lock:
            to_write = {k: self._store[k] for k in self._dirty if k in self._store}
            self._dirty.clear()

        for key, entry in to_write.items():
            self._write_entry(key, entry)

        if to_write:
            logger.debug("Flushed %d cache entries to disk", len(to_write))

    def clear(self) -> None:
        """Remove all cached entries and delete disk files."""
        with self._lock:
            keys_to_delete = set(self._store.keys())
            self._store.clear()
            self._dirty.clear()

        if self._cache_dir is not None:
            for key in keys_to_delete:
                self._remove_entry_file(key)
            for path in self._cache_dir.glob("*.json"):
                try:
                    path.unlink()
                except OSError:
                    pass

    def close(self) -> None:
        """Flush pending writes and stop the background thread."""
        if self._flush_thread is not None:
            self._flush_stop.set()
            self._flush_thread.join(timeout=5.0)
            self._flush_thread = None
        self.flush()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
