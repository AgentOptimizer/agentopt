"""EvalCache — file-backed cache for per-item evaluation results."""

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class EvalCache:
    """Thread-safe, file-backed cache for per-item evaluation results.

    Each entry stores the latency, response, and token counts for a
    ``(model, input_data)`` pair.  On re-runs, cached items skip LLM calls
    entirely, making iterative benchmarking fast and cheap.  Scores are
    recomputed from the cached response so changing ``eval_fn`` does not
    require clearing the cache.

    Args:
        cache_path: Path to the JSON file.  Created (with parent dirs) on
            first write.  An existing file is loaded on construction.
        flush_interval: Number of new entries to buffer before writing to
            disk.  Defaults to ``10``.  Set to ``1`` for per-item saves
            (safest against crashes) or higher for less I/O.
        save_score: If ``True``, persist the evaluated score in the cache
            entry.  Defaults to ``False`` — the score is recomputed from
            the cached response on retrieval.  Set to ``True`` when the
            evaluation function is expensive (e.g. LLM-as-judge).
    """

    def __init__(
        self,
        cache_path: Union[str, Path] = ".cache/eval_cache.json",
        flush_interval: int = 10,
        save_score: bool = False,
    ) -> None:
        self._path = Path(cache_path)
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._flush_interval = max(1, flush_interval)
        self._dirty: int = 0  # unsaved writes since last flush
        self.save_score = save_score
        self._load()

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(model_name: str, input_data: Any) -> str:
        """Return a deterministic hex key for a ``(model, input)`` pair.

        Uses the full SHA-256 hex digest (64 characters / 256 bits) to
        ensure effectively zero collision probability at any scale.

        The expected answer is intentionally excluded from the key so that
        changing the expected answer or the evaluation function does not
        invalidate cached LLM responses.
        """
        raw = f"{model_name}::{input_data}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return the cached entry for *key*, or ``None`` on miss."""
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, entry: Dict[str, Any]) -> None:
        """Store *entry* under *key*.

        Writes to disk every *flush_interval* entries.  Call :meth:`flush`
        to force an immediate write (e.g. at the end of an evaluation loop).
        """
        with self._lock:
            self._data[key] = entry
            self._dirty += 1
            if self._dirty >= self._flush_interval:
                self._save()
                self._dirty = 0

    def flush(self) -> None:
        """Write any buffered entries to disk immediately."""
        with self._lock:
            if self._dirty > 0:
                self._save()
                self._dirty = 0

    def clear(self) -> None:
        """Remove all entries and delete the backing file."""
        with self._lock:
            self._data.clear()
            if self._path.exists():
                self._path.unlink()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            text = self._path.read_text(encoding="utf-8")
            self._data = json.loads(text) if text.strip() else {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Corrupt or unreadable cache file %s — starting fresh: %s",
                self._path,
                exc,
            )
            self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class NoCache:
    """Drop-in replacement that never caches.

    All methods match :class:`EvalCache` signatures so callers need no
    ``if`` guards.
    """

    @staticmethod
    def make_key(model_name: str, input_data: Any) -> str:  # noqa: ARG004
        return ""

    def get(self, key: str) -> None:  # noqa: ARG002
        return None

    def set(self, key: str, entry: Dict[str, Any]) -> None:  # noqa: ARG002
        pass

    def flush(self) -> None:
        pass

    def clear(self) -> None:
        pass

    def __len__(self) -> int:
        return 0

    def __contains__(self, key: str) -> bool:  # noqa: ARG002
        return False
