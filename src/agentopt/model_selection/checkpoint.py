"""Durable observation journal for deterministic Gittins continuation.

Only JSON is read from disk (never pickle). The separate SQLite lock is held
for the entire run; the journal itself commits each successful cell separately.
"""

import hashlib
import json
import math
from contextlib import contextmanager

from .base import DatapointResult
import sqlite3
from pathlib import Path


class CheckpointError(ValueError):
    """A checkpoint problem that can be displayed without provider details."""


class CheckpointInUseError(RuntimeError):
    """Another process currently owns the checkpoint."""


def checkpoint_paths(path):
    database = Path(path).resolve()
    lock = Path(str(database) + ".lock")
    paths = {"checkpoint": database, "checkpoint lock": lock}
    for role, base in list(paths.items()):
        for suffix in ("-journal", "-wal", "-shm"):
            paths[role + suffix] = Path(str(base) + suffix)
    return paths


def require_distinct_paths(paths):
    seen = []
    for role, path in paths.items():
        for other_role, other in seen:
            if path.resolve() == other.resolve() or (path.exists() and other.exists() and path.samefile(other)):
                raise ValueError(f"Path conflict: {role} and {other_role} must refer to different files")
        seen.append((role, path))


def fingerprint(payload):
    try:
        def typed(value):
            if type(value) is tuple:
                return ["tuple", [typed(x) for x in value]]
            if type(value) is list:
                return ["list", [typed(x) for x in value]]
            if type(value) is dict:
                return ["dict", [[typed(k), typed(v)] for k, v in value.items()]]
            if type(value) not in (str, int, float, bool, type(None)):
                raise TypeError("unsupported checkpoint value")
            return [type(value).__name__, value]
        encoded = json.dumps(typed(payload), sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint inputs must be JSON-serializable; use plain dataset and model specifications") from error
    return hashlib.sha256(encoded).hexdigest()


class EvaluationCheckpoint:
    def __init__(self, path, identity, *, legacy_identity=None):
        self.paths = checkpoint_paths(path)
        self.path = self.paths["checkpoint"]
        self.identity = identity
        self.legacy_identity = legacy_identity
        self._needs_normalization = False
        self.db = self.lock = None

    def __enter__(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_nlink > 1:
                raise CheckpointError("Checkpoint hard links are not supported; use a single-link file")
            self.lock = sqlite3.connect(self.paths["checkpoint lock"], timeout=0, isolation_level=None)
            try:
                self.lock.execute("BEGIN EXCLUSIVE")
            except sqlite3.OperationalError as error:
                raise CheckpointInUseError("This checkpoint is already in use by another run") from error
            existed = self.path.exists()
            self.db = sqlite3.connect(self.path, isolation_level=None)
            if self.path.stat().st_nlink > 1:
                raise CheckpointError("Checkpoint hard links are not supported")
            if not existed:
                self.db.execute("BEGIN IMMEDIATE")
                try:
                    self.db.execute("CREATE TABLE metadata (id INTEGER PRIMARY KEY, identity TEXT NOT NULL)")
                    self.db.execute("INSERT INTO metadata VALUES (1, ?)", (self.identity,))
                    self.db.execute("CREATE TABLE batches (id INTEGER PRIMARY KEY, arm TEXT NOT NULL, questions TEXT NOT NULL)")
                    self.db.execute("CREATE TABLE cells (batch INTEGER NOT NULL, question INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY (batch, question))")
                    self.db.execute("PRAGMA user_version = 3")
                    self.db.execute("COMMIT")
                except BaseException:
                    self.db.execute("ROLLBACK")
                    raise
            tables = {row[0] for row in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"metadata", "batches", "cells"} <= tables or self.db.execute("PRAGMA user_version").fetchone()[0] != 3:
                raise CheckpointError(f"Invalid or legacy checkpoint schema in '{self.path}'; original retained. Use a new path.")
            identity = self.db.execute("SELECT identity FROM metadata WHERE id=1").fetchone()
            if identity is None:
                raise CheckpointError(f"Missing checkpoint identity in '{self.path}'; original retained")
            self._needs_normalization = identity[0] != self.identity and identity[0] == self.legacy_identity
            if identity[0] != self.identity and not self._needs_normalization:
                raise CheckpointError("Checkpoint setup differs: check dataset, models, seed, workflow and algorithm settings; use a new checkpoint path for a new run")
            self.db.execute("PRAGMA synchronous = FULL")
            self.saved_cells = self.db.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
            self.saved_batches = self.db.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
            self.batch_index = 0
            return self
        except BaseException as error:
            self.__exit__(None, None, None)
            if isinstance(error, (sqlite3.Error, OSError)):
                raise CheckpointError(f"Cannot open/read checkpoint '{self.path}'. Check the file and permissions; the checkpoint has not been replaced.") from error
            raise

    def _normalize_identity(self):
        # Called after static validation and deterministic replay of all saved manifests.
        with self._operation("update"):
            self.db.execute("UPDATE metadata SET identity=? WHERE id=1", (self.identity,))
        self._needs_normalization = False

    def __exit__(self, *_):
        if self.db is not None:
            self.db.close()
        if self.lock is not None:
            self.lock.close()

    @contextmanager
    def _operation(self, action):
        try:
            yield
        except (sqlite3.Error, OSError) as error:
            raise CheckpointError(
                f"Cannot {action} checkpoint '{self.path}'. Existing saved cells are retained; "
                "in-flight work not committed to disk may need to run again."
            ) from error

    def validate(self, *, names, questions, score_bounds, prices):
        """Check the entire journal before any new evaluation is dispatched."""
        from ..model_price import compute_price
        with self._operation("read"):
            batches = self.db.execute("SELECT id, arm, questions FROM batches ORDER BY id").fetchall()
            rows = self.db.execute("SELECT batch, question, data FROM cells").fetchall()
        try:
            manifests = {}
            for expected, (batch, arm, encoded) in enumerate(batches, 1):
                ids = json.loads(encoded)
                if batch != expected or arm not in names or not isinstance(ids, list) or not ids:
                    raise ValueError("invalid batch identity")
                if any(type(q) is not int or not 0 <= q < questions for q in ids) or len(set(ids)) != len(ids):
                    raise ValueError("invalid batch question indices")
                manifests[batch] = (arm, ids)
            seen, counts = set(), {}
            for batch, question, encoded in rows:
                arm, ids = manifests[batch]
                if type(question) is not int or question not in ids or (arm, question) in seen:
                    raise ValueError("invalid or duplicate cell identity")
                seen.add((arm, question))
                counts[batch] = counts.get(batch, 0) + 1
                item = json.loads(encoded)
                raw = item["datapoint"]
                dp = DatapointResult.model_validate(raw)
                if type(raw["datapoint_index"]) is not int or dp.datapoint_index != question:
                    raise ValueError("cell index does not match batch")
                for value in (item["cost"], item["charge"], raw["score"], raw["latency_seconds"]):
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError("non-finite observation")
                if item["cost"] < 0 or item["charge"] < 0 or dp.latency_seconds < 0 or not score_bounds[0] <= dp.score <= score_bounds[1]:
                    raise ValueError("observation outside valid bounds")
                if not dp.input_tokens and not dp.output_tokens:
                    raise ValueError("missing token usage")
                for field in ("input_tokens", "output_tokens"):
                    if any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in raw[field].items()):
                        raise ValueError("invalid token usage")
                cost = compute_price(dp.input_tokens, dp.output_tokens, custom_prices=prices)
                if cost is None or not math.isclose(item["cost"], cost, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError("cost does not match token usage")
                if item["charge"] > item["cost"] + 1e-12:
                    raise ValueError("charge exceeds recorded token cost")
            if any(counts.get(batch, 0) != len(ids) for batch, (_, ids) in manifests.items() if batch < len(batches)):
                raise ValueError("incomplete non-terminal batch")
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise CheckpointError(f"Invalid checkpoint contents in '{self.path}': {error}. The original file is retained; use a valid checkpoint or a new path.") from error

    def begin_batch(self, arm, questions):
        with self._operation("read/write"):
            self.batch_index += 1
            encoded = json.dumps(questions)
            row = self.db.execute("SELECT arm, questions FROM batches WHERE id=?", (self.batch_index,)).fetchone()
            if row is not None and row != (arm, encoded):
                raise CheckpointError(f"Checkpoint evaluation order differs in '{self.path}'; use a new checkpoint path")
            if self._needs_normalization and self.batch_index >= self.saved_batches:
                self._normalize_identity()
            if row is None:
                self.db.execute("INSERT INTO batches VALUES (?, ?, ?)", (self.batch_index, arm, encoded))
            return {question: json.loads(data) for question, data in self.db.execute(
                "SELECT question, data FROM cells WHERE batch=?", (self.batch_index,))}

    def save_cell(self, question, data):
        with self._operation("write"):
            self.db.execute("INSERT INTO cells VALUES (?, ?, ?)",
                            (self.batch_index, question, json.dumps(data, allow_nan=False)))
