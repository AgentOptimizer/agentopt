"""Durable observation journal for deterministic Gittins continuation.

Only JSON is read from disk (never pickle). The separate SQLite lock is held
for the entire run; the journal itself commits each successful cell separately.
"""

import hashlib
import json
import sqlite3
from pathlib import Path


class CheckpointError(ValueError):
    """A checkpoint problem that can be displayed without provider details."""


class CheckpointInUseError(RuntimeError):
    """Another process currently owns the checkpoint."""


def fingerprint(payload):
    try:
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint inputs must be JSON-serializable; use plain dataset and model specifications") from error
    return hashlib.sha256(encoded).hexdigest()


class EvaluationCheckpoint:
    def __init__(self, path, identity):
        self.path = Path(path)
        self.identity = identity
        self.db = self.lock = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.lock = sqlite3.connect(str(self.path) + ".lock", timeout=0, isolation_level=None)
            try:
                self.lock.execute("BEGIN EXCLUSIVE")
            except sqlite3.OperationalError as error:
                raise CheckpointInUseError("This checkpoint is already in use by another run") from error
            self.db = sqlite3.connect(self.path, isolation_level=None)
            self.db.execute("PRAGMA synchronous = FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, identity TEXT NOT NULL)")
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES (1, ?)", (self.identity,))
            if self.db.execute("SELECT identity FROM metadata WHERE id=1").fetchone()[0] != self.identity:
                raise CheckpointError("Checkpoint setup differs: check dataset, models, seed, workflow and algorithm settings; use a new checkpoint path for a new run")
            self.db.execute("CREATE TABLE IF NOT EXISTS batches (id INTEGER PRIMARY KEY, arm TEXT NOT NULL, questions TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS cells (batch INTEGER NOT NULL, question INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY (batch, question))")
            self.saved_cells = self.db.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
            self.saved_batches = self.db.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
            self.batch_index = 0
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.db is not None:
            self.db.close()
        if self.lock is not None:
            self.lock.close()

    def begin_batch(self, arm, questions):
        self.batch_index += 1
        encoded = json.dumps(questions)
        row = self.db.execute("SELECT arm, questions FROM batches WHERE id=?", (self.batch_index,)).fetchone()
        if row is None:
            self.db.execute("INSERT INTO batches VALUES (?, ?, ?)", (self.batch_index, arm, encoded))
        elif row != (arm, encoded):
            raise CheckpointError("Checkpoint evaluation order differs; use a new checkpoint path")
        return {question: json.loads(data) for question, data in self.db.execute(
            "SELECT question, data FROM cells WHERE batch=?", (self.batch_index,))}

    def save_cell(self, question, data):
        self.db.execute("INSERT INTO cells VALUES (?, ?, ?)",
                        (self.batch_index, question, json.dumps(data, allow_nan=False)))
