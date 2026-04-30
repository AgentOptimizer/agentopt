"""Small BIRD benchmark helpers for NL2SQL examples.

The full BIRD dev dataset is large, so AgentOpt keeps setup/download logic in
``benchmarks/bird`` and exposes lightweight loaders here.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_BIRD_DATA_DIR = Path("benchmarks/bird/data")
DEFAULT_SQL_TIMEOUT_SECONDS = 30.0
SQL_PROGRESS_HANDLER_STEP_COUNT = 10_000
NUMERIC_RELATIVE_EPSILON = 1e-4
NUMERIC_ABSOLUTE_EPSILON = 1e-9


@dataclass(frozen=True)
class BirdPaths:
    """Resolved paths for a local BIRD dev dataset checkout."""

    root: Path
    questions: Path
    tables: Path
    databases: Path

    @classmethod
    def from_data_dir(cls, data_dir: str | Path | None = None) -> "BirdPaths":
        root = resolve_bird_data_dir(data_dir)
        return cls(
            root=root,
            questions=root / "dev.json",
            tables=root / "dev_tables.json",
            databases=root / "dev_databases",
        )

    def validate(self) -> None:
        missing = [
            path
            for path in (self.questions, self.tables, self.databases)
            if not path.exists()
        ]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                "BIRD data is not set up. Missing: "
                f"{missing_text}. Run `uv run python benchmarks/bird/setup_bird.py` "
                "or pass --data-dir to an existing BIRD dev dataset."
            )


@dataclass(frozen=True)
class BirdExample:
    """One BIRD NL2SQL example."""

    question_id: int
    db_id: str
    question: str
    gold_sql: str
    evidence: str = ""
    difficulty: str = ""
    gold_sql_tables: tuple[str, ...] = ()
    gold_sql_columns: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, row: dict[str, Any], fallback_question_id: int) -> "BirdExample":
        return cls(
            question_id=int(row.get("question_id", fallback_question_id)),
            db_id=str(row["db_id"]),
            question=str(row["question"]),
            gold_sql=str(row.get("SQL") or row.get("query") or ""),
            evidence=str(row.get("evidence") or ""),
            difficulty=str(row.get("difficulty") or ""),
            gold_sql_tables=tuple(
                str(value) for value in row.get("gold_sql_tables", [])
            ),
            gold_sql_columns=tuple(
                str(value) for value in row.get("gold_sql_columns", [])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SqlExecution:
    """Result of executing a SQLite query."""

    rows: list[tuple[Any, ...]] | None
    error: str | None
    elapsed_seconds: float


def resolve_bird_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve common BIRD dev dataset layouts to the directory with dev.json."""

    root = Path(data_dir or DEFAULT_BIRD_DATA_DIR).expanduser()
    candidates = [
        root,
        root / "data",
        root / "dev_20240627",
        root / "bird" / "data",
    ]
    for candidate in candidates:
        if (
            (candidate / "dev.json").exists()
            and (candidate / "dev_tables.json").exists()
        ):
            return candidate
    return root


def load_bird_examples(
    data_dir: str | Path | None = None,
    *,
    limit: int | None = None,
    question_ids: Iterable[int] | None = None,
    db_ids: Iterable[str] | None = None,
) -> list[BirdExample]:
    """Load BIRD dev examples from a local dataset directory."""

    paths = BirdPaths.from_data_dir(data_dir)
    paths.validate()
    raw_rows = json.loads(paths.questions.read_text(encoding="utf-8"))
    selected_question_ids = (
        {int(question_id) for question_id in question_ids}
        if question_ids is not None
        else None
    )
    selected_db_ids = {str(db_id) for db_id in db_ids} if db_ids is not None else None

    examples: list[BirdExample] = []
    for idx, row in enumerate(raw_rows):
        example = BirdExample.from_json(row, fallback_question_id=idx)
        if (
            selected_question_ids is not None
            and example.question_id not in selected_question_ids
        ):
            continue
        if selected_db_ids is not None and example.db_id not in selected_db_ids:
            continue
        examples.append(example)
        if limit is not None and len(examples) >= limit:
            break
    return examples


def load_bird_table_metadata(data_dir: str | Path | None = None) -> dict[str, Any]:
    """Load BIRD table metadata keyed by database id."""

    paths = BirdPaths.from_data_dir(data_dir)
    paths.validate()
    rows = json.loads(paths.tables.read_text(encoding="utf-8"))
    return {str(row["db_id"]): row for row in rows}


def bird_db_path(data_dir: str | Path | None, db_id: str) -> Path:
    """Return the SQLite path for a BIRD database id."""

    paths = BirdPaths.from_data_dir(data_dir)
    paths.validate()
    db_path = paths.databases / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing SQLite database for {db_id!r}: {db_path}")
    return db_path


def read_sqlite_schema(db_path: str | Path, *, max_chars: int | None = None) -> str:
    """Read CREATE statements from a SQLite database in read-only mode."""

    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database file not found: {path}")

    statements: list[str] = []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            """
            SELECT type, name, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
    for _, name, sql in rows:
        if sql:
            statement = str(sql).strip().rstrip(";") + ";"
        else:
            statement = f"-- Could not read schema for {name}"
        statements.append(statement)

    schema = "\n\n".join(statements)
    if max_chars is not None and max_chars > 0 and len(schema) > max_chars:
        return schema[:max_chars].rstrip() + "\n-- schema truncated"
    return schema


def build_bird_prompt(
    example: BirdExample,
    schema: str,
    *,
    include_evidence: bool = True,
) -> str:
    """Build a compact SQLite NL2SQL prompt for one BIRD example."""

    evidence_block = ""
    if include_evidence and example.evidence:
        evidence_block = f"\nHints:\n{example.evidence}\n"

    return (
        "You are a SQLite SQL expert. Write one SQLite query that answers the "
        "question using only the provided database schema.\n\n"
        "Rules:\n"
        "- Return only the SQL query, with no markdown fences or explanation.\n"
        "- The query must be a single read-only SELECT or WITH statement.\n"
        "- Quote table or column names when needed for SQLite.\n\n"
        f"Database: {example.db_id}\n\n"
        "Schema:\n"
        f"{schema}\n"
        f"{evidence_block}\n"
        "Question:\n"
        f"{example.question}\n"
    )


def extract_sql(text: Any) -> str:
    """Extract SQL from common LLM response formats."""

    sql = str(text or "").strip()
    if "</think>" in sql:
        sql = sql.split("</think>")[-1].strip()

    fence_pattern = re.compile(r"```(?P<lang>[^\n`]*)\n?(?P<body>.*?)```", re.DOTALL)
    blocks = list(fence_pattern.finditer(sql))
    if blocks:
        preferred = None
        for block in blocks:
            lang = block.group("lang").strip().lower()
            if not lang or "sql" in lang or lang in {"sqlite", "duckdb"}:
                preferred = block.group("body").strip()
                break
        sql = preferred if preferred is not None else blocks[0].group("body").strip()

    first_line, _, rest = sql.partition("\n")
    if first_line.strip().lower() in {"sql", "sqlite"} and rest.strip():
        sql = rest.strip()

    return sql.strip().rstrip(";") + ";" if sql.strip() else ""


def is_read_only_sql(sql: str) -> bool:
    stripped = str(sql or "").strip().lstrip("(").upper()
    return stripped.startswith("SELECT") or stripped.startswith("WITH")


def execute_sql(
    db_path: str | Path,
    sql: str,
    *,
    timeout_seconds: float | None = DEFAULT_SQL_TIMEOUT_SECONDS,
) -> SqlExecution:
    """Execute a read-only SQLite query with a soft timeout."""

    start = time.monotonic()
    path = Path(db_path)
    if not path.exists():
        return SqlExecution(None, f"Database file not found: {path}", 0.0)
    if not is_read_only_sql(sql):
        return SqlExecution(
            None,
            "Refusing to execute non-read-only SQL. Expected SELECT or WITH.",
            0.0,
        )

    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            if timeout_seconds and timeout_seconds > 0:

                def progress_handler() -> int:
                    elapsed = time.monotonic() - start
                    return 1 if elapsed >= float(timeout_seconds) else 0

                conn.set_progress_handler(
                    progress_handler, SQL_PROGRESS_HANDLER_STEP_COUNT
                )
            rows = conn.execute(sql).fetchall()
    except sqlite3.OperationalError as exc:
        message = str(exc)
        if "interrupted" in message.lower() and timeout_seconds:
            message = f"SQL execution timed out after {float(timeout_seconds):.2f}s"
        return SqlExecution(None, message, time.monotonic() - start)
    except Exception as exc:
        return SqlExecution(None, str(exc), time.monotonic() - start)
    return SqlExecution(rows, None, time.monotonic() - start)


def evaluate_sql(
    db_path: str | Path,
    predicted_sql: str,
    gold_sql: str,
    *,
    timeout_seconds: float | None = DEFAULT_SQL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Evaluate SQL by comparing execution results against the gold query."""

    gold = execute_sql(db_path, gold_sql, timeout_seconds=timeout_seconds)
    if gold.error:
        return {
            "is_correct": False,
            "reason": f"Error executing gold SQL: {gold.error}",
            "gold_error": gold.error,
        }

    predicted = execute_sql(db_path, predicted_sql, timeout_seconds=timeout_seconds)
    if predicted.error:
        return {
            "is_correct": False,
            "reason": f"Error executing predicted SQL: {predicted.error}",
            "predicted_error": predicted.error,
            "gold_row_count": len(gold.rows or []),
        }

    order_sensitive = _is_order_sensitive_query(gold_sql)
    is_correct = _results_equal(
        gold.rows or [],
        predicted.rows or [],
        order_sensitive=order_sensitive,
    )
    return {
        "is_correct": is_correct,
        "reason": "Execution results match."
        if is_correct
        else "Execution results do not match.",
        "comparison_mode": "ordered" if order_sensitive else "unordered",
        "gold_row_count": len(gold.rows or []),
        "predicted_row_count": len(predicted.rows or []),
        "gold_elapsed_seconds": gold.elapsed_seconds,
        "predicted_elapsed_seconds": predicted.elapsed_seconds,
    }


def _is_order_sensitive_query(sql: str) -> bool:
    return bool(
        re.search(r"\border\s+by\b", sql, flags=re.IGNORECASE)
        and re.search(r"\blimit\b", sql, flags=re.IGNORECASE)
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _row_tuple(row: Sequence[Any] | Any) -> tuple[Any, ...]:
    return tuple(row) if isinstance(row, (list, tuple)) else (row,)


def _cell_equal(left: Any, right: Any) -> bool:
    if _is_number(left) and _is_number(right):
        left_num = float(left)
        right_num = float(right)
        scale = max(abs(left_num), abs(right_num), 1.0)
        tolerance = max(NUMERIC_ABSOLUTE_EPSILON, NUMERIC_RELATIVE_EPSILON * scale)
        return abs(left_num - right_num) <= tolerance
    return left == right


def _row_equal(left: Sequence[Any] | Any, right: Sequence[Any] | Any) -> bool:
    left_tuple = _row_tuple(left)
    right_tuple = _row_tuple(right)
    return len(left_tuple) == len(right_tuple) and all(
        _cell_equal(left_cell, right_cell)
        for left_cell, right_cell in zip(left_tuple, right_tuple)
    )


def _row_sort_key(row: Sequence[Any] | Any) -> tuple[tuple[int, Any], ...]:
    parts = []
    for value in _row_tuple(row):
        if _is_number(value):
            parts.append((0, float(value)))
        else:
            parts.append((1, str(value)))
    return tuple(parts)


def _results_equal(
    left_rows: list[Sequence[Any] | Any],
    right_rows: list[Sequence[Any] | Any],
    *,
    order_sensitive: bool,
) -> bool:
    if len(left_rows) != len(right_rows):
        return False
    if not order_sensitive:
        left_rows = sorted(left_rows, key=_row_sort_key)
        right_rows = sorted(right_rows, key=_row_sort_key)
    return all(_row_equal(left, right) for left, right in zip(left_rows, right_rows))
