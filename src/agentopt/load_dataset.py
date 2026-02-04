"""
Dataset loading utilities for agentopt.
"""

import json
from pathlib import Path
from typing import List, Optional

from .types import EvaluationTask, Message


def load_dataset(dataset_dir: Optional[str]) -> List[EvaluationTask]:
    """
    Load evaluation tasks from dataset directory.

    Looks for JSONL files in the dataset_dir and loads tasks.
    Expected format: JSONL with "question" and "output" fields.
    Each line is a JSON object: {"question": "...", "output": "..."}

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        List of evaluation tasks
    """
    if not dataset_dir:
        return []

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    # Look for JSONL files
    jsonl_files = list(dataset_path.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in dataset directory: {dataset_dir}")

    # Load first JSONL file (can be extended to load multiple)
    tasks: List[EvaluationTask] = []
    with open(jsonl_files[0], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            # Validate required fields
            if "question" not in item:
                raise ValueError(f"Missing 'question' field in dataset entry: {item}")
            if "output" not in item:
                raise ValueError(f"Missing 'output' field in dataset entry: {item}")

            task = EvaluationTask(
                messages=[Message(role="user", content=item["question"])],
                expected_answer=item["output"],
            )
            tasks.append(task)

    return tasks
