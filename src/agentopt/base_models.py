"""
Type definitions for agentopt.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence, Tuple, Union

if TYPE_CHECKING:
    from .model_proxy import ModelProxy


# Type aliases
EvalFn = Callable[[str, Any], Union[bool, float]]
ModelSpec = Union[str, Any]  # Model name string or model object
ModelsConfig = Dict["ModelProxy", List[ModelSpec]]

# Dataset: any Sequence of (input_data, expected_answer) pairs.
# Accepts list, tuple, or any custom class with __getitem__ and __len__.
Dataset = Sequence[Tuple[Any, Any]]


def validate_dataset(dataset: Any) -> None:
    """Validate that *dataset* is usable for evaluation.

    Checks:
    1. Has ``__len__`` (so ``len(dataset)`` works).
    2. Has ``__getitem__`` (so indexing / iteration works).
    3. Is not empty.
    4. First element unpacks into exactly 2 values (input_data, expected_answer).

    Raises:
        TypeError: If the dataset is missing required capabilities.
        ValueError: If the dataset is empty or elements have wrong structure.
    """
    if not hasattr(dataset, "__len__"):
        raise TypeError(
            f"dataset must support len(), but got {type(dataset).__name__}. "
            "Provide a list, tuple, or any object with __len__ and __getitem__."
        )
    if not hasattr(dataset, "__getitem__"):
        raise TypeError(
            f"dataset must support indexing (dataset[i]), but got {type(dataset).__name__}. "
            "Provide a list, tuple, or any object with __len__ and __getitem__."
        )
    if len(dataset) == 0:
        raise ValueError("dataset must not be empty.")
    try:
        first = dataset[0]
    except (KeyError, IndexError, TypeError) as e:
        raise TypeError(
            f"dataset must support integer indexing (dataset[0]), "
            f"but got {type(dataset).__name__}. {e}"
        ) from None
    try:
        input_data, expected_answer = first
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Each dataset element must unpack into (input_data, expected_answer), "
            f"but the first element is {type(first).__name__}: {first!r}. {e}"
        ) from None
