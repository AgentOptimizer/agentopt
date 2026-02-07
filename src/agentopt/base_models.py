"""
Type definitions for agentopt.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

if TYPE_CHECKING:
    from .model_proxy import ModelProxy


# Type aliases
EvalFn = Callable[[str, Any], Union[bool, float]]
ModelSpec = Union[str, Any]  # Model name string or model object
ModelsConfig = Dict["ModelProxy", List[ModelSpec]]
