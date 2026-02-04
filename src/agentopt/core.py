"""
Core model selection functionality.
"""

from typing import Any


class ModelProxy:
    """
    Proxy that forwards attribute access to the current model.
    Stores the model directly in _model attribute.
    """

    def __init__(self, attr_name: str, initial_model: Any = None) -> None:
        self.attr_name = attr_name
        self._model = initial_model  # Store model directly

    def __getattr__(self, name):
        if self._model is None:
            raise RuntimeError(
                f"No model set for attribute '{self.attr_name}'. "
                f"Call bind_model(obj, '{self.attr_name}', model) first."
            )
        return getattr(self._model, name)


def bind_model(obj: Any, attr_path: str, model: Any) -> Any:
    """
    Bind a model to an object's attribute, supporting nested paths.

    The attr_path can be:
    - A simple attribute: "model" → obj.model
    - A nested path: "B.C" → obj.B.C
    - A nested path with leading dot: ".B.C" → obj.B.C (dot is stripped)

    For LangChain agents created with ModelProxy, updates the existing proxy.
    For other objects, creates/updates a ModelProxy at the specified path.

    Usage:
        # Simple attribute
        bind_model(agent, "model", candidate_model)

        # Nested attribute
        bind_model(agent, "B.C", candidate_model)  # Sets agent.B.C

        # For model selection, rebind with different models:
        for model in candidate_models:
            bind_model(agent, "B.C", model)
            result = agent.invoke(...)

    Args:
        obj: Object to bind model to
        attr_path: Path to the attribute (e.g., "model", "B.C", ".B.C")
        model: The model object to bind
    """
    # Strip leading dot if present
    attr_path = attr_path.lstrip(".")

    # Split path into parts
    parts = attr_path.split(".")

    # Navigate to the target object (all parts except the last)
    target_obj = obj
    path_so_far = []
    for part in parts[:-1]:
        path_so_far.append(part)
        if not hasattr(target_obj, part):
            current_path = ".".join(path_so_far)
            raise AttributeError(
                f"Cannot bind model: path '{attr_path}' is invalid. "
                f"Object has no attribute '{part}' at '{current_path}'"
            )
        target_obj = getattr(target_obj, part)

    # Get the final attribute name
    final_attr = parts[-1]

    # Check if attribute already exists and is a ModelProxy
    if hasattr(target_obj, final_attr):
        existing = getattr(target_obj, final_attr)
        if isinstance(existing, ModelProxy):
            # Update existing proxy directly
            existing._model = model
            return obj

    # Create new proxy and set it as attribute
    proxy = ModelProxy(attr_path, model)
    setattr(target_obj, final_attr, proxy)
    return obj
