"""
Core model selection functionality.
"""

from typing import Any


class ModelProxy:
    """
    Transparent proxy that wraps an LLM and allows model swapping.

    Usage:
        from crewai import Agent, LLM

        # 1. Wrap the LLM
        llm = ModelProxy(LLM(model='gpt-4o-mini'))

        # 2. Use directly in agent
        agent = Agent(role="...", llm=llm, ...)

        # 3. ModelSelector can swap the underlying model
        llm.set_model(LLM(model='gpt-4o'))
    """

    def __init__(self, initial_model: Any) -> None:
        object.__setattr__(self, "_model", initial_model)
        object.__setattr__(self, "_model_class", type(initial_model))

    def set_model(self, model: Any) -> None:
        """Swap the underlying model. Accepts a model object or, in limited cases, a string name."""
        if isinstance(model, str):
            current_model = object.__getattribute__(self, "_model")
            if current_model is None:
                raise AttributeError("No model set")

            model_copy = getattr(current_model, "model_copy", None)
            if callable(model_copy):
                try:
                    new_model = model_copy(update={"model": model})
                except TypeError:
                    new_model = None
                else:
                    object.__setattr__(self, "_model", new_model)
                    return

            if hasattr(current_model, "model"):
                setattr(current_model, "model", model)
                return

            raise TypeError(
                "Cannot swap model using a string for this model type. Pass a fully-constructed model instance instead."
            )

        object.__setattr__(self, "_model", model)

    def get_model(self) -> Any:
        """Get the underlying model."""
        return object.__getattribute__(self, "_model")

    def __getattr__(self, name: str) -> Any:
        model = object.__getattribute__(self, "_model")
        if model is None:
            raise AttributeError("No model set")
        return getattr(model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_model":
            object.__setattr__(self, name, value)
        else:
            model = object.__getattribute__(self, "_model")
            if model is None:
                raise AttributeError("No model set")
            setattr(model, name, value)

    def __call__(self, *args, **kwargs):
        """Forward calls to the underlying model."""
        model = object.__getattribute__(self, "_model")
        if model is None:
            raise AttributeError("No model set")
        return model(*args, **kwargs)

    def __repr__(self) -> str:
        model = object.__getattribute__(self, "_model")
        return f"ModelProxy({model!r})"
