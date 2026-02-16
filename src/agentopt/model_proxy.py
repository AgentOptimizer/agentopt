"""
Core model selection functionality.
"""

from typing import Any

from pydantic import BaseModel


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
        object.__setattr__(
            self, "_optmodel", initial_model
        )  # the model class that will be wrapped
        object.__setattr__(self, "_optmodel_class", type(initial_model))

    def set_model(self, model: Any) -> None:
        """Swap the underlying model. Accepts a model object or, in limited cases, a string name."""
        if isinstance(model, str):
            current_model = object.__getattribute__(self, "_optmodel")
            if current_model is None:
                raise AttributeError("No model set")

            # Try framework-specific rebuild first (handles cross-provider).
            new_model = self._rebuild_from_string(model, current_model)
            if new_model is not None:
                object.__setattr__(self, "_optmodel", new_model)
                object.__setattr__(self, "_optmodel_class", type(new_model))
                return

            # .model for crewai
            # .model_name for langchain and langgraph
            support_fields = ("model", "model_name", "model_id")

            if isinstance(current_model, BaseModel):
                ## Pydantic path: use model_copy to create an immutable update
                target_field = next(
                    (
                        f
                        for f in support_fields
                        if f in type(current_model).model_fields
                    ),
                    None,
                )
                if target_field:
                    try:
                        new_model = current_model.model_copy(
                            update={target_field: model}
                        )
                    except (TypeError, ValueError):
                        pass
                    else:
                        object.__setattr__(self, "_optmodel", new_model)
                        return
            else:
                ## Non-pydantic fallback: directly set the attribute
                target_field = next(
                    (a for a in support_fields if hasattr(current_model, a)), None
                )
                if target_field:
                    setattr(current_model, target_field, model)
                    return

            raise TypeError(
                "Cannot swap model using a string for this model type. Pass a fully-constructed model instance instead."
            )

        # Full model object passed — just swap it in.
        object.__setattr__(self, "_optmodel", model)
        object.__setattr__(self, "_optmodel_class", type(model))

    @staticmethod
    def _rebuild_from_string(model_name: str, current_model: Any) -> Any:
        """Try to rebuild the LLM via a framework-specific factory.

        Returns a new LLM object, or ``None`` if not applicable.
        """
        module = getattr(type(current_model), "__module__", "") or ""

        if module.startswith("crewai"):
            try:
                from crewai import LLM

                return LLM(model=model_name)
            except Exception:
                return None

        if module.startswith(
            (
                "langchain_openai",
                "langchain_anthropic",
                "langchain_google",
                "langchain_aws",
                "langchain_community",
            )
        ):
            try:
                from .model_factory import create_model_from_string

                return create_model_from_string(model_name)
            except Exception:
                return None

        return None

    def get_model(self) -> Any:
        """Get the underlying model."""
        return object.__getattribute__(self, "_optmodel")

    def __getattr__(self, name: str) -> Any:
        model = object.__getattribute__(self, "_optmodel")
        if model is None:
            raise AttributeError("No model set")
        return getattr(model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_optmodel":
            object.__setattr__(self, name, value)
        else:
            model = object.__getattribute__(self, "_optmodel")
            if model is None:
                raise AttributeError("No model set")
            setattr(model, name, value)

    def __call__(self, *args, **kwargs):
        """Forward calls to the underlying model."""
        model = object.__getattribute__(self, "_optmodel")
        if model is None:
            raise AttributeError("No model set")
        return model(*args, **kwargs)

    def __repr__(self) -> str:
        model = object.__getattribute__(self, "_optmodel")
        return f"ModelProxy({model!r})"
