"""Core model selection functionality."""

from typing import Any, List

from pydantic import BaseModel

from .builders import build_llm
from .constants import MODEL_FIELDS


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
        object.__setattr__(self, "_crewai_agents", [])
        object.__setattr__(self, "_langchain_executors", [])
        object.__setattr__(self, "_llamaindex_agents", [])

    def register_crewai_agents(self, agents: List[Any]) -> None:
        """Register CrewAI agents whose LLM will be updated automatically on set_model()."""
        object.__setattr__(self, "_crewai_agents", list(agents))

    def register_langchain_executor(
        self, executor: Any, tools: List[Any], prompt: Any
    ) -> None:
        """Register a LangChain AgentExecutor for automatic chain rebuild on set_model()."""
        executors = object.__getattribute__(self, "_langchain_executors")
        executors.append((executor, tools, prompt))

    def register_llamaindex_agents(self, agents: List[Any]) -> None:
        """Register LlamaIndex agents whose LLM will be updated automatically on set_model()."""
        object.__setattr__(self, "_llamaindex_agents", list(agents))

    def set_model(self, model: Any) -> None:
        """Swap the underlying model. Accepts a model object or, in limited cases, a string name."""
        if isinstance(model, str):
            current_model = object.__getattribute__(self, "_optmodel")
            if current_model is None:
                raise AttributeError("No model set")

            # Try framework-specific rebuild first (handles cross-provider).
            new_model = build_llm(model, current_model)
            if new_model is not None:
                object.__setattr__(self, "_optmodel", new_model)
                object.__setattr__(self, "_optmodel_class", type(new_model))
                self._sync_registered_frameworks()
                return

            if isinstance(current_model, BaseModel):
                ## Pydantic path: use model_copy to create an immutable update
                target_field = next(
                    (f for f in MODEL_FIELDS if f in type(current_model).model_fields),
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
                        self._sync_registered_frameworks()
                        return
            else:
                ## Non-pydantic fallback: directly set the attribute
                target_field = next(
                    (a for a in MODEL_FIELDS if hasattr(current_model, a)), None
                )
                if target_field:
                    setattr(current_model, target_field, model)
                    self._sync_registered_frameworks()
                    return

            raise TypeError(
                "Cannot swap model using a string for this model type. "
                "Pass a fully-constructed model instance instead."
            )

        # Full model object passed — just swap it in.
        object.__setattr__(self, "_optmodel", model)
        object.__setattr__(self, "_optmodel_class", type(model))
        self._sync_registered_frameworks()

    def _sync_registered_frameworks(self) -> None:
        """Sync the current model to registered CrewAI agents and LangChain executors."""
        model = object.__getattribute__(self, "_optmodel")

        # CrewAI agent sync
        agents = object.__getattribute__(self, "_crewai_agents")
        if agents:
            from .crewai import build_crewai_llm

            model_name = next(
                (str(getattr(model, f)) for f in MODEL_FIELDS if hasattr(model, f)),
                None,
            )
            if model_name is not None:
                for ag in agents:
                    ag.llm = build_crewai_llm(model_name)

        # LangChain executor chain rebuild
        executors = object.__getattribute__(self, "_langchain_executors")
        if executors:
            from .langchain import sync_langchain_executor

            for executor, tools, prompt in executors:
                sync_langchain_executor(executor, model, tools, prompt)

        # LlamaIndex agent sync
        llamaindex_agents = object.__getattribute__(self, "_llamaindex_agents")
        if llamaindex_agents:
            from .llamaindex import sync_llamaindex_agents

            sync_llamaindex_agents(llamaindex_agents, model)

    def get_model(self) -> Any:
        """Get the underlying model."""
        return object.__getattribute__(self, "_optmodel")

    def __getattr__(self, name: str) -> Any:
        model = object.__getattribute__(self, "_optmodel")
        if model is None:
            raise AttributeError("No model set")
        return getattr(model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in (
            "_optmodel",
            "_crewai_agents",
            "_langchain_executors",
            "_llamaindex_agents",
        ):
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
