"""Core model selection functionality."""

from typing import Any, Callable

from pydantic import BaseModel

from .builders import build_llm
from .constants import MODEL_FIELDS


class ModelProxy:
    """
    Transparent proxy that wraps an LLM and allows model swapping.

    Usage (single-agent, auto-wired via selector):
        from crewai import Agent, LLM
        llm = ModelProxy(LLM(model='gpt-4o-mini'))
        agent = Agent(role="...", llm=llm, ...)
        # Selector auto-registers — no manual register() call needed.

    Usage (multi-agent, manual register):
        llm = ModelProxy(ChatOpenAI(model='gpt-4o-mini'))
        executor = AgentExecutor(agent=..., tools=tools)
        llm.register(executor)          # framework auto-detected
    """

    def __init__(self, initial_model: Any) -> None:
        object.__setattr__(self, "_optmodel", initial_model)
        object.__setattr__(self, "_optmodel_class", type(initial_model))
        # List of (new_llm: Any) -> None callables, populated by adapters.
        object.__setattr__(self, "_sync_callbacks", [])

    # ------------------------------------------------------------------
    # Internal sync machinery (called by FrameworkAdapters)
    # ------------------------------------------------------------------

    def _add_sync(self, fn: Callable[[Any], None]) -> None:
        """Register a sync callback.  Called by FrameworkAdapters only."""
        callbacks = object.__getattribute__(self, "_sync_callbacks")
        callbacks.append(fn)

    def _sync_registered_frameworks(self) -> None:
        """Fire all registered sync callbacks with the current model."""
        model = object.__getattribute__(self, "_optmodel")
        callbacks = object.__getattribute__(self, "_sync_callbacks")
        for fn in callbacks:
            fn(model)

    # ------------------------------------------------------------------
    # Public API — framework-agnostic
    # ------------------------------------------------------------------

    def register(self, agent: Any) -> None:
        """Auto-detect the agent's framework and register sync callbacks.

        This is the preferred registration method.  The framework is detected
        automatically.

        Args:
            agent: Any supported agent object (CrewAI Crew, LangChain
                AgentExecutor, LlamaIndex FunctionAgent / AgentWorkflow,
                OpenAI Agents SDK Agent).
        """
        from .adapter import get_adapter

        adapter = get_adapter(agent)
        if adapter is None:
            raise TypeError(
                f"Unsupported agent type: {type(agent).__name__}. "
                "Pass invoke_fn= to the selector instead, or implement a "
                "FrameworkAdapter for this agent type."
            )
        adapter.register_with_proxy(self, agent, [self])

    def set_model(self, model: Any) -> None:
        """Swap the underlying model.  Accepts a model object or a string name."""
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

    def get_model(self) -> Any:
        """Get the underlying model."""
        return object.__getattribute__(self, "_optmodel")

    # ------------------------------------------------------------------
    # Proxy protocol
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        model = object.__getattribute__(self, "_optmodel")
        if model is None:
            raise AttributeError("No model set")
        return getattr(model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_optmodel", "_optmodel_class", "_sync_callbacks", "_ag2_agents"):
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
