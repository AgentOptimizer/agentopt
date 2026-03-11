"""Core model selection functionality."""

import functools
import inspect
import threading
from typing import Any, Callable

from pydantic import BaseModel

from .builders import build_llm
from .constants import MODEL_FIELDS
from .token_tracking import TokenAccumulator, extract_usage


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
        # Token tracker for proxy-level response interception.
        object.__setattr__(self, "_token_tracker", None)
        # Thread-local storage for per-thread model/tracker overrides (parallel eval).
        object.__setattr__(self, "_thread_local", threading.local())

    # ------------------------------------------------------------------
    # Thread-local model/tracker overrides (for parallel evaluation)
    # ------------------------------------------------------------------

    def _get_effective_model(self) -> Any:
        """Return thread-local model if set, otherwise the default."""
        tl = object.__getattribute__(self, "_thread_local")
        model = getattr(tl, "model", None)
        if model is not None:
            return model
        return object.__getattribute__(self, "_optmodel")

    def _get_effective_tracker(self) -> Any:
        """Return thread-local tracker if set, otherwise the default."""
        tl = object.__getattribute__(self, "_thread_local")
        tracker = getattr(tl, "tracker", None)
        if tracker is not None:
            return tracker
        return object.__getattribute__(self, "_token_tracker")

    def _set_thread_model(self, model: Any, tracker: TokenAccumulator) -> None:
        """Set per-thread model and tracker override.

        For AG2 agents, acquires a per-proxy lock first to serialize access
        to the shared ProxyAwareWrapper instances (AG2 spawns internal threads
        for LLM calls, making pure thread-local insufficient).
        """
        self._acquire_ag2_lock()
        tl = object.__getattribute__(self, "_thread_local")
        tl.model = model
        tl.tracker = tracker
        self._propagate_ag2_override(model, tracker)

    def _clear_thread_model(self) -> None:
        """Clear per-thread model/tracker overrides and release AG2 lock."""
        tl = object.__getattribute__(self, "_thread_local")
        tl.model = None
        tl.tracker = None
        self._propagate_ag2_override(None, None)
        self._release_ag2_lock()

    def _acquire_ag2_lock(self) -> None:
        """Acquire the AG2 eval lock if present (serializes parallel evals)."""
        try:
            lock = object.__getattribute__(self, "_ag2_eval_lock")
            lock.acquire()
        except AttributeError:
            pass

    def _release_ag2_lock(self) -> None:
        """Release the AG2 eval lock if present."""
        try:
            lock = object.__getattribute__(self, "_ag2_eval_lock")
            lock.release()
        except AttributeError:
            pass

    def _propagate_ag2_override(self, model: Any, tracker: Any) -> None:
        """If this proxy has AG2 agents, set/clear override on their wrappers."""
        try:
            ag2_agents = object.__getattribute__(self, "_ag2_agents")
        except AttributeError:
            return
        for agent in ag2_agents:
            client = getattr(agent, "client", None)
            if client is not None and hasattr(client, "_override_model"):
                client._override_model = model
                client._override_tracker = tracker

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
        """Get the underlying model (respects thread-local overrides)."""
        return self._get_effective_model()

    # ------------------------------------------------------------------
    # Proxy protocol
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        model = self._get_effective_model()
        if model is None:
            raise AttributeError("No model set")
        attr = getattr(model, name)
        tracker = self._get_effective_tracker()
        if tracker is not None and callable(attr):
            return self._wrap_for_usage(attr, tracker)
        return attr

    @staticmethod
    def _wrap_for_usage(method: Callable, tracker: TokenAccumulator) -> Callable:
        """Wrap a callable to extract token usage from its return value."""
        if inspect.iscoroutinefunction(method):

            @functools.wraps(method)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await method(*args, **kwargs)
                in_tok, out_tok = extract_usage(result)
                if in_tok or out_tok:
                    tracker.add(in_tok, out_tok)
                return result

            return _async_wrapper

        @functools.wraps(method)
        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = method(*args, **kwargs)
            in_tok, out_tok = extract_usage(result)
            if in_tok or out_tok:
                tracker.add(in_tok, out_tok)
            return result

        return _sync_wrapper

    def _set_token_tracker(self, tracker: TokenAccumulator) -> None:
        """Attach a token tracker for proxy-level response interception."""
        object.__setattr__(self, "_token_tracker", tracker)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in (
            "_optmodel",
            "_optmodel_class",
            "_sync_callbacks",
            "_ag2_agents",
            "_ag2_eval_lock",
            "_token_tracker",
            "_thread_local",
        ):
            object.__setattr__(self, name, value)
        else:
            model = self._get_effective_model()
            if model is None:
                raise AttributeError("No model set")
            setattr(model, name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward calls to the underlying model, intercepting for token usage."""
        model = self._get_effective_model()
        if model is None:
            raise AttributeError("No model set")
        tracker = self._get_effective_tracker()
        if tracker is not None:
            result = model(*args, **kwargs)
            in_tok, out_tok = extract_usage(result)
            if in_tok or out_tok:
                tracker.add(in_tok, out_tok)
            return result
        return model(*args, **kwargs)

    def __repr__(self) -> str:
        model = self._get_effective_model()
        return f"ModelProxy({model!r})"
