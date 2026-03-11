"""AG2 (AutoGen 2) support for ModelProxy.

AG2 validates llm_config and rejects ModelProxy, so we patch
ConversableAgent._validate_llm_config to accept ModelProxy and
silently convert LLMConfig → AG2ConfigWrapper in ModelProxy.__init__
so that set_model() can update the config in place.

Additionally, ConversableAgent.__init__ is patched to auto-register
agents with the proxy and replace the agent's ``client`` with a
``ProxyAwareWrapper`` — a subclass of ``OpenAIWrapper`` that resolves
the LLM config from the proxy at call time.  This makes AG2 agents
respect ``ModelProxy.set_model()`` and thread-local overrides for
parallel evaluation.
"""

from __future__ import annotations

import copy
import os
import threading

from ..adapter import FrameworkAdapter
from ..token_tracking import TokenAccumulator
from typing import Any, Callable, List, Tuple


def _read_usage_summary(summary: Any) -> Tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from a usage summary dict."""
    if not isinstance(summary, dict):
        return (0, 0)
    in_tok = out_tok = 0
    for key, val in summary.items():
        if key == "total_cost" or not isinstance(val, dict):
            continue
        in_tok += val.get("prompt_tokens", 0)
        out_tok += val.get("completion_tokens", 0)
    return (in_tok, out_tok)


def is_ag2_llm(llm: Any) -> bool:
    """Check if an LLM object is an AG2ConfigWrapper."""
    return isinstance(llm, AG2ConfigWrapper)


class AG2ConfigWrapper:
    """Mutable wrapper around AG2's config_list.

    Exposes a ``model`` property so ModelProxy.set_model() can find and
    update the model name.  The config_list is shared by reference with
    the LLMConfig created during validation, so in-place mutations
    propagate to the agent automatically.
    """

    @staticmethod
    def _build_config(model: str) -> dict:
        """Build an AG2 config_list entry for the given model name."""
        bare = model.split("/", 1)[-1] if "/" in model else model
        if bare.startswith("claude") or model.startswith("anthropic/"):
            return {
                "api_type": "anthropic",
                "model": bare,
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            }
        return {
            "api_type": "openai",
            "model": bare,
            "api_key": os.getenv("OPENAI_API_KEY"),
        }

    def __init__(self, model: str) -> None:
        self.config_list = [self._build_config(model)]

    @property
    def model(self) -> str:
        return self.config_list[0]["model"]

    @model.setter
    def model(self, value: str) -> None:
        self.config_list[0] = self._build_config(value)


# ---------------------------------------------------------------------------
# ProxyAwareWrapper — makes AG2 agents route LLM calls through the proxy
# ---------------------------------------------------------------------------


class ProxyAwareWrapper:
    """Wrapper that resolves the LLM config from a ``ModelProxy`` at call time.

    Each ``create()`` call reads the proxy's effective model (which is
    thread-local–aware), obtains or rebuilds the appropriate ``OpenAIWrapper``,
    and feeds token usage into the proxy's effective tracker.

    **AG2 thread note:** AG2's ``agent.run()`` spawns an internal thread
    (``initiate_chat``), so the LLM call happens in a *different* thread
    from the one that set the proxy's thread-local override.  To handle
    this, ``ProxyAwareWrapper`` snapshots the proxy's effective model and
    tracker into instance-level attributes (``_override_model`` and
    ``_override_tracker``) that are visible across threads.  These are
    set/cleared by ``_set_thread_model``/``_clear_thread_model`` on the
    proxy, which are called from the selectors.
    """

    def __init__(self, proxy: Any) -> None:
        self._proxy = proxy
        # Instance-level override — set by proxy._set_thread_model() hook.
        self._override_model: Any = None  # AG2ConfigWrapper or None
        self._override_tracker: Any = None  # TokenAccumulator or None
        # Build the initial inner wrapper.
        self._inner = self._build_inner_from(proxy._get_effective_model())
        self._current_model = (
            self._inner._config_list[0].get("model")
            if self._inner._config_list
            else None
        )
        # Expose usage summaries for _read_total_tokens compat.
        self.total_usage_summary = self._inner.total_usage_summary
        self.actual_usage_summary = self._inner.actual_usage_summary

    @staticmethod
    def _build_inner_from(wrapper: Any) -> Any:
        """Build an OpenAIWrapper from an AG2ConfigWrapper."""
        from autogen import LLMConfig
        from autogen.oai.client import OpenAIWrapper

        config = wrapper.config_list[0]
        return OpenAIWrapper(**LLMConfig(config))

    def _get_effective_model_and_tracker(self) -> Tuple[Any, Any]:
        """Return (AG2ConfigWrapper, tracker), preferring instance override."""
        if self._override_model is not None:
            return self._override_model, self._override_tracker
        return self._proxy._get_effective_model(), self._proxy._get_effective_tracker()

    def create(self, **config: Any) -> Any:
        """Resolve effective model, call inner wrapper, and track tokens."""
        wrapper, tracker = self._get_effective_model_and_tracker()
        model = wrapper.config_list[0].get("model")

        # Rebuild inner wrapper if model changed.
        if model != self._current_model:
            self._inner = self._build_inner_from(wrapper)
            self._current_model = model

        response = self._inner.create(**config)

        # Sync usage summaries for _read_total_tokens compat.
        self.total_usage_summary = self._inner.total_usage_summary
        self.actual_usage_summary = self._inner.actual_usage_summary

        # Feed tokens into the effective tracker.
        if tracker is not None and self._inner.actual_usage_summary:
            in_tok, out_tok = _read_usage_summary(self._inner.actual_usage_summary)
            if in_tok or out_tok:
                tracker.add(in_tok, out_tok, model_name=model or "unknown")

        return response

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else to the inner wrapper."""
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# _TrackingClientWrapper — token tracking for cloned agents (parallel path)
# ---------------------------------------------------------------------------


class _TrackingClientWrapper:
    """Thin wrapper around OpenAIWrapper that feeds token usage into a tracker.

    Used for cloned agents in parallel evaluation where there is no
    ModelProxy / ProxyAwareWrapper — just a bare OpenAIWrapper that
    needs its usage reported to a TokenAccumulator.
    """

    def __init__(
        self, inner: Any, tracker: TokenAccumulator, model_name: str = "unknown"
    ) -> None:
        self._inner = inner
        self._tracker = tracker
        self._model_name = model_name

    def create(self, **kwargs: Any) -> Any:
        response = self._inner.create(**kwargs)
        if self._inner.actual_usage_summary:
            in_tok, out_tok = _read_usage_summary(self._inner.actual_usage_summary)
            if in_tok or out_tok:
                self._tracker.add(in_tok, out_tok, model_name=self._model_name)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# FrameworkAdapter
# ---------------------------------------------------------------------------


class AG2Adapter(FrameworkAdapter):
    """Adapter for AG2 ``ConversableAgent`` objects."""

    invoke_method_name = "run"

    def detect_model(self, model: Any) -> bool:
        return is_ag2_llm(model)

    def create_token_tracker(self, agent: Any = None) -> TokenAccumulator:
        """Create a TokenAccumulator for AG2 token tracking.

        When *agent* is provided and its client is a bare OpenAIWrapper
        (not a ProxyAwareWrapper), wrap the client so that each create()
        call feeds token counts into the tracker automatically.
        """
        tracker = TokenAccumulator()
        if agent is not None and hasattr(agent, "client"):
            client = agent.client
            if not isinstance(client, ProxyAwareWrapper):
                # Extract model name from agent's llm_config for tracking.
                mname = "unknown"
                cfg = getattr(agent, "llm_config", None)
                if cfg is not None:
                    cl = getattr(cfg, "config_list", None) or []
                    if cl:
                        mname = (
                            cl[0].get("model", "unknown")
                            if isinstance(cl[0], dict)
                            else "unknown"
                        )
                agent.client = _TrackingClientWrapper(client, tracker, model_name=mname)
        return tracker

    def wrap_invoke_fn_with_tracker(
        self, invoke_fn: Any, tracker: TokenAccumulator
    ) -> Any:
        """No-op for AG2 — token tracking is handled by ProxyAwareWrapper.

        ProxyAwareWrapper.create() feeds token counts into the proxy's
        effective tracker on every LLM call, so no external wrapping is needed.
        """
        return invoke_fn

    @classmethod
    def patch_proxy_class(cls, proxy_cls: type) -> None:
        """Patch *proxy_cls* to work transparently with AG2's LLMConfig.

        1. Patches ``ModelProxy.__init__`` to silently convert LLMConfig →
           AG2ConfigWrapper so ``set_model()`` works via the wrapper's
           ``model`` property.
        2. Patches ``ConversableAgent._validate_llm_config`` to accept
           ModelProxy and return a linked LLMConfig whose config_list is
           shared by reference with the wrapper.
        3. Patches ``ConversableAgent.__init__`` to auto-register agents
           with the proxy for explicit sync on ``set_model()``.
        """
        from autogen import ConversableAgent, LLMConfig

        @staticmethod
        def _is_ag2_llm_config(obj: Any) -> bool:
            """Check if an object is an AG2 LLMConfig."""
            return (type(obj).__module__ or "").startswith("autogen") and type(
                obj
            ).__name__ == "LLMConfig"

        # --- Patch ModelProxy.__init__ ---
        original_init = proxy_cls.__init__

        def _patched_init(self: Any, initial_model: Any) -> None:
            if _is_ag2_llm_config(initial_model):
                config_list = list(initial_model.config_list)
                if not config_list:
                    raise ValueError("LLMConfig has an empty config_list")
                first = config_list[0]
                cfg = first if isinstance(first, dict) else first.model_dump()
                model_name = cfg.get("model")
                if not model_name:
                    raise ValueError("LLMConfig entry missing 'model' key")
                wrapper = AG2ConfigWrapper(model_name)
                original_init(self, wrapper)
                object.__setattr__(self, "_ag2_agents", [])
                object.__setattr__(self, "_ag2_eval_lock", threading.Lock())
            else:
                original_init(self, initial_model)

        proxy_cls.__init__ = _patched_init

        # --- Patch ConversableAgent._validate_llm_config ---
        original_validate = ConversableAgent._validate_llm_config.__func__

        @classmethod  # type: ignore[misc]
        def _patched_validate(klass: type, llm_config: Any) -> Any:
            if isinstance(llm_config, proxy_cls):
                wrapper = object.__getattribute__(llm_config, "_optmodel")
                if isinstance(wrapper, AG2ConfigWrapper):
                    return LLMConfig(*wrapper.config_list)
            return original_validate(klass, llm_config)

        ConversableAgent._validate_llm_config = _patched_validate

        # --- Patch ConversableAgent.__init__ for auto-registration ---
        original_agent_init = ConversableAgent.__init__

        def _patched_agent_init(self_agent: Any, *args: Any, **kwargs: Any) -> None:
            llm_config_arg = kwargs.get("llm_config")
            original_agent_init(self_agent, *args, **kwargs)
            if isinstance(llm_config_arg, proxy_cls):
                ag2_agents = object.__getattribute__(llm_config_arg, "_ag2_agents")
                if self_agent not in ag2_agents:
                    ag2_agents.append(self_agent)
                # Replace the agent's client with a proxy-aware wrapper so
                # all LLM calls route through the proxy (thread-local aware).
                self_agent.client = ProxyAwareWrapper(llm_config_arg)

        ConversableAgent.__init__ = _patched_agent_init

    def detect(self, agent: Any) -> bool:
        """Check if an agent is an AG2 ConversableAgent."""
        return (type(agent).__module__ or "").startswith("autogen") and hasattr(
            agent, "run"
        )

    def get_invoke_fn(self, agent: Any) -> Callable:
        """Wrap agent.run() to handle input dict and extract content.

        Token tracking is handled by ProxyAwareWrapper.create() — no
        before/after delta tracking is needed here.
        """

        def _invoke(input_data: Any) -> Any:
            if isinstance(input_data, dict):
                message = input_data.get("input", str(input_data))
            else:
                message = str(input_data)
            response = agent.run(message=message, max_turns=1, user_input=False)

            def _extract(r: Any) -> str:
                for _ in r.events:
                    pass
                if hasattr(r, "summary") and r.summary:
                    return r.summary
                if r.messages:
                    last = r.messages[-1]
                    if isinstance(last, dict):
                        return last.get("content") or str(last)
                    return last.content if hasattr(last, "content") else str(last)
                return ""

            return _extract(response)

        return _invoke

    def register_with_proxy(
        self, proxy: Any, agent: Any, all_proxies: List[Any]
    ) -> None:
        """Register sync callback to update agent llm_config on set_model().

        The ProxyAwareWrapper auto-resolves the model on each create() call,
        so client recreation is not needed.  We update llm_config for
        consistency (e.g. if user code reads agent.llm_config).
        """
        try:
            ag2_agents = object.__getattribute__(proxy, "_ag2_agents")
        except AttributeError:
            ag2_agents: list = []
            object.__setattr__(proxy, "_ag2_agents", ag2_agents)
            object.__setattr__(proxy, "_ag2_eval_lock", threading.Lock())

        if agent is None:
            return  # invoke_fn path: no agent sync needed

        if agent not in ag2_agents:
            ag2_agents.append(agent)

        def _sync(new_llm: Any, _agents: list = ag2_agents) -> None:
            if not isinstance(new_llm, AG2ConfigWrapper):
                return
            from autogen import LLMConfig

            new_config = LLMConfig(*new_llm.config_list)
            for ag in _agents:
                ag.llm_config = new_config

        proxy._add_sync(_sync)

    def clone_for_parallel(
        self,
        agent: Any,
        proxies: List[Any],
        combo: tuple,
        get_model_name: Callable[[Any], str],
    ) -> Any:
        """Clone the AG2 agent with a fresh LLMConfig for this combination."""
        model_spec = combo[0]
        model_name = (
            model_spec if isinstance(model_spec, str) else get_model_name(model_spec)
        )
        from autogen import LLMConfig

        new_config = LLMConfig(AG2ConfigWrapper._build_config(model_name))
        agent_copy = copy.copy(agent)
        agent_copy.llm_config = new_config
        agent_copy.client = agent_copy._create_client(new_config)
        return agent_copy


# Self-register — runs when this module is first imported.
from ..adapter import register_adapter  # noqa: E402

register_adapter(AG2Adapter())
