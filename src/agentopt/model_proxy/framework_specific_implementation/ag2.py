"""AG2 (AutoGen 2) support for ModelProxy.

AG2 validates llm_config and rejects ModelProxy, so we patch
ConversableAgent._validate_llm_config to accept ModelProxy and
silently convert LLMConfig → AG2ConfigWrapper in ModelProxy.__init__
so that set_model() can update the config in place.

Additionally, ConversableAgent.__init__ is patched to auto-register
agents with the proxy, so _sync_registered_frameworks() can recreate
the LLMConfig and force client recreation on set_model().

This is the same registration pattern used for OpenAI Agents SDK
in openai_sdk.py.
"""

from __future__ import annotations

import copy
import os
import threading

from ..adapter import FrameworkAdapter
from typing import Any, Callable, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Thread-local token tracking for parallel clone_fn path
# ---------------------------------------------------------------------------

_token_tracking_local = threading.local()


def _read_total_tokens(agent: Any) -> Tuple[int, int]:
    """Read cumulative prompt/completion tokens from agent.client.total_usage_summary."""
    summary = getattr(getattr(agent, "client", None), "total_usage_summary", None)
    if not isinstance(summary, dict):
        return (0, 0)
    in_tok = out_tok = 0
    for key, val in summary.items():
        if key == "total_cost" or not isinstance(val, dict):
            continue
        in_tok += val.get("prompt_tokens", 0)
        out_tok += val.get("completion_tokens", 0)
    return (in_tok, out_tok)


class _TrackedResponse:
    """Wraps an AG2 response to accumulate token usage after events are consumed.

    AG2 stores token usage in ``agent.client.total_usage_summary``, not in
    conversation messages.  This wrapper intercepts ``events`` iteration and
    records the delta in the accumulator after the generator is exhausted.
    """

    def __init__(
        self, response: Any, agent: Any, before: Tuple[int, int], acc: Any
    ) -> None:
        self._response = response
        # Store as an instance attribute so it shadows any class-level property.
        self.events = self._build_tracked_events(response, agent, before, acc)

    @staticmethod
    def _build_tracked_events(response: Any, agent: Any, before: Tuple[int, int], acc: Any):  # type: ignore[override]
        for event in response.events:
            yield event
        # Events exhausted — compute token delta from agent's usage summary.
        after_in, after_out = _read_total_tokens(agent)
        acc.input_tokens += after_in - before[0]
        acc.output_tokens += after_out - before[1]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


class _AG2TokenAccumulator:
    """Accumulates token counts across multiple ConversableAgent.run() calls."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0


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
# FrameworkAdapter
# ---------------------------------------------------------------------------


class AG2Adapter(FrameworkAdapter):
    """Adapter for AG2 ``ConversableAgent`` objects."""

    invoke_method_name = "run"

    def __init__(self) -> None:
        # Maps agent id → last response, set by the invoke closure.
        self._last_responses: Dict[int, Any] = {}

    def detect_model(self, model: Any) -> bool:
        return is_ag2_llm(model)

    def get_token_usage(self, agent: Any) -> Tuple[int, int]:
        stored = self._last_responses.pop(id(agent), None)
        if stored is None:
            return (0, 0)
        if isinstance(stored, _AG2TokenAccumulator):
            result = (stored.input_tokens, stored.output_tokens)
            stored.input_tokens = 0
            stored.output_tokens = 0
            return result
        return (0, 0)

    def wrap_invoke_fn_for_parallel(self, invoke_fn: Any) -> Any:
        """Wrap invoke_fn to capture token usage from any ConversableAgent.run() calls.

        Introspects invoke_fn.__closure__ to find ConversableAgent instances and
        patches their run() at the instance level (takes precedence over class method).
        Uses thread-local storage so each parallel task tracks its own tokens
        without cross-thread contamination.
        """
        fresh_agents = []
        try:
            from autogen import ConversableAgent as _CA

            if hasattr(invoke_fn, "__closure__") and invoke_fn.__closure__:
                for cell in invoke_fn.__closure__:
                    try:
                        val = cell.cell_contents
                        if isinstance(val, _CA):
                            fresh_agents.append(val)
                    except ValueError:
                        pass
        except ImportError:
            pass

        acc = _AG2TokenAccumulator()
        self._last_responses[id(acc)] = acc

        for agent in fresh_agents:
            _orig_run = agent.run

            def _instance_patched_run(
                *args: Any, _orig: Any = _orig_run, _agent: Any = agent, **kwargs: Any
            ) -> Any:
                before = _read_total_tokens(_agent)
                response = _orig(*args, **kwargs)
                _acc = getattr(_token_tracking_local, "accumulator", None)
                if _acc is not None:
                    return _TrackedResponse(response, _agent, before, _acc)
                return response

            agent.run = _instance_patched_run

        def wrapped(input_data: Any) -> Any:
            _token_tracking_local.accumulator = acc
            try:
                return invoke_fn(input_data)
            finally:
                _token_tracking_local.accumulator = None

        return acc, wrapped

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

        ConversableAgent.__init__ = _patched_agent_init

    def detect(self, agent: Any) -> bool:
        """Check if an agent is an AG2 ConversableAgent."""
        return (type(agent).__module__ or "").startswith("autogen") and hasattr(
            agent, "run"
        )

    def get_invoke_fn(self, agent: Any) -> Callable:
        """Wrap agent.run() to handle input dict and extract content."""
        acc = _AG2TokenAccumulator()
        self._last_responses[id(agent)] = acc

        def _invoke(input_data: Any) -> Any:
            if isinstance(input_data, dict):
                message = input_data.get("input", str(input_data))
            else:
                message = str(input_data)
            before = _read_total_tokens(agent)
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

            result = _extract(response)
            after_in, after_out = _read_total_tokens(agent)
            acc.input_tokens += after_in - before[0]
            acc.output_tokens += after_out - before[1]
            return result

        return _invoke

    def register_with_proxy(
        self, proxy: Any, agent: Any, all_proxies: List[Any]
    ) -> None:
        """Register sync callback that calls sync_ag2_agents on set_model()."""
        try:
            ag2_agents = object.__getattribute__(proxy, "_ag2_agents")
        except AttributeError:
            ag2_agents: list = []
            object.__setattr__(proxy, "_ag2_agents", ag2_agents)

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
                ag.client = ag._create_client(new_config)

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
