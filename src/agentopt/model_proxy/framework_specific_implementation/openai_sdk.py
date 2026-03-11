"""OpenAI Agents SDK compatibility: ABC registration, model builder, and FrameworkAdapter."""

import copy
from typing import Any, Callable, List

from ..adapter import FrameworkAdapter
from ..constants import MODEL_FIELDS
from ..token_tracking import extract_usage


def build_openai_agents_model(model_name: str) -> Any:
    """Create an OpenAI Agents SDK ``Model`` from a model name string.

    Uses the default ``OpenAIProvider`` to resolve the name (e.g.
    ``"gpt-4o-mini"``) into a concrete ``Model`` instance.

    Returns the ``Model``.
    """
    from agents.models.openai_provider import OpenAIProvider

    provider = OpenAIProvider()
    return provider.get_model(model_name)


def _get_model_name(proxy: Any) -> str:
    """Extract the current model name string from the proxy's wrapped object."""
    model = object.__getattribute__(proxy, "_optmodel")
    for field in MODEL_FIELDS:
        if hasattr(model, field):
            return str(getattr(model, field))
    raise ValueError("Cannot determine model name from wrapped object")


async def _get_response(self: Any, *args: Any, **kwargs: Any) -> Any:
    """OpenAI Agents SDK ``Model`` interface — non-streaming.

    Delegates to the wrapped model if it implements ``get_response``,
    otherwise resolves a real SDK ``Model`` from the current model name.
    Extracts token usage from the response and feeds into the proxy's tracker.
    """
    model = self._get_effective_model()
    if hasattr(model, "get_response"):
        result = await model.get_response(*args, **kwargs)
    else:
        resolved = build_openai_agents_model(_get_model_name(self))
        result = await resolved.get_response(*args, **kwargs)
    tracker = self._get_effective_tracker()
    if tracker is not None:
        in_tok, out_tok = extract_usage(result)
        if in_tok or out_tok:
            tracker.add(in_tok, out_tok)
    return result


def _stream_response(self: Any, *args: Any, **kwargs: Any) -> Any:
    """OpenAI Agents SDK ``Model`` interface — streaming variant."""
    model = self._get_effective_model()
    if hasattr(model, "stream_response"):
        return model.stream_response(*args, **kwargs)
    resolved = build_openai_agents_model(_get_model_name(self))
    return resolved.stream_response(*args, **kwargs)


# ---------------------------------------------------------------------------
# FrameworkAdapter
# ---------------------------------------------------------------------------


class OpenAISDKAdapter(FrameworkAdapter):
    """Adapter for OpenAI Agents SDK ``Agent`` objects.

    ``patch_proxy_class`` registers ModelProxy as a virtual subclass of the
    SDK's ``Model`` ABC and patches ``get_response`` / ``stream_response``.
    These delegate to the proxy at call time, meaning model swaps take
    effect immediately without any explicit sync callbacks.
    """

    invoke_method_name = None  # uses a custom wrapper, not a named method

    def __init__(self) -> None:
        # Maps id(agent_copy) → TokenAccumulator for parallel clones.
        self._clone_trackers: dict = {}

    @classmethod
    def patch_proxy_class(cls, proxy_cls: type) -> None:
        """Register *proxy_cls* as a virtual subclass of the OpenAI Agents SDK
        ``Model`` ABC and patch the required interface methods.
        """
        from agents.models.interface import Model

        Model.register(proxy_cls)
        proxy_cls.get_response = _get_response
        proxy_cls.stream_response = _stream_response

    def detect(self, agent: Any) -> bool:
        return (type(agent).__module__ or "").startswith("agents")

    def detect_model(self, model: Any) -> bool:
        """Detect ModelProxy instances registered as OpenAI SDK Model virtual subclasses."""
        try:
            from agents.models.interface import Model

            return isinstance(model, Model)
        except ImportError:
            return False

    def get_invoke_fn(self, agent: Any) -> Callable:
        """Return a synchronous invoke callable that runs the agent via Runner."""
        from agents import Runner

        def _invoke(input_data: Any) -> Any:
            if isinstance(input_data, dict):
                prompt = input_data.get("input", str(input_data))
            else:
                prompt = str(input_data)
            result = Runner.run_sync(agent, prompt)
            return (
                result.final_output if hasattr(result, "final_output") else str(result)
            )

        return _invoke

    def register_with_proxy(
        self, proxy: Any, agent: Any, all_proxies: List[Any]
    ) -> None:
        pass  # Model swapping is handled transparently via Model ABC registration.

    def wrap_invoke_fn_with_tracker(
        self, invoke_fn: Callable, tracker: Any
    ) -> Callable:
        """Wrap Agent models in ModelProxy for token tracking.

        Introspects invoke_fn's closure to find Agent instances and wraps
        their .model in ModelProxy with the tracker. Since _get_response
        is already patched on ModelProxy to call extract_usage(), this
        enables automatic token tracking.
        """
        try:
            from agents import Agent
        except ImportError:
            return invoke_fn

        from ..proxy import ModelProxy

        if hasattr(invoke_fn, "__closure__") and invoke_fn.__closure__:
            for cell in invoke_fn.__closure__:
                try:
                    val = cell.cell_contents
                    if isinstance(val, Agent) and not isinstance(val.model, ModelProxy):
                        proxy = ModelProxy(val.model)
                        proxy._set_token_tracker(tracker)
                        val.model = proxy
                except ValueError:
                    pass

        return invoke_fn

    def create_token_tracker(self, agent: Any = None) -> Any:
        """Return the tracker pre-attached to a parallel clone's proxy, if any."""
        if agent is not None:
            tracker = self._clone_trackers.pop(id(agent), None)
            if tracker is not None:
                return tracker
        from ..token_tracking import TokenAccumulator

        return TokenAccumulator()

    def clone_for_parallel(
        self,
        agent: Any,
        proxies: List[Any],
        combo: tuple,
        get_model_name: Callable[[Any], str],
    ) -> Any:
        """Clone the Agent with a fresh ModelProxy wrapping a concrete Model.

        Each clone gets its own proxy with a per-thread TokenAccumulator so
        that proxy-level response interception tracks tokens correctly.
        """
        model_spec = combo[0]
        model_name = (
            model_spec if isinstance(model_spec, str) else get_model_name(model_spec)
        )
        fresh_model = build_openai_agents_model(model_name)

        # Wrap in a per-thread proxy so token interception works in parallel.
        from ..proxy import ModelProxy
        from ..token_tracking import TokenAccumulator

        fresh_proxy = ModelProxy(fresh_model)
        tracker = TokenAccumulator()
        fresh_proxy._set_token_tracker(tracker)

        agent_copy = copy.copy(agent)
        agent_copy.model = fresh_proxy
        self._clone_trackers[id(agent_copy)] = tracker
        return agent_copy


# Self-register — runs when this module is first imported.
from ..adapter import register_adapter  # noqa: E402

register_adapter(OpenAISDKAdapter())
