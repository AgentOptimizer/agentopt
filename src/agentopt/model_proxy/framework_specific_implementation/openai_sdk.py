"""OpenAI Agents SDK compatibility: ABC registration, model builder, and FrameworkAdapter."""

import copy
from typing import Any, Callable, Dict, List, Tuple

from ..adapter import FrameworkAdapter
from ..constants import MODEL_FIELDS


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
    """
    model = object.__getattribute__(self, "_optmodel")
    if hasattr(model, "get_response"):
        return await model.get_response(*args, **kwargs)
    resolved = build_openai_agents_model(_get_model_name(self))
    return await resolved.get_response(*args, **kwargs)


def _stream_response(self: Any, *args: Any, **kwargs: Any) -> Any:
    """OpenAI Agents SDK ``Model`` interface — streaming variant."""
    model = object.__getattribute__(self, "_optmodel")
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
        # Maps agent id → last RunnerResult, set by the invoke closure.
        self._last_results: Dict[int, Any] = {}

    def get_token_usage(self, agent: Any) -> Tuple[int, int]:
        result = self._last_results.pop(id(agent), None)
        if result is None:
            return (0, 0)
        in_tok = out_tok = 0
        for resp in result.raw_responses:
            if resp.usage is not None:
                in_tok += resp.usage.input_tokens
                out_tok += resp.usage.output_tokens
        return (in_tok, out_tok)

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

    def get_invoke_fn(self, agent: Any) -> Callable:
        """Return a synchronous invoke callable that runs the agent via Runner."""
        from agents import Runner

        def _invoke(input_data: Any) -> Any:
            if isinstance(input_data, dict):
                prompt = input_data.get("input", str(input_data))
            else:
                prompt = str(input_data)
            result = Runner.run_sync(agent, prompt)
            self._last_results[id(agent)] = result
            return (
                result.final_output if hasattr(result, "final_output") else str(result)
            )

        return _invoke

    def register_with_proxy(
        self, proxy: Any, agent: Any, all_proxies: List[Any]
    ) -> None:
        # No-op: OpenAI SDK resolves the model via _get_response at call time.
        pass

    def clone_for_parallel(
        self,
        agent: Any,
        proxies: List[Any],
        combo: tuple,
        get_model_name: Callable[[Any], str],
    ) -> Any:
        """Clone the Agent with a fresh concrete Model for this combination.

        Bypasses the proxy entirely on the clone — assigns a real
        ``OpenAIProvider`` Model directly so threads don't share mutable state.
        """
        model_spec = combo[0]
        model_name = (
            model_spec if isinstance(model_spec, str) else get_model_name(model_spec)
        )
        fresh_model = build_openai_agents_model(model_name)
        agent_copy = copy.copy(agent)
        agent_copy.model = fresh_model
        return agent_copy


# Self-register — runs when this module is first imported.
from ..adapter import register_adapter  # noqa: E402

register_adapter(OpenAISDKAdapter())
