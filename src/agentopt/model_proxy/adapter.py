"""Framework adapter registry for ModelProxy.

Each supported LLM framework (CrewAI, LangChain, LlamaIndex, OpenAI SDK, …)
ships its own ``FrameworkAdapter`` subclass that knows how to:

  1. Detect whether an agent object belongs to it (``detect``).
  2. Return the right invoke callable for the selector (``get_invoke_fn``).
  3. Register per-instance sync callbacks on a ``ModelProxy`` so that
     ``proxy.set_model()`` propagates to the agent (``register_with_proxy``).
  4. Clone the agent independently for parallel evaluation
     (``clone_for_parallel``).

Adding a new framework means creating one new file with a
``FrameworkAdapter`` subclass and calling ``register_adapter(MyAdapter())``
at module level.  The adapter is auto-loaded when the framework module is
first imported (which happens as part of the normal ``model_proxy`` package
import chain).
"""

import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple

from .token_tracking import TokenAccumulator


class FrameworkAdapter(ABC):
    """Abstract base for all framework adapters.

    Subclasses must implement ``detect`` and ``get_invoke_fn``.
    The remaining methods have safe default implementations that work
    for stateless / closure-based frameworks.
    """

    # Subclasses should override this with the agent's invoke method name
    # ("kickoff", "invoke", "run") so the parallel selector can re-bind
    # the method after cloning.  ``None`` means the adapter wraps it itself.
    invoke_method_name: Optional[str] = None

    @classmethod
    def patch_proxy_class(cls, proxy_cls: type) -> None:
        """One-time class-level patching to make ModelProxy compatible.

        Default: no-op.  Override for frameworks that require ABC registration
        or method patching (e.g. CrewAI ``BaseLLM``, OpenAI SDK ``Model``).

        Called once per adapter from ``model_proxy/__init__.py`` after all
        adapters have been imported.
        """
        pass

    @abstractmethod
    def detect(self, agent: Any) -> bool:
        """Return True if this adapter can handle *agent*."""
        ...

    @abstractmethod
    def get_invoke_fn(self, agent: Any) -> Callable:
        """Return a callable that runs *agent* on a single input."""
        ...

    def register_with_proxy(
        self,
        proxy: Any,
        agent: Any,
        all_proxies: List[Any],
    ) -> None:
        """Register sync callbacks on *proxy* for sequential model swapping.

        Called by ``BaseModelSelector.__init__`` and ``ModelProxy.register()``.
        Default: no-op (correct for OpenAI SDK / Claude SDK where no
        per-instance sync is needed).

        Args:
            proxy: The ``ModelProxy`` instance to register callbacks on.
            agent: The top-level agent object.
            all_proxies: All proxies in the current selector's ``models`` dict,
                used for positional mapping (proxy i → sub-agent i).
        """
        pass

    def detect_model(self, model: Any) -> bool:
        """Return True if this adapter's framework owns this LLM model object.

        Used to detect the adapter for token tracking on the ``invoke_fn=`` path,
        where there is no top-level agent — only proxy-wrapped model objects.
        Default: False (opt in per framework).
        """
        return False

    def create_token_tracker(self, agent: Any = None) -> TokenAccumulator:
        """Set up framework-specific token tracking hooks and return an accumulator.

        Subclasses install whatever framework-specific hooks are needed
        (callbacks, patches, etc.) that feed into the returned accumulator.
        The consumer reads tokens via ``tracker.reset()``.

        Args:
            agent: The agent object (for agent= path) or None (for invoke_fn= path).

        Default: returns a bare ``TokenAccumulator`` with no hooks.
        """
        return TokenAccumulator()

    def wrap_invoke_fn_with_tracker(
        self, invoke_fn: Callable, tracker: TokenAccumulator
    ) -> Callable:
        """Wrap invoke_fn so that each call feeds token counts into *tracker*.

        Used for the sequential invoke_fn= path.
        Default: no-op — returns invoke_fn unchanged.
        """
        return invoke_fn

    def clone_for_parallel(
        self,
        agent: Any,
        proxies: List[Any],
        combo: tuple,
        get_model_name: Callable[[Any], str],
    ) -> Any:
        """Create an independent agent clone for one parallel combination.

        Default: shallow Pydantic copy (or deepcopy for non-Pydantic).
        Framework adapters that embed the LLM structurally (e.g. LangChain)
        must override this to rebuild the agent with a fresh LLM.

        Args:
            agent: The original top-level agent.
            proxies: All ``ModelProxy`` instances for this selection run.
            combo: The model specs (one per proxy) for this combination.
            get_model_name: Callable to extract a display name from a spec.

        Returns:
            An independent copy of *agent* configured for this combination.
        """
        try:
            from pydantic import BaseModel

            if isinstance(agent, BaseModel):
                return agent.model_copy(deep=False)
        except ImportError:
            pass
        return copy.deepcopy(agent)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: List[FrameworkAdapter] = []


def register_adapter(adapter: FrameworkAdapter) -> None:
    """Add *adapter* to the global registry.  Call at module level."""
    _REGISTRY.append(adapter)


def get_adapter(agent: Any) -> Optional[FrameworkAdapter]:
    """Return the first adapter whose ``detect(agent)`` returns True."""
    for adapter in _REGISTRY:
        if adapter.detect(agent):
            return adapter
    return None


def get_adapter_for_model(model: Any) -> Optional[FrameworkAdapter]:
    """Return the first adapter whose ``detect_model(model)`` returns True.

    Used to find the adapter for a raw LLM model object (not a top-level agent),
    for example when the ``invoke_fn=`` path is used and we need to detect the
    framework from the proxy's underlying model to enable token tracking.
    """
    for adapter in _REGISTRY:
        if adapter.detect_model(model):
            return adapter
    return None
