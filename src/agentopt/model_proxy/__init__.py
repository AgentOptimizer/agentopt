"""Model proxy package: transparent LLM wrapper with model swapping."""

from .proxy import ModelProxy

# crewai.py and langchain.py are already imported transitively via
# proxy.py → builders.py, so their adapters self-register automatically.

# llamaindex is an optional extra — import eagerly but swallow ImportError.
try:
    from .framework_specific_implementation import (
        llamaindex as _,
    )  # noqa: F401 — registers LlamaIndexAdapter
except ImportError:
    pass

# OpenAI Agents SDK: register the ABC virtual subclass AND the adapter.
try:
    from .framework_specific_implementation.openai_sdk import (
        register_openai_agents_model,
    )

    register_openai_agents_model(ModelProxy)
    # OpenAISDKAdapter self-registers at the bottom of openai_sdk.py,
    # so no explicit register_adapter() call is needed here.
except Exception:
    # Swallow any optional-dep issues (e.g., OpenAI Agents SDK pulling old TF).
    pass

try:
    from .framework_specific_implementation.ag2 import register_ag2_llm_config

    register_ag2_llm_config(ModelProxy)
except Exception:
    # Swallow any optional-dep issues (autogen not installed).
    pass

__all__ = ["ModelProxy"]
