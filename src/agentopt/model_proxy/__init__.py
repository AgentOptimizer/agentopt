"""Model proxy package: transparent LLM wrapper with model swapping."""

from .proxy import ModelProxy

# proxy.py → builders.py transitively imports crewai and langchain,
# so their adapters self-register automatically.  Eagerly import
# remaining optional frameworks to populate the adapter registry.
for _mod in ("llamaindex", "openai_sdk", "ag2"):
    try:
        __import__(
            f"{__name__}.framework_specific_implementation.{_mod}", fromlist=["_"]
        )
    except Exception:
        pass

# Apply one-time class-level patches (ABC registration, method patching)
# for each adapter that needs them.
from .adapter import _REGISTRY

for _adapter in _REGISTRY:
    try:
        type(_adapter).patch_proxy_class(ModelProxy)
    except Exception:
        pass

__all__ = ["ModelProxy"]
