"""Provider registry and auto-detection for LLM API routing."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a known LLM API provider."""

    name: str
    base_url: str
    path_patterns: Tuple[str, ...]


# Built-in providers — used for subprocess agent auto-detection.
DEFAULT_PROVIDERS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="openai",
        base_url="https://api.openai.com",
        path_patterns=("/v1/chat/completions", "/chat/completions", "/v1/responses",),
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        base_url="https://api.anthropic.com",
        path_patterns=("/v1/messages",),
    ),
    "google": ProviderConfig(
        name="google",
        base_url="https://generativelanguage.googleapis.com",
        path_patterns=("/v1beta/models", "/v1/models",),
    ),
}

# Header name used by the httpx monkey-patch to pass the original base URL.
TARGET_HEADER = "x-agentopt-target"


def detect_provider(
    path: str, providers: Optional[Dict[str, ProviderConfig]] = None,
) -> Optional[ProviderConfig]:
    """Match a request path against registered provider patterns.

    Returns the first matching ``ProviderConfig``, or ``None``.
    """
    if providers is None:
        providers = DEFAULT_PROVIDERS

    for provider in providers.values():
        for pattern in provider.path_patterns:
            if pattern in path:
                return provider
    return None


def resolve_target(
    path: str,
    headers: Dict[str, str],
    providers: Optional[Dict[str, ProviderConfig]] = None,
) -> Tuple[str, str]:
    """Determine the upstream base URL and path for a proxied request.

    Resolution order:
    1. Path-pattern auto-detection against the **provider registry**.
       This is authoritative: if a provider is registered for this path
       pattern, its ``base_url`` is used even when an
       ``X-AgentOpt-Target`` header is present.  This lets tests (and
       production configs) override where traffic for a given API path
       is forwarded.
    2. ``X-AgentOpt-Target`` header — fallback for providers that are
       **not** in the registry (custom / unknown endpoints).

    Returns
    -------
    (target_base_url, upstream_path)
        *target_base_url* is e.g. ``"https://api.openai.com"``.
        *upstream_path* is e.g. ``"/v1/chat/completions"``.

    Raises
    ------
    ValueError
        If the provider cannot be resolved by either mechanism.
    """
    if providers is None:
        providers = DEFAULT_PROVIDERS

    # 1. Registered provider (authoritative — allows overrides)
    provider = detect_provider(path, providers)
    if provider is not None:
        return provider.base_url, path

    # 2. Explicit header (fallback for unregistered / custom providers)
    target = headers.get(TARGET_HEADER)
    if target:
        return target.rstrip("/"), path

    raise ValueError(
        f"Cannot resolve LLM provider for path {path!r}. "
        "Either register the provider via register_provider() or "
        "set the X-AgentOpt-Target header (in-process agents)."
    )
