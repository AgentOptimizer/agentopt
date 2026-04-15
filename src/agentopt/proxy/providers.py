"""Provider registry and auto-detection for LLM API routing."""

import fnmatch
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple


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

# ---------------------------------------------------------------------------
# Known LLM API hostnames — used for CONNECT-mode interception.
# Only CONNECT requests to these hosts are TLS-terminated and inspected.
# Everything else is tunneled through as a transparent passthrough.
# ---------------------------------------------------------------------------

INTERCEPT_HOSTS: Set[str] = {
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.mistral.ai",
    "api.groq.com",
    "api.together.xyz",
    "api.deepseek.com",
}

# Wildcard patterns (matched with fnmatch).
INTERCEPT_PATTERNS: Tuple[str, ...] = (
    "bedrock-runtime.*.amazonaws.com",
    "*.openai.azure.com",
)


def should_intercept(hostname: str) -> bool:
    """Return ``True`` if CONNECT traffic to *hostname* should be intercepted."""
    if hostname in INTERCEPT_HOSTS:
        return True
    return any(fnmatch.fnmatch(hostname, pat) for pat in INTERCEPT_PATTERNS)


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
    1. ``X-AgentOpt-Target`` header — set by the httpx monkey-patch for
       in-process agents.  Contains the exact original base URL, which
       may be a custom / self-hosted endpoint (e.g. Azure OpenAI).
    2. Path-pattern auto-detection against the **provider registry** —
       fallback for subprocess agents where custom headers are not
       available.

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

    # 1. Explicit header (in-process agents — exact target, supports
    #    Azure / self-hosted / custom OpenAI-compatible endpoints)
    target = headers.get(TARGET_HEADER)
    if target:
        return target.rstrip("/"), path

    # 2. Auto-detect from path pattern (subprocess agents)
    provider = detect_provider(path, providers)
    if provider is not None:
        return provider.base_url, path

    raise ValueError(
        f"Cannot resolve LLM provider for path {path!r}. "
        "Either set the X-AgentOpt-Target header (in-process) or "
        "register the provider via register_provider()."
    )
