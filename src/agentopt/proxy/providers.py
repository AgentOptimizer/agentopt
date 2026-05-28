"""LLM provider catalog: hostnames, base URLs, path patterns.

A provider tells the two interception paths what to look for:

* its **hostname** — the mitmproxy addon's ``tls_clienthello`` hook uses
  this to decide whether to TLS-terminate the subprocess's HTTPS, or
  pass it through encrypted.
* its **path patterns** — the in-process httpx wrapper uses these to
  detect LLM requests it should record.

``ProviderRegistry`` owns the catalog per ``LLMTracker`` instance, so a
``register_provider`` call on one tracker doesn't leak to another.

The free functions ``detect_provider`` / ``resolve_target`` /
``should_intercept`` operate against the built-in defaults and exist
mainly as a stable read-only API for tests and external callers.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Set, Tuple
from urllib.parse import urlparse


# Legacy header from the previous "rewrite to localhost" interceptor
# design.  Kept only because ``resolve_target`` exposes it for callers
# that may still set it; the current httpx wrapper records directly
# without rewriting.
TARGET_HEADER = "x-agentopt-target"


@dataclass(frozen=True)
class Provider:
    """An LLM API provider known to the proxy."""

    name: str
    base_url: str
    path_patterns: Tuple[str, ...]

    @property
    def hostname(self) -> str:
        return urlparse(self.base_url).hostname or ""


# Backward-compat alias.
ProviderConfig = Provider


# Built-in providers — the proxy MITMs their hostnames and auto-detects
# their request paths.
_BUILTIN_PROVIDERS: Tuple[Provider, ...] = (
    Provider(
        name="openai",
        base_url="https://api.openai.com",
        path_patterns=("/v1/chat/completions", "/chat/completions", "/v1/responses",),
    ),
    Provider(
        name="anthropic",
        base_url="https://api.anthropic.com",
        path_patterns=("/v1/messages",),
    ),
    Provider(
        name="google",
        base_url="https://generativelanguage.googleapis.com",
        path_patterns=("/v1beta/models", "/v1/models"),
    ),
    Provider(
        name="gemini-cli-oauth",
        base_url="https://cloudcode-pa.googleapis.com",
        path_patterns=(
            "/v1internal:generateContent",
            "/v1internal:streamGenerateContent",
        ),
    ),
)

# Extra LLM hostnames the proxy MITMs in CONNECT mode but has no in-process
# path patterns for — they're only ever hit by subprocess agents going
# through HTTPS_PROXY.
_EXTRA_INTERCEPT_HOSTS: Tuple[str, ...] = (
    "api.mistral.ai",
    "api.groq.com",
    "api.together.xyz",
    "api.deepseek.com",
)

# Wildcard hostname patterns (fnmatch).  Used for providers whose
# concrete hostname varies per deployment.
_INTERCEPT_PATTERNS: Tuple[str, ...] = (
    "bedrock-runtime.*.amazonaws.com",
    "*.openai.azure.com",
)


class ProviderRegistry:
    """Per-instance LLM provider catalog used by ``LLMTracker``.

    Holds one mutable copy of the built-in providers plus the intercept
    set used by the mitmproxy addon's ``tls_clienthello`` hook.  Each
    ``LLMTracker`` instance gets its own.
    """

    def __init__(self) -> None:
        self.providers: Dict[str, Provider] = {p.name: p for p in _BUILTIN_PROVIDERS}
        self._intercept_hosts: Set[str] = set(_EXTRA_INTERCEPT_HOSTS) | {
            p.hostname for p in self.providers.values() if p.hostname
        }
        self._intercept_patterns = list(_INTERCEPT_PATTERNS)
        # Aggregate of all providers' path_patterns — consulted by the
        # in-process httpx wrapper.  Rebuilt-on-write so the wrapper can
        # iterate it lock-free.
        self.path_patterns: FrozenSet[str] = frozenset(
            pattern
            for provider in self.providers.values()
            for pattern in provider.path_patterns
        )

    def register(
        self, name: str, base_url: str, path_patterns: Tuple[str, ...],
    ) -> None:
        """Add or replace a provider.

        Updates the path-pattern registry (direct mode) and the hostname
        intercept set (CONNECT mode).
        """
        provider = Provider(name=name, base_url=base_url, path_patterns=path_patterns)
        self.providers[name] = provider
        if provider.hostname:
            self._intercept_hosts.add(provider.hostname)
        self.path_patterns = frozenset(
            pattern for p in self.providers.values() for pattern in p.path_patterns
        )

    def detect(self, path: str) -> Optional[Provider]:
        """First provider whose path pattern is a substring of *path*."""
        for provider in self.providers.values():
            for pattern in provider.path_patterns:
                if pattern in path:
                    return provider
        return None

    def resolve_target(self, path: str, headers: Dict[str, str],) -> Tuple[str, str]:
        """Determine ``(upstream_base_url, upstream_path)`` for a request.

        Resolution order:
        1. ``X-AgentOpt-Target`` header (set by the httpx monkey-patch).
        2. Path-pattern auto-detection.

        Raises ``ValueError`` if neither resolves.
        """
        target = headers.get(TARGET_HEADER)
        if target:
            return target.rstrip("/"), path

        provider = self.detect(path)
        if provider is not None:
            return provider.base_url, path

        raise ValueError(
            f"Cannot resolve LLM provider for path {path!r}. "
            "Either set the X-AgentOpt-Target header (in-process) or "
            "register the provider via register_provider()."
        )

    def should_intercept(self, hostname: str) -> bool:
        """True if CONNECT traffic to *hostname* should be TLS-terminated."""
        if hostname in self._intercept_hosts:
            return True
        return any(fnmatch.fnmatch(hostname, pat) for pat in self._intercept_patterns)


# ---------------------------------------------------------------------------
# Module-level read-only API — used by tests and as a stable public surface.
# These reflect only the built-ins; calls to ``register_provider`` on a
# tracker do NOT mutate them.
# ---------------------------------------------------------------------------

DEFAULT_PROVIDERS: Dict[str, Provider] = {p.name: p for p in _BUILTIN_PROVIDERS}

INTERCEPT_HOSTS: Set[str] = set(_EXTRA_INTERCEPT_HOSTS) | {
    p.hostname for p in _BUILTIN_PROVIDERS if p.hostname
}

INTERCEPT_PATTERNS: Tuple[str, ...] = _INTERCEPT_PATTERNS


def detect_provider(
    path: str, providers: Optional[Dict[str, Provider]] = None,
) -> Optional[Provider]:
    """Match a path against the built-in providers (or a custom dict)."""
    catalog = providers if providers is not None else DEFAULT_PROVIDERS
    for provider in catalog.values():
        for pattern in provider.path_patterns:
            if pattern in path:
                return provider
    return None


def resolve_target(
    path: str, headers: Dict[str, str], providers: Optional[Dict[str, Provider]] = None,
) -> Tuple[str, str]:
    """Free-function form of ``ProviderRegistry.resolve_target``."""
    target = headers.get(TARGET_HEADER)
    if target:
        return target.rstrip("/"), path
    provider = detect_provider(path, providers)
    if provider is not None:
        return provider.base_url, path
    raise ValueError(
        f"Cannot resolve LLM provider for path {path!r}. "
        "Either set the X-AgentOpt-Target header (in-process) or "
        "register the provider via register_provider()."
    )


def should_intercept(hostname: str) -> bool:
    """Built-in form of ``ProviderRegistry.should_intercept``."""
    if hostname in INTERCEPT_HOSTS:
        return True
    return any(fnmatch.fnmatch(hostname, pat) for pat in INTERCEPT_PATTERNS)
