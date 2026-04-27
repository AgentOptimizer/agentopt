"""Tests for LLMTracker.register_provider covering Direct + CONNECT modes."""

import pytest

from agentopt.proxy import LLMTracker


@pytest.fixture
def tracker():
    t = LLMTracker(cache=False, cache_dir=None)
    t.start()
    yield t
    t.stop()


def test_register_provider_adds_hostname_to_intercept_set(tracker):
    """register_provider should extend both the provider registry and
    the CONNECT intercept set."""
    assert not tracker._server._should_intercept("openrouter.ai")

    tracker.register_provider(
        name="openrouter",
        base_url="https://openrouter.ai",
        path_patterns=("/api/v1/chat/completions",),
    )

    # Direct mode: path-pattern auto-detection registry updated.
    assert "openrouter" in tracker._server._providers
    # CONNECT mode: hostname added to intercept set.
    assert tracker._server._should_intercept("openrouter.ai")


def test_register_provider_strips_port_and_scheme(tracker):
    """base_url with port / path / trailing slash should still resolve to hostname."""
    tracker.register_provider(
        name="local-vllm",
        base_url="http://localhost:8000/v1",
        path_patterns=("/chat/completions",),
    )
    assert tracker._server._should_intercept("localhost")


def test_unknown_host_not_intercepted(tracker):
    """Hosts not registered should not be intercepted — they pass through."""
    assert not tracker._server._should_intercept("example.com")
    assert not tracker._server._should_intercept("random.site.net")


def test_default_hosts_still_intercepted(tracker):
    """Built-in providers stay intercepted after init."""
    assert tracker._server._should_intercept("api.openai.com")
    assert tracker._server._should_intercept("api.anthropic.com")
    assert tracker._server._should_intercept("generativelanguage.googleapis.com")
    assert tracker._server._should_intercept("cloudcode-pa.googleapis.com")


def test_wildcard_patterns_work(tracker):
    """Wildcard patterns (e.g. Azure OpenAI) still match after refactor."""
    assert tracker._server._should_intercept("my-resource.openai.azure.com")
    assert tracker._server._should_intercept("bedrock-runtime.us-east-1.amazonaws.com")


def test_instance_isolation(tracker):
    """register_provider on one tracker must not leak into a fresh one."""
    tracker.register_provider(
        name="openrouter",
        base_url="https://openrouter.ai",
        path_patterns=("/api/v1/chat/completions",),
    )
    assert tracker._server._should_intercept("openrouter.ai")

    other = LLMTracker(cache=False, cache_dir=None)
    other.start()
    try:
        assert not other._server._should_intercept("openrouter.ai")
    finally:
        other.stop()
