"""Tests for LLMTracker.register_provider covering Direct + CONNECT modes."""

import httpx
import pytest

from agentopt.proxy import LLMTracker
from agentopt.proxy.interceptor import _is_llm_request


@pytest.fixture
def tracker():
    t = LLMTracker(cache=False, cache_dir=None)
    t.start()
    yield t
    t.stop()


def test_register_provider_adds_hostname_to_intercept_set(tracker):
    """register_provider should extend both the provider registry and
    the CONNECT intercept set."""
    assert not tracker._registry.should_intercept("openrouter.ai")

    tracker.register_provider(
        name="openrouter",
        base_url="https://openrouter.ai",
        path_patterns=("/api/v1/chat/completions",),
    )

    # Path-pattern registry updated (in-process httpx wrapper detects it).
    assert "openrouter" in tracker._registry.providers
    # Hostname added to intercept set (subprocess CONNECT path).
    assert tracker._registry.should_intercept("openrouter.ai")


def test_register_provider_strips_port_and_scheme(tracker):
    """base_url with port / path / trailing slash should still resolve to hostname."""
    tracker.register_provider(
        name="local-vllm",
        base_url="http://localhost:8000/v1",
        path_patterns=("/chat/completions",),
    )
    assert tracker._registry.should_intercept("localhost")


def test_unknown_host_not_intercepted(tracker):
    """Hosts not registered should not be intercepted — they pass through."""
    assert not tracker._registry.should_intercept("example.com")
    assert not tracker._registry.should_intercept("random.site.net")


def test_default_hosts_still_intercepted(tracker):
    """Built-in providers stay intercepted after init."""
    assert tracker._registry.should_intercept("api.openai.com")
    assert tracker._registry.should_intercept("api.anthropic.com")
    assert tracker._registry.should_intercept("generativelanguage.googleapis.com")
    assert tracker._registry.should_intercept("cloudcode-pa.googleapis.com")


def test_wildcard_patterns_work(tracker):
    """Wildcard patterns (e.g. Azure OpenAI) still match after refactor."""
    assert tracker._registry.should_intercept("my-resource.openai.azure.com")
    assert tracker._registry.should_intercept("bedrock-runtime.us-east-1.amazonaws.com")


def test_instance_isolation(tracker):
    """register_provider on one tracker must not leak into a fresh one."""
    tracker.register_provider(
        name="openrouter",
        base_url="https://openrouter.ai",
        path_patterns=("/api/v1/chat/completions",),
    )
    assert tracker._registry.should_intercept("openrouter.ai")

    other = LLMTracker(cache=False, cache_dir=None)
    other.start()
    try:
        assert not other._registry.should_intercept("openrouter.ai")
    finally:
        other.stop()


def test_register_provider_updates_interceptor_paths(tracker):
    """register_provider must extend the httpx-patch path set too.

    Without this, in-process calls to a custom provider's path would not
    match ``_is_llm_request`` and the proxy would never see them — silent
    loss of tracking.
    """
    custom_path = "/openrouter-custom/respond"
    request = httpx.Request(
        method="POST", url=f"https://openrouter.ai{custom_path}", json={"prompt": "hi"},
    )
    assert not _is_llm_request(request, tracker._registry.path_patterns)

    tracker.register_provider(
        name="openrouter",
        base_url="https://openrouter.ai",
        path_patterns=(custom_path,),
    )
    assert _is_llm_request(request, tracker._registry.path_patterns)


def test_register_provider_paths_isolated_between_instances(tracker):
    """Path patterns registered on one tracker must not leak to another."""
    custom_path = "/openrouter-custom/respond"
    tracker.register_provider(
        name="openrouter",
        base_url="https://openrouter.ai",
        path_patterns=(custom_path,),
    )
    assert custom_path in tracker._registry.path_patterns

    other = LLMTracker(cache=False, cache_dir=None)
    other.start()
    try:
        assert custom_path not in other._registry.path_patterns
    finally:
        other.stop()


def test_register_provider_records_in_process_call(mock_upstream):
    """End-to-end: a custom provider's call is rewritten and recorded."""
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        # Custom path that none of the built-ins cover.
        tracker.register_provider(
            name="local",
            base_url=mock_upstream.base_url,
            path_patterns=("/custom/llm/v9/respond",),
        )
        with tracker.track(data_id="dp", combo_id="c"):
            with httpx.Client(base_url=mock_upstream.base_url) as client:
                resp = client.post(
                    "/custom/llm/v9/respond",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert resp.status_code == 200

        records = tracker.get_records()
        assert len(records) == 1
        assert records[0].request_url.endswith("/custom/llm/v9/respond")
        assert records[0].status_code == 200
    finally:
        tracker.stop()
