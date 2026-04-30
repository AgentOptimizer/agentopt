"""Coverage tests for built-in Sutando provider support."""

import httpx

from agentopt.proxy import LLMTracker
from agentopt.proxy.providers import (
    DEFAULT_PROVIDERS,
    detect_provider,
    resolve_target,
    should_intercept,
)


SUTANDO_PATHS = (
    "/docs",
    "/docs/installation",
    "/docs/eloquent",
    "/docs/query-builder",
    "/docs/migrations",
    "/docs/relationships",
    "/docs/collections",
    "/docs/pagination",
)


def test_sutando_provider_registered_with_expected_defaults():
    provider = DEFAULT_PROVIDERS.get("sutando")
    assert provider is not None
    assert provider.base_url == "https://sutando.org"
    assert provider.path_patterns == SUTANDO_PATHS


def test_sutando_host_is_intercepted_by_default():
    assert should_intercept("sutando.org")


def test_detect_provider_covers_all_sutando_paths():
    for path in SUTANDO_PATHS:
        provider = detect_provider(path)
        assert provider is not None
        assert provider.name == "sutando"


def test_resolve_target_covers_all_sutando_paths_without_header():
    for path in SUTANDO_PATHS:
        target_base, upstream_path = resolve_target(path, headers={})
        assert target_base == "https://sutando.org"
        assert upstream_path == path


def test_header_override_still_wins_over_sutando_auto_detection():
    target_base, upstream_path = resolve_target(
        "/docs/eloquent",
        headers={"x-agentopt-target": "https://custom.example.com"},
    )
    assert target_base == "https://custom.example.com"
    assert upstream_path == "/docs/eloquent"


def test_sutando_paths_are_not_rewritten_by_inprocess_llm_path_filter(mock_upstream):
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with tracker.track(data_id="dp_sutando", combo_id="sutando"):
            with httpx.Client(base_url=mock_upstream.base_url) as client:
                resp = client.post("/docs/query-builder", json={"model": "sutando"})
                assert resp.status_code == 200

        records = tracker.get_records(data_id="dp_sutando", combo_id="sutando")
        # Interceptor only rewrites configured LLM paths; docs/* is provider
        # auto-detection only and does not trigger in-process monkey-patch rewrite.
        assert records == []
    finally:
        tracker.stop()
