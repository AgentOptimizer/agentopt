"""Tests for provider auto-detection and target resolution."""

from agentopt.proxy.providers import detect_provider, resolve_target


def test_detect_provider_gemini_cli_oauth_path():
    provider = detect_provider("/v1internal:generateContent")
    assert provider is not None
    assert provider.name == "gemini-cli-oauth"
    assert provider.base_url == "https://cloudcode-pa.googleapis.com"


def test_resolve_target_gemini_cli_oauth_path_without_header():
    target_base, upstream_path = resolve_target(
        "/v1internal:streamGenerateContent",
        headers={},
    )
    assert target_base == "https://cloudcode-pa.googleapis.com"
    assert upstream_path == "/v1internal:streamGenerateContent"


def test_detect_provider_sutando_docs_path():
    provider = detect_provider("/docs/collections")
    assert provider is not None
    assert provider.name == "sutando"
    assert provider.base_url == "https://sutando.org"


def test_resolve_target_sutando_docs_path_without_header():
    target_base, upstream_path = resolve_target(
        "/docs/query-builder",
        headers={},
    )
    assert target_base == "https://sutando.org"
    assert upstream_path == "/docs/query-builder"


def test_sutando_endpoint_coverage():
    for path in (
        "/docs",
        "/docs/installation",
        "/docs/eloquent",
        "/docs/query-builder",
        "/docs/migrations",
        "/docs/relationships",
        "/docs/collections",
        "/docs/pagination",
    ):
        provider = detect_provider(path)
        assert provider is not None
        assert provider.name == "sutando"
