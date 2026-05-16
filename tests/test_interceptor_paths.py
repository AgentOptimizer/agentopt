"""Tests for LLM path detection in the httpx interceptor."""

import httpx

from agentopt.proxy.interceptor import _is_llm_request
from agentopt.proxy.providers import ProviderRegistry


_DEFAULT_PATTERNS = ProviderRegistry().path_patterns


def _post(url: str) -> httpx.Request:
    return httpx.Request(method="POST", url=url, json={"prompt": "hi"})


def test_gemini_cli_oauth_paths_are_treated_as_llm_calls():
    assert _is_llm_request(
        _post("https://cloudcode-pa.googleapis.com/v1internal:generateContent"),
        _DEFAULT_PATTERNS,
    )
    assert _is_llm_request(
        _post("https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent"),
        _DEFAULT_PATTERNS,
    )


def test_non_post_requests_are_not_treated_as_llm_calls():
    request = httpx.Request(
        method="GET", url="https://api.openai.com/v1/chat/completions"
    )
    assert not _is_llm_request(request, _DEFAULT_PATTERNS)
