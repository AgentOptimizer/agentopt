"""Tests for LLM path detection in the httpx interceptor."""

import httpx

from agentopt.proxy.interceptor import _is_llm_request


def _post(url: str) -> httpx.Request:
    return httpx.Request(method="POST", url=url, json={"prompt": "hi"})


def test_gemini_cli_oauth_paths_are_treated_as_llm_calls():
    assert _is_llm_request(_post("https://cloudcode-pa.googleapis.com/v1internal:generateContent"))
    assert _is_llm_request(
        _post("https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent")
    )


def test_non_post_requests_are_not_treated_as_llm_calls():
    request = httpx.Request(method="GET", url="https://api.openai.com/v1/chat/completions")
    assert not _is_llm_request(request)
