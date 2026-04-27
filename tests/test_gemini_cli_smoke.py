"""Smoke test for Gemini CLI-style paths through LLMTracker proxy."""

import httpx

from agentopt.proxy import LLMTracker


def test_gemini_cli_path_is_intercepted_and_recorded(mock_upstream):
    """Gemini CLI internal path should be redirected and recorded like other LLM calls."""
    tracker = LLMTracker(cache=False, cache_dir=None)
    tracker.start()
    try:
        with tracker.track(data_id="dp_gemini", combo_id="gemini-cli"):
            # Base URL points to mock upstream. The interceptor rewrites this request
            # to the session proxy because the path matches Gemini CLI patterns.
            with httpx.Client(base_url=mock_upstream.base_url) as client:
                resp = client.post(
                    "/v1internal:generateContent",
                    json={
                        "model": "gemini-2.5-flash",
                        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                    },
                )
                assert resp.status_code == 200

        records = tracker.get_records(data_id="dp_gemini", combo_id="gemini-cli")
        assert len(records) == 1
        assert records[0].status_code == 200
        assert records[0].request_url.endswith("/v1internal:generateContent")
    finally:
        tracker.stop()
