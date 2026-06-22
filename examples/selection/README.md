# Selection examples

`ModelSelector` picks the best combination of models for a multi-step agent over a dataset. Same script runs in either mode:

- [`local/`](local/) — default; in-process mitmproxy per session.
- [`daemon/`](daemon/) — set `AGENTOPT_GATEWAY_URL` and let a shared `agentopt serve` process do the work.

## What's in `local/`

| File | What it shows |
|---|---|
| [`custom_agent.py`](local/custom_agent.py) | Plain Python + OpenAI SDK — the canonical starter. |
| [`advanced_algorithms.py`](local/advanced_algorithms.py) | Every `method=` available on `ModelSelector` (auto, arm_elimination, matrix_ucb, bayesian, …). |
| [`openai_sdk.py`](local/openai_sdk.py) | OpenAI Agents SDK. |
| [`langchain.py`](local/langchain.py), [`langgraph.py`](local/langgraph.py) | LangChain / LangGraph. |
| [`llamaindex.py`](local/llamaindex.py) | LlamaIndex. |
| [`crewai.py`](local/crewai.py) | CrewAI. |
| [`ag2.py`](local/ag2.py) | AG2. |
| [`gemini_cli.py`](local/gemini_cli.py) | Subprocess agent (`gemini` CLI) — `HTTPS_PROXY` is set for you. |
| [`openharness.py`](local/openharness.py) | OpenHarness subprocess. |
| [`terminal_bench.py`](local/terminal_bench.py) | Terminal Bench subprocess. |
| [`openclaw.py`](local/openclaw.py) | OpenClaw (uses the shared `OpenClawAgent` adapter at [`../shared/openclaw_agent.py`](../shared/openclaw_agent.py)). |

## What's in `daemon/`

| File | What it shows |
|---|---|
| [`basic.py`](daemon/basic.py) | The exact same selector script, run against `agentopt serve` instead of an in-process proxy. The only difference is one env var. |
