# Routing examples

A `Router` decides which model to send each LLM call to, at request time. The whole user-facing API is one pattern:

```python
from agentopt import LLMTracker, RandomRouter

router = RandomRouter(candidates=["gpt-4o", "gpt-4o-mini"], seed=0)
with LLMTracker(combo_id="demo", router=router) as tracker:
    for dp in datapoints:
        agent.run(dp)
tracker.print_summary()    # model sequence, per-model tokens, total latency
```

Same pattern in local mode and daemon mode — the only deployment switch is whether `AGENTOPT_GATEWAY_URL` is set.

## `local/` — in-process and subprocess agents

| File | Framework / agent | Notes |
|---|---|---|
| [`custom_agent.py`](local/custom_agent.py) | Plain Python + OpenAI SDK | Each call returns `(answer, resp.model)` so the routing decision is visible. |
| [`langchain.py`](local/langchain.py) | LangChain tool-calling agent | |
| [`langgraph.py`](local/langgraph.py) | LangGraph planner+solver | Each node's LLM call is routed independently. |
| [`llamaindex.py`](local/llamaindex.py) | LlamaIndex `AgentWorkflow` | Async `run()` shape. |
| [`openai_sdk.py`](local/openai_sdk.py) | OpenAI Agents SDK | Planner+solver agents. |
| [`crewai.py`](local/crewai.py) | CrewAI researcher+writer crew | |
| [`ag2.py`](local/ag2.py) | AG2 (autogen) planner+solver | Uses `initiate_chat()` for contextvar propagation. |
| [`gemini_cli.py`](local/gemini_cli.py) | Gemini CLI subprocess | **v1 limitation**: model lives in URL path, not body — router is called but the decision passes through unrouted. |
| [`openharness.py`](local/openharness.py) | `oh` CLI subprocess | Routes via `HTTPS_PROXY`; `oh` uses official SDKs that honour the env var. |
| [`terminal_bench.py`](local/terminal_bench.py) | `tb run` subprocess | Routes LLM calls made *inside* the Docker container `tb` launches. |
| [`openclaw.py`](local/openclaw.py) | OpenClaw subprocess | Reuses the shared `OpenClawAgent` adapter at `examples/shared/openclaw_agent.py` (config-file patching). |
| [`custom_router.py`](local/custom_router.py) | Custom `Router` subclass | `LengthBasedRouter`: long prompts → big, short → small. |

Run any of them directly:

```bash
uv run python examples/routing/local/custom_agent.py
```

## `daemon/` — same scripts, behind `agentopt serve`

Set `AGENTOPT_GATEWAY_URL=http://127.0.0.1:9000` and the same Python code runs against a long-lived daemon.

| File | What it shows |
|---|---|
| [`default_policy.py`](daemon/default_policy.py) | Daemon-wide default via `--routing-policy random --candidate-models …`. The client just opens `with LLMTracker(combo_id=...)` — no router code. |
| [`per_session_override.py`](daemon/per_session_override.py) | Daemon default + client passes `router=` to `LLMTracker` to override for its own session. |
| [`custom_policy_module.py`](daemon/custom_policy_module.py) | User-supplied `Router` ([`my_policies.py`](daemon/my_policies.py)) loaded by the daemon via `--policy-module` and shared by the client. |

Each daemon example's docstring includes the exact `agentopt serve` invocation it expects.
