# Selection examples

`ModelSelector` picks the best combination of models for a multi-step agent over a dataset. Same script runs in either mode:

- [`local/`](local/) — default; in-process mitmproxy per session.
- [`daemon/`](daemon/) — set `AGENTOPT_GATEWAY_URL` and let a shared `agentopt serve` process do the work.

## Run your own benchmark

[`benchmark_runner.py`](benchmark_runner.py) loads a JSONL file of `input` and
`expected` records, prints current recommendations as work completes, and saves
results after a cooperative Ctrl+C stop. From the repository root:

```bash
python -m pip install -e . openai
python examples/selection/benchmark_runner.py \
  --dataset examples/selection/benchmark_sample.jsonl \
  --models gpt-4o-mini gpt-4.1-nano \
  --output results.json
```

Set `OPENAI_API_KEY` first and use models available to your account; this makes
real API calls. The sample is a format illustration, not evidence of model
quality. Use `--help` to see the options, or add `--objective-mode multi` for
accuracy/cost alternatives. A stop lets scheduled work finish before saving;
there is no resumable search checkpoint.

The [user guide](../../docs/getting-started/your-benchmark.md) explains the dataset
and scorer contract, custom agents, evaluation fractions, dollar limits, live
callbacks, and [contributing an example](../../docs/getting-started/your-benchmark.md#contribute-a-benchmark-example).

## What's in `local/`

| File | What it shows |
|---|---|
| [`custom_agent.py`](local/custom_agent.py) | Plain Python + OpenAI SDK — the canonical starter. |
| [`advanced_algorithms.py`](local/advanced_algorithms.py) | Adaptive selection methods, including Gittins single and multi-objective modes. |
| [`openai_sdk.py`](local/openai_sdk.py) | OpenAI Agents SDK. |
| [`langchain.py`](local/langchain.py), [`langgraph.py`](local/langgraph.py) | LangChain / LangGraph. |
| [`llamaindex.py`](local/llamaindex.py) | LlamaIndex. |
| [`crewai.py`](local/crewai.py) | CrewAI. |
| [`ag2.py`](local/ag2.py) | AG2. |
| [`gemini_cli.py`](local/gemini_cli.py) | Subprocess agent (`gemini` CLI) — `HTTPS_PROXY` is set for you. |
| [`openharness.py`](local/openharness.py) | OpenHarness subprocess. |
| [`terminal_bench.py`](local/terminal_bench.py) | Terminal Bench subprocess. |
| [`openclaw.py`](local/openclaw.py) | OpenClaw (uses the shared `OpenClawAgent` adapter at [`../shared/openclaw_agent.py`](../shared/openclaw_agent.py)). |

## Selection modes

Install `agentopt-py` and the example's OpenAI/python-dotenv dependencies,
then provide `OPENAI_API_KEY`. These examples make real model calls:

```bash
python examples/selection/local/advanced_algorithms.py --method gittins --objective-mode single --max-search-cost-usd 1.0
python examples/selection/local/advanced_algorithms.py --method gittins --objective-mode multi --max-search-cost-usd 1.0
```

Single mode defaults to Gittins (`auto`) and also supports
`random_configurations`, `random_questions`, `ucb`, and `bayesian`.
`random` is an alias for single-objective `random_configurations`.
Multi mode supports `auto`/`gittins`, `random_configurations`, and `random_questions`.
The random-question baseline evaluates shared random questions on every configuration.

Single mode prints one best combination. Multi mode prints the terminal
recommendation set and a deployment choice under a $0.01/question cost limit.
The search cap counts new token charges including warm start; predictions
can underestimate the next batch's charge, so actual spend may overshoot.

## What's in `daemon/`

| File | What it shows |
|---|---|
| [`basic.py`](daemon/basic.py) | The exact same selector script, run against `agentopt serve` instead of an in-process proxy. The only difference is one env var. |
