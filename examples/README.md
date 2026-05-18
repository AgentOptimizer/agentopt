# Examples

Two orthogonal features, two deployment modes — four quadrants.

| | Local backend (in-process proxy) | Daemon backend (`agentopt serve`) |
|---|---|---|
| **Offline model selection** | [`selection/local/`](selection/local/) | [`selection/daemon/`](selection/daemon/) |
| **Online per-call routing** | [`routing/local/`](routing/local/) | [`routing/daemon/`](routing/daemon/) |

The user-facing API is identical between local and daemon mode — setting `AGENTOPT_GATEWAY_URL` is the entire switch. See [`docs/api/router.md`](../docs/api/router.md) and [`docs/api/proxy.md`](../docs/api/proxy.md).

[`shared/`](shared/) holds adapters reused by more than one example (e.g. the `OpenClawAgent` wrapper used by both selection and routing).

## Selection vs. routing — when to use which

- **Selection** picks **one** combination of models and evaluates it over a whole dataset. The decision is made once, before the run.
- **Routing** picks a model **per individual LLM call** at request time, based on prompt content, session history, or a fixed policy.

They're orthogonal and compose: a `ModelSelector` could evaluate combinations that each include a router.

## Prerequisites

Most examples need:

```bash
pip install "agentopt-py[examples]"
export OPENAI_API_KEY=...   # or the relevant provider key
```

Daemon-mode examples need a daemon running in a separate terminal:

```bash
agentopt serve                            # default port 9000
# in another shell:
export AGENTOPT_GATEWAY_URL=http://127.0.0.1:9000
```
