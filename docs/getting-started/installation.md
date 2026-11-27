# Installation

## Quick Install

=== "pip"

    ```bash
    pip install agentopt-py
    ```

=== "uv"

    ```bash
    uv add agentopt-py
    ```

Gittins, UCB, and random selection are included in the standard installation.
SciPy is installed automatically for Gittins’ normal-distribution calculations
and numerical root finding.

## Optional Dependencies

=== "Bayesian Optimization"

    Requires PyTorch and BoTorch:

    ```bash
    pip install "agentopt-py[bayesian]"
    ```

## Development Setup

```bash
git clone https://github.com/AgentOptimizer/agentopt.git
cd agentopt
uv sync --extra dev
uv run pytest
```

## Requirements

- **Python** >= 3.12
- **LLM API key** (e.g., `OPENAI_API_KEY` set in your environment)

!!! tip "No framework dependency"
    AgentOpt uses `httpx`, `pydantic`, `aiohttp`, `mitmproxy`, NumPy, and SciPy. It works alongside whatever LLM framework you already use — no need to install framework-specific adapters.
