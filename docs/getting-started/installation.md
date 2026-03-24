# Installation

## Quick Install

=== "pip"

    ```bash
    pip install agentopt-py
    ```

=== "uv"

    ```bash
    uv add agentopt
    ```

## Optional Dependencies

=== "Bayesian Optimization"

    Requires PyTorch and BoTorch:

    ```bash
    pip install "agentopt-py[bayesian]"
    ```

=== "All Extras"

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

- **Python** >= 3.10
- **LLM API key** (e.g., `OPENAI_API_KEY` set in your environment)

!!! tip "No framework dependency"
    AgentOpt has minimal core dependencies (`httpx`, `pydantic`). It works alongside whatever LLM framework you already use — no need to install framework-specific adapters.
