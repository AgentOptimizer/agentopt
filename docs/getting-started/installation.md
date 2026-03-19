# Installation

## Basic Install

```bash
pip install agentopt
```

## Optional Dependencies

```bash
# Bayesian optimization (requires PyTorch)
pip install "agentopt[bayesian]"

# All example dependencies
pip install "agentopt[examples]"
```

## Development Install

```bash
git clone https://github.com/AgentOptimizer/agentopt.git
cd agentopt
uv sync --extra dev
uv run pytest
```

## Requirements

- Python >= 3.10
- An LLM API key (e.g., `OPENAI_API_KEY`)
