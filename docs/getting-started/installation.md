# Installation

AgentOpt uses [uv](https://docs.astral.sh/uv/) for dependency management.

## Base install

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync
git clone https://github.com/TianyiPeng/agentopt.git
cd agentopt
uv sync
```

## Framework extras

Install only the frameworks you need:

```bash
uv sync --extra crewai           # CrewAI
uv sync --extra llamaindex       # LlamaIndex
uv sync --extra ag2              # AG2 (AutoGen 2)
uv sync --extra openai-agents    # OpenAI Agents SDK
uv sync --extra claude-agent-sdk # Claude Agent SDK
uv sync --extra bayesian         # Bayesian optimization (requires PyTorch)
```

## Environment setup

Set API keys for the providers you want to use:

```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here   # for claude-* models
export GOOGLE_API_KEY=your_key_here      # for gemini-* models
```

Or create a `.env` file (AgentOpt loads it automatically via `python-dotenv`):

```bash
cp .env.example .env
# Edit .env with your keys
```

## Verify installation

```python
from agentopt import ModelProxy, ModelSelector
print("AgentOpt installed successfully!")
```
