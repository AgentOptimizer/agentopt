# Project Structure

```
agentopt/
├── agentopt/              # Main package
│   ├── __init__.py       # Package exports
│   ├── core.py           # Core functionality (ModelProxy, ModelSelector, bind_model)
│   └── model_factory.py  # Model creation from strings
├── examples/             # Example usage
│   ├── agent.py          # Example agent implementation
│   ├── main.py           # Example model selection script
│   └── datasets/         # Example datasets
│       └── math_problems.jsonl
├── tests/                # Test directory (for future tests)
├── README.md             # Documentation
├── pyproject.toml        # Package configuration
└── uv.lock               # Dependency lock file
```
