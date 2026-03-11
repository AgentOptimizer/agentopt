# Project Structure

```
agentopt/
├── src/agentopt/
│   ├── __init__.py              # Public API exports (ModelProxy, ModelSelector, other selectors, types, utilities)
│   ├── base_models.py           # Type aliases (EvalFn, ModelSpec, ModelsConfig)
│   ├── model_factory.py         # create_model_from_string — multi-provider LLM factory
│   ├── model_topology.py        # Model quality/speed rankings for hill climbing
│   ├── model_proxy/
│   │   ├── proxy.py             # ModelProxy — transparent proxy, set_model(), register()
│   │   ├── adapter.py           # FrameworkAdapter ABC + registry (get_adapter, register_adapter)
│   │   ├── constants.py         # Framework detection helpers + MODEL_FIELDS
│   │   ├── builders.py          # Generic LLM rebuild helpers
│   │   └── framework_specific_implementation/
│   │       ├── crewai.py        # CrewAI support + CrewAIAdapter
│   │       ├── langchain_compat.py  # LangChain support + LangChainAdapter
│   │       ├── llamaindex.py    # LlamaIndex support + LlamaIndexAdapter + build_llamaindex_llm
│   │       ├── openai_sdk.py    # OpenAI Agents SDK support + OpenAISDKAdapter
│   │       └── ag2.py           # AG2 support + AG2Adapter + _build_ag2_config
│   └── model_selection/
│       ├── base.py              # BaseModelSelector, ModelResult, SelectionResults
│       ├── brute_force.py       # BruteForceModelSelector (grid search over all combinations; default, aliased as ModelSelector)
│       ├── random_search.py     # RandomSearchModelSelector (random subset of combinations; sampled brute-force)
│       ├── hill_climbing.py     # HillClimbingModelSelector (local search / hill-climbing over combinations; experimental)
│       ├── arm_elimination.py   # ArmEliminationModelSelector (bandit-style successive elimination; experimental)
│       ├── hyperband.py         # HyperbandModelSelector (bandit-style, multi-bracket successive halving over dataset samples)
│       ├── bayesian_optimization.py  # BayesianOptimizationModelSelector (Bayesian optimization over model combinations; experimental)
│       └── utils.py             # Compat re-export of extract_prompt
├── examples/
│   ├── crewai_example.py        # CrewAI: single, multi-agent, multi-LLM
│   ├── langchain_example.py     # LangChain: single, multi-LLM
│   ├── langgraph_example.py     # LangGraph: multi-agent, multi-LLM
│   ├── llamaindex_example.py    # LlamaIndex: single, multi-agent, multi-LLM
│   ├── openai_sdk_example.py    # OpenAI SDK: single, multi-LLM
│   ├── claude_sdk_example.py    # Claude SDK: single, multi-agent, multi-LLM
│   ├── ag2_example.py           # AG2: single, multi-agent, multi-LLM
│   └── datasets/
│       └── math_problems.jsonl  # Example dataset
├── README.md                    # Documentation
├── pyproject.toml               # Package configuration
└── uv.lock                      # Dependency lock file
```
