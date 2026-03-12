# CrewAI

AgentOpt integrates with CrewAI via duck typing — `ModelProxy` can be passed directly as the `llm` parameter.

## Installation

```bash
uv sync --extra crewai
```

## Example

```python
from crewai import Agent, Task, Crew, LLM
from agentopt import ModelProxy, ModelSelector

# 1. Wrap the LLM
llm = ModelProxy(LLM(model="openai/gpt-4o-mini"))

# 2. Build your agent normally
agent = Agent(role="Researcher", goal="Answer questions", backstory="...", llm=llm)
task = Task(description="{input}", expected_output="A clear answer", agent=agent)
crew = Crew(agents=[agent], tasks=[task])

# 3. Prepare dataset
dataset = [
    ({"input": "What is 2 + 2?"}, "4"),
    ({"input": "Capital of France?"}, "Paris"),
]

# 4. Run optimization
selector = ModelSelector(
    models={llm: ["openai/gpt-4o-mini", "openai/gpt-4o"]},
    eval_fn=lambda expected, actual: expected.lower() in str(actual).lower(),
    dataset=dataset,
    agent=crew,
)
results = selector.select_best()
```

## Multi-agent

```python
researcher_llm = ModelProxy(LLM(model="openai/gpt-4o-mini"))
writer_llm = ModelProxy(LLM(model="openai/gpt-4o-mini"))

researcher = Agent(role="Researcher", llm=researcher_llm, ...)
writer = Agent(role="Writer", llm=writer_llm, ...)

selector = ModelSelector(
    models={
        researcher_llm: ["openai/gpt-4o-mini", "openai/gpt-4o"],
        writer_llm: ["openai/gpt-4o-mini", "openai/gpt-4o"],
    },
    eval_fn=eval_fn,
    dataset=dataset,
    agent=crew,
)
results = selector.select_best(parallel=True)
```

## How it works

- **Detection:** `CrewAIAdapter.detect()` checks `type(agent).__module__.startswith("crewai")`
- **Invocation:** Calls `crew.kickoff(inputs=input_data)`
- **Parallel cloning:** Uses `model_copy(deep=False)` + clones crew agents with fresh LLMs
