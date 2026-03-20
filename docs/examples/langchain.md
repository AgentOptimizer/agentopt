# LangChain

Using AgentOpt with LangChain's `ChatOpenAI` model instances.

!!! info "Model objects, not strings"
    With LangChain, the `models` dict contains `ChatOpenAI` instances rather than plain strings.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from agentopt import BruteForceModelSelector

def agent_maker(models):
    def run(input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]

        planner = models["planner"]  # ChatOpenAI instance
        solver = models["solver"]    # ChatOpenAI instance

        # Step 1: Plan
        plan = planner.invoke([
            SystemMessage(content="Create a brief plan to answer the question."),
            HumanMessage(content=question),
        ]).content

        # Step 2: Solve
        answer = solver.invoke([
            SystemMessage(content=f"Follow this plan and answer concisely:\n{plan}"),
            HumanMessage(content=question),
        ]).content
        return answer
    return run

def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0

dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
]

selector = BruteForceModelSelector(
    agent_fn=agent_maker,
    models={
        "planner": [ChatOpenAI(model="gpt-4o"), ChatOpenAI(model="gpt-4o-mini")],
        "solver":  [ChatOpenAI(model="gpt-4o"), ChatOpenAI(model="gpt-4o-mini")],
    },
    eval_fn=eval_fn,
    dataset=dataset,
)

results = selector.select_best(parallel=True)
results.print_summary()
```

[:octicons-file-code-24: Full example on GitHub](https://github.com/AgentOptimizer/agentopt/blob/main/examples/langchain_example.py)
