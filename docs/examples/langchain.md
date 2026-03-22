# LangChain

Using AgentOpt with LangChain's `ChatOpenAI` model instances.

!!! info "Model objects, not strings"
    With LangChain, the `models` dict contains `ChatOpenAI` instances rather than plain strings.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from agentopt import BruteForceModelSelector

class LangChainAgent:
    def __init__(self, models):
        self.planner = models["planner"]  # ChatOpenAI instance
        self.solver = models["solver"]    # ChatOpenAI instance

    def run(self, input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]

        # Step 1: Plan
        plan = self.planner.invoke([
            SystemMessage(content="Create a brief plan to answer the question."),
            HumanMessage(content=question),
        ]).content

        # Step 2: Solve
        answer = self.solver.invoke([
            SystemMessage(content=f"Follow this plan and answer concisely:\n{plan}"),
            HumanMessage(content=question),
        ]).content
        return answer

def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0

dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
]

selector = BruteForceModelSelector(
    agent=LangChainAgent,
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
