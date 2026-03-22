# LlamaIndex

Using AgentOpt with LlamaIndex LLM instances.

!!! info "LlamaIndex LLM objects"
    Similar to LangChain, pass LlamaIndex `OpenAI` instances in the `models` dict.

```python
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from agentopt import BruteForceModelSelector

class MyAgent:
    def __init__(self, models):
        self.models = models

    def run(self, input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]

        planner = LlamaOpenAI(model=self.models["planner"])
        solver = LlamaOpenAI(model=self.models["solver"])

        plan = planner.complete(f"Create a brief plan to answer: {question}").text

        answer = solver.complete(
            f"Follow this plan and answer concisely:\n{plan}\n\nQuestion: {question}"
        ).text
        return answer

selector = BruteForceModelSelector(
    agent=MyAgent,
    models={
        "planner": ["gpt-4o", "gpt-4o-mini"],
        "solver":  ["gpt-4o", "gpt-4o-mini"],
    },
    eval_fn=eval_fn,
    dataset=dataset,
)

results = selector.select_best()
results.print_summary()
```

[:octicons-file-code-24: Full example on GitHub](https://github.com/AgentOptimizer/agentopt/blob/main/examples/llamaindex_example.py)
