# LlamaIndex Example

Using AgentOpt with LlamaIndex LLMs.

```python
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from agentopt import BruteForceModelSelector

def agent_maker(models):
    def run(input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]

        planner = LlamaOpenAI(model=models["planner"])
        solver = LlamaOpenAI(model=models["solver"])

        plan = planner.complete(f"Create a brief plan to answer: {question}").text

        answer = solver.complete(
            f"Follow this plan and answer concisely:\n{plan}\n\nQuestion: {question}"
        ).text
        return answer
    return run

selector = BruteForceModelSelector(
    agent_fn=agent_maker,
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

See full example: [examples/llamaindex_example.py](https://github.com/AgentOptimizer/agentopt/blob/main/examples/llamaindex_example.py)
