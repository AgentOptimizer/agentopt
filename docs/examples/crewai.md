# CrewAI

Using AgentOpt with CrewAI agents and tasks.

!!! info "Model strings as `llm` parameter"
    CrewAI agents accept a model string via the `llm` parameter. Pass values from the `models` dict directly.

```python
from crewai import Agent, Crew, Task
from agentopt import BruteForceModelSelector

class CrewAgent:
    def __init__(self, models):
        self.models = models

    def run(self, input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]

        planner = Agent(
            role="Planner",
            goal="Create a plan to answer the question",
            backstory="You are a planning assistant.",
            llm=self.models["planner"],
        )
        solver = Agent(
            role="Solver",
            goal="Answer the question following the plan",
            backstory="You are a problem solver.",
            llm=self.models["solver"],
        )

        plan_task = Task(description=f"Plan: {question}", agent=planner, expected_output="A plan")
        solve_task = Task(description=question, agent=solver, expected_output="An answer")

        crew = Crew(agents=[planner, solver], tasks=[plan_task, solve_task])
        result = crew.kickoff()
        return str(result)

selector = BruteForceModelSelector(
    agent=CrewAgent,
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

[:octicons-file-code-24: Full example on GitHub](https://github.com/AgentOptimizer/agentopt/blob/main/examples/crewai_example.py)
