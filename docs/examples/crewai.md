# CrewAI Example

Using AgentOpt with CrewAI agents and tasks.

```python
from crewai import Agent, Crew, Task
from agentopt import BruteForceModelSelector

def agent_maker(models):
    def run(input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]

        planner = Agent(
            role="Planner",
            goal="Create a plan to answer the question",
            backstory="You are a planning assistant.",
            llm=models["planner"],
        )
        solver = Agent(
            role="Solver",
            goal="Answer the question following the plan",
            backstory="You are a problem solver.",
            llm=models["solver"],
        )

        plan_task = Task(description=f"Plan: {question}", agent=planner, expected_output="A plan")
        solve_task = Task(description=question, agent=solver, expected_output="An answer")

        crew = Crew(agents=[planner, solver], tasks=[plan_task, solve_task])
        result = crew.kickoff()
        return str(result)
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

See full example: [examples/crewai_example.py](https://github.com/AgentOptimizer/agentopt/blob/main/examples/crewai_example.py)
