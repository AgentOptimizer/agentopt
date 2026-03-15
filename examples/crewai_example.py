"""
Example: CrewAI agent with agentopt + LiteLLM.

Prerequisites:
    1. pip install crewai agentopt
    2. Start LiteLLM proxy: litellm --config litellm_config.yaml --port 4000
"""

import argparse
import os
from typing import Dict

from crewai import Agent, Crew, LLM, Task

from agentopt import (
    ArmEliminationModelSelector,
    BruteForceModelSelector,
    HillClimbingModelSelector,
    HyperbandModelSelector,
    RandomSearchModelSelector,
)

SELECTORS = {
    "brute_force": BruteForceModelSelector,
    "random": RandomSearchModelSelector,
    "hill_climbing": HillClimbingModelSelector,
    "arm_elimination": ArmEliminationModelSelector,
    "hyperband": HyperbandModelSelector,
}

LITELLM_BASE_URL = "http://localhost:4000/v1"


def agent_maker(models: Dict[str, str]):
    """Factory: builds a CrewAI crew with researcher + writer agents."""
    researcher = Agent(
        role="Researcher",
        goal="Research the topic and provide accurate information",
        backstory="You are a knowledgeable researcher.",
        llm=LLM(
            model=f"openai/{models['researcher']}",
            base_url=LITELLM_BASE_URL,
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
    )
    writer = Agent(
        role="Writer",
        goal="Write a concise answer based on research",
        backstory="You are a skilled writer who distills information.",
        llm=LLM(
            model=f"openai/{models['writer']}",
            base_url=LITELLM_BASE_URL,
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
    )

    def run(input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]

        research_task = Task(
            description=f"Research this question: {question}",
            expected_output="Factual information about the topic",
            agent=researcher,
        )
        write_task = Task(
            description=f"Write a concise answer to: {question}",
            expected_output="A clear, concise answer",
            agent=writer,
        )

        crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task],)
        result = crew.kickoff()
        return str(result)

    return run


def eval_fn(expected: str, actual) -> float:
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
]


def main():
    parser = argparse.ArgumentParser(description="CrewAI model selection example")
    parser.add_argument("--selector", choices=SELECTORS, default="brute_force")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=20)
    args = parser.parse_args()

    selector_cls = SELECTORS[args.selector]
    selector = selector_cls(
        agent_fn=agent_maker,
        models={
            "researcher": ["gpt-5.1", "gpt-4o-mini", "gpt-4.1"],
            "writer": ["gpt-5.1", "gpt-4o-mini", "gpt-4.1"],
        },
        eval_fn=eval_fn,
        dataset=dataset,
    )

    results = selector.select_best(
        parallel=args.parallel, max_concurrent=args.max_concurrent
    )
    results.print_summary()

    best = results.get_best_combo()
    if best:
        print(f"\nBest combination: {best}")


if __name__ == "__main__":
    main()
