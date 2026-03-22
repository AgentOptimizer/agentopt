"""
Example: CrewAI agent with agentopt.

Prerequisites:
    1. pip install crewai agentopt
    2. Set OPENAI_API_KEY environment variable
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import inspect
from typing import Any, Dict

from crewai import Agent, Crew, LLM, Task

from agentopt import (
    ArmEliminationModelSelector,
    BruteForceModelSelector,
    EpsilonLUCBModelSelector,
    HillClimbingModelSelector,
    LMProposalModelSelector,
    RandomSearchModelSelector,
    ThresholdBanditSEModelSelector,
)

SELECTORS = {
    "brute_force": BruteForceModelSelector,
    "random": RandomSearchModelSelector,
    "hill_climbing": HillClimbingModelSelector,
    "arm_elimination": ArmEliminationModelSelector,
    "epsilon_lucb": EpsilonLUCBModelSelector,
    "threshold_successive_elimination": ThresholdBanditSEModelSelector,
    "lm_proposal": LMProposalModelSelector,
}

try:
    from agentopt import BayesianOptimizationModelSelector

    SELECTORS["bayesian_optimization"] = BayesianOptimizationModelSelector
except ImportError:
    pass


class MyAgent:
    """CrewAI crew with researcher + writer agents."""

    def __init__(self, models: Dict[str, Any]):
        self.researcher_llm = (
            models["researcher"]
            if not isinstance(models["researcher"], str)
            else LLM(model=models["researcher"])
        )
        self.writer_llm = (
            models["writer"]
            if not isinstance(models["writer"], str)
            else LLM(model=models["writer"])
        )

    def run(self, input_data):
        question = input_data if isinstance(input_data, str) else input_data["question"]

        researcher = Agent(
            role="Researcher",
            goal="Research the topic and provide accurate information",
            backstory="You are a knowledgeable researcher.",
            llm=self.researcher_llm,
        )
        writer = Agent(
            role="Writer",
            goal="Write a concise answer based on research",
            backstory="You are a skilled writer who distills information.",
            llm=self.writer_llm,
        )

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

        crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
        result = crew.kickoff()
        return str(result)


def eval_fn(expected: str, actual) -> float:
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


dataset = [
    # Easy – every combo should get these
    ("What is 7 * 8?", "56"),
    ("What is the derivative of x^3?", "3x^2"),
    # Medium
    ("What is the integral of 1/(1+x^2) dx?", "arctan"),
    ("If log base 2 of x equals 5, what is x?", "32"),
    # Hard – weaker combos likely fail
    (
        "What is the sum of the series 1/1! + 1/2! + 1/3! + ... + 1/10! "
        "rounded to 6 decimal places?",
        "1.718282",
    ),
    (
        "A bag has 5 red and 3 blue balls. Two are drawn without replacement. "
        "What is the probability both are red? Give the fraction.",
        "5/14",
    ),
    ("Find the remainder when 2^100 is divided by 7.", "2",),
    ("What is the determinant of the matrix [[1,2,3],[4,5,6],[7,8,9]]?", "0",),
]


def _filter_selector_kwargs(
    selector_cls, selector_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    params = inspect.signature(selector_cls.__init__).parameters
    return {k: v for k, v in selector_kwargs.items() if k in params}


def main():
    parser = argparse.ArgumentParser(description="CrewAI model selection example")
    parser.add_argument("--selector", choices=SELECTORS, default="brute_force")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument(
        "--use-instances",
        action="store_true",
        help="Pass pre-built LLM instances instead of model name strings",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.25,
        help="Fraction of combinations to evaluate when --selector=random",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Batch size for batched selectors "
            "(hill_climbing neighbours and bayesian_optimization candidates)."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Epsilon for --selector=epsilon_lucb.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for --selector=threshold_successive_elimination.",
    )
    args = parser.parse_args()

    candidates = ["gpt-5.2", "gpt-4o-mini", "gpt-4.1"]
    if args.use_instances:
        models = {
            "researcher": [LLM(model=m) for m in candidates],
            "writer": [LLM(model=m) for m in candidates],
        }
    else:
        models = {"researcher": candidates, "writer": candidates}

    selector_cls = SELECTORS[args.selector]
    selector_kwargs: Dict[str, Any] = {}
    if args.selector == "random":
        selector_kwargs["sample_fraction"] = args.sample_fraction
    if args.selector in ("hill_climbing", "bayesian_optimization"):
        selector_kwargs["batch_size"] = args.batch_size
    if args.selector == "epsilon_lucb":
        selector_kwargs["epsilon"] = args.epsilon
    if args.selector == "threshold_successive_elimination":
        selector_kwargs["threshold"] = args.threshold

    selector = selector_cls(
        agent=MyAgent,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        **_filter_selector_kwargs(selector_cls, selector_kwargs),
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
