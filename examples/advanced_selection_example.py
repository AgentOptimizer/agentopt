"""
Example: Advanced model selection algorithms.

The framework-specific examples (custom_agent_example.py, langchain_example.py, etc.)
all use method="brute_force" for simplicity. This example demonstrates the other
selection algorithms available via ModelSelector(method=...).

Prerequisites:
    1. pip install openai agentopt
    2. Set OPENAI_API_KEY environment variable
    3. For Bayesian optimization: pip install "agentopt[bayesian]"
"""

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from agentopt import ModelSelector


# ---------------------------------------------------------------------------
# Agent, dataset, and eval_fn (same as custom_agent_example.py)
# ---------------------------------------------------------------------------

class MyAgent:
    def __init__(self, models):
        self.client = OpenAI()
        self.planner_model = models["planner"]
        self.solver_model = models["solver"]

    def run(self, input_data):
        plan = self.client.chat.completions.create(
            model=self.planner_model,
            messages=[
                {"role": "system", "content": "Create a brief plan to answer the question."},
                {"role": "user", "content": input_data},
            ],
        ).choices[0].message.content

        answer = self.client.chat.completions.create(
            model=self.solver_model,
            messages=[
                {"role": "system", "content": f"Follow this plan and answer concisely:\n{plan}"},
                {"role": "user", "content": input_data},
            ],
        ).choices[0].message.content
        return answer


def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is H2O commonly known as?", "water"),
]

models = {
    "planner": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
    "solver": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
}


# ---------------------------------------------------------------------------
# Selection algorithms
# ---------------------------------------------------------------------------

def run_auto():
    """method="auto" — automatically picks the best algorithm (default)."""
    selector = ModelSelector(
        agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
        method="auto",
    )
    return selector.select_best(parallel=True)


def run_random():
    """method="random" — evaluate a random subset of combinations."""
    selector = ModelSelector(
        agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
        method="random",
        sample_fraction=0.5,  # evaluate 50% of all combinations
    )
    return selector.select_best(parallel=True)


def run_hill_climbing():
    """method="hill_climbing" — greedy search using model quality/speed rankings."""
    selector = ModelSelector(
        agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
        method="hill_climbing",
        batch_size=4,  # number of neighbors to evaluate per step
    )
    return selector.select_best(parallel=True)


def run_arm_elimination():
    """method="arm_elimination" — eliminates statistically dominated combinations early."""
    selector = ModelSelector(
        agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
        method="arm_elimination",
    )
    return selector.select_best(parallel=True)


def run_epsilon_lucb():
    """method="epsilon_lucb" — stops when the best arm is identified within epsilon."""
    selector = ModelSelector(
        agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
        method="epsilon_lucb",
        epsilon=0.05,  # acceptable gap from the true best
    )
    return selector.select_best(parallel=True)


def run_threshold():
    """method="threshold" — classify combinations as above/below a quality threshold."""
    selector = ModelSelector(
        agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
        method="threshold",
        threshold=0.8,  # minimum acceptable accuracy
    )
    return selector.select_best(parallel=True)


def run_lm_proposal():
    """method="lm_proposal" — use a proposer LLM to shortlist promising combinations."""
    selector = ModelSelector(
        agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
        method="lm_proposal",
    )
    return selector.select_best(parallel=True)


def run_bayesian():
    """method="bayesian" — GP-based Bayesian optimization (requires agentopt[bayesian])."""
    selector = ModelSelector(
        agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
        method="bayesian",
        batch_size=4,
    )
    return selector.select_best(parallel=True)


# ---------------------------------------------------------------------------
# Main — pick which algorithm to demo
# ---------------------------------------------------------------------------

METHODS = {
    "auto": run_auto,
    "random": run_random,
    "hill_climbing": run_hill_climbing,
    "arm_elimination": run_arm_elimination,
    "epsilon_lucb": run_epsilon_lucb,
    "threshold": run_threshold,
    "lm_proposal": run_lm_proposal,
    "bayesian": run_bayesian,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Demo advanced model selection algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available methods:
  auto              Automatically picks the best algorithm (default)
  random            Evaluate a random subset of combinations
  hill_climbing     Greedy search using model quality/speed rankings
  arm_elimination   Eliminate statistically dominated combinations early
  epsilon_lucb      Stop when best arm is identified within epsilon
  threshold         Classify combinations above/below a quality threshold
  lm_proposal       Use a proposer LLM to shortlist promising combinations
  bayesian          GP-based Bayesian optimization (requires agentopt[bayesian])
""",
    )
    parser.add_argument("--method", choices=METHODS, default="auto")
    args = parser.parse_args()

    print(f"Running method: {args.method}")
    print("-" * 60)

    results = METHODS[args.method]()
    results.print_summary()

    best = results.get_best_combo()
    if best:
        print(f"\nBest combination: {best}")
