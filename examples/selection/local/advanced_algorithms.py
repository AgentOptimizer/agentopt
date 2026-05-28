"""
Example: Advanced model selection algorithms.

The framework-specific examples (custom_agent_example.py, langchain_example.py, etc.)
all use method="brute_force" for simplicity. This example demonstrates the other
selection algorithms available via ModelSelector(method=...). The default
method="auto" automatically finds the best combination (wired to arm_elimination —
strong best-arm identification with lower search cost than brute_force).

Prerequisites:
    1. pip install openai agentopt-py
    2. Set OPENAI_API_KEY environment variable (this script uses python-dotenv; install
       it or use ``pip install \"agentopt-py[examples]\"``)
    3. For Bayesian optimization: pip install "agentopt-py[bayesian]"
    4. For matrix UCB-LRF: pip install "agentopt-py[ucb_lrf]"

The matrix UCB demos use ``sample_fraction=0.1`` (~10% of the combination × datapoint
grid), like ``random`` / ``bayesian``. Matrix UCB-LRF also accepts ``warmup_fraction``
(alias for ``warmup_percentage``). Use ``1.0`` for a full grid; tune ``max_concurrent`` for step size.
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
        plan = (
            self.client.chat.completions.create(
                model=self.planner_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Create a brief plan to answer the question.",
                    },
                    {"role": "user", "content": input_data},
                ],
            )
            .choices[0]
            .message.content
        )

        answer = (
            self.client.chat.completions.create(
                model=self.solver_model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Follow this plan and answer concisely:\n{plan}",
                    },
                    {"role": "user", "content": input_data},
                ],
            )
            .choices[0]
            .message.content
        )
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
    """method="auto" — automatically finds the best combination (default; wired to arm_elimination — strong best-arm identification, cheaper than brute_force)."""
    selector = ModelSelector(
        agent=MyAgent,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        method="auto",
        objective_mode="weighted",
        lambda_latency=0.2,
    )
    return selector.select_best(parallel=True)


def run_random():
    """method="random" — evaluate a random subset of combinations."""
    selector = ModelSelector(
        agent=MyAgent,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        method="random",
        sample_fraction=0.25,  # evaluate 25% of all combinations
        objective_mode="weighted",
        lambda_latency=0.2,
    )
    return selector.select_best(parallel=True)


def run_hill_climbing():
    """method="hill_climbing" — greedy search using model quality/speed rankings."""
    selector = ModelSelector(
        agent=MyAgent,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        method="hill_climbing",
        batch_size=4,  # number of neighbors to evaluate per step
        objective_mode="weighted",
        lambda_latency=0.2,
    )
    return selector.select_best(parallel=True)


def run_arm_elimination():
    """method="arm_elimination" — eliminates statistically dominated combinations early."""
    selector = ModelSelector(
        agent=MyAgent,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        method="arm_elimination",
        objective_mode="weighted",
        lambda_latency=0.2,
    )
    return selector.select_best(parallel=True)


def run_epsilon_lucb():
    """method="epsilon_lucb" — stops when the best arm is identified within epsilon."""
    selector = ModelSelector(
        agent=MyAgent,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        method="epsilon_lucb",
        epsilon=0.01,  # acceptable gap from the true best
        objective_mode="weighted",
        lambda_latency=0.2,
    )
    return selector.select_best(parallel=True)


def run_threshold():
    """method="threshold" — classify combinations as above/below a quality threshold."""
    selector = ModelSelector(
        agent=MyAgent,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        method="threshold",
        threshold=0.75,  # minimum acceptable accuracy
        objective_mode="weighted",
        lambda_latency=0.2,
    )
    return selector.select_best(parallel=True)


def run_lm_proposal():
    """method="lm_proposal" — use a proposer LLM to shortlist promising combinations."""
    selector = ModelSelector(
        agent=MyAgent,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        method="lm_proposal",
        objective_mode="weighted",
        lambda_latency=0.2,
    )
    return selector.select_best(parallel=True)


def run_bayesian():
    """method="bayesian" — GP-based Bayesian optimization (requires agentopt[bayesian])."""
    selector = ModelSelector(
        agent=MyAgent,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        method="bayesian",
        batch_size=4,
        sample_fraction=0.25,  # evaluate 25% of all combinations
        objective_mode="weighted",
        lambda_latency=0.2,
    )
    return selector.select_best(parallel=True)


def run_matrix_ucb():
    """method="matrix_ucb" — UCB exploration over the combination × datapoint matrix."""
    selector = ModelSelector(
        agent=MyAgent,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        method="matrix_ucb",
        a=1.0,
        sample_fraction=0.1,
        objective_mode="pareto",
    )
    return selector.select_best(max_concurrent=4)


def run_matrix_ucb_lrf():
    """method="matrix_ucb_lrf" — matrix UCB with low-rank ensemble (requires agentopt[ucb_lrf])."""
    selector = ModelSelector(
        agent=MyAgent,
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        method="matrix_ucb_lrf",
        rank=1,
        ensemble_size=16,
        iterations=5,
        eta=5.0,
        warmup_fraction=0.05,
        sample_fraction=0.1,
        objective_mode="pareto",
    )
    # Unlike matrix_ucb (which always uses async eval), LRF still uses parallel=True
    # for concurrent cell evaluation; sequential path is sync-only.
    return selector.select_best(parallel=True, max_concurrent=4)


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
    "matrix_ucb": run_matrix_ucb,
    "matrix_ucb_lrf": run_matrix_ucb_lrf,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Demo advanced model selection algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available methods:
  auto              Automatically finds the best combination (wired to arm_elimination; lower search cost than brute_force) (default)
  random            Evaluate a random subset of combinations
  hill_climbing     Greedy search using model quality/speed rankings
  arm_elimination   Eliminate statistically dominated combinations early
  epsilon_lucb      Stop when best arm is identified within epsilon
  threshold         Classify combinations above/below a quality threshold
  lm_proposal       Use a proposer LLM to shortlist promising combinations
  bayesian          GP-based Bayesian optimization (requires agentopt[bayesian])
  matrix_ucb        UCB on the combination × datapoint matrix (demo: 10%% budget)
  matrix_ucb_lrf    Same with low-rank uncertainty (requires agentopt[ucb_lrf])
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
