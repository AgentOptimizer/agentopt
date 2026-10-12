"""Compare Gittins and selection baselines with live OpenAI calls.

Install openai, python-dotenv and agentopt-py, and set OPENAI_API_KEY.
Bayesian optimization additionally needs agentopt-py[bayesian].
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


STOP_REASON_MESSAGES = {
    "user_stopped": "You requested a stop.",
    "question_budget": "The remaining evaluation allowance cannot cover the next batch.",
    "search_cost_budget": "The search cost limit was reached, or the next batch would exceed it.",
    "warm_start_budget": "The budget was reached during initial sampling; recommendations may not be available.",
    "gittins_stop": "The search ended on Gittins stopping advice.",
    "all_arms_completed": "All configurations have been evaluated on every test question.",
    "direction_eta_numerical_floor": "The multi-objective search reached its numerical stopping limit.",
    "evaluation_failed": "The evaluation batch produced no usable results.",
    "completed": "The search completed.",
}


def stop_reason_message(reason):
    return STOP_REASON_MESSAGES.get(reason, (reason or "The search ended.").replace("_", " "))


def run_selection(method="auto", objective_mode="single", max_search_cost_usd=None,
                  cost_scaling_factor=1e-4, evaluation_fraction=0.5,
                  stop_on_recommendation=False):
    options = {}
    if method in ("auto", "gittins"):
        options.update(warm_start_batch_size=2, batch_size=2,
                       observation_budget_fraction=evaluation_fraction,
                       max_search_cost_usd=max_search_cost_usd)
        if objective_mode == "single":
            options.update(cost_scaling_factor=cost_scaling_factor,
                           stop_on_recommendation=stop_on_recommendation)
    elif method in ("random", "random_configurations", "random_questions"):
        if max_search_cost_usd is not None:
            raise ValueError("Random baselines use an evaluation fraction or user stop, not a dollar cap")
        options.update(observation_budget_fraction=evaluation_fraction, seed=42)
    elif method == "ucb":
        options.update(a=1.0, observation_budget_fraction=evaluation_fraction,
                       max_search_cost_usd=max_search_cost_usd, seed=42)
    elif method == "bayesian":
        options.update(batch_size=1, sample_fraction=evaluation_fraction,
                       max_search_cost_usd=max_search_cost_usd)
    selector = ModelSelector(
        agent=MyAgent, models=models, eval_fn=eval_fn, dataset=dataset,
        method=method, objective_mode=objective_mode, **options,
    )

    print("Each evaluation runs one model configuration on one test question and scores its output.")
    if max_search_cost_usd is not None:
        print(f"Search cost limit: ${max_search_cost_usd:.2f} (soft limit; scheduled work may exceed it).")
    else:
        print("Search cost limit: not set.")

    def show_progress(snapshot):
        status = ("Finished" if snapshot.is_final else
                  "Collecting initial samples" if snapshot.stop_reason == "warming_up" else "Searching")
        cost = "Unavailable" if snapshot.selection_cost is None else f"${snapshot.selection_cost:.6f}"
        print(f"\nStatus: {status}")
        print(f"Completed evaluations: {snapshot.total_evaluations}")
        print(f"Search cost: {cost}")
        if objective_mode == "single":
            best = snapshot.get_best_combo()
            if best is None:
                print("No recommendation available yet.")
            elif hasattr(snapshot, "get_recommendations"):
                candidate = next(r for r in snapshot.get_recommendations() if r.combo == best)
                print("Current recommendation (conservative score and USD/question):")
                print(f"  {best} | score={candidate.conservative_accuracy:.4f} | "
                      f"cost=${candidate.conservative_cost_usd:.6f}")
            else:
                candidate = snapshot.get_best()
                price = candidate.price
                cost_text = "unavailable" if price is None else f"${price:.6f}"
                print("Current recommendation (observed score and USD/question):")
                print(f"  {best} | score={candidate.accuracy:.4f} | cost={cost_text}")
        else:
            candidates = snapshot.get_recommendations()
            if candidates:
                print("Current recommendations (conservative score and USD/question):")
                for candidate in candidates:
                    print(f"  {candidate.combo} | score={candidate.conservative_accuracy:.4f} | "
                          f"cost=${candidate.conservative_cost_usd:.6f}")
            else:
                print("No recommendation available yet.")
        if snapshot.stop_recommended and not snapshot.is_final:
            print("Stopping advice: Gittins suggests that further evaluation may not be worth its cost.")
            print("The search can continue unless automatic stopping is enabled or a stop is requested.")
        if snapshot.is_final:
            print(f"Reason: {stop_reason_message(snapshot.stop_reason)}")
            if snapshot.search_cost_overshoot_usd > 0:
                print(f"Search cost limit exceeded by: ${snapshot.search_cost_overshoot_usd:.6f}")
        # A UI may call selector.request_stop(), or this callback may return False.

    return selector.select_best(parallel=True, max_concurrent=4,
                                on_update=show_progress)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("auto", "gittins", "random", "ucb", "bayesian",
                                             "random_configurations", "random_questions"), default="auto")
    parser.add_argument("--objective-mode", choices=("single", "multi"), default="single")
    parser.add_argument("--max-search-cost-usd", type=float,
                        help="Optional soft cap on new evaluation charges, for Gittins, UCB, or Bayesian optimization")
    parser.add_argument("--cost-scaling-factor", type=float, default=1e-4,
                        help="Single Gittins search-cost coefficient in score units per USD")
    parser.add_argument("--evaluation-fraction", type=float, default=0.5,
                        help="Fraction of configuration-question cells; for Bayesian, fraction of full configurations")
    parser.add_argument("--stop-on-recommendation", action="store_true",
                        help="End single-objective Gittins automatically on its stopping advice")
    args = parser.parse_args()
    supported = {"single": {"auto", "gittins", "random", "random_configurations", "random_questions", "ucb", "bayesian"},
                 "multi": {"auto", "gittins", "random_configurations", "random_questions"}}
    if args.method not in supported[args.objective_mode]:
        parser.error(f"{args.method} does not support {args.objective_mode} mode")
    if args.method in ("random", "random_configurations", "random_questions") and args.max_search_cost_usd is not None:
        parser.error("Random baselines use an evaluation fraction or user stop, not a dollar cap")
    if args.stop_on_recommendation and (args.method not in ("auto", "gittins") or args.objective_mode != "single"):
        parser.error("--stop-on-recommendation applies only to single-objective Gittins")
    if not 0 < args.evaluation_fraction <= 1:
        parser.error("--evaluation-fraction must be in (0, 1]")
    results = run_selection(args.method, args.objective_mode, args.max_search_cost_usd,
                            args.cost_scaling_factor, args.evaluation_fraction,
                            args.stop_on_recommendation)
    results.print_summary()
    if args.objective_mode == "single":
        print("Recommended configuration:", results.get_best_combo() or "No recommendation available.")
    else:
        for candidate in results.get_recommendations():
            print(candidate.combo, candidate.conservative_accuracy, candidate.conservative_cost_usd)
        chosen = results.choose(max_cost_usd=0.01)
        print("Deployment choice at $0.01/question:", chosen.combo if chosen else None)
