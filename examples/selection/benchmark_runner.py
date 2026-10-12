"""Run a local JSONL benchmark with live recommendations and cooperative stopping.

From this repository checkout: pip install -e . && pip install openai
Set OPENAI_API_KEY, then run this file with --dataset and --models.
The bundled benchmark_sample.jsonl illustrates the format; it is not a
representative evaluation benchmark. Replace it and the scoring function for
real use. Ctrl+C ends the run after the scheduled batch and saves its results;
it does not create a resumable checkpoint.
"""

import argparse
import json
import math
import signal
from pathlib import Path
from threading import Event


METHODS = {
    "single": ("auto", "gittins", "random", "random_configurations", "random_questions", "ucb", "bayesian"),
    "multi": ("auto", "gittins", "random_configurations", "random_questions"),
}
RANDOM_METHODS = ("random", "random_configurations", "random_questions")


class AnswerAgent:
    """Replace this class to evaluate your own agent or multiple agent nodes."""

    def __init__(self, models):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = models["answer"]

    def run(self, input_data):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return only the short answer, with no explanation."},
                {"role": "user", "content": input_data},
            ],
        )
        return response.choices[0].message.content or ""


def eval_fn(expected, actual):
    """Case-insensitive exact match after trimming surrounding whitespace."""
    return float(expected.strip().casefold() == str(actual).strip().casefold())


def load_dataset(path):
    """Load nonempty string input/expected pairs, identifying invalid line numbers."""
    dataset = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("input", "expected")
            ):
                raise ValueError(f"{path}:{line_number}: input and expected must be nonempty strings")
            dataset.append((row["input"], row["expected"]))
    if not dataset:
        raise ValueError(f"{path}: the benchmark must contain at least one example")
    return dataset


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL file with input and expected strings")
    parser.add_argument("--models", nargs="+", required=True, help="OpenAI model names available to your account")
    parser.add_argument("--method", choices=METHODS["single"], default="auto")
    parser.add_argument("--objective-mode", choices=("single", "multi"), default="single")
    parser.add_argument("--evaluation-fraction", type=float,
                        help="Fraction of configuration-question cells (default 1); Bayesian uses full configurations (default .25)")
    parser.add_argument("--max-search-cost-usd", type=float,
                        help="Soft dollar cap for Gittins, UCB, or Bayesian; scheduled work can overshoot")
    parser.add_argument("--cost-scaling-factor", type=float,
                        help="Single Gittins search-cost coefficient (default 1e-4)")
    parser.add_argument("--stop-on-recommendation", action="store_true",
                        help="Automatically stop single Gittins on its stopping advice; default is advice only")
    parser.add_argument("--seed", type=int, help="Seed for Gittins, random, or UCB; Bayesian has no seed option")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Maximum concurrent evaluations (default 4)")
    parser.add_argument("--output", type=Path, default=Path("results.json"), help="Result JSON path (default results.json)")
    return parser


def selector_options(args):
    """Route only supported options, before creating a tracker or making requests."""
    if args.method not in METHODS[args.objective_mode]:
        raise ValueError(f"{args.method} does not support {args.objective_mode} mode")
    single_gittins = args.method in ("auto", "gittins") and args.objective_mode == "single"
    if (args.cost_scaling_factor is not None or args.stop_on_recommendation) and not single_gittins:
        raise ValueError("--cost-scaling-factor and --stop-on-recommendation apply only to single Gittins")
    if args.max_search_cost_usd is not None and args.method in RANDOM_METHODS:
        raise ValueError("Random baselines use --evaluation-fraction or Ctrl+C, not a dollar cap")
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be positive")
    if args.seed is not None and (args.seed < 0 or args.method == "bayesian"):
        raise ValueError("--seed must be nonnegative and applies only to Gittins, random, or UCB")
    options = {}
    if args.evaluation_fraction is not None:
        if not math.isfinite(args.evaluation_fraction) or not 0 < args.evaluation_fraction <= 1:
            raise ValueError("--evaluation-fraction must be in (0, 1]")
        name = "sample_fraction" if args.method == "bayesian" else "observation_budget_fraction"
        options[name] = args.evaluation_fraction
    for name in ("max_search_cost_usd", "cost_scaling_factor"):
        value = getattr(args, name)
        if value is not None:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
            options[name] = value
    if args.seed is not None:
        options["seed"] = args.seed
    if args.stop_on_recommendation:
        options["stop_on_recommendation"] = True
    return options


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


def show_progress(snapshot):
    cost = "unavailable" if snapshot.selection_cost is None else f"${snapshot.selection_cost:.6f}"
    status = ("Finished" if snapshot.is_final else
              "Collecting initial samples" if snapshot.stop_reason == "warming_up" else "Searching")
    print(f"\nStatus: {status}", flush=True)
    print(f"Completed evaluations: {snapshot.total_evaluations}", flush=True)
    print(f"Search cost: {cost}", flush=True)
    if getattr(snapshot, "objective_mode", "single") == "multi":
        recommendations = snapshot.get_recommendations()
        if recommendations:
            print("Current recommendations (conservative score and USD/question):", flush=True)
            for candidate in recommendations:
                print(f"  {candidate.combo} | score={candidate.conservative_accuracy:.4f} | "
                      f"cost=${candidate.conservative_cost_usd:.6f}", flush=True)
        else:
            print("No recommendation available yet.", flush=True)
    else:
        best = snapshot.get_best_combo()
        if best is None:
            print("No recommendation available yet.", flush=True)
        elif hasattr(snapshot, "get_recommendations"):
            candidate = next(r for r in snapshot.get_recommendations() if r.combo == best)
            print("Current recommendation (conservative score and USD/question):", flush=True)
            print(f"  {best} | score={candidate.conservative_accuracy:.4f} | "
                  f"cost=${candidate.conservative_cost_usd:.6f}", flush=True)
        else:
            candidate = snapshot.get_best()
            price = candidate.price
            cost_text = "unavailable" if price is None else f"${price:.6f}"
            print("Current recommendation (observed score and USD/question):", flush=True)
            print(f"  {best} | score={candidate.accuracy:.4f} | cost={cost_text}", flush=True)
    if snapshot.stop_recommended and not snapshot.is_final:
        print("Stopping advice: Gittins suggests that further evaluation may not be worth its cost.", flush=True)
        print("The search can continue unless automatic stopping is enabled or a stop is requested.", flush=True)
    if snapshot.is_final:
        print(f"Reason: {stop_reason_message(snapshot.stop_reason)}", flush=True)
        if snapshot.search_cost_overshoot_usd > 0:
            print(f"Search cost limit exceeded by: ${snapshot.search_cost_overshoot_usd:.6f}", flush=True)


def run_selection(selector, *, max_concurrent, output):
    """Run on the main thread so Ctrl+C requests a stop without cancelling work."""
    stop_event = Event()

    def request_stop(signum, frame):
        # Avoid raising KeyboardInterrupt while the selector drains its batch.
        stop_event.set()

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        print("Each evaluation runs one model configuration on one test question and scores its output.", flush=True)
        print("Press Ctrl+C to stop after the scheduled batch; its completed results will be saved.", flush=True)
        results = selector.select_best(
            parallel=True, max_concurrent=max_concurrent,
            on_update=show_progress, stop_event=stop_event,
        )
        Path(output).write_text(results.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"Saved results to {output}", flush=True)
        return results
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        options = selector_options(args)
        dataset = load_dataset(args.dataset)
        if not args.output.parent.is_dir() or args.output.is_dir():
            raise ValueError("--output must name a file in an existing directory")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    try:
        import openai  # noqa: F401 -- fail before starting the tracker if the SDK is absent
    except ImportError:
        parser.error("This example requires the OpenAI SDK: pip install openai")
    from agentopt import ModelSelector

    selector = ModelSelector(
        agent=AnswerAgent, models={"answer": args.models}, eval_fn=eval_fn,
        dataset=dataset, method=args.method, objective_mode=args.objective_mode, **options,
    )
    if args.max_search_cost_usd is not None:
        print(f"Search cost limit: ${args.max_search_cost_usd:.2f} (soft limit; scheduled work may exceed it).", flush=True)
    else:
        print("Search cost limit: not set.", flush=True)
    return run_selection(selector, max_concurrent=args.max_concurrent, output=args.output)


if __name__ == "__main__":
    main()
