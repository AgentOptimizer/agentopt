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
import os
import signal
import sys
import tempfile
from pathlib import Path
from threading import Event, Lock


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


class BenchmarkRunError(RuntimeError):
    """A reported terminal failure, without exposing provider response bodies."""


def failure_message(error, *, scoring=False):
    if scoring:
        return "Scoring failed. Check eval_fn; it must return a finite numeric score between 0 and 1."
    # SDK errors can be wrapped; inspect their causes without printing credentials,
    # prompts, reference answers, or raw provider responses.
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        chain.append(error)
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    for cause in chain:
        status = getattr(cause, "status_code", None)
        if status == 401:
            return "Authentication failed. Check OPENAI_API_KEY and ensure the key is valid."
        if status == 403:
            return "Access denied. Check your API key permissions and access to the selected models."
        if status in (400, 404):
            return "The model API rejected the request. Check the selected model names, endpoint, and request parameters."
        if status == 429:
            return "The API rate limit or quota was reached. Check usage and billing, or retry later with lower --max-concurrent."
        if type(cause).__name__ == "APITimeoutError":
            return "The model API request timed out. Check connectivity and retry, or lower --max-concurrent."
        if type(cause).__name__ == "APIConnectionError":
            return "Could not connect to the model API. Check your network and API endpoint, then retry."
        if "api_key" in str(cause).lower() and "must be set" in str(cause).lower():
            return "No API key is configured. Set OPENAI_API_KEY before running the benchmark."
    if any(isinstance(cause, ImportError) for cause in chain):
        return "A required dependency is missing. Install the checkout and dependencies for the selected method."
    if any(isinstance(cause, OSError) for cause in chain):
        return "Could not access a required file. Check file paths, permissions, and available disk space."
    return "The benchmark failed. Check the agent, selected models, scorer, pricing, and selector settings."


class FailureState:
    """Retain the first actionable failure even when the SDK skips a failed sample."""

    def __init__(self):
        self.message = None
        self.stop_event = Event()
        self._lock = Lock()

    def record(self, error, *, scoring=False):
        with self._lock:
            if self.message is None:
                self.message = failure_message(error, scoring=scoring)
        self.stop_event.set()
        return BenchmarkRunError(self.message)

    def wrap_agent(self, agent_class):
        state = self

        class CheckedAgent:
            def __init__(self, models):
                try:
                    self.agent = agent_class(models)
                except Exception as error:
                    raise state.record(error) from error

            def run(self, input_data):
                try:
                    return self.agent.run(input_data)
                except Exception as error:
                    raise state.record(error) from error

        return CheckedAgent

    def wrap_scorer(self, scorer):
        def checked_score(expected, actual):
            try:
                score = float(scorer(expected, actual))
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid score")
                return score
            except Exception as error:
                raise self.record(error, scoring=True) from error
        return checked_score


def save_json(path, payload):
    """Replace a result only after a complete JSON write succeeds."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def report_failure(message, *, output, snapshot=None):
    print("\nStatus: Failed", file=sys.stderr, flush=True)
    print(f"Error: {message}", file=sys.stderr, flush=True)
    if snapshot is None or not snapshot.results:
        print("No evaluation results were saved; no completed result snapshot is available.", file=sys.stderr, flush=True)
        return
    partial = snapshot.model_copy(deep=True)
    partial.is_final = True
    partial.stop_reason = "evaluation_failed"
    payload = partial.model_dump(mode="json")
    payload["is_partial"] = True
    payload["failure_message"] = message
    payload["snapshot_scope"] = "Last available result snapshot; failed or unfinished work and its charges may be missing."
    output = Path(output)
    partial_path = output.with_name(f"{output.stem}.partial{output.suffix or '.json'}")
    try:
        save_json(partial_path, payload)
    except (OSError, ValueError, TypeError):
        print("Partial results could not be saved. Check file permissions and available disk space.", file=sys.stderr, flush=True)
    else:
        print(f"Saved partial results ({partial.total_evaluations} evaluations) to {partial_path}", file=sys.stderr, flush=True)
        print("Only the last available snapshot was saved; failed or unfinished work and its charges may be missing.", file=sys.stderr, flush=True)
        print("This file is not a resumable checkpoint.", file=sys.stderr, flush=True)


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


def show_progress(snapshot, *, total_questions):
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
                      f"cost=${candidate.conservative_cost_usd:.6f} | "
                      f"evaluated={candidate.observed_samples}/{total_questions}", flush=True)
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
                  f"cost=${candidate.conservative_cost_usd:.6f} | "
                  f"evaluated={candidate.observed_samples}/{total_questions}", flush=True)
        else:
            candidate = snapshot.get_best()
            price = candidate.price
            cost_text = "unavailable" if price is None else f"${price:.6f}"
            print("Current recommendation (observed score and USD/question):", flush=True)
            print(f"  {best} | score={candidate.accuracy:.4f} | cost={cost_text} | "
                  f"evaluated={len(candidate.datapoint_results)}/{total_questions}", flush=True)
    if snapshot.stop_recommended and not snapshot.is_final:
        print("Stopping advice: Gittins suggests that further evaluation may not be worth its cost.", flush=True)
        print("The search can continue unless automatic stopping is enabled or a stop is requested.", flush=True)
    if snapshot.is_final:
        print(f"Reason: {stop_reason_message(snapshot.stop_reason)}", flush=True)
        if snapshot.search_cost_overshoot_usd > 0:
            print(f"Search cost limit exceeded by: ${snapshot.search_cost_overshoot_usd:.6f}", flush=True)


def run_selection(selector, *, max_concurrent, output, failures=None):
    """Run on the main thread so Ctrl+C requests a stop without cancelling work."""
    failures = failures or FailureState()
    stop_event = failures.stop_event
    last_snapshot = None

    def update(snapshot):
        nonlocal last_snapshot
        last_snapshot = snapshot
        if failures.message is None:
            show_progress(snapshot, total_questions=len(selector.dataset))

    def request_stop(signum, frame):
        # Avoid raising KeyboardInterrupt while the selector drains its batch.
        if stop_event.is_set():
            return
        stop_event.set()
        print("\nStop requested. Waiting for the current batch to finish…", flush=True)

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        print("Each evaluation runs one model configuration on one test question and scores its output.", flush=True)
        print("Press Ctrl+C to stop after the scheduled batch; its completed results will be saved.", flush=True)
        results = selector.select_best(
            parallel=True, max_concurrent=max_concurrent,
            on_update=update,
            stop_event=stop_event,
        )
        if failures.message is not None:
            raise BenchmarkRunError(failures.message)
        save_json(output, results.model_dump(mode="json"))
        print(f"Saved results to {output}", flush=True)
        return results
    except Exception as error:
        message = failures.message or failure_message(error)
        if last_snapshot is None:
            last_snapshot = getattr(selector, "latest_results", None)
        report_failure(message, output=output, snapshot=last_snapshot)
        raise BenchmarkRunError(message) from None
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

    failures = FailureState()
    try:
        selector = ModelSelector(
            agent=failures.wrap_agent(AnswerAgent), models={"answer": args.models},
            eval_fn=failures.wrap_scorer(eval_fn), dataset=dataset,
            method=args.method, objective_mode=args.objective_mode, **options,
        )
    except Exception as error:
        report_failure(failure_message(error), output=args.output)
        raise SystemExit(1) from None
    if args.max_search_cost_usd is not None:
        print(f"Search cost limit: ${args.max_search_cost_usd:.2f} (soft limit; scheduled work may exceed it).", flush=True)
    else:
        print("Search cost limit: not set.", flush=True)
    try:
        return run_selection(selector, max_concurrent=args.max_concurrent, output=args.output, failures=failures)
    except BenchmarkRunError:
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
