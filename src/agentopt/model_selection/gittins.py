"""Cost-coupled Gittins selection over live configuration-question evaluations.

Single-objective selection uses fixed-cost finite-dataset Gittins indices.
Multi-objective selection uses the exact deployment and quality axes and
returns a finite-LCB Pareto set. Both modes share observations across visits,
freeze warm-start costs. Multi mode decays eta independently on direction stops.
"""

from __future__ import annotations

import asyncio
from bisect import bisect_left
from copy import deepcopy
from fractions import Fraction
from itertools import chain
import math
import operator
from pathlib import Path
from uuid import uuid4
import warnings

import numpy as np
from pydantic import BaseModel, Field

from ..model_price import MODEL_PRICES, get_model_price
from .axis_gittins_dp import AxisGittinsBoundaryCache
from .base import BaseModelSelector, DatapointResult, ModelResult, SelectionResults, save_json
from .checkpoint import (AUTO_CHECKPOINT, CheckpointError, EvaluationCheckpoint,
                         default_checkpoint_path, fingerprint, workflow_identity)
from .gittins_statistics import (
    GaussianPosterior, finite_dataset_moments, fit_warm_start, raw_pareto_front_indices,
)


DEFAULT_GITTINS_BATCH_SIZE = 4


def _ordered_prices(prices):
    if prices is None:
        return None
    # Both lookup levels use names, while dataset and model-spec mappings can
    # be observed in insertion order by user code.
    return {model: dict(sorted(rates.items())) for model, rates in sorted(prices.items())}


class _PartialBatchStopped(Exception):
    def __init__(self, result):
        self.result = result


class GittinsRecommendation(BaseModel):
    """A recommended configuration with finite-dataset Gaussian predictions.

    ``accuracy`` and ``cost_usd`` are predictions for the supplied dataset.
    Conservative coordinates use mean minus beta*std for score, and mean
    plus beta*std for cost. They are model estimates, not coverage guarantees.
    """

    model_name: str
    combo: dict[str, str]
    observed_samples: int
    accuracy: float
    accuracy_std: float
    cost_usd: float
    cost_std_usd: float
    conservative_accuracy: float
    conservative_cost_usd: float
    objective: float | None = None
    conservative_objective: float | None = None


class GittinsSelectionResults(SelectionResults):
    """Evaluation records plus the recommendation at the actual stopping time.

    Multi mode marks no arbitrary single ``is_best``. Use ``choose`` to
    express a deployment preference, or inspect ``recommendations`` directly.
    """

    objective_mode: str
    unmeasured_failed_attempts: int = Field(
        default=0, description="Evaluation attempts with unknown charges, counted once per attempt even after multiple HTTP timeouts."
    )
    pending_batch_evaluations: int = 0
    recommendations: list[GittinsRecommendation] = Field(default_factory=list)
    stop_reason: str
    total_evaluations: int
    search_cost_overshoot_usd: float = 0.0
    direction_eta: list[float] = Field(default_factory=list)

    def get_recommendations(self) -> list[GittinsRecommendation]:
        return list(self.recommendations)

    def get_best_combo(self) -> dict[str, str] | None:
        # Archived Gittins and random runs already carry the structured
        # mapping on their recommendation, even with an ambiguous label.
        best = self.get_best()
        if best is not None:
            for candidate in self.recommendations:
                if candidate.model_name == best.model_name:
                    return dict(candidate.combo)
        return super().get_best_combo()

    def choose(self, *, max_cost_usd=None, min_score=None, cost_weight=None):
        """Choose a deployment candidate using conservative score/cost.

        With a cost limit, maximize score among feasible candidates. With
        only a score floor, minimize cost. A weight maximizes score minus
        weight * raw USD. Infeasible constraints return ``None``. This does
        not change ``is_best`` or guarantee performance on future questions.
        """
        for name, value in (("max_cost_usd", max_cost_usd), ("min_score", min_score),
                            ("cost_weight", cost_weight)):
            if value is not None and (not math.isfinite(value) or (name != "min_score" and value < 0)):
                raise ValueError(f"{name} must be finite" + (" and nonnegative" if name != "min_score" else ""))
        if all(value is None for value in (max_cost_usd, min_score, cost_weight)):
            if len(self.recommendations) > 1:
                raise ValueError("Specify a cost limit, score floor, or cost weight to choose a combination")
            return self.recommendations[0] if self.recommendations else None
        feasible = [r for r in self.recommendations
                    if (max_cost_usd is None or r.conservative_cost_usd <= max_cost_usd)
                    and (min_score is None or r.conservative_accuracy >= min_score)]
        if not feasible:
            return None
        if cost_weight is not None:
            return max(feasible, key=lambda r: (r.conservative_accuracy - cost_weight * r.conservative_cost_usd,
                                               -r.conservative_cost_usd))
        if max_cost_usd is not None:
            return max(feasible, key=lambda r: (r.conservative_accuracy, -r.conservative_cost_usd))
        return min(feasible, key=lambda r: (r.conservative_cost_usd, -r.conservative_accuracy))


def _integer(value, name, minimum=1):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class GittinsModelSelector(BaseModelSelector):
    """Gittins with scalar or two-axis accuracy/cost objectives.

    Included in the standard installation (NumPy and SciPy). No complete score matrix
    is needed. Each combination receives a seeded independent question
    order and a uniform warm start; no cell is evaluated twice in one run.

    ``single`` optimizes score, optionally penalizing cost and latency with
    fixed warm-start min/max scales. ``multi`` keeps score and mean USD as
    separate objectives; scalar lambda weights are invalid in that mode.

    ``observation_budget_fraction`` counts cells including warm start.
    ``max_search_cost_usd`` counts actual non-cached token charges including
    warm start. Before adaptive batches, their frozen warm-start charge
    estimates are reserved. Actual costs may exceed estimates, so this is
    a soft spend cap; any overshoot is reported. A cap interrupting warm
    start returns observed records and an empty recommendation set.

    Single mode prices each question with ``cost_scaling_factor=1e-4`` times
    its estimated USD cost. It uses a fixed prior (``prior_mean=0.5`` in
    normalized units), finite-dataset means and per-cell DP transitions.
    Its default recommendation_beta is zero; multi mode defaults to one.
    Single mode recommends stopping when a completed empirical value reaches
    every unfinished index, keeping the cost coefficient fixed. By default it
    continues selecting the largest unfinished index until a budget, user stop,
    or full completion. Set ``stop_on_recommendation=True`` to stop automatically.

    In multi mode, a direction stops when a completed arm's value reaches the
    maximum unfinished Gittins index. Its eta then decays; all directions must stop
    at numerical floors without an intervening observation to halt globally.
    Budget exhaustion and full completion can halt earlier. Evaluation or
    pricing failures abort the run, with the base selector stopping tracking.

    Successful cells are checkpointed automatically for identifiable workflows.
    Use checkpoint_id to version external dependencies and checkpoint_path=None to opt out.
    Dataset, ordered model specifications, seed, prices and algorithm settings
    must match, except that the total cell and spending limits may increase.
    Saved evaluations and spending count toward both new limits.
    Change checkpoint_id when prompts, tools or evaluator behavior
    change. Inputs must be JSON-serializable. Finish stops the current run but
    keeps the checkpoint; use a different path for a fresh run. An interrupted
    request not yet saved may run again. Other algorithms do not support this.
    """

    # Random baselines reuse the observation methods without this initializer.
    # Only Gittins itself enables persistence.
    checkpoint_path = None
    _failed_spend = 0
    _unmeasured_attempts = 0
    _journal = None

    def __init__(self, agent=None, models=None, eval_fn=None, dataset=None, *,
                 objective_mode="single", batch_size=None, warm_start_batch_size=None,
                 observation_budget_fraction=1.0, max_total_question_evaluations=None,
                 max_search_cost_usd=None, recommendation_beta=None,
                 recommendation_min_samples=0, score_bounds=(0.0, 1.0),
                 prior_mean=0.5, prior_variance=0.04, score_noise_variance=0.25,
                 cost_scaling_factor=1e-4, stop_on_recommendation=False,
                 cost_reference_usd=None, search_cost_scale_eta=1.0,
                 eta_initial=1.0, eta_decay=0.5, effective_cost_bin_ratio=2.0,
                 effective_cost_bin_anchor=1e-4, grid_size=129, stop_tolerance=1e-9,
                 seed=42, model_prices=None, tracker=None,
                 lambda_cost=0.0, lambda_latency=0.0,
                 checkpoint_path=AUTO_CHECKPOINT, checkpoint_id=None):
        try:
            import scipy  # noqa: F401
        except ImportError as error:
            raise ImportError('SciPy is required by AgentOpt; install with `pip install agentopt-py`') from error
        if objective_mode not in ("single", "multi"):
            raise ValueError("objective_mode must be 'single' or 'multi'")
        self.objective_mode = objective_mode
        if not isinstance(stop_on_recommendation, bool):
            raise ValueError("stop_on_recommendation must be a boolean")
        if objective_mode != "single" and stop_on_recommendation:
            raise ValueError("stop_on_recommendation requires single mode")
        self.stop_on_recommendation = stop_on_recommendation
        if recommendation_beta is None:
            recommendation_beta = 0.0 if objective_mode == "single" else 1.0
        self.prior_mean = float(prior_mean)
        if not math.isfinite(self.prior_mean):
            raise ValueError("prior_mean must be finite")
        self.cost_scaling_factor = _positive(cost_scaling_factor, "cost_scaling_factor")
        for name, value in (("lambda_cost", lambda_cost), ("lambda_latency", lambda_latency),
                            ("recommendation_beta", recommendation_beta), ("stop_tolerance", stop_tolerance)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if objective_mode == "multi" and (lambda_cost or lambda_latency):
            raise ValueError("multi mode keeps accuracy and cost separate; lambda weights require single mode")
        if warm_start_batch_size is not None:
            legacy_size = _integer(warm_start_batch_size, "warm_start_batch_size")
            if batch_size is None:
                batch_size = legacy_size
            elif legacy_size != batch_size:
                raise ValueError("warm_start_batch_size must match batch_size. Use warm_start_max_concurrent "
                                 "in select_best() to change warm-up concurrency.")
        self.batch_size = _integer(DEFAULT_GITTINS_BATCH_SIZE if batch_size is None else batch_size, "batch_size")
        self.warm_start_batch_size = self.batch_size  # Preserve the normalized checkpoint settings.
        self.grid_size = _integer(grid_size, "grid_size", 17)
        self.seed = _integer(seed, "seed", 0)
        self.min_samples = _integer(recommendation_min_samples, "recommendation_min_samples", 0)
        self.beta = float(recommendation_beta)
        self.stop_tolerance = float(stop_tolerance)
        low, high = map(float, score_bounds)
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("score_bounds must be finite with upper > lower")
        self.score_bounds = (low, high)
        self.prior_variance = _positive(prior_variance, "prior_variance")
        self.score_noise_variance = _positive(score_noise_variance, "score_noise_variance")
        self.cost_reference = None if cost_reference_usd is None else _positive(cost_reference_usd, "cost_reference_usd")
        self.cost_scale = _positive(search_cost_scale_eta, "search_cost_scale_eta")
        self.eta_initial = _positive(eta_initial, "eta_initial")
        self.eta_decay = float(eta_decay)
        if not math.isfinite(self.eta_decay) or not 0 < self.eta_decay < 1:
            raise ValueError("eta_decay must lie in (0, 1)")
        self.cost_bin_ratio = effective_cost_bin_ratio
        if effective_cost_bin_ratio is not None and _positive(effective_cost_bin_ratio, "effective_cost_bin_ratio") <= 1:
            raise ValueError("effective_cost_bin_ratio must exceed one or be None")
        self.cost_bin_anchor = _positive(effective_cost_bin_anchor, "effective_cost_bin_anchor")
        fraction = float(observation_budget_fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("observation_budget_fraction must lie in (0, 1]")
        self.max_spend = None if max_search_cost_usd is None else _positive(max_search_cost_usd, "max_search_cost_usd")
        # Validate all budget dimensions before BaseModelSelector starts tracking.
        if dataset is None or models is None:
            raise TypeError("'models' and 'dataset' are required")
        if not models or any(not candidates for candidates in models.values()):
            raise ValueError("Every model node must have at least one candidate")
        self.warm_size = min(self.warm_start_batch_size, len(dataset))
        n_arms = math.prod(len(candidates) for candidates in models.values())
        self.cell_cap = math.ceil(fraction * n_arms * len(dataset))
        if max_total_question_evaluations is not None:
            self.cell_cap = min(self.cell_cap, _integer(max_total_question_evaluations, "max_total_question_evaluations"))
        if self.cell_cap < n_arms * self.warm_size:
            raise ValueError("The cell budget cannot fund the mandatory uniform warm start")
        # Explicit workflow identity covers prompts, evaluator versions, provider
        # endpoints and other configuration outside the model/dataset arguments.
        self._journal = None
        self.restored_cells = 0
        inferred_workflow = None
        automatic = checkpoint_path is AUTO_CHECKPOINT
        if automatic:
            if checkpoint_id is None:
                try:
                    checkpoint_id = inferred_workflow = workflow_identity(agent, eval_fn)
                except (ValueError, TypeError, RecursionError):
                    # A fresh ID prevents silently restoring another runtime configuration.
                    checkpoint_id = uuid4().hex
                    warnings.warn(f"Could not infer a stable workflow identity. This run will save a "
                                  f"new checkpoint. To resume it, pass checkpoint_id='{checkpoint_id}' "
                                  "from this working directory with the same workflow.", RuntimeWarning,
                                  stacklevel=2)
            try:
                fingerprint([[dataset[i] for i in range(len(dataset))], list(models.items()), model_prices])
            except (ValueError, TypeError, RecursionError):
                warnings.warn("Automatic checkpoint unavailable for these dataset or model values. "
                              "Use plain JSON values or tuples for checkpointing, or checkpoint_path=None "
                              "to run without it.", RuntimeWarning, stacklevel=2)
                checkpoint_path = None
        self.checkpoint_path = checkpoint_path
        if checkpoint_path is not None:
            # Own the input graph so edits to caller-owned containers cannot
            # change what a checkpoint identity represents.
            dataset = deepcopy([dataset[i] for i in range(len(dataset))])
            models = deepcopy(models)
            model_prices = deepcopy(model_prices)
            if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
                raise ValueError("checkpoint_id is required: give a stable workflow/evaluator version")
            settings = {key: value for key, value in vars(self).items()
                        if key not in {"_journal", "restored_cells", "checkpoint_path"}}
            identity_inputs = {
                "format": 1, "algorithm": "gittins-v1", "workflow": checkpoint_id,
                "dataset": [dataset[i] for i in range(len(dataset))],
                "models": list(models.items()), "settings": settings,
                "lambda_cost": float(lambda_cost), "lambda_latency": float(lambda_latency),
                "prices": _ordered_prices(model_prices), "price_table": MODEL_PRICES,
            }

            def identity_for_limits(cell_cap, max_spend):
                return fingerprint({**identity_inputs, "settings": {
                    **settings, "cell_cap": cell_cap, "max_spend": max_spend,
                }})

            if automatic:
                # The journal validates accumulated limits separately on every reopen.
                self.checkpoint_path = default_checkpoint_path(identity_for_limits(None, None))
            self._identity_for_limits = identity_for_limits
            self._checkpoint_input_identity = fingerprint({"dataset": dataset, "models": list(models.items()), "prices": _ordered_prices(model_prices)})
            self._checkpoint_setting_keys = tuple(settings)
            self._checkpoint_settings_identity = fingerprint(settings)
        self._inferred_workflow = inferred_workflow
        super().__init__(agent=agent, models=models, eval_fn=eval_fn, dataset=dataset,
                         model_prices=model_prices, tracker=tracker,
                         lambda_cost=lambda_cost, lambda_latency=lambda_latency)
        if checkpoint_path is not None:
            self._frozen_checkpoint_settings = deepcopy({key: getattr(self, key)
                for key in (*self._checkpoint_setting_keys, "lambda_cost", "lambda_latency")})

    def __setattr__(self, name, value):
        frozen = self.__dict__.get("_frozen_checkpoint_settings", {})
        if name in frozen and (type(value) is not type(frozen[name]) or value != frozen[name]):
            raise CheckpointError(f"Checkpoint setting '{name}' is immutable; create a new selector for changed settings")
        super().__setattr__(name, value)

    def select_best(self, parallel=None, max_concurrent=None, *, show_ui=None, warm_start_max_concurrent=None,
                    on_update=None, stop_event=None, _warm_resume=None):
        """Run once, optionally opening the multi-objective live view.

        Gittins defaults to batch_size rolling warm-up slots. Use parallel=False
        for a serial run, or warm_start_max_concurrent to change warm-up only.
        Each adaptive batch can run all its questions together. An explicit
        max_concurrent limits both phases. A dollar cap reserves estimated spend
        for in-flight cells after an initial price probe.

        By default, multi-objective Gittins opens its window at a local Python
        prompt with a desktop backend. Scripts can opt in with show_ui=True.
        A configured LiveParetoPlot can also be supplied as show_ui for custom
        titles and exports. Its owner controls the window lifetime.
        """
        # Random search also inherits this entry point, but has no warm phase.
        has_warmup = hasattr(self, "warm_size")
        if parallel is None:
            parallel = has_warmup
        if warm_start_max_concurrent is None:
            warm_start_max_concurrent = self.batch_size if has_warmup else 1
        if show_ui is None:
            from ..visualization.live_pareto import local_desktop_available
            show_ui = has_warmup and self.objective_mode == "multi" and local_desktop_available()
        if show_ui is False:
            return self._select_with_tracking(parallel, max_concurrent, on_update, stop_event,
                                              warm_start_max_concurrent=warm_start_max_concurrent,
                                              warm_resume=_warm_resume)
        if self.objective_mode != "multi" or not hasattr(self, "warm_size"):
            raise ValueError("The live UI requires multi-objective Gittins")
        from threading import current_thread, main_thread
        from ..visualization import LiveParetoPlot
        if current_thread() is not main_thread():
            raise RuntimeError("The live UI must be opened on the main thread")
        owns_view = show_ui is True
        if owns_view:
            export_dir = Path.cwd() / ".agentopt_cache" / "exports" / uuid4().hex
            plot = LiveParetoPlot(output=export_dir / "frontier.png", total_questions=len(self.dataset),
                                  warm_total=len(self._all_combos()) * self.warm_size,
                                  stop_event=stop_event)
        elif isinstance(show_ui, LiveParetoPlot):
            plot = show_ui
            if stop_event is not None and stop_event is not plot.stop_event:
                raise ValueError("Use the same stop_event for the plot and selector")
        else:
            raise TypeError("show_ui must be a boolean or a LiveParetoPlot")
        try:
            if owns_view:
                export_dir.mkdir(parents=True, exist_ok=False)
            results = plot.run(self, parallel=parallel, max_concurrent=max_concurrent, on_update=on_update,
                               warm_start_max_concurrent=warm_start_max_concurrent)
            if owns_view:
                output = export_dir / "results.json"
                save_json(output, results.model_dump(mode="json"))
                plot.show_saved_results(output)
                plot.wait_for_close()
            return results
        finally:
            if owns_view:
                plot.plt.close(plot.figure)

    def _run_selection(self, parallel=False, max_concurrent=None, *, warm_start_max_concurrent=1, warm_resume=None):
        warm_width = _integer(warm_start_max_concurrent, "warm_start_max_concurrent")
        if warm_width != 1 and not hasattr(self, "warm_size"):
            raise ValueError("Warm-up concurrency requires Gittins")
        if max_concurrent is None:
            concurrency = self.batch_size if hasattr(self, "warm_size") else 20
        else:
            concurrency = _integer(max_concurrent, "max_concurrent")
            warm_width = min(warm_width, concurrency)
        self._warm_width = warm_width if parallel else 1
        self._warm_resume = warm_resume
        self._warm_inflight = 0
        self._run_dataset = self.dataset
        self._run_combos = self._all_combos()
        if self.checkpoint_path is not None:
            if self._inferred_workflow is not None:
                try:
                    current_workflow = workflow_identity(self.agent, self.eval_fn)
                except (ValueError, TypeError, RecursionError) as exc:
                    raise CheckpointError("Workflow changed after construction; create a new selector") from exc
                if current_workflow != self._inferred_workflow:
                    raise CheckpointError("Workflow changed after construction; create a new selector")
            self._run_dataset = deepcopy([self.dataset[i] for i in range(len(self.dataset))])
            models = deepcopy(self._models)
            current = fingerprint({"dataset": self._run_dataset, "models": list(models.items()), "prices": _ordered_prices(self.model_prices)})
            current_settings = fingerprint({key: getattr(self, key) for key in self._checkpoint_setting_keys})
            if current != self._checkpoint_input_identity or current_settings != self._checkpoint_settings_identity:
                raise CheckpointError("Checkpoint inputs changed after selector construction; create a new selector for the changed dataset or models")
            self._run_combos = deepcopy(self._all_combos())
        if self.checkpoint_path is None:
            try:
                return asyncio.run(self._select(parallel, concurrency))
            except _PartialBatchStopped as stopped:
                return stopped.result
        with EvaluationCheckpoint(self.checkpoint_path, self._identity_for_limits,
                                  limits=(self.cell_cap, self.max_spend)) as journal:
            journal.validate(names=[self._combo_name(combo) for combo in self._run_combos],
                             questions=len(self._run_dataset), score_bounds=self.score_bounds,
                             prices=self._custom_prices, warm_cells=len(self._run_combos) * self.warm_size)
            self._journal = journal
            self.restored_cells = journal.saved_cells
            try:
                result = asyncio.run(self._select(parallel, concurrency))
                if journal.batch_index < journal.saved_batches:
                    raise CheckpointError(
                        f"Checkpoint evaluation order differs in '{journal.path}': "
                        f"selection ended ({result.stop_reason}) before all saved batches could be replayed"
                    )
                return result
            except _PartialBatchStopped as stopped:
                # Partial restores are only published on the last saved batch.
                return stopped.result
            finally:
                self._journal = None

    def _publish_update(self, result, *, final=False):
        # Do not animate the historical warm start or let callbacks stop its
        # reconstruction. The first update contains all restored observations.
        if self._journal is not None and self._journal.batch_index < self._journal.saved_batches and not final:
            return result
        return super()._publish_update(result, final=final)

    def _selection_cost_so_far(self):
        # Charged calls only, accumulated before tracker archival.
        return float(self._spent + self._failed_spend)

    def _token_charge(self, model, input_tokens, output_tokens):
        rates = get_model_price(model, custom_prices=self._custom_prices)
        if rates is None or any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError(f"Cannot account for Gittins search cost for {model!r}; supply model_prices")
        # Rates and budget limits are decimal USD. Preserve that value through
        # aggregation and batch prediction; SDK results and saved cells use floats.
        input_rate, output_rate = (Fraction(str(rate)) for rate in rates)
        return (input_rate * input_tokens + output_rate * output_tokens) / 1_000_000

    def _record_charge(self, record):
        return self._token_charge(record.model, record.prompt_tokens, record.completion_tokens)

    def _record_failed_attempt(self, arm, question):
        attempt_id = f"{self._names[arm]}::gittins_{self._run_id}::dp_{question + 1}"
        charge = Fraction(0)
        unmeasured = False
        for record in self._tracker.get_records():
            if record.data_id != attempt_id or record.cached:
                continue
            if getattr(record, "request_not_sent", False):
                continue
            if getattr(record, "usage_unavailable", False) or getattr(record, "transport_charge_unknown", False):
                unmeasured = True
            else:
                try:
                    charge += self._record_charge(record)
                except ValueError:
                    unmeasured = True
        if charge or unmeasured:
            self._save_attempt_charge(attempt_id, charge, unmeasured)

    def _save_attempt_charge(self, attempt_id, charge, unmeasured):
        if self._journal is not None:
            self._journal.save_failed_attempt(attempt_id, charge, unmeasured)
        self._failed_spend += charge
        self._unmeasured_attempts += int(unmeasured)

    def _restored_cell(self, item):
        dp = DatapointResult.model_validate(item["datapoint"])
        return dp, item["cost"], item["charge"]

    def _update_spend(self):
        records = self._tracker.get_records()
        new_records = records[self._record_cursor:]
        self._record_cursor = len(records)
        for record in new_records:
            if record.cached:
                continue
            self._spent += self._record_charge(record)

    async def _evaluate_cell(self, arm, question, agent, parallel):
        try:
            return await self._evaluate_new_cell(arm, question, agent, parallel)
        except Exception:
            self._record_failed_attempt(arm, question)
            raise

    async def _evaluate_new_cell(self, arm, question, agent, parallel):
        label = f"{self._names[arm]}::gittins_{self._run_id}"
        tasks = [deepcopy(self._run_dataset[question]) if self._journal else self._run_dataset[question]]
        if parallel or self._is_async_run:
            outcome = await self._evaluate_agent_async(agent, tasks, label, max_concurrent=1, dp_offset=question)
        else:
            # A synchronous SDK may run its own event loop. Await one cell at
            # a time while keeping its tracking context in the worker thread.
            outcome = await asyncio.to_thread(self._evaluate_agent, agent, tasks, label, dp_offset=question)
        scores, latencies, dp_ids = outcome
        if len(scores) != 1:
            raise RuntimeError(f"Evaluation failed for {self._names[arm]} at dataset index {question}")
        cost = self._per_sample_costs(dp_ids)[0]
        low, high = self.score_bounds
        if not math.isfinite(scores[0]) or not low <= scores[0] <= high:
            raise ValueError("eval_fn score must be finite and within score_bounds")
        if cost is None or not math.isfinite(cost) or cost < 0:
            raise ValueError("Gittins requires known nonnegative per-evaluation costs; supply model_prices")
        if not math.isfinite(latencies[0]) or latencies[0] < 0:
            raise ValueError("Evaluation latency must be finite and nonnegative")
        dp = self._build_datapoint_results(scores, latencies, dp_ids)[0]
        if not dp.input_tokens and not dp.output_tokens:
            raise ValueError("No LLM token usage was recorded; Gittins requires tracked evaluations")
        dp.datapoint_index = question
        records = [record for record in self._tracker.get_records()
                   if record.data_id in dp_ids and not record.cached]
        charge = float(sum(self._record_charge(record) for record in records
                           if not getattr(record, "request_not_sent", False)
                           and not self._charge_unknown(record)))
        # A provider SDK can retry a timed-out call inside an otherwise successful cell.
        if any(self._charge_unknown(record) for record in records):
            self._save_attempt_charge(dp_ids[0], 0, True)
        return dp, cost, charge

    @staticmethod
    def _cell_data(observation):
        dp, cost, charge = observation
        return {"datapoint": dp.model_dump(mode="json"), "cost": cost, "charge": charge}

    def _record_cell(self, arm, observation, position):
        dp, cost, charge = observation
        self._datapoints[arm].insert(position, dp)
        self._scores[arm].insert(position, dp.score)
        self._costs[arm].insert(position, cost)
        self._latencies[arm].insert(position, dp.latency_seconds)
        self._positions[arm] += 1
        self._evaluations += 1
        self._spent += Fraction(str(charge))

    async def _pull(self, arm, question_ids, parallel, max_concurrent, expected_charge=0):
        """Persist each successful cell before advancing the adaptive batch."""
        saved = self._journal.begin_batch(self._names[arm], question_ids) if self._journal else {}
        if self._journal and self._journal.batch_index <= self._journal.saved_batches and len(saved) < len(question_ids):
            self._publish_partial_batch(arm, question_ids, saved, expected_charge)
        # Constructing an agent can itself initialize clients; avoid it entirely
        # when every cell in the batch already exists.
        agent = self.agent(deepcopy(self._combos[arm]) if self._journal else self._combos[arm]) if len(saved) < len(question_ids) else None
        async def evaluate(question):
            if question in saved:
                return self._restored_cell(saved[question])
            if self._unknown_cost_stop():
                return None
            observation = await self._evaluate_cell(arm, question, agent, parallel)
            if self._journal:
                self._journal.save_cell(question, self._cell_data(observation))
            return observation

        if parallel:
            semaphore = asyncio.Semaphore(max_concurrent)
            async def limited(question):
                async with semaphore:
                    return await evaluate(question)
            # Drain other successful tasks if one fails, so their completed cells
            # are saved before the selector releases its checkpoint lock.
            evaluated = await asyncio.gather(*(limited(q) for q in question_ids), return_exceptions=True)
            for item in evaluated:
                if isinstance(item, BaseException):
                    raise item
        else:
            evaluated = [await evaluate(q) for q in question_ids]
        if any(observation is None for observation in evaluated):
            completed = {q: self._cell_data(obs) for q, obs in zip(question_ids, evaluated) if obs is not None}
            self._publish_partial_batch(arm, question_ids, completed, expected_charge, stop_reason="search_cost_unknown")
        before = self._spent
        for observation in evaluated:
            self._record_cell(arm, observation, len(self._datapoints[arm]))
        return self._spent - before

    def _publish_partial_batch(self, arm, question_ids, saved, expected_charge, *, stop_reason=None):
        """Include the measured part of a batch in a restore or final snapshot."""
        count = len(saved)
        before_spent = self._spent
        posterior = deepcopy(self._posteriors[arm]) if self._calibrated else None
        scalar = deepcopy(self._scalar_posteriors[arm]) if self._calibrated else None
        for question in question_ids:
            if question not in saved:
                continue
            dp, cost, charge = self._restored_cell(saved[question])
            self._datapoints[arm].append(dp)
            self._scores[arm].append(dp.score)
            self._costs[arm].append(cost)
            self._latencies[arm].append(dp.latency_seconds)
            self._spent += Fraction(str(charge))
        self._evaluations += count
        self._positions[arm] += count
        if count and self._calibrated:
            observations = self._calibration.observations(self._scores[arm][-count:], self._costs[arm][-count:])
            self._posteriors[arm].update(observations.mean(axis=0), self._calibration.question_noise, count)
            values = self._scalar_observations(arm)[-count:]
            self._scalar_posteriors[arm].update(np.array([np.mean(values)]), np.array([self._scalar_noise]), count)
        snapshot = self._results(stop_reason or "restoring_batch", self._active_eta, calibrated=self._calibrated)
        snapshot.pending_batch_evaluations = len(question_ids) - count
        if stop_reason is not None:
            raise _PartialBatchStopped(snapshot)
        self._publish_update(snapshot)
        if self._should_stop():
            snapshot.stop_reason = "user_stopped"
            raise _PartialBatchStopped(snapshot)
        if self.max_spend is not None:
            remaining = snapshot.pending_batch_evaluations
            if self._unknown_cost_stop():
                snapshot.stop_reason = "search_cost_unknown"
                raise _PartialBatchStopped(snapshot)
            if self._budget_stop(remaining, expected_charge * remaining) == "search_cost_budget":
                snapshot.stop_reason = "search_cost_budget" if self._calibrated else "warm_start_budget"
                raise _PartialBatchStopped(snapshot)
        # The algorithm will incorporate the entire batch once; this temporary
        # view must not update its posterior twice when the missing cells finish.
        if count:
            for values in (self._datapoints, self._scores, self._costs, self._latencies):
                del values[arm][-count:]
        self._evaluations -= count
        self._positions[arm] -= count
        self._spent = before_spent
        if self._calibrated:
            self._posteriors[arm], self._scalar_posteriors[arm] = posterior, scalar

    def _unknown_cost_stop(self):
        return self.max_spend is not None and (self._unmeasured_attempts > 0 or self._live_unknown_charge_attempts() > 0)

    def _budget_stop(self, count, predicted_charge=0):
        # Reconstruct the prefix before honoring a new stop request. The original
        # scheduling budgets still apply to each batch, including a partial one.
        restoring = self._journal is not None and self._journal.batch_index < self._journal.saved_batches
        if not restoring and self._should_stop():
            return "user_stopped"
        if not restoring and self._unknown_cost_stop():
            return "search_cost_unknown"
        cell_cap, max_spend = self._journal.saved_limits if restoring else (self.cell_cap, self.max_spend)
        if self._evaluations + count > cell_cap:
            return "question_budget"
        if max_spend is not None:
            limit = Fraction(str(max_spend))
            spent = self._spent if restoring else self._spent + self._failed_spend
            if spent >= limit or spent + predicted_charge > limit:
                return "search_cost_budget"
        return None

    def _direction_status(self, indices, completed, direction):
        # G0 uses an ordered, tolerance-aware fold for unfinished arms.
        arm, best_index = None, -math.inf
        for candidate in sorted(indices):
            if indices[candidate] > best_index + self.stop_tolerance:
                arm, best_index = candidate, indices[candidate]
        completed_arm, completed_index = None, -math.inf
        if completed:
            terminal = self._completed_states[direction]
            for i in completed:
                if i not in terminal:
                    secondary = (-float(np.mean(self._costs[i]))
                                 if self.objective_mode == "single" or direction == 1
                                 else float(np.mean(self._scores[i])))
                    terminal[i] = (self._active_state(i, direction)[0], secondary)
            values = {i: terminal[i][0] for i in completed}
            maximum = max(values.values())
            tied = [i for i in completed if abs(values[i] - maximum) <= self.stop_tolerance]
            completed_arm = max(tied, key=lambda i: (terminal[i][1], -i))
            completed_index = values[completed_arm]
        return arm, best_index, completed_arm, completed_index

    def _scalar_observations(self, arm):
        low, high = self.score_bounds
        objective = np.asarray(self._scores[arm], dtype=float).copy()
        for values, bounds, weight in ((self._costs[arm], self._fixed_cost_bounds, self.lambda_cost),
                                       (self._latencies[arm], self._fixed_latency_bounds, self.lambda_latency)):
            lo, hi = bounds
            if weight and hi > lo:
                objective -= weight * (np.asarray(values) - lo) / (hi - lo)
        return (objective - low) / (high - low)

    def _active_state(self, arm, direction):
        if self.objective_mode == "single":
            values = self._scalar_observations(arm)
            mean, _ = finite_dataset_moments(
                self._scalar_posteriors[arm], np.array([values.sum()]),
                len(values), len(self._run_dataset), np.array([self._scalar_noise]),
            )
            # The full-dataset posterior-mean process is an affine transform
            # of the latent mean process. Transform transition variances,
            # retaining the original monetary scale of evaluation costs.
            scale = 1.0 + self._scalar_noise / (len(self._run_dataset) * self.prior_variance)
            return float(mean[0]), self._initial_scalar_var * scale ** 2, self._scalar_noise * scale ** 2
        if direction == 0:
            scale = self._calibration.cost_reference_usd
            return (1.0 - self._posteriors[arm].mean[1] / scale,
                    self._initial_var[1] / scale ** 2, self._calibration.question_noise[1] / scale ** 2)
        return self._posteriors[arm].mean[0], self._initial_var[0], self._calibration.question_noise[0]

    async def _warm_up(self, charges, eta):
        # One manifest per seeded cell, independent of worker count. A crash can
        # leave holes when later cells finish first. Restore all durable work
        # before dispatching any of those holes or publishing the first update.
        journal = self._journal
        total = len(self._combos) * self.warm_size
        historical = min(journal.saved_batches, total) if journal else 0
        missing = []
        recorded_slots = [[] for _ in self._combos]
        width = min(total, self._warm_width)

        def cell(slot):
            arm, offset = divmod(slot, self.warm_size)
            return arm, self._orders[arm][offset]

        def record(slot, observation):
            arm, _ = cell(slot)
            # Keep even partially restored results in seeded order.
            positions = recorded_slots[arm]
            index = bisect_left(positions, slot)
            positions.insert(index, slot)
            self._record_cell(arm, observation, index)
            charges[arm] += Fraction(str(observation[2]))

        for slot in range(historical):
            # Several admitted cells can finish after a concurrent run crosses
            # its soft limit. Restore their actual charges before admitting new work.
            if not journal.parallel_warm_history and self._budget_stop(1):
                raise CheckpointError("Checkpoint evaluation order differs: saved warm start exceeds its budget")
            arm, question = cell(slot)
            saved = journal.begin_batch(self._names[arm], [question])
            if question in saved:
                record(slot, self._restored_cell(saved[question]))
            else:
                missing.append(slot)
        if historical and self._evaluations < total:
            self._publish_update(self._results("warming_up", eta, calibrated=False))

        async def evaluate(slot):
            arm, question = cell(slot)
            combo = self._combos[arm]
            agent = self.agent(deepcopy(combo) if journal else combo)
            observation = await self._evaluate_cell(arm, question, agent, width > 1)
            if journal:
                data = self._cell_data(observation)
                journal.save_cell_at(slot + 1, question, data)
            return observation

        if historical == total and not missing:
            return None
        pending = chain(missing, range(historical, total))
        active = {}
        next_slot = next(pending, None)
        reserved = Fraction(0)
        reason = None
        last_update_count = self._evaluations
        loop = asyncio.get_running_loop()
        last_update_time = loop.time()
        try:
            while active or (next_slot is not None and not reason):
                while next_slot is not None and len(active) < width and not reason:
                    if self._warm_resume is not None and not self._warm_resume.is_set():
                        break
                    slot = next_slot
                    arm, question = cell(slot)
                    estimate = 0
                    if self.max_spend is not None:
                        if not self._spent and active:
                            break  # Only one probe until a positive charge is known.
                        if charges[arm]:
                            estimate = charges[arm] / len(recorded_slots[arm])
                        elif self._evaluations:
                            estimate = self._spent / self._evaluations
                    decision = self._budget_stop(len(active) + 1, reserved + estimate)
                    if decision:
                        # A cheaper settlement may make this same cell affordable.
                        # Keep it at the front of the queue while other cells drain.
                        if not active or decision == "user_stopped":
                            reason = decision
                        break
                    if journal:
                        if slot >= historical:
                            journal.begin_batch(self._names[arm], [question])
                        if width > 1:
                            journal.reserve_warm_cell(slot + 1)
                    active[asyncio.create_task(evaluate(slot))] = (slot, estimate)
                    reserved += estimate
                    next_slot = next(pending, None)
                if not active and (next_slot is None or reason) and self._evaluations == last_update_count:
                    break
                if active:
                    done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        observation = task.result()
                        slot, estimate = active.pop(task)
                        reserved -= estimate
                        record(slot, observation)
                self._warm_inflight = len(active)
                # Coalesce progress snapshots independently of dispatch: a slow
                # cell holds just its own slot. Pause drains active work before
                # the view waits for Continue; Finish also drains before return.
                if (not active or self._evaluations - last_update_count >= width
                        or loop.time() - last_update_time >= 0.2):
                    self._publish_update(self._results("warming_up", eta, calibrated=False))
                    last_update_count = self._evaluations
                    last_update_time = loop.time()
                if self._should_stop():
                    reason = "user_stopped"
        finally:
            # A failed cell or progress callback must not abandon the other
            # workers while they are still charging or writing the checkpoint.
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            self._warm_inflight = 0
        return None if self._evaluations == total else reason

    async def _select(self, parallel, max_concurrent):
        self._combos = self._run_combos
        self._names = [self._combo_name(combo) for combo in self._combos]
        if len(set(self._names)) != len(self._names):
            raise ValueError("Gittins requires distinct combination labels")
        n_arms, n_questions = len(self._combos), len(self._run_dataset)
        self._run_id = uuid4().hex
        self._scores = [[] for _ in range(n_arms)]
        self._costs = [[] for _ in range(n_arms)]
        self._latencies = [[] for _ in range(n_arms)]
        self._datapoints = [[] for _ in range(n_arms)]
        self._positions = np.zeros(n_arms, dtype=int)
        children = np.random.SeedSequence(self.seed).spawn(n_arms + 1)
        self._orders = [np.random.default_rng(child).permutation(n_questions).tolist() for child in children[1:]]
        self._record_cursor = len(self._tracker.get_records())
        self._spent, self._evaluations = 0, 0
        self._failed_spend = self._journal.failed_spend if self._journal else 0
        self._unmeasured_attempts = self._journal.unmeasured_attempts if self._journal else 0
        self._stop_recommended = False
        dimensions = 1 if self.objective_mode == "single" else 2
        # Completed posteriors and observed tie-break statistics never change.
        self._completed_states = [{} for _ in range(dimensions)]
        eta = [1.0] if self.objective_mode == "single" else [self.eta_initial] * dimensions
        self._active_eta, self._calibrated = eta, False
        warm_charges = [Fraction(0) for _ in range(n_arms)]
        reason = await self._warm_up(warm_charges, eta)
        if reason:
            return self._results(reason if reason in ("user_stopped", "search_cost_unknown") else "warm_start_budget",
                                 eta, calibrated=False)
        self._calibration = fit_warm_start(self._scores, self._costs, score_bounds=self.score_bounds,
                                          prior_variance=self.prior_variance,
                                          score_noise_variance=self.score_noise_variance,
                                          cost_reference_usd=self.cost_reference)
        self._posteriors = [self._calibration.posterior(s, c) for s, c in zip(self._scores, self._costs)]
        if self.objective_mode == "single":
            # A fixed accuracy prior also underlies reported finite-dataset
            # accuracy predictions; deployment weights affect only the scalar.
            for arm, posterior in enumerate(self._posteriors):
                low, high = self.score_bounds
                normalized = (np.mean(self._scores[arm]) - low) / (high - low)
                variance = 1.0 / (1.0 / self.prior_variance + self.warm_size / self.score_noise_variance)
                posterior.mean[0] = variance * (self.prior_mean / self.prior_variance + self.warm_size * normalized / self.score_noise_variance)
                posterior.variance[0] = variance
        self._initial_var = self._posteriors[0].variance.copy()
        self._fixed_cost_bounds = (float(np.min(self._costs)), float(np.max(self._costs)))
        self._fixed_latency_bounds = (float(np.min(self._latencies)), float(np.max(self._latencies)))
        warm_objectives = [self._scalar_observations(arm) for arm in range(n_arms)]
        scalar_prior_mean = self.prior_mean if self.objective_mode == "single" else float(np.mean(warm_objectives))
        score_span = self.score_bounds[1] - self.score_bounds[0]
        self._scalar_noise = self.score_noise_variance * (1 + (self.lambda_cost + self.lambda_latency) / score_span) ** 2
        self._scalar_posteriors = []
        for values in warm_objectives:
            posterior = GaussianPosterior(np.array([scalar_prior_mean]), np.array([self.prior_variance]))
            posterior.update(np.array([np.mean(values)]), np.array([self._scalar_noise]), self.warm_size)
            self._scalar_posteriors.append(posterior)
        self._initial_scalar_var = self._scalar_posteriors[0].variance[0]
        # Continuation penalties use observed workflow cost, as in research;
        # dollar reservations instead estimate new non-cached charges.
        single = self.objective_mode == "single"
        expected_costs = np.mean(self._costs, axis=1) * (1 if single else self.batch_size)
        multiplier = self.cost_scaling_factor if single else self.cost_scale
        base_penalties = np.maximum(multiplier * expected_costs, 1e-12)
        if not single and self.cost_bin_ratio is not None:
            exponents = np.rint(np.log(base_penalties / self.cost_bin_anchor) / math.log(self.cost_bin_ratio))
            base_penalties = self.cost_bin_anchor * np.power(self.cost_bin_ratio, exponents)
        charge_per_question = [charge / self.warm_size for charge in warm_charges]
        horizon = n_questions - self.warm_size if single else math.ceil((n_questions - self.warm_size) / self.batch_size)
        if single:
            eta = [1.0]
        stages = np.zeros(n_arms, dtype=int)
        cache = AxisGittinsBoundaryCache()
        # An arm's index changes only after observing that arm or changing
        # the current direction's eta. G0 uses the same invalidation rule.
        direction_indices = [{} for _ in range(dimensions)]
        direction, floor_stops = 0, set()
        self._active_eta, self._calibrated = eta, True
        if not single:
            self._publish_update(self._results("running", eta))
        while True:
            completed = [i for i in range(n_arms) if self._positions[i] == n_questions]
            unfinished = [i for i in range(n_arms) if self._positions[i] < n_questions]
            if not unfinished:
                self._stop_recommended = single
                reason = "search_cost_unknown" if self._unknown_cost_stop() else "all_arms_completed"
                break
            # Single mode recomputes its advice for the current observations,
            # including when a budget ends the run after the preceding batch.
            if not single:
                reason = self._budget_stop(1)
                if reason:
                    break
            indices = {}
            for arm in unfinished:
                if arm not in direction_indices[direction]:
                    mean, initial, noise = self._active_state(arm, direction)
                    table = cache.get(effective_pull_cost=eta[direction] * base_penalties[arm],
                                      initial_var=float(initial), obs_noise_var=float(noise if single else noise / self.batch_size),
                                      horizon=horizon, grid_size=self.grid_size)
                    direction_indices[direction][arm] = table.index(int(stages[arm]), float(mean))
                indices[arm] = direction_indices[direction][arm]
            arm, best_index, completed_arm, best_completed = self._direction_status(indices, completed, direction)
            direction_stop = bool(completed and best_completed >= best_index - self.stop_tolerance)
            if single:
                self._stop_recommended = direction_stop
                self._publish_update(self._results("running", eta))
                reason = self._budget_stop(1)
                if reason:
                    break
                if direction_stop and self.stop_on_recommendation:
                    reason = "gittins_stop"
                    break
            elif direction_stop:
                remaining_penalty = max((horizon - stages[i]) * eta[direction] * base_penalties[i] for i in unfinished)
                threshold = max(self.stop_tolerance, np.finfo(float).eps * max(1.0, abs(best_completed), abs(best_index)))
                next_eta = eta[direction] * self.eta_decay
                if remaining_penalty <= threshold or next_eta <= 0 or np.any(next_eta * base_penalties <= 0):
                    floor_stops.add(direction)
                else:
                    eta[direction] = next_eta
                    direction_indices[direction].clear()
                    floor_stops.discard(direction)
                if len(floor_stops) == dimensions:
                    reason = "direction_eta_numerical_floor"
                    break
                direction = (direction + 1) % dimensions
                continue
            start = int(self._positions[arm])
            questions = self._orders[arm][start:start + self.batch_size]
            reason = self._budget_stop(len(questions), charge_per_question[arm] * len(questions))
            if reason:
                break
            await self._pull(arm, questions, parallel, max_concurrent, charge_per_question[arm])
            observations = self._calibration.observations(self._scores[arm][-len(questions):], self._costs[arm][-len(questions):])
            self._posteriors[arm].update(observations.mean(axis=0), self._calibration.question_noise, len(questions))
            values = self._scalar_observations(arm)[-len(questions):]
            self._scalar_posteriors[arm].update(np.array([np.mean(values)]), np.array([self._scalar_noise]), len(questions))
            for local_indices in direction_indices:
                local_indices.pop(arm, None)
            stages[arm] += len(questions) if single else 1
            floor_stops.clear()
            direction = (direction + 1) % dimensions
            if not single:
                self._publish_update(self._results("running", eta))
        return self._results(reason, eta)

    def _results(self, reason, eta, calibrated=True):
        records = []
        for arm, name in enumerate(self._names):
            dps = self._datapoints[arm]
            if not dps:
                continue
            input_tokens, output_tokens = {}, {}
            for dp in dps:
                for target, tokens in ((input_tokens, dp.input_tokens), (output_tokens, dp.output_tokens)):
                    for model, count in tokens.items():
                        target[model] = target.get(model, 0) + count
            objective = None
            if calibrated and self.objective_mode == "single" and self._has_combined_objective:
                low, high = self.score_bounds
                objective = float(np.mean(self._scalar_observations(arm)) * (high - low) + low)
            records.append(self._make_result(model_name=name, combo=self._combos[arm],
                                            accuracy=float(np.mean(self._scores[arm])),
                                            latency_seconds=float(np.mean(self._latencies[arm])),
                                            input_tokens=input_tokens, output_tokens=output_tokens,
                                            attribute="combination", datapoint_results=dps, combined_objective=objective))
        candidates = []
        if calibrated:
            low, high = self.score_bounds
            for arm, name in enumerate(self._names):
                if len(self._scores[arm]) < self.min_samples:
                    continue
                observations = self._calibration.observations(self._scores[arm], self._costs[arm])
                mean, variance = finite_dataset_moments(self._posteriors[arm], observations.sum(axis=0),
                                                        len(observations), len(self._run_dataset), self._calibration.question_noise)
                std = np.sqrt(variance)
                accuracy, score_std = low + (high - low) * mean[0], (high - low) * std[0]
                if len(observations) == len(self._run_dataset):
                    accuracy = float(np.mean(self._scores[arm]))
                    mean[1] = float(np.mean(self._costs[arm]))
                objective = conservative_objective = None
                if self.objective_mode == "single":
                    values = self._scalar_observations(arm)
                    scalar_mean, scalar_var = finite_dataset_moments(self._scalar_posteriors[arm], np.array([values.sum()]),
                                                                    len(values), len(self._run_dataset), np.array([self._scalar_noise]))
                    objective = float(low + (high - low) * scalar_mean[0])
                    conservative_objective = objective - self.beta * (high - low) * math.sqrt(scalar_var[0])
                candidates.append(GittinsRecommendation(
                    model_name=name, combo={node: self._candidate_label(model) for node, model in self._combos[arm].items()},
                    observed_samples=len(observations), accuracy=float(accuracy), accuracy_std=float(score_std),
                    cost_usd=float(mean[1]), cost_std_usd=float(std[1]),
                    conservative_accuracy=float(accuracy - self.beta * score_std),
                    conservative_cost_usd=float(mean[1] + self.beta * std[1]),
                    objective=objective, conservative_objective=conservative_objective,
                ))
        if self.objective_mode == "single" and candidates:
            winner = max(candidates, key=lambda r: (r.conservative_objective, -r.conservative_cost_usd))
            recommendations = [winner]
            for record in records:
                record.is_best = record.model_name == winner.model_name
        elif candidates:
            positions = raw_pareto_front_indices([[r.conservative_accuracy, r.conservative_cost_usd] for r in candidates])
            recommendations = [candidates[i] for i in positions]
        else:
            recommendations = []
        return GittinsSelectionResults(results=records, recommendations=recommendations,
                                       objective_mode=self.objective_mode, stop_reason=reason,
                                       stop_recommended=self._stop_recommended,
                                       total_evaluations=self._evaluations,
                                       unmeasured_failed_attempts=self._unmeasured_attempts,
                                       search_cost_overshoot_usd=max(0.0, self._selection_cost_so_far() - self.max_spend) if self.max_spend is not None else 0.0,
                                       direction_eta=list(eta))
