# Offline Model Selection

> **Core idea:** Given a multi-agent system with N agent roles, each with M candidate models, find the best model combination by evaluating candidates on a dataset — optimizing for accuracy first, then latency, then cost.

---

## 1. Overview

Model selection is framed as a combinatorial optimization problem. A multi-agent system has several agent roles (e.g. "planner", "solver"). Each role has a list of candidate models. The search space is the Cartesian product of all candidate lists.

The user provides:
1. **`agent_fn`** — a factory that builds a runnable agent from a model combination
2. **`models`** — candidate models per agent role
3. **`eval_fn`** — a scoring function that compares expected and actual outputs
4. **`dataset`** — (input, expected_answer) pairs

The framework evaluates combinations, tracks token usage and latency via `agentproxy`, and returns ranked results.

```python
selector = ModelSelector(
    agent_fn=agent_maker,
    models={"planner": ["gpt-4o", "gpt-4o-mini"], "solver": ["gpt-4o", "gpt-4o-mini"]},
    eval_fn=lambda expected, actual: 1.0 if expected in str(actual) else 0.0,
    dataset=[("What is 2+2?", "4"), ...],
)
results = selector.select_best(parallel=True)
```

---

## 2. Core Abstractions

Defined in `agentopt/src/agentopt/base_models.py` and `model_selection/base.py`.

### Types

| Type | Definition | Purpose |
|------|-----------|---------|
| `ModelCandidate` | `Any` | A model: string name, config dict, or LLM instance |
| `ModelsConfig` | `Dict[str, List[ModelCandidate]]` | Node names → candidate lists |
| `Dataset` | `Sequence[Tuple[Any, Any]]` | (input_data, expected_answer) pairs |
| `EvalFn` | `Callable[[str, Any], Union[bool, float]]` | Scoring function |
| `AgentFn` | `Callable[[Dict[str, ModelCandidate]], Any]` | Factory: combo → runnable agent |

### Candidate labeling

`BaseModelSelector._candidate_label(candidate)` extracts a human-readable name:

1. **String** → used as-is
2. **Dict** → looks for keys `"model"`, `"model_name"`, `"id"`, `"name"` (in order)
3. **Object** → checks attributes `model`, `model_name`, `id`, `name`
4. **Fallback** → `TypeName@hexid`

This allows passing pre-built LLM instances (e.g. `ChatOpenAI(model="gpt-4o")`) as candidates — the label is extracted automatically.

---

## 3. Combo Generation

All combinations are generated via `itertools.product`:

```python
models = {
    "planner": ["gpt-4o", "gpt-4o-mini"],
    "solver": ["gpt-4o", "gpt-4o-mini"],
}
# → 4 combos:
#   {"planner": "gpt-4o",      "solver": "gpt-4o"}
#   {"planner": "gpt-4o",      "solver": "gpt-4o-mini"}
#   {"planner": "gpt-4o-mini", "solver": "gpt-4o"}
#   {"planner": "gpt-4o-mini", "solver": "gpt-4o-mini"}
```

**Display name format:** `"planner=gpt-4o + solver=gpt-4o-mini"`

---

## 4. Evaluation Pipeline

### Sequential evaluation

```
for each combo in combinations:
    agent = agent_fn(combo)                    # build agent
    for each (input_data, expected) in dataset:
        with tracker.track(data_id, combo_id):
            result = agent(input_data)         # run agent (LLM calls tracked)
        score = eval_fn(expected, result)       # score output
    aggregate scores → ModelResult
```

### Parallel evaluation

```python
async def _evaluate_agent_async(combo, datapoint, semaphore):
    async with semaphore:  # asyncio.Semaphore(max_concurrent)
        agent = agent_fn(combo)
        with tracker.track(data_id, combo_id):
            ctx = contextvars.copy_context()
            result = await loop.run_in_executor(None, ctx.run, agent, input_data)
        return eval_fn(expected, result)
```

Key details:
- `asyncio.Semaphore(max_concurrent)` controls concurrency
- `contextvars.copy_context()` ensures ContextVars propagate to executor threads
- Each `agent_fn` call produces a fresh, stateless agent

### Latency correction

When caching is active, wall-clock time underestimates true latency (cache hits are instant). The evaluator corrects for this:

```python
wall_clock = time.time() - start_time
cached_latency = tracker.get_cached_latency(data_id=dp_id)
latency = wall_clock + cached_latency   # fair comparison
```

---

## 5. Result Types

### `DatapointResult`

Per-datapoint metrics:

```python
class DatapointResult(BaseModel):
    datapoint_index: int
    score: float
    latency_seconds: float
    input_tokens: Dict[str, int]    # {model: tokens}
    output_tokens: Dict[str, int]
```

### `ModelResult`

Per-combination aggregate:

```python
class ModelResult(BaseModel):
    model_name: str                  # "planner=gpt-4o + solver=gpt-4o-mini"
    accuracy: float                  # mean of datapoint scores
    latency_seconds: float           # mean latency per datapoint
    input_tokens: Dict[str, int]     # {model: total_input_tokens}
    output_tokens: Dict[str, int]    # {model: total_output_tokens}
    is_best: bool
    datapoint_results: List[DatapointResult]

    @property
    def price(self) -> Optional[float]:  # total USD cost
```

### `SelectionResults`

Collection of all evaluated combinations:

| Method | Description |
|--------|-------------|
| `print_summary()` | Formatted table with rank, accuracy, latency, tokens, price |
| `get_best()` | `ModelResult` with highest accuracy |
| `get_best_combo()` | Parses best combo name → `{"planner": "gpt-4o", ...}` |
| `to_csv(path)` | Export all results as CSV |
| `export_config(path)` | Export best combo as YAML config |

### Best-combo tiebreaker

When multiple combos have equal accuracy:
1. **Accuracy** (higher is better)
2. **Latency** (lower is better)
3. **Price** (lower is better)

---

## 6. Selection Algorithms

All selectors inherit from `BaseModelSelector` and implement `select_best(parallel, max_concurrent) -> SelectionResults`.

### 6.1 Brute Force

**File:** `model_selection/brute_force.py`

Evaluates every combination in the Cartesian product.

- **Complexity:** O(C × N) where C = number of combos, N = dataset size
- **Parallel:** Yes, via `asyncio.gather` over all combos
- **Best for:** Small search spaces where exhaustive evaluation is affordable

```python
selector = BruteForceModelSelector(agent_fn=..., models=..., eval_fn=..., dataset=...)
results = selector.select_best(parallel=True, max_concurrent=20)
```

### 6.2 Random Search

**File:** `model_selection/random_search.py`

Samples a random subset of combinations and evaluates only those.

- **Parameters:**
  - `sample_fraction: float = 0.25` — fraction of total combos to evaluate
  - `seed: Optional[int]` — for reproducibility
- **Samples:** `ceil(total_combos × sample_fraction)` combinations
- **Parallel:** Yes
- **Best for:** Large search spaces where exhaustive evaluation is too expensive

```python
selector = RandomSearchModelSelector(..., sample_fraction=0.3, seed=42)
```

### 6.3 Hill Climbing with Random Restarts

**File:** `model_selection/hill_climbing.py`

Greedy local search using model topology to define neighbors. Starts from a random combination and iteratively moves to a better neighbor.

**Algorithm:**

```
for each restart:
    combo = random_unseen_combination()
    for each iteration:
        evaluate(combo)
        if accuracy < 1.0:
            try move to higher-quality neighbor
        if no quality move:
            try move to faster neighbor
        if no move available or patience exhausted:
            stop this restart
    track best combo across restarts
```

**Neighbor definition** (from `model_topology.py`):
- **Quality neighbor:** the next higher-quality model in `QUALITY_RANKING` among the candidates for that node
- **Speed neighbor:** the next faster model in `SPEED_RANKING` among the candidates for that node

**Parameters:**
- `max_iterations: int = 20` — max iterations per restart
- `num_restarts: int = 3` — number of random restarts
- `patience: int = 3` — stop after this many iterations without improvement
- `seed: Optional[int]` — for reproducibility

**Parallel:** No (sequential only). Caches evaluations in `_eval_cache` to avoid recomputation across restarts.

**Best for:** Large search spaces where model quality/speed rankings are meaningful. Efficient when the topology accurately reflects real performance.

### 6.4 Successive Arm Elimination

**File:** `model_selection/arm_elimination.py`

Treats each combination as a bandit arm. Evaluates all arms on growing batches of data and eliminates statistically dominated arms each round.

**Algorithm:**

```
active = all combinations
offset = 0
batch_size = n_initial

while active > 1 and data remaining:
    batch = dataset[offset : offset + batch_size]
    evaluate all active arms on batch
    for each pair (i, j) in active:
        if arm i is dominated by arm j:
            eliminate arm i
    offset += batch_size
    batch_size *= growth_factor
```

**Dominance test:**

Arm i is dominated by arm j when:

```
μ_i + confidence × SE_i < μ_j − confidence × SE_j
```

where `μ` is the mean score, `SE = σ / √n` is the standard error, and `confidence` controls the elimination threshold.

- A higher `confidence` value requires stronger evidence to eliminate (more conservative).
- A lower `confidence` value eliminates more aggressively (risk of eliminating good arms).

**Parameters:**
- `n_initial: Optional[int]` — initial batch size (default: `max(1, dataset_size // 10)`)
- `growth_factor: float = 2.0` — batch size multiplier each round
- `confidence: float = 1.0` — elimination threshold (number of standard errors)

**Parallel:** Yes, via `asyncio.gather` over active arms per round.

**Best for:** Medium-to-large search spaces where you want to quickly discard poor combinations without evaluating them on the full dataset.

### 6.5 Bayesian Optimization

**File:** `model_selection/bayesian_optimization.py`

Uses a Gaussian Process surrogate model to predict combination performance and an acquisition function to select the most promising unseen combination to evaluate next.

**Algorithm:**

```
1. Randomly evaluate `2 * (n_nodes + 1)` combinations (clamped by the sample budget)
2. For each BO iteration:
    a. Fit GP (MixedSingleTaskGP) to all evaluations
    b. Compute LogExpectedImprovement for all unseen combos
    c. Evaluate the combo with highest EI
    d. Add result to training data
3. Return best combination found
```

**Key components:**
- **Surrogate model:** BoTorch `MixedSingleTaskGP` — a Gaussian Process that handles categorical inputs (each node's model choice is a categorical dimension)
- **Acquisition function:** `LogExpectedImprovement` — balances exploration (high uncertainty) and exploitation (high predicted value)
- **Training:** `ExactMarginalLogLikelihood` fitted via `fit_gpytorch_mll`

**Parameters:**
- `sample_fraction: float = 0.25` — fraction of total combinations to evaluate (includes the initial random evaluations)
- Initial random evaluations use `2 * (n_nodes + 1)` and are clamped by the sample budget.

**Dependencies:** Requires `torch`, `botorch`, `gpytorch`. Install via `pip install "agentopt[bayesian]"`.

**Parallel:** No (sequential only — GP fitting is inherently sequential).

**Best for:** Small-to-medium search spaces with expensive evaluations, where each evaluation is costly and you want to minimize the total number of evaluations.

---

## 7. Model Topology

**File:** `agentopt/src/agentopt/model_topology.py`

Pre-built rankings that encode general-impression ordering of LLM models. Used by hill climbing to define neighbor moves.

### Quality Ranking (best → worst)

```
openai/o3 > openai/o4-mini > openai/o3-mini
> openai/gpt-5.2 > openai/gpt-5.1 > openai/gpt-4.1 > openai/gpt-4o > openai/gpt-4.1-mini > openai/gpt-4o-mini > openai/gpt-4.1-nano
> anthropic/claude-opus-4 > anthropic/claude-sonnet-4 > anthropic/claude-3.7-sonnet > anthropic/claude-3.5-sonnet > anthropic/claude-3.5-haiku > anthropic/claude-3-haiku
> google/gemini-2.5-pro > google/gemini-2.5-flash > google/gemini-2.0-flash > google/gemini-2.0-flash-lite
```

### Speed Ranking (fastest → slowest)

```
openai/gpt-4.1-nano > google/gemini-2.0-flash-lite > openai/gpt-4o-mini > anthropic/claude-3-haiku
> anthropic/claude-3.5-haiku > openai/gpt-4.1-mini > google/gemini-2.0-flash > google/gemini-2.5-flash
> openai/gpt-4o > openai/gpt-4.1 > openai/o4-mini > openai/o3-mini
> ... > openai/o3
```

### Neighbor functions

```python
get_higher_quality_neighbor(current: str, candidates: List[str]) -> Optional[str]
get_faster_neighbor(current: str, candidates: List[str]) -> Optional[str]
```

Both functions sort the candidate list according to the relevant ranking, find the current model's position, and return the next better model. Returns `None` if already at the top.

Models not in the ranking are appended after known models in their original order.

Uses **OpenRouter naming convention** (`provider/model-name`).

---

## 8. Pricing

**File:** `agentopt/src/agentopt/model_price.py`
**Data:** `model_price.json` (repo root)

### Lookup strategy

1. **Exact match** in custom prices (if provided)
2. **Exact match** in built-in price table
3. **Suffix match** — strips provider prefix (e.g. `openai/gpt-4o` → `gpt-4o`)

### Cost computation

```python
compute_price(
    input_tokens: Dict[str, int],    # {model: total_input_tokens}
    output_tokens: Dict[str, int],   # {model: total_output_tokens}
    custom_prices: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[float]   # total USD, or None if any model price unknown
```

Prices are in **dollars per million tokens**.

### Custom pricing

Any selector accepts `model_prices` to override or extend the built-in table:

```python
selector = ModelSelector(
    ...,
    model_prices={
        "my-custom-model": {"input_price": 2.50, "output_price": 10.00},
    },
)
```

---

## 9. Package Structure

```
agentopt/
├── pyproject.toml
└── src/
    └── agentopt/
        ├── __init__.py                # Public API; ModelSelector = BruteForceModelSelector
        ├── base_models.py             # EvalFn, ModelCandidate, ModelsConfig, Dataset, AgentFn
        ├── model_price.py             # get_model_price(), compute_price(), MODEL_PRICES
        ├── model_topology.py          # QUALITY_RANKING, SPEED_RANKING, neighbor functions
        └── model_selection/
            ├── __init__.py
            ├── base.py                # BaseModelSelector, DatapointResult, ModelResult, SelectionResults
            ├── brute_force.py         # BruteForceModelSelector
            ├── random_search.py       # RandomSearchModelSelector
            ├── hill_climbing.py       # HillClimbingModelSelector
            ├── arm_elimination.py     # ArmEliminationModelSelector
            └── bayesian_optimization.py  # BayesianOptimizationModelSelector
```
