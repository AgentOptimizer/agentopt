# HotpotQA / MathQA missing-cell supplement

This branch was created from `cache-benchmark-run`, the branch that contains
the Bedrock application-inference-profile mappings, benchmark agents, cache
interceptor, and the original 200-sample run path.

## Audited scope

- HotpotQA: 32 cells across 13 configurations.
- MathQA: 1,239 cells across 63 configurations.
- Total: 1,271 cells.
- Every incomplete configuration is missing a contiguous suffix ending at
  question 199. `plan.csv` stores those 76 suffix ranges and the runner
  expands them to exact cells.
- The five lookup views (score, cost, input tokens, output tokens, total
  tokens) share the same missing mask.

Source lookup fingerprints:

- HotpotQA:
  `2e1260c986c4be0cf71fbaf31ce07d4b3fb2abbc6a599f09b246c4971d3ded8a`
- MathQA:
  `10b569c8371356bfc41bdfa4e84173d5a3d4f1352340acd45f3473390c65b9be`

Dataset identity:

- HotpotQA JSON SHA-256:
  `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea`
- MathQA Hugging Face snapshot:
  `c4f1cc784c04c4957b50c97858f23893b633eea6`

## Agent behavior is intentionally unchanged

The supplement uses the same benchmark factories and parameters as the
original run by default.  The outer MathQA reflection limit remains 3 and the
HotpotQA reflection limit remains 1.  One pathological MathQA row
(`answer=Qwen3 32B + critic=Ministral 3 8B`, question 187) repeatedly exceeded
Qwen's 32,768-token context window in the inner tool loop.  That row was rerun
with `--max-tool-rounds 3`; the exception is recorded in the completed CSV and
manifest.  All other rows retain the original unbounded inner-loop behavior.

The original cache reconstruction called brute force with
`parallel=True, max_concurrent=20`.  In global-concurrency mode that means
one configuration at a time with up to 20 datapoints concurrently.  The
supplement runner preserves that scheduling shape: configurations are
processed serially and missing datapoints within each configuration use
`--max-concurrent 20` by default.

## Safe workflow

The runner defaults to a local dry-run and makes no AWS calls:

```powershell
python experiments/run_missing_cells.py
python experiments/run_missing_cells.py --benchmark hotpotqa
```

Execution requires an explicit flag and the exact DAPLab resource-tag values:

```powershell
python experiments/run_missing_cells.py --execute --benchmark hotpotqa --project-tag "<project>" --billing-tag1 "<UNI>"
```

Before any model invocation, the script:

1. verifies AWS account `920736616554`;
2. requires region `us-east-1`;
3. verifies that every model resolves to an application inference profile;
4. calls the Bedrock control-plane tag API and requires exact `project` and
   `billing-tag1` values on every profile used by the selected cells.

Successful cells are appended and fsynced to
`experiments/missing_cells/output/<benchmark>.jsonl`.  Errors are retained
for audit but are retried on the next invocation.  Successful cells are
skipped, and API responses are persisted under
`.agentopt_cache/missing-cells`.

Use `--limit 1` for a paid smoke test after the dry-run and tag preflight.
The scripts do not create EC2 instances or other AWS resources.

### qx2278 bearer-token smoke test

The short-term Bedrock API key is read from
`AWS_BEARER_TOKEN_BEDROCK`; it is never stored in this repository. Copy
`profile_arns.example.json` to a locally named profile map (for example,
`profile_arns.local.json`) and fill in the account-specific profile ARNs. Local
profile maps are ignored by Git.

From the repository root, first run the single-cell plan without `--execute`:

```powershell
.\.venv\Scripts\python.exe experiments\run_missing_cells.py `
  --plan experiments\missing_cells\smoke_ministral_mathqa.csv `
  --profile-map experiments\missing_cells\profile_arns.local.json `
  --project-tag agentopt --billing-tag1 qx2278 --max-concurrent 1
```

Then repeat the command with `--preflight-only` to validate the bearer token,
account-scoped profile ARN, and both required resource tags without invoking a
model.  After both checks pass, replace `--preflight-only` with `--execute` to
invoke the model.  The smoke cell is a real missing MathQA cell (question 141),
so a successful result can remain in the normal supplemental output and will
be skipped by later resumptions.

## Merge without overwriting

The merge command checks the source pickle hash, refuses to overwrite any
existing cell, and writes a separate pickle:

```powershell
python experiments/merge_missing_cells.py --benchmark hotpotqa --source experiments/results/cache_db_results/hotpotqa_lookup.pkl --patch-jsonl experiments/missing_cells/output/hotpotqa.jsonl --output experiments/missing_cells/completed/hotpotqa_lookup.pkl
```

Run the same command with the MathQA paths after its supplemental run.  Do not
replace the original lookup until the reported remaining-missing count is zero
and the completed pickle has been independently audited.

## Audited complete 8x8 datasets

The historical lookup contains nine models, but the incomplete Claude 3 Haiku
row and column are outside the eight-model supplement.  Build the strict 8x8
submatrices (64 configurations x 200 datapoints) without modifying either
historical source pickle:

```powershell
python experiments/build_completed_8x8.py
```

This produces pipeline-compatible pickles, flat CSV exports, and a manifest
under `experiments/missing_cells/completed/`.  The builder verifies the source
hashes, refuses to overwrite historical cells during the merge, and fails if
any result cell is still missing.  The CSV `record_source` field distinguishes
historical and supplemental rows; optional execution metadata such as
`max_tool_rounds` is populated for supplemental rows when applicable.

## DeepSeek 9x9 completion

Four MathQA rows in `answer=Qwen3 32B + critic=DeepSeek R1` reached Qwen's
32,768-token context limit after already receiving successful Qwen and
DeepSeek responses. They are preserved in
`adjudicated_failures.9x9.json`, scored as incorrect, and retain the recorded
token, cost, client-latency, and server-latency values from those successful
responses. The rejected request itself returned no usage metadata and is not
estimated.

The supplemental runner now applies the same policy automatically: only an
explicit provider `ValidationException` identifying a context-length overflow
is emitted as `status=ok`, `score=0`, with
`terminal_reason=context_length_exceeded`. Exact metrics from all successful
responses before the rejection are retained. Authentication failures,
timeouts, throttling, missing profiles, and other infrastructure exceptions
remain `status=error` and are therefore retryable.

While unresolved rows remain, create separately named staged artifacts with:

```powershell
.\.venv\Scripts\python.exe experiments\build_completed_9x9.py --allow-incomplete
```

The staged pickle retains unresolved cells as `None`, and the manifest lists
their exact configuration and question IDs. Do not rename staged artifacts as
final datasets. After all retries succeed, run the same command without
`--allow-incomplete`; strict mode refuses to publish a final MathQA artifact
if any result or server latency is missing.

The remaining transient MathQA rows can be resumed under the original
experiment settings (default three reflection iterations, unbounded inner
tool loop, and cache enabled) with:

```powershell
.\.venv\Scripts\python.exe experiments\run_missing_cells.py `
  --plan experiments\missing_cells\retry_deepseek_9x9_timeout3.csv `
  --profile-map experiments\missing_cells\profile_arns.local.json `
  --project-tag agentopt `
  --billing-tag1 qx2278 `
  --max-concurrent 20 `
  --execute
```

Already successful rows in the retry plan are skipped automatically.
