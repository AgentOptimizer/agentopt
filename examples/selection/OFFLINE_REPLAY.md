# Test the live UI with offline replay

The `package-internal-test` branch can replay eight datasets. Each evaluation reads the recorded score and cost for one workflow configuration on one question. Setup requires an internet connection to download the repositories and install dependencies. The replay runs offline without a model API key.

## Set up the test branch

If you already have an AgentOpt checkout, open **PowerShell** in the repository root and update the test branch.

```powershell
git fetch origin
git switch package-internal-test
git pull --ff-only origin package-internal-test
```

For a fresh checkout, run `git clone --branch package-internal-test https://github.com/AgentOptimizer/agentopt.git`, then `cd agentopt`.

The replay looks for the matrices in `../agentopt-research/data`. Run the following commands to prepare the data and install AgentOpt with plotting support. The example uses Python 3.12.

```powershell
if (!(Test-Path ..\agentopt-research)) {
    git clone --depth 1 --filter=blob:none --sparse https://github.com/AgentOptimizer/agentopt-research.git ..\agentopt-research
    git -C ..\agentopt-research sparse-checkout set data/hotpotqa data/mathqa data/scope
}
if (!(Test-Path .\.venv\Scripts\python.exe)) {
    py -3.12 -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install -e ".[plot]"
```

## Open the UI and switch datasets

Paste the block below into the same PowerShell window to open BIRD's four-role workflow. The SCOPE demos use a reproducible sample of 12 recorded configurations per dataset. HotpotQA and MathQA use all 100 configurations. The optional `Limit` counts evaluated cells across runs. The replay stops at a batch boundary.

```powershell
function Show-Replay([string]$Dataset, [int]$Limit = 0) {
    $out = Join-Path $env:LOCALAPPDATA "AgentOpt\ui-check\$Dataset"
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $options = @()
    if ($Dataset -in 'hotpotqa', 'mathqa') {
        if ($Limit -eq 0) { $Limit = 500 }
    } else {
        $options = @('--max-configs', '12')
        if ($Limit -eq 0) { $Limit = 144 }
    }
    .\.venv\Scripts\python.exe .\examples\selection\replay_10x10.py --benchmark $Dataset --live-plot --delay-seconds 0 --grid-size 33 --stop-after $Limit --output "$out\result.json" --plot-output "$out\frontier.png" @options
}

Show-Replay bird_mini_dev
```

Close the plot window, then run any one of these commands to switch datasets.

```powershell
Show-Replay bird_dev
Show-Replay bing_querylogs
Show-Replay restaurant_test
Show-Replay restaurant_valid
Show-Replay stackoverflow
Show-Replay hotpotqa
Show-Replay mathqa
```

## Check the cards, controls, and checkpoint

1. In the BIRD window, wait for recommendations to appear. Click a numbered point on the chart. Check that the matching card is highlighted and shows the model selected for each of the four roles.
2. Click **Pause**. Once the status shows **Paused**, check that the evaluated cell count and evaluation spend stay steady. Click **Continue** and watch both update as the next batch completes.
3. Click **Finish** and note the final evaluated cell count. Once the terminal prints `Saved results to`, close the window. Run `Show-Replay bird_mini_dev 200` to resume from the checkpoint and stop at about 200 total cells. At the end, check that `restored` matches the count you noted and `newly evaluated` counts only the additional cells.

Each dataset's result JSON, chart, and SQLite checkpoint are saved under `$env:LOCALAPPDATA\AgentOpt\ui-check\<dataset>`. In a new PowerShell session, define `Show-Replay` again from the repository root to reuse the checkpoint.

## Continue with a larger budget

Use the same checkpoint and raise the total cell or spending limit. For example,
with the data in its default location, run these commands from the repository root.

```powershell
$run = Join-Path $env:TEMP ("agentopt-budget-" + [guid]::NewGuid())
New-Item -ItemType Directory $run | Out-Null
$replay = @(".\examples\selection\replay_10x10.py", "--benchmark", "bird_mini_dev", "--max-configs", "12", "--delay-seconds", "0", "--output", "$run\result.json")
.\.venv\Scripts\python.exe @replay --max-total-question-evaluations 60
.\.venv\Scripts\python.exe @replay --max-total-question-evaluations 96
```

The second command can evaluate up to 36 more cells after the first 60.
Use `--max-search-cost-usd` to set or raise a total simulated spending limit.
When both limits are set, retain or raise each one on restart. For an older checkpoint, use a fresh output or checkpoint path.

### Warm-up concurrency

Pass `--warm-start-max-concurrent 16` for up to 16 rolling warm-up cells (default: the batch size).
Set `--batch-size 12` to evaluate 12 questions per configuration in warm-up and up to 12 questions for
each selected configuration (default four). Add `--max-concurrent 6` to limit both
phases to six simultaneous evaluations. The live window uses the same settings.

With a dollar limit, warm-up probes for a price before reserving estimated charges
for concurrent cells. Reuse the output path when changing concurrency. Choose a
fresh checkpoint when changing the adaptive batch size, which affects selection.

## Test the user package before release

To check what a user will receive, open a separate PowerShell environment and install the user branch directly. Use a separate virtual environment so the replay checkout keeps its editable installation.

```powershell
py -3.12 -m venv .venv-user
.\.venv-user\Scripts\python.exe -m pip install "agentopt-py @ git+https://github.com/AgentOptimizer/agentopt.git@gittins-objective-modes"
.\.venv-user\Scripts\agentopt.exe --help
```
