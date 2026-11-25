# 用 8 个数据集离线检查运行界面

本指南适用于 `package-internal-test` 分支。回放从已保存的矩阵读取每个格子的分数和费用，全程无需模型 API key。首次下载仓库和安装 Python 依赖需要联网。

## 准备环境

已有 AgentOpt 仓库时，在仓库根目录的 **PowerShell** 中运行：

```powershell
git fetch origin
git switch package-internal-test
git pull --ff-only origin package-internal-test
```

首次下载可直接运行 `git clone --branch package-internal-test https://github.com/AgentOptimizer/agentopt.git`，然后 `cd agentopt`。

回放默认从同级目录 `../agentopt-research/data` 读取矩阵。以下命令准备数据和支持绘图的 Python 环境（示例使用 Python 3.12）：

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

## 打开界面和切换数据集

在同一个 PowerShell 窗口粘贴整段，先打开 BIRD 的四角色界面。参数 `Limit` 表示累计评测格子数，按批次停止。SCOPE 示例固定抽取 12 个已有配置，QA 示例使用完整的 100 个配置。

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

关闭图形窗口后，任选一行切换数据集：

```powershell
Show-Replay bird_dev
Show-Replay bing_querylogs
Show-Replay restaurant_test
Show-Replay restaurant_valid
Show-Replay stackoverflow
Show-Replay hotpotqa
Show-Replay mathqa
```

## 检查界面和续跑

1. 在 BIRD 窗口等待推荐出现。点击左图的编号，查看右侧对应卡片的高亮，以及四个角色各自选用的模型。
2. 点击 **Pause**，等状态变成已暂停，检查进度和费用是否稳定。点击 **Continue** 后，观察下一批评测是否更新。
3. 点击 **Finish**，记下已评测格子数。终端显示 `Saved results to` 后关窗。运行 `Show-Replay bird_mini_dev 200`，从相同断点续跑到累计约 200 格。结束时，终端的 `restored` 和 `newly evaluated` 分别给出复用和新增的格子数。

每个数据集的 JSON、图表和 SQLite 断点均保存在 `$env:LOCALAPPDATA\AgentOpt\ui-check\<数据集>`。新开 PowerShell 时重新定义上面的函数即可使用已有断点。
