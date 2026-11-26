# AgentOpt 本地开发与演示验收记录

2026-11-26 · Windows · Python 3.12.3

## 仓库与测试基线

本次验收以 `package-internal-test@d382b7f` 为基线，在 Windows 上对照 `gittins-objective-modes` 测试。

| 分支或数据仓库 | 基线 | 用途 |
|---|---|---|
| `gittins-objective-modes` | `39a9732` | 单目标、多目标选型，实时 Pareto 图与 SQLite 断点 |
| `package-internal-test` | `d382b7f` | 完整包含上一个分支，另有 3 个提交提供离线回放入口和操作指南 |
| `agentopt-research` | `9be395d` | 配套研究仓库，提供八套历史评测矩阵 |

单目标和多目标选择的早期版本通过 PR #117 进入主分支。核对时，主分支最近合并的 PR #131 更新了演示视频。两项 PR 都由 `yuelihe2-svg` 发起、`QianJaneXie` 合并。[PR #117](https://github.com/AgentOptimizer/agentopt/pull/117) · [PR #131](https://github.com/AgentOptimizer/agentopt/pull/131)

论文《Which Model for Which Role?》提出 CC-Gittins，用于权衡工作流中不同模型组合的质量与部署成本。本仓库提供选择器、代理跟踪和运行界面。离线演示读取历史分数与成本，实际执行选择、绘图和断点保存。[论文](https://arxiv.org/abs/2610.26676) · [研究仓库](https://github.com/AgentOptimizer/agentopt-research)

## 界面、回放与结果导出

| 操作 | 当前行为 |
|---|---|
| 检查推荐 | 可以点击图上的候选点查看各角色使用的模型。终端按结构化推荐显示组合，含 ` + ` 或 `=` 的模型别名可以完整呈现。 |
| 暂停、继续、结束 | 进入 Paused 状态后，评测数和金额保持稳定。继续后按批次更新。图形模式也响应进度回调返回的 `False`，保存最终快照。 |
| 回放与续跑 | 回放按题目索引历史用量，保留可确认的零费用记录。SQLite 保存已完成的评测单元，重启后分别报告恢复数与新增数。 |
| 读取与保存文件 | 支持有无 UTF-8 BOM 的输入矩阵、配置和基准 JSONL。运行前解析实际输出路径，检查结果、图片、输入和 SQLite 附属文件的重名；图片路径自动补上扩展名后也参与检查。 |
| 导出结果 | CSV 按汇总列写入组合名、分数、价格和 token 等数据。读取赢家及导出 YAML 时使用结构化角色映射；旧结果可按原有显示标签读取。 |

入门命令行使用[仓库自带的 10 道算术题](examples/selection/benchmark_sample.jsonl)。图形演示使用历史矩阵，可按 [Windows 离线回放指南](examples/selection/OFFLINE_REPLAY.md) 选择数据集、打开窗口、保存图片和断点。

## 代理、缓存与费用

- **流式响应。** 本地 HTTPX 代理缓冲上游响应并读取 SSE 用量。同步、异步客户端都能消费响应；中途断流时释放连接并传递原始错误。使用 OpenAI Python SDK 和模拟传输验证了文本内容与末帧用量。[代理回归用例](tests/test_call_recording.py)
- **耗时。** 本地 HTTPX 与子进程代理使用高分辨率计时器测量请求耗时。缓存命中沿用原请求耗时，覆盖了 Windows 上很快完成的成功请求和 HTTP 错误响应。
- **缓存匹配。** 缓存键包含完整 URL、路由后的请求体、响应格式和选定的服务商请求头。测试覆盖切换服务商、API 版本、流式格式及重开磁盘缓存。升级后的首次真实请求可能产生上游费用。
- **预算与用量。** 成功调用缺少可计量用量时，本地与 daemon 都会报告原因。默认 Gittins 选择器在写入该单元的断点前停止，重试时可重新读取上游用量。回归覆盖真实零值、跨 SSE 帧的计数和无效计数。搜索按评测批次核算预算。若历史运行有用量缺失告警，可用新的断点复核其费用与推荐。

## 演示验收

| 数据与操作 | 实测结果 |
|---|---|
| 八套数据回放 | HotpotQA、MathQA 各使用全部 100 个组合，运行到 424 个单元。BIRD mini/dev、Bing Query Logs、Restaurant test/valid 和 Stack Overflow 六套 SCOPE 数据各抽 12 个配置，先跑 60 个单元。八套均生成推荐并保存结果和断点。 |
| BIRD 四角色窗口 | 点击图点核对 schema linking、classification、execution 和 debug 四个角色，经过 Pause、Continue、Finish 后保存图片。从 60 个单元续跑时新增 4 个；再从 64 个续跑到 200 个时新增 136 个。[窗口与断点用例](tests/test_live_pareto_tk.py) |
| HotpotQA 全矩阵 | 单目标、多目标分别评估 100 个组合 × 200 道题。用独立读取的 CSV 计算每个组合的平均分、单题成本和非支配集合，与选择器输出逐项比对。完整复核使用 196 题的自适应批次和 33 点计算网格，短回放使用 4 题批次。 |

### 演示边界用例

现有回归覆盖了预热中途停止、暂停后直接结束、重合推荐点的点选，以及关闭窗口后从断点继续。另在同一个 1 配置 × 1 问题的最小输入上验证了两个界面状态：

| 操作 | 界面预期 |
|---|---|
| 首次评测前触发 Finish | 显示已结束、0/1、零费用和 0 项推荐；显示空状态，仍能导出 PNG。 |
| 完成一次评测，得分和费用均为 0 | 显示 1/1、100% 和 1 项推荐；坐标原点有可点击的点，点击后选中对应组合，卡片显示 1/1。 |

这两个用例使用真实选择器和无窗口画布，覆盖相同零费用在不同完成状态下的显示。[界面边界用例](tests/test_live_pareto.py)

## 本地复验

从工作区根目录运行。模型调用相关测试使用历史数据、本地服务或模拟响应。

```powershell
$env:PYTHONUTF8 = '1'
$env:AGENTOPT_TEST_TK = '1'
$env:NO_PROXY = '127.0.0.1:1'
$env:PATH = (Resolve-Path .venv/Scripts).Path + ';' + $env:PATH
.venv/Scripts/python -m pytest -q -ra --tb=short
```

交付前使用包含上述两个用例的最终工作区复验，全仓 **443 项通过、0 项失败、2 项跳过**，另有 19 个子用例通过。两个跳过项需要 Windows 文件符号链接权限。HotpotQA 单目标推荐为 Qwen3 32B 规划、DeepSeek R1 求解，平均得分 0.7466375，平均每题 0.00403817 美元。多目标的 15 项推荐与独立计算的前沿完全一致。
