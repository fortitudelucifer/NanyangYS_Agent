# NanyangYS Agent：3 分钟 Demo

目标：用三分钟展示两件彼此独立、都可核验的成果。

1. Agent 控制平面能够运行，并在没有合格 Evidence 时 fail-closed。
2. GPU 机器已经产生真实的 Train-only BL0 / BL1 / BL2 数值快照，但它仍在等待 provenance 和 Validation 闭环。

## 演示前准备

下面所有 PowerShell 命令都可从任意当前目录运行：

```powershell
$AgentRoot = "D:\BaiduSyncdisk\master\NTU\course\CA6002\GOAI\NanyangYS_Agent"
$Snapshot = Join-Path $AgentRoot "demo\gpu_train_only_snapshot"
```

如环境尚未安装项目：

```powershell
python -m pip install -e "${AgentRoot}[test]"
```

## 0:00–0:40：先讲清研究边界

打开快照状态说明：

```powershell
Get-Content -LiteralPath (Join-Path $Snapshot "STATUS.md")
```

口头说明：

- Silver 保持冻结，本 Demo 不重新清洗数据。
- GPU 实验是 Train-only、7 个 assessment origins、BL0 / BL1 / BL2。
- 数值结果已经复核；provenance、Agent Evidence 准入和 Validation 尚未完成。
- 因此当前是“有真实实验快照，但没有可发布科学证据”的状态。

## 0:40–1:25：展示真实 GPU 数字

先看三个模型的汇总指标：

```powershell
Import-Csv (Join-Path $Snapshot "pooled_metrics.csv") |
  Select-Object model_id, rows, users, average_precision, log_loss, brier, user_gauc_event_weighted |
  Format-Table -AutoSize
```

再看 BL2 相对 BL1 的配对 bootstrap：

```powershell
Import-Csv (Join-Path $Snapshot "paired_user_cluster_bootstrap.csv") |
  Where-Object contrast -eq "BL2_minus_BL1" |
  Select-Object metric, point_estimate, ci95_lower, ci95_upper |
  Format-Table -AutoSize
```

应强调的结论：BL2 相对 BL1 的 AP、两种 user-GAUC 均提升，Log Loss 与 Brier 均下降；AP 差值为 `+0.041295`，95% CI 为 `[0.037661, 0.045054]`。这只是一项 Train-only 初步结论。

可选展示 7 天逐日指标：

```powershell
Import-Csv (Join-Path $Snapshot "daily_metrics.csv") |
  Where-Object model_id -in @("BL1", "BL2") |
  Select-Object origin, model_id, average_precision, log_loss, brier |
  Format-Table -AutoSize
```

## 1:25–2:25：运行 Agent 并验证 fail-closed

不要给 Agent 传入快照中的任何 manifest，直接运行默认 Demo：

```powershell
$Run = python "$AgentRoot\scripts\run_agent_system_demo_v001.py" `
  --config "$AgentRoot\configs\agent_system_v001.yaml" `
  --output-root "$AgentRoot\artifacts\agent_runs" `
  --verify | ConvertFrom-Json

$Run
Get-Content -LiteralPath (Join-Path $Run.run_dir "reports\final_report.md")
```

应看到：

```text
terminal_state = waiting_for_evidence
verification.verified = true
```

解释：默认 `NullEvidenceProvider` 明确返回 unavailable。Agent 仍完成合同、边界、特征、泄漏、证据、Gate、主张和发布就绪度等受治理步骤，并把全过程写成可哈希验证的运行制品；因为没有合格 Evidence，最终安全停止。

## 2:25–3:00：解释下一步接入点

快照中两个 manifest 的含义必须分清：

| 文件 | 用途 | 能否传给 `--evidence-manifest` |
|---|---|:--:|
| `run_manifest.json` | GPU producer 的运行记录 | 否 |
| `evidence_bundle_manifest.yaml` | 本 Demo 小型快照的文件/哈希清单 | 否 |

它们都不是 Agent 的消费侧 Evidence manifest。真正接入前还要：

1. 补齐并冻结 contract、code、input、model config、authorization、target manifest 等 provenance。
2. 按当前 Provider schema 生成版本化、可移植的独立 Evidence manifest。
3. 在 Agent 配置中冻结相同摘要并做 scope、文件大小和 SHA-256 校验。
4. 完成独立 Validation 后，再讨论可发布结论。

结束语：**这个 Demo 已证明 Agent 会管理研究证据，也展示了真实 Train-only 数字；它没有把尚未闭环的实验包装成最终结论。**

## 演示失败时的快速检查

```powershell
python -m pytest "$AgentRoot\tests\agent_system" -q
python "$AgentRoot\scripts\run_agent_system_demo_v001.py" --help
```

如果当前机器没有全局 `python`，将命令中的 `python` 替换为已安装 Python 3.11+ 的完整可执行文件路径。
