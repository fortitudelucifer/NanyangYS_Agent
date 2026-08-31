# KuaiRand Long-Sequence Agent

## CausalFeatureOps Agent Infra v0.1

GPU 研究结果与 Agent 控制平面已经解耦。当前可运行一个不读数据、不训练模型、不会伪造指标的六角色离线 Harness Demo；缺少真实科研证据时会终止为 `waiting_for_evidence`。

- 设计：[docs/CAUSAL_FEATURE_OPS_AGENT_SYSTEM_V001.md](docs/CAUSAL_FEATURE_OPS_AGENT_SYSTEM_V001.md)
- Checkpoint：[reports/agent_system_checkpoint_2026-08-16.md](reports/agent_system_checkpoint_2026-08-16.md)

```powershell
..\.venv\Scripts\python.exe scripts\run_agent_system_demo_v001.py --verify
```

当前 backend 是用于验证 Harness 的 `local_structured_v1` 确定性角色 fixture，不等同于六个 AgentTeams worker；真实 AgentTeams adapter 仍是比赛前必须完成的工作。

> 当前研究状态（2026-08-24）：正式 Silver 不变；Ubuntu 5070 Ti 上完成的 H2 History-Value GPU Adam v007/v008/v010 证据已导入并通过哈希复核。它支持严格时点 BL2 的离线排序增量，但 random exposure 域的绝对概率门未通过，且 SGD 尚未确认。继续前必须阅读 [`reports/post_gpu_import_reconciliation_2026-08-24.md`](reports/post_gpu_import_reconciliation_2026-08-24.md)，不得把其结果自动扩展为 Gold、序列模型或概率部署授权。

面向 KuaiRand-1K 的时点安全长序列候选视频偏好预测与自动验证 Agent 系统。

正式 Silver 已完成并由 manifests 验收；当前只获准进行 v002 Train-only 数据建模/描述性关联。原始 CSV 与正式 Silver 保持只读；Gold、正式模型训练、Validation 和测试模型评估尚未开放。

## 数据流

```text
../KuaiRand-1K/data（只读原始 CSV）
        ↓ 数据合同与质量审计
data/silver（去重、类型统一、质量标记）
        ↓ point-in-time 序列构造
data/gold（训练、验证和封存测试样本）
```

## 当前约定

- 主标签：`long_view`
- 辅助标签：`is_like`
- 安全标签：`is_hate`
- 批次键：`user_id + time_ms`
- 历史规则：只允许 `history_time < target_time`
- 原始数据：永久只读
- 完全重复行：Silver 层保留一份，副本进入审计记录
- 不确定或冲突记录：进入 `data/quarantine`，不自动修改官方标签
- `video_features_statistic_1k.csv`：默认禁止用于同月预测

## 当前分析入口

日常迭代使用 `--quick`（8 线程 CPU、独立非正式输出）；checkpoint 使用 `--release`（完整 SHA-256、单线程确定性输出）：

```powershell
..\.venv\Scripts\python.exe .\scripts\analyze_train_associations_v002.py --quick
..\.venv\Scripts\python.exe .\scripts\analyze_train_associations_v002.py --release
```

两种模式均不会重清洗 Silver，也不会读取 Validation、late、random 或 statistic 文件。quick 结果不能升级研究进度或 Gate 状态。

Gate 2 的 Train-only 设计证据使用独立 release 入口：

```powershell
..\.venv\Scripts\python.exe .\scripts\analyze_gate2_train_design_v002.py --release
```

当前已冻结 calendar burn-in 的 assessment-only 规则、source-identity 固定行原型、diagnostic modality proxy v1 和 compute/search 规划上限；Gold、正式训练及后续数据访问仍未开放。

## 目录

- `configs/`：数据路径、清洗规则、实验合同
- `data/`：项目生成的数据层和审计清单
- `src/kuairand_longseq/`：数据、特征、模型、评估、Agent、Skill 和 Harness 代码
- `scripts/`：可直接执行的流水线入口
- `tests/`：数据合同和 point-in-time 测试
- `notebooks/`：只用于探索，不作为生产流水线入口
- `reports/`：研究报告和审计报告
- `artifacts/`：模型、预测、运行清单等可复现产物
