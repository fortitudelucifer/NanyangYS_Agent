# NanyangYS Agent

面向 KuaiRand-1K 长序列推荐研究的可审计 Agent 控制平面。它负责把研究任务约束为有角色权限、预算、证据门禁、停止条件和哈希审计的流程；它不重新清洗 Silver，也不在本机训练模型。

## 当前真实状态

| 模块 | 当前状态 |
|---|---|
| Agent Harness | 已可离线运行和校验；无合格证据时确定性停在 `waiting_for_evidence` |
| GPU 实验快照 | Train-only 的 BL0 / BL1 / BL2 结果已做数值核验 |
| 科学证据准入 | 未完成；provenance 冻结与 Agent Evidence manifest 尚未闭环 |
| Validation / sealed test | 未使用、未授权 |
| 发布 | 未授权；系统保持 fail-closed |

GPU 快照中，BL2 相对 BL1 的初步结果为：AP `+0.041295`、event-weighted user-GAUC `+0.048758`、Log Loss `-0.022953`、Brier `-0.008907`，AP 的用户聚类 bootstrap 95% CI 为 `[0.037661, 0.045054]`。这些数字只支持 **Train-only 初步结论**，不等于 Validation、Gold 或发布结论。

## 直接运行 Demo

以下 PowerShell 命令可从任意当前目录执行：

```powershell
$AgentRoot = "D:\BaiduSyncdisk\master\NTU\course\CA6002\GOAI\NanyangYS_Agent"
python "$AgentRoot\scripts\run_agent_system_demo_v001.py" `
  --config "$AgentRoot\configs\agent_system_v001.yaml" `
  --output-root "$AgentRoot\artifacts\agent_runs" `
  --verify
```

预期关键输出：

```text
terminal_state = waiting_for_evidence
verification.verified = true
```

`waiting_for_evidence` 是正确的安全终局：默认 Provider 不提供科研证据，Agent 不能自行把 GPU 快照升级为可发布结论。

运行测试：

```powershell
$AgentRoot = "D:\BaiduSyncdisk\master\NTU\course\CA6002\GOAI\NanyangYS_Agent"
python -m pip install -e "${AgentRoot}[test]"
python -m pytest "$AgentRoot\tests\agent_system" -q
```

完整的 3 分钟展示顺序见 [demo/DEMO_GUIDE.md](demo/DEMO_GUIDE.md)。

## Agent 体系

六个角色按最小权限协作：

| 角色 | 职责 |
|---|---|
| Manager | 冻结研究合同与任务边界 |
| Data Auditor | 审计数据范围和时间泄漏，不重洗数据 |
| Feature Miner | 生成 point-in-time 特征规格 |
| Causal Evaluator | 请求、核验和评价研究证据 |
| Safety Reviewer | 限制结论范围，阻止越界表述 |
| Feature Publisher | 只在证据和审批同时满足时发布，并支持回滚 |

十个类型化 Skill：

`register_research_contract` · `audit_project_boundary` · `propose_feature_specs` · `detect_temporal_leakage` · `request_research_evidence` · `evaluate_research_evidence` · `review_claim_boundaries` · `assess_release_readiness` · `publish_feature_package` · `rollback_feature_package`

默认工作流：

```text
研究合同 → 边界审计 → 特征规格 → 泄漏审计
        → 证据请求 → Gate 评价 → 主张审查 → 发布就绪度
```

每次 Skill 调用依次检查角色权限、预算、幂等键和输出 schema；事件写入哈希链。证据缺失或不匹配时流程停止，不会回填占位结果，也不会自动发布。

## 最小提交结构

```text
NanyangYS_Agent/
├─ README.md
├─ pyproject.toml
├─ configs/
│  └─ agent_system_v001.yaml
├─ scripts/run_agent_system_demo_v001.py
├─ src/kuairand_longseq/
│  ├─ agents/
│  ├─ evidence/
│  ├─ harness/
│  └─ skills/
├─ tests/agent_system/test_agent_system_v001.py
├─ demo/
│  ├─ DEMO_GUIDE.md
│  └─ gpu_train_only_snapshot/
│     ├─ STATUS.md
│     ├─ evidence_bundle_manifest.yaml
│     ├─ run_manifest.json
│     ├─ pooled_metrics.csv
│     ├─ daily_metrics.csv
│     └─ paired_user_cluster_bootstrap.csv
└─ artifacts/agent_runs/                 # 运行时生成，可重建
```

`demo/gpu_train_only_snapshot` 只保留演示所需的小型摘要，没有复制 54 MB 的逐行预测 Parquet、原始数据或训练环境。

## 明确禁止的主张

- 不能称 GPU 快照为 Validation、Gold、sealed-test 或正式发布结果。
- 不能称 BL2 为已经验证的长序列模型；当前只是 Train-only 的 BL0 / BL1 / BL2 比较。
- 不能把 producer 的 `run_manifest.json` 或 `evidence_bundle_manifest.yaml` 直接传给 `--evidence-manifest`；它们不是 Agent 消费合同所需的 Evidence manifest。
- 不能宣称线上因果收益、全库召回质量或任意策略的离线评估能力。
- 不能宣称 Agent 已训练模型、调用 LLM/GPU，或已取得发布审批。

运行时合同以 `configs/agent_system_v001.yaml`、`src/kuairand_longseq/` 与安全回归测试为准。
