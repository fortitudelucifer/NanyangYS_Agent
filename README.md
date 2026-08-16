# NanyangYS Agent

NanyangYS Agent 是一个面向**真实开源大厂视频播放数据**的轻量级、可审计的视频推荐 Agent 系统闭环。目标不是堆叠大模型，而是用更小巧、更精确的方式挖掘用户长时序视频偏好，并让系统在收到新的用户反馈后能够持续优化迭代。

### 数据基础与扩展路线

当前以快手 **KuaiRand-1K** 公开数据集为起点开展研究，后续计划逐步接入更多真实开源视频播放数据，包括腾讯、字节跳动、YouTube 等平台，以验证偏好建模与推荐策略在多源数据上的可迁移性与稳健性。

### 系统设计理念

系统参考 **Harness** 与 **Loop Engineering** 思想构建：把研究任务约束为有角色权限、预算、证据门禁、停止条件和哈希审计的可迭代闭环流程。用户更新信息（新的观看、反馈、长时序行为）可作为新一轮 loop 的输入，驱动 Agent 重新审计边界、生成特征规格、请求与评价证据、审查主张边界，并在证据与审批同时满足时发布可回滚的特征包。它不重新清洗 Silver，也不在本机训练模型，而是把"研究能否被信任"这件事本身工程化。

### 研究进展

当前仍在持续研究数据集阶段。已在 **RTX 4090** 上完成部分用户长时序视频偏好研究并产出实验数据（Train-only 的 BL0 / BL1 / BL2 基线对比与数值核验），后续仍在继续深入建模与验证工作。

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

clone 仓库后，在仓库根目录下执行：

```powershell
python scripts/run_agent_system_demo_v001.py `
  --config configs/agent_system_v001.yaml `
  --output-root artifacts/agent_runs `
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
python -m pip install -e ".[test]"
python -m pytest tests/agent_system -q
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

## 许可证

### 代码许可证

本项目代码以 **Apache License 2.0** 授权，详见 [LICENSE](LICENSE)。

### 数据集声明

本项目使用以下真实开源视频推荐数据集进行研究：

| 数据集 | 来源 | 数据集许可证 | 引用 |
|---|---|---|---|
| **KuaiRand-1K** | 快手 (Kuaishou) | [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Gao et al., CIKM 2022 |
| 腾讯视频数据 | 腾讯 (Tencent) | 待接入 | 待补充 |
| 字节跳动数据 | 字节跳动 (ByteDance) | 待接入 | 待补充 |
| YouTube 数据 | YouTube | 待接入 | 待补充 |

**KuaiRand 数据集使用声明：**

- KuaiRand 数据集由 Chongming Gao, Shijun Li, Yuan Zhang, Jiawei Chen, Biao Li, Wenqiang Lei, Peng Jiang, Xiangnan He 等人发布，原始论文为：
  > Gao, C., Li, S., Zhang, Y., Chen, J., Li, B., Lei, W., Jiang, P., & He, X. (2022). KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos. *CIKM '22*, Atlanta, GA, USA. [DOI: 10.1145/3511808.3557624](https://doi.org/10.1145/3511808.3557624)
- 数据集主页：<https://kuairand.com> · GitHub 仓库：<https://github.com/chongminggao/KuaiRand>
- 该数据集以 **CC-BY-SA-4.0** 许可发布，使用时需署名，且基于该数据集的衍生作品须以相同许可证共享。
- 本项目**不分发数据集本身**，仅提供基于该数据集的研究代码与实验结果。使用者需自行从官方渠道获取数据集并遵守其许可证条款。
