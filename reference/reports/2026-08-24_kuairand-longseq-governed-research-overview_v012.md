# KuaiRand 长序列受治理研究总览：从 Raw 到 v012 结论

> 生成日期：2026-08-24（Asia/Singapore）  
> 范围：`KuaiRand-1K/`、`governed_data/` 与 `kuairand-longseq-agent/` 的受治理 `candidate_long_view_prediction` 研究。  
> 阅读原则：以各版本的合同、审批、artifact manifest 和 `final_decision.json` 为当前事实来源；早期交接文档、启动记录与实验 README 仅用于还原过程。

## 一句话最终结论

在冻结的 GPU Adam 协议下，严格时点 H2 用户总体历史相对相同静态特征基线具有稳定的离线排序增量；这个增量先后通过 Validation、后置 standard 和 random-exposure audit。random 域发生显著的基准率迁移，故 v011 用独立目标域数据选择并冻结了不改变排序的截距校准器，v012 再以成对目标域适配获得了额外、统计上可信的排序与概率损失改进。

当前工程候选是 `NEW_BL2 + v012 calibrator`，但它的证据等级是 **post-audit target-adaptation temporal replay，不是新的 pristine confirmation**；仍须以新数据确认，且不能推导线上因果收益、跨优化器稳健性或用户×标签偏好机制。

## 0. 当前事实、证据优先级与版本状态

| 层级 | 当前事实 | 主证据 |
|---|---|---|
| 数据层 | 正式 Silver 为 `silver-20260814-155536`，Raw 与 Silver 不重清洗、不覆盖。 | [`silver_run_manifest.json`](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/silver_run_manifest.json) |
| v010 | H2 历史在 Validation、sealed standard 与 random audit 上均有显著排序增量；random 的绝对概率并不可直接部署。 | [`history_value_final_evidence_summary_v010.md`](kuairand-longseq-agent/reports/analysis/history_value_final_evidence_summary_v010.md) |
| v011 | 选择 `M2_intercept_only` 校准，严格保持排序不变并显著修复目标域概率质量。 | [`v011 final_decision.json`](kuairand-longseq-agent/experiments/bl2_target_domain_calibration_v011/outputs/final_decision.json) |
| v012 | `C2_balanced` 目标域成对重训练通过全部门，优于 `OLD_BL2 + v011`。 | [`v012 final_decision.json`](kuairand-longseq-agent/experiments/bl2_target_domain_retraining_v012/outputs/final_decision.json) |

2026-08-24 已复核 Ubuntu 导入的 v007/v008/v010 共 31 个制品，并确认本地 6 个正式 Silver Parquet 与冻结快照哈希一致。本文另按 `kuairand-longseq-agent/` 作为路径根，复核 v011 和 v012 顶层 artifact manifest 所列 6 个文件的 SHA-256，均匹配。

注意：v011/v012 的 README 中仍保留“待授权”的历史叙述；其 `outputs/final_decision.json`、`experiment_manifest.json`、stage decision 与哈希制品是后续完成后写出的更高优先级状态，本文按后者表述，不把二者混为同一时点的事实。

## 1. 从原始数据到可建模问题

### 1.1 Raw 数据与可回答的问题

原始 KuaiRand-1K 由 early standard（5,055,984 行）、late standard（6,657,061 行）、random exposure（43,028 行）、1,000 个用户和 4,371,868 个基础视频组成。研究不试图模拟完整线上推荐系统，而是回答受限的离线问题：

```text
在目标视频曝光前，只使用该用户的严格历史、用户静态信息和视频基础元数据，
预测候选事件发生 long_view=1 的概率或排序分数。
```

`video_features_statistic_1k.csv` 是同期/事后统计，默认拒绝作为预测特征；random exposure 只用于主结论冻结后的迁移、偏差与校准审计，不能回流选特征或调参。

### 1.2 标签、场景与时点规则

- 主标签是来源提供的官方 `long_view`；`is_like` 与 `is_hate` 仅为合同允许时的辅助/安全标签。
- 标准阶段目标场景固定为 `tab=1`；`is_click` 不能脱离 `tab` 被当作语义一致的统一标签。
- 同一 `(user_id, time_ms)` 的候选属于同时批次：先为整批构造特征，再更新历史。因此唯一允许的历史边界是 `history_time < target_time`，当前批次反馈绝不进入当前批次特征。
- 早期 standard 用于训练与模型选择；后置 standard 用于时间外评估；random 用于独立曝光审计。这些数据权限不是“文件能读就能用”，而是由版本合同授予。

这些规则来自 [`PROJECT_HANDOFF.md`](kuairand-longseq-agent/PROJECT_HANDOFF.md) 与 [`data-governance.md`](skills/kuairand-governed-research/references/data-governance.md)。

## 2. Silver 清洗：不以“更干净”为名丢失证据

Silver 不是简单删行：它保留重复记录的 canonical copy，并将 duplicate copy、payload conflict、非法域和 `long_view` 公式不一致记录分别进入可审计输出。最终清洗构建的关键对账如下。

| 来源 | Raw 行 | Silver 行 | 关键治理结果 |
|---|---:|---:|---|
| early standard | 5,055,984 | 4,992,443 | 34,562 duplicate copies、5,455 conflicts、23,524 label mismatches 均被显式处理。 |
| late standard | 6,657,061 | 6,556,501 | 59,662 duplicate copies、8,514 conflicts、32,384 label mismatches 均被显式处理。 |
| random | 43,028 | 42,982 | 1 duplicate copy、45 label mismatches；不作为调参输入。 |
| users / videos basic | 1,000 / 4,371,868 | 1,000 / 4,371,868 | 事件—用户、事件—视频连接缺失均为 0。 |

清洗合同、实现与输出均有 SHA-256；行对账、二元域、连接覆盖及“事后 video statistic 未物化/未 join”门全部通过。正式 Silver 因此是研究输入锚点，不是可被后续模型结果反向重洗的对象。

## 3. EDA 如何收敛到研究方向与假设

### 3.1 Canonical Train EDA

在只读 early-standard 的 2022-04-08 至 04-17、`tab=1` 范围内，canonical Train 有 2,399,844 个目标事件、950 个用户、974,550 个视频和 765,417 个 `long_view` 正例（31.8944%）；加入的同源公式不一致行有 14,070 条。用户与视频连接缺失均为 0。

EDA 给出的不是“直接可选的特征”，而是几个必须控制的事实：

- event-micro 正例率 31.89%，但 user-macro 平均率 39.55%，说明活跃用户权重足以改变总体判断；评估必须同时报告用户层面的结果。
- 日正例率约在 29.90%–34.88% 间变化，历史深度又与日期和活跃度共变；不能把深历史的低正例率误读成因果效应。
- `prior_user_lv_rate` 的 pooled AUC 为 0.7088、user-GAUC 仅 0.5170；`recent10_batch_lv_rate` 的 user-GAUC 为 0.5857，但二者在同一 timestamp 批次中均是常数。这排除了“只凭用户总体率就能做候选内排序”的错误方向。
- 非正/缺失时长、上传类型与用户活跃度存在显著异质性，但部分字段的采集时点未知，只能用于诊断而不能静默进入 point-in-time allowlist。

因此研究方向从“用静态 ID 预测”收敛为：在相同目标行上，检验**严格历史 H2 是否给 BL1 静态基线增加预测信息**；若要回答用户偏好某类内容，必须另建 H3（用户×作者/标签/内容）合同，而不能从 H2 推断。

### 3.2 冻结的可证伪假设

`hypothesis_registry_v002.yaml` 将主问题登记为：在相同候选事件、用户/视频静态信息和曝光前上下文下，严格先验统计历史是否改善 `long_view` 预测。比较要求固定目标行、同一切分、同一指标与配对用户簇 bootstrap；不允许用单变量关联作为硬筛选，也不允许用 future partition 选择特征、阈值或校准器。

最初模型族为：

- `BL0`：冻结常数概率参考；
- `BL1`：静态用户/视频/作者/音乐/标签组合、时长和时间元数据；
- `BL2`：与 BL1 完全相同，额外加入 H2 的 lifetime 与最近 10/50/200 个批次事件数、正例数、平滑 `long_view` rate、历史深度/间隔/覆盖标记。

对应证据：[`train_association_report_v002.md`](kuairand-longseq-agent/reports/analysis/train_association_report_v002.md)、[`hypothesis_registry_v002.yaml`](kuairand-longseq-agent/configs/hypothesis_registry_v002.yaml)。

## 4. 冒烟测试、优化器选择与早期失败的价值

### 4.1 先验证工程正确性，而非直接打开后置标签

GPU 合成 smoke test 确认 Adam 与 SGD 在 RTX 5070 Ti 上完成同一正则化逻辑回归目标、没有 CPU fallback；严格历史构造遵守“同 timestamp 整批先预测、后更新”，random 构造器只允许 standard events 更新状态。仓库回归测试为 106 passed，`--validate-only` 未读取 governed split。

v001 随后在 Train-only preflight 安全停止：CPU-LBFGS 参考解的最大 500 次迭代上限产生 convergence warning。把相同参考解上限提高到 2,000 后，6/6 个参考解在 644–952 次内收敛，且首个目标差仅约 `6.17e-10`；这证明失败来自参考求解器上限，而非数据、标签或历史定义。

### 4.2 Adam 被选择，原 SGD 路线被冻结

所有优化器决策只用 Train-only objective adequacy，不借后置指标反推：

| 优化器路线 | 冻结试验 | 充分性结果 | 决策 |
|---|---|---|---|
| GPU Adam | `lr=0.03`，30/100/300/1000 steps；BL1/BL2 × 3 probes | 100 steps 起 6/6 通过 objective-regret；100 是最小充分 checkpoint。 | 选择 `lr=0.03, steps=100`。 |
| 全批 SGD，momentum=0 | `lr=0.001/0.003/0.01/0.03/0.1`，至多 3,000 steps | 最佳 `lr=0.1` 仍为 0/6 通过；BL1 regret 远超允许值。 | 冻结 SGD，不进入 Validation、sealed 或 random。 |
| 事后扩展的 SGD 草案 | `lr=1.0, 10,000` steps | 仅部分 Train probe 诊断，未覆盖冻结协议。 | 不执行、不用于与 Adam 的科学对照。 |

因此“排除 SGD”不是 SGD 证明历史无用，而是避免把两个模型优化充分度不同误写成历史特征差异。此后所有正式主张明确限定为 **GPU Adam 下**。

## 5. 从 Train-only 发现到三轮版本化实验

### 5.1 前导发现：Gate 2B 指出了必须修复的风险

Train rolling-origin 固定行实验曾发现 BL2−BL1 的 ΔAP 为 `+0.110821`，95% CI `[+0.101479, +0.120377]`，7/7 天为正；但 BL1/BL2 相对 BL0 的绝对 Log Loss 和 Brier 更差，且概率大面积饱和。它保留了“历史有排序信号”的假设，却拒绝把该 Train-only 模型升级为概率模型、序列模型或 Validation 结论。

这一步是后续三轮实验必须显式报告绝对概率门、概率健康、逐日稳定性和配对不确定性的原因，而不是一个被隐去的失败。

### 5.2 三轮正式结论

| 版本 | 问题与冻结参数 | 一致性/统计结果 | 结论与边界 |
|---|---|---|---|
| **v010 H2 历史价值** | GPU Adam `lr=0.03`、100 steps；相同 BL1 静态特征与 BL2=BL1+H2；按 Validation → sealed standard → random audit 顺序释放。 | 三阶段均使用 2,000 次配对用户簇 bootstrap。ΔAP：Validation `+0.035626 [0.031302, 0.040028]`；sealed `+0.040651 [0.036937, 0.044402]`；random `+0.026188 [0.018296, 0.034667]`。 | H2 排序增量可在时间外与 random exposure 迁移；random 中 BL1/BL2 绝对概率仍不合格，不能直接当部署概率。 |
| **v011 目标域校准** | 在 v010 排序分数上，以目标域前段选择 `M2_intercept_only`；最终参数：intercept `-2.0952099481`、slope `1.0169043468`，17 次迭代收敛；选择、refit、held-out test 分离。 | 23,752 行、967 用户的 held-out-to-calibrator 阶段，AP/ROC/user-gAUC 精确不变；Log Loss `0.512076 → 0.268939`，Δ `-0.243137`，95% CI `[-0.258633, -0.226836]`；Brier `0.169978 → 0.074827`，Δ `-0.095151`，95% CI `[-0.102219, -0.087959]`。 | 校准通过，且严格保持排序顺序；工程上可用校准器，但为 post-audit temporal test，仍待新数据确认。 |
| **v012 目标域重训练** | 从 sealed BL1/BL2 warm start 的成对适配，三候选均合格，冻结 `C2_balanced`：Adam `lr=0.01`、100 steps、tether `0.001`；完成后分别对 NEW_BL1/NEW_BL2 做 M2 校准。 | 最终 12,399 行、857 用户、2,000 次配对 bootstrap。`NEW_BL2 − OLD_BL2+v011`：ΔAP `+0.017032 [0.005265, 0.031319]`；ΔLog Loss `-0.003307 [-0.005145, -0.001725]`；ΔBrier `-0.000752 [-0.001243, -0.000333]`；Δevent-gAUC `+0.024183 [0.007412, 0.041419]`。3/3 天历史 AP 为正。 | v012 重训练增加价值；工程推荐为 `deploy_NEW_BL2_with_v012_calibrator_pending_new_data_confirmation`，不可包装成 pristine confirmatory test。 |

v010 的 random 阶段还应保留一个细节：其 Δlog-loss 与 ΔBrier 点估计均改善，但 95% CI 跨 0；这并不削弱已显著的排序迁移结论，却禁止把该阶段写成“概率损失已被确定改善”。

## 6. 最终模型选择与可复现实验参数

| 作用 | 选中的参数/状态 | 为什么不是任意调参 |
|---|---|---|
| H2 主实验优化器 | GPU Adam，`lr=0.03`、100 steps | 以 Train-only regret 对 CPU-LBFGS 参考解验证，选择最小充分 checkpoint。 |
| H2 历史定义 | lifetime + recent 10/50/200 batch 的事件数、正例数、平滑率、历史深度/间隔/覆盖标记；平滑先验强度 20。 | 由冻结 H2 bundle 与严格时点构造器定义，不从 random 标签反推。 |
| v011 校准 | `M2_intercept_only`，intercept `-2.0952099481`、slope `1.0169043468` | 三个 family 先在 selection 段选择，final held-out 未参与选择；排名不变是硬门。 |
| v012 重训练 | `C2_balanced`，Adam `lr=0.01`、100 steps、tether `0.001`；`NEW_BL2` 最终状态 SHA-256 `5f4023f32749a44b555559d25a7793cd7b2dff856a594998f03c8bf138b63308`。 | C1/C2/C3 全部 eligibility 合格，C2 在 final replay 前按冻结 selection rule 唯一胜出。 |

所有关键 run 都有合同、审批/访问账本、输入与制品哈希、切分审计、逐行预测、pooled/daily metrics、概率分布审计、2,000 次 bootstrap 与 final decision。对应入口为 [`v010 evidence manifest`](kuairand-longseq-agent/reports/analysis/history_value_final_evidence_manifest_v010.json)、[`v011 experiment manifest`](kuairand-longseq-agent/experiments/bl2_target_domain_calibration_v011/outputs/experiment_manifest.json)、[`v012 experiment manifest`](kuairand-longseq-agent/experiments/bl2_target_domain_retraining_v012/outputs/experiment_manifest.json)。

## 7. 最终可说与不可说的话

**可以说：**

1. 在本任务、冻结特征、严格时点规则和 GPU Adam 协议下，H2 用户历史对候选 `long_view` 的离线排序有稳定增量，并跨 standard 与 random exposure 表现出迁移。
2. random 域的基准率变化会破坏未经目标域适配的绝对概率；独立校准与成对重训练在后审计时间回放中修复/提升了概率质量，且没有抹去排序价值。
3. `NEW_BL2 + v012 calibrator` 是当前最强的**工程候选**，其上线前仍需要新数据确认与人工审批。

**不能说：**

- 线上观看时长、留存、商业指标或任意新策略的因果提升；
- Adam 以外优化器同样稳健（SGD 已冻结，未形成充分对照）；
- H2 证明用户对某个标签/内容语义“喜欢”（H2 不是 H3 用户×内容机制）；
- v011/v012 是未污染的新验证或可绕过后续新数据确认的最终发布证据；
- 已完成 Gold、序列模型、全目录检索或线上模型训练。

## 8. 这些证据能为 NanyangYS Agent 提供什么

NanyangYS Agent 当前是**控制平面**：它管理合同、边界、特征规格、泄漏审计、证据准入、主张审查与可回滚发布；它不重新清洗 Silver、不在本机训练，也不会自动把外部 GPU 摘要提升为科学结论。当前 Agent 配置仍正确地处于 fail-closed 状态。

| 项目研究证据 | 可支持的 Agent 角色/能力 | 允许给 Agent 的事实 | 不允许的升级 |
|---|---|---|---|
| Silver row reconciliation、SHA-256、禁止重洗、statistic denylist | Data Auditor | 可审计数据边界、正式 Silver 身份、Raw/Silver 不可覆盖。 | Agent 不能重新清洗或自行探索新数据文件。 |
| `history_time < target_time`、timestamp-batch 先预测后更新、H2 特征定义 | Feature Miner + Data Auditor | 可提出并审计 H2 的 point-in-time Feature Spec；拒绝当前反馈、同批反馈和 future labels。 | 不能把 H2 说成用户×标签偏好，更不能以当前标签构造特征。 |
| v010 的同目标行 BL1/BL2、跨阶段 ΔAP、2,000 次用户簇 bootstrap、哈希链 | Causal Evaluator + Safety Reviewer | 可作为“static baseline vs strict statistical history”的候选外部科学证据，特别是 Validation 阶段。 | 不支持 short/long sequence 模型，不支持线上因果或全库召回主张。 |
| v011 的排序不变校准与显著 Log Loss/Brier 改善 | Causal Evaluator + Feature Publisher | 可证明部署域校准应独立选择、冻结、并以概率健康门验收。 | 不能将 post-audit 校准写成 untouched random audit，也不能跳过新数据确认。 |
| v012 的成对 target adaptation、选择冻结、final replay、回滚友好的 model state hash | Feature Publisher + Safety Reviewer | 可形成版本化、可回滚的工程候选包：`NEW_BL2 + v012 calibrator`，状态为 pending confirmation。 | 不能自动发布；必须保留“post-audit、人工批准、新数据确认”条件。 |

### 8.1 为什么当前 NanyangYS Agent 仍不能直接接收这些目录

这不是缺点，而是 Agent 正确的安全门：

1. `configs/agent_system_v001.yaml` 当前把 `expected_provenance` 的 contract/code/input/model-config/authorization/target-manifest 六个 SHA-256 都设为 `null`。其 `ManifestEvidenceProvider` 会拒绝任何外部证据，直到消费端先冻结这些摘要。
2. 该 Agent 当前请求 `split: validation`、`tier: validation`，且要求四个模型：`static_baseline`、`strict_statistical_history`、`short_sequence`、`long_sequence`。v010 只覆盖 static/H2；本项目没有获批或完成 short/long sequence，因此它**不能**满足当前四模型合同。
3. Agent 的 provider 不接受 producer 的 `run_manifest.json` 或 `artifact_hash_manifest.json` 作为消费侧 Evidence manifest；它要求独立 schema，包含 scope、模型集、指标、gates、limitations、六类 provenance digest 与逐文件路径/大小/SHA-256。
4. v011/v012 的证据等级是 post-audit temporal test/replay，不应伪装成 Agent 枚举中的 `validation` 或 `sealed_test` tier。

因此，当前可交付给 Agent 的最强价值是**证据包的原材料和消费合同设计**，不是“把现有目录丢进去后自动发布”。

### 8.2 合法接入路径

若未来要让该 Agent 消费本项目证据，应先建立一个新的、版本化的 Agent Evidence handoff，而不是改写旧结果：

1. 选择单一 scope，例如 v010 的 Validation：映射 `ADAM_BL1 → static_baseline`、`ADAM_BL2 → strict_statistical_history`，并明确不声明 sequence 模型。
2. 从 v010 合同、批准、code/input/target manifest 与 validation artifact manifest 生成一个自包含 `research_evidence_manifest_v001.yaml`；其 artifact 路径必须相对该 manifest，逐一固定 size 与 SHA-256。
3. 创建新的 Agent 消费合同，预先写入六个 provenance digest，并把 `required_models` 收窄到证据实际覆盖的两个模型；不得篡改现有 `agent_system_v001.yaml` 的四模型要求来伪造兼容性。
4. 让 Agent 的 Causal Evaluator 校验外部 manifest，Safety Reviewer 限制可说的话，Feature Publisher 最多进入 `waiting_for_approval`；v011/v012 作为工程扩展证据单独登记为 post-audit，不与 validation 证据混合。
5. 在独立新数据确认和人工批准前，Agent 只能输出“工程候选/待确认”，不能发布为线上有效或因果有效。

相关 Agent 消费规则位于 [`NanyangYS Agent README`](archive/external-projects/nanyangys-agent-20260816/source/README.md)、[`agent_system_v001.yaml`](archive/external-projects/nanyangys-agent-20260816/source/configs/agent_system_v001.yaml) 和 [`evidence/admission.py`](archive/external-projects/nanyangys-agent-20260816/source/src/kuairand_longseq/evidence/admission.py)。

## 9. 继续研究时的最短正确路径

1. 保持 Silver、v010/v011/v012 的既有路径、合同和哈希证据不变；不因整理而重命名。
2. 若目标是部署，优先设计**新数据**上的确认合同：冻结 `NEW_BL2 + v012 calibrator`、监控 exposure domain、历史深度、warm/cold video、duration-valid 状态及概率健康。
3. 若目标是科学机制，另建 H3 用户×内容/标签历史增量合同，以 v012 的 BL2 为固定基线；不得从 H2 直接跳到内容偏好叙事。
4. 若目标是 Agent 集成，先完成上述独立 Evidence manifest 与消费端 digest pinning，再让 Agent 运行准入；当前 fail-closed 行为应保留。

## 10. 主要证据入口

- [Silver 清洗报告](kuairand-longseq-agent/reports/generated/silver_cleaning_report.md)
- [Train-only EDA / 关联报告](kuairand-longseq-agent/reports/analysis/train_association_report_v002.md)
- [Gate 2B 的排序发现与概率失败](kuairand-longseq-agent/reports/analysis/gate2b_baseline_results_v002.md)
- [GPU 优化器预检](kuairand-longseq-agent/reports/analysis/history_value_gpu_preflight_results_v001_v002.md)
- [SGD 冻结记录](kuairand-longseq-agent/reports/analysis/sgd_optimizer_freeze_2026-08-23.md)
- [v010 最终证据总结](kuairand-longseq-agent/reports/analysis/history_value_final_evidence_summary_v010.md)
- [v011 最终决策](kuairand-longseq-agent/experiments/bl2_target_domain_calibration_v011/outputs/final_decision.json)
- [v012 最终决策](kuairand-longseq-agent/experiments/bl2_target_domain_retraining_v012/outputs/final_decision.json)
- [2026-08-24 GPU 导入与状态对账](kuairand-longseq-agent/reports/post_gpu_import_reconciliation_2026-08-24.md)
