# KuaiRand 神经序列候选模型设计 v013

> 状态：`design_registered_fail_closed`  
> 合同：[contract_v013.yaml](../../../experiments/neural_sequence_candidate_model_v013/contract_v013.yaml)  
> 执行授权：[approval_v013.json](../../../experiments/neural_sequence_candidate_model_v013/approval_v013.json)  
> 设计快照：[design_manifest_v013.json](../../../experiments/neural_sequence_candidate_model_v013/design_manifest_v013.json)  
> 证据边界：本报告新增设计和预注册内容，不代表 Gold、神经训练、Validation、restricted test 或 random audit 已获授权或完成。

## 1. 继承事实与新问题

正式 Silver 继续固定为 `silver-20260814-155536`，不得因神经模型设计重新清洗或覆盖。v010 已证明冻结 GPU Adam 下严格 H2 用户总体历史相对静态模型具有排序增量；v011 通过独立截距校准修复目标域概率；v012 的成对目标域适配进一步产生增量，当前工程候选是 `NEW_BL2 + v012 calibrator`，仍待新数据确认。

v013 不重复回答“H2 有没有用”，而是把问题推进为：在完全相同的 point-in-time 目标行上，非线性交互、10/50/200 批次细粒度序列以及更早历史摘要分别贡献多少额外价值，并且这些增量是否同时通过排序、用户内判别和绝对概率质量门。

## 2. 完整模型阶梯

| 层级 | 模型 ID | 输入/结构 | 主要作用 |
|---|---|---|---|
| B0 | `BL0_CONSTANT` | 冻结常数概率 | 绝对 Log Loss/Brier 参考 |
| B1 | `LINEAR_STATIC_V013` | 同 Gold 静态特征逻辑回归 | 同目标行线性静态基线 |
| B2 | `LINEAR_H2_V013` | B1 + lifetime/10/50/200 H2 | 同目标行线性历史基线 |
| N0 | `MLP_STATIC` | 静态内容 Embedding + MLP | 检验无历史非线性交互 |
| N1 | `MLP_H2` | N0 + H2 统计历史分支 | 检验神经架构中的统计历史增量 |
| S10 | `DIN10` | N1 + 最近10批候选感知 Attention | 检验首次细粒度序列增量 |
| S50 | `DIN50` | 与 DIN10 同结构，仅窗口改为50批 | 检验10→50批增量 |
| S200 | `DIN200` | 与 DIN10/50 同结构，仅窗口改为200批 | 检验50→200及10→200批增量 |
| H500 | `HIER500` | 最近200批细粒度 + 201–500批门控摘要 | 探索更早历史增量，不使用全长 Transformer |

`DIN10`、`DIN50`、`DIN200` 必须共享候选编码器、历史事件编码器、Attention、融合 MLP、优化器家族、损失函数、种子和目标行；窗口长度是这三个模型之间唯一允许的科学差异。短历史样本使用 padding 与 mask，不得通过分别删行制造窗口收益。

## 3. 必须报告的对照

### 3.1 嵌套主对照

| 对照 | 回答的问题 |
|---|---|
| `MLP_STATIC − LINEAR_STATIC_V013` | 非线性静态交互有没有价值？ |
| `MLP_H2 − MLP_STATIC` | 在同一神经架构下，H2 统计历史有没有增量？ |
| `DIN10 − MLP_H2` | 最近10批细粒度事件是否超过统计摘要？ |
| `DIN50 − DIN10` | 历史从10扩展到50批是否稳定增益？ |
| `DIN200 − DIN50` | 历史从50扩展到200批是否稳定增益？ |
| `DIN200 − DIN10` | 10到200批的总增量是多少？ |
| `HIER500 − DIN200` | 201–500批压缩摘要是否还有增量？ |

### 3.2 支持性与最终 successor 对照

- `LINEAR_H2_V013 − LINEAR_STATIC_V013`：在 v013 Gold 上复核线性 H2 增量；
- `MLP_H2 − LINEAR_H2_V013`：相同 H2 信息下的非线性价值；
- `DIN50/DIN200 − MLP_H2`：细粒度序列相对统计摘要的累计价值；
- `SELECTED_NEURAL_ENSEMBLE − FROZEN_NEW_BL2_V012`：在合法的新确认数据上回答 v013 是否应替代当前工程候选。

最终 successor 对照只有在新确认数据、v012 冻结模型状态和 v013 选定模型能在完全相同目标行上评分时才成立；不得把 v012 已消耗的 post-audit replay 重新包装成 v013 的未见确认集。

## 4. 推荐架构

### 4.1 候选内容编码器

候选视频表示以 `author_id`、`music_id`、匿名 `tag`、`upload_type`、基础时长、上传年龄和基础几何属性为主。`video_id` 只作辅助，并必须单独报告 behavior-cold-video；不能让四百多万个视频 ID 的记忆替代内容泛化。

### 4.2 历史事件编码器

每个历史事件包含当时已知的内容属性、严格过去反馈、历史场景和距目标的时间差。当前事件的 `long_view`、播放/停留时长及任何当前反馈均在 denylist。历史记录必须满足 `history_time_ms < target_time_ms`，同一 `(user_id, time_ms)` 批次先全部生成样本，再更新历史状态。

### 4.3 候选感知 Attention 与长期摘要

候选表示作为 Query，最近 N 批历史事件作为 Key/Value；DIN10/50/200 只改变 N。H2 分支继续提供 lifetime 与10/50/200批统计摘要。HIER500 保留最近200批细粒度 Attention，并将201–500批压缩为按作者、标签、时长桶和日期的门控摘要。v013 禁止直接训练全长 Transformer。

### 4.4 融合与输出

拟议的单配置为32维类别 Embedding、64维事件/Attention 表示和 `256→128→64` GELU MLP，dropout `0.10`，最后输出单个 logit。具体 batch size、学习率、weight decay、epoch 和 early stopping 必须在 Gold pilot 后、科学回测前冻结；当前值不得由 Validation、restricted 或 random 结果反推。

## 5. 五随机种子协议

所有进入科学比较的神经候选固定使用：

```text
20260824
20260825
20260826
20260827
20260828
```

执行要求：

1. `MLP_STATIC`、`MLP_H2`、`DIN10`、`DIN50`、`DIN200` 以及满足门禁时的 `HIER500` 均运行完整5个种子；
2. 任何候选缺一个种子就不能进入正式对照；
3. 主预测为五个模型 logit 的算术平均，再使用一个独立、冻结、时间有序的校准器；
4. 每种子分别保存预测、指标和优化轨迹，并报告 min/median/max/mean/std；
5. 种子波动不是5份独立用户样本，不能把5个 seed 当成显著性样本量；
6. 用户簇 bootstrap 在五种子 ensemble 的同一目标行预测上执行2,000次，并与 seed variability 分开报告；
7. 增删种子属于协议变更，需要新合同版本。

在配置冻结后，科学神经拟合规划上限为6个神经候选 × 5个种子 = 30次。工程 smoke 与 Gold pilot 不得进入科学指标；pilot 后不得追加超参数网格，除非建立新版本并重新批准。

## 6. Gold 数据与固定目标行

Gold 的每个目标样本至少需要 `sample_id`、`user_id`、`video_id`、`target_time_ms`、`target_batch_id`、split、标签、静态/历史特征和审计字段。正式构建前先在固定 Train 子集试制，并满足：

- 任一历史记录严格早于目标时间；
- 同时间戳反馈不可见；
- 当前反馈未进入特征；
- 相同输入重复构建得到相同 sample ID、行数和哈希；
- 10/50/200/500窗口使用同一 `target_row_manifest`；
- 历史不足只使用 mask，不删目标样本；
- canonical、公式重算 sensitivity 与 legacy reproduction 分开保存和对账；
- 手工抽取用户时间线可以从 Silver 逐行复算。

当前 `data/gold/` 只有 `.gitkeep`，尚无已批准 Gold 合同、run manifest 或特征 manifest，因此本报告不授权训练。

## 7. 分阶段执行与停止条件

1. **Design gate**：模型注册表、完整对照、5个种子、fail-closed 合同完成；本轮达到该中间状态。
2. **Gold pilot gate**：批准 Gold 合同，试制、手工时间线复算和泄漏测试全部通过。
3. **Implementation gate**：runner、专项测试、仓库回归、五种子确定性 smoke 和禁止透明 CPU fallback 全部通过。
4. **Train rolling-origin gate**：冻结单一训练配置和 operational GPU 预算，所有候选使用相同目标行并完成5种子。
5. **Validation gate**：只在冻结候选集合和预算内选择模型/校准器，不新增窗口或搜索空间。
6. **Final evaluation gate**：冻结模型、校准器、阈值、SESOI、MDE、代码与制品哈希，并获得精确合同哈希授权后只执行一次。

以下任一情况必须停止：Gold/feature/target-row manifest 缺失；时点违规；窗口模型行集不同；候选缺少任一种子；预算或超参数未冻结；合同哈希未获批准；概率健康门失败；请求的数据访问未明确授予。

## 8. 评估和发布边界

主指标为 event-micro AP、event-weighted user-GAUC、Log Loss 和 Brier；ROC-AUC、ECE20、user-equal gAUC 为描述性。每个模型同时报告逐日、历史深度、活跃度、warm/cold user、warm/behavior-cold video、时长/缺失和内容模态代理切片。

排序提升不能替代概率门。每个候选必须通过有限概率、饱和审计以及 Log Loss/Brier 不劣于 BL0 的绝对 sanity。AP SESOI、user-GAUC 非劣/优效边界和 bootstrap seed 仍须在首次正式 Validation 前冻结；MDE 只能由同一 Validation 行上的配对模型差值方差估计。

v013 最强允许结论只能是：在冻结数据、目标行、五种子 ensemble、神经结构和评估合同下，某个候选相对指定基线具有离线预测增量。它不能证明全库召回、线上因果收益、用户真实语义偏好或无需新数据确认的部署效果。

## 9. 预期产物

获准执行后，v013 至少要产生 Gold/feature/target-row manifests、preflight、每种子 run manifest 与优化轨迹、五种子模型状态、ensemble/校准器规格、逐行预测、pooled/daily/slice 指标、2,000次配对用户簇 bootstrap、概率分布/可靠性审计、访问账本、artifact hash manifest、run manifest、`final_decision.json` 和链接全部证据的人类可读结果报告。

当前完成的是**可审查设计与 fail-closed 合同**；剩余条件是 Gold 合同与 pilot、训练超参数/预算冻结、实现与测试、精确合同哈希批准。没有生成任何模型结果。
