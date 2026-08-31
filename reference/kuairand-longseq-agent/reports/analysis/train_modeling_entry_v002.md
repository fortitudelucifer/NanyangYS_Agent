# KuaiRand-1K v002 Train-only 数据建模入口

> 日期：2026-08-14  
> 协议：`configs/experiment_v002_proposal.yaml`  
> 授权状态：`approved_for_train_only_data_modeling`  
> 本文件性质：建模入口与 canonical Train 人群基线，不是模型训练报告

## 1. 本阶段允许做什么

当前只允许：

- 在 2022-04-08 至 04-17 的 `early_standard` 上做描述性关联和测量设计。
- 建立 feature availability、point-in-time 规则和 RQ/hypothesis registry。
- 在固定目标行上做轻量基线原型，用于估计覆盖率、方差和后续预算。
- 起草 Gold 合同，但不构建全量 Gold。

当前不允许：正式候选模型训练、Validation 模型选择、restricted test 模型评估或 random model audit。

## 2. Canonical Train 构造

目标事件只使用：

```text
data/silver/events_early_standard.parquet
UNION ALL
data/quarantine/label_formula_mismatch_rows.parquet
  WHERE source_table = 'early_standard'
    AND exclusion_reason = 'LONG_VIEW_FORMULA_MISMATCH'

WHERE event_date BETWEEN 2022-04-08 AND 2022-04-17
  AND tab = 1
```

约束：

- 保留数据提供的官方 `long_view`。
- 使用 `(source_table, source_row_number)` 作为源身份键。
- 不加回 conflicting event、invalid domain 或 dimension conflict。
- 不读取 04-18 至 04-21 的 Validation 标签关系。
- 不读取 late/random 做新的特征—标签关系筛选。
- 当前事件的标签、反馈、播放和停留结果禁止作为特征。

## 3. 入口人群基线

只读 DuckDB 对上述口径的核验结果：

| 项目 | 数值 |
|---|---:|
| Canonical Train `tab=1` 目标事件 | 2,399,844 |
| 用户 | 950 |
| 视频 | 974,550 |
| 官方 `long_view=1` | 765,417 |
| 官方 `long_view` 率 | 31.8944% |
| 从 exclusive mismatch 加回的事件 | 14,070 |
| 加回事件中的官方正例 | 14,069 |
| 加回事件中的官方负例 | 1 |

这些数字只冻结建模入口人群，不构成特征有效性、模型表现或因果结论。

## 4. 第一批真实数据建模任务

按以下顺序执行：

1. **Feature availability registry**：逐字段登记来源、连接键、目标时点可用性、缺失状态和 allow/diagnostic/deny 角色。
2. **Canonical Train profile**：逐日、用户活跃度、历史深度、内容模态代理和时长状态的样本量与标签率。
3. **三口径关联**：事件边际、user-conditional、within-timestamp diagnostic；任何单变量结果都不得直接成为硬筛选器。
4. **固定目标行设计**：为无历史、统计历史、10/50/200 批次窗口定义可比较的相同行集。
5. **RQ registry**：预注册概率预测增量、用户内判别、历史深度和时间/内容泛化问题。
6. **Gold pilot 与预算基准**：在进入正式训练前估计 CPU/GPU、RAM、磁盘、trial 和 Validation look 预算。

## 5. 下一产物

- `configs/feature_availability_v002.yaml`
- `reports/analysis/train_association_report_v002.md`
- `configs/hypothesis_registry_v002.yaml`
- `configs/compute_search_budget_v002.yaml`

本入口通过后才讨论 Gold pilot；它不提供 Gold 构建或模型训练权限。

## 6. 依据

- [`../../configs/experiment_v002_proposal.yaml`](../../configs/experiment_v002_proposal.yaml)
- [`../../NEXT_RESEARCH_SEQUENCE.md`](../../NEXT_RESEARCH_SEQUENCE.md)
- [`../research_adjustment_checkpoint_2026-08-14.md`](../research_adjustment_checkpoint_2026-08-14.md)
- [`../partial_unblinding_access_log.md`](../partial_unblinding_access_log.md)
