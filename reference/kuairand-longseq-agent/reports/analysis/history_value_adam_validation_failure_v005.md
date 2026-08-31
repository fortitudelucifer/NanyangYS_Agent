# 历史特征价值 Adam Validation v005：数据完整性停止记录

记录时间：2026-08-23

## 结论先行

v005 已按批准合同打开 Validation，但在 GPU 拟合、预测和指标计算之前被目标行数完整性门停止。停止不是模型失败，也不能用于判断历史是否有用。

根因已经通过源表聚合精确定位：合同中的 `881,035` 是 early-standard Silver 主表在 2022-04-18 至 04-21 的 `tab=1` 行数；同一合同还要求加回 `5,417` 条 `LONG_VIEW_FORMULA_MISMATCH` 隔离行。因此 canonical 目标总数应为 `886,452`。

## 数据核对

| 来源 | tab=1 行数 | 正例数 | 唯一来源身份数 |
|---|---:|---:|---:|
| early-standard Silver | 881,035 | 279,081 | 881,035 |
| 官方公式不一致隔离行加回 | 5,417 | 5,417 | 5,417 |
| canonical union | 886,452 | 284,498 | 886,452 |

逐日计数保存在 `reports/generated/history_value_adam_validation_v005/validation_target_count_diagnosis.csv`，可直接绘图。

## 执行边界

- Train-only Adam 充分性证据已通过哈希验证。
- Validation 已打开并完成 SQL 临时表的行数检查。
- 特征 Parquet 尚未正式复制完成。
- GPU BL1/BL2 拟合未启动。
- Prediction、bootstrap 和任何 Validation 指标均未计算。
- sealed test 与 random audit 均未访问。

## 修复原则

后继合同只允许把 Validation canonical union 的预期目标行数从 `881,035` 改为 `886,452`。数据来源、加回规则、特征 SQL、Adam `lr=0.03/100 steps`、BL0/BL1/BL2、2,000 次配对用户簇 bootstrap、统计门和 Validation-only 权限全部保持不变。

这属于预期计数元数据修复，不是根据模型结果调整协议，因为停止点早于模型拟合和指标读取。
