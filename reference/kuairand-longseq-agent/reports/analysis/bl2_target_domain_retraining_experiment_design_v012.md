# BL2 目标域成对重训练实验设计 v012

状态：冻结设计，等待精确合同哈希批准  
设计日期：2026-08-23（Asia/Shanghai）

## 1. 问题与对照

v012 只回答一个问题：允许模型权重使用 random 目标域的早期标签进行适配后，`NEW_BL2` 是否显著优于已经通过 v011 的 `OLD_BL2 + M2 intercept calibration`。

为了继续识别历史价值，每个配置必须同时训练：

- `NEW_BL1`：sealed 静态模型 warm-start；
- `NEW_BL2`：sealed 历史模型 warm-start；
- `OLD_BL2_PLUS_V011`：冻结不训练的工程对照；
- `TARGET_BL0`：只使用 train 与 calibration 标签形成的常数概率。

SGD 保持冻结。v012 不能修改 v010“历史排序有用”或 v011“截距校准有效”的既有结论。

## 2. 固定时间切分

| 角色 | 日期 | 行数 | 用户 | 正例 | 正例率 |
|---|---|---:|---:|---:|---:|
| target-adaptation train | 04-22 至 04-29 | 11,999 | 760 | 947 | 0.078923 |
| target calibration | 04-30 至 05-02 | 7,276 | 778 | 611 | 0.083975 |
| model selection | 05-03 至 05-05 | 11,353 | 875 | 1,024 | 0.090196 |
| final temporal replay | 05-06 至 05-08 | 12,399 | 857 | 1,039 | 0.083797 |

四段均按连续日期切分。同一行在 v012 内只能承担一个角色。最后三天已在 v010/v011 中被总体查看，因此只能称为回顾性 temporal replay。

## 3. 固定模型与三个候选配置

不重新拟合 one-hot 词表或 scaler；使用 sealed 重建模型的冻结设计，使所有 random 行保持 BL1 13,527 列、BL2 13,545 列，并允许精确 warm-start。

目标函数为：

```text
mean BCE(target train) + tether_strength / 2 × ||w - w_sealed||²
```

截距不做 tether；训练后另用独立 calibration 段修正概率截距。

| 配置 | Adam lr | steps | tether | 解释 |
|---|---:|---:|---:|---|
| C1_conservative | 0.003 | 100 | 0.01 | 强约束、小更新 |
| C2_balanced | 0.01 | 100 | 0.001 | 中等更新 |
| C3_aggressive | 0.03 | 100 | 0.0001 | 弱约束、大更新 |

三个配置使用完全相同的 train 行、初始化、特征、模型对和预算定义。禁止根据 selection/test 结果新增配置。

## 4. 校准、选择与冻结

每个候选在 target calibration 上只拟合截距：BL1 斜率固定为 sealed 的 `1.194068183024429`，BL2 斜率固定为 v011/ sealed 的 `1.0169043468053687`。这是 v011 已选 M2 家族在新模型上的预注册延续。

selection 上先检查历史增量、静态基线、概率健康和相对旧方案的 AP 非劣。合格候选中选择 `NEW_BL2` log-loss 最低者；在 `0.0001` 内按 C1 → C2 → C3 选择更保守的配置。若没有候选合格，仍按同一 log-loss 规则冻结一个诊断候选并继续 final replay，但 selection gate 保持失败，不允许借此发布新模型。

选择后不使用 selection 标签重训练或重校准。模型、校准器、预测代码与 SHA-256 先冻结，再打开最后三天。

## 5. 最终判断门

v012 只有同时满足以下条件，才能判断“重训练提供额外价值”：

1. `NEW_BL2 - NEW_BL1`：ΔAP ≥ `+0.005` 且用户簇 bootstrap 95% CI 下界 > 0；event-gAUC 不下降；log-loss/Brier 点估计不恶化；
2. `NEW_BL1 - TARGET_BL0`：AP 区间下界 > 0，且排序/损失方向正确；
3. `NEW_BL2 - OLD_BL2_PLUS_V011`：log-loss 或 Brier 至少一项的 95% CI 上界 < 0，另一项点估计不恶化；
4. NEW_BL2 相对旧方案的 AP 95% CI 下界 > `-0.005`；
5. NEW_BL2 的 ECE20 ≤ `0.02`、平均概率误差 ≤ `0.01`、概率有限且无异常饱和；
6. 至少 2/3 天的 `NEW_BL2 - NEW_BL1` AP 为正，每天平均概率误差 ≤ `0.02`；
7. 主要区间均使用 2,000 次配对用户簇 bootstrap，seed=`20260824`。

未同时通过时，工程结论为继续保留冻结 `OLD_BL2 + v011`，而不是宣称历史无用。

## 6. 证据边界

v012 是已知数据上的目标域适配和时间回放，能用于当前 demo 的工程选择，但不能作为新数据独立确认。若未来采集新的目标域日志，应固定本实验选出的单一训练与校准流程，再在新 untouched test 上确认一次。
