# Gate 2B Train-only 固定行基线结果（2026-08-15）

> 结论：BL2 相对 BL1 的七日 Train rolling-origin **排序/相对稳定性门禁通过**，但相对常数 BL0 的**绝对概率质量 sanity 失败**；Gate 2B 已完成但**不得晋级序列模型或 Validation**。  
> 边界：这不是 Gold、Validation 或测试集结果，也不是因果或线上效果证明。

![Gate 2B 结果概览](../figures/gate2b_baseline_results_v002.png)

## 做了什么

- 数据：canonical source-Train 2022-04-08 至 04-17；目标 `tab=1`，历史使用全 tab；所有历史满足 `history_time_ms < target_time_ms`。
- 固定行：04-11 至 04-17 共 7 个逐日 assessment origin，BL0/BL1/BL2 在每一天使用完全相同的 source identity 和标签。
- BL0：仅使用 origin 之前目标行的正例率；BL1：静态 ID/内容元数据的稀疏线性基线；BL2：在 BL1 上增加严格用户历史。高基数 user×content 历史因 pre-metric quick 性能止损而延后独立评估。
- 选择：只在预注册的 04-11、04-14、04-17 三个搜索 origin 内选择 alpha 与 BL2 bundle；冻结后再重新拟合七日回测。
- 不确定性：950 用户、2,000 次共享 PCG64 user-cluster bootstrap；只解释为 Train-only 设计证据。

## 冻结模型

- BL1：`BL1_S_ID_CONTENT_V1_A1em06`
- BL2：`BL2_H2_USER_STRICT_V1_A1em04`
- 实际拟合：32 / 32；总耗时 452.4 秒。

## 同一批 04-11 至 04-17 目标行的 pooled 指标

| 模型 | Average precision | user-GAUC（事件加权） | Log Loss | Brier | ECE20（描述） |
|---|---:|---:|---:|---:|---:|
| BL0 | 0.324906 | 0.498371 | 0.627359 | 0.217828 | 0.007802 |
| BL1 | 0.324844 | 0.501942 | 5.162145 | 0.320341 | 0.320339 |
| BL2 | 0.435665 | 0.523877 | 4.269645 | 0.275823 | 0.275223 |

BL2 − BL1：ΔAP=0.110821，Δuser-GAUC=0.021935，ΔLog Loss=-0.892500，ΔBrier=-0.044518；7 天中 7 天 ΔAP>0。

## Post-release 概率 sanity 审计

预注册的相对门禁只比较 BL2 与冻结 BL1，因此它能回答“历史特征是否改善了同一线性模型”，却没有阻止 BL1 和 BL2 同时成为糟糕的概率模型。保守的 post-release 审计发现：

- BL2 相对 BL0：ΔAP=+0.110759，但 ΔLog Loss=+3.642286、ΔBrier=+0.057995、ΔECE20=+0.267421；后三项数值上明显更差。这里没有为 BL2−BL0 计算推断区间，因此不使用“统计显著”措辞。
- BL1 有 98.50% 的预测落在下限 clip，另有 1.47% 落在上限 clip；BL2 分别为 82.89% 与 11.88%。这是严重的概率饱和，不是可接受的校准状态。
- 因此只能保留“严格用户历史带来稳定排序增量”的 Train-only 证据，不能发布“概率预测已经改善”或“模型已准备好进入下一阶段”的结论。

该 sanity 判定是在结果产生后的保守审计，不伪装成预注册确认性门禁。可复算证据见 [`probability_sanity_audit.csv`](../generated/gate2b_baselines_v002/probability_sanity_audit.csv) 与 [`postrelease_audit_manifest.json`](../generated/gate2b_baselines_v002/postrelease_audit_manifest.json)。

## 配对用户聚类不确定性

| 对比指标（BL2−BL1） | 点估计 | 95% percentile CI | Train-only 解读 |
|---|---:|---:|---|
| Average precision | 0.110821 | [0.101479, 0.120377] | 聚合判别增量 |
| user-GAUC | 0.021935 | [0.019190, 0.025003] | 用户内判别增量 |
| Log Loss | -0.892500 | [-1.019237, -0.773907] | 小于 0 更好 |
| Brier | -0.044518 | [-0.051376, -0.038125] | 小于 0 更好 |

## 当前只能得出的结论

1. 严格用户历史在 source-Train 固定行上带来稳定的排序增量：ΔAP 与 Δuser-GAUC 在 7/7 天均为正，用户聚类 CI 也不跨 0。
2. 当前 SGD 概率输出严重饱和；虽然 BL2 相对 BL1 的 Log Loss/Brier 改善，但二者均明显差于常数 BL0，因此概率质量不合格。
3. ECE20 保持描述性，不单独作为通过门禁；但 BL0 对照暴露出的绝对 Log Loss/Brier 失败足以阻止晋级。
4. 下一步只能先版本化预注册概率基线修复（优化器/尺度/时间内校准及 BL0 绝对 sanity 门），不能根据本轮结果静默加网格重跑。Silver 的重洗、重建或覆盖仍关闭；Gold、序列模型和 Validation 也未获授权。
