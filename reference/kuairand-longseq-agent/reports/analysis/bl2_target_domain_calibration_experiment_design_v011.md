# BL2 目标域概率校准：下一步实验设计 v011

状态：已形成精确合同并于 2026-08-23 执行完成，最终状态 `pass`  
设计日期：2026-08-23（Asia/Shanghai）  
输入说明：`/data/master course/Kuai/BL2概率偏高与目标域校准说明.md`  
冻结前置合同：random-only v010，SHA-256 `a73b254c43de31be6d9490a7b33398f77cf7fd49e7b0624c931886beca8fea04`

执行合同 SHA-256：`d32a086e3deb96f326887b7cf600f769f81034c9160acc14173dfbd164fbe485`  
完整结果：`experiments/bl2_target_domain_calibration_v011/results_v011.md`

## 1. 实验目标

保持 BL2 模型、静态特征、H2 历史特征、类别编码、scaler 和 Adam 参数全部冻结，只在 BL2 输出之后增加一个轻量目标域校准层，回答两个可证伪问题：

1. random 域的概率偏高是否主要能由基准率/截距变化解释？
2. 校准后能否显著降低 log-loss 与 Brier，同时严格保持 BL2 已验证的排序能力？

本实验不重新验证“历史有没有用”，也不允许因校准结果修改 BL2、H2、Adam、SGD 或 v010 结论。

## 2. 当前数据条件与证据等级

本地 governed snapshot 只有 2022-04-22 至 05-08 这一份 random 目标域数据。它已经被 v010 最终审计完整消费，目前不存在新的 untouched random 时间段。

因此实验分为两层：

- **阶段 A：现有数据上的 post-audit 探索性校准实验。** 校准器的拟合、选择和评价按时间严格隔离，但全量 random 的标签率、原模型指标和逐日指标已经被看过。阶段 A 可以指导工程方案，不能冒充 pristine confirmatory test。
- **阶段 B：未来新目标域数据上的正式确认。** 先冻结阶段 A 选出的校准器家族，再用新的目标域校准集拟合，用新的 untouched 测试集只评估一次。

## 3. 阶段 A：现有 v010 冻结预测上的探索性实验

### 3.1 唯一输入

只读取 v010 已封存的逐行预测：

`reports/generated/history_value_adam_random_v010/random_audit/predictions.parquet`

- 行数：`43,027`；
- SHA-256：`48ae0793e554b1eb82d3ba75684915f471cf52f005039a6ad674075b4afffc31`；
- 使用字段：目标身份、日期、用户、标签、`raw_ADAM_BL2`、`p_ADAM_BL2`；
- 不重新打开原始 random parquet；
- 不拟合 BL1/BL2，不重新生成特征，不更新历史。

### 3.2 冻结时间切分

按日期而非随机行切分，避免未来标签进入过去校准：

| 角色 | 日期 | 行数 | 正例数 | 正例率 | 用途 |
|---|---|---:|---:|---:|---|
| calibration-fit | 04-22 至 04-27 | 8,731 | 686 | 7.857% | 只拟合候选校准器 |
| calibration-selection | 04-28 至 05-02 | 10,544 | 872 | 8.270% | 只选择候选与冻结参数 |
| held-out-to-calibrator test | 05-03 至 05-08 | 23,752 | 2,063 | 8.686% | 只进行一次最终方法评价 |

最后一段只能称为“对校准器留出”，不能称为全项目 untouched test，因为 v010 已经看过该时段的原始模型结果。

### 3.3 模型与候选校准器

`M0` 为不调整的冻结 BL2，仅作对照，不参加选择。最多比较三个候选：

1. `M1_prior_shift`：在冻结 `p_ADAM_BL2` 的 logit 上，只加 calibration-fit 正例率相对源校准集正例率的先验偏移。源校准集为 `70,457 / 222,256 = 0.31700831473616009`，目标 calibration-fit 为 `686 / 8,731 = 0.078570610468445767`，所以选择阶段偏移为 `-1.6943737490338504`；无迭代参数搜索。若 M1 被选中，使用前两段合并率 `1,558 / 19,275 = 0.080830090791180281` 重算最终偏移 `-1.6635670005029957`。
2. `M2_intercept_only`：冻结 sealed BL2 校准斜率 `1.0169043468053687`，只在 calibration-fit 上最大似然拟合新截距。实现使用单调 score equation 的 bracketed root，不正则化截距；搜索区间、容差和最大迭代必须在合同中冻结。
3. `M3_platt`：在 `raw_ADAM_BL2` 上拟合 `sigmoid(a × raw + b)`，沿用现有校准实现：scikit-learn LBFGS、L2、`C=1.0`、`tol=1e-10`、`max_iter=1000`、float64；只有正常收敛且 `a > 0` 时才合格。

不把 isotonic、分桶查表、神经校准器或分群校准器放进第一轮，避免在 686 个 calibration-fit 正例上扩大搜索与过拟合。

### 3.4 候选选择规则

所有候选先只在 calibration-fit 拟合，再一次性读取 calibration-selection 指标：

1. 先淘汰非有限概率、`a ≤ 0`、排序变化或优化不收敛的候选；
2. 在剩余候选中选择 calibration-selection log-loss 最低者；
3. 若与最低值相差不超过 `0.0001`，选择更简单者，优先级为 `M1 → M2 → M3`；
4. 选择家族后，使用 calibration-fit 与 calibration-selection 合并数据重新拟合该家族；随后冻结最终参数、代码与预测哈希，才允许读取 test 指标；
5. test 失败不得回到 selection 重新选方法或降低门槛。

### 3.5 主统计分析

主总体为全部 random exposures；warm users 为敏感性分析。比较为同一行上的 `selected_calibrator − M0`。

- 点指标：log-loss、Brier、ECE20、calibration-in-the-large、校准斜率/截距；
- 排序审计：AP、ROC-AUC、event-weighted user-gAUC；
- 推断：`2,000` 次配对用户簇 bootstrap，以用户为重采样单位；
- 概率计算使用 float64，并记录 clipping 规则；
- reliability diagram 使用固定 20 个等宽桶，同时输出等频桶作为描述性敏感性分析。

### 3.6 冻结通过门

阶段 A 的 held-out-to-calibrator test 必须同时满足：

1. `Δlog-loss < 0`，且配对用户簇 95% CI 上界 `< 0`；
2. `ΔBrier < 0`，且配对用户簇 95% CI 上界 `< 0`；
3. `ECE20 ≤ 0.02`；
4. `|mean(calibrated_probability) − observed_prevalence| ≤ 0.01`；
5. 所有概率有限、位于 `[0,1]`，极端饱和比例不超过 v010 概率健康门；
6. 校准映射严格单调递增；稳定排序索引不变，AP/ROC-AUC/user-gAUC 的数值差不超过 `1e-10`；
7. 任何 test 结果都不得修改 BL2、候选集合、选择规则或门槛。

ECE 与均值概率门用于判断概率刻度是否达到工程可解释水平；主要统计证据仍是配对 log-loss/Brier 改善。

### 3.7 切片与监控

以下只作预注册敏感性分析，不用于选择校准器：

- history `0–49`、`50–199`、`200+`、探索性 `500+`；
- warm/cold user；
- warm/behavior-cold video；
- tag known/unknown；
- video duration valid/invalid；
- 每日 prevalence、mean predicted probability、log-loss、Brier 和 ECE。

仅对至少 `500` 行且至少 `50` 个正例的切片报告 paired interval；更小切片明确标记为不稳定描述。重点监控 v010 已暴露的低历史组和 duration-invalid 组。

## 4. 结果如何解释

| 观察结果 | 解释 | 下一步 |
|---|---|---|
| M1/M2 已通过，M3 无实质额外改善 | 偏差主要是基准率/截距漂移 | 部署截距校准，优先简单方案 |
| M3 明显优于 M1/M2 且通过 | 除基准率外还有 score scale/conditional shift | 部署二参数 Platt，监控斜率 |
| 三者都改善但未过 ECE/损失门 | 全局单调校准不足 | 另立合同研究分段或非线性校准 |
| 排序指标发生变化 | 实现错误、数值饱和或非单调映射 | fail-closed，不发布 |
| 总体通过但关键大切片明显恶化 | 单一全局校准器掩盖异质漂移 | 暂不发布概率，设计分群校准实验 |

无论哪种结果，都不改变 v010 已经成立的“BL2 排序历史增量”结论。

## 5. 阶段 B：新目标域数据的正式确认实验

阶段 A 只能选工程方案。要允许 Agent 把输出解释为真实概率，需要额外收集与实际部署曝光机制一致的新数据，并在收集前冻结阶段 B 合同。

建议最低规模：

- 新 calibration-fit：约 `12,000` 行，按 8.4% 预计约 `1,000` 个正例；
- 新 untouched test：至少 `20,000` 行，预计约 `1,680` 个正例；
- 若仍需在新域比较多个校准家族，应再增加独立 selection 段约 `10,000` 行；若阶段 A 已冻结唯一家族，则阶段 B 不再进行方法选择。

切分必须按连续时间块或完整请求批次进行，同 timestamp/请求不能跨边界。数据收集结束日期、行范围和输入哈希必须在查看 test 标签或预测指标前冻结；不得因结果接近门槛而追加样本。

阶段 B 使用与阶段 A 相同的主门，并额外要求：

- test 只打开一次；
- calibration-fit 标签不能更新 BL2/H2，只能拟合后置校准参数；
- test 标签不参与拟合、方法选择、阈值选择或历史更新；
- 通过后序列化 `a/b` 或截距、训练域、适用 exposure domain、输入 score 版本与 SHA-256；
- 输出同时保留 `ranking_score` 和 `calibrated_probability`。

## 6. 计算与工程建议

本实验不需要重新训练大模型。两参数 logistic calibration 和 43k 行的 bootstrap 在 conda `Kuai` 环境的 CPU 上即可高效完成；GPU 不会带来有意义的收益。应把 GPU 资源留给后续模型训练实验。

实现前应先写 v011 精确合同并固定：

- v010 合同、预测与制品 manifest 哈希；
- 三段日期与行身份清单；
- 三个候选的精确定义、优化器、正则化、容差和最大迭代；
- 选择规则、tie-break、bootstrap seed/replicates、全部 gates；
- 输出 CSV/Parquet/JSON/MD 清单；
- 禁止重新打开原始 random、禁止 BL2 重训、禁止 test 后返工。

## 7. 推荐执行顺序

1. 先把本设计转成 v011 exploration-only 精确合同和测试；
2. 获得合同哈希批准后，从 v010 冻结 predictions 运行阶段 A；
3. 根据阶段 A 结果决定工程候选，但报告明确标注 post-audit；
4. 收集新目标域数据；
5. 另立确认性合同运行阶段 B；
6. 阶段 B 通过后，才允许 CPU Agent 把 `calibrated_probability` 用于概率阈值、收益估计或面向用户的概率解释。
