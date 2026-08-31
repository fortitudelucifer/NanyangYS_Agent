# BL2 目标域成对重训练结果 v012

- 生成时间：2026-08-23T22:37:21.935070+08:00
- 合同 SHA-256：`bd362bbd40a6cedcb7f6859713337bdadb2e9e03690e9117224801a35e1cb6d5`
- 设备：conda Kuai / CUDA / RTX 5070 Ti；优化器：GPU full-batch Adam。
- SGD 继续冻结；v010/v011 制品未修改。
- 证据等级：post-audit temporal replay，不是新的 pristine test。

## 时间切分

| split | rows | users | positives | prevalence |
|---|---:|---:|---:|---:|
| target_adaptation_train | 11,999 | 760 | 947 | 0.078923 |
| target_calibration | 7,276 | 778 | 611 | 0.083975 |
| model_selection | 11,353 | 875 | 1,024 | 0.090196 |
| final_temporal_replay_test | 12,399 | 857 | 1,039 | 0.083797 |

## Selection

选择配置：`C2_balanced`；selection 合格：`True`；fallback：`False`。

| config | eligible | ΔAP BL2-BL1 | Δlogloss BL2-BL1 | ΔAP NEW-OLD | NEW_BL2 logloss |
|---|---|---:|---:|---:|---:|
| C1_conservative | True | 0.029004 | -0.008959 | 0.007893 | 0.274048 |
| C2_balanced | True | 0.040310 | -0.010101 | 0.021058 | 0.271899 |
| C3_aggressive | True | 0.038384 | -0.013004 | 0.022749 | 0.274999 |

最终冻结参数摘要：

- NEW_BL1: displacement L2=`3.224509`，raw intercept=`-0.238640084863`，calibration slope=`1.19406818302`，calibration intercept=`0.464839839243`。
- NEW_BL2: displacement L2=`3.052129`，raw intercept=`-0.215435877442`，calibration slope=`1.01690434681`，calibration intercept=`0.0740855644936`。

## Final temporal replay

| model | AP | event-gAUC | log-loss | Brier | ECE20 | mean p |
|---|---:|---:|---:|---:|---:|---:|
| TARGET_BL0 | 0.083797 | 0.500000 | 0.288005 | 0.076784 | 0.002967 | 0.080830 |
| OLD_BL2_PLUS_V011 | 0.187896 | 0.621353 | 0.263925 | 0.072850 | 0.006645 | 0.081573 |
| NEW_BL1 | 0.176476 | 0.613925 | 0.269303 | 0.073613 | 0.006894 | 0.077465 |
| NEW_BL2 | 0.204928 | 0.645537 | 0.260618 | 0.072098 | 0.006200 | 0.079689 |

| contrast / metric | point | 95% CI |
|---|---:|---:|
| NEW_BL1_minus_TARGET_BL0 / average_precision | 0.092679 | [0.073879, 0.118385] |
| NEW_BL1_minus_TARGET_BL0 / user_gauc_event_weighted | 0.113925 | [0.088970, 0.138765] |
| NEW_BL1_minus_TARGET_BL0 / log_loss | -0.018702 | [-0.023458, -0.014445] |
| NEW_BL1_minus_TARGET_BL0 / brier | -0.003171 | [-0.004216, -0.002270] |
| NEW_BL2_minus_NEW_BL1 / average_precision | 0.028452 | [0.015166, 0.041826] |
| NEW_BL2_minus_NEW_BL1 / user_gauc_event_weighted | 0.031612 | [0.011644, 0.050783] |
| NEW_BL2_minus_NEW_BL1 / log_loss | -0.008684 | [-0.011364, -0.006119] |
| NEW_BL2_minus_NEW_BL1 / brier | -0.001515 | [-0.002099, -0.000970] |
| NEW_BL2_minus_OLD_BL2_PLUS_V011 / average_precision | 0.017032 | [0.005265, 0.031319] |
| NEW_BL2_minus_OLD_BL2_PLUS_V011 / user_gauc_event_weighted | 0.024183 | [0.007412, 0.041419] |
| NEW_BL2_minus_OLD_BL2_PLUS_V011 / log_loss | -0.003307 | [-0.005145, -0.001725] |
| NEW_BL2_minus_OLD_BL2_PLUS_V011 / brier | -0.000752 | [-0.001243, -0.000333] |

## 决策

- selection_eligibility_gate：True
- history_value_gate：True
- static_baseline_gate：True
- retraining_probability_value_gate：True
- old_ranking_noninferiority_gate：True
- calibration_health_gate：True
- daily_stability_gate：True
- 最终状态：`retraining_adds_value`

若最终状态不是 `retraining_adds_value`，CPU Agent 应继续使用冻结 BL2 + v011 截距校准。
任何结果都不推翻 v010 的历史排序结论，也不升级为新数据独立确认。
