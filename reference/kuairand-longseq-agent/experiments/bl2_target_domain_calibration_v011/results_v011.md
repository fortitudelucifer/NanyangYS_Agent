# BL2 目标域后置校准结果 v011

- 生成时间：2026-08-23T21:56:13.364758+08:00
- 合同 SHA-256：`d32a086e3deb96f326887b7cf600f769f81034c9160acc14173dfbd164fbe485`
- 证据等级：post-audit、held-out-to-calibrator temporal test；不是新的 pristine test。
- BL2、H2、Adam、特征和历史全部冻结；只拟合后置校准器。

## 时间切分

| split | rows | users | positives | prevalence |
|---|---:|---:|---:|---:|
| calibration_fit | 8,731 | 645 | 686 | 0.078571 |
| calibration_selection | 10,544 | 844 | 872 | 0.082701 |
| held_out_test | 23,752 | 967 | 2,063 | 0.086856 |

## 方法选择

| family | log-loss | Brier | ECE20 | mean p | eligible |
|---|---:|---:|---:|---:|---|
| M1_prior_shift | 0.262257 | 0.072477 | 0.033304 | 0.115979 | True |
| M2_intercept_only | 0.255865 | 0.070951 | 0.004318 | 0.080861 | True |
| M3_platt | 0.255849 | 0.070952 | 0.003854 | 0.080871 | True |

选择家族：`M2_intercept_only`；合并前两段后的最终参数：slope=`1.01690434681`，intercept=`-2.09520994809`。

## 六天留出结果

| model | log-loss | Brier | ECE20 | AP | event-gAUC | mean p |
|---|---:|---:|---:|---:|---:|---:|
| original_BL2 | 0.512076 | 0.169978 | 0.281335 | 0.197730 | 0.623640 | 0.368191 |
| selected_calibrator | 0.268939 | 0.074827 | 0.006724 | 0.197730 | 0.623640 | 0.082333 |

| contrast | point | 95% CI |
|---|---:|---:|
| Δlog_loss | -0.243137 | [-0.258633, -0.226836] |
| Δbrier | -0.095151 | [-0.102219, -0.087959] |
| Δaverage_precision | 0.000000 | [0.000000, 0.000000] |
| Δuser_gauc_event_weighted | 0.000000 | [0.000000, 0.000000] |

## 决策

- log-loss gate：True
- Brier gate：True
- ECE gate：True
- mean probability gate：True
- probability health gate：True
- ranking invariance gate：True
- 最终状态：`pass`

该结果只评价后置概率校准；无论通过或失败，都不改变 v010 的 BL2 历史排序结论。
