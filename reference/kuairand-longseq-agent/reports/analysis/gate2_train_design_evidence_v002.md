# Gate 2 Train-only 设计证据 v002

> 状态：**设计证据已生成；不代表 Gate 2 已通过，也不是模型训练结论**  
> 数据边界：canonical source-Train，2022-04-08 至 2022-04-17；目标 `tab=1`；历史使用同范围全 tab，严格 `history_time < target_time`  
> 禁止访问：late、random、Validation、restricted test、`video_features_statistic_1k.csv`；未重清洗 Silver，未构建 Gold

## 1. 本轮回答什么

本轮只为四个待定设计提供可复算证据：calendar burn-in 的覆盖代价、10/50/200 固定目标行、`picture-like` 候选代理的映射审计、以及 user-cluster bootstrap 的最小实现。canonical 目标仍为 2,399,844 行、950 个用户、765,417 个正例；源身份键唯一。既有 release 覆盖表的最大绝对复算差为 `0.0e+00`。

## 2. Calendar burn-in：只作敏感性和 rolling-origin 设计证据

主目标人群继续使用 **all rows + available-history mask**。B1–B2 只是敏感性候选；B3 只定义未来 Train rolling-origin 的 assessment target 日期，不定义新的 canonical 人群。

| policy | 目标起点 | 保留行 | 行保留率 | history≥50 | history≥200 | history≥500 | 用户 | 双标签用户 | 正例率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 2022-04-08 | 2,399,844 | 100.00% | 92.56% | 72.03% | 42.67% | 950 | 946 | 31.89% |
| B1 | 2022-04-09 | 2,167,921 | 90.34% | 97.25% | 79.09% | 47.22% | 950 | 944 | 31.93% |
| B2 | 2022-04-10 | 1,909,252 | 79.56% | 98.57% | 84.68% | 52.86% | 947 | 938 | 31.93% |
| B3 | 2022-04-11 | 1,641,098 | 68.38% | 98.99% | 88.43% | 57.91% | 943 | 935 | 32.05% |

B3（从 04-11 开始）把 `history≥200` 覆盖从 72.03% 提高到 88.43%，但若把它误当成人群过滤就会丢失 758,746 行（31.62%）和前三个日历日；保留期每日正例率仍覆盖 29.90%–34.88%。因此 coverage 不能证明 B3 更“正确”，也不能消除时间漂移。

- **协议决定**：canonical 主人群不按 calendar burn-in 删除目标行，短历史用显式 mask；04-08 至 04-10 冻结为历史积累期，04-11 至 04-17 冻结为未来七个逐日 Train assessment origins。该选择只依据 Train 覆盖和需要七个日回测点，不使用标签率优化。
- **不可声称**：04-11 是更“正确”的主人群起点、全局资格阈值或已被模型增量验证的最优日期；B0–B2 继续作为覆盖/敏感性证据。

## 3. 10/50/200 固定行可用性

| cohort | 比较窗口 | 目标行 | 覆盖 | 用户 | 双标签用户 | 正例率 | 源身份 digest |
|---|---|---:|---:|---:|---:|---:|---|
| all_rows_masked | 10 / 50 / 200 | 2,399,844 | 100.00% | 950 | 946 | 31.89% | `bf058e955f3f...` |
| history_10_plus | 10 | 2,364,626 | 98.53% | 945 | 939 | 31.73% | `a45a92c9c091...` |
| history_50_plus | 10 / 50 | 2,221,326 | 92.56% | 908 | 903 | 31.17% | `928b3ea8efec...` |
| history_200_plus | 10 / 50 / 200 | 1,728,489 | 72.03% | 758 | 752 | 29.16% | `3ae8532f84a4...` |
| history_500_plus | 10 / 50 / 200 / 500 | 1,024,000 | 42.67% | 452 | 449 | 24.95% | `73e8eca52afa...` |

`history_200_plus` 仍有 1,728,489 行、758 个用户，可以承担 10/50/200 的主长序列消融；但它在 04-08 的日内覆盖仅 5.97%，到 04-17 才达到 94.21%，所以任何窗口增量都必须逐日报告，不能只看池化指标。`history_500_plus` 仅保留 452 个用户，继续限定为 exploratory 是合理的。

每个 cohort 已记录过滤式和按 `(source_table, source_row_number)` 排序流式计算的 SHA-256。但这些 digest 是 **logical/source-identity prototype**，不是 Gold `sample_id` manifest。

- **可冻结**：all-row masked 主设计；`history≥50` 的 10/50 固定行消融；`history≥200` 的 10/50/200 固定行消融；500 仅 exploratory；比较内不得按模型删行。
- **仍待完成**：Gold builder 生成正式 target-row manifests、`sample_id`、特征矩阵 denylist 检查及其最终哈希。

## 4. `candidate_mapping_audit`：不是官方内容模态

名称候选只使用大小写不敏感 token `picture|photo|album`，命中的值为 `LongPicture`、`PictureSet`、`PictureCopy`、`PhotoCopy`、`FlashPhoto`、`OriginPicture`、`LocalIntelligenceAlbum`。它覆盖 183,249 行（7.64%）、936 个用户。

与 event-duration audit signal 的交集为 177,885 行，Jaccard=0.929；union 内诊断不一致为 13,582 行（7.09%）。与 `videos_basic.video_duration` signal 的 Jaccard=0.929，union 内不一致为 7.12%。这些只是两个非权威 signal 的一致性，不是 accuracy。

保守候选 `proxy_v1` 定义如下：

- `picture_like = name_flag AND videos_basic duration missing/nonpositive`；
- `video_like = NOT name_flag AND videos_basic duration positive finite`；
- 其余全部为 `unknown`。

| proxy_v1 | 行 | 覆盖 | 用户 | 正例 | 负例 | 正例率（仅结果描述） |
|---|---:|---:|---:|---:|---:|---:|
| video_like | 2,208,377 | 92.02% | 949 | 745,177 | 1,463,200 | 33.74% |
| picture_like | 177,827 | 7.41% | 933 | 18,718 | 159,109 | 10.53% |
| unknown | 13,640 | 0.57% | 830 | 1,522 | 12,118 | 11.16% |

`unknown` 的来源：

| 原因 | 行 | 覆盖 | 用户 |
|---|---:|---:|---:|
| nonpicture_name_but_missing_or_nonpositive_video_duration | 8,218 | 0.34% | 647 |
| picture_name_but_positive_video_duration | 5,422 | 0.23% | 718 |

正负例计数只用于审计异质性，**没有参与映射规则**。`video_type` 和几何完整性也没有提供官方语义确认。

- **可冻结**：`content_modality_proxy_v1` 的 diagnostic-only 规则、unknown 默认和三切片审计输出。
- **不可冻结**：把 `proxy_v1` 称为真实模态、用作正式 predictor，或据此把主人群改为 video-only。未来未见值仍自动进入 unknown。

## 5. 用户聚类不确定性最小方案

| cohort | 用户 | top 10% 用户行占比 | top 25% | 单用户最大行 | cluster SE（正例率烟测） | 相对 naive-iid SE |
|---|---:|---:|---:|---:|---:|---:|
| all_rows_masked | 950 | 35.85% | 61.38% | 23,236 | 0.00837 | 27.8x |
| history_50_plus | 908 | 36.71% | 62.24% | 23,044 | 0.00853 | 27.5x |
| history_200_plus | 758 | 38.91% | 64.65% | 22,158 | 0.00914 | 26.5x |
| history_500_plus | 452 | 41.83% | 67.08% | 21,413 | 0.01055 | 24.7x |

all-row 主人群中 top 10% 用户贡献 35.85% 事件，逐行独立假设把 event-micro 正例率 SE 低估约 27.8 倍。该数值不是模型性能不确定性，但足以否决 row-iid 推断。

最小实现为：在升序 950 用户全集上，用 `numpy.PCG64(seed=20260814)` 生成同一个 `2000 × 950` 用户 multiplicity plan；所有 cohort、estimand、baseline/candidate 共用。矩阵 digest 为 `582d0ef006da0f61fb753fa6d15d6ee801f7fce5f820fc1c92a376268112d972`。每次抽中用户时携带该用户在固定 cohort 的全部行；比较模型时复算 PR-AUC、Log Loss、Brier 和 user-GAUC 的**配对差**，不对两个模型各自独立 bootstrap。当前 CSV 只是标签率 plumbing smoke test，2000 次均有效。

- **可冻结**：cluster=`user_id`、共享 multiplicities、固定用户顺序、同一 replicate 内所有模型和指标配对。
- **不可冻结**：Train 模型差的方差必须等固定 baseline/candidate 配对预测；MDE 必须进一步等待获批的 Validation 配对预测。SESOI 和非劣界也不能由本次 label-rate smoke test 推出。

## 6. Gate 2 当前结论

| 项目 | 状态 |
|---|---|
| canonical Train 边界与唯一源身份 | 可冻结 |
| all-row + mask 主人群 | 可冻结 |
| Calendar burn-in | canonical 全行保留；B3 冻结为 assessment-only 起点，B1/B2 为 sensitivity |
| 10/50/200 cohort 逻辑和 prototype digests | 可冻结逻辑；正式 Gold manifests 待建 |
| 500 窗口 | 可冻结为 exploratory-only |
| `proxy_v1` | diagnostic-only 规则已冻结；真实模态与 predictor 声明禁止 |
| user-cluster 共享重采样设计 | 可冻结 plumbing；模型差异边界未冻结 |

因此，本轮推进了 Gate 2 的**设计证据**，但尚不能宣布 Gate 2 通过。下一步应先冻结相同 fixed rows 上的 baseline/candidate 协议，再在获得正式训练批准后产生 Train rolling-origin 配对预测，以估计 Train 原型差值和逐日稳定性；Validation MDE 仍必须等待获批的配对 Validation 预测。Gold build 也需单独批准。

## 7. 可复算产物

- `scripts/analyze_gate2_train_design_v002.py`
- `reports/generated/gate2_train_design_v002/burn_in_tradeoff.csv`
- `reports/generated/gate2_train_design_v002/fixed_row_cohort_summary.csv`
- `reports/generated/gate2_train_design_v002/fixed_row_by_date.csv`
- `reports/generated/gate2_train_design_v002/candidate_mapping_audit.csv`
- `reports/generated/gate2_train_design_v002/candidate_mapping_contingency.csv`
- `reports/generated/gate2_train_design_v002/candidate_mapping_summary.csv`
- `reports/generated/gate2_train_design_v002/candidate_proxy_v1_summary.csv`
- `reports/generated/gate2_train_design_v002/candidate_proxy_v1_unknown_audit.csv`
- `reports/generated/gate2_train_design_v002/user_cluster_summary.csv`
- `reports/generated/gate2_train_design_v002/cluster_bootstrap_label_rate.csv`
- `reports/generated/gate2_train_design_v002/design_manifest.json`
