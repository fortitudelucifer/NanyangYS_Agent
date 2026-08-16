# GPU Train-only 最小证据快照

这个目录仅用于当前 Demo：它保留 GPU 实验最小的可读结果和生产端运行记录，不包含 54 MB 预测 Parquet，也不触发数据清洗或训练。

## 当前结论

状态是 `numerically_verified`，但 Agent 接入状态仍是 `pending_provenance_completion`。因此 `claim_eligible=false`、`release_eligible=false`。

在 2022-04-11 至 2022-04-17 的 7 个 Train-only origins 上，共评估 1,641,098 行、943 名用户、525,942 个正例。BL2 相对 BL1 的池化变化为：

| 指标 | BL2 − BL1 | 方向 |
|---|---:|---|
| Average Precision | +0.041295 | 更好 |
| user-GAUC（事件加权） | +0.048758 | 更好 |
| Log Loss | -0.022953 | 更好 |
| Brier | -0.008907 | 更好 |

AP 的用户簇 bootstrap 95% CI 为 `[0.037661, 0.045054]`；四项指标在 7/7 个 origins 上方向一致。

## 为什么还不能发布结论

本快照缺少生产端合同正文、运行脚本正文、执行授权、独立 assessment identity 产物和结果报告。它只能证明“这些 Train-only 数值存在且内部一致”，不能证明完整来源链已满足 Agent 的准入合同。

因此禁止把本目录描述为 Validation、Test、Gold、长序列模型、正式 release 或样本外泛化结果。完整文件清单、目标文件哈希和边界见 `evidence_bundle_manifest.yaml`。
