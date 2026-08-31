# v012 状态勘误（先读本文件，再读 README.md）

> **本目录的 `README.md` 不可修改**：它的 SHA-256 被
> `outputs/preflight/integrity_and_data_audit.json` 的 `implementation` 段钉住。
> 修改它会使 v012 自身的完整性记录失配。因此状态更正写在这里。

## README.md 的"当前状态"段落已过期

`README.md` 末尾写着：

> 当前状态：……正式读取四段标签及 GPU 训练仍需要精确合同哈希批准。

这句话记录的是**实验启动前**的时点。实验此后已完整执行完毕。

## 当前真实状态（以 outputs 为准）

| 项 | 状态 | 权威来源 |
|---|---|---|
| 实验执行 | **已完成** | `outputs/final_decision.json` |
| 科学状态 | `retraining_adds_value` | 同上 |
| 全部必需门 | `all_required_gates_passed: true`（7 个门全通过） | 同上 |
| 选中配置 | `C2_balanced`（Adam `lr=0.01`、100 steps、tether `0.001`） | 同上 |
| 冻结模型状态 SHA-256 | `5f4023f32749a44b555559d25a7793cd7b2dff856a594998f03c8bf138b63308` | `outputs/frozen_selected_model/` |
| 证据等级 | `post_audit_target_adaptation_temporal_replay_not_pristine_confirmation` | `outputs/final_decision.json` |
| 工程建议 | `deploy_NEW_BL2_with_v012_calibrator_pending_new_data_confirmation` | 同上 |
| 逐日门 | 3/3 天 H2 AP 为正（要求 ≥2 天） | 同上 |

## 前置完整性

`predecessor_integrity_snapshot.json` pin 的 **10 个输入全部存在且 SHA-256 全部匹配**
（2026-08-24 独立复验）。v012 是本包中前置链最完整的实验。

## 两处解读时必须带上的限制

1. **暖启动来自重建模型。** `contract_v012.yaml` 的 `frozen_design.source` 为
   `reconstructed_sealed_GroupedDesign`；v010 原始权重从未落盘。重建的概率复现
   最大差 5.14e-07、相关系数 1，结论不受影响。见 `INTEGRITY_ERRATA.md` 第 E-2 节。

2. **`cat_user` 在特征集内。** `required_feature_columns` 含用户 one-hot，
   目标域适配在 760 个用户上更新了包含用户维度的权重，最终回放评估的 857 个用户
   与之高度重叠。这在时点上完全合法（无未来泄漏），但意味着
   `NEW_BL2 − OLD_BL2+v011` 的增量中含有**用户级适配**成分，
   **新用户 / 冷启动用户不会获得同等收益**。相对干净的对照是
   `NEW_BL2 − NEW_BL1 = +0.028452 [0.015166, 0.041826]`（两者都被同样适配过）。
