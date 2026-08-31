# v011 状态勘误（先读本文件，再读 README.md）

> **本目录的 `README.md` 不可修改**：它的 SHA-256 被
> `outputs/preflight/integrity_verification.json` 的 `implementation` 段钉住
> （`8bd154bd…`）。修改它会使 v011 自身的完整性记录失配。
> 因此状态更正写在这里。

## README.md 的"当前状态"段落已过期

`README.md` 末尾写着：

> 当前状态：前置完整性已复核；合同、runner 和测试构建中；结果执行需要最终精确合同哈希批准。

这句话记录的是**实验启动前**的时点。实验此后已完整执行完毕。

## 当前真实状态（以 outputs 为准）

| 项 | 状态 | 权威来源 |
|---|---|---|
| 实验执行 | **已完成** | `outputs/final_decision.json` |
| 科学状态 | `pass` | 同上 |
| 选中校准族 | `M2_intercept_only`（intercept `-2.0952099481`，slope `1.0169043468`） | `outputs/final_refit/selected_calibrator.json` |
| 证据等级 | `post_audit_held_out_to_calibrator_temporal_test` | `outputs/final_decision.json` |
| 部署状态 | `engineering_calibrator_supported_pending_new_data_confirmation` | 同上 |
| 排序不变门 | 通过（AP / ROC / user-gAUC 差值精确为 0） | `outputs/final_decision.json` → `ranking_metric_differences` |
| 是否改变 v010 结论 | 否 | `changes_v010_history_ranking_conclusion: false` |

## 前置完整性：已完全复验

2026-08-24 复算结果：

| 校验对象 | 结果 |
|---|---|
| `predecessor_integrity_snapshot.json` 的 6 条 manifest pin | **6 / 6 通过** |
| `calibration_input`（v010 `random_audit/predictions.parquet`，SHA `48ae0793…`） | ✅ 通过 |
| `v010_run_manifest`、`v010_final_decision` | ✅ 2 / 2 通过 |

`all_declared_predecessor_artifacts_verified` 的声明成立。

> 曾有一段时间为 4/6：2026-08-16 Ubuntu → Windows 导入丢失了 2 份 stage 级
> `artifact_hash_manifest.json`。原件已于 2026-08-24 由 Ubuntu 源机补传并归位，
> SHA-256 与本快照 pin 的摘要一致。完整记录见包根
> [`INTEGRITY_ERRATA.md`](../../../INTEGRITY_ERRATA.md) 第 E-1 节。

## 一处容易误读的点

v011 使用的"冻结 BL2"来自**重建**的 v010 模型状态
（`reconstructed_frozen_model_state.npz`，概率复现最大差 5.14e-07、相关系数 1），
而非 v010 运行时的原始权重文件——后者从未落盘。详见 `INTEGRITY_ERRATA.md` 第 E-2 节。
