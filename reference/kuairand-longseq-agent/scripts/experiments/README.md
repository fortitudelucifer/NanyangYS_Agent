# Versioned experiment scripts

未来新增的实验脚本必须使用 `<脚本功能或作用>_vNNN.py`，并与报告使用同一实验流目录：

| 实验流 | 目录 | 版本范围 |
|---|---|---|
| v010 之前 | `pre-v010/` | v001--v009 |
| v011 目标域校准 | `v011-target-domain-calibration/` | v011 |
| v012 目标域重训练 | `v012-target-domain-retraining/` | v012 |
| v013 神经序列 | `v013-neural-sequence/` | v013 |

示例：`run_target_domain_calibration_v011.py`、`audit_target_domain_calibration_v011.py`。运行、分析、审计和诊断脚本都归入所属实验流。已有扁平 `scripts/` 入口保持原路径，避免破坏合同、清单和历史说明的引用。

创建 v014 或更高主版本前，先新建对应的 `vNNN-<topic>/` 报告/脚本目录对，并更新索引、工作区规范和审计规则。
