# Versioned experiment reports

未来新增的人工实验报告必须使用 `YYYY-MM-DD_<实验内容>_vNNN.md`，并按实验流放置：

| 实验流 | 目录 | 版本范围 |
|---|---|---|
| v010 之前 | `pre-v010/` | v001--v009 |
| v011 目标域校准 | `v011-target-domain-calibration/` | v011 |
| v012 目标域重训练 | `v012-target-domain-retraining/` | v012 |
| v013 神经序列 | `v013-neural-sequence/` | v013 |

示例：`2026-08-24_target-domain-calibration_v011.md`。已有 `reports/analysis/` 及根目录报告保持原路径；它们是历史或已发布证据，不因本次分类而改名。

创建 v014 或更高主版本前，先新建 `vNNN-<topic>/`，随后更新 `PROJECT_INDEX.md`、根目录 `WORKSPACE_CONVENTIONS.md` 和 `scripts/audit_workspace_layout.py`。
