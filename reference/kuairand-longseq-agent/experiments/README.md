# Versioned Experiments

每个新实验用一个 `experiments/<experiment-id>_vNNN/` 目录，至少包含 README、冻结合同、批准记录（如需要）和前序完整性快照。生成的预测、指标与模型状态写到对应的 `reports/generated/<experiment-id>_vNNN/`，人类可读结论写到 `reports/analysis/`。

已有实验目录按其历史结构保留；新实验必须遵循根目录 `WORKSPACE_CONVENTIONS.md` 与 `PROJECT_INDEX.md` 的生命周期规则。

当前新实验入口：`neural_sequence_candidate_model_v013/`。其合同为 fail-closed 设计合同；在精确合同哈希获批、Gold point-in-time 门禁通过和可执行预算冻结前，不得据此构建 Gold、训练神经模型或读取新的受限标签。
