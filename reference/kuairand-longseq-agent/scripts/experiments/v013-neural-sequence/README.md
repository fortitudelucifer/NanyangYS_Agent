# v013 neural-sequence scripts

本目录已为 v013 神经序列实验流登记，但当前没有 runner。设计阶段不以空脚本或占位预测冒充可执行实验。

后续脚本只能在 Gold 合同、v013 精确合同哈希与执行批准齐备后添加，并使用以下命名角色：

- `build_gold_sequence_v013.py`：确定性 point-in-time Gold 试制/构建；
- `train_neural_sequence_v013.py`：五种子神经候选训练；
- `evaluate_neural_sequence_v013.py`：同目标行指标、逐日/切片和配对 bootstrap；
- `audit_neural_sequence_v013.py`：泄漏、输入身份、种子、预算和输出哈希审计。

任何实现都必须先提供 `--validate-only` 或等价 fail-closed preflight，并在未授权时停在读取受控标签或 GPU 训练之前。
