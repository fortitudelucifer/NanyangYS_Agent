# 历史特征价值 Adam Validation v006：预测身份列加载停止记录

记录时间：2026-08-23

## 结论先行

v006 已完成正确的 `886,452` 行 Validation 数据完整性检查，并在 RTX 5070 Ti 上完成 ADAM_BL1 与 ADAM_BL2 拟合；两者都通过 objective adequacy，随后才进入评估函数。执行在写出预测和计算指标之前因内存 Frame 漏载 `time_ms` 而停止，因此仍不能判断历史是否有用。

## 已完成的有效证据

- canonical 特征共 3,286,296 行，Validation 目标 886,452 行。
- 来源身份 3,286,296 个，零重复。
- PIT、历史窗口单调性、正例计数和视频连接检查均无违规。
- ADAM_BL1 与 ADAM_BL2 均实际在 GPU 上完成 100 步，并通过相对 CPU 参考目标的充分性门。
- 已验证特征文件 SHA-256：`8f900c5432feecc60d04f2ffa9cd1a6942678d8de60db8832b5046f1b152eade`。

## 实现错误

特征 Parquet 本身包含 `time_ms`。共享 `read_frame()` 只加载 canonical 模型列和三个人群标记，而 canonical 模型列不包含 `time_ms`；后续逐行预测身份清单又强制要求 `time_ms`，因此触发 `KeyError`。

这是列加载错误，不是数据、优化器或科学门失败。目标分数曾在进程内存中生成，但没有落盘、没有冻结哈希，也没有计算 bootstrap 或指标。

## 权限边界与修复

sealed test 与 random audit 均未访问。后继合同只能把 `time_ms` 加入内存 Frame 的加载列，并复用上述经过完整性验证的特征文件；模型、超参数、目标行、统计规则和 Validation-only 权限不得改变。
