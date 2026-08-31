# Tests

计划覆盖：

- 原始 Schema 和取值域合同
- 完全重复行去重
- 冲突键隔离
- 时间切分边界
- 同时间戳批次处理
- `history_time < target_time`
- 当前标签和未来反馈泄漏检查
- 封存测试集访问控制
- `--quick` / `--release` 互斥与显式选择
- quick/release 输出路径隔离和 checkpoint 资格真实性
- quick/release invariants 精确一致、浮点统计在冻结容差内一致
- release 完整 SHA-256 与重复运行哈希一致
- 实际加速后端真实性；检测到显卡但仍走 DuckDB CPU 时不得声明 GPU used

当前运行模式合同测试使用标准库 `unittest`，不依赖 pytest：

```powershell
..\.venv\Scripts\python.exe -m unittest -v tests.test_train_association_modes
```
