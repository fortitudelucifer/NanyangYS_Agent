# BL2 target-domain calibration v011

本目录与第一轮历史价值实验物理分离。第一轮 Validation、sealed、模型重建诊断和 random audit 目录保持只读，本实验只允许读取哈希固定的 v010 `predictions.parquet`。

实验顺序：

1. 校验 `predecessor_integrity_snapshot.json` 中的全部 SHA-256、文件大小和 v010 结论；
2. 04-22 至 04-27 只拟合三个后置校准候选；
3. 04-28 至 05-02 只选择校准家族；
4. 用前两段合并数据重新拟合选定家族并冻结参数；
5. 05-03 至 05-08 只评价一次；
6. 输出逐行预测、用户簇 bootstrap、reliability 数据、分析报告和 SHA-256 manifest。

禁止事项：BL1/BL2 重训、特征重算、历史更新、原始 random 重开、留出测试后返工。

当前状态：前置完整性已复核；合同、runner 和测试构建中；结果执行需要最终精确合同哈希批准。
