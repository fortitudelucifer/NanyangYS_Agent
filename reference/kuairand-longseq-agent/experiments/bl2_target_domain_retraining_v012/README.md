# BL2 target-domain retraining v012

本目录用于判断：相对已冻结的 `OLD_BL2 + v011 M2 intercept calibration`，在 random 目标域上成对重新训练 `NEW_BL1/NEW_BL2` 是否还能提供统计上可信的额外价值。

固定流程：

1. 复核 v010、sealed 重建模型与 v011 全部固定输入；
2. 04-22 至 04-29：从 sealed BL1/BL2 权重 warm-start，按三个冻结配置成对进行 GPU Adam 目标域适配；
3. 04-30 至 05-02：只拟合冻结斜率的目标域截距校准；
4. 05-03 至 05-05：选择唯一成对配置；
5. 冻结模型、校准器和哈希；
6. 05-06 至 05-08：只进行一次 final temporal replay；
7. 输出逐行预测、可作图 CSV/Parquet、2,000 次配对用户簇 bootstrap 和 MD 报告。

本实验不修改 v010/v011，不解冻 SGD，不把同一数据上的 temporal replay 表述为新的 pristine test。若新模型不能通过全部额外价值门，工程决策保持 `OLD_BL2 + v011`。

当前状态：目录、合同、runner、专项测试、仓库回归测试和合成 CUDA 烟雾测试均已完成；正式读取四段标签及 GPU 训练仍需要精确合同哈希批准。
