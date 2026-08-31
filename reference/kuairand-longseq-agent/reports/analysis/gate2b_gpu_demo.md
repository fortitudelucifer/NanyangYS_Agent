# Gate 2B GPU engineering demo

> **非正式实验结果。** 本运行仅证明 CUDA 工程路径与速度；它不是 v003 canonical release，不支持 Gold、Validation 或序列模型晋级。

## 运行证据

- 设备：`NVIDIA GeForce RTX 5070 Ti`
- PyTorch：`2.11.0+cu128`；编译 CUDA：`12.8`
- 冻结输入 SHA-256：`4981ee32c4c367bf42f27fdff81171e2bedbe5eb7fabf66d6b3e2796b9e428cd`
- Origin：`2022-04-14`；fit/calibration/assessment 行数：`400,000` / `100,000` / `100,000`
- CUDA 负责：稀疏线性模型优化、calibration/assessment raw-score 计算。
- CPU 负责：Parquet 读取、冻结切分、OneHot/缩放、前一日 sigmoid calibration 和指标。

## 模型结果（demo-only）

| 模型 | GPU 训练秒 | AP | user-GAUC | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|
| BL1 | 0.239 | 0.543252 | 0.604947 | 0.554313 | 0.187264 |
| BL2 | 0.345 | 0.583455 | 0.632166 | 0.532914 | 0.178872 |

BL2−BL1：`ΔAP=+0.040203`、`Δuser-GAUC=+0.027219`、`ΔLog Loss=-0.021398`、`ΔBrier=-0.008391`。

## 同矩阵 CPU/GPU 训练计时

- 模型：`BL2`；相同步数：`30`。
- CPU：`0.579245` 秒/步。
- GPU：`0.011506` 秒/步。
- 训练核速度比：`50.34x`。

## 解释边界

- GPU demo 使用 `torch.optim.Adam`，不复现 sklearn `SGDClassifier` 的 adaptive schedule 与 averaged-SGD 语义。
- 如果使用行数上限，行由 source identity 的固定哈希选择；没有读取标签来抽样，但仍不属于正式固定行协议。
- 任何正式 v003 结论仍必须来自 CPU canonical runner 的全绿测试、最终哈希、独立复核、所有者批准和一次受管 release。
