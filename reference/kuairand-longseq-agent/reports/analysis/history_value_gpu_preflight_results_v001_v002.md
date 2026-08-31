# 历史特征价值 GPU 实验：v001/v002 当前预检结果

生成日期：2026-08-23

## 结论先行

当前尚不能回答“历史有没有用”。v001 和 v002 都在 Train-only 优化器充分性预检阶段按合同停止，Validation、sealed test、random audit 均未打开，因此没有任何正式 BL2-vs-BL1 泛化结果。

已经得到的确定结论是：

1. CUDA 管线真实运行，RTX 5070 Ti 并未回退到 CPU。
2. GPU Adam（lr=0.03）在 100 步时已对 3 个 Train probe × BL1/BL2 共 6 个组合全部通过 objective-regret 门。
3. 原冻结的普通全批 GPU SGD 网格（lr 0.001–0.1，最多 3,000 步，momentum=0）没有任何配置对 6 个组合全部通过，因此不能作为有效优化器对照进入 Validation。
4. 这属于求解器健康失败，不是“历史无用”的科学反证，也不能把先前 Train 上的 Adam 探索结果升级为确认结论。

## 数据与实验边界

预检只使用冻结的 Train 严格时点特征矩阵：2022-04-08 至 2022-04-17 的标准曝光 `tab=1` 目标，历史来自严格早于目标时间的全 tab 标准事件。三个 objective-only probe origin 为 2022-04-11、04-14、04-17。

每个 probe 同时构建静态 BL1 和静态+H2 历史 BL2。CPU-LBFGS 只作为同一正则化逻辑回归目标的参考最优解，不参与预测；GPU Adam/SGD 的训练目标为：

`mean binary log-loss(raw score) + alpha/2 × ||w||²`，其中 `alpha=1e-4`，截距不正则化。

充分性要求：`GPU objective - reference objective <= max(1e-4, 0.005 × |reference objective|)`，且参考解正常收敛。

## v001 停止结果

v001 的首个 BL1 参考解在冻结的 500 次上限达到 `n_iter=500` 并产生 1 个 convergence warning，故在任何 GPU 正式网格和 Validation 访问前停止。

Train-only 诊断将相同 LBFGS 上限提高到 2,000 后，6/6 个参考解均在 644–952 次内无警告收敛。首个 BL1 的 500 次与收敛解目标只相差约 `6.17e-10`，说明 v001 是迭代上限过紧，不是数据或目标异常。

## v002 GPU 运行证据

v002 完成 36 条 GPU 轨迹：6 条 Adam、30 条 SGD。

| 项目 | Adam | SGD |
|---|---:|---:|
| GPU 轨迹数 | 6 | 30 |
| GPU 训练秒数合计 | 166.925 | 2,497.264 |
| 单轨迹中位秒数 | 23.376 | 69.919 |
| 单轨迹最长秒数 | 59.391 | 177.965 |
| 进程峰值 CUDA 显存 | 4.684 GiB | 4.684 GiB |

总计记录的 GPU 训练时间为 2,664.189 秒（44.40 GPU 分钟）。`nvitop` 先前长期不升，是因为 v001 停在 CPU 参考解，随后 6 个参考诊断也在 CPU；v002 进入 Adam/SGD 轨迹后才出现持续 GPU 负载。CPU 参考解与 GPU 轨迹交替执行，因此利用率不是全程恒高。

## Adam objective 充分性结果

| lr | steps | 通过/要求 | 最大 regret | regret 门槛下界 | 结果 |
|---:|---:|---:|---:|---:|---|
| 0.03 | 30 | 0/6 | 0.006750 | 0.002606 | 失败 |
| 0.03 | 100 | 6/6 | 0.000540 | 0.002606 | 通过 |
| 0.03 | 300 | 6/6 | 0.000218 | 0.002606 | 通过 |
| 0.03 | 1000 | 6/6 | 0.000040 | 0.002606 | 通过 |

按“选择最小充分 checkpoint”规则，Adam 应冻结为 `lr=0.03, steps=100`。

## SGD objective 充分性结果

下表为每个学习率的最大冻结 checkpoint（3,000 步）。

| lr | steps | 通过/要求 | 最大 regret | regret 门槛下界 | 结果 |
|---:|---:|---:|---:|---:|---|
| 0.001 | 3000 | 0/6 | 0.068308 | 0.002606 | 失败 |
| 0.003 | 3000 | 0/6 | 0.061822 | 0.002606 | 失败 |
| 0.01 | 3000 | 0/6 | 0.054133 | 0.002606 | 失败 |
| 0.03 | 3000 | 0/6 | 0.046245 | 0.002606 | 失败 |
| 0.1 | 3000 | 0/6 | 0.034773 | 0.002606 | 失败 |

最好的冻结配置是 `lr=0.1, steps=3000`，但仍明显不足：

| probe | 模型 | GPU objective | reference objective | regret | allowed | 
|---|---|---:|---:|---:|---:|
| 2022-04-11 | BL1 | 0.581199 | 0.546426 | 0.034773 | 0.002732 |
| 2022-04-11 | BL2 | 0.532177 | 0.526305 | 0.005872 | 0.002632 |
| 2022-04-14 | BL1 | 0.577298 | 0.543548 | 0.033750 | 0.002718 |
| 2022-04-14 | BL2 | 0.525729 | 0.521117 | 0.004612 | 0.002606 |
| 2022-04-17 | BL1 | 0.580203 | 0.547231 | 0.032972 | 0.002736 |
| 2022-04-17 | BL2 | 0.527400 | 0.523330 | 0.004069 | 0.002617 |

BL1 的不足远大于 BL2，因此如果直接让这个 SGD 进入 Validation，会把“静态 vs 历史”的差异与“两个模型被优化到不同程度”混在一起，无法作为可信对照。

## 关于 SGD lr=1.0 是否异常

`lr=1.0` 相对常见 mini-batch SGD 经验确实偏大，但这里是对均值损失做确定性的全批梯度下降，特征经过标准化/裁剪，学习率数值不能直接与常见深度学习训练比较。因此它不必然在数学上无效。

但当前证据不足以把它直接写入正式实验：仅有一个 Train probe 的诊断显示 `lr=1.0, steps=3000` 对 BL2 已通过，而 BL1 regret 仍为 0.004381，高于 0.002732 门槛；`lr=3.0` 已出现明显不稳定。把学习率从 0.1 扩到 1.0、把步数扩到 10,000 属于失败后的优化器协议修订，必须重新冻结并在全部 Train probes 上验证，不能直接进入 Validation。

因此，对“这么大的学习率不正常”的审慎回答是：它在本目标上有稳定迹象，但目前没有足够证据证明它是合适、充分且稳健的正式 SGD 配置，暂不应继续执行 v003。

## 当前能够和不能够声称的结果

能够声称：

- GPU 环境和稀疏训练路径有效。
- Adam 可以充分优化 BL1 与 BL2 的共同目标。
- 原普通 SGD 网格不足，Adam-vs-SGD 科学对照尚未成立。
- 两次停止均发生在 Train-only 预检，保护了后续数据访问顺序。

不能声称：

- 历史有用或历史无用。
- Validation、sealed 或 random 上存在任何 BL2-BL1 增益。
- Adam 与 SGD 得到了相同方向或相同量级的历史效应。

作为背景而非当前确认结果，先前 Train-only Adam 探索观察到 BL2-BL1 AP `+0.041295`，用户簇 95% CI `[0.037661, 0.045054]`；它只用于提出假设，不能替代尚未运行的 Validation。

## 下一步决策建议

今天需要 demo 证据时，最稳妥的是把科学确认改成 Adam-only：使用已充分的 `lr=0.03, steps=100`，仍完整运行静态 BL1、历史 BL2、Validation、sealed、random 和 2,000 次用户簇 bootstrap；同时明确声明“SGD 鲁棒性对照因求解不足未完成”。这样最快回答核心问题，但放弃“两种优化器均支持”的强结论。

如果必须保留 Adam+SGD 双优化器强结论，则应暂停后置数据访问，重新设计有数值依据的 SGD 协议，例如先在 Train 上估计稳定步长或使用预注册衰减/动量方案，再冻结 successor。该路径更慢，而且对照的含义会从“普通 SGD”改变为“经过调度或动量的 SGD”。

## 可复核与作图文件

- 完整逐 probe/模型/配置数据：`reports/generated/history_value_gpu_confirmation_v002/preflight/optimization_adequacy.csv`
- 汇总作图表：`reports/generated/history_value_gpu_confirmation_v002/preflight/configuration_summary.csv`
- v002 停止记录：`reports/generated/history_value_gpu_confirmation_v002/preflight/preflight_failure.json`
- v001 停止记录：`reports/generated/history_value_gpu_confirmation_v001/preflight/preflight_failure.json`

所有表均为 CSV/JSON，可直接绘制 checkpoint-vs-regret、learning-rate-vs-regret、通过率和 GPU 用时图。
