# GPU SGD 优化器线路冻结记录

冻结日期：2026-08-23

## 冻结决定

项目负责人决定暂停 SGD 优化器研究，先完成历史价值主实验。SGD 不再作为 Validation、sealed test 或 random audit 的强制预测流或科学 gate；所有已经产生的 Train-only objective 结果完整保留，待主实验完成后再研究失败原因。

## 冻结状态

- 原协议：全批 `torch.optim.SGD`，momentum=0，固定学习率，最大 3,000 步。
- 已完成：3 个 Train probe × 2 个模型 × 5 个学习率，共 30 条 GPU SGD 轨迹。
- 结果：冻结网格没有任何配置在 6/6 个 probe/model 组合上通过 objective-regret 门。
- 最优冻结配置：`lr=0.1, steps=3000`，仍为 0/6 通过。
- Validation、sealed test、random audit：均未因 SGD 线路而打开。
- v003 的 `lr=1.0 / 10,000 steps` 草案：未批准、未执行、已放弃。

## 冻结证据

- 完整 objective 数据：`reports/generated/history_value_gpu_confirmation_v002/preflight/optimization_adequacy.csv`
- 该文件 SHA-256：`79867555a31a96f5ffe1be38930de69ad491d70fea5aed56008d7997d8cbcf16`
- v002 停止记录：`reports/generated/history_value_gpu_confirmation_v002/preflight/preflight_failure.json`
- 当前分析报告：`reports/analysis/history_value_gpu_preflight_results_v001_v002.md`

## 冻结后主实验边界

主实验只运行三条对齐预测流：

- BL0：冻结常数概率参考；
- ADAM_BL1：静态特征基线；
- ADAM_BL2：相同静态特征加严格时点 H2 历史。

Adam 配置继承 v002 已完成的 6/6 Train objective 充分性证据：`lr=0.03, steps=100`。主实验继续保持 Validation → sealed test → random audit、2,000 次配对用户簇 bootstrap、AP 最小效应 0.005，以及“科学失败继续、工程无效停止”的规则。

最终结论只能描述 Adam 下的历史增量，不能声称对 SGD 或不同优化器具有稳健性。

## 主实验后拟研究的问题

SGD 线路恢复时应先在 Train-only 环境回答：

1. BL1 与 BL2 的曲率、条件数代理和分特征组梯度尺度是否不同；
2. BL1 的更大 regret 是否由稀疏类别块、参数化或全局固定步长造成；
3. full-batch、mini-batch、momentum、学习率衰减分别改变什么；
4. 如何在不读取后置预测指标的前提下冻结新的 SGD 充分性协议。

这些问题与当前“历史有没有用”的主实验分离，不能用后置数据结果倒推 SGD 配置。
