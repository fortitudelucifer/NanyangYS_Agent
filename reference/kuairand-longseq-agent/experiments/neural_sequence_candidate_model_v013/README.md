# Neural sequence candidate model v013

## 状态

`design_registered_fail_closed`。本目录登记下一阶段神经网络与序列长度消融，但不授权 Gold 构建、模型训练、Validation/restricted/random 新标签读取或结果发布。

当前已冻结的设计决定：

- 完整模型阶梯：常数、同 Gold 线性静态、同 Gold 线性 H2、MLP 静态、MLP H2、DIN10、DIN50、DIN200、分层500；
- `DIN10`、`DIN50`、`DIN200` 使用同一编码器和训练协议，仅改变严格历史窗口；
- 每个可发布神经候选必须完整运行5个固定种子：`20260824`、`20260825`、`20260826`、`20260827`、`20260828`；
- 主评估使用五种子 logit ensemble，单种子波动与用户簇 bootstrap 分开报告；
- 所有嵌套比较使用相同目标行、相同标签、相同切分和相同指标实现。

## 文件

- [contract_v013.yaml](contract_v013.yaml)：机器可读的 fail-closed 合同；
- [approval_v013.json](approval_v013.json)：仅授权文档与合同设计，明确拒绝执行；
- [predecessor_integrity_requirements_v013.json](predecessor_integrity_requirements_v013.json)：当前 Silver/v010/v011/v012 证据锚点及 Gold 缺口；
- [design_manifest_v013.json](design_manifest_v013.json)：本轮设计、合同与边界文件的 SHA-256 快照；
- [设计报告](../../reports/experiments/v013-neural-sequence/2026-08-24_neural-sequence-model-design_v013.md)：架构、完整对照、门禁和未来产物。

## 下一授权点

执行前必须另外完成并批准：Gold 数据合同与小规模时间线复算、特征 allowlist/denylist、训练超参数和 operational compute budget、精确合同 SHA-256、实现与专项测试、五种子 CUDA smoke/preflight。缺少任一项时必须停止，不得创建正式预测或 `final_decision.json`。
