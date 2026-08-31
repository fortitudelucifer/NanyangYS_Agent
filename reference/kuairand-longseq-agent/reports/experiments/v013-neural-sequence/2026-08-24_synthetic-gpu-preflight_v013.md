# v013 合成 GPU 工程预检与 RTX 5060 压力测试

> 日期：2026-08-24  
> 证据等级：`engineering_synthetic_only`  
> 科学结果：否  
> Gold、正式训练、Validation/Test/Random 访问：均未授权

## 1. 问题与边界

本轮回答“当前 RTX 5060 Laptop 能否承担 v013 的实现、CUDA smoke 和 Gold pilot
准备，是否现在就需要切换桌面版 RTX 5070 Ti”。项目所有者明确授权安装 PyTorch、编写
合成 v013 测试和运行 HIER500 压力测试；授权记录为
[`synthetic_gpu_preflight_authorization_v013.json`](../../../experiments/neural_sequence_candidate_model_v013/synthetic_gpu_preflight_authorization_v013.json)。

本轮没有读取 Raw、Silver、Gold 或任何真实标签。所有输入和标签均由测试进程生成；
`contract_v013.yaml` 的 `execution_authorized: false`、Gold 禁止和正式神经训练禁止保持不变。
合成模型只复现 v013 已登记的 32/64 维表示、`256→128→64` 融合头、
`DIN10/DIN50/DIN200` 窗口和 HIER500 的 `200 + 201..500` 分层几何，不能替代尚未冻结的
Gold 特征编码器与正式模型实现。

## 2. 环境

- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8,546,484,224 bytes 可见显存；
- 驱动层：本机 `nvidia-smi` 报告 CUDA 13.1 能力；
- Python：3.11.9；
- PyTorch：`2.12.1+cu130`；
- CUDA 可用性与实际矩阵运算：通过；compute capability 为 `12.0`；
- 安装锁：[`requirements-cu130-v013.txt`](../../../experiments/neural_sequence_candidate_model_v013/requirements-cu130-v013.txt)；
- `pip check`：`No broken requirements found`。

环境与完整 `pip freeze` 见最终压力运行的
[`environment.json`](../../../experiments/neural_sequence_candidate_model_v013/outputs/gpu-preflight-20260824-042812/environment.json)
和
[`environment_lock.txt`](../../../experiments/neural_sequence_candidate_model_v013/outputs/gpu-preflight-20260824-042812/environment_lock.txt)。

## 3. 测试实现

- 合成模型几何：[`neural_sequence_v013.py`](../../../src/kuairand_longseq/models/neural_sequence_v013.py)；
- fail-closed CUDA 入口：[`gpu_preflight_v013.py`](../../../scripts/experiments/v013-neural-sequence/gpu_preflight_v013.py)；
- 聚焦测试：[`test_neural_sequence_v013.py`](../../../tests/test_neural_sequence_v013.py)。

实现检查包括：6 个候选与合同注册表一致、5 个种子未改变、全部候选输出有限、历史 padding
不能改变 DIN 预测、HIER500 分支长度正确、合成批次可复现、正式执行仍关闭，以及一次真实
CUDA HIER500 前向/反向/AdamW 更新。压力 worker 明确禁止透明 CPU fallback，并在独立进程
中测试每个 batch，避免一次高压失败污染后续档位。

## 4. 测试结果

聚焦 v013 测试：`12 passed in 8.37s`。包含既有 GPU 模块的定向回归：
`27 passed in 9.41s`。

全仓库回归为 `172 passed, 15 failed in 120.50s`。15 个失败位于既有 Gate2B/历史实验：
合同与测试期待值漂移、当前已有 approval 文件但旧测试仍断言其不存在、历史导入 manifest
缺失，以及 Gate2B runner 哈希/线程池字段不一致。它们不涉及本轮新增的 v013 模型与 CUDA
路径；本轮没有修改这些旧合同、runner、approval 或历史制品，也没有放宽断言。

## 5. HIER500 压力边界

最终边界运行使用 BF16、合成 cardinality（author 500k、music 1M、tag 100k）、
51,362,946 个参数、500 历史长度、1 次完整 AdamW 更新，并用 85% reserved 显存作为
保守门槛。完整结果见
[`stress_results.csv`](../../../experiments/neural_sequence_candidate_model_v013/outputs/gpu-preflight-20260824-042812/stress_results.csv)。

| Batch | 峰值 allocated | 峰值 reserved | reserved/可见显存 | 结论 |
|---:|---:|---:|---:|---|
| 4096 | 3.69 GB | 6.11 GB | 71.48% | 通过，仍保留至少15%门槛 |
| 6144 | 5.40 GB | 8.18 GB | 95.70% | 运行完成但余量不足，不采用 |
| 8192 | 7.10 GB | 10.81 GB | 126.49% | WDDM 虚拟保留超出物理显存，不采用 |

较小档位 32、64、128、256、512、1024、2048 也均通过。这里的吞吐量只反映合成 batch
和短运行，不能外推正式 epoch 时间。

高档单独 sweep `gpu-preflight-20260824-042722` 因不含任何安全基线，旧版判断逻辑曾把
“本轮无通过档位”错误写成“现在需要 5070 Ti”。该输出保留作失败路径证据，但其硬件推荐
已由修正后的 `gpu-preflight-20260824-042812` 取代；没有覆盖或删除旧输出。

## 6. 结论与下一门禁

当前 RTX 5060 Laptop 足以继续 v013 的代码实现、聚焦 CUDA smoke 和 Gold pilot 准备；
现在不需要使用桌面版 RTX 5070 Ti。合成 HIER500 在 batch 4096 仍通过保守显存余量门，
实际正式 micro-batch 很可能远低于这个档位。

这不是正式训练设备资格。Gold 特征 cardinality、实际 DataLoader、正式模型编码器、混合精度、
梯度累积、学习率、batch size 和 operational wall-clock 均未冻结。取得 Gold 合同与执行授权后，
必须在真实 Train-only Gold pilot 上重复同一显存/吞吐检查，再决定是否切换 5070 Ti，并在科学
训练前冻结统一配置。最终工程判断见
[`hardware_decision.json`](../../../experiments/neural_sequence_candidate_model_v013/outputs/gpu-preflight-20260824-042812/hardware_decision.json)，
其配套哈希见
[`artifact_hash_manifest.json`](../../../experiments/neural_sequence_candidate_model_v013/outputs/gpu-preflight-20260824-042812/artifact_hash_manifest.json)。

