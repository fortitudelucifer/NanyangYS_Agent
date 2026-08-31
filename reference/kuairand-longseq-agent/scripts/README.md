# Scripts

已有扁平脚本是受合同、README 或哈希引用的历史执行面，保持其原路径。未来新增的实验脚本放入 [experiments/](experiments/) 中与报告相同的实验流目录，并使用：

```text
<脚本功能或作用>_vNNN.py
```

例如 `run_target_domain_calibration_v011.py`、`audit_target_domain_calibration_v011.py`。当前分类为：

| 实验流 | 脚本目录 | 仅接受的版本 |
|---|---|---|
| v010 之前 | `experiments/pre-v010/` | v001--v009 |
| v011 目标域校准 | `experiments/v011-target-domain-calibration/` | v011 |
| v012 目标域重训练 | `experiments/v012-target-domain-retraining/` | v012 |
| v013 神经序列 | `experiments/v013-neural-sequence/` | v013 |

同一实验流的 `run_`、`analyze_`、`audit_` 和 `diagnose_` 脚本都放在该流目录中；新主版本先创建对应的 `vNNN-<topic>/` 目录并更新索引与审计规则。

历史编号化、可重复执行入口的例子：

```text
00_audit_raw.py
01_build_silver.py
02_build_gold_sequences.py
03_train_baseline.py
04_evaluate_validation.py
05_run_sealed_test.py
```

当前已有入口：

- `build_silver.py`：正式 Silver 构建入口；本轮 Train-only 建模没有重新执行。
- `analyze_train_associations_v002.py`：只读的 canonical Train 描述性关联分析；使用显式四文件白名单，不构建 Gold、不训练模型，也不读取 Validation/late/random/statistic 数据。
- `analyze_gate2_train_design_v002.py`：release-only 的 Gate 2 设计证据入口；复算 burn-in 覆盖、source-identity 固定行 digest、diagnostic modality proxy v1 和共享 user-cluster bootstrap 计划，不构建 Gold 或训练模型。

## Train association 的运行档位

必须显式选择一个档位；不带档位参数会直接退出，避免把快速结果误当正式 release。

| 档位 | 用途 | 线程与验证 | 输出 |
|---|---|---|---|
| `--quick` | 日常研究迭代 | 默认 8 个 CPU 线程；核对正式 manifest 成员和文件大小，不重复计算四个大输入的 SHA-256 | 独立写入 `reports/generated/train_association_v002_quick/`；永远不是 checkpoint 证据 |
| `--release` | checkpoint 与可引用正式结果 | 固定 1 个 CPU 线程；完整输入 SHA-256；生成正式 PNG、报告和 canonical manifest | 写入 `reports/generated/train_association_v002/` 与正式分析报告 |

```powershell
# 通常约数秒；不覆盖 release
..\.venv\Scripts\python.exe .\scripts\analyze_train_associations_v002.py --quick

# quick 如需临时图表
..\.venv\Scripts\python.exe .\scripts\analyze_train_associations_v002.py --quick --with-chart

# 只在 checkpoint 时运行
..\.venv\Scripts\python.exe .\scripts\analyze_train_associations_v002.py --release
```

当前两种档位都使用 DuckDB CPU。检测到 NVIDIA 显卡不等于已使用 GPU；manifest 必须以 `accelerator_used=false`、`gpu_used=false` 如实记录，直到出现经过对照验证且禁止透明 CPU fallback 的独立 GPU 后端。

## Gate 2 设计证据

该入口只提供正式 release；完整验证四个白名单输入的 size 与 SHA-256，输出到独立目录：

```powershell
..\.venv\Scripts\python.exe .\scripts\analyze_gate2_train_design_v002.py --release
```

产物是 Train-only 设计证据，不代表 Gate 2、Gold 或模型训练已经获批。
