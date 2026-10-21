# Bank Marketing — CA6003

[English](#english) | [中文](#中文)

## English

A coursework project on predicting bank term deposit subscriptions using the UCI Bank Marketing dataset. It examines data quality, feature semantics, chronological splits, preprocessing comparisons, class imbalance, and contact budgets. The main notebook contains the analysis and historical results; the supplementary notebook documents and implements the extended experiments.

### Files

| File | Purpose |
|---|---|
| [Bank_Marketing_Assignment.ipynb](Bank_Marketing_Assignment.ipynb) | Main assignment, including code, explanations, and saved results |
| [Bank_Marketing_Supplementary_Experiments.ipynb](Bank_Marketing_Supplementary_Experiments.ipynb) | Formulas, parameter tables, and self-contained execution code for 17 supplementary experiment families |
| [bank-additional-full.csv](bank-additional-full.csv) | Unmodified source data: 41,188 records, 20 input fields, and target `y` |
| [bank-additional-names.txt](bank-additional-names.txt) | Original field descriptions and references supplied by the dataset authors |
| [requirements.txt](requirements.txt) | Notebook dependencies and versions verified locally |
| [.gitignore](.gitignore) | Excludes caches, virtual environments, large models, videos, and archives |

### Running the notebook

The locally verified environment uses Python 3.11.9. Use a separate virtual environment and run the following commands from this directory:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab Bank_Marketing_Assignment.ipynb
```

Open the notebook, select the corresponding Python kernel, and choose **Restart Kernel → Run All**. Keep the CSV beside both notebooks; no additional scripts or historical model files are required.

To inspect or run the supplementary experiments:

```bash
python -m jupyterlab Bank_Marketing_Supplementary_Experiments.ipynb
```

Its default `EXPERIMENTS=set()` and `RUN_TEST=False` load development data and display method definitions and parameter tables without fitting models or reading test rows. Select experiment families in its configuration cell to train them. For example, `EXPERIMENTS={"refinement", "window", "kalman"}` selects the corresponding branches. Full development replay is substantially more work than the default run.

Balanced Random Forest additionally requires the following optional packages in the notebook kernel's environment:

```bash
python -m pip install imbalanced-learn==0.14.2 sklearn-compat==0.1.6
```

The BRF branch checks both exact package versions and dependency constraints. Missing or mismatched packages stop that branch; it does not silently substitute ordinary RF. Other branches do not require these optional packages.

### Results and reproducibility scope

- The main notebook's default run repeats the EDA, fits 8 core/contact models from the original CSV, and generates 9 main figures. Enable `SHOW_ALL_20_FIELDS` for additional field plots.
- The default execution reads only the first 35,009 training/validation rows and creates no new test predictions. Modeling excludes `duration` because it is known only after a call ends.
- Historical results from extended experiments are embedded in the notebook: 5,897 metric rows covering 1,454 prediction series across 20 stages. These tables contain saved historical results; Run All does not retrain every extended experiment.
- The supplementary notebook includes executable training, calibration, threshold, and metric routines for 17 experiment families. It uses the original CSV and declared packages, without local helper scripts or archived predictions.
- This minimal repository does not include historical fitted models, individual experiment records, or private verification files. New supplementary runs generate their own outputs under `outputs/supplementary_reproduction/`. The historical test set has already been examined, so its later evaluations are posthoc supplements rather than independent blind confirmation.
- Existing outputs and experimental values are preserved. Runs in different environments may produce small floating-point differences.

### Supplementary test results (20 October 2026)

Sections 7–10 and the extended-algorithm appendix present supplementary test results on the same 6,179 records. The notebook embeds 2,726 complete metric rows across 21 reporting groups. Models can appear in more than one group; these counts are not numbers of independent algorithms or experiments.

Each section shows up to ten candidates and folds the remaining candidates, alternative decision rules, and original train/validation results. Training ranges and development decisions are described beside the tables. The Kalman comparison uses a frozen end-of-development state throughout test, without test-label feedback or state updates. Its earlier rolling validation experiment remains available as a development supplement.

All reported tables and figures can be read in the main notebook without separate result files. Its Run All does not retrain every extended experiment; use the supplementary notebook's explicit experiment switches for that purpose. The updated main notebook has passed metric recalculation, structural checks, and selected rendering checks; a fresh full Run All of that updated notebook has not been performed.

### Supplementary implementation and verification

The supplementary notebook covers feature and macro ablations, class weighting and resampling, RF/BRF/HGB, binning, temporal weighting, rolling windows, calibration, and Kalman-style intercept updates. Long parameter tables show ten rows first and retain the remainder in expandable sections.

Recall thresholds are transferred from the preceding development window. The final validation model retains the D4 thresholds for the corresponding supplementary test rules. Running validation alone does not fabricate missing previous-window thresholds; select at least `FOLD_NAMES=["D4","val"]` to reproduce that transfer.

The review fixes were checked against 395 saved development prediction series, including 1,185 selected recall thresholds and 948 transferred thresholds. Four existing development models were reused to check automatically generated contrasts, subgroup slices, and Top-k diagnostics. The review introduced zero new base-model fits and zero test reads. A fresh-kernel default execution of the supplementary notebook passed; this is not a claim that all configurations were retrained. The notebook states the earlier numerical-check scope separately.

The repository cleanup on 21 October 2026 preserved both notebooks and the source data. Local research outputs and audit files were archived outside the repository and are not runtime dependencies.

### Data source and references

Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), by S. Moro, P. Rita, and P. Cortez. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The CSV and original data description in this repository are unmodified.

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. [doi:10.24432/C5K306](https://doi.org/10.24432/C5K306).

Related paper: Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems. [doi:10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001).

---

## 中文

银行定期存款订阅预测课程作业。以 UCI Bank Marketing 数据为例，讨论数据质量、变量语义、时间顺序划分、预处理对照、类别不平衡与联系预算。主 Notebook 保存分析与历史结果；补充 Notebook 保存扩展实验的方法说明和执行代码。

### 文件

| 文件 | 用途 |
|---|---|
| [Bank_Marketing_Assignment.ipynb](Bank_Marketing_Assignment.ipynb) | 主作业，包含代码、解释和已保存的结果 |
| [Bank_Marketing_Supplementary_Experiments.ipynb](Bank_Marketing_Supplementary_Experiments.ipynb) | 17类补充实验的公式、参数表和可独立运行的代码 |
| [bank-additional-full.csv](bank-additional-full.csv) | 未修改的原始数据：41,188 条记录，20 个输入字段及目标 `y` |
| [bank-additional-names.txt](bank-additional-names.txt) | 数据作者提供的字段说明和参考文献 |
| [requirements.txt](requirements.txt) | Notebook 所需依赖及本地验证版本 |
| [.gitignore](.gitignore) | 排除缓存、虚拟环境、大型模型、视频和归档 |

### 运行

本地验证环境为 Python 3.11.9。建议使用独立虚拟环境，在本目录运行：

```bash
python -m pip install -r requirements.txt
python -m jupyterlab Bank_Marketing_Assignment.ipynb
```

打开 Notebook，选择对应 Python 内核，然后执行 **Restart Kernel → Run All**。CSV 与两份 Notebook 放在同一目录即可，不需要额外脚本或历史模型文件。

查看或运行补充实验：

```bash
python -m jupyterlab Bank_Marketing_Supplementary_Experiments.ipynb
```

补充 Notebook 默认 `EXPERIMENTS=set()`、`RUN_TEST=False`，只加载开发数据、定义方法并显示参数表，不训练模型、不读取test。修改配置单元中的实验支线即可开始训练，例如 `EXPERIMENTS={"refinement", "window", "kalman"}`。全量开发实验比默认执行耗时明显更长。

Balanced Random Forest另需在对应内核环境安装：

```bash
python -m pip install imbalanced-learn==0.14.2 sklearn-compat==0.1.6
```

BRF支线检查两个可选包的精确版本及依赖约束。缺包或版本不符时停止，不会自动替换为普通RF；其他支线不需要这两个可选包。

### 结果与复现范围

- 主 Notebook 默认从原始 CSV 重跑 EDA、8 个核心/联系模型，并生成 9 张主要图；可通过 `SHOW_ALL_20_FIELDS` 开关生成额外字段图。
- 默认执行仅读取前 35,009 行训练/验证数据，不新增 test 预测。`duration` 在建模时排除，以免使用通话结束后才知道的信息。
- 扩展实验的历史结果已嵌入 Notebook，包括 20 个阶段、1,454 份预测序列对应的 5,897 行指标。这些表格是已保存的历史结果，Run All 不会重新训练全部扩展实验。
- 补充 Notebook 内含17类实验的训练、校准、阈值和指标计算代码，只依赖原始CSV与声明的包，不依赖本地辅助脚本或归档预测。
- 本精简仓库不包含历史已拟合模型、逐次实验记录或私人验证文件。补充实验运行时会在 `outputs/supplementary_reproduction/` 生成自己的输出。历史test已被查看，后续评价属于事后补充，不能视为新的独立盲测。
- 原有输出和实验数值予以保留；不同环境的运行数值可能存在浮点差异。

### 补充test结果（2026-10-20）

第7–10节及扩展算法附录统一展示同一段6,179条test记录的补充评价。Notebook嵌入21个展示组、2,726行完整指标；同一模型可能在不同组重复出现，这些数量不代表独立算法或独立实验数量。

各节最多显示10个候选，其余候选、其他决策规则与原train/val结果折叠保留；表旁说明实际训练范围和开发阶段决策。卡尔曼test对照固定开发结束时的状态，整个test区间不接收标签反馈、不更新状态；此前的滚动验证实验作为开发补充保留。

所有结果表和图均可直接在主Notebook内阅读，无需额外结果文件。主Notebook的Run All不会重新训练全部扩展实验；需使用补充Notebook的实验开关。更新后的主Notebook已完成指标复算、结构及部分渲染检查，尚未对该版本整本重新执行Run All。

### 补充实现与验证范围

补充Notebook覆盖特征与宏观消融、类加权与重采样、RF/BRF/HGB、分箱、时间加权、滑动窗口、概率校准和卡尔曼式截距更新。长参数表先显示10行，其余可展开。

召回阈值逐开发窗向后迁移；最终val模型保留D4阈值，供对应的补充test规则使用。单独运行val不会伪造前窗阈值；要重现该迁移，至少选择 `FOLD_NAMES=["D4","val"]`。

审核修复核对了395组历史开发预测，覆盖1,185个本窗召回阈值、948个迁移阈值；复用4个已有开发模型核对自动生成的参照对照、人群切片和Top-k诊断。该次验证新增基模型拟合为0、test读取为0。补充Notebook已通过新内核默认执行，但这不代表所有配置均已重新训练；此前数值检查的范围另在Notebook中说明。

2026年10月21日的仓库整理保留了两份Notebook和原始数据。本地研究输出及审计文件归档到仓库外，不是运行依赖。

### 数据来源与引用

数据来源：[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)，作者为 S. Moro、P. Rita 和 P. Cortez。数据按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 授权；本仓库中的 CSV 和原始数据说明未作修改。

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. [doi:10.24432/C5K306](https://doi.org/10.24432/C5K306).

相关论文：Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems. [doi:10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001).
