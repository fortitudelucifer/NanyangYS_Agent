# Bank Marketing — CA6003

[English](#english) | [中文](#中文)

## English

A coursework project on predicting bank term deposit subscriptions using the UCI Bank Marketing dataset. It examines data quality, feature semantics, chronological splits, preprocessing comparisons, class imbalance, and contact budgets. The notebook contains the analysis, code, figures, and historical results.

### Files

| File | Purpose |
|---|---|
| [Bank_Marketing_Assignment.ipynb](Bank_Marketing_Assignment.ipynb) | Main assignment, including code, explanations, and saved results |
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

Open the notebook, select the corresponding Python kernel, and choose **Restart Kernel → Run All**. Keep the CSV beside the notebook; no additional scripts or model files are required.

### Results and reproducibility scope

- The default run repeats the EDA, fits 8 core/contact models from the original CSV, and generates 8 main figures. Enable `SHOW_ALL_20_FIELDS` for additional field plots.
- The default execution reads only the first 35,009 training/validation rows and creates no new test predictions. Modeling excludes `duration` because it is known only after a call ends.
- Historical results from extended experiments are embedded in the notebook: 5,897 metric rows covering 1,454 prediction series across 20 stages. These tables contain saved historical results; Run All does not retrain every extended experiment.
- This minimal repository does not include historical models, individual experiment records, or the complete retraining tools. The historical test set has already been examined, so the existing results do not constitute a new independent blind evaluation.
- Existing outputs and experimental values are preserved. Runs in different environments may produce small floating-point differences.

### Data source and references

Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), by S. Moro, P. Rita, and P. Cortez. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The CSV and original data description in this repository are unmodified.

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. [doi:10.24432/C5K306](https://doi.org/10.24432/C5K306).

Related paper: Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems. [doi:10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001).

---

## 中文

银行定期存款订阅预测课程作业。以 UCI Bank Marketing 数据为例，讨论数据质量、变量语义、时间顺序划分、预处理对照、类别不平衡与联系预算。完整分析、代码、图表和历史结果保存在 Notebook 中。

### 文件

| 文件 | 用途 |
|---|---|
| [Bank_Marketing_Assignment.ipynb](Bank_Marketing_Assignment.ipynb) | 主作业，包含代码、解释和已保存的结果 |
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

打开 Notebook，选择对应 Python 内核，然后执行 **Restart Kernel → Run All**。CSV 与 Notebook 放在同一目录即可，不需要额外脚本或模型文件。

### 结果与复现范围

- 默认从原始 CSV 重跑 EDA、8 个核心/联系模型，并生成 8 张主要图；可通过 `SHOW_ALL_20_FIELDS` 开关生成额外字段图。
- 默认执行仅读取前 35,009 行训练/验证数据，不新增 test 预测。`duration` 在建模时排除，以免使用通话结束后才知道的信息。
- 扩展实验的历史结果已嵌入 Notebook，包括 20 个阶段、1,454 份预测序列对应的 5,897 行指标。这些表格是已保存的历史结果，Run All 不会重新训练全部扩展实验。
- 本精简仓库不包含历史模型、逐次实验记录或完整重训工具。历史 test 已经被查看，已有结果不构成新的独立盲测。
- 原有输出和实验数值予以保留；不同环境的运行数值可能存在浮点差异。

### 数据来源与引用

数据来源：[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)，作者为 S. Moro、P. Rita 和 P. Cortez。数据按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 授权；本仓库中的 CSV 和原始数据说明未作修改。

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. [doi:10.24432/C5K306](https://doi.org/10.24432/C5K306).

相关论文：Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems. [doi:10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001).
