# KuaiRand-1K 长序列受治理研究：从 Raw 数据到 v012 工程候选

> **JKRec 包内位置：**根目录总览文件；文中证据链接已按该位置重定向。

> **汇报日期：**2026-08-24（Asia/Singapore）  
> **本次补充：**Raw 逐 CSV 分布、H1–H4 假设与指标/CI 解释、Silver 链接化对账、特征来源与行数、CPU-LBFGS/regret、v011/v012 过程及 v010 §9.3 基线修正。  
> **研究问题：**`candidate_long_view_prediction`  
> **交付性质：**可读汇报与证据导航；不产生新实验结论。  
> **当前工程候选：**`NEW_BL2 + v012 calibrator`，但仅为 *pending new-data confirmation* 的工程候选。
>
> **修订记录（2026-08-24，分发前独立评审后）：**新增 §7.4a（v010 冻结模型为重建权重）、
> §9.1.1（v011/v012 切分边界对齐设计）、§9.5.1（最终回放逐日 ΔAP 趋势）；
> §4.2 补入多重比较声明；§9.2 修正 random 行的概率门表述；§12.1 补入 random 域数据耗尽事实；
> 证据索引新增 E14、E15。所有数值未变，全部改动为补充说明与措辞更正。

<a id="toc"></a>

## 阅读导航

1. [先读结论与证据边界](#executive-summary)
2. [项目问题、数据集背景与研究范围](#scope)
3. [Raw 数据规模、分布、字段与标签](#raw-data)
4. [粗筛：EDA 如何收敛为可检验假设](#screening)
5. [Silver：清洗、治理与可复核数据层](#silver)
6. [Gold：设计已明确，但正式数据层尚未物化](#gold)
7. [模型、特征与公式：BL0、BL1、BL2](#models)
8. [预实验与优化器比较：为什么保留 Adam、冻结 SGD](#optimizer)
9. [三轮正式实验：v010、v011、v012](#three-rounds)
10. [最终对比、结论与可说/不可说](#conclusions)
11. [对 NanyangYS Agent 的可执行指导](#agent-guidance)
12. [下一步与完整证据索引](#evidence-index)

<a id="executive-summary"></a>

## 1. 结论先行：研究回答了什么

本研究不是“复现一个可直接上线的视频推荐系统”，而是回答一个受控离线问题：在候选视频曝光的时刻，只用当时已可见的用户历史、用户静态信息与视频基础元数据，能否更好地预测该候选事件的官方 `long_view` 标签。

最终证据支持以下三级结论：

| 层级 | 已验证的事实 | 不可越过的边界 |
|---|---|---|
| **v010：历史价值** | 冻结 GPU Adam 协议下，加入 H2 严格用户总体历史的 BL2 相比相同静态特征的 BL1，在 Validation、后置 standard 与 random exposure audit 的 AP 增量均显著为正。 | 是离线排序增量；不代表线上因果收益，也不代表所有优化器均有效。 |
| **v011：概率校准** | 目标域截距校准显著降低 Log Loss/Brier，并严格保持候选排序不变。 | 是 post-audit 的 held-out-to-calibrator temporal test，不是全新、未触碰的确认试验。 |
| **v012：受约束适配** | 成对目标域重训练在最终 temporal replay 上同时提升排序和概率损失，优于 `OLD_BL2 + v011`。 | 是 post-audit replay；上线或科学发布前仍需新的独立数据确认与人工审批。 |

最精炼的最终表述是：**在本任务、冻结特征、严格时点规则与 GPU Adam 协议下，H2 用户总体历史有可复现的离线排序价值；目标域校准和受约束适配可带来工程价值，但尚不构成线上或因果证明。**

> **证据优先级。**本文以合同、审批、输入/输出哈希、`artifact_hash_manifest.json` 与 `final_decision.json` 为事实锚点；较早的交接文档和 README 用于还原过程。部分旧文档仍记载“Gold/Validation 未开放”的历史状态，不能覆盖 2026-08-24 已导入并复核的 v007/v008/v010 GPU 证据。[项目状态对账](kuairand-longseq-agent/reports/post_gpu_import_reconciliation_2026-08-24.md)

[返回导航](#toc)

<a id="scope"></a>

## 2. 项目问题、数据集背景与研究范围

### 2.1 为什么选 KuaiRand-1K

KuaiRand 是来自快手移动视频应用推荐日志的序列推荐数据集。KuaiRand-1K 从更大版本中抽取 1,000 名用户，保留其相关视频；它同时提供标准推荐日志与插入随机曝光的视频，因此可观察到不同曝光域下的行为差异。[数据集 README](KuaiRand-1K/README.md)

本项目选择 KuaiRand-1K，而非更小的 Pure 版本，原因是问题依赖完整、严格按时间排序的用户行为史。随机曝光并不把问题自动变成“任意策略的无偏离线评估”：它只提供一个独立的曝光分布审计域，不能覆盖全平台、任意候选池或任意新策略。

### 2.2 研究单位、输入与输出

对用户 \(u\) 在目标时刻 \(t_i\) 的候选视频事件 \(i\)，研究的输出是：

$$
\hat p_i \approx \Pr(y_i=1\mid \text{prediction-time information}),
\qquad y_i=\operatorname{long\_view}_i\in\{0,1\}.
$$

| 项目 | 本研究的定义 |
|---|---|
| 样本单位 | 一条已观测的候选视频曝光事件；不是全库召回结果。 |
| 主要场景 | standard 目标固定为 `tab=1`；历史可来自全部 `tab`，但必须严格早于目标时间。 |
| 输入 | 严格历史、用户/视频的允许静态字段、视频基础元数据及质量标记。 |
| 输出 | `long_view=1` 的排序分数或概率。 |
| 核心比较 | 同一目标行、同一静态特征、同一训练协议下 `BL2 − BL1` 的增量。 |
| 不做的事 | 不重洗 Raw/Silver；不从当期统计表取特征；不将模型结论外推为线上因果或业务收益。 |

### 2.3 数据使用与访问顺序

| 时间段 | 数据域 | 在研究中的用途 | 关键限制 |
|---|---|---|---|
| 2022-04-08 至 04-17 | early standard | Train、EDA、优化器充分性和模型拟合。 | 仅在 Train 内选模型/超参数。 |
| 2022-04-18 至 04-21 | standard | Validation。 | 先完成、哈希复核后才能解锁下一阶段。 |
| 2022-04-22 至 05-08 | late standard | sealed standard 时间外测试。 | 配置冻结后一次性主评估。 |
| 2022-04-22 至 05-08 | random | random exposure audit。 | 在主结论冻结后才打开；标签不得回流调参。 |

这条访问顺序将“提出假设”“确认历史增量”“检查曝光迁移”“事后工程适配”分开，避免用后置结果反向选择特征或阈值。

[返回导航](#toc)

<a id="raw-data"></a>

## 3. Raw 数据：规模、分布、字段与标签

### 3.1 本地 Raw 数据锚点

Raw CSV 永久只读。下表是本项目实际使用的本地文件行数；后续清洗和建模均以这些文件及其 manifest 为锚点，而不是以外部简介中的近似统计替代本地核验。[数据合同](kuairand-longseq-agent/configs/data_contract.yaml)

| 表 | 日期/性质 | Raw 行数 | 研究角色 |
|---|---|---:|---|
| `log_standard_4_08_to_4_21_1k.csv` | 早期 standard | 5,055,984 | Train 与 Validation 的历史来源。 |
| `log_standard_4_22_to_5_08_1k.csv` | 后期 standard | 6,657,061 | 时间外 sealed standard。 |
| `log_random_4_22_to_5_08_1k.csv` | 随机曝光 | 43,028 | 独立 random audit。 |
| `user_features_1k.csv` | 用户静态特征 | 1,000 | 用户维表。 |
| `video_features_basic_1k.csv` | 视频基础元数据 | 4,371,868 | 视频静态特征。 |
| `video_features_statistic_1k.csv` | 当月聚合统计 | 4,371,868 | 默认拒绝进入同月预测特征，防止事后泄漏。 |

标准曝光共有 \(5{,}055{,}984+6{,}657{,}061=11{,}713{,}045\) 条事件；random 另有 43,028 条。Raw 文件总解压规模约 4.3 GB，适合用 DuckDB/Polars 做列式、惰性处理，而不适合将全量表直接载入普通 pandas DataFrame。

#### 3.1.1 六个 CSV 分别记录了什么、分布大致怎样

三张日志 CSV 的 schema 完全相同，均有 19 列；区别在日期、曝光机制与 `is_rand`。下表的标签率来自正式清洗报告中的**同源 Raw 分布**，不是模型训练或测试结果。

| CSV | 19 列 / 时间范围 | 记录量与可见分布 | 数据质量轮廓 | 在本研究中的正确位置 |
|---|---|---|---|---|
| `log_standard_4_08_to_4_21_1k.csv` | 2022-04-08–04-21；`is_rand=0` | 5,055,984 行；Raw `long_view` 率 26.3463%。 | 34,562 个完全重复副本（约 0.68%）、5,455 个冲突业务键、23,538 个公式一致性疑点；409,259 行日志时长缺失/非正。 | 早期 standard；提供 Train 与 Validation 之前的严格历史。 |
| `log_standard_4_22_to_5_08_1k.csv` | 2022-04-22–05-08；`is_rand=0` | 6,657,061 行；Raw `long_view` 率 26.0986%。 | 59,662 个重复副本（约 0.90%）、8,514 个冲突业务键、32,392 个公式一致性疑点；525,657 行日志时长缺失/非正。 | late standard；只在配置冻结后作为 sealed 时间外评估。 |
| `log_random_4_22_to_5_08_1k.csv` | 2022-04-22–05-08；`is_rand=1` | 43,028 行；Raw `long_view` 率 8.4155%，明显低于 standard。 | 只有 1 个完全重复副本、0 个冲突业务键、45 个公式一致性疑点；1,310 行日志时长缺失/非正。 | random exposure audit；只检查冻结结论能否跨曝光域迁移，不能参与选参。 |

| CSV | 列数与主体字段 | 规模/缺失分布 | 可用性结论 |
|---|---|---|---|
| `user_features_1k.csv` | 31 列：`user_id`、活跃度、直播/作者标记、关注/粉丝/好友数及区间、注册天数及区间、18 个加密 `onehot_feat`。 | 1,000 行；其中 `is_live_streamer=-124` 有 782 个用户，必须解释为未知而不是“不直播”。 | 本轮冻结 BL1 并未把这 30 个用户画像字段逐一作为特征组；它使用事件中的 `user_id` 形成 `cat_user`。用户表仍是后续扩展与审计的数据锚点。 |
| `video_features_basic_1k.csv` | 12 列：`video_id`、作者、视频/上传类型、上传日、可见性、时长、宽高、音乐、标签。 | 4,371,868 行；`tag` 缺失 137,484（约 3.14%），上传日期缺失 58，时长缺失/非正 573,057（约 13.11%）。 | 是当前静态视频特征的唯一维表来源；缺失以状态/标记建模，不删行。 |
| `video_features_statistic_1k.csv` | 52 列：展示、播放、完播、互动、转发等月内平均统计。 | 4,371,868 行；按月内每天、场景聚合，Raw 文件约 3.1 GB。 | **默认禁止**加入同月预测：它会把目标期之后的信息带回当前事件。只可做独立质量画像。 |

日志中约 4.95M（early）和 6.51M（late）行共享同一用户、同一 `time_ms` 而对应不同视频；它们是时间戳批次，不能被去重成一条事件。random 日志没有这种多视频批次模式，不能据此倒推 standard 中批次内视频的真实展示顺序。

### 3.2 事件、用户与视频字段的简要分层

| 数据层 | 代表字段 | 对问题的意义 | 使用限制 |
|---|---|---|---|
| 事件键与时间 | `user_id`, `video_id`, `date`, `time_ms`, `tab`, `is_rand` | 定义候选事件、场景、数据域与时点。 | 同一 `(user_id,time_ms)` 是一个候选批次，批次内顺序未知。 |
| 当前反馈 | `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `long_view`, `play_time_ms` | 记录事件后结果。 | 预测当前事件时全部禁止作为输入。 |
| 用户维表 | 活跃度、社交规模、注册天数、加密 one-hot 字段等 | 可用于静态用户描述或诊断。 | 时点未知的字段不得静默进入 point-in-time allowlist。 |
| 视频基础维表 | 作者、音乐、上传日期/类型、时长、宽高、标签等 | 支持静态基线。 | 缺失必须作为状态处理；标签数字没有公开语义字典。 |
| 视频统计表 | `show_cnt` 等月内平均统计 | 对同月预测有明显后验泄漏风险。 | 在本项目的 Silver/主模型中拒绝物化或 join。 |

#### 3.2.1 每类字段的语义、分布形态与权限

| CSV / 字段块 | 具体字段（完整组别） | 数据分布/语义 | 模型权限 |
|---|---|---|---|
| 三张事件日志（各 19 列） | 键与时间：`user_id`, `video_id`, `date`, `hourmin`, `time_ms`, `tab`, `is_rand`。 | `time_ms` 是毫秒交互时间；`tab\in[0,14]` 是场景 ID，不是内容类目。`(user_id,time_ms)` 可能有多视频且批内顺序未知。 | 可定义目标、split、严格历史与 `cat_user`/`cat_video`；不能把批内反馈当历史。 |
| 三张事件日志（各 19 列） | 反馈：`is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `long_view`, `play_time_ms`, `profile_stay_time`, `comment_stay_time`, `is_profile_enter`。 | 多个二元行为与时长/停留反馈；`is_click` 的产品语义随 UI/`tab` 改变。 | 只有官方 `long_view` 是主标签；所有当前事件反馈都在当前预测特征 denylist。 |
| `user_features_1k.csv`（31 列） | 画像：活跃度、社交数值/区间、注册天数/区间；身份标记；`onehot_feat0`–`onehot_feat17`。 | 既有类别、计数、区间和加密类别。字段采集时点并未对所有列逐项证明。 | 可用于未来、经可用性合同批准的静态用户特征；当前 BL1 不以这些画像列替代 `cat_user`。 |
| `video_features_basic_1k.csv`（12 列） | 标识/类别：`video_id`,`author_id`,`video_type`,`upload_type`,`music_id`,`music_type`,`tag`；数值/日期：`upload_dt`,`video_duration`,`server_width`,`server_height`；状态：`visible_status`。 | 高基数 ID/标签组合、长尾时长、缺失标签/日期、图片类或未知时长。 | 当前 BL1 的作者、音乐、视频/上传/音乐类型、标签、时长、上传年龄和画面几何均由此派生。 |
| `video_features_statistic_1k.csv`（52 列） | `counts`、show/play/complete/valid/long/short-time、comment/like/follow/share/download 等聚合统计。 | 是一个月内按日与场景平均后的**结果型统计**，不是目标时刻的原生内容元数据。 | 禁止输入；读取它来预测同一月相当于潜在地使用 future exposure/feedback。 |

因此，字段“存在于 CSV”不等于“可进入模型”：是否可用首先由时点、数据合同和目标域决定，其次才是相关性或模型分数。

### 3.3 标签：为何只预测 `long_view`

主标签为官方提供的二元 `long_view`：

$$
y_i=\operatorname{long\_view}_i\in\{0,1\}.
$$

原始数据说明中，`long_view=1` 的规则为：视频时长不超过 18 秒时，播放时长至少达到视频时长；视频时长大于 18 秒时，播放时长至少达到 18 秒。可写作：

$$
\operatorname{long\_view}_i=
\begin{cases}
\mathbb{1}[\operatorname{play\_time}_i\ge \operatorname{duration}_i], & \operatorname{duration}_i\le18\text{s},\\
\mathbb{1}[\operatorname{play\_time}_i\ge18\text{s}], & \operatorname{duration}_i>18\text{s}.
\end{cases}
$$

| 标签/字段 | 本项目角色 | 处理理由 |
|---|---|---|
| `long_view` | 主训练、验证与评估标签 | 与离线候选偏好预测问题直接对应。 |
| `is_like`, `is_hate` | 辅助/安全标签登记 | 未用于主模型。 |
| `is_click` | 描述性诊断字段 | 在双列 UI 是 click、单列 UI 含 valid-play 含义；脱离 `tab` 不可被当作单一语义标签。 |
| 当前事件的播放/停留/反馈 | 禁止输入 | 它们在预测时尚不可见，使用即泄漏。 |
| 重算版 `long_view` | 未来敏感性分析候选 | 仅在公式可定义范围内分析，不能和官方标签混合或回填。 |

### 3.4 分布：为什么 random 是必要但不能被滥用的审计域

清洗后的总体标签率与 canonical 训练子集显示两个重要分布事实：标准与 random 的正例率明显不同，且用户权重会改变宏观结论。

| 范围 | 事件/目标行 | 用户 | `long_view` 正例率 | 解释 |
|---|---:|---:|---:|---|
| early Silver | 4,992,443 | 1,000 | 26.1296% | 清洗后的早期 standard 总体。 |
| late Silver | 6,556,501 | 1,000 | 25.9234% | 清洗后的后期 standard 总体。 |
| random Silver | 42,982 | 1,000 | 8.3198% | random 与 standard 存在显著基准率迁移。 |
| canonical Train (`tab=1`) | 2,399,844 | 950 | 31.8944% | early standard 的模型目标范围；另有 14,070 条同源公式疑点行按官方标签 re-attach。 |
| Train 的 user-macro 平均率 | — | 950 | 39.55% | 与 event-micro 相差 7.65 个百分点，故不能只看 pooled 指标。 |

因此，后续同时报告 pooled AP、event-gAUC、概率损失和用户簇不确定性；random 只用于冻结结论后的迁移审计，不能回流成为选参数据。

[返回导航](#toc)

<a id="screening"></a>

## 4. 粗筛：先设计假设和判据，再让 H2 进入模型

### 4.1 H1–H4 的设计背景，以及后来为何收紧为 RQ1–RQ4

最初研究设计提出 H1–H4，是为了让“长序列可能有价值”成为可被否定的命题，而不是看到关联后才写结论。[原始研究设计 §4.4](docs/design/KuaiRand_1K_Long_Sequence_Preference_Prediction_Agent_Research_Design.md)

| 原始假设 | 设计问题 | 需要的对照 | 2026-08-24 的实际状态 |
|---|---|---|---|
| **H1：历史增量** | 在相同用户、视频、上下文下，时点安全历史能否提升后期 `long_view` 预测？ | 静态无历史 BL1 vs 加历史候选。 | **已由 v010 检验**，但只限冻结 Adam 下的用户总体历史统计量。 |
| **H2：长序列增量** | 长/分层历史是否显著优于仅用最近 10 或 50 批次的短历史？ | 同一目标行上只改变 10/50/200 或序列窗口。 | **未完成**；v013 只登记了 DIN10/DIN50/DIN200 等设计，尚未有正式 Gold/训练结果。 |
| **H3：内容泛化** | 加入候选内容属性后，能否在前期未见视频上优于纯 video-ID？ | 内容/作者/tag 等特征 vs 纯 ID，对 warm/cold video 切片。 | 部分静态内容字段已在 BL1 中；高基数 `H3_USER_CONTENT_STRICT_V1` 交互在 55 分钟 quick stop 后、读取模型指标前被延后，**没有独立 H3 结果**。 |
| **H4：稳定性** | 改善是否不依赖单日、高活跃用户或特定时长/内容切片？ | 预注册逐日与切片检查。 | v010 有 4/4、17/17、14/17 的逐日信息和描述性切片；它支持稳定性线索，但小切片不能替代主 bootstrap。 |

有一个容易混淆的命名：原始**假设 H2**指“长序列窗口的额外价值”；而实际运行的 `H2_USER_STRICT_V1` 是“用户总体历史特征包”的名称。v010 直接回答的是**原始 H1**，使用的候选特征包恰好叫 `H2_USER_STRICT_V1`；它没有完成原始 H2 的 10/50/200 消融，也没有完成 H3 的用户×内容交互实验。

随后方法审查指出：池化 AP/AUC 可能主要区分不同用户的基础率，不能自动等同于“同一用户内的视频排序”。因此假设登记将问题收紧为 RQ1–RQ4：历史概率增量、用户内内容判别、历史深度增量、时间/内容稳定性，并把 user-GAUC、固定目标行和配对用户簇推断写入合同。[假设登记](kuairand-longseq-agent/configs/hypothesis_registry_v002.yaml) · [建模审查](kuairand-longseq-agent/reports/modeling_standard_review_2026-08-14.md)

### 4.2 指标、差值与 95% CI：表中的数字应如何阅读

同一指标不能回答所有问题。报告中的 \(\Delta\) 默认是“候选模型 − 基线模型”；因此 AP/user-GAUC 为正较好，Log Loss/Brier 为负较好。

| 指标 | 数学定义/方向 | 它回答什么 | 常见误读 |
|---|---|---|---|
| **AP（Average Precision）** | \(\mathrm{AP}=\frac{1}{N_+}\sum_{r=1}^{N}\mathrm{Precision}@r\cdot\mathbb{1}[y_r=1]\)；越大越好。 | 把所有事件按分数排序后，正例是否集中在前部；是主排序指标。 | AP 受正例率影响，不能跨 prevalence 很不同的切片只比较裸 AP。 |
| **event-gAUC / user-GAUC** | 先在每名同时有正负例的用户内计算 \(\mathrm{AUC}_u\)，再按其合格事件数加权平均；越大越好。 | 同一用户内，正例是否通常得分高于负例，防止 pooled 提升全来自用户基础率差异。 | 它不是已验证的线上曝光 slate 排序；同 `time_ms` pair AUC 只能是探索性诊断。 |
| **Log Loss** | \(-\frac1n\sum_i[y_i\log\hat p_i+(1-y_i)\log(1-\hat p_i)]\)；越小越好。 | 概率是否既正确又不过度自信；把 0.99 预测错会重罚。 | 只看 AP 不能保证 Log Loss 好；Gate 2B 正是反例。 |
| **Brier Score** | \(\frac1n\sum_i(\hat p_i-y_i)^2\)；越小越好。 | 概率误差的平方平均，直观反映“预测概率离 0/1 标签有多远”。 | 它与 Log Loss相关但不等价；应同时报告。 |
| **ECE20** | 将预测按 20 个桶分组，汇总每桶平均预测与实际率的偏差；越小越好。 | 概率校准的描述性诊断。 | 在本项目中不单独决定 pass/fail；分桶方法本身也会影响数值。 |

95% CI 不是“95% 的单次预测会落在区间内”。本项目的主 CI 通过 **2,000 次配对用户簇 bootstrap**得到：

1. 以**用户**而不是单条事件为重采样单位，抽样时保留同一用户所有事件的相关性；
2. 对每个 bootstrap 样本，同时重算 BL1 与 BL2，并计算同一批重采样用户上的模型差值；
3. 将 2,000 个差值的 2.5% 与 97.5% 分位数作为 percentile 95% CI。

所以，\(\Delta\mathrm{AP}=+0.035626\;[0.031302,0.040028]\) 表示：在该合同、样本和重采样方案下，历史模型的 AP 增量始终落在正区间；它支持“离线增量不太像抽样波动”，但**不**自动证明因果、跨数据集可迁移或线上收益。若 Log Loss/Brier 的 CI 跨 0，只能说点估计方向改善，不能声称该概率改善已被统计确定。

还有一点必须在读表前声明：**本项目所有 95% CI 均为 per-comparison（逐对比）区间，未做族错误率（FWER/FDR）校正。**
v010 涉及 3 个阶段 × 4 个主指标，v011 涉及 3 个校准族，v012 涉及 3 个候选配置 × 7 个门。
项目对多重比较的控制手段是**预注册**——主指标、最小效应量、判据与选择规则在看到结果前已写入合同并冻结，
而不是事后从多个结果里挑显著的。这在设计上是有效的，但它不等价于统计校正。
因此，对于点估计接近门槛、CI 下界贴近 0 的对比（例如 §9.5 中 v012 的 ΔBrier 下界 `-0.000333`），
应当按"单次预设检验的边缘证据"来读，而不是按"已排除多重比较风险的确证结果"来读。

### 4.3 Train-only EDA：用来登记混杂和可检验方向，而不是硬筛字段

粗筛只读取 2022-04-08 至 04-17 的 canonical source-Train：所有目标行固定为 `tab=1`，历史只来自严格更早的全 `tab` standard 事件。它不读取 late/random，也不构建正式 Gold 或训练正式候选模型。[Train-only 关联报告](kuairand-longseq-agent/reports/analysis/train_association_report_v002.md)

| 观察 | 数据 | 正确解释 | 不能推出的结论 |
|---|---|---|---|
| 时间漂移 | 日正例率为 29.90%–34.88%。 | 后续比较要用 rolling-origin 并控制日期。 | 不能把某日差异当作特征效果。 |
| 历史深度混杂 | 历史越深，原始正例率越低。 | 深度与日期、活跃度、曝光量共同变化。 | 不能说“长历史导致偏好变弱”。 |
| 用户总体率 | `prior_user_lv_rate`：pooled AUC 0.7088，user-GAUC 0.5170。 | 跨用户基础率有信息。 | 同一用户、同一时间批次内并不能区分视频。 |
| 最近 10 批次率 | pooled AUC 0.7220，user-GAUC 0.5857，批次内 pair AUC 0.5000。 | 最近历史在用户内有状态信息。 | 不是候选内容偏好的直接证据。 |
| 时长/上传类型 | 非正/缺失时长、上传类型有明显异质性。 | 需要显式质量状态与可用性审计。 | 不能将字段名直接解释为真实内容语义。 |

EDA 的作用是把 H1 落成一个可复核的嵌套比较：在相同候选行、相同静态特征与同一训练协议下，`BL2 = BL1 + H2_USER_STRICT_V1` 是否改善指标；任何单变量 AUC 都不直接淘汰字段。

### 4.4 Gate 2B：有排序信号，但概率失败阻止了错误晋级

在正式 GPU 验证前，Gate 2B 以 7 个 Train rolling-origin 固定行回测进行了小范围预实验。它保留 H1 的排序线索，却通过绝对概率审计阻止了错误晋级。[Gate 2B 报告](kuairand-longseq-agent/reports/analysis/gate2b_baseline_results_v002.md)

| 模型 | AP | user-GAUC | Log Loss | Brier | 面向读者的解释 |
|---|---:|---:|---:|---:|---|
| BL0 常数概率 | 0.324906 | 0.498371 | 0.627359 | 0.217828 | 不区分任何事件；给出“只报平均概率”时的绝对概率下限。 |
| BL1 静态线性基线 | 0.324844 | 0.501942 | 5.162145 | 0.320341 | 排序几乎没有超过 BL0，且极端概率使两项概率损失急剧恶化。 |
| BL2 = BL1 + `H2_USER_STRICT_V1` | 0.435665 | 0.523877 | 4.269645 | 0.275823 | 相比 BL1 的排序更好，却仍远差于 BL0 的绝对概率；不能被称为可用概率模型。 |

`BL2 − BL1` 的 Train-only \(\Delta\mathrm{AP}=+0.110821\)，95% CI \([0.101479,0.120377]\)，7/7 天为正；但 BL1/BL2 的大量预测被裁剪在极端区间。这个结果只保留“严格历史值得在健康优化协议下继续检验”的动机，**不**允许声称概率预测已可靠、Gold 已完成、或可以进入序列模型。

[返回导航](#toc)

<a id="silver"></a>

## 5. Silver：清洗不是删除麻烦数据，而是建立可审计输入

### 5.1 清洗原则

正式 Silver 运行是 `silver-20260814-155536`（v0.1.0）。Raw CSV 不移动、不覆盖、不原地修改；每种问题行都被保留为 canonical、quarantine 或质量标记，而非“清干净了就消失”。[Silver 清洗报告](kuairand-longseq-agent/reports/generated/silver_cleaning_report.md)

| 规则 | 处理 | 研究意义 |
|---|---|---|
| 完全重复行 | 保留一条 canonical，重复副本写入审计制品。 | 防止重复计数，同时保留可追溯性。 |
| 同一 `user_id+video_id+time_ms` 载荷冲突 | 整组进入 quarantine。 | 不擅自选第一条/最后一条造成隐性标签选择。 |
| 同用户同时间、不同视频 | 全部保留为时间戳批次。 | 它是候选批次，不是重复行。 |
| `tab` / 二元值 / 时间等非法域 | 隔离。 | 保障输入合同；合法 `tab` 是 \([0,14]\)。 |
| 缺失或非正时长 | 保留并标记。 | 不将未知错误填为 0，也不选择性删除用户。 |
| `long_view` 公式不一致 | 隔离为公式一致性疑点，保留官方标签和重算值。 | 不将它直接宣称为官方错误标签。 |
| `is_live_streamer=-124` | 映射为 `UNKNOWN_SENTINEL`。 | 它不是普通的 0；删除会改变研究人群。 |
| `video_features_statistic` | 不物化、不 join。 | 防止同月后验统计泄漏。 |

### 5.2 Silver 行数对账（逐表嵌入引用）

所有数值来自同一份 [Silver 清洗报告](kuairand-longseq-agent/reports/generated/silver_cleaning_report.md)，并可由已保存的 [row reconciliation CSV](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/row_reconciliation.csv)、[质量规则汇总](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/quality_rule_summary.csv)、[正式运行 manifest](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/silver_run_manifest.json) 交叉核验；这里显式链接到 `governed_data/formal_silver_snapshot`，因为主项目的 `data/manifests/` 只保留占位符，不能把占位符误称为证据文件。

| 来源（Raw 链接） | Raw 行 | canonical 行 | 重复副本 | 冲突隔离 | 公式疑点隔离 | Silver 行 | 对账/逐表核验 |
|---|---:|---:|---:|---:|---:|---:|---|
| [early standard](KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv) | 5,055,984 | 5,021,422 | 34,562 | 5,455 | 23,524 | 4,992,443 | [行数](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/row_reconciliation.csv) · [规则](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/quality_rule_summary.csv) · PASS |
| [late standard](KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv) | 6,657,061 | 6,597,399 | 59,662 | 8,514 | 32,384 | 6,556,501 | [行数](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/row_reconciliation.csv) · [规则](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/quality_rule_summary.csv) · PASS |
| [random](KuaiRand-1K/data/log_random_4_22_to_5_08_1k.csv) | 43,028 | 43,027 | 1 | 0 | 45 | 42,982 | [行数](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/row_reconciliation.csv) · [规则](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/quality_rule_summary.csv) · PASS |
| [users](KuaiRand-1K/data/user_features_1k.csv) | 1,000 | 1,000 | 0 | 0 | 0 | 1,000 | [行数](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/row_reconciliation.csv) · [规则](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/quality_rule_summary.csv) · PASS |
| [videos basic](KuaiRand-1K/data/video_features_basic_1k.csv) | 4,371,868 | 4,371,868 | 0 | 0 | 0 | 4,371,868 | [行数](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/row_reconciliation.csv) · [规则](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/quality_rule_summary.csv) · PASS |

不要将 §3.1 的“规则命中数”与本表的“最终隔离数”机械相减：同一 Raw 行可同时命中重复、冲突、时长或公式规则，而本表按确定性处理顺序作**互斥的最终去向**对账。例如 early 的 23,538 个公式一致性检测并不等于 23,538 个最终独立公式隔离行；最终对账为 23,524。质量规则表记录的是发现轮廓，row reconciliation 记录的是输出行守恒，两者共同才是完整证据。

独立验收还验证了二元域、`tab` 范围、事件—用户/视频连接缺失均为 0、Raw 文件元信息不变、输出文件 SHA-256 匹配，以及统计表未进入 Silver；正式输出哈希见 [Silver output manifest](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/manifests/silver_output_manifest.json)。

### 5.3 训练视图为何会 re-attach 公式疑点行

Silver 是治理锚点；模型的 canonical 视图可以在明确合同下做只读 union：

$$
\text{canonical model view}
=\text{Silver event rows}
\;\cup\;
\text{exclusive formula-mismatch rows with official }long\_view.
$$

这并不“改写”Silver。它只是在模型合同中重新加入与 Silver 身份键不重叠的同源公式疑点行，并继续使用官方标签；冲突行与非法域行仍不加入。这样既不静默丢失官方标签，也把公式敏感性分析与主标签严格分离。

[返回导航](#toc)

<a id="gold"></a>

## 6. Gold：设计已明确，正式版本化数据层尚未物化

### 6.1 需要澄清的状态

项目目录中存在 `data/gold/` 作为未来层的保留位置，但截至 2026-08-24 仍没有获批、非空、带 Gold run/feature/target manifests 的正式快照；v013 的先决条件也明确把它列为缺口。[Gold 数据层说明](kuairand-longseq-agent/data/README.md) · [v013 前置条件](kuairand-longseq-agent/experiments/neural_sequence_candidate_model_v013/README.md)

同时，v010–v012 的受控运行确实构造了满足本节规则的、按阶段合同固定的特征矩阵和目标行。这些是 **Gold-equivalent experiment matrices**，支持相应实验的可复核结论；但它们不能被回填命名为已经发布的通用 `data/gold` 数据集。下表区分二者：

| 层次 | 当前状态 | 可做什么 | 不能声称什么 |
|---|---|---|---|
| Silver | 已冻结、已验收 | 作为后续构建的只读输入锚点。 | 不是一行一候选的时点特征宽表。 |
| Gold 设计/合同 | 已明确 | 定义 sample 粒度、split、allowlist、泄漏测试与 manifest 要求。 | 不等于 Gold 已运行完成。 |
| v010–v012 合同特征矩阵 | 已存在于具体实验链 | 支持该合同下的模型训练、评估与哈希复核。 | 不等于可供任意未来实验复用的 Gold snapshot。 |
| `data/gold/` 正式层 | 尚未物化/验收 | 未来应版本化构建。 | 不能说 Gold 已完成或序列模型已获授权。 |

### 6.2 Gold 的目标粒度与防泄漏算法

Gold 的一行应是一条候选事件，而不是把整个用户或视频压成一行。对用户 \(u\) 在目标时刻 \(t_i\)，允许的历史集合是：

$$
\mathcal{H}_u(t_i)=\{j:u_j=u,\;t_j<t_i\}.
$$

严格不等式 \(t_j<t_i\) 是核心控制。对相同 `(user_id,time_ms)` 的多个候选，执行顺序必须是：

```text
读取目标时间戳批次
→ 用此前已冻结的用户状态为批内每个候选构造同一份历史特征
→ 预测/保存这些候选特征
→ 才用该批次的已观测反馈更新用户历史状态
```

因此，批内视频可共享用户状态，却绝不能互相看到当前批次的标签、播放时长或点击。未来正式 Gold 还必须输出 `sample_id`、split/行数对账、输入/特征清单哈希、时间线复算与泄漏测试；这些正是 v013 未执行前的硬前置条件。

[返回导航](#toc)

<a id="models"></a>

## 7. 模型、特征与公式

### 7.1 三个嵌套模型：同一候选行上只改变允许的信息

所有三个模型预测的是第 \(i\) 个候选事件的 `long_view`。它来自 [early Silver events（4,992,443 行）](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/silver/events_early_standard.parquet)、[late Silver events（6,556,501 行）](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/silver/events_late_standard.parquet) 和 [random Silver events（42,982 行）](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/silver/events_random.parquet) 的 canonical 模型视图；v010 训练合同的最终训练行数为 **2,399,844**（950 用户、974,550 视频、765,417 正例）。[Silver users（1,000 行）](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/silver/users.parquet)、[Silver videos basic（4,371,868 行）](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/silver/videos_basic.parquet) 只以允许的静态字段 join；`video_features_statistic_1k.csv`（4,371,868 行）明确 denylist，完全没有进入模型。[训练合同与特征设计](预测标签、模型公式、设计由来与可行性.md) · [Silver 清洗报告](kuairand-longseq-agent/reports/generated/silver_cleaning_report.md)

| 模型 | 输入文件 / 已使用行数 | 公式/组成 | 该比较回答什么 |
|---|---|---|---|
| BL0 | 同一目标行；无学习特征。 | \(\hat p_i^{(\mathrm{BL0})}=\hat\pi_{\mathrm{train}}\)。 | 只报告训练期总体正例率时，概率损失可达到的基础参照。 |
| BL1 | 同一目标行 + [users 1,000 行](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/silver/users.parquet) + [videos basic 4,371,868 行](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/silver/videos_basic.parquet)；实际 join 到训练候选的用户 950、视频 974,550。 | 静态字段的正则化稀疏逻辑回归。 | 静态 ID、元数据和质量状态本身能解释多少。 |
| BL2 | 与 BL1 完全相同的静态 join/目标行；再从此前事件批次生成历史。 | \(\mathbf{x}^{(\mathrm{BL2})}_i=[\mathbf{x}^{(\mathrm{BL1})}_i,\mathbf{h}_u(t_i)]\)。 | **H1：严格用户总体历史是否在同一静态模型之上带来增量。** |

对 BL1/BL2，线性分数与概率为：

$$
s_i=\mathbf{x}_i^\top\mathbf{w}+b,
\qquad
\hat p_i=\sigma(s_i)=\frac{1}{1+\exp(-s_i)}.
$$

其中，\(\mathbf{x}_i\) 是第 \(i\) 个候选的完整允许特征向量；\(\mathbf{w}\) 是每一个展开列对应的待学习权重；\(b\) 是不依赖具体特征的全局截距；\(s_i\) 是未校准的 log-odds 分数；\(\sigma(\cdot)\) 把任意实数压到 \((0,1)\) 并解释为预测的 long-view 概率。BL1 与 BL2 的唯一系统差异是 \(\mathbf{h}_u(t_i)\)，所以两者的差异可归因于 H2，而不是换了数据行、标签或静态特征。

### 7.2 BL1：19 个原始字段组，而不是固定 19 维

下表的“行数”是**来源表的行数**，不是每个字段被展开后的列数；类别字段只从训练拟合集学习词典，因此最终稀疏维度随合同变化。字段映射和转换实现可由 [基线特征代码](kuairand-longseq-agent/src/kuairand_longseq/models/gate2b_repair_v003.py) 复核。

| 字段组 | 具体字段（完整列名） | 数据来源 / 来源行数 | 变换与缺失语义 | 设计意图 |
|---|---|---|---|---|
| 类别静态（9） | `cat_user`; `cat_video`; `cat_author`; `cat_music`; `cat_video_type`; `cat_upload_type`; `cat_music_type`; `cat_tag_combo`; `cat_duration_bucket` | 事件键：三张已链接 Silver events（共 11,591,926 行）；视频侧字段：[videos basic 4,371,868 行](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/silver/videos_basic.parquet)。 | 仅用训练拟合行 One-Hot；每字段最小频次 20、最多 4,096 类；未知类别不会从验证/审计标签中学习。 | 保留实体、发布与内容的粗粒度静态差异，并限制超高基数记忆。 |
| 连续静态（5） | `static_log_duration`; `static_log_upload_age`; `static_log_width`; `static_log_height`; `static_aspect` | [videos basic 4,371,868 行](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/silver/videos_basic.parquet) 的时长/上传日/宽高；目标时刻来自三张已链接 Silver events。 | 对正值作 `log1p`；上传年龄由 `target_time-upload_time` 计算；宽高比为宽/高。 | 降低时长、年龄和分辨率长尾的尺度支配，并保留内容新旧与画面形状。 |
| 质量/状态（5） | `static_duration_valid`; `static_upload_age_valid`; `static_upload_future`; `static_geometry_valid`; `static_tag_missing` | [videos basic 4,371,868 行](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/silver/videos_basic.parquet)；`tag` 缺失 137,484 行、非法/非正 duration 573,057 行已在 §3.1 说明。 | 二元状态列不把异常伪装成数值 0；`upload_future` 显式标出上传时间晚于目标时刻。 | 让模型知道“未知/异常”本身，而不是让填补值制造不存在的含义。 |

因此，“BL1 有 19 个字段组”指 \(9+5+5\) 个**语义输入组**；例如 `cat_video` 在训练中可能展开成很多列，而 `static_aspect` 只是一列。`user_features_1k.csv` 的完整 31 列没有被整体喂入：本基线实际只从事件中的 `user_id` 构造 `cat_user`；其余个人属性并非本合同的特征来源。`video_features_statistic_1k.csv` 即使有 52 列和同样 4,371,868 行，也没有进入表中任何一项，因为它可能包含同月行为后验统计。[Raw 字段与口径](KuaiRand-1K/README.md)

### 7.3 H2：18 个严格用户总体历史特征及其 20 伪计数

H2 的来源不是另一张静态表，而是该用户在**严格过去的 standard 事件**中的状态：对 standard 目标，使用 [early（4,992,443 行）](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/silver/events_early_standard.parquet) 与允许的更早 [late（6,556,501 行）](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/silver/events_late_standard.parquet) 批次；对 [random 目标（42,982 行）](governed_data/formal_silver_snapshot/kuairand-longseq-agent/data/silver/events_random.parquet)，状态表**只含 standard 事件**，绝不让任何 random 标签更新 random 自己或其他 random 行的 H2。对目标用户 \(u\)、目标时间 \(t_i\)，先构造：

$$
\mathcal H_u(t_i)=\{j:u_j=u,\;t_j<t_i\}.
$$

这里 \(j\) 是此前候选事件，\(t_j<t_i\) 是严格不等式。若多条候选共享同一 `(user_id,time_ms)`，它们先共享同一份旧状态并被预测，随后才整体更新状态；当前批次的 `long_view`、播放时长和同批反馈都不可用。实现见 [严格历史特征代码](kuairand-longseq-agent/src/kuairand_longseq/features/history_value_feature_sql.py)，防泄漏顺序见 [Gold 设计](#gold)。

对窗口 \(k\in\{10,50,200\}\)，以**历史时间批次**而非原始事件行统计事件数 \(E_{u,t}^{(k)}\) 与正例数 \(P_{u,t}^{(k)}\)，并定义：

$$
R_{u,t}^{(k)}=
\frac{P_{u,t}^{(k)}+\lambda\hat\pi_{\mathrm{train}}}
{E_{u,t}^{(k)}+\lambda},
\qquad\lambda=20.
$$

\(\hat\pi_{\mathrm{train}}\) 是训练拟合集的总体 long-view 正例率；\(E\) 是此前允许的事件总数；\(P\) 是其中 `long_view=1` 的数目；\(R\) 是供模型使用的平滑历史率；\(\lambda\) 是**伪计数/先验强度**，不是第 20 个特征、更不是从 random audit 调出的超参数。

为什么是 20？冻结的 H2 合同把 \(\lambda=20\) 规定为“小样本收缩”的保守常量：它等价于在观察数据前加入 20 个、正例比例为 \(\hat\pi_{\mathrm{train}}\) 的虚拟历史事件。于是当 \(E=0\) 时 \(R=\hat\pi_{\mathrm{train}}\)；当 \(E=1\) 时，单次正/负反馈的权重仅为 \(1/21\)，不会把比率推到 0 或 1；当 \(E=20\) 时，数据和先验各贡献一半；当 \(E\gg20\) 时，实际历史主导。20 是一项**预先固定的稳定性选择**，并没有在 Validation、sealed 或 random 上搜索“最优 20”，因此不能声称它是统计最优、普适或因果正确的强度。

| H2 部分 | 完整输出字段 | 来源 / 使用行 | 特征数 | 每一项的意义 |
|---|---|---|---:|---|
| 最近 10 批次 | `log1p_w10_event_count`, `log1p_w10_positive_count`, `smoothed_w10_long_view_rate`, `w10_full_window_mask` | 允许的严格先前 **standard** 批次；random 目标也只读 standard 状态。 | 4 | 短期历史量、正例量、平滑率，以及是否已拥有 10 个完整先前批次。 |
| 最近 50 批次 | `log1p_w50_event_count`, `log1p_w50_positive_count`, `smoothed_w50_long_view_rate`, `w50_full_window_mask` | 同上。 | 4 | 中期状态，避免只把“最近一次”误作稳定偏好。 |
| 最近 200 批次 | `log1p_w200_event_count`, `log1p_w200_positive_count`, `smoothed_w200_long_view_rate`, `w200_full_window_mask` | 同上。 | 4 | 较长观察尺度及历史是否足够深。 |
| 全历史 | `log1p_prior_batch_count`, `log1p_prior_event_count`, `log1p_prior_positive_count`, `smoothed_lifetime_long_view_rate`, `log1p_last_user_gap_seconds` | 同上；间隔由目标批次与上一个允许历史批次的时间差算出。 | 5 | 分别描述批次数、事件量、正例量、长期平均状态和新近性。 |
| 冷启动状态 | `has_history` | 同上。 | 1 | 将“完全没有历史”与“有历史但未出现正例”区分开。 |
| **总计** | 上述全部字段 | 训练 2,399,844 目标行；评估时各自只用该阶段允许的历史。 | **18** | \(3\times4+5+1=18\)。 |

H2 仅为用户**总体**先前 long-view 状态：没有 `user_id × tag`、`user_id × author`、候选视频标签或相似度交互。因此，BL2 的成功可支持 H1，却不能支持原始 H2（长窗口相对短窗口的消融）或 H3（用户—内容偏好）的成功。

### 7.4 v010 的学习目标、参数与信息边界

给定二元标签 \(y_i\in\{0,1\}\)（1 代表 `long_view`），v010 最小化：

\[
\mathcal{L}(\mathbf{w},b)=
-\frac{1}{n}\sum_{i=1}^{n}
\left[y_i\log\sigma(s_i)+(1-y_i)\log(1-\sigma(s_i))\right]
+\frac{\alpha}{2}\lVert\mathbf{w}\rVert_2^2,
\qquad\alpha=10^{-4}.
\]

\(n\) 是当前训练目标行数；第一项是每行的二元交叉熵/负对数似然，真实为 1 却预测接近 0（或反过来）的高置信错误会被重罚；第二项为 L2 正则，\(\alpha=10^{-4}\) 控制对所有非截距权重 \(\mathbf w\) 的收缩强度；\(\lVert\mathbf w\rVert_2^2\) 是这些权重平方和。截距 \(b\) 不参与 L2 正则，以免机械地把总体基准率压向 0。损失中没有 `watch_ratio`、原始 `play_time_ms`、当次/同批 `long_view`、未来事件或 statistic 表后验字段；这些均会违反目标或时点边界。

### 7.4a 必须先说明的前提：v010 的冻结模型权重是**重建**得到的

v010 的三个阶段完整保存了特征矩阵、逐行预测、指标、bootstrap 与 manifest，
但**没有把训练完成的模型权重写成文件**。因此后文 §7.6 与 §9.5 所说的
"从 sealed BL1/BL2 暖启动"，实际加载的不是 v010 运行时的原始权重，而是事后
从已封存的 sealed 特征重新拟合复原出来的状态：

```text
reports/generated/sealed_model_reconstruction_diagnostic_v010/reconstructed_frozen_model_state.npz
SHA-256  c36c6bec407569d81734d1809c13d4fe2b96e410da6493dee04af77b6dfaeb60   (4,927,967 bytes)
```

这一点在 v012 的冻结合同里是明写的（`frozen_design.source: reconstructed_sealed_GroupedDesign`），
本节把它提到正文，避免读者把"暖启动"理解成加载了原始冻结模型。

重建质量由独立诊断给出，逐行比对已封存的 sealed 预测：

| 量 | 平均绝对差 | p99 | 最大差 | 相关系数 |
|---|---:|---:|---:|---:|
| `ADAM_BL1_probability` | 1.28e-08 | 4.74e-08 | 1.59e-07 | 1 |
| `ADAM_BL2_probability` | 1.37e-08 | 4.87e-08 | 5.14e-07 | 1 |
| `ADAM_BL1_raw`（logit） | 5.98e-08 | 2.38e-07 | 9.54e-07 | 1 |
| `ADAM_BL2_raw`（logit） | 1.05e-07 | 4.77e-07 | 3.70e-06 | 1 |

差异在 float32 数值容差之内，对 AP、gAUC、Log Loss、Brier 的任何一位有效数字都不可见，
**v011/v012 的结论不受影响**。[重建诊断](kuairand-longseq-agent/reports/analysis/sealed_model_reconstruction_diagnostic_v010.md)

两点必须同时说清楚：

1. 这次补救之所以成立，是因为 sealed 阶段的**特征矩阵恰好被保留了**。若当时特征也未落盘，
   整条 v011/v012 链无法建立。
2. 因此，后续任何实验合同都应把**"模型权重必须持久化并进入 artifact hash manifest"**
   列为硬性输出要求，而不是可选项。

### 7.5 v011：只修正目标域概率尺度，不制造排序增量

**设计意图。** v010 已证实 random exposure 中 BL2 比 BL1 排序更好，但其约 8.4% 的目标域正例率远低于 standard 训练期约 31%–32%。冻结分数的排序可用，并不等于它输出的 0.35 就真代表 35% 概率。v011 的工作是让概率刻度适配目标域，明确不允许借“校准”制造排序收益。

**实验过程。** 冻结 v010 BL2 和其特征/历史定义后，random 目标域按时间分成 8,731 行 calibration-fit、10,544 行 selection、23,752 行 held-out test。候选校准器先仅在 fit 拟合，再在 selection 选择，最终可在 fit+selection 重拟合；held-out 完全不参与模型族或参数选择。三个候选为：M1 先验/截距平移；M2 固定正斜率、仅拟合截距；M3 同时拟合斜率和截距的 Platt 型 sigmoid。M2 与 M3 的 selection Log Loss 仅差 0.000016，在合同的 tie 处理下选择更简单的 M2。[v011 合同与结果](kuairand-longseq-agent/experiments/bl2_target_domain_calibration_v011/results_v011.md)

对冻结 BL2 分数 \(s_i\)，形式统一为：

$$
\hat p_i^{(\mathrm{v011})}=\sigma(a s_i+c).
$$

最终存档的是 M2：

$$
a=1.0169043468,
\qquad c=-2.0952099481.
$$

这里 \(a\) 是预冻结的正斜率，控制 logit 尺度；M2 的 “intercept only” 是指选择/最终拟合不重新学习斜率，而只从目标域标签估计截距 \(c\)。\(c\) 是整体 log-odds 平移，因此可把平均预测从过高的 standard 尺度下调。由于 \(a>0\)，任意 \(s_i>s_j\) 仍然得到 \(\hat p_i>\hat p_j\)：AP、ROC-AUC、event-gAUC 和 user-gAUC 应精确不变，这是 eligibility 硬门。

| held-out 指标（23,752 行 / 967 用户 / 2,063 正例） | 原始冻结 BL2 | M2 校准后 | 差异 / 95% CI | 该结果说明什么 |
|---|---:|---:|---|---|
| AP | 0.197730 | 0.197730 | 0 | 校准没有改动排名次序。 |
| event-gAUC | 0.623640 | 0.623640 | 0 | 同上；不是新的 H2 证据。 |
| Log Loss | 0.512076 | 0.268939 | -0.243137 [-0.258633, -0.226836] | 概率对数损失显著下降。 |
| Brier | 0.169978 | 0.074827 | -0.095151 [-0.102219, -0.087959] | 平方概率误差显著下降。 |
| ECE20 | 0.281335 | 0.006724 | — | 20-bin 的校准偏差描述大幅减小。 |

总结：v011 支持“该 post-audit 目标域上，可用独立数据把冻结排序变成可信得多的概率”；它不把 v010 的 random audit 变回 untouched，也不检验 H2 的内容偏好机制。

### 7.6 v012：受 tether 约束的目标域适配，随后独立校准

**设计意图。** v011 只改变概率映射，理论上不能提高 AP；若目标域的特征—标签关系也变了，需检验有限度重训练是否有增量。但小目标域样本上自由重训会遗忘 sealed 模型、追逐噪声，因此 v012 用 tether 将参数拉回已冻结的 sealed 解，并设置独立校准与最终时间回放。

**实验过程。** 从 sealed BL1/BL2 暖启动，按时间依次使用 11,999 行 target-adaptation-train 更新、7,276 行 target-calibration 分别校准 NEW_BL1/NEW_BL2、11,353 行 model-selection 在 C1/C2/C3 中按冻结规则择一，最后才在 12,399 行 final temporal replay 测试。每一步都不回看后续段标签；选择后冻结 `C2_balanced`。它不是 v010 的 confirmatory 阶段，而是后审计的工程验证。[v012 合同与结果](kuairand-longseq-agent/experiments/bl2_target_domain_retraining_v012/results_v012.md)

v012 训练目标为：

$$
\mathcal{L}_{\mathrm{v012}}(\mathbf{w},b)=
-\frac{1}{n}\sum_{i=1}^{n}
\left[y_i\log\sigma(s_i)+(1-y_i)\log(1-\sigma(s_i))\right]
+\frac{\tau}{2}\lVert\mathbf{w}-\mathbf{w}_{\mathrm{sealed}}\rVert_2^2.
$$

\(\mathbf{w}_{\mathrm{sealed}}\) 是 v010 已冻结模型的非截距权重；\(\mathbf w\) 是待适配权重；\(b\) 是可随目标域基准率变化的截距；\(\tau\) 是 tether 强度，惩罚 \(\mathbf w\) 偏离 \(\mathbf w_{\mathrm{sealed}}\) 的平方距离。最终 C2 参数为：

$$
\eta=0.01,
\qquad\text{steps}=100,
\qquad\tau=0.001.
$$

\(\eta\) 是 Adam 的单次更新学习率；`steps` 是全批更新次数；\(\tau=0.001\) 表示适配可以改变权重，但不是不受约束的重新训练。截距不受 tether，之后又在独立 calibration 段重新做概率校准，这把“排序结构的适配”和“平均概率的修正”分离开。

| final temporal replay（12,399 行 / 857 用户 / 1,039 正例） | AP | event-gAUC | Log Loss | Brier | ECE20 | 角色 |
|---|---:|---:|---:|---:|---:|---|
| TARGET_BL0 | 0.083797 | 0.500000 | 0.288005 | 0.076784 | 0.002967 | 只报目标域总体率的概率参照。 |
| OLD_BL2 + v011 | 0.187896 | 0.621353 | 0.263925 | 0.072850 | 0.006645 | 不重训练的部署候选。 |
| NEW_BL1 | 0.176476 | 0.613925 | 0.269303 | 0.073613 | 0.006894 | 适配后不含历史的嵌套对照。 |
| NEW_BL2 + v012 calibrator | 0.204928 | 0.645537 | 0.260618 | 0.072098 | 0.006200 | 选中的可回滚工程候选。 |

同一行配对的 `NEW_BL2 − OLD_BL2+v011` 为 \(\Delta\mathrm{AP}=+0.017032\,[0.005265,0.031319]\)、\(\Delta\mathrm{event\text{-}gAUC}=+0.024183\,[0.007412,0.041419]\)、\(\Delta\mathrm{LogLoss}=-0.003307\,[-0.005145,-0.001725]\)、\(\Delta\mathrm{Brier}=-0.000752\,[-0.001243,-0.000333]\)。H2 AP 在 3/3 天为正（门为至少 2 天）；因此结果支持 `retraining_adds_value`，但工程决策仍是 **pending new-data confirmation**，不是自动上线。

[返回导航](#toc)

<a id="optimizer"></a>

## 8. 预实验与优化器比较：为什么最终只保留 Adam

### 8.1 先比较“是否充分优化”，再比较科学结果

Adam/SGD 的比较只在 Train-only objective preflight 中进行；还没有用 Validation、sealed 或 random 的标签来挑选优化器。这里的 **CPU-LBFGS** 是在 CPU 上运行的 Limited-memory Broyden-Fletcher-Goldfarb-Shanno（有限内存拟牛顿）优化器：它不用完整 Hessian 矩阵，而用最近若干梯度/参数变化近似曲率，因而适合为凸的 L2 逻辑回归求一个高精度、可复查的参考解。它既不是第三个部署模型，也不参加 AP、gAUC、Log Loss 的科学比较；它的唯一角色是回答“GPU 上的 Adam/SGD 是否已把**同一个**目标函数优化得足够接近”。

对每个冻结 Train probe、每个 BL1/BL2，记 CPU-LBFGS 收敛后的目标为 \(\mathcal L_{\mathrm{reference}}\)，GPU 优化器同一损失（含同一 \(\alpha=10^{-4}\)）为 \(\mathcal L_{\mathrm{GPU}}\)。GPU 训练需要满足：

$$
\text{regret}=\mathcal{L}_{\mathrm{GPU}}-\mathcal{L}_{\mathrm{reference}}
\le \max\left(10^{-4},\;0.005\left|\mathcal{L}_{\mathrm{reference}}\right|\right).
$$

这里的 regret 是**优化 regret/目标函数缺口**，不是用户体验 regret、强化学习 regret，也不是 BL2 相对 BL1 的指标差。理论上参考解应不差于同一凸目标的未充分 GPU 解，因此越接近 0 越好；负的极小数可视作数值容差内的等价解。右边的“允许值”采用双重保护：绝对下限 \(10^{-4}\) 避免在极小目标时容差荒谬地苛刻，0.5% 相对容差则让不同 probe 的损失尺度可比较；取 `max` 表示允许较宽的那一个。例如 BL1 参考目标约 0.546426 时，允许值为 \(\max(0.0001,0.005\times0.546426)=0.002732\)。

这一步防止 BL1、BL2 因优化充分程度不同而被错误比较。v001 曾在 CPU-LBFGS 500 次上限出现 warning；提高到 2,000 后，6/6 个参考解在 644–952 次无 warning 收敛，说明先前是参考求解器迭代上限过紧，而不是数据、标签或历史定义已经失效。[GPU 预检报告](kuairand-longseq-agent/reports/analysis/history_value_gpu_preflight_results_v001_v002.md)

### 8.2 冻结的 Adam 与 SGD 充分性结果

| 优化器 | 冻结配置 | 通过 / 6 probes | 最大 regret | 如何读这一行 |
|---|---|---:|---:|---|
| Adam | \(\eta=0.03\), 30 steps | 0 / 6 | 0.006750 | 缺口大于各 probe 允许值；30 次全批更新不足。 |
| Adam | \(\eta=0.03\), 100 steps | 6 / 6 | 0.000540 | 每个 BL1/BL2 probe 都过阈值；这是**最少**充分 checkpoint，故冻结。 |
| Adam | \(\eta=0.03\), 300 / 1,000 steps | 6 / 6 | 0.000218 / 0.000040 | 更接近参考解，但没有理由以更多计算覆盖已预定的“最小充分”选择。 |
| 全批 SGD, momentum=0 | \(\eta=0.001\)–0.1，最多 3,000 steps | 0 / 6 | 0.068308 至 0.034773 | 在这组预冻结网格内，均离参考解过远。 |

最好的冻结 SGD（\(\eta=0.1\), 3,000 steps）仍让 BL1 regret 为 0.034773；相对于上例 0.002732 的允许值约为 **12.7 倍**，BL2 也没有通过。若把它带入 Validation，会把“历史带来什么”与“两个模型根本没有被同等优化”混在一起。表中的 `6` 不是六个用户或六个随机种子，而是六个合同固定的 Train-only BL1/BL2 probe。

因此决策是：**Adam 进入 v010；原 SGD 路线冻结。**这不等于数学上证明 SGD 永远不行，也不等于“Adam 比 SGD 科学上更优”；只表示原已冻结 SGD 协议没有达到可以进入后置评估的数值充分性标准。任何 `lr=1.0`、更多步数、动量或衰减的新 SGD 尝试，都必须作为新合同重新在全部 Train probes 上验证，不能追认改写本次结论。[GPU 预检报告](kuairand-longseq-agent/reports/analysis/history_value_gpu_preflight_results_v001_v002.md)

[返回导航](#toc)

<a id="three-rounds"></a>

## 9. 三轮版本化实验：问题、数据权限、方法与结论的变化

### 9.1 三轮的关系

```text
Train-only 粗筛与优化器充分性
        ↓（Adam 冻结；Validation 解锁）
v010：H2 是否有离线排序增量？
        ↓（发现 random 域基准率迁移）
v011：冻结排序在目标域能否得到可信概率？
        ↓（校准后仍检验受限适配价值）
v012：小幅、可回滚的目标域重训练能否进一步改善？
```

| 版本 | 核心问题 | 保持冻结的部分 | 唯一主要变化 | 证据等级 |
|---|---|---|---|---|
| v010 | H2 是否比静态 BL1 更有价值？ | 数据治理、BL1 静态项、H2 定义、Adam \(0.03,100\) | 按 Validation → sealed → random 顺序开放数据。 | 确认性离线阶段链。 |
| v011 | random 域的绝对概率能否修复而不损失排序？ | v010 BL2、特征、历史、Adam 结果 | 目标域选择的单调校准器。 | post-audit held-out temporal test。 |
| v012 | 在保持可回滚的前提下能否超越 `OLD_BL2+v011`？ | 前代模型、目标定义、配对比较与校准健康门 | 带 tether 的目标域适配及独立校准。 | post-audit temporal replay。 |

所有表都应按“同一阶段、同一目标行、同一基线”的纵向比较阅读，不能跨行把 v010 standard 的 32% 正例率与 v011/v012 random 的约 8% 正例率直接比绝对 Log Loss。AP / event-gAUC 越大越好，Log Loss / Brier / ECE 越小越好；它们的定义、是否衡量排序或概率、以及 CI 的簇 bootstrap 含义在 [§4.2](#screening) 已给出。v010 的 CI 是配对用户簇 bootstrap；v011/v012 的差异也以同目标行的用户簇配对重抽样报告。故“CI 不跨 0”只说明该**指定阶段和指定对比**有稳定差异，绝不自动等价于跨域、线上或因果效果。

#### 9.1.1 v011 与 v012 的切分边界在行级上精确对齐

三轮之间最容易被质疑的一点是："v012 是不是换了个切法，把对自己有利的数据切出来了？"
答案是否定的，而且可以逐行核对。v011 与 v012 使用同一个 random 目标域，
两者的切分边界**完全同界**：

```text
random 域 canonical 总量                                      43,027 行
├─ 前置区间（04-22 → 05-02）                                  19,275 行
│   v011:  calibration fit  8,731 + selection 10,544        = 19,275
│   v012:  adaptation train 11,999 + calibration  7,276     = 19,275   ← 边界完全相同
└─ 后置区间（05-03 → 05-08）                                  23,752 行
    v011:  held-out-to-calibrator test                       = 23,752
    v012:  model selection 11,353 + final replay 12,399      = 23,752   ← 对同一区间的再切分
```

这带来两个可核验的性质：

| 性质 | 为什么重要 | 核验入口 |
|---|---|---|
| `OLD_BL2 + v011` 与 `NEW_BL2` 的**信息量对等** | v011 校准器所用数据与 v012 适配+校准所用数据在时间上同界，两个候选看到的历史信息完全相同。§9.5 的对比因此是公平对照，而不是"新模型多看了几天"。 | 各段 `split_audit.json` 的 `rows` / `date_min` / `date_max` |
| v012 的最终回放段**从未参与任何选择** | 05-06 → 05-08 的 12,399 行在 v011 报告时点之后才被单独切出；v012 的 C1/C2/C3 选择只用 05-03 → 05-05 的 11,353 行。 | `outputs/model_selection/split_audit.json` 与 `outputs/final_temporal_replay_test/split_audit.json` |

同时必须说清楚它的**代价**：v012 的 model-selection 段（11,353 行）取自
v011 曾报告过的 held-out 区间。这在时序上完全合法（v011 早已冻结并发布），
但意味着**不能把 §9.4 的 v011 held-out 结果与 §9.5 的 v012 选择结果当作两份独立证据相加**——
它们共享数据。两者只能各自在自己的段上解读，这也是 §10.1 拒绝合并三轮指标的原因之一。

### 9.2 v010：H2 历史价值的三阶段确认

所有增量均为 \(\text{ADAM\_BL2}-\text{ADAM\_BL1}\)，主分析采用同目标行的 **warm users**（该用户在目标点已有允许历史），并以 2,000 次**配对用户簇 bootstrap**估计 95% CI。每次重抽样抽的是用户及该用户的全部目标行，并对 BL1/BL2 共同重算，因而区间针对模型差异而非两个独立样本。正的 AP/event-gAUC 和负的 Log Loss/Brier 表示 BL2 更好。

| 阶段 | 主分析行 / 用户 | \(\Delta\)AP (95% CI) | \(\Delta\)event-gAUC (95% CI) | \(\Delta\)Log Loss (95% CI) | \(\Delta\)Brier (95% CI) | 结论 |
|---|---|---|---|---|---|---|
| Validation v007 | 886,452 / 902 | +0.035626 [0.031302, 0.040028] | +0.038450 [0.034904, 0.041906] | -0.021001 [-0.022926, -0.019150] | -0.008278 [-0.009120, -0.007494] | 4/4 天为正，pass。 |
| sealed v008 | 4,401,690 / 974 | +0.040651 [0.036937, 0.044402] | +0.046205 [0.043415, 0.049011] | -0.023128 [-0.024797, -0.021537] | -0.009028 [-0.009722, -0.008372] | 17/17 天为正，pass。 |
| random v010 | 42,372 / 983 | +0.026188 [0.018296, 0.034667] | +0.059032 [0.044595, 0.073569] | -0.009589 [-0.019104, 0.000062] | -0.001654 [-0.005735, 0.002461] | 排序迁移 pass；**绝对概率门 fail**（见 §9.3）。 |

v010 预注册历史门要求 \(\Delta\mathrm{AP}\ge0.005\)、其 95% CI 下界大于 0、\(\Delta\)event-gAUC 不低于 0、\(\Delta\)Log Loss/\(\Delta\)Brier 不高于 0，且不发生大面积饱和。三个阶段的 2,000/2,000 bootstrap replicate 均有效。

阅读这张表时，`4/4` 与 `17/17` 是按天切片中 \(\Delta\)AP 为正的天数，不是 4 次或 17 次独立重训；它用于发现单日反向退化，不能代替 CI。random 的 AP 增量为最小效应 0.005 的约 5.24 倍；但 random 中概率损失的置信区间跨 0。这允许说“历史排序迁移、概率损失点估计方向改善”，**不**允许说“random 上概率损失已被统计确定改善”。

### 9.3 v010 暴露的 random 基准率迁移（已修正基线解释）

random warm 正例率为 **8.3971%**，远低于 standard 训练/验证中的约 31.5%–32.1%。表中的 BL0 是**从 standard 训练阶段冻结**的常数概率参考，不是用 random 标签现算的 8.3971%“目标域真均值”；若当时改用 random 均值，就会使用审计标签重设基线，破坏 v010 的 untouched audit 身份。也正因此，BL0 在 random 的 AP 等于正例率、event-gAUC 为 0.5，却因预测尺度偏高而有较差的 Log Loss/Brier/ECE。

| random warm 模型 | 预测/参考概率来自哪里 | AP（排序） | event-gAUC（用户内排序） | Log Loss（概率） | Brier（概率） | ECE20（校准偏差） | 应如何解释 |
|---|---|---:|---:|---:|---:|---:|---|
| BL0 | v010 前在 standard train 冻结的全局常数；无 random refit。 | 0.083971 | 0.500000 | 0.448396 | 0.132386 | 0.235512 | 常数分数没有排序能力；AP 与随机域正例率一致是 ties 的正常结果。它是冻结概率尺度的对照，**不是** random 域最优常数校准器。 |
| ADAM_BL1 | 仅 standard 训练的静态模型；无 random refit。 | 0.170240 | 0.556485 | 0.516692 | 0.169610 | 0.293838 | 静态特征能排出部分正例，但概率总体偏高，三项概率健康均比冻结 BL0 更差。 |
| ADAM_BL2 | 与 BL1 同一冻结训练；仅增加 H2 历史。 | 0.196428 | 0.615517 | 0.507102 | 0.167955 | 0.282172 | 相对 BL1 的排序和概率点估计更好；但相对冻结 BL0 仍不具部署级概率健康。 |

所以该表必须同时读出两件不矛盾的事：第一，BL2 相对 BL1 的历史**排序增量通过**（见 §9.2）；第二，BL1/BL2 的绝对概率尺度在 random 域不合格。它引出 v011 的独立时间校准，而不是允许用 random 标签偷偷调 v010 后再称其为 untouched audit。v011 随后在合法的 calibration-fit/selection 段学习目标域截距，并把最后 23,752 行留作测试。

### 9.4 v011：目标域校准

v011 将目标域数据顺序分成 calibration fit、selection 与 held-out test，避免在最终测试段选择校准器。

| split | 行数 | 用户 | 正例 | 正例率 |
|---|---:|---:|---:|---:|
| calibration fit | 8,731 | 645 | 686 | 7.8571% |
| calibration selection | 10,544 | 844 | 872 | 8.2701% |
| held-out-to-calibrator test | 23,752 | 967 | 2,063 | 8.6856% |

`fit` 只允许估计校准参数，`selection` 只允许在预先列出的校准族中选择，最后一段只用于一次报告；同一用户跨时段出现不改变按时间防止未来标签进入过去参数的约束。三类候选都通过 eligibility；selection 的完整读数如下。M3 的 Log Loss 虽比 M2 低 0.000016，但在冻结 tie 规则下没有足够优势覆盖“只调整截距”的更简单、可解释选择：

| selection 校准族 | Log Loss | Brier | ECE20 | mean \(\hat p\) | eligibility | 它改变什么 |
|---|---:|---:|---:|---:|---|---|
| M1 `prior_shift` | 0.262257 | 0.072477 | 0.033304 | 0.115979 | pass | 用先验比例作整体平移。 |
| M2 `intercept_only` | 0.255865 | 0.070951 | 0.004318 | 0.080861 | pass / **chosen** | 冻结正斜率，只用目标域标签拟合截距。 |
| M3 `platt` | 0.255849 | 0.070952 | 0.003854 | 0.080871 | pass | 同时放开斜率与截距；数值差异在选择容差内。 |

最终 held-out 结果如下。`mean \(\hat p\)` 是这一段所有预测概率的平均值，用于直接对照真实正例率 8.6856%，不是新的排序指标：

| 指标 | 原始 BL2 | M2 校准后 | 差异 / 95% CI | 读法 |
|---|---:|---:|---|---|
| AP | 0.197730 | 0.197730 | 0 | 严格保序，排序不变。 |
| event-gAUC | 0.623640 | 0.623640 | 0 | 严格保序，排序不变。 |
| Log Loss | 0.512076 | 0.268939 | -0.243137 [-0.258633, -0.226836] | 概率质量显著改善。 |
| Brier | 0.169978 | 0.074827 | -0.095151 [-0.102219, -0.087959] | 概率质量显著改善。 |
| ECE20 | 0.281335 | 0.006724 | — | 校准描述大幅改善。 |
| mean \(\hat p\) | 0.368191 | 0.082333 | — | 从明显高于真实 0.086856 的冻结尺度，回到接近目标域基准率；不是要求逐行概率都正确。 |

v011 的正确结论是：**在该后审计的目标域测试中，独立校准使概率更可信且不触碰排序；它不改变 v010 的历史价值结论。**[v011 结果](kuairand-longseq-agent/experiments/bl2_target_domain_calibration_v011/results_v011.md) · [final decision](kuairand-longseq-agent/experiments/bl2_target_domain_calibration_v011/outputs/final_decision.json)

### 9.5 v012：目标域成对重训练

v012 从 sealed BL1/BL2 暖启动；它在每个候选中同时保留 NEW_BL1（无 H2）和 NEW_BL2（有 H2），所以可同时检查“重训练本身有价值吗”与“历史项在重训练后仍有增量吗”。三个候选皆通过 selection eligibility，按冻结规则选择 `C2_balanced`。它的四段时间结构与最终结果是：

| split | 行数 | 用户 | 正例 | 作用 |
|---|---:|---:|---:|---|
| target adaptation train | 11,999 | 760 | 947 | tether 适配。 |
| target calibration | 7,276 | 778 | 611 | 为 NEW_BL1/NEW_BL2 独立校准。 |
| model selection | 11,353 | 875 | 1,024 | 在 C1/C2/C3 中冻结选择。 |
| final temporal replay test | 12,399 | 857 | 1,039 | 最后一次回放评估。 |

模型选择段没有查看 final replay 标签。`ΔAP BL2-BL1`/`ΔLogLoss BL2-BL1` 检查 H2 在适配后仍有嵌套增量；`ΔAP NEW-OLD` 检查重训练相对旧 BL2+v011 是否值得保留；`NEW_BL2 Log Loss` 用于选择候选的概率质量比较：

| selection 配置 | H2 \(\Delta\)AP BL2−BL1 | H2 \(\Delta\)Log Loss BL2−BL1 | \(\Delta\)AP NEW−OLD | NEW_BL2 Log Loss | eligibility / 结论 |
|---|---:|---:|---:|---:|---|
| C1 `conservative` | +0.029004 | -0.008959 | +0.007893 | 0.274048 | pass。 |
| C2 `balanced` | +0.040310 | -0.010101 | +0.021058 | **0.271899** | pass / **chosen**。 |
| C3 `aggressive` | +0.038384 | **-0.013004** | **+0.022749** | 0.274999 | pass；排序增量略高，但目标 Log Loss 较 C2 差，未被选择。 |

final replay 表同时放入概率参照、旧的已校准候选、适配后的无历史对照和适配后的 H2 模型。这里 `mean \(\hat p\)` 应接近该段真实正例率 0.083797，但接近并不代替逐行校准或排序能力：

| final replay 模型 | AP | event-gAUC | Log Loss | Brier | ECE20 | mean \(\hat p\) | 应如何读 |
|---|---:|---:|---:|---:|---:|---:|---|
| TARGET_BL0 | 0.083797 | 0.500000 | 0.288005 | 0.076784 | 0.002967 | 0.080830 | 目标域常数概率参照；无排序能力但平均刻度健康。 |
| OLD_BL2 + v011 | 0.187896 | 0.621353 | 0.263925 | 0.072850 | 0.006645 | 0.081573 | 旧部署候选，给新模型一个严格同段对照。 |
| NEW_BL1 | 0.176476 | 0.613925 | 0.269303 | 0.073613 | 0.006894 | 0.077465 | 适配后的静态嵌套基线。 |
| NEW_BL2 + v012 calibrator | 0.204928 | 0.645537 | 0.260618 | 0.072098 | 0.006200 | 0.079689 | 最佳 AP/gAUC 和概率损失；是候选而非已发布模型。 |

关键的同目标行对比为 `NEW_BL2 − OLD_BL2 + v011`：

$$
\Delta\mathrm{AP}=+0.017032\;[0.005265,0.031319],
$$

$$
\Delta\mathrm{event\text{-}gAUC}=+0.024183\;[0.007412,0.041419],
$$

$$
\Delta\mathrm{LogLoss}=-0.003307\;[-0.005145,-0.001725],
\qquad
\Delta\mathrm{Brier}=-0.000752\;[-0.001243,-0.000333].
$$

所有 required gates 通过；H2 AP 在 3/3 天为正（要求至少 2 天）。结论是 `retraining_adds_value`，并给出工程建议 `deploy_NEW_BL2_with_v012_calibrator_pending_new_data_confirmation`。[v012 结果](kuairand-longseq-agent/experiments/bl2_target_domain_retraining_v012/results_v012.md) · [final decision](kuairand-longseq-agent/experiments/bl2_target_domain_retraining_v012/outputs/final_decision.json)

四个配对 CI 都不跨 0，说明在这一次最终时间回放中，新 BL2 相对旧 BL2+v011 同时提升排序（AP、event-gAUC）并降低概率损失（Log Loss、Brier）。这仍是**有限证据**：它没有检验新的独立日期、未观测曝光策略、在线干预效果或长期稳定性，因此决定词必须保持为 `pending_new_data_confirmation`。

#### 9.5.1 最终回放的逐日 ΔAP 呈衰减趋势

`positive_history_AP_days: 3`（门槛 ≥2）是一个通过的门，但汇总数字掩盖了一个
对部署决策更重要的信息：**增量在三天内单调衰减**。下表由
`outputs/final_temporal_replay_test/daily_metrics.csv` 逐日 AP 相减得到：

| 日期 | 行数 | 正例 | `OLD_BL2+v011` AP | `NEW_BL2` AP | 逐日 ΔAP |
|---|---:|---:|---:|---:|---:|
| 2022-05-06 | 4,290 | 371 | 0.179321 | 0.211508 | **+0.032187** |
| 2022-05-07 | 3,885 | 318 | 0.203874 | 0.225166 | **+0.021292** |
| 2022-05-08 | 4,224 | 350 | 0.190282 | 0.194420 | **+0.004138** |

目标域适配的最后一天是 2022-04-29，因此三个回放日距适配时点分别是 7、8、9 天。
增量从 `+0.0322` 降到 `+0.0041`，到第三天已接近 0。

如何解读，需要同时给出两面：

- **不能过度解读。** 每天只有约 4,000 行、320–370 个正例，单日 AP 的抽样噪声很大；
  本项目没有对逐日差值做 bootstrap，因此这三个点没有 CI，**三点不足以拟合衰减率**。
  池化的 `ΔAP = +0.017032 [0.005265, 0.031319]` 仍是该段的主结论。
- **也不能忽略。** 这个方向与目标域适配的机制预期一致：适配学到的是"适配窗口附近"的
  特征—标签关系，距离越远越失效。它直接关系到部署时的 **retrain 频率**，
  是新数据确认必须回答的问题。

因此，§12.1 的新数据确认合同应当**显式包含一个足够长的回放窗口**（不是 3 天），
并预注册"逐日增量对 time-since-adaptation 的回归"作为次要终点，
用来估计衰减速率与合理的重训周期。在此之前，不应假设该增量在部署中会长期保持。

[返回导航](#toc)

<a id="conclusions"></a>

## 10. 最终比较、验证内容与边界

### 10.1 为什么不能把三轮的指标粗暴合并

v010 的三个阶段覆盖不同时间/曝光域，且 sealed 与 random 时间重叠、用户可能重叠；v011/v012 又是 post-audit 的目标域切分。因此不能把所有 \(\Delta\)AP 拼成单一的“总置信区间”，也不能说 v012 推翻或替代 v010。正确阅读方式是：

| 证据链 | 验证的命题 | 状态 |
|---|---|---|
| v010 Validation → sealed | H2 是否在两个连续 standard 时间窗稳定提高离线预测？ | 支持。 |
| v010 random audit | 已冻结的 H2 排序增量能否迁移到 random exposure？ | 支持；绝对概率尺度未直接部署。 |
| v011 | 在目标域，能否不改排序而修复概率尺度？ | 支持，post-audit。 |
| v012 | 受约束目标域适配能否比旧 BL2+校准再增加价值？ | 支持，post-audit replay。 |

### 10.2 可以说与不能说

| 可以说 | 不能说 |
|---|---|
| 冻结 GPU Adam 下，H2 严格用户总体历史相对相同静态基线有稳定离线排序增量。 | 模型已证明线上观看时长、留存或商业指标提升。 |
| 排序增量在 standard 与 random exposure audit 都出现。 | random 覆盖任意策略、全平台候选池，或可做无假设精确 OPE。 |
| v011 校准在后审计目标域中修复概率质量且保序。 | v011 是 untouched random audit 或新的 pristine confirmatory test。 |
| v012 在最终 temporal replay 中增加排序和概率价值。 | v012 已完成新数据确认，可以无审批上线。 |
| `NEW_BL2 + v012 calibrator` 是当前最强工程候选。 | Adam 以外的优化器同样稳健；原 SGD 已被确认失败或成功。 |
| H2 是用户总体历史状态。 | H2 证明用户喜欢某标签/作者/内容语义；这需要 H3 交互合同。 |
| v013 已登记序列模型设计与 5-seed 门。 | Gold、神经序列训练或新的标签访问已授权/完成。 |

### 10.3 推荐的部署/研究姿态

如果目标是工程试运行，应把 `NEW_BL2 + v012 calibrator` 当作**可回滚候选**：保留 BL1/BL0 回退路径，监控 exposure domain、历史深度、warm/cold video、duration-valid 状态、AP 代理指标与概率健康。若目标是科学机制，应另建 H3 用户×作者/标签/内容增量合同，并以当前 BL2 为固定基线；不能把 H2 的成功改写为内容偏好结论。

[返回导航](#toc)

<a id="agent-guidance"></a>

## 11. 这份证据如何指导 NanyangYS Agent

### 11.1 Agent 的正确角色：控制平面，而非训练或自动发布器

归档中的 NanyangYS Agent 定义了 Manager、Data Auditor、Feature Miner、Causal Evaluator、Safety Reviewer、Feature Publisher 六个角色。它的核心价值是把合同、泄漏审计、证据准入、主张边界、批准与回滚串成 fail-closed 工作流；不是重新清洗 Silver、在本机训练，或自动把 GPU 摘要提升为发布结论。[Agent README](archive/external-projects/nanyangys-agent-20260816/source/README.md)

| 本项目可提供的输入 | Agent 可以做什么 | Agent 不可做什么 |
|---|---|---|
| Silver run、row reconciliation、SHA-256、统计表 denylist | Data Auditor 可核验数据锚点与禁止重洗。 | 不能自行读 Raw/Silver 后重清洗或发现新结论。 |
| `history_time < target_time`、批次先预测后更新、H2 定义 | Feature Miner/Data Auditor 可提出并审计 point-in-time H2 feature spec。 | 不能把当前反馈、同批反馈或 future label 写入特征。 |
| v010 同目标行 BL1/BL2、2,000 次用户簇 bootstrap | Causal Evaluator 可审查“static vs strict statistical history”的离线证据。 | 不支持短/长神经序列模型、线上因果或全库召回主张。 |
| v011 校准 | 可将“部署域校准需独立选择、冻结并验收概率健康”写入控制规则。 | 不得把 post-audit 校准伪装成 untouched validation。 |
| v012 可回滚模型状态与 replay 结果 | Feature Publisher 可形成 *pending confirmation* 候选包。 | 不得自动发布；必须保留新数据与人工审批门。 |

### 11.2 当前归档 Agent 不能直接接收本项目目录的原因

这不是缺点，而是应保留的安全门。归档 Agent 的当前 `agent_system_v001.yaml`：

1. 六类 `expected_provenance` SHA-256 都是 `null`，因此消费端尚未冻结可接受的外部证据身份；
2. 请求的模型集合是 `static_baseline`、`strict_statistical_history`、`short_sequence`、`long_sequence` 四个模型，而 v010 只覆盖前两个；
3. 消费端要求独立 Evidence manifest，不能把生产端 `run_manifest.json` 或 `artifact_hash_manifest.json` 直接塞进去；
4. v011/v012 的 `post_audit` 证据等级不应混入其当前要求 `validation` 的准入路径。

`EvidenceAdmissionPolicy` 会严格校验 provider 类型、六类 provenance digest、scope/tier、模型集合、制品数量/大小及每件制品的 SHA-256。缺少任何一项都应停在 `waiting_for_evidence`，而不是降低门槛。[消费端合同](archive/external-projects/nanyangys-agent-20260816/source/configs/agent_system_v001.yaml) · [准入实现](archive/external-projects/nanyangys-agent-20260816/source/src/kuairand_longseq/evidence/admission.py)

### 11.3 正确的接入路径（不修改归档原件）

| 步骤 | 要创建/冻结的东西 | 关键约束 |
|---|---|---|
| 1 | 新的版本化 Agent Evidence handoff | 归档目录保留 provenance；在当前项目或新的集成工作区建立 successor，不直接改写 archive。 |
| 2 | 单一 v010 Validation scope | 映射 `ADAM_BL1 → static_baseline`、`ADAM_BL2 → strict_statistical_history`；明确“不覆盖 sequence 模型”。 |
| 3 | 消费侧 `research_evidence_manifest_v001.yaml` | 从 v010 合同、审批、代码/输入/目标 manifest 和 validation 制品生成；路径相对 manifest，逐项固定 size/SHA-256。 |
| 4 | 新的 Agent 消费合同 | 在运行前 pin 住 contract、code、input、model-config、authorization、target 六类摘要；`required_models` 缩窄到实际覆盖的两个。 |
| 5 | 分层登记 v011/v012 | 作为 post-audit 工程扩展证据单独保存，不改名为 validation/sealed tier。 |
| 6 | 审批与发布 | 即使准入通过，也仅能进入 `waiting_for_approval`；新数据确认和人工批准后才可发布。 |

这让 Agent 获得最有价值的能力：它能拒绝时间泄漏、证据篡改、模型集不完整和越界主张，而不是把“有报告文件”误当成“可以发布”。

[返回导航](#toc)

<a id="evidence-index"></a>

## 12. 后续工作与证据索引

### 12.1 下一步优先级

1. **不重洗、不覆盖**：保持 Raw、Silver、v010/v011/v012 合同和哈希产物的当前路径。
2. **新数据确认**：若目标是部署，冻结 `NEW_BL2 + v012 calibrator`，在真正独立的新数据上复测排序、概率、分布漂移与切片健康。

   **必须先认清一个硬约束：KuaiRand-1K 的 random 曝光域已被本轮实验 100% 消耗，包内不存在可用于确认的剩余 random 数据。**

   ```text
   random 域 canonical 总量                                            43,027 行
   v011 用掉   fit 8,731 + selection 10,544 + held-out 23,752       = 43,027   (100%)
   v012 用掉   adapt 11,999 + calib 7,276 + sel 11,353 + replay 12,399 = 43,027   (100%)
   剩余未被任何拟合、选择或评估触碰的 random 行                          0 行
   ```

   因此"新数据确认"**不可能通过在本数据集内重新切分实现**。任何再切分都会重用
   已参与过拟合或选择的行，产出的将是第四次 post-audit replay，而不是确认性证据。
   合法的确认路径只有三条：

   | 路径 | 说明 | 需要先做什么 |
   |---|---|---|
   | A. 换数据集 | 使用 KuaiRand-Pure 或 KuaiRand-27K 的独立 random 曝光域 | 新的数据合同、Silver 清洗、特征词典与 `cat_user`/`cat_video` 的跨数据集映射策略（用户/视频 ID 不通用，这是主要工程量） |
   | B. 新日志 | 取真实线上新时间窗的曝光日志 | 数据获取授权；本课程/研究设定下通常不可得 |
   | ~~C. 改口径~~ | ~~在 late standard 的未使用日期上做时间外确认~~ | **已排除。**sealed v008 的 `daily_metrics.csv` 覆盖 2022-04-22 至 05-08 全部 17 天，late standard 域同样没有未使用日期。 |

   在完成 A/B/C 之一之前，`NEW_BL2 + v012 calibrator` 的状态必须保持
   `pending_new_data_confirmation`，且**不应向外表述为"只差最后一步验证"**——
   那一步在当前数据条件下尚不具备执行前提。
3. **正式 Gold**：如需序列模型，先批准 Gold 合同、sample_id、allowlist/denylist、时间线复算与 manifests；不能以本报告代替 Gold。
4. **机制/序列研究**：H3 用户×内容历史或 v013 DIN/MLP 必须在 Gold 与 5-seed 前置门完成后才执行。
5. **Agent 集成**：生成独立消费侧 Evidence manifest 与 digest pinning，保持 fail-closed 到证据和批准齐备。

### 12.2 索引：事实、公式、结果与审计入口

| 编号 | 证据/文件 | 作用 |
|---|---|---|
| E1 | [工作区规范](WORKSPACE_CONVENTIONS.md) | 本汇报放入 `deliverables/` 的分类依据；历史证据不移动。 |
| E2 | [KuaiRand-1K 数据集 README](KuaiRand-1K/README.md) | 数据集背景、原始字段、标签公式与 `tab` 域。 |
| E3 | [正式 Silver 清洗报告](kuairand-longseq-agent/reports/generated/silver_cleaning_report.md) | 行数对账、质量规则、标签率与验收门。 |
| E4 | [项目交接文档](kuairand-longseq-agent/PROJECT_HANDOFF.md) | 数据合同、清洗边界、Gold 设计与禁止事项。 |
| E5 | [Train-only 关联报告](kuairand-longseq-agent/reports/analysis/train_association_report_v002.md) | 粗筛、数据分布和假设收敛。 |
| E6 | [Gate 2B 基线结果](kuairand-longseq-agent/reports/analysis/gate2b_baseline_results_v002.md) | 早期排序信号与绝对概率 sanity 失败。 |
| E7 | [GPU 优化器预检](kuairand-longseq-agent/reports/analysis/history_value_gpu_preflight_results_v001_v002.md) | Adam/SGD objective-regret 充分性对比。 |
| E8 | [v010 最终证据总结](kuairand-longseq-agent/reports/analysis/history_value_final_evidence_summary_v010.md) | Validation、sealed、random 的主结论和边界。 |
| E9 | [预测标签、模型公式、设计由来与可行性](预测标签、模型公式、设计由来与可行性.md) | 本文 BL0/BL1/BL2、校准和 tether 公式的详细依据。 |
| E10 | [v011 结果](kuairand-longseq-agent/experiments/bl2_target_domain_calibration_v011/results_v011.md) / [决策](kuairand-longseq-agent/experiments/bl2_target_domain_calibration_v011/outputs/final_decision.json) | 目标域校准的切分、指标与 post-audit 边界。 |
| E11 | [v012 结果](kuairand-longseq-agent/experiments/bl2_target_domain_retraining_v012/results_v012.md) / [决策](kuairand-longseq-agent/experiments/bl2_target_domain_retraining_v012/outputs/final_decision.json) | 受约束重训练的选择、最终回放与工程建议。 |
| E12 | [GPU 导入与状态对账](kuairand-longseq-agent/reports/post_gpu_import_reconciliation_2026-08-24.md) | 导入文件级完整性、当前 Silver 一致性和冲突处理。 |
| E13 | [NanyangYS Agent README](archive/external-projects/nanyangys-agent-20260816/source/README.md) | Agent 的当前控制平面、角色和 fail-closed 状态。 |
| E14 | [sealed 模型跨进程重建诊断](kuairand-longseq-agent/reports/analysis/sealed_model_reconstruction_diagnostic_v010.md) | v011/v012 暖启动所用重建权重的数值等价性证据（见 §7.4a）。 |
| E15 | [完整性勘误](INTEGRITY_ERRATA.md) | 分发前独立评审的完整记录：曾因 Ubuntu 导入丢失的 7 个 v007/v008 制品（**已补传归位，v007 17/17、v008 18/18、v011 前置 6/6 复算通过**），以及刻意保留并披露的重建模型、原始运行环境路径等事实。 |

### 12.3 本报告自身的验证边界

本文件是对 E1–E13 的可读整合：未读取或修改 Raw 数据、未运行模型、未生成新图或新指标、未修改 Silver、未修改已发布分析报告与哈希化产物。所有数值都应以相应合同、manifest、`final_decision.json` 和上述原始报告为准。

[返回导航](#toc)
