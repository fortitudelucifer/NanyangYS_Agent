# JKRec 总览报告入口

本文件是 `JKRec` 包根目录的一页式阅读入口。根目录有两份正式汇报：

| 文件 | 定位 | 用时 |
|---|---|---|
| **[主报告](2026-08-24_kuairand-longseq-governed-research-main-report_v012.md)** | 带 4 张图、按论证顺序组织、数值统一 4 位小数。**新读者从这里开始。** | ~16 分钟 |
| [完整档案](2026-08-24_kuairand-longseq-governed-research-full-report_v012.md) | 字段级大表、逐条制品链接、Agent 集成章节。作主报告的附录。 | ~41 分钟 |

两份数值同源，均以各阶段 `final_decision.json` 与哈希制品为准。

## 一页结论

- 原始 KuaiRand-1K CSV 与冻结 Silver 数据都在本包中；清洗仅可审计、不可覆盖。
- v010 支持：冻结 GPU Adam 下，严格用户总体历史 BL2 相比同静态特征 BL1 有离线排序增量。
- v011 支持：在 post-audit 目标域中，独立截距校准改善概率质量但不改变排序。
- v012 支持：带 tether 的目标域适配在最终 temporal replay 增加价值；仍需新数据确认和人工审批。
- 正式 Gold snapshot 尚未物化；v010–v012 的严格时点特征矩阵只能称为实验级 Gold-equivalent。
- **random 曝光域的 43,027 行已被 v011/v012 100% 消耗**，包内无剩余随机曝光数据；"待新数据确认"无法在本数据集内完成（完整汇报 §12.1）。
- v011/v012 的"从 sealed 暖启动"加载的是**重建**权重（v010 原始权重从未落盘，重建误差 ~1e-7）；见完整汇报 §7.4a 与 `INTEGRITY_ERRATA.md`。

## 按复现目的进入

| 目的 | 先读/先看 |
|---|---|
| 快速判断结论有多硬 | [主报告 §1](2026-08-24_kuairand-longseq-governed-research-main-report_v012.md) + 图 1、图 3 |
| 理解全流程、公式和边界 | [主报告](2026-08-24_kuairand-longseq-governed-research-main-report_v012.md)；需要字段级细节再翻[完整档案](2026-08-24_kuairand-longseq-governed-research-full-report_v012.md) |
| 验证 Raw 与 Silver | [Silver 清洗报告](kuairand-longseq-agent/reports/generated/silver_cleaning_report.md) · `governed_data/formal_silver_snapshot/` |
| 复查 v010–v012 | `kuairand-longseq-agent/reports/analysis/` · `kuairand-longseq-agent/experiments/` · `reports/generated/` 的白名单运行目录 |
| 了解重跑边界 | [项目交接](kuairand-longseq-agent/PROJECT_HANDOFF.md) · [本包范围](PACKAGE_SCOPE.md) |
| 接入 NanyangYS Agent | [Agent 源快照](archive/external-projects/nanyangys-agent-20260816/source/README.md) |

## 分发前必读

| 文件 | 为什么必须先看 |
|---|---|
| [完整性勘误](INTEGRITY_ERRATA.md) | 证据核验的完整记录。**当前无未解决缺口**：曾因 Ubuntu 导入丢失的 7 个 v007/v008 制品已补传归位（v007 17/17、v008 18/18、v011 前置 6/6 通过）；E-2 起为刻意披露的事实，含 v011/v012 暖启动使用重建权重。在引用任何结论前先读。 |
| [变更记录](CHANGELOG.md) | v001 → v002 的全部改动。所有实验数值未变。 |
| [发布环境](requirements-release-v010-v012.txt) | 复现环境以此为准，**不是** `pyproject.toml`。 |

## 完整性校验

```bash
python verify_package.py            # 全量，约 40 秒
python verify_package.py --quick    # 只校验小文件，约 10 秒
```

所有包内文件的 SHA-256 见 [SHA256SUMS.txt](SHA256SUMS.txt)。清单为 PowerShell 格式
（大写 hex + CRLF + 反斜杠），GNU `sha256sum -c` 在 macOS/Linux 上会对每行报 FAILED——
**那是格式不兼容，不是文件被篡改**，请改用上面的脚本。
