#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 KuaiRand 长序列受治理研究报告的图表资产 (v001)。

输入：包内既有的哈希化制品（CSV），不重新计算任何指标。
输出：<包根>/figures/*.png

用法：
    python render_report_figures_v001.py --package-root <JKRec 根目录> --out <输出目录>

设计约束（见 docs/review 的文档评审）：
  * 单轴，不使用双 y 轴；
  * 分类色仅用 blue #2a78d6 / orange #eb6834（对比度通过校验），
    参考线与网格用中性灰，文字不着色；
  * ≥2 个系列必带图例；细线 2px、标记 ≥8px；网格弱化。
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm

# ---------------------------------------------------------------- 样式
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE = "#e4e3df", "#ffffff"

for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK HK"):
    if any(name in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        break
plt.rcParams.update({
    "font.size": 10, "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def read_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def title_block(ax, title, subtitle=None):
    ax.set_title(title, color=INK, fontsize=12.5, fontweight="bold",
                 loc="left", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes,
                color=INK2, fontsize=9.2, va="bottom")


# ---------------------------------------------------------------- 图 1
def fig1_forest(root, out):
    """四个证据阶段的 ΔAP 点估计与 95% CI。

    上组三行是同一对比（ADAM_BL2 − ADAM_BL1）；下组一行是不同对比
    （NEW_BL2 − OLD_BL2+v011），用分隔线与标注隔开，避免被读成一条链。
    """
    G = os.path.join(root, "kuairand-longseq-agent", "reports", "generated")
    E = os.path.join(root, "kuairand-longseq-agent", "experiments")
    src = [
        ("v010 Validation\n886,452 行 / 902 用户",
         f"{G}/history_value_adam_validation_v007/validation/paired_user_cluster_bootstrap.csv",
         "ADAM_BL2_minus_ADAM_BL1"),
        ("v010 sealed standard\n4,401,690 行 / 974 用户",
         f"{G}/history_value_adam_sealed_v008/sealed_test/paired_user_cluster_bootstrap.csv",
         "ADAM_BL2_minus_ADAM_BL1"),
        ("v010 random audit\n42,372 行 / 983 用户",
         f"{G}/history_value_adam_random_v010/random_audit/paired_user_cluster_bootstrap.csv",
         "ADAM_BL2_minus_ADAM_BL1"),
        ("v012 final replay\n12,399 行 / 857 用户",
         f"{E}/bl2_target_domain_retraining_v012/outputs/final_temporal_replay_test/paired_user_cluster_bootstrap.csv",
         "NEW_BL2_minus_OLD_BL2_PLUS_V011"),
    ]
    rows = []
    for label, path, contrast in src:
        for r in read_csv(path):
            if r["contrast"] == contrast and r["metric"] == "average_precision":
                rows.append((label, float(r["point_estimate"]),
                             float(r["ci95_lower"]), float(r["ci95_upper"])))
                break

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ys = [3, 2, 1, -0.35]                      # 第四行下沉，制造视觉分组
    for (label, pt, lo, hi), y in zip(rows, ys):
        c = BLUE if y > 0 else ORANGE
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=3)
        ax.plot([lo, lo], [y - .1, y + .1], color=c, lw=2, zorder=3)
        ax.plot([hi, hi], [y - .1, y + .1], color=c, lw=2, zorder=3)
        ax.plot([pt], [y], "o", ms=9, color=c, mec=SURFACE, mew=1.6, zorder=4)
        ax.text(hi + 0.0016, y, f"{pt:+.4f}  [{lo:.4f}, {hi:.4f}]",
                va="center", ha="left", fontsize=9.2, color=INK)

    ax.axvline(0, color=INK3, lw=1.2, zorder=2)
    ax.axvline(0.005, color=INK3, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.text(0.005, 3.62, " 预注册最小效应 0.005", fontsize=8.6, color=INK2, va="bottom")
    ax.axhline(0.35, color=GRID, lw=1.2)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.2, color=INK)
    ax.set_ylim(-1.05, 4.05)
    ax.set_xlim(-0.004, 0.062)
    ax.set_xlabel("ΔAP（候选模型 − 基线模型），2,000 次配对用户簇 bootstrap 的 95% CI")
    ax.grid(axis="y", visible=False)

    ax.text(0.062, 3.62, "对比：ADAM_BL2 − ADAM_BL1（H2 历史价值）",
            fontsize=8.8, color=BLUE, ha="right", va="bottom", fontweight="bold")
    ax.text(0.062, -0.02, "不同对比：NEW_BL2 −（OLD_BL2 + v011 校准）",
            fontsize=8.8, color=ORANGE, ha="right", va="bottom", fontweight="bold")
    ax.text(0.062, -0.72, "上下两组回答不同问题，不可相加或串成一条链",
            fontsize=8.4, color=INK2, ha="right", va="bottom", style="italic")

    title_block(ax, "H2 历史特征的离线排序增量：四个阶段的 ΔAP 与 95% CI",
                "四个区间下界均大于 0；random 域区间最宽（该域仅 42,372 行）")
    fig.savefig(os.path.join(out, "fig1_delta_ap_forest.png"))
    plt.close(fig)
    return "fig1_delta_ap_forest.png"


# ---------------------------------------------------------------- 图 2
def fig2_reliability(root, out):
    """v011 目标域校准前后的可靠性曲线（20 等宽分箱）。"""
    p = os.path.join(root, "kuairand-longseq-agent", "experiments",
                     "bl2_target_domain_calibration_v011", "outputs",
                     "held_out_to_calibrator_test", "reliability_equal_width_20.csv")
    series = defaultdict(list)
    for r in read_csv(p):
        if not r["rows"] or int(r["rows"]) == 0:
            continue                      # 空分箱（rows=0）无预测/实际值，跳过
        series[r["model_id"]].append(
            (float(r["mean_probability"]), float(r["observed_rate"]), int(r["rows"])))

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.plot([0, 1], [0, 1], color=INK3, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(0.63, 0.60, "完美校准", rotation=39, fontsize=9, color=INK2,
            ha="center", va="center")

    spec = [("original_BL2", "冻结 BL2（未校准）", ORANGE),
            ("selected_calibrator", "M2 截距校准后", BLUE)]
    for key, label, color in spec:
        pts = sorted(series[key])
        xs = [a for a, _, _ in pts]
        ys = [b for _, b, _ in pts]
        ss = [max(28, min(320, n / 9.0)) for _, _, n in pts]
        ax.plot(xs, ys, color=color, lw=2, zorder=3, alpha=.85)
        ax.scatter(xs, ys, s=ss, color=color, edgecolor=SURFACE, linewidth=1.4,
                   zorder=4, label=label)

    ax.axhline(0.086856, color=INK3, lw=1, ls=":", zorder=2)
    ax.text(0.985, 0.093, "该段真实正例率 0.0869", fontsize=8.6, color=INK2, ha="right")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("分箱内平均预测概率")
    ax.set_ylabel("分箱内实际正例率")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9.5, labelcolor=INK,
              handletextpad=.6, borderaxespad=1.1)
    ax.text(0.035, 0.855,
            "ECE20      0.2813 → 0.0067\nLog Loss   0.5121 → 0.2689\nBrier      0.1700 → 0.0748",
            transform=ax.transAxes, fontsize=8.8, color=INK2, va="top",
            family="DejaVu Sans Mono", linespacing=1.6)
    ax.text(0.035, 0.700, "AP 与 event-gAUC 的差值精确为 0",
            transform=ax.transAxes, fontsize=8.8, color=INK2, va="top")
    ax.text(0.5, -0.13, "圆点面积 ∝ 该分箱样本量；held-out 段共 23,752 行 / 967 用户",
            transform=ax.transAxes, fontsize=8.4, color=INK2, ha="center")

    title_block(ax, "v011 目标域校准：概率刻度被修正，排序完全不动",
                "校准前曲线远低于对角线（普遍高估）；校准后贴合对角线")
    fig.savefig(os.path.join(out, "fig2_reliability_v011.png"))
    plt.close(fig)
    return "fig2_reliability_v011.png"


# ---------------------------------------------------------------- 图 3
def fig3_daily(root, out):
    """v012 最终回放的逐日 ΔAP，与池化估计对照。"""
    p = os.path.join(root, "kuairand-longseq-agent", "experiments",
                     "bl2_target_domain_retraining_v012", "outputs",
                     "final_temporal_replay_test", "daily_metrics.csv")
    ap, rows_n = defaultdict(dict), {}
    for r in read_csv(p):
        ap[r["event_date"]][r["model_id"]] = float(r["average_precision"])
        rows_n[r["event_date"]] = int(r["rows"])
    dates = sorted(ap)
    delta = [ap[d]["NEW_BL2"] - ap[d]["OLD_BL2_PLUS_V011"] for d in dates]

    POOL, LO, HI = 0.017032, 0.005265, 0.031319
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.axhspan(LO, HI, color=BLUE, alpha=.10, zorder=1)
    ax.axhline(POOL, color=BLUE, lw=2, zorder=2)
    ax.text(2.42, POOL, f"  池化 ΔAP {POOL:+.4f}\n  95% CI [{LO:.4f}, {HI:.4f}]",
            fontsize=8.8, color=BLUE, va="center")
    ax.axhline(0, color=INK3, lw=1.2, zorder=2)

    xs = range(len(dates))
    ax.plot(xs, delta, color=ORANGE, lw=2, zorder=3)
    ax.scatter(xs, delta, s=110, color=ORANGE, edgecolor=SURFACE, lw=1.6, zorder=4)
    for x, d, dt in zip(xs, delta, dates):
        ax.text(x, d + 0.0022, f"{d:+.4f}", ha="center", fontsize=9.4,
                color=INK, fontweight="bold")
        ax.text(x, -0.0055, f"{rows_n[dt]:,} 行", ha="center", fontsize=8.4, color=INK2)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{d}\n距适配窗口结束 {n} 天" for d, n in zip(dates, (7, 8, 9))],
                       fontsize=9.2, color=INK)
    ax.set_xlim(-0.45, 3.15)
    ax.set_ylim(-0.008, 0.040)
    ax.set_ylabel("逐日 ΔAP（NEW_BL2 − OLD_BL2+v011）")
    ax.grid(axis="x", visible=False)
    ax.text(0.5, -0.30,
            "逐日值未做 bootstrap，无置信区间；每日仅约 4,000 行、320–370 正例，单日噪声大。\n"
            "三点不足以拟合衰减率，但方向与目标域适配的机制预期一致——新数据确认应使用更长回放窗口。",
            transform=ax.transAxes, fontsize=8.4, color=INK2, ha="center", linespacing=1.6)

    title_block(ax, "v012 最终回放：3/3 天为正，但增量在三天内衰减至接近 0",
                "预注册门槛为「至少 2 天为正」——门通过了，趋势值得单独记录")
    fig.savefig(os.path.join(out, "fig3_daily_delta_ap_v012.png"))
    plt.close(fig)
    return "fig3_daily_delta_ap_v012.png"


# ---------------------------------------------------------------- 图 4
def fig4_timeline(root, out):
    """数据域与实验切分的时间轴：展示 v011/v012 的边界对齐与 random 域耗尽。"""
    D0 = 8  # 以 4 月 8 日为 0 点，单位「天」
    def x(m, d):
        return (d - D0) if m == 4 else (30 - D0 + d)

    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    fig.subplots_adjust(bottom=0.26, left=0.135, right=0.985, top=0.86)
    bars = [
        (6.2, x(4, 8),  x(4, 21) + 1, "#c9ddf5", "early standard  5,055,984 行 → Silver 4,992,443", INK),
        (5.2, x(4, 8),  x(4, 17) + 1, BLUE,      "Train（EDA / 优化器充分性 / 拟合）", "#ffffff"),
        (5.2, x(4, 18), x(4, 21) + 1, "#6ea6e6", "Validation", "#ffffff"),
        (3.9, x(4, 22), x(5, 8) + 1,  "#c9ddf5", "late standard  6,657,061 行 → Silver 6,556,501", INK),
        (2.9, x(4, 22), x(5, 8) + 1,  BLUE,      "sealed standard 时间外一次性测试（4,401,690 行 / 17 天）", "#ffffff"),
        (1.5, x(4, 22), x(5, 8) + 1,  "#fbd9c8", "random exposure  43,028 行 → Silver 42,982（canonical 43,027）", INK),
        (0.4, x(4, 22), x(4, 27) + 1, ORANGE,    "fit\n8,731", "#ffffff"),
        (0.4, x(4, 28), x(5, 2) + 1,  "#f2915f", "selection\n10,544", "#ffffff"),
        (0.4, x(5, 3),  x(5, 8) + 1,  "#f7bc9c", "held-out test  23,752", INK),
        (-1.0, x(4, 22), x(4, 29) + 1, ORANGE,   "adaptation  11,999", "#ffffff"),
        (-1.0, x(4, 30), x(5, 2) + 1,  "#f2915f", "calib\n7,276", "#ffffff"),
        (-1.0, x(5, 3),  x(5, 5) + 1,  "#f7bc9c", "selection\n11,353", INK),
        (-1.0, x(5, 6),  x(5, 8) + 1,  "#8a3a12", "final replay\n12,399", "#ffffff"),
    ]
    for y, x0, x1, color, label, tc in bars:
        w = x1 - x0
        h = .62 if "\n" not in label else .74
        ax.add_patch(Rectangle((x0, y - h / 2), w - 0.10, h,
                               facecolor=color, edgecolor=SURFACE, lw=1.8, zorder=3))
        ax.text((x0 + x1) / 2 - 0.05, y, label, ha="center", va="center",
                fontsize=8.0 if "\n" in label else 8.4, color=tc, zorder=4,
                linespacing=1.35)

    ax.axvline(x(5, 3), color=INK3, lw=1.4, ls=(0, (5, 3)), zorder=5)
    ax.annotate("05-03：v011 held-out 与\nv012 selection+replay 的共同边界",
                xy=(x(5, 3), 7.05), xytext=(x(4, 24), 7.35),
                fontsize=8.6, color=INK2, va="center", ha="center", linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=INK3, lw=1.1,
                                connectionstyle="arc3,rad=-0.15"))

    for y, lab in [(6.2, "early standard"), (3.9, "late standard"), (1.5, "random exposure")]:
        ax.text(-1.6, y, lab, ha="right", va="center", fontsize=9.6,
                color=INK, fontweight="bold")
    for y, lab in [(0.4, "└ v011"), (-1.0, "└ v012")]:
        ax.text(-1.6, y, lab, ha="right", va="center", fontsize=9.2, color=ORANGE,
                fontweight="bold")

    ticks = [x(4, d) for d in (8, 12, 16, 20, 24, 28)] + [x(5, d) for d in (2, 6, 8)]
    labs = ["04-08", "04-12", "04-16", "04-20", "04-24", "04-28", "05-02", "05-06", "05-08"]
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=9)
    ax.set_yticks([]); ax.set_ylim(-1.75, 7.8); ax.set_xlim(-10.5, x(5, 8) + 1.8)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)

    fig.text(0.5, 0.135,
             "v011 的 fit+selection（8,731+10,544）与 v012 的 adaptation+calibration（11,999+7,276）均为 19,275 行，边界完全同界；",
             fontsize=8.6, color=INK2, ha="center")
    fig.text(0.5, 0.088,
             "v011 held-out（23,752）= v012 selection+replay（11,353+12,399）。两个候选看到的历史信息量相同，对照公平。",
             fontsize=8.6, color=INK2, ha="center")
    fig.text(0.5, 0.038,
             "random 域 43,027 行已被 100% 消耗——「待新数据确认」无法在本数据集内完成。",
             fontsize=8.6, color=ORANGE, ha="center", fontweight="bold")

    title_block(ax, "数据域、访问顺序与实验切分：v011 与 v012 的边界在行级上精确对齐",
                "浅色条为数据域，深色条为该域上实际使用的切分")
    fig.savefig(os.path.join(out, "fig4_data_timeline.png"))
    plt.close(fig)
    return "fig4_data_timeline.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for fn in (fig1_forest, fig2_reliability, fig3_daily, fig4_timeline):
        name = fn(a.package_root, a.out)
        p = os.path.join(a.out, name)
        print(f"  ✅ {name}  ({os.path.getsize(p):,} bytes)")


if __name__ == "__main__":
    main()
