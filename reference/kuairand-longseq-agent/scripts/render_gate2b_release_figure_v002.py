"""Re-render the frozen Gate 2B figure without fitting or reading source data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
DAILY_ORIGINS = [f"2022-04-{day:02d}" for day in range(11, 18)]


def typed_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, object] = {}
            for key, value in row.items():
                if key in {"origin", "model_id", "slice"}:
                    converted[key] = value
                else:
                    try:
                        converted[key] = float(value)
                    except (TypeError, ValueError):
                        converted[key] = value
            rows.append(converted)
    return rows


def main() -> None:
    generated = ROOT / "reports/generated/gate2b_baselines_v002"
    daily_rows = typed_rows(generated / "daily_metrics.csv")
    slice_rows = typed_rows(generated / "pooled_and_slice_metrics.csv")
    by_model = {
        model: [row for row in daily_rows if row["model_id"] == model]
        for model in ["BL0", "BL1", "BL2"]
    }
    colors = {"BL0": "#9ca3af", "BL1": "#2563eb", "BL2": "#dc2626"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.arange(7)
    labels = [origin[-5:] for origin in DAILY_ORIGINS]
    for model, rows in by_model.items():
        axes[0, 0].plot(
            x,
            [row["average_precision"] for row in rows],
            marker="o",
            label=model,
            color=colors[model],
        )
    axes[0, 0].set_title("A. Daily Train rolling-origin average precision")
    axes[0, 0].set_xticks(x, labels, rotation=30)
    axes[0, 0].set_ylabel("Average precision")
    axes[0, 0].legend()
    delta_ap = [
        by_model["BL2"][index]["average_precision"]
        - by_model["BL1"][index]["average_precision"]
        for index in range(7)
    ]
    axes[0, 1].bar(
        x, delta_ap, color=["#16a34a" if value > 0 else "#dc2626" for value in delta_ap]
    )
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("B. Daily increment: BL2 minus BL1")
    axes[0, 1].set_xticks(x, labels, rotation=30)
    axes[0, 1].set_ylabel("Delta average precision")
    pooled = {
        row["model_id"]: row
        for row in slice_rows
        if row["slice"] == "all_assessment_rows"
    }
    model_x = np.arange(3)
    axes[1, 0].bar(
        model_x - 0.18,
        [pooled[m]["average_precision"] for m in ["BL0", "BL1", "BL2"]],
        0.36,
        label="AP",
    )
    axes[1, 0].bar(
        model_x + 0.18,
        [pooled[m]["user_gauc_event_weighted"] for m in ["BL0", "BL1", "BL2"]],
        0.36,
        label="user-GAUC",
    )
    axes[1, 0].set_xticks(model_x, ["BL0", "BL1", "BL2"])
    axes[1, 0].set_title("C. Pooled discrimination (same rows)")
    axes[1, 0].legend()
    axes[1, 1].bar(
        model_x - 0.18,
        [pooled[m]["log_loss"] for m in ["BL0", "BL1", "BL2"]],
        0.36,
        label="Log Loss",
    )
    axes[1, 1].bar(
        model_x + 0.18,
        [pooled[m]["brier"] for m in ["BL0", "BL1", "BL2"]],
        0.36,
        label="Brier",
    )
    axes[1, 1].set_xticks(model_x, ["BL0", "BL1", "BL2"])
    axes[1, 1].set_title("D. Pooled probability error (lower is better)")
    axes[1, 1].legend()
    axes[1, 1].text(
        0.02,
        0.96,
        "Absolute sanity FAIL: BL1/BL2 worse than BL0",
        transform=axes[1, 1].transAxes,
        va="top",
        color="#b91c1c",
        fontweight="bold",
    )
    fig.suptitle(
        "KuaiRand-1K Gate 2B — ranking stability PASS / probability sanity FAIL",
        fontsize=15,
    )
    output = ROOT / "reports/figures/gate2b_baseline_results_v002.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
