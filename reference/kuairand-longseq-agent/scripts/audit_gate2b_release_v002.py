"""Read-only post-release probability sanity audit for Gate 2B."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "reports/generated/gate2b_baselines_v002"
PREDICTIONS = GENERATED / "daily_predictions.parquet"
METRICS = GENERATED / "pooled_and_slice_metrics.csv"
OUTPUT = GENERATED / "probability_sanity_audit.csv"
MANIFEST = GENERATED / "postrelease_audit_manifest.json"
RUN_MANIFEST = GENERATED / "run_manifest.json"
REPORT = ROOT / "reports/analysis/gate2b_baseline_results_v002.md"
FIGURE = ROOT / "reports/figures/gate2b_baseline_results_v002.png"
CHECKPOINT = ROOT / "reports/gate2b_checkpoint_2026-08-15.md"
AUDIT_REPORT = ROOT / "reports/gate2b_probability_sanity_audit_2026-08-15.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    con = duckdb.connect()
    # Keep floating-point aggregate serialization stable across repeated audit
    # runs.  This audit is small enough that deterministic single-threaded
    # aggregation is preferable to parallel reduction-order noise.
    con.execute("SET threads=1")
    rows = con.execute(
        """
        WITH long_form AS (
          SELECT 'BL0' model_id, p_bl0 p FROM read_parquet(?)
          UNION ALL SELECT 'BL1', p_bl1 FROM read_parquet(?)
          UNION ALL SELECT 'BL2', p_bl2 FROM read_parquet(?)
        )
        SELECT model_id, count(*) row_count, avg(p) mean_probability,
               min(p) minimum, quantile_cont(p,0.01) p01,
               quantile_cont(p,0.10) p10, quantile_cont(p,0.50) median,
               quantile_cont(p,0.90) p90, quantile_cont(p,0.99) p99, max(p) maximum,
               avg((p <= 1.0000001e-7)::INT) lower_clip_share,
               avg((p >= 0.99999989)::INT) upper_clip_share,
               avg((p <= 0.01)::INT) probability_le_001_share,
               avg((p >= 0.99)::INT) probability_ge_099_share
        FROM long_form GROUP BY model_id ORDER BY model_id
        """,
        [str(PREDICTIONS)] * 3,
    ).fetchall()
    columns = [item[0] for item in con.description]
    con.close()
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
    with METRICS.open("r", encoding="utf-8", newline="") as handle:
        pooled = {
            row["model_id"]: row
            for row in csv.DictReader(handle)
            if row["slice"] == "all_assessment_rows"
        }
    bl0, bl2 = pooled["BL0"], pooled["BL2"]
    absolute = {
        "BL2_minus_BL0_average_precision": float(bl2["average_precision"])
        - float(bl0["average_precision"]),
        "BL2_minus_BL0_log_loss": float(bl2["log_loss"]) - float(bl0["log_loss"]),
        "BL2_minus_BL0_brier": float(bl2["brier"]) - float(bl0["brier"]),
        "BL2_minus_BL0_ece20": float(bl2["ece20_equal_width"])
        - float(bl0["ece20_equal_width"]),
    }
    manifest = {
        "audit_id": "gate2b_probability_sanity_postrelease_v002",
        "status": "complete_read_only_postrelease_audit",
        "scientific_metrics_observed": True,
        "role": "conservative_postrelease_sanity_review_not_preregistered_gate",
        "decision": "relative_BL2_vs_BL1_stability_pass_but_absolute_probability_sanity_fail",
        "advancement_allowed": False,
        "inputs": [
            {"path": PREDICTIONS.relative_to(ROOT).as_posix(), "sha256": sha256(PREDICTIONS)},
            {"path": METRICS.relative_to(ROOT).as_posix(), "sha256": sha256(METRICS)},
            {"path": RUN_MANIFEST.relative_to(ROOT).as_posix(), "sha256": sha256(RUN_MANIFEST)},
        ],
        "absolute_BL2_minus_BL0": absolute,
        "output": {
            "path": OUTPUT.relative_to(ROOT).as_posix(),
            "size_bytes": OUTPUT.stat().st_size,
            "sha256": sha256(OUTPUT),
        },
        "script": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256(Path(__file__)),
        },
        "governance_outputs": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in [REPORT, FIGURE, CHECKPOINT, AUDIT_REPORT]
        ],
        "scope": {
            "prediction_artifacts_only": True,
            "model_refit": False,
            "silver_recleaned": False,
            "gold_built": False,
            "validation_accessed": False,
            "late_accessed": False,
            "random_accessed": False,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
