"""Finalize Gate 2B governance after a conservative post-release audit.

Scientific predictions and metric artifacts must still match the hashes written
by the original release.  Only human-facing documents, the figure, audit files,
and the governance decision are revised.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "reports/generated/gate2b_baselines_v002"
MANIFEST = GENERATED / "run_manifest.json"
IMMUTABLE_SCIENCE = {
    "gate2b_feature_matrix.parquet",
    "feature_manifest.json",
    "target_row_manifests.csv",
    "search_trial_manifest.csv",
    "search_predictions_origin_2022-04-11.parquet",
    "search_predictions_origin_2022-04-14.parquet",
    "search_predictions_origin_2022-04-17.parquet",
    "search_metrics.csv",
    "selected_models.json",
    "daily_predictions.parquet",
    "daily_metrics.csv",
    "pooled_and_slice_metrics.csv",
    "paired_user_cluster_bootstrap.csv",
    "calibration_bins.csv",
    "usage_ledger.csv",
}
POSTRELEASE_FILES = [
    GENERATED / "probability_sanity_audit.csv",
    GENERATED / "feature_promotion_manifest.json",
    ROOT / "reports/analysis/gate2b_baseline_results_v002.md",
    ROOT / "reports/figures/gate2b_baseline_results_v002.png",
    ROOT / "reports/gate2b_checkpoint_2026-08-15.md",
    ROOT / "reports/gate2b_probability_sanity_audit_2026-08-15.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    original = {Path(item["path"]).name: item for item in payload["outputs"]}
    for name in sorted(IMMUTABLE_SCIENCE):
        item = original.get(name)
        if item is None:
            raise RuntimeError(f"original release manifest is missing {name}")
        path = ROOT / item["path"]
        observed = sha256(path)
        if observed != item["sha256"]:
            raise RuntimeError(f"scientific artifact changed after release: {name}")
    by_path = {item["path"]: item for item in payload["outputs"]}
    for path in POSTRELEASE_FILES:
        by_path[relative(path)] = {
            "path": relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    payload["outputs"] = [by_path[key] for key in sorted(by_path)]
    payload["manifest_revision"] = 2
    payload["status"] = "complete_train_only_gate2b_not_advanced_probability_sanity_failed"
    payload["checkpoint_eligible"] = True
    payload["advancement_allowed"] = False
    payload["gates"]["postrelease_absolute_probability_sanity"] = {
        "role": "conservative_postrelease_sanity_review_not_preregistered_gate",
        "passed": False,
        "BL2_minus_BL0_log_loss": 3.642285888711356,
        "BL2_minus_BL0_brier": 0.0579948961610082,
        "BL2_minus_BL0_ece20_descriptive": 0.26742113369493486,
        "failure_action": "do_not_advance_to_sequence_models_or_validation",
    }
    payload["postrelease_governance"] = {
        "scientific_artifacts_rehashed_unchanged": True,
        "model_refit_after_release": False,
        "scope_expansion_after_results": False,
        "documents_and_figure_revised_to_show_probability_sanity_failure": True,
        "audit_manifest_path": "reports/generated/gate2b_baselines_v002/postrelease_audit_manifest.json",
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "manifest_sha256": sha256(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
