#!/usr/bin/env python3
"""Measure cross-process sealed-model reconstruction drift without opening random data."""

from __future__ import annotations

import csv
import gc
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_history_value_gpu_confirmation_v001 as v1
import run_history_value_adam_confirmation_v004 as v4
import run_history_value_adam_validation_v007 as v7
import run_history_value_adam_random_v009 as v9


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "reports/generated/sealed_model_reconstruction_diagnostic_v010"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/sealed_model_reconstruction_diagnostic_v010.md"
OLD_ADEQUACY = PROJECT_ROOT / "reports/generated/history_value_adam_sealed_v008/sealed_test/optimization_adequacy.csv"
OLD_CALIBRATION = PROJECT_ROOT / "reports/generated/history_value_adam_sealed_v008/sealed_test/calibration_audit.csv"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric_difference(label: str, old: np.ndarray, new: np.ndarray) -> dict[str, Any]:
    left = np.asarray(old, dtype=np.float64)
    right = np.asarray(new, dtype=np.float64)
    difference = np.abs(left - right)
    quantile = np.quantile(difference, [0.5, 0.9, 0.99, 0.999, 1.0])
    return {
        "quantity": label,
        "rows": int(left.size),
        "exact_equal_share": float(np.mean(left == right)),
        "mean_absolute_difference": float(difference.mean()),
        "p50_absolute_difference": float(quantile[0]),
        "p90_absolute_difference": float(quantile[1]),
        "p99_absolute_difference": float(quantile[2]),
        "p999_absolute_difference": float(quantile[3]),
        "maximum_absolute_difference": float(quantile[4]),
        "pearson_correlation": float(np.corrcoef(left, right)[0, 1]),
    }


def main() -> int:
    if OUTPUT_ROOT.exists():
        raise v1.ContractStop("diagnostic output already exists")
    v9.install_overrides()
    contract, digest = v9.load_contract()
    v9.verify_sealed_prerequisite(contract)
    environment = v1.environment_manifest(contract, require_cuda=True)
    device = torch.device("cuda:0")
    OUTPUT_ROOT.mkdir(parents=True)
    v1.progress("diagnostic: reconstruct exact sealed fit without random access")
    frame = v7.read_frame_with_time(v9.SEALED_FEATURES)
    selected = {"ADAM": {"learning_rate": 0.03, "steps": 100}}
    target_index, prediction, frozen, adequacy, calibration = v4.fit_standard_stage(
        frame, stage="sealed_reconstruction_diagnostic",
        fit_range=("2022-04-08", "2022-04-20"), calibration_date="2022-04-21",
        target_range=("2022-04-22", "2022-05-08"), selected=selected, device=device,
    )
    prior = v1.pq.read_table(v9.SEALED_PREDICTIONS, columns=[
        "source_table", "source_row_number", "p_BL0", "p_ADAM_BL1", "p_ADAM_BL2",
        "raw_ADAM_BL1", "raw_ADAM_BL2",
    ])
    identity_exact = bool(
        np.array_equal(
            prior.column("source_table").combine_chunks().to_numpy(zero_copy_only=False),
            frame.columns["source_table"][target_index],
        )
        and np.array_equal(
            prior.column("source_row_number").combine_chunks().to_numpy(zero_copy_only=False),
            frame.columns["source_row_number"][target_index],
        )
    )
    if not identity_exact:
        raise v1.ContractStop("diagnostic target identity mismatch")
    mapping = {
        "BL0_probability": ("p_BL0", "BL0"),
        "ADAM_BL1_probability": ("p_ADAM_BL1", "ADAM_BL1"),
        "ADAM_BL2_probability": ("p_ADAM_BL2", "ADAM_BL2"),
        "ADAM_BL1_raw": ("raw_ADAM_BL1", "raw_ADAM_BL1"),
        "ADAM_BL2_raw": ("raw_ADAM_BL2", "raw_ADAM_BL2"),
    }
    differences: list[dict[str, Any]] = []
    old_probability: dict[str, np.ndarray] = {}
    for label, (old_column, new_key) in mapping.items():
        old = prior.column(old_column).combine_chunks().to_numpy(zero_copy_only=False)
        new = np.asarray(prediction[new_key])
        differences.append(numeric_difference(label, old, new))
        if "probability" in label:
            old_probability[new_key] = np.asarray(old, dtype=np.float64)
    labels = frame.labels()[target_index]
    users = frame.users()[target_index]
    metric_rows: list[dict[str, Any]] = []
    for stream in ("BL0", "ADAM_BL1", "ADAM_BL2"):
        old_metric = v1.metrics.point_metrics(labels, old_probability[stream], users, epsilon=v1.repair.METRIC_CLIP_LOW)
        new_metric = v1.metrics.point_metrics(labels, prediction[stream], users, epsilon=v1.repair.METRIC_CLIP_LOW)
        for metric in ("average_precision", "user_gauc_event_weighted", "log_loss", "brier", "ece20_equal_width"):
            metric_rows.append({
                "stream": stream, "metric": metric,
                "v008_value": old_metric[metric], "reconstructed_value": new_metric[metric],
                "difference": float(new_metric[metric]) - float(old_metric[metric]),
            })
    old_adequacy = csv_rows(OLD_ADEQUACY)
    old_calibration = csv_rows(OLD_CALIBRATION)
    v1.write_csv(OUTPUT_ROOT / "prediction_difference_distribution.csv", differences)
    v1.write_csv(OUTPUT_ROOT / "sealed_metric_reproduction.csv", metric_rows)
    v1.write_csv(OUTPUT_ROOT / "reconstructed_optimization_adequacy.csv", adequacy)
    v1.write_csv(OUTPUT_ROOT / "reconstructed_calibration_audit.csv", calibration)
    with (OUTPUT_ROOT / "reconstructed_frozen_models.pkl").open("wb") as handle:
        pickle.dump(frozen, handle, protocol=5)
    v9._save_model_state(OUTPUT_ROOT / "reconstructed_frozen_model_state.npz", frozen)
    audit = {
        "status": "complete_no_random_data_access",
        "source_v009_contract_sha256": digest,
        "sealed_target_identity_exact": identity_exact,
        "prediction_differences": differences,
        "metric_reproduction": metric_rows,
        "old_adequacy_rows": old_adequacy,
        "old_calibration_rows": old_calibration,
        "reconstructed_models_pickle_sha256": v1.sha256_file(OUTPUT_ROOT / "reconstructed_frozen_models.pkl"),
        "reconstructed_model_state_sha256": v1.sha256_file(OUTPUT_ROOT / "reconstructed_frozen_model_state.npz"),
        "random_input_opened": False,
        "environment": environment,
    }
    v1.write_json(OUTPUT_ROOT / "diagnostic_audit.json", audit)
    lines = [
        "# sealed 模型跨进程重建数值诊断 v010", "",
        "本诊断只读取已经封存的 sealed 特征和预测，不访问 random 数据。", "",
        "| quantity | mean abs diff | p99 | p999 | max | correlation |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in differences:
        lines.append(
            f"| {row['quantity']} | {row['mean_absolute_difference']:.12g} | "
            f"{row['p99_absolute_difference']:.12g} | {row['p999_absolute_difference']:.12g} | "
            f"{row['maximum_absolute_difference']:.12g} | {row['pearson_correlation']:.12g} |"
        )
    lines.extend(["", "逐指标复现差值保存在 `sealed_metric_reproduction.csv`。", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    v1.finalize_hashes(OUTPUT_ROOT)
    del frame, prediction, prior, labels, users
    gc.collect()
    v1.progress(f"diagnostic complete; report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
