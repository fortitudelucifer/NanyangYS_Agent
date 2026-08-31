#!/usr/bin/env python3
"""Run the frozen supplementary BL1/BL2 learning-curve diagnostic."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kuai-matplotlib-v013")

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn
import threadpoolctl
import torch
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from kuairand_longseq.evaluation import gate2b_metrics as metrics  # noqa: E402
from kuairand_longseq.models import gate2b_repair_v003 as repair  # noqa: E402
from kuairand_longseq.models import history_value_gpu as gpu  # noqa: E402
import run_gate2b_probability_repair_v003 as canonical  # noqa: E402
import run_history_value_gpu_confirmation_v001 as common  # noqa: E402


CONTRACT_PATH = EXPERIMENT_ROOT / "contract_v013.yaml"
APPROVAL_PATH = EXPERIMENT_ROOT / "approval_v013.json"
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
FEATURE_PATH = PROJECT_ROOT / "reports/generated/history_value_adam_validation_v006/validation/features.parquet"
FEATURE_MANIFEST_PATH = PROJECT_ROOT / "reports/generated/history_value_adam_validation_v006/validation/feature_manifest.json"
MODEL_IDS = ("BL1", "BL2")
AP_BOOTSTRAP_BLOCK_SIZE = 64
METRIC_NAMES = (
    "average_precision", "log_loss", "brier",
    "user_gauc_event_weighted", "user_gauc_user_equal",
)


class ContractStop(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> tuple[dict[str, Any], str]:
    contract = yaml.load(CONTRACT_PATH.read_text(encoding="utf-8"), Loader=common.UniqueKeyLoader)
    return contract, sha256_file(CONTRACT_PATH)


def write_json(path: Path, payload: Any) -> None:
    common.write_json(path, payload)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    common.write_csv(path, rows)


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def verify_contract_shape(contract: dict[str, Any]) -> None:
    if contract["contract_id"] != "supplementary_learning_curve_v013":
        raise ContractStop("wrong contract id")
    if contract["scope"]["SGD_status"] != "frozen_deferred":
        raise ContractStop("SGD must remain frozen")
    if contract["optimizer"] != {
        "implementation": "kuairand_longseq.models.history_value_gpu.fit_trajectory",
        "name": "ADAM", "learning_rate": 0.03, "steps": 100, "alpha": 0.0001,
        "initialization": "all_zero", "deterministic_algorithms": True,
        "fit_seed": 20260814, "device": "cuda:0", "CPU_fallback": "forbidden",
    }:
        raise ContractStop("frozen optimizer block changed")
    b = contract["designs"]["B_user_cluster_subsample"]
    a = contract["designs"]["A_training_window"]
    if b["fractions"] != [0.10, 0.25, 0.50, 0.75, 1.00]:
        raise ContractStop("B fractions changed")
    if b["seeds"] != [20260814, 20260815, 20260816]:
        raise ContractStop("B seeds changed")
    if a["training_starts"] != ["2022-04-15", "2022-04-13", "2022-04-11", "2022-04-09", "2022-04-08"]:
        raise ContractStop("A windows changed")
    expected_fits = 2 * (len(b["fractions"]) * len(b["seeds"]) + len(a["training_starts"]))
    if expected_fits != int(contract["execution_environment"]["maximum_GPU_fits"]):
        raise ContractStop("fit budget does not match the design")


def environment_manifest(contract: dict[str, Any], require_cuda: bool) -> dict[str, Any]:
    observed = {
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "pyarrow": pa.__version__,
        "duckdb": duckdb.__version__, "threadpoolctl": threadpoolctl.__version__,
    }
    expected = contract["execution_environment"]["required_versions"]
    if observed != expected:
        raise ContractStop(f"environment mismatch: expected={expected}, observed={observed}")
    result: dict[str, Any] = {"versions": observed, "python_executable": sys.executable}
    if require_cuda:
        if not torch.cuda.is_available():
            raise ContractStop("CUDA unavailable; CPU fallback forbidden")
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        if torch.cuda.get_device_name(device) != contract["execution_environment"]["expected_device_name"]:
            raise ContractStop("CUDA device name differs from contract")
        if int(props.total_memory) < int(contract["execution_environment"]["minimum_device_memory_bytes"]):
            raise ContractStop("CUDA memory below contract minimum")
        result["cuda"] = {
            "device": str(device), "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "memory_bytes": int(props.total_memory), "compiled_cuda": torch.version.cuda,
        }
    return result


def implementation_manifest() -> list[dict[str, Any]]:
    paths = [
        Path(__file__), CONTRACT_PATH,
        PROJECT_ROOT / "scripts/run_gate2b_probability_repair_v003.py",
        PROJECT_ROOT / "src/kuairand_longseq/models/gate2b_repair_v003.py",
        PROJECT_ROOT / "src/kuairand_longseq/models/history_value_gpu.py",
        PROJECT_ROOT / "src/kuairand_longseq/evaluation/gate2b_metrics.py",
        EXPERIMENT_ROOT / "implementation_revision_v013_01.md",
        EXPERIMENT_ROOT / "implementation_revision_v013_02.md",
    ]
    return [
        {"path": str(path.relative_to(PROJECT_ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]


def preflight(contract: dict[str, Any], require_cuda: bool) -> dict[str, Any]:
    verify_contract_shape(contract)
    frozen = contract["frozen_input"]
    checks = []
    for name, path, size_key, sha_key in (
        ("features", FEATURE_PATH, "feature_size_bytes", "feature_sha256"),
        ("feature_manifest", FEATURE_MANIFEST_PATH, None, "feature_manifest_sha256"),
    ):
        if not path.is_file():
            raise ContractStop(f"missing frozen input: {path}")
        observed_size = path.stat().st_size
        observed_sha = sha256_file(path)
        if size_key and observed_size != int(frozen[size_key]):
            raise ContractStop(f"{name} size mismatch")
        if observed_sha != frozen[sha_key]:
            raise ContractStop(f"{name} SHA mismatch")
        checks.append({"name": name, "path": str(path.relative_to(PROJECT_ROOT)), "size_bytes": observed_size, "sha256": observed_sha})
    metadata = pq.ParquetFile(FEATURE_PATH).metadata
    if metadata.num_rows != int(frozen["feature_rows"]):
        raise ContractStop("feature row count mismatch")
    required = set(canonical.REQUIRED_COLUMNS) | {"time_ms"} | set(common.EXTRA_COLUMNS)
    missing = sorted(required - set(pq.ParquetFile(FEATURE_PATH).schema_arrow.names))
    if missing:
        raise ContractStop(f"feature schema missing {missing}")
    manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["status"] != frozen["strict_point_in_time_status"]:
        raise ContractStop("point-in-time status mismatch")
    if int(manifest["validation"]["target_rows"]) != int(frozen["expected_validation_rows"]):
        raise ContractStop("frozen Validation row count mismatch")
    return {
        "status": "pass", "inputs": checks, "feature_rows": metadata.num_rows,
        "feature_validation": manifest["validation"],
        "implementation": implementation_manifest(),
        "environment": environment_manifest(contract, require_cuda=require_cuda),
    }


def verify_approval(contract_sha: str) -> dict[str, Any]:
    if not APPROVAL_PATH.is_file():
        raise ContractStop("exact-hash approval receipt missing")
    receipt = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    required = {
        "contract_id": "supplementary_learning_curve_v013",
        "contract_sha256": contract_sha,
        "execution_authorized": True,
        "authorized_stages": ["preflight", "validation_reuse_learning_curve"],
        "approved_by": "project_owner",
    }
    for key, value in required.items():
        if receipt.get(key) != value:
            raise ContractStop(f"approval mismatch: {key}")
    return receipt


def read_frame() -> canonical.Frame:
    columns = list(canonical.REQUIRED_COLUMNS)
    columns.insert(columns.index("long_view"), "time_ms")
    columns.extend(common.EXTRA_COLUMNS)
    table = pq.read_table(FEATURE_PATH, columns=columns)
    return canonical.Frame(columns={
        name: table.column(name).combine_chunks().to_numpy(zero_copy_only=False)
        for name in columns
    })


def nested_user_prefixes(users: np.ndarray, fractions: list[float], seed: int) -> dict[float, np.ndarray]:
    universe = np.unique(np.asarray(users, dtype=np.int64))
    permutation = np.random.Generator(np.random.PCG64(int(seed))).permutation(universe)
    result: dict[float, np.ndarray] = {}
    previous: set[int] = set()
    for fraction in sorted(float(value) for value in fractions):
        count = min(universe.size, max(1, int(math.ceil(fraction * universe.size))))
        selected = np.sort(permutation[:count].astype(np.int64, copy=False))
        current = set(map(int, selected))
        if not previous.issubset(current):
            raise ContractStop("nested user sampling invariant failed")
        previous = current
        result[fraction] = selected
    return result


def sample_digest(users: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(users, dtype="<i8").tobytes(order="C")).hexdigest()


def build_point_specs(frame: canonical.Frame, contract: dict[str, Any]) -> list[dict[str, Any]]:
    dates = frame.dates()
    users = frame.users()
    cal_day = np.datetime64(contract["temporal_protocol"]["calibration_date"], "D")
    val_lo = np.datetime64(contract["temporal_protocol"]["validation_start"], "D")
    val_hi = np.datetime64(contract["temporal_protocol"]["validation_end"], "D")
    calibration_index = np.flatnonzero(dates == cal_day)
    validation_index = np.flatnonzero((dates >= val_lo) & (dates <= val_hi))
    if calibration_index.size == 0 or validation_index.size != int(contract["frozen_input"]["expected_validation_rows"]):
        raise ContractStop("calibration or validation split mismatch")
    specs: list[dict[str, Any]] = []
    a = contract["designs"]["A_training_window"]
    a_end = np.datetime64(a["training_end"], "D")
    for start in a["training_starts"]:
        lo = np.datetime64(start, "D")
        fit_index = np.flatnonzero((dates >= lo) & (dates <= a_end))
        selected = np.unique(users[fit_index])
        specs.append({
            "point_id": f"A_{start}_to_{a['training_end']}", "design": "A_training_window",
            "size_value": start, "fraction": None, "replicate_seed": int(a["seed"]),
            "training_start": start, "training_end": a["training_end"],
            "fit_index": fit_index, "calibration_index": calibration_index,
            "validation_index": validation_index, "sampled_users": selected,
        })
    b = contract["designs"]["B_user_cluster_subsample"]
    b_lo = np.datetime64(b["training_start"], "D")
    b_hi = np.datetime64(b["training_end"], "D")
    full_fit = np.flatnonzero((dates >= b_lo) & (dates <= b_hi))
    full_users = users[full_fit]
    for seed in b["seeds"]:
        prefixes = nested_user_prefixes(full_users, b["fractions"], int(seed))
        for fraction in b["fractions"]:
            selected = prefixes[float(fraction)]
            fit_index = full_fit[np.isin(full_users, selected)]
            specs.append({
                "point_id": f"B_f{float(fraction):.2f}_s{int(seed)}", "design": "B_user_cluster_subsample",
                "size_value": float(fraction), "fraction": float(fraction), "replicate_seed": int(seed),
                "training_start": b["training_start"], "training_end": b["training_end"],
                "fit_index": fit_index, "calibration_index": calibration_index,
                "validation_index": validation_index, "sampled_users": selected,
            })
    return specs


def point_origin(frame: canonical.Frame, spec: dict[str, Any]) -> canonical.OriginSplit:
    labels = frame.labels()
    fit_index = spec["fit_index"]
    return canonical.OriginSplit(
        origin=spec["point_id"], fit_index=fit_index,
        calibration_index=spec["calibration_index"], assessment_index=spec["validation_index"],
        calibration_date="2022-04-17", bl0_probability=float(labels[fit_index].mean()),
        fit_prevalence=float(labels[fit_index].mean()),
    )


def summary_row(point_id: str, target: str, metric: str, point: float, values: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    return {
        "point_id": point_id, "target": target, "metric": metric,
        "point_estimate": float(point), "bootstrap_replicates_requested": int(values.size),
        "effective_replicates": int(valid.size), "bootstrap_mean": float(valid.mean()),
        "bootstrap_se": float(valid.std(ddof=1)),
        "ci95_lower": float(np.quantile(valid, 0.025)),
        "ci95_upper": float(np.quantile(valid, 0.975)),
    }


def bootstrap_point(
    point_id: str, labels: np.ndarray, users: np.ndarray,
    predictions: dict[str, np.ndarray], multiplicities: np.ndarray,
) -> tuple[list[dict[str, Any]], pa.Table]:
    universe = np.unique(users)
    if multiplicities.shape[1] != universe.size:
        raise ContractStop("bootstrap user universe changed")
    row_user = metrics._user_index(users, universe)
    user_count = universe.size
    user_rows = np.bincount(row_user, minlength=user_count).astype(np.float64)
    replicate_rows = multiplicities @ user_rows
    points = {
        model: metrics.point_metrics(labels, predictions[model], users, epsilon=repair.METRIC_CLIP_LOW)
        for model in MODEL_IDS
    }
    values: dict[str, dict[str, np.ndarray]] = {name: {} for name in METRIC_NAMES}
    for model in MODEL_IDS:
        p = metrics.clipped(predictions[model], repair.METRIC_CLIP_LOW)
        values["average_precision"][model] = metrics.weighted_ap_replicates(
            labels, p, row_user, multiplicities,
            block_size=AP_BOOTSTRAP_BLOCK_SIZE, epsilon=repair.METRIC_CLIP_LOW,
        )
        log_row = -(labels * np.log(p) + (1 - labels) * np.log1p(-p))
        brier_row = np.square(p - labels)
        values["log_loss"][model] = (
            multiplicities @ metrics._per_user_sums(log_row, row_user, user_count)
        ) / replicate_rows
        values["brier"][model] = (
            multiplicities @ metrics._per_user_sums(brier_row, row_user, user_count)
        ) / replicate_rows
        gauc = metrics.user_gauc_components(
            labels, p, users, user_universe=universe, epsilon=repair.METRIC_CLIP_LOW,
        )
        eligible = gauc.eligible.astype(np.float64)
        eligible_rows = gauc.event_counts.astype(np.float64) * eligible
        auc = np.nan_to_num(gauc.auc, nan=0.0)
        values["user_gauc_event_weighted"][model] = (
            multiplicities @ (auc * eligible_rows)
        ) / (multiplicities @ eligible_rows)
        values["user_gauc_user_equal"][model] = (
            multiplicities @ (auc * eligible)
        ) / (multiplicities @ eligible)
    summaries: list[dict[str, Any]] = []
    tables: list[pa.Table] = []
    replicate_id = np.arange(multiplicities.shape[0], dtype=np.int32)
    for metric_name in METRIC_NAMES:
        for model in MODEL_IDS:
            replicate = values[metric_name][model]
            summaries.append(summary_row(
                point_id, model, metric_name, float(points[model][metric_name]), replicate,
            ))
            tables.append(pa.table({
                "point_id": pa.array(np.repeat(point_id, replicate.size)),
                "target": pa.array(np.repeat(model, replicate.size)),
                "metric": pa.array(np.repeat(metric_name, replicate.size)),
                "replicate_id": replicate_id, "value": replicate,
            }))
        delta = values[metric_name]["BL2"] - values[metric_name]["BL1"]
        point = float(points["BL2"][metric_name] - points["BL1"][metric_name])
        summaries.append(summary_row(point_id, "BL2_minus_BL1", metric_name, point, delta))
        tables.append(pa.table({
            "point_id": pa.array(np.repeat(point_id, delta.size)),
            "target": pa.array(np.repeat("BL2_minus_BL1", delta.size)),
            "metric": pa.array(np.repeat(metric_name, delta.size)),
            "replicate_id": replicate_id, "value": delta,
        }))
    return summaries, pa.concat_tables(tables)


def fit_one_point(
    frame: canonical.Frame, spec: dict[str, Any], contract: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    labels = frame.labels()
    users = frame.users()
    origin = canonical.build_origin_matrices(frame, point_origin(frame, spec))
    point_metrics_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    validation_predictions: dict[str, np.ndarray] = {}
    config = contract["optimizer"]
    for model_id in MODEL_IDS:
        progress(f"{spec['point_id']}: GPU fit {model_id}")
        fit = gpu.fit_trajectory(
            origin.matrices["fit"][model_id], origin.labels["fit"], device=device,
            optimizer_name="ADAM", learning_rate=float(config["learning_rate"]),
            checkpoints=[int(config["steps"])], alpha=float(config["alpha"]),
        )
        raw_train, train_score_seconds = gpu.score(
            origin.matrices["fit"][model_id], fit.coefficient, fit.intercept, device=device,
        )
        raw_cal, cal_score_seconds = gpu.score(
            origin.matrices["calibration"][model_id], fit.coefficient, fit.intercept, device=device,
        )
        raw_val, val_score_seconds = gpu.score(
            origin.matrices["assessment"][model_id], fit.coefficient, fit.intercept, device=device,
        )
        calibrator = repair.fit_previous_day_sigmoid(
            raw_cal, origin.labels["calibration"], user_id=origin.users["calibration"],
        )
        train_probability = calibrator.apply(raw_train)
        val_probability = calibrator.apply(raw_val)
        repair.assert_calibration_monotone(raw_val, val_probability)
        validation_predictions[model_id] = val_probability
        fit_rows.append({
            "point_id": spec["point_id"], "design": spec["design"], "model": model_id,
            "optimizer": fit.optimizer, "learning_rate": fit.learning_rate, "steps": fit.steps,
            "alpha": float(config["alpha"]), "objective": fit.objective,
            "terminal_gradient_norm": fit.terminal_gradient_norm,
            "fit_elapsed_seconds": fit.elapsed_seconds,
            "train_score_seconds": train_score_seconds, "calibration_score_seconds": cal_score_seconds,
            "validation_score_seconds": val_score_seconds,
            "peak_cuda_memory_bytes": fit.peak_cuda_memory_bytes,
            "matrix_rows": int(origin.matrices["fit"][model_id].shape[0]),
            "matrix_columns": int(origin.matrices["fit"][model_id].shape[1]),
            "matrix_nonzeros": int(origin.matrices["fit"][model_id].nnz),
        })
        calibration_rows.append({
            "point_id": spec["point_id"], "design": spec["design"], "model": model_id,
            "fit_rows": calibrator.fit_rows, "fit_users": calibrator.fit_users,
            "fit_positives": calibrator.fit_positives, "fit_prevalence": calibrator.fit_prevalence,
            "intercept": calibrator.intercept, "slope": calibrator.slope,
            "n_iter": calibrator.n_iter,
            "convergence_warning_count": calibrator.convergence_warning_count,
        })
        for split, y, p, u in (
            ("train", origin.labels["fit"], train_probability, origin.users["fit"]),
            ("validation", origin.labels["assessment"], val_probability, origin.users["assessment"]),
        ):
            row = metrics.point_metrics(y, p, u, epsilon=repair.METRIC_CLIP_LOW)
            point_metrics_rows.append({
                "point_id": spec["point_id"], "design": spec["design"],
                "size_value": spec["size_value"], "fraction": spec["fraction"],
                "replicate_seed": spec["replicate_seed"], "model": model_id, "split": split,
                "probability_basis": "previous_day_sigmoid_applied_to_model_score", **row,
            })
    coverage = [
        {"point_id": spec["point_id"], "design": spec["design"], **row}
        for row in origin.categorical_audit
    ]
    del origin
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "metrics": point_metrics_rows, "fits": fit_rows, "calibration": calibration_rows,
        "coverage": coverage, "validation_predictions": validation_predictions,
        "validation_labels": labels[spec["validation_index"]],
        "validation_users": users[spec["validation_index"]],
    }


def metric_lookup(rows: list[dict[str, Any]], point_id: str, model: str, split: str) -> dict[str, Any]:
    return next(row for row in rows if row["point_id"] == point_id and row["model"] == model and row["split"] == split)


def bootstrap_lookup(rows: list[dict[str, Any]], point_id: str, target: str, metric: str) -> dict[str, Any]:
    return next(row for row in rows if row["point_id"] == point_id and row["target"] == target and row["metric"] == metric)


def add_uplift_rows(
    spec: dict[str, Any], point_rows: list[dict[str, Any]], bootstrap_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    train1 = metric_lookup(point_rows, spec["point_id"], "BL1", "train")
    train2 = metric_lookup(point_rows, spec["point_id"], "BL2", "train")
    val1 = metric_lookup(point_rows, spec["point_id"], "BL1", "validation")
    val2 = metric_lookup(point_rows, spec["point_id"], "BL2", "validation")
    ap_boot = bootstrap_lookup(bootstrap_rows, spec["point_id"], "BL2_minus_BL1", "average_precision")
    return {
        "point_id": spec["point_id"], "design": spec["design"], "size_value": spec["size_value"],
        "fraction": spec["fraction"], "replicate_seed": spec["replicate_seed"],
        "n_train_users": train1["users"], "n_train_events": train1["rows"],
        "train_prevalence": train1["prevalence"],
        "train_delta_ap_BL2_minus_BL1": train2["average_precision"] - train1["average_precision"],
        "validation_delta_ap_BL2_minus_BL1": val2["average_precision"] - val1["average_precision"],
        "validation_delta_ap_ci95_lower": ap_boot["ci95_lower"],
        "validation_delta_ap_ci95_upper": ap_boot["ci95_upper"],
        "BL1_train_minus_validation_ap": train1["average_precision"] - val1["average_precision"],
        "BL2_train_minus_validation_ap": train2["average_precision"] - val2["average_precision"],
        "BL2_specific_excess_ap_gap": (
            (train2["average_precision"] - val2["average_precision"])
            - (train1["average_precision"] - val1["average_precision"])
        ),
    }


def aggregate_for_plot(rows: list[dict[str, Any]], design: str, model: str, split: str) -> dict[str, np.ndarray]:
    selected = [row for row in rows if row["design"] == design and row["model"] == model and row["split"] == split]
    groups: dict[float, list[dict[str, Any]]] = {}
    for row in selected:
        x = float(row["fraction"]) if design.startswith("B_") else float(row["rows"])
        groups.setdefault(x, []).append(row)
    x = np.asarray(sorted(groups), dtype=np.float64)
    y = np.asarray([np.mean([row["average_precision"] for row in groups[value]]) for value in x])
    return {"x": x, "y": y}


def make_plots(point_rows: list[dict[str, Any]], uplift_rows: list[dict[str, Any]]) -> None:
    colors = {"BL1": "#377eb8", "BL2": "#e41a1c"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, model in zip(axes[:2], MODEL_IDS):
        for split, style in (("train", "--"), ("validation", "-")):
            data = aggregate_for_plot(point_rows, "B_user_cluster_subsample", model, split)
            axis.plot(data["x"], data["y"], style, marker="o", color=colors[model],
                      alpha=0.55 if split == "train" else 1.0, label=split)
        axis.set_xscale("log")
        axis.set_title(model)
        axis.set_xlabel("Training user fraction")
        axis.set_ylabel("Average precision")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    b_rows = [row for row in uplift_rows if row["design"] == "B_user_cluster_subsample"]
    fractions = sorted({float(row["fraction"]) for row in b_rows})
    mean = np.asarray([np.mean([row["validation_delta_ap_BL2_minus_BL1"] for row in b_rows if float(row["fraction"]) == value]) for value in fractions])
    low = np.asarray([min(row["validation_delta_ap_ci95_lower"] for row in b_rows if float(row["fraction"]) == value) for value in fractions])
    high = np.asarray([max(row["validation_delta_ap_ci95_upper"] for row in b_rows if float(row["fraction"]) == value) for value in fractions])
    axes[2].plot(fractions, mean, marker="o", color="#4daf4a")
    axes[2].fill_between(fractions, low, high, color="#4daf4a", alpha=0.2, label="pointwise CI envelope")
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_xscale("log")
    axes[2].set_title("Validation history uplift")
    axes[2].set_xlabel("Training user fraction")
    axes[2].set_ylabel("AP(BL2) - AP(BL1)")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("B: nested user-cluster learning curve (mean across 3 sampling seeds)")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "learning_curve_B_main.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    a_training_events = {
        row["point_id"]: float(row["n_train_events"])
        for row in uplift_rows if row["design"] == "A_training_window"
    }
    for axis, model in zip(axes[:2], MODEL_IDS):
        for split, style in (("train", "--"), ("validation", "-")):
            selected = sorted(
                (
                    (a_training_events[row["point_id"]], float(row["average_precision"]))
                    for row in point_rows
                    if row["design"] == "A_training_window"
                    and row["model"] == model and row["split"] == split
                ),
                key=lambda pair: pair[0],
            )
            x_value = np.asarray([pair[0] for pair in selected], dtype=np.float64)
            y_value = np.asarray([pair[1] for pair in selected], dtype=np.float64)
            axis.plot(x_value, y_value, style, marker="o", color=colors[model],
                      alpha=0.55 if split == "train" else 1.0, label=split)
        axis.set_title(model)
        axis.set_xlabel("Training events")
        axis.set_ylabel("Average precision")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    a_rows = sorted(
        [row for row in uplift_rows if row["design"] == "A_training_window"],
        key=lambda row: row["n_train_events"],
    )
    x = np.asarray([row["n_train_events"] for row in a_rows])
    y = np.asarray([row["validation_delta_ap_BL2_minus_BL1"] for row in a_rows])
    low = np.asarray([row["validation_delta_ap_ci95_lower"] for row in a_rows])
    high = np.asarray([row["validation_delta_ap_ci95_upper"] for row in a_rows])
    axes[2].plot(x, y, marker="o", color="#4daf4a")
    axes[2].fill_between(x, low, high, color="#4daf4a", alpha=0.2)
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_title("Validation history uplift")
    axes[2].set_xlabel("Training events")
    axes[2].set_ylabel("AP(BL2) - AP(BL1)")
    axes[2].grid(alpha=0.25)
    fig.suptitle("A: labeled training-window learning curve (H2 definition unchanged)")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "learning_curve_A_support.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def final_decision(
    specs: list[dict[str, Any]], failures: list[dict[str, Any]], uplift: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]], coverage: list[dict[str, Any]],
) -> dict[str, Any]:
    positive = [row for row in uplift if row["validation_delta_ap_BL2_minus_BL1"] > 0]
    ci_positive = [row for row in uplift if row["validation_delta_ap_ci95_lower"] > 0]
    b = [row for row in uplift if row["design"] == "B_user_cluster_subsample"]
    a = [row for row in uplift if row["design"] == "A_training_window"]
    unknown = [
        row for row in coverage
        if row.get("split") == "assessment" and row.get("field") in {"cat_user", "cat_video", "cat_author"}
    ]
    max_unknown = {
        field: max((float(row["unknown_event_share"]) for row in unknown if row["field"] == field), default=float("nan"))
        for field in ("cat_user", "cat_video", "cat_author")
    }
    excess = [float(row["BL2_specific_excess_ap_gap"]) for row in uplift]
    return {
        "scientific_status": "complete" if not failures and len(uplift) == len(specs) else "partial",
        "evidence_level": "supplementary_generalization_diagnostic_on_previously_consumed_validation",
        "changes_v010_v011_or_v012_conclusions": False,
        "expected_curve_points": len(specs), "completed_curve_points": len(uplift),
        "failed_curve_points": len(failures),
        "positive_validation_history_uplift_points": len(positive),
        "validation_history_uplift_CI_strictly_above_zero_points": len(ci_positive),
        "B_points": len(b), "A_points": len(a),
        "B_validation_delta_AP_range": [
            min((row["validation_delta_ap_BL2_minus_BL1"] for row in b), default=None),
            max((row["validation_delta_ap_BL2_minus_BL1"] for row in b), default=None),
        ],
        "A_validation_delta_AP_range": [
            min((row["validation_delta_ap_BL2_minus_BL1"] for row in a), default=None),
            max((row["validation_delta_ap_BL2_minus_BL1"] for row in a), default=None),
        ],
        "BL2_specific_excess_AP_gap_range_descriptive": [min(excess, default=None), max(excess, default=None)],
        "maximum_validation_unknown_category_share": max_unknown,
        "interpretation_rule": (
            "Primary evidence is paired BL2-minus-BL1 Validation uplift. Train-minus-Validation gaps and "
            "unknown-category coverage are descriptive and cannot alone prove or refute overfitting."
        ),
    }


def render_report(
    contract_sha: str, decision: dict[str, Any], uplift: list[dict[str, Any]],
    point_rows: list[dict[str, Any]], failures: list[dict[str, Any]],
) -> str:
    def fmt(value: Any, digits: int = 6) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"
    lines = [
        "# Supplementary learning curve v013 结果", "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        f"- 合同 SHA-256：`{contract_sha}`",
        "- 环境：conda `Kuai`，RTX 5070 Ti，GPU Adam。",
        "- 证据等级：已经使用过的 Validation 上的补充泛化诊断；不是新的 pristine confirmation。", "",
        "## 设计", "",
        "B 按训练用户簇进行 10%/25%/50%/75%/100% 的嵌套下采样，每个比例三个冻结种子。A 只改变带标签训练窗起点；所有点共享 04-17 校准日和 04-18 至 04-21 Validation 目标行，H2 特征定义不变。", "",
        "## Validation 历史增量", "",
        "| design | size | seed | train users | train events | ΔAP BL2−BL1 | 95% CI | excess AP gap |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(uplift, key=lambda item: (item["design"], str(item["size_value"]), item["replicate_seed"])):
        lines.append(
            f"| {row['design']} | {row['size_value']} | {row['replicate_seed']} | {row['n_train_users']} | "
            f"{row['n_train_events']} | {fmt(row['validation_delta_ap_BL2_minus_BL1'])} | "
            f"[{fmt(row['validation_delta_ap_ci95_lower'])}, {fmt(row['validation_delta_ap_ci95_upper'])}] | "
            f"{fmt(row['BL2_specific_excess_ap_gap'])} |"
        )
    lines.extend(["", "## 汇总结论", "",
        f"- 完成 {decision['completed_curve_points']}/{decision['expected_curve_points']} 个曲线点；失败 {decision['failed_curve_points']} 个。",
        f"- Validation 上 BL2−BL1 AP 点估计为正：{decision['positive_validation_history_uplift_points']}/{decision['completed_curve_points']} 个点。",
        f"- 95% 用户簇 bootstrap 区间严格高于零：{decision['validation_history_uplift_CI_strictly_above_zero_points']}/{decision['completed_curve_points']} 个点。",
        f"- B 的 Validation ΔAP 范围：{decision['B_validation_delta_AP_range']}。",
        f"- A 的 Validation ΔAP 范围：{decision['A_validation_delta_AP_range']}。", "",
        "这里最强的允许表述是：观察 BL2 相对 BL1 的 Validation 增量是否随训练规模保持同向，以及是否出现 BL2 特有的额外泛化差距。train–Validation 的裸间距同时受时间漂移和目标率变化影响，不能单独证明没有过拟合。", "",
        "## 产物", "",
        "主表为 `point_metrics.csv`、`history_uplift.csv` 和 `bootstrap_summary.csv`；完整重复值在 `bootstrap_replicates.parquet`，固定预测在 `validation_predictions.parquet`；主图为 `learning_curve_B_main.png`，辅助图为 `learning_curve_A_support.png`。", "",
    ])
    if failures:
        lines.extend(["## 失败点", "", "```json", json.dumps(failures, indent=2, ensure_ascii=False), "```", ""])
    return "\n".join(lines)


def finalize_hashes(directory: Path) -> None:
    manifest_path = directory / "artifact_hash_manifest.json"
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path != manifest_path:
            rows.append({
                "path": str(path.relative_to(PROJECT_ROOT)), "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    write_json(manifest_path, {"hash_algorithm": "SHA-256", "artifacts": rows})


def validate_only() -> None:
    contract, digest = load_contract()
    result = preflight(contract, require_cuda=False)
    progress(f"V013_VALIDATE_ONLY_OK contract_sha256={digest} feature_rows={result['feature_rows']}")


def release(approved_hash: str) -> None:
    contract, digest = load_contract()
    if approved_hash != digest:
        raise ContractStop(f"approved hash mismatch: observed {digest}")
    verify_approval(digest)
    if OUTPUT_ROOT.exists():
        raise ContractStop("outputs already exist; overwrite forbidden")
    preflight_result = preflight(contract, require_cuda=True)
    OUTPUT_ROOT.mkdir(parents=True)
    write_json(OUTPUT_ROOT / "preflight.json", {"contract_sha256": digest, **preflight_result})
    started = time.perf_counter()
    frame = read_frame()
    specs = build_point_specs(frame, contract)
    expected_points = 5 + 5 * 3
    if len(specs) != expected_points:
        raise ContractStop("curve point count mismatch")
    labels = frame.labels()
    users = frame.users()
    validation_index = specs[0]["validation_index"]
    validation_users = users[validation_index]
    user_universe = np.unique(validation_users)
    multiplicities, multiplicity_sha = metrics.make_multiplicities(
        user_count=user_universe.size,
        replicates=int(contract["statistics"]["bootstrap_replicates"]),
        seed=int(contract["statistics"]["bootstrap_seed"]),
    )
    np.save(OUTPUT_ROOT / "bootstrap_multiplicity.npy", multiplicities, allow_pickle=False)
    sample_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_tables: list[pa.Table] = []
    uplift_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    prediction_columns: dict[str, pa.Array] = {
        "source_table": pa.array(frame.columns["source_table"][validation_index]),
        "source_row_number": pa.array(frame.columns["source_row_number"][validation_index]),
        "user_id": pa.array(validation_users),
        "event_date": pa.array(frame.columns["event_date"][validation_index]),
        "long_view": pa.array(labels[validation_index]),
    }
    device = torch.device("cuda:0")
    for number, spec in enumerate(specs, start=1):
        sample_rows.append({
            "point_id": spec["point_id"], "design": spec["design"], "size_value": spec["size_value"],
            "fraction": spec["fraction"], "replicate_seed": spec["replicate_seed"],
            "sampled_user_count": int(spec["sampled_users"].size),
            "sampled_user_sha256": sample_digest(spec["sampled_users"]),
        })
        for split_name, index in (
            ("train", spec["fit_index"]), ("calibration", spec["calibration_index"]),
            ("validation", spec["validation_index"]),
        ):
            split_rows.append({
                "point_id": spec["point_id"], "design": spec["design"], "split": split_name,
                "rows": int(index.size), "users": int(np.unique(users[index]).size),
                "positives": int(labels[index].sum()), "prevalence": float(labels[index].mean()),
                "date_min": str(frame.dates()[index].min()), "date_max": str(frame.dates()[index].max()),
            })
        progress(f"point {number}/{len(specs)} start: {spec['point_id']}")
        try:
            result = fit_one_point(frame, spec, contract, device)
            point_rows.extend(result["metrics"])
            fit_rows.extend(result["fits"])
            calibration_rows.extend(result["calibration"])
            coverage_rows.extend(result["coverage"])
            summaries, replicate_table = bootstrap_point(
                spec["point_id"], result["validation_labels"], result["validation_users"],
                result["validation_predictions"], multiplicities,
            )
            bootstrap_rows.extend(summaries)
            bootstrap_tables.append(replicate_table)
            uplift_rows.append(add_uplift_rows(spec, point_rows, bootstrap_rows))
            for model_id in MODEL_IDS:
                prediction_columns[f"{spec['point_id']}__{model_id}"] = pa.array(
                    result["validation_predictions"][model_id].astype(np.float32)
                )
            progress(f"point {number}/{len(specs)} complete: {spec['point_id']}")
        except Exception as exc:  # contracted record-and-continue policy
            failures.append({
                "point_id": spec["point_id"], "error_type": type(exc).__name__,
                "message": str(exc), "traceback": traceback.format_exc(),
            })
            progress(f"point {number}/{len(specs)} FAILED: {spec['point_id']} {type(exc).__name__}: {exc}")
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(OUTPUT_ROOT / "sample_manifests.csv", sample_rows)
    write_csv(OUTPUT_ROOT / "split_audit.csv", split_rows)
    write_csv(OUTPUT_ROOT / "fit_audit.csv", fit_rows)
    write_csv(OUTPUT_ROOT / "calibration_audit.csv", calibration_rows)
    write_csv(OUTPUT_ROOT / "categorical_coverage_audit.csv", coverage_rows)
    write_csv(OUTPUT_ROOT / "point_metrics.csv", point_rows)
    write_csv(OUTPUT_ROOT / "history_uplift.csv", uplift_rows)
    write_csv(OUTPUT_ROOT / "bootstrap_summary.csv", bootstrap_rows)
    pq.write_table(pa.concat_tables(bootstrap_tables), OUTPUT_ROOT / "bootstrap_replicates.parquet", compression="zstd")
    pq.write_table(pa.table(prediction_columns), OUTPUT_ROOT / "validation_predictions.parquet", compression="zstd")
    write_json(OUTPUT_ROOT / "point_failures.json", failures)
    if uplift_rows:
        make_plots(point_rows, uplift_rows)
    decision = final_decision(specs, failures, uplift_rows, bootstrap_rows, coverage_rows)
    write_json(OUTPUT_ROOT / "final_decision.json", decision)
    report = render_report(digest, decision, uplift_rows, point_rows, failures)
    (OUTPUT_ROOT / "results_v013.md").write_text(report, encoding="utf-8")
    write_json(OUTPUT_ROOT / "run_manifest.json", {
        "status": decision["scientific_status"], "contract_sha256": digest,
        "generated_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "feature_sha256": contract["frozen_input"]["feature_sha256"],
        "bootstrap_multiplicity_sha256": multiplicity_sha,
        "implementation": implementation_manifest(), "decision": decision,
    })
    finalize_hashes(OUTPUT_ROOT)
    progress(f"v013 complete status={decision['scientific_status']} outputs={OUTPUT_ROOT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--release", action="store_true")
    parser.add_argument("--approved-contract-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_only:
        validate_only()
        return 0
    if not args.approved_contract_sha256:
        raise ContractStop("--approved-contract-sha256 required")
    release(args.approved_contract_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
