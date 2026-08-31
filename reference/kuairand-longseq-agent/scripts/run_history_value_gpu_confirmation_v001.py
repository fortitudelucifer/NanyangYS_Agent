#!/usr/bin/env python3
"""Run the frozen Validation -> sealed -> random GPU history confirmation."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn
import threadpoolctl
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from kuairand_longseq.evaluation import gate2b_metrics as metrics  # noqa: E402
from kuairand_longseq.features.history_value_feature_sql import (  # noqa: E402
    materialize_random_features,
    materialize_standard_features,
)
from kuairand_longseq.models import gate2b_repair_v003 as repair  # noqa: E402
from kuairand_longseq.models import history_value_gpu as gpu  # noqa: E402
import run_gate2b_probability_repair_v003 as canonical  # noqa: E402


CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_gpu_confirmation_contract_v001.yaml"
APPROVAL_PATH = PROJECT_ROOT / "configs/history_value_gpu_confirmation_approval_v001.json"
OUTPUT_ROOT = PROJECT_ROOT / "reports/generated/history_value_gpu_confirmation_v001"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/history_value_gpu_confirmation_results_v001.md"
TRAIN_FEATURE = PROJECT_ROOT / "reports/generated/gate2b_baselines_v002/gate2b_feature_matrix.parquet"
EXTRA_COLUMNS = ("warm_user", "warm_video", "behavior_cold_video")
STREAMS = ("BL0", "ADAM_BL1", "ADAM_BL2", "SGD_BL1", "SGD_BL2")
LEARNED = STREAMS[1:]
CONTRASTS: dict[str, dict[str, float]] = {
    "ADAM_BL1_minus_BL0": {"ADAM_BL1": 1.0, "BL0": -1.0},
    "SGD_BL1_minus_BL0": {"SGD_BL1": 1.0, "BL0": -1.0},
    "ADAM_BL2_minus_ADAM_BL1": {"ADAM_BL2": 1.0, "ADAM_BL1": -1.0},
    "SGD_BL2_minus_SGD_BL1": {"SGD_BL2": 1.0, "SGD_BL1": -1.0},
    "ADAM_BL1_minus_SGD_BL1": {"ADAM_BL1": 1.0, "SGD_BL1": -1.0},
    "ADAM_BL2_minus_SGD_BL2": {"ADAM_BL2": 1.0, "SGD_BL2": -1.0},
    "Adam_history_effect_minus_SGD_history_effect": {
        "ADAM_BL2": 1.0, "ADAM_BL1": -1.0, "SGD_BL2": -1.0, "SGD_BL1": 1.0,
    },
}


class ContractStop(RuntimeError):
    pass


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: yaml.Loader, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ContractStop(f"duplicate contract key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_contract() -> tuple[dict[str, Any], str]:
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    return yaml.load(text, Loader=UniqueKeyLoader), sha256_file(CONTRACT_PATH)


def resolve_input(contract: dict[str, Any], key: str) -> Path:
    return (PROJECT_ROOT / contract["data_governance"]["allowlisted_inputs"][key]["archive_path"]).resolve()


def verify_input(contract: dict[str, Any], key: str) -> dict[str, Any]:
    entry = contract["data_governance"]["allowlisted_inputs"][key]
    path = resolve_input(contract, key)
    if not path.is_file():
        raise ContractStop(f"missing allowlisted input: {key}")
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != int(entry["expected_size_bytes"]) or digest != entry["expected_sha256"]:
        raise ContractStop(f"size/SHA mismatch for {key}")
    return {"input": key, "path": str(path), "size_bytes": size, "sha256": digest}


def environment_manifest(contract: dict[str, Any], *, require_cuda: bool) -> dict[str, Any]:
    observed = {
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "pyarrow": pa.__version__,
        "duckdb": duckdb.__version__, "threadpoolctl": threadpoolctl.__version__,
    }
    expected = contract["execution_environment"]["required_versions"]
    if observed != expected:
        raise ContractStop(f"environment version mismatch: expected {expected}, observed {observed}")
    result: dict[str, Any] = {"versions": observed, "executable": sys.executable}
    if require_cuda:
        if not torch.cuda.is_available():
            raise ContractStop("CUDA is unavailable; CPU fallback is forbidden")
        device = torch.device("cuda:0")
        memory = int(torch.cuda.get_device_properties(device).total_memory)
        result["cuda"] = {
            "device": str(device), "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "memory_bytes": memory, "compiled_cuda": torch.version.cuda,
        }
        if result["cuda"]["name"] != contract["execution_environment"]["expected_device_name"]:
            raise ContractStop("CUDA device name differs from contract")
        if memory < int(contract["execution_environment"]["minimum_device_memory_bytes"]):
            raise ContractStop("CUDA device memory is below the contract minimum")
    return result


def implementation_records(contract: dict[str, Any]) -> list[dict[str, Any]]:
    declared = contract["implementation_status"].get("result_producing_files", [])
    records = []
    for item in declared:
        path = PROJECT_ROOT / item["path"]
        observed = sha256_file(path)
        if observed != item["sha256"]:
            raise ContractStop(f"implementation hash mismatch: {item['path']}")
        records.append({"path": item["path"], "sha256": observed})
    return records


def verify_approval_receipt(contract_sha: str) -> dict[str, Any]:
    if not APPROVAL_PATH.is_file():
        raise ContractStop(
            "exact-hash approval receipt is missing; governed data access remains locked"
        )
    receipt = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    required = {
        "contract_id": "history_value_gpu_confirmation_v001",
        "contract_sha256": contract_sha,
        "execution_authorized": True,
        "automatic_ordered_transitions_authorized": True,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ContractStop(f"approval receipt mismatch: {key}")
    if receipt.get("approved_by") != "project_owner":
        raise ContractStop("approval receipt must identify the project owner")
    return receipt


def validate_only() -> None:
    contract, digest = load_contract()
    if contract["contract_id"] != "history_value_gpu_confirmation_v001":
        raise ContractStop("wrong contract id")
    if tuple(contract["model_matrix"]["required_prediction_streams"]) != STREAMS:
        raise ContractStop("five prediction streams differ from runner")
    if int(contract["bootstrap"]["replicates"]) != 2000:
        raise ContractStop("bootstrap replicate count differs from runner")
    implementation_records(contract)
    environment_manifest(contract, require_cuda=False)
    progress(f"VALIDATE_ONLY_OK contract_sha256={digest}; governed data not opened")


def read_frame(path: Path) -> canonical.Frame:
    columns = list(canonical.REQUIRED_COLUMNS) + list(EXTRA_COLUMNS)
    table = pq.read_table(path, columns=columns)
    return canonical.Frame(
        columns={
            name: table.column(name).combine_chunks().to_numpy(zero_copy_only=False)
            for name in columns
        }
    )


def raw_blocks(frame: canonical.Frame, index: np.ndarray, prevalence: float) -> dict[str, np.ndarray]:
    return canonical._group_blocks(frame, index, prevalence)


def configure_duckdb(con: duckdb.DuckDBPyConnection, temp_dir: Path) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute("SET threads=8")
    con.execute("SET memory_limit='20GB'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory=?", [str(temp_dir)])


def feature_manifest(path: Path, validation: dict[str, Any], inputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "complete_strict_point_in_time", "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
        "validation": validation, "verified_inputs": inputs,
    }


def materialize_stage_features(
    contract: dict[str, Any], stage: str, stage_dir: Path
) -> tuple[Path, dict[str, Any]]:
    path = stage_dir / "features.parquet"
    temp = stage_dir / ".features.building.parquet"
    if path.exists():
        raise ContractStop(f"refusing to overwrite completed/incomplete stage feature: {path}")
    keys = ["events_early_standard", "videos_basic", "label_formula_mismatch_rows"]
    if stage in {"sealed_test", "random_audit"}:
        keys.append("events_late_standard")
    if stage == "random_audit":
        keys.append("events_random")
    verified = [verify_input(contract, key) for key in keys]
    con = duckdb.connect()
    configure_duckdb(con, stage_dir / "duckdb_tmp")
    try:
        if stage == "validation":
            validation = materialize_standard_features(
                con, early_path=resolve_input(contract, "events_early_standard"), late_path=None,
                mismatch_path=resolve_input(contract, "label_formula_mismatch_rows"),
                videos_path=resolve_input(contract, "videos_basic"), output_path=temp,
                end_date="2022-04-21", target_start="2022-04-18", target_end="2022-04-21",
                expected_target_rows=881035,
            )
        elif stage == "sealed_test":
            validation = materialize_standard_features(
                con, early_path=resolve_input(contract, "events_early_standard"),
                late_path=resolve_input(contract, "events_late_standard"),
                mismatch_path=resolve_input(contract, "label_formula_mismatch_rows"),
                videos_path=resolve_input(contract, "videos_basic"), output_path=temp,
                end_date="2022-05-08", target_start="2022-04-22", target_end="2022-05-08",
                expected_target_rows=4404167,
            )
        elif stage == "random_audit":
            validation = materialize_random_features(
                con, early_path=resolve_input(contract, "events_early_standard"),
                late_path=resolve_input(contract, "events_late_standard"),
                random_path=resolve_input(contract, "events_random"),
                mismatch_path=resolve_input(contract, "label_formula_mismatch_rows"),
                videos_path=resolve_input(contract, "videos_basic"), output_path=temp,
                expected_target_rows=43027,
            )
        else:
            raise ContractStop(f"no feature materializer for {stage}")
    finally:
        con.close()
    temp.replace(path)
    manifest = feature_manifest(path, validation, verified)
    write_json(stage_dir / "feature_manifest.json", manifest)
    return path, manifest


def reference_objectives(
    matrices: dict[str, Any], labels: np.ndarray, *, origin: str
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    objectives: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for model_id in ("BL1", "BL2"):
        progress(f"{origin}: CPU reference {model_id}")
        _, record, c_value = repair.fit_reference(matrices[model_id], labels, alpha=1e-4)
        if not record.converged:
            raise ContractStop(f"reference solver did not converge: {origin}/{model_id}")
        objectives[model_id] = record.objective
        rows.append({
            "origin": origin, "model_id": model_id, "reference_C": c_value,
            **record.as_row(),
        })
    return objectives, rows


def preflight(contract: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = OUTPUT_ROOT / "preflight"
    if out.exists():
        raise ContractStop("preflight output already exists; completed stages are immutable")
    out.mkdir(parents=True)
    declared = contract["implementation_status"]["optimizer_preflight_feature_artifact"]
    if TRAIN_FEATURE.stat().st_size != int(declared["size_bytes"]) or sha256_file(TRAIN_FEATURE) != declared["sha256"]:
        raise ContractStop("frozen Train preflight feature artifact hash mismatch")
    frame = read_frame_with_defaults(TRAIN_FEATURE)
    dates = frame.dates()
    labels = frame.labels()
    adequacy_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    trajectories: dict[tuple[str, float, int, str, str], dict[str, Any]] = {}
    for origin_text in contract["optimizer_adequacy"]["train_probe_origins"]:
        origin = np.datetime64(origin_text, "D")
        cutoff = origin - np.timedelta64(1, "D")
        fit_index = np.flatnonzero(dates < cutoff)
        prevalence = float(labels[fit_index].mean())
        blocks = raw_blocks(frame, fit_index, prevalence)
        _, bl1, bl2 = repair.fit_grouped_design(prevalence=prevalence, **blocks)
        matrices = {"BL1": bl1, "BL2": bl2}
        references, ref_rows = reference_objectives(matrices, labels[fit_index], origin=origin_text)
        reference_rows.extend(ref_rows)
        specs = [("ADAM", 0.03, [30, 100, 300, 1000])]
        specs.extend(
            ("SGD", float(lr), [100, 300, 1000, 3000])
            for lr in [0.1, 0.03, 0.01, 0.003, 0.001]
        )
        for optimizer_name, lr, checkpoints in specs:
            for model_id in ("BL1", "BL2"):
                progress(f"preflight {origin_text} {optimizer_name} lr={lr:g} {model_id}")
                fit = gpu.fit_trajectory(
                    matrices[model_id], labels[fit_index], device=device,
                    optimizer_name=optimizer_name, learning_rate=lr,
                    checkpoints=checkpoints, alpha=1e-4,
                )
                for trace in fit.objective_trace:
                    decision = gpu.adequacy(
                        float(trace["objective"]), references[model_id], reference_converged=True,
                    )
                    row = {
                        "origin": origin_text, "optimizer": optimizer_name,
                        "learning_rate": lr, "steps": int(trace["step"]),
                        "model_id": model_id, "terminal_gradient_norm_at_max_checkpoint":
                        fit.terminal_gradient_norm if int(trace["step"]) == fit.steps else None,
                        "trajectory_elapsed_seconds": fit.elapsed_seconds,
                        "trajectory_peak_cuda_memory_bytes": fit.peak_cuda_memory_bytes,
                        **decision,
                    }
                    adequacy_rows.append(row)
                    trajectories[(optimizer_name, lr, int(trace["step"]), origin_text, model_id)] = row
        del matrices, bl1, bl2, blocks
        gc.collect()

    origins = list(contract["optimizer_adequacy"]["train_probe_origins"])
    def eligible(name: str, lr: float, steps: int) -> tuple[bool, float]:
        rows = [trajectories[(name, lr, steps, origin, model)] for origin in origins for model in ("BL1", "BL2")]
        return all(bool(row["adequacy_passed"]) for row in rows), max(float(row["objective_regret"]) for row in rows)

    adam_candidates = []
    for steps in [30, 100, 300, 1000]:
        passed, regret = eligible("ADAM", 0.03, steps)
        if passed:
            adam_candidates.append((steps, regret, 0.03))
    sgd_candidates = []
    for steps in [100, 300, 1000, 3000]:
        for lr in [0.1, 0.03, 0.01, 0.003, 0.001]:
            passed, regret = eligible("SGD", lr, steps)
            if passed:
                sgd_candidates.append((steps, regret, lr))
    if not adam_candidates or not sgd_candidates:
        write_csv(out / "optimization_adequacy.csv", adequacy_rows)
        raise ContractStop("no adequate configuration for one or both GPU optimizers")
    adam = min(adam_candidates)
    sgd = min(sgd_candidates)
    selected = {
        "status": "complete_frozen_before_validation",
        "ADAM": {"learning_rate": adam[2], "steps": adam[0], "maximum_regret": adam[1]},
        "SGD": {"learning_rate": sgd[2], "steps": sgd[0], "maximum_regret": sgd[1]},
    }
    write_csv(out / "optimization_adequacy.csv", adequacy_rows)
    write_csv(out / "reference_solver_audit.csv", reference_rows)
    write_json(OUTPUT_ROOT / "selected_optimizer_configuration_manifest.json", selected)
    finalize_hashes(out)
    return selected


def read_frame_with_defaults(path: Path) -> canonical.Frame:
    table = pq.read_table(path)
    columns = {
        name: table.column(name).combine_chunks().to_numpy(zero_copy_only=False)
        for name in canonical.REQUIRED_COLUMNS
    }
    n = table.num_rows
    columns.update({"warm_user": np.ones(n, bool), "warm_video": np.ones(n, bool), "behavior_cold_video": np.zeros(n, bool)})
    return canonical.Frame(columns=columns)


@dataclass
class FrozenModels:
    design: repair.GroupedDesign
    fits: dict[str, gpu.GpuFit]
    calibrators: dict[str, repair.Calibrator]
    bl0: float


def fit_standard_stage(
    frame: canonical.Frame,
    *,
    stage: str,
    fit_range: tuple[str, str],
    calibration_date: str,
    target_range: tuple[str, str],
    selected: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray], FrozenModels, list[dict[str, Any]], list[dict[str, Any]]]:
    dates = frame.dates()
    labels = frame.labels()
    fit_lo, fit_hi = (np.datetime64(value, "D") for value in fit_range)
    cal_date = np.datetime64(calibration_date, "D")
    target_lo, target_hi = (np.datetime64(value, "D") for value in target_range)
    fit_index = np.flatnonzero((dates >= fit_lo) & (dates <= fit_hi))
    cal_index = np.flatnonzero(dates == cal_date)
    target_index = np.flatnonzero((dates >= target_lo) & (dates <= target_hi))
    if np.unique(labels[cal_index]).size != 2:
        raise ContractStop(f"{stage} calibration lacks both classes")
    split = canonical.OriginSplit(
        origin=stage, fit_index=fit_index, calibration_index=cal_index,
        assessment_index=target_index, calibration_date=calibration_date,
        bl0_probability=float(labels[dates < target_lo].mean()),
        fit_prevalence=float(labels[fit_index].mean()),
    )
    origin = canonical.build_origin_matrices(frame, split)
    references, reference_rows = reference_objectives(
        origin.matrices["fit"], origin.labels["fit"], origin=stage,
    )
    predictions: dict[str, np.ndarray] = {
        "BL0": np.full(target_index.size, split.bl0_probability, dtype=np.float64)
    }
    fits: dict[str, gpu.GpuFit] = {}
    calibrators: dict[str, repair.Calibrator] = {}
    adequacy_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for optimizer_name in ("ADAM", "SGD"):
        config = selected[optimizer_name]
        for model_id in ("BL1", "BL2"):
            stream = f"{optimizer_name}_{model_id}"
            progress(f"{stage}: fit {stream} on GPU")
            fit = gpu.fit_trajectory(
                origin.matrices["fit"][model_id], origin.labels["fit"], device=device,
                optimizer_name=optimizer_name, learning_rate=float(config["learning_rate"]),
                checkpoints=[int(config["steps"])], alpha=1e-4,
            )
            decision = gpu.adequacy(fit.objective, references[model_id], reference_converged=True)
            adequacy_rows.append({
                "stage": stage, "stream": stream, "model_id": model_id,
                "optimizer": optimizer_name, "learning_rate": fit.learning_rate,
                "steps": fit.steps, "terminal_gradient_norm": fit.terminal_gradient_norm,
                "elapsed_seconds": fit.elapsed_seconds,
                "peak_cuda_memory_bytes": fit.peak_cuda_memory_bytes, **decision,
            })
            if not decision["adequacy_passed"]:
                raise ContractStop(f"{stage} optimizer adequacy failed: {stream}")
            raw_cal, _ = gpu.score(
                origin.matrices["calibration"][model_id], fit.coefficient, fit.intercept,
                device=device,
            )
            raw_target, _ = gpu.score(
                origin.matrices["assessment"][model_id], fit.coefficient, fit.intercept,
                device=device,
            )
            calibrator = repair.fit_previous_day_sigmoid(
                raw_cal, origin.labels["calibration"], user_id=origin.users["calibration"],
            )
            probability = calibrator.apply(raw_target)
            repair.assert_calibration_monotone(raw_target, probability)
            predictions[stream] = probability
            predictions[f"raw_{stream}"] = raw_target
            fits[stream] = fit
            calibrators[stream] = calibrator
            calibration_rows.append({
                "stage": stage, "stream": stream, "fit_rows": calibrator.fit_rows,
                "fit_users": calibrator.fit_users, "fit_positives": calibrator.fit_positives,
                "intercept": calibrator.intercept, "slope": calibrator.slope,
                "n_iter": calibrator.n_iter, "convergence_warning_count": calibrator.convergence_warning_count,
            })
    reference_rows = [{"stage": stage, **row} for row in reference_rows]
    return target_index, predictions, FrozenModels(origin.design, fits, calibrators, split.bl0_probability), adequacy_rows + reference_rows, calibration_rows


def score_random_stage(
    frame: canonical.Frame, frozen: FrozenModels, *, device: torch.device
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    index = np.arange(frame.size)
    blocks = raw_blocks(frame, index, frozen.design.prevalence)
    bl1, bl2 = repair.transform_grouped(frozen.design, **blocks)
    repair.assert_column_prefix(bl1, bl2)
    matrices = {"BL1": bl1, "BL2": bl2}
    predictions: dict[str, np.ndarray] = {"BL0": np.full(frame.size, frozen.bl0)}
    for stream, fit in frozen.fits.items():
        model_id = stream.split("_")[1]
        raw, _ = gpu.score(matrices[model_id], fit.coefficient, fit.intercept, device=device)
        probability = frozen.calibrators[stream].apply(raw)
        repair.assert_calibration_monotone(raw, probability)
        predictions[stream] = probability
        predictions[f"raw_{stream}"] = raw
    return index, predictions


def probability_distribution(probability: np.ndarray) -> dict[str, Any]:
    p = np.asarray(probability, dtype=np.float64)
    q = np.quantile(p, [0, .001, .01, .1, .5, .9, .99, .999, 1])
    return {
        "rows": p.size, "finite_share": float(np.mean(np.isfinite(p))),
        "below_or_equal_1e_6_share": float(np.mean(p <= 1e-6)),
        "above_or_equal_1_minus_1e_6_share": float(np.mean(p >= 1 - 1e-6)),
        **dict(zip(("minimum", "p001", "p01", "p10", "median", "p90", "p99", "p999", "maximum"), map(float, q))),
    }


def safe_point(labels: np.ndarray, probability: np.ndarray, users: np.ndarray) -> dict[str, Any] | None:
    if labels.size == 0 or np.unique(labels).size != 2:
        return None
    return metrics.point_metrics(labels, probability, users, epsilon=repair.METRIC_CLIP_LOW)


def bootstrap_all(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    users: np.ndarray,
    user_universe: np.ndarray,
    multiplicities: np.ndarray,
) -> list[dict[str, Any]]:
    y = np.asarray(labels, dtype=np.int8)
    row_user = metrics._user_index(users, user_universe)
    user_count = user_universe.size
    user_rows = np.bincount(row_user, minlength=user_count).astype(np.float64)
    replicate_rows = multiplicities @ user_rows
    points = {stream: metrics.point_metrics(y, predictions[stream], users, epsilon=repair.METRIC_CLIP_LOW) for stream in STREAMS}
    values: dict[str, dict[str, np.ndarray]] = {name: {} for name in ("average_precision", "log_loss", "brier", "user_gauc_event_weighted", "user_gauc_user_equal")}
    for stream in STREAMS:
        p = metrics.clipped(predictions[stream], repair.METRIC_CLIP_LOW)
        values["average_precision"][stream] = metrics.weighted_ap_replicates(
            y, p, row_user, multiplicities, block_size=8, epsilon=repair.METRIC_CLIP_LOW,
        )
        log_row = -(y * np.log(p) + (1-y) * np.log1p(-p))
        brier_row = np.square(p-y)
        values["log_loss"][stream] = (multiplicities @ metrics._per_user_sums(log_row, row_user, user_count)) / replicate_rows
        values["brier"][stream] = (multiplicities @ metrics._per_user_sums(brier_row, row_user, user_count)) / replicate_rows
        gauc = metrics.user_gauc_components(y, p, users, user_universe=user_universe, epsilon=repair.METRIC_CLIP_LOW)
        eligible = gauc.eligible.astype(float)
        eligible_rows = gauc.event_counts.astype(float) * eligible
        auc = np.nan_to_num(gauc.auc, nan=0.0)
        values["user_gauc_event_weighted"][stream] = (multiplicities @ (auc * eligible_rows)) / (multiplicities @ eligible_rows)
        values["user_gauc_user_equal"][stream] = (multiplicities @ (auc * eligible)) / (multiplicities @ eligible)
    rows: list[dict[str, Any]] = []
    for contrast, coefficients in CONTRASTS.items():
        for metric_name, by_stream in values.items():
            replicate = sum(coefficient * by_stream[stream] for stream, coefficient in coefficients.items())
            point = sum(coefficient * float(points[stream][metric_name]) for stream, coefficient in coefficients.items())
            valid = replicate[np.isfinite(replicate)]
            rows.append({
                "contrast": contrast, "metric": metric_name, "point_estimate": point,
                "bootstrap_replicates_requested": multiplicities.shape[0],
                "effective_replicates": valid.size, "bootstrap_mean": float(valid.mean()),
                "bootstrap_se": float(valid.std(ddof=1)),
                "ci95_lower": float(np.quantile(valid, .025)),
                "ci95_upper": float(np.quantile(valid, .975)),
            })
    return rows


def stage_decision(
    stage: str, pooled: list[dict[str, Any]], bootstrap: list[dict[str, Any]], daily: list[dict[str, Any]], probability_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    point = {(row["scope"], row["model_id"]): row for row in pooled}
    boot = {(row["contrast"], row["metric"]): row for row in bootstrap}
    warm = "primary_warm_user"
    static: dict[str, bool] = {}
    history: dict[str, bool] = {}
    absolute: dict[str, bool] = {}
    for optimizer in ("ADAM", "SGD"):
        static_contrast = f"{optimizer}_BL1_minus_BL0"
        h_contrast = f"{optimizer}_BL2_minus_{optimizer}_BL1"
        static[optimizer] = bool(
            boot[(static_contrast, "average_precision")]["ci95_lower"] > 0
            and boot[(static_contrast, "user_gauc_event_weighted")]["point_estimate"] >= 0
            and boot[(static_contrast, "log_loss")]["point_estimate"] <= 0
            and boot[(static_contrast, "brier")]["point_estimate"] <= 0
        )
        history[optimizer] = bool(
            boot[(h_contrast, "average_precision")]["point_estimate"] >= .005
            and boot[(h_contrast, "average_precision")]["ci95_lower"] > 0
            and boot[(h_contrast, "user_gauc_event_weighted")]["point_estimate"] >= 0
            and boot[(h_contrast, "log_loss")]["point_estimate"] <= 0
            and boot[(h_contrast, "brier")]["point_estimate"] <= 0
        )
        for model in ("BL1", "BL2"):
            stream = f"{optimizer}_{model}"
            absolute[stream] = bool(
                point[(warm, stream)]["log_loss"] <= point[(warm, "BL0")]["log_loss"] + 1e-10
                and point[(warm, stream)]["brier"] <= point[(warm, "BL0")]["brier"] + 1e-10
            )
    positive_days: dict[str, int] = {}
    for optimizer in ("ADAM", "SGD"):
        by_date: dict[str, dict[str, float]] = {}
        for row in daily:
            if row["population"] == warm and row["model_id"] in {f"{optimizer}_BL1", f"{optimizer}_BL2"}:
                by_date.setdefault(row["event_date"], {})[row["model_id"]] = row["average_precision"]
        positive_days[optimizer] = sum(values[f"{optimizer}_BL2"] > values[f"{optimizer}_BL1"] for values in by_date.values() if len(values) == 2)
    saturation = all(
        row["below_or_equal_1e_6_share"] <= .05 and row["above_or_equal_1_minus_1e_6_share"] <= .05
        for row in probability_rows if row["model_id"] in LEARNED
    )
    daily_requirement = 0 if stage == "random_audit" else (3 if stage == "validation" else 12)
    if stage == "random_audit":
        # Static/absolute comparisons to the standard BL0 are descriptive under
        # the random-exposure prevalence shift; only transport + health gate it.
        passed = bool(saturation and all(history.values()))
    else:
        passed = bool(
            saturation and all(static.values()) and all(absolute.values())
            and all(history.values())
            and all(value >= daily_requirement for value in positive_days.values())
        )
    return {
        "stage": stage, "scientific_status": "pass" if passed else "fail_or_mixed",
        "scientific_failure_blocks_next_stage": False,
        "static_baseline_gate": static, "absolute_probability_gate": absolute,
        "history_gate": history, "positive_AP_days": positive_days,
        "required_positive_AP_days": daily_requirement, "probability_saturation_gate": saturation,
    }


def evaluate_and_freeze(
    frame: canonical.Frame,
    target_index: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    stage: str,
    stage_dir: Path,
    contract_sha: str,
    feature_info: dict[str, Any],
    adequacy_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    environment: dict[str, Any],
) -> dict[str, Any]:
    labels = frame.labels()[target_index]
    users = frame.users()[target_index]
    dates = frame.dates()[target_index].astype(str)
    identities = {
        name: frame.columns[name][target_index]
        for name in ("source_table", "source_row_number", "user_id", "video_id", "event_date", "time_ms", "long_view", "prior_batch_n")
    }
    table_data = {
        key: value for key, value in identities.items() if key != "prior_batch_n"
    }
    table_data["prior_batch_count"] = identities["prior_batch_n"]
    for stream in STREAMS:
        table_data[f"p_{stream}"] = predictions[stream]
    for stream in LEARNED:
        table_data[f"raw_{stream}"] = predictions[f"raw_{stream}"]
    prediction_path = stage_dir / "predictions.parquet"
    pq.write_table(pa.table(table_data), prediction_path, compression="zstd", row_group_size=100000)
    prediction_sha = sha256_file(prediction_path)
    target_rows = [
        {"source_table": str(source), "source_row_number": int(row)}
        for source, row in zip(identities["source_table"], identities["source_row_number"])
    ]
    write_csv(stage_dir / "target_row_manifest.csv", target_rows)
    target_sha = sha256_file(stage_dir / "target_row_manifest.csv")
    user_universe = np.unique(users)
    multiplicities, multiplicity_sha = metrics.make_multiplicities(
        user_count=user_universe.size, replicates=2000, seed=20260814,
    )
    np.save(stage_dir / "bootstrap_multiplicity.npy", multiplicities, allow_pickle=False)

    warm = frame.columns["warm_user"][target_index].astype(bool)
    masks: dict[str, np.ndarray] = {
        "primary_warm_user": warm, "all_target_rows": np.ones(labels.size, bool),
        "cold_user": ~warm,
        "history_0_49": identities["prior_batch_n"] < 50,
        "history_50_199": (identities["prior_batch_n"] >= 50) & (identities["prior_batch_n"] < 200),
        "history_200_plus": identities["prior_batch_n"] >= 200,
        "history_500_plus_exploratory": identities["prior_batch_n"] >= 500,
        "warm_video": frame.columns["warm_video"][target_index].astype(bool),
        "behavior_cold_video": frame.columns["behavior_cold_video"][target_index].astype(bool),
        "tag_known": frame.columns["static_tag_missing"][target_index] == 0,
        "tag_unknown": frame.columns["static_tag_missing"][target_index] == 1,
        "video_duration_valid": frame.columns["static_duration_valid"][target_index] == 1,
        "video_duration_invalid": frame.columns["static_duration_valid"][target_index] == 0,
    }
    pooled_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    for scope, mask in masks.items():
        for stream in STREAMS:
            point = safe_point(labels[mask], predictions[stream][mask], users[mask])
            if point is not None:
                row = {"scope": scope, "model_id": stream, **point}
                (pooled_rows if scope in {"primary_warm_user", "all_target_rows", "cold_user"} else slice_rows).append(row)
    daily_rows: list[dict[str, Any]] = []
    for day in sorted(np.unique(dates)):
        for population, pop_mask in (("primary_warm_user", warm), ("all_target_rows", np.ones(labels.size, bool))):
            mask = (dates == day) & pop_mask
            for stream in STREAMS:
                point = safe_point(labels[mask], predictions[stream][mask], users[mask])
                if point is not None:
                    daily_rows.append({"event_date": day, "population": population, "model_id": stream, **point})
    if not warm.any() or np.unique(labels[warm]).size != 2:
        raise ContractStop(f"{stage} warm primary population cannot be evaluated")
    bootstrap_rows = bootstrap_all(labels[warm], {key: value[warm] for key, value in predictions.items() if key in STREAMS}, users[warm], user_universe, multiplicities)
    if min(int(row["effective_replicates"]) for row in bootstrap_rows) < 1950:
        raise ContractStop(f"{stage} has insufficient effective bootstrap replicates")
    contrast_rows = [{key: row[key] for key in ("contrast", "metric", "point_estimate")} for row in bootstrap_rows]
    probability_rows = [{"scope": "all_target_rows", "model_id": stream, **probability_distribution(predictions[stream])} for stream in LEARNED]
    if any(row["finite_share"] != 1.0 or row["minimum"] < 0 or row["maximum"] > 1 for row in probability_rows):
        raise ContractStop(f"{stage} invalid probability")

    write_csv(stage_dir / "pooled_metrics.csv", pooled_rows)
    write_csv(stage_dir / "daily_metrics.csv", daily_rows)
    write_csv(stage_dir / "model_contrasts.csv", contrast_rows)
    write_csv(stage_dir / "paired_user_cluster_bootstrap.csv", bootstrap_rows)
    write_csv(stage_dir / "history_depth_slice_metrics.csv", slice_rows)
    write_csv(stage_dir / "probability_distribution_audit.csv", probability_rows)
    write_csv(stage_dir / "optimization_adequacy.csv", adequacy_rows)
    write_csv(stage_dir / "calibration_audit.csv", calibration_rows)
    write_csv(stage_dir / "usage_ledger.csv", [row for row in adequacy_rows if "elapsed_seconds" in row])
    write_json(stage_dir / "model_manifest.json", {
        "streams": list(STREAMS), "prediction_sha256": prediction_sha,
        "target_manifest_sha256": target_sha, "bootstrap_multiplicity_sha256": multiplicity_sha,
    })
    write_json(stage_dir / "environment_and_hardware_manifest.json", environment)
    decision = stage_decision(stage, pooled_rows, bootstrap_rows, daily_rows, probability_rows)
    write_json(stage_dir / "stage_decision.json", decision)
    write_json(stage_dir / "run_manifest.json", {
        "status": "complete", "stage": stage, "contract_sha256": contract_sha,
        "generated_at": datetime.now().astimezone().isoformat(),
        "rows": labels.size, "users": user_universe.size, "positives": int(labels.sum()),
        "feature_manifest": feature_info, "prediction_sha256": prediction_sha,
        "target_manifest_sha256": target_sha, "bootstrap_multiplicity_sha256": multiplicity_sha,
        "decision": decision,
    })
    finalize_hashes(stage_dir)
    return {"decision": decision, "pooled": pooled_rows, "bootstrap": bootstrap_rows, "daily": daily_rows}


def finalize_hashes(directory: Path) -> None:
    path = directory / "artifact_hash_manifest.json"
    files = [item for item in sorted(directory.iterdir()) if item.is_file() and item != path]
    def manifest_name(item: Path) -> str:
        try:
            return str(item.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(item.relative_to(directory))
    write_json(path, {"hash_algorithm": "SHA-256", "artifacts": [
        {"path": manifest_name(item), "size_bytes": item.stat().st_size, "sha256": sha256_file(item)}
        for item in files
    ]})


def append_access(stage: str, status: str, contract_sha: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "stage_access_ledger.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "timestamp": datetime.now().astimezone().isoformat(), "stage": stage,
            "status": status, "contract_sha256": contract_sha,
        }, sort_keys=True) + "\n")


def metric_lookup(result: dict[str, Any], stage: str, contrast: str, metric: str) -> dict[str, Any]:
    for row in result["bootstrap"]:
        if row["contrast"] == contrast and row["metric"] == metric:
            return row
    raise KeyError((stage, contrast, metric))


def render_report(results: dict[str, dict[str, Any]], contract_sha: str, selected: dict[str, Any]) -> str:
    lines = [
        "# 历史特征价值 GPU 顺序确认实验 v001", "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        f"- 最终合同 SHA-256：`{contract_sha}`", "- 环境：conda `Kuai`；GPU Adam 与 GPU SGD；五条对齐预测流。",
        "- 证据限制：restricted 与 random 窗口此前看过聚合标签/质量摘要，因此不是 pristine sealed data；此前未读取候选模型预测或 BL2-BL1 对比。", "",
        "## 数据来源与背景", "",
        "数据来自 KuaiRand 正式 Silver 快照：early standard、late standard、random exposures，以及仅回补官方 `long_view` 的公式不一致隔离行；静态内容元数据来自 `videos_basic`。标准目标为 `tab=1`，random 目标为全部正式随机曝光。历史严格使用 `history_time_ms < target_time_ms`；random 标签从不更新历史。", "",
        "核心问题只有一个：在完全相同的静态基线 BL1 上增加 H2 严格用户历史，是否在 Validation、partially-unblinded sealed test 和 random audit 上稳定提升，并同时经得住 GPU Adam 与 GPU SGD。", "",
        "## 冻结优化器配置", "",
        f"- Adam：lr={selected['ADAM']['learning_rate']}，steps={selected['ADAM']['steps']}。",
        f"- SGD：lr={selected['SGD']['learning_rate']}，steps={selected['SGD']['steps']}。", "",
        "## 主要结果", "",
        "| 阶段 | Adam ΔAP (95% CI) | SGD ΔAP (95% CI) | Adam gate | SGD gate | 阶段结论 |", "|---|---:|---:|---|---|---|",
    ]
    for stage in ("validation", "sealed_test", "random_audit"):
        result = results[stage]
        adam = metric_lookup(result, stage, "ADAM_BL2_minus_ADAM_BL1", "average_precision")
        sgd = metric_lookup(result, stage, "SGD_BL2_minus_SGD_BL1", "average_precision")
        decision = result["decision"]
        lines.append(
            f"| {stage} | {adam['point_estimate']:.6f} [{adam['ci95_lower']:.6f}, {adam['ci95_upper']:.6f}] | "
            f"{sgd['point_estimate']:.6f} [{sgd['ci95_lower']:.6f}, {sgd['ci95_upper']:.6f}] | "
            f"{decision['history_gate']['ADAM']} | {decision['history_gate']['SGD']} | {decision['scientific_status']} |"
        )
    lines.extend(["", "## 分析与结论", ""])
    validation = results["validation"]["decision"]
    sealed = results["sealed_test"]["decision"]
    random = results["random_audit"]["decision"]
    if all(validation["history_gate"].values()) and all(sealed["history_gate"].values()) and all(random["history_gate"].values()):
        conclusion = "三个阶段在两种优化器下均支持 H2 历史增量；在冻结的离线候选响应协议内，历史有稳定且可迁移的预测价值。"
    elif all(validation["history_gate"].values()) and all(sealed["history_gate"].values()):
        conclusion = "标准推荐曝光上的时间外证据支持历史增量，但 random audit 未在两种优化器下同时通过；结论应限制在标准曝光。"
    elif not all(validation["history_gate"].values()):
        conclusion = "Validation 未在两种优化器下同时通过，因此广义稳定历史增量已在首个确认阶段被证伪；后续结果只用于刻画时间异质性，不能抹去该失败。"
    else:
        conclusion = "Validation 支持但 sealed test 未持续支持，历史增量没有跨更长时间窗保持稳定。"
    lines.extend([
        conclusion, "", "静态基线、概率健康、逐日方向、gAUC、log-loss 与 Brier 门均保存在各阶段 `stage_decision.json`；所有逐模型与对比结果为可直接作图的 CSV。", "",
        "这是一项离线预测确认，不证明因果价值、线上业务提升或全量召回质量。", "",
        "## 可作图与统计文件", "",
        "每个阶段目录均包含 `pooled_metrics.csv`、`daily_metrics.csv`、`model_contrasts.csv`、`paired_user_cluster_bootstrap.csv`、`history_depth_slice_metrics.csv` 和 `probability_distribution_audit.csv`。Bootstrap 为 2,000 次配对用户簇重采样。", "",
    ])
    return "\n".join(lines)


def run_release(approved_hash: str) -> None:
    contract, digest = load_contract()
    if approved_hash != digest:
        raise ContractStop(f"approved contract hash mismatch: observed {digest}")
    verify_approval_receipt(digest)
    implementation_records(contract)
    environment = environment_manifest(contract, require_cuda=True)
    device = torch.device("cuda:0")
    if OUTPUT_ROOT.exists():
        raise ContractStop("release output root already exists; overwrite is forbidden")
    OUTPUT_ROOT.mkdir(parents=True)
    append_access("preflight", "opened", digest)
    selected = preflight(contract, device)
    append_access("preflight", "complete_hashed", digest)

    results: dict[str, dict[str, Any]] = {}
    completed_hashed = {"preflight"}
    gpu.assert_stage_access("validation", completed_hashed)
    append_access("validation", "opened_after_preflight", digest)
    val_dir = OUTPUT_ROOT / "validation"
    val_dir.mkdir()
    val_path, val_feature = materialize_stage_features(contract, "validation", val_dir)
    val_frame = read_frame(val_path)
    val_index, val_predictions, _, val_adequacy, val_calibration = fit_standard_stage(
        val_frame, stage="validation", fit_range=("2022-04-08", "2022-04-16"),
        calibration_date="2022-04-17", target_range=("2022-04-18", "2022-04-21"),
        selected=selected, device=device,
    )
    results["validation"] = evaluate_and_freeze(
        val_frame, val_index, val_predictions, stage="validation", stage_dir=val_dir,
        contract_sha=digest, feature_info=val_feature, adequacy_rows=val_adequacy,
        calibration_rows=val_calibration, environment=environment,
    )
    append_access("validation", "complete_hashed_scientific_result_preserved", digest)
    completed_hashed.add("validation")
    del val_frame, val_predictions
    gc.collect()

    gpu.assert_stage_access("sealed_test", completed_hashed)
    append_access("sealed_test", "opened_after_validation_complete_regardless_of_scientific_result", digest)
    sealed_dir = OUTPUT_ROOT / "sealed_test"
    sealed_dir.mkdir()
    sealed_path, sealed_feature = materialize_stage_features(contract, "sealed_test", sealed_dir)
    sealed_frame = read_frame(sealed_path)
    sealed_index, sealed_predictions, frozen, sealed_adequacy, sealed_calibration = fit_standard_stage(
        sealed_frame, stage="sealed_test", fit_range=("2022-04-08", "2022-04-20"),
        calibration_date="2022-04-21", target_range=("2022-04-22", "2022-05-08"),
        selected=selected, device=device,
    )
    results["sealed_test"] = evaluate_and_freeze(
        sealed_frame, sealed_index, sealed_predictions, stage="sealed_test", stage_dir=sealed_dir,
        contract_sha=digest, feature_info=sealed_feature, adequacy_rows=sealed_adequacy,
        calibration_rows=sealed_calibration, environment=environment,
    )
    append_access("sealed_test", "complete_hashed_scientific_result_preserved", digest)
    completed_hashed.add("sealed_test")
    del sealed_frame, sealed_predictions
    gc.collect()

    gpu.assert_stage_access("random_audit", completed_hashed)
    append_access("random_audit", "opened_after_sealed_complete_regardless_of_scientific_result", digest)
    random_dir = OUTPUT_ROOT / "random_audit"
    random_dir.mkdir()
    random_path, random_feature = materialize_stage_features(contract, "random_audit", random_dir)
    random_frame = read_frame(random_path)
    random_index, random_predictions = score_random_stage(random_frame, frozen, device=device)
    copied_adequacy = [{"stage": "random_audit", "source": "frozen_sealed_test", **row} for row in sealed_adequacy]
    copied_calibration = [{"stage": "random_audit", "source": "frozen_sealed_test", **row} for row in sealed_calibration]
    results["random_audit"] = evaluate_and_freeze(
        random_frame, random_index, random_predictions, stage="random_audit", stage_dir=random_dir,
        contract_sha=digest, feature_info=random_feature, adequacy_rows=copied_adequacy,
        calibration_rows=copied_calibration, environment=environment,
    )
    append_access("random_audit", "complete_hashed", digest)
    final = {
        "status": "complete", "contract_sha256": digest,
        "stage_decisions": {stage: result["decision"] for stage, result in results.items()},
    }
    write_json(OUTPUT_ROOT / "final_claim_decision.json", final)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(results, digest, selected), encoding="utf-8")
    finalize_hashes(OUTPUT_ROOT)
    progress(f"release complete; report={REPORT_PATH}")


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
        raise ContractStop("--approved-contract-sha256 is required for release")
    run_release(args.approved_contract_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
