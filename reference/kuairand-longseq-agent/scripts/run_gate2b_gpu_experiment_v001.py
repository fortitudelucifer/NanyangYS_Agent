"""Full seven-origin Train-only GPU experiment and report generator.

This owner-authorized exploratory experiment inherits the frozen data,
temporal splits, grouped BL1/BL2 preprocessing, previous-day calibration,
metrics and gates from the reviewed v003 protocol.  The fitted sparse logistic
optimizer is PyTorch Adam on CUDA, so the run is intentionally separate from
the CPU/sklearn canonical v003 release.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_gate2b_gpu_demo as gpu  # noqa: E402
import run_gate2b_probability_repair_v003 as canonical  # noqa: E402
from kuairand_longseq.evaluation import gate2b_metrics as metrics  # noqa: E402
from kuairand_longseq.models import gate2b_repair_v003 as repair  # noqa: E402


CONTRACT_PATH = PROJECT_ROOT / "configs/gate2b_gpu_experiment_contract_v001.yaml"
OUTPUT_DIR = PROJECT_ROOT / "reports/generated/gate2b_gpu_experiment_v001"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/gate2b_gpu_experiment_results_v001.md"


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def point_metrics(labels: np.ndarray, probability: np.ndarray, users: np.ndarray) -> dict[str, Any]:
    return metrics.point_metrics(
        labels, probability, users, epsilon=repair.METRIC_CLIP_LOW
    )


def distribution(probability: np.ndarray) -> dict[str, float]:
    return canonical.probability_distribution(probability)


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(candidate[name] - baseline[name])
        for name in (
            "average_precision",
            "user_gauc_event_weighted",
            "log_loss",
            "brier",
            "ece20_equal_width",
        )
    }


def probability_gate(
    source_contract: dict[str, Any],
    daily_rows: list[dict[str, Any]],
    pooled: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    spec = source_contract["probability_quality_gate"]
    tolerance = float(spec["numerical_comparison_tolerance"])
    stability = spec["daily_stability_requirements"]
    results: list[dict[str, Any]] = []
    for model_id in ("BL1", "BL2"):
        same_day_log = sum(
            row["log_loss"] <= row["BL0_log_loss"] + tolerance
            for row in daily_rows
            if row["model_id"] == model_id
        )
        same_day_brier = sum(
            row["brier"] <= row["BL0_brier"] + tolerance
            for row in daily_rows
            if row["model_id"] == model_id
        )
        result = {
            "model_id": model_id,
            "pooled_log_loss_minus_BL0": float(
                pooled[model_id]["log_loss"] - pooled["BL0"]["log_loss"]
            ),
            "pooled_brier_minus_BL0": float(
                pooled[model_id]["brier"] - pooled["BL0"]["brier"]
            ),
            "nonworse_log_loss_days": int(same_day_log),
            "nonworse_brier_days": int(same_day_brier),
            "total_days": 7,
        }
        result["passed"] = bool(
            result["pooled_log_loss_minus_BL0"] <= tolerance
            and result["pooled_brier_minus_BL0"] <= tolerance
            and same_day_log >= int(stability["minimum_nonworse_log_loss_days"])
            and same_day_brier >= int(stability["minimum_nonworse_brier_days"])
        )
        results.append(result)
    return {"per_model": results, "passed": all(row["passed"] for row in results)}


def relative_gate(
    source_contract: dict[str, Any],
    daily_rows: list[dict[str, Any]],
    pooled: dict[str, dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    requirements = source_contract["relative_history_gate"]["daily_requirements"]
    daily_by_key = {(row["origin"], row["model_id"]): row for row in daily_rows}
    origins = sorted({row["origin"] for row in daily_rows})
    positive_ap_days = sum(
        daily_by_key[(origin, "BL2")]["average_precision"]
        > daily_by_key[(origin, "BL1")]["average_precision"]
        for origin in origins
    )
    nonnegative_gauc_days = sum(
        daily_by_key[(origin, "BL2")]["user_gauc_event_weighted"]
        >= daily_by_key[(origin, "BL1")]["user_gauc_event_weighted"]
        for origin in origins
    )
    ap_bootstrap = next(
        row
        for row in bootstrap_rows
        if row["contrast"] == "BL2_minus_BL1" and row["metric"] == "average_precision"
    )
    delta = metric_delta(pooled["BL2"], pooled["BL1"])
    result = {
        **{f"delta_{name}": value for name, value in delta.items()},
        "positive_average_precision_days": int(positive_ap_days),
        "nonnegative_user_gauc_days": int(nonnegative_gauc_days),
        "total_days": 7,
        "average_precision_ci_lower": float(ap_bootstrap["ci95_lower"]),
        "average_precision_ci_upper": float(ap_bootstrap["ci95_upper"]),
    }
    result["passed"] = bool(
        result["delta_average_precision"] > 0
        and result["delta_user_gauc_event_weighted"] >= 0
        and result["delta_log_loss"] <= 0
        and result["delta_brier"] <= 0
        and positive_ap_days
        >= int(requirements["minimum_positive_average_precision_days"])
        and nonnegative_gauc_days
        >= int(requirements["minimum_nonnegative_user_gauc_days"])
        and result["average_precision_ci_lower"] > 0
    )
    return result


def render_report(result: dict[str, Any]) -> str:
    pooled = result["pooled_metrics"]
    relative = result["relative_history_gate"]
    probability = result["probability_quality_gate"]
    lines = [
        "# Gate 2B 全七日 GPU 探索实验结果（v001）",
        "",
        "> 结论边界：这是项目所有者授权的完整 Train-only GPU 探索实验。它使用全部七个"
        " rolling-origin 和冻结输入，但 PyTorch Adam 不等同于 canonical v003 的 sklearn "
        "SGD，因此不能标记为 canonical release 或独立确认性证据。",
        "",
        "## 结论先行",
        "",
    ]
    if probability["passed"] and relative["passed"]:
        lines.append(
            "GPU 协议下，BL1/BL2 的绝对概率质量门与 BL2 相对 BL1 的历史增量门均通过。"
            "这支持“严格历史在该 GPU 训练协议中同时改善排序与概率质量”的 Train-only "
            "探索性结论；仍需 canonical sklearn release 才能决定正式 Gate 2B 状态。"
        )
    elif not probability["passed"]:
        lines.append(
            "GPU 协议未通过绝对概率质量门。即使 BL2 的排序优于 BL1，也不能把该输出解释为"
            "合格概率模型；应停留在基线修复层。"
        )
    else:
        lines.append(
            "GPU 协议通过绝对概率质量门，但 BL2 相对 BL1 的历史增量门未通过；严格历史"
            "没有在全部注册条件上形成稳定增量。"
        )
    lines += [
        "",
        "## 实验范围与实现",
        "",
        f"- 数据：冻结 Train-only 特征矩阵，SHA-256 `{result['input']['sha256']}`。",
        f"- 样本：七日 assessment 共 `{result['scope']['assessment_rows']:,}` 行，"
        f"`{result['scope']['assessment_users']}` 个用户。",
        f"- GPU：`{result['hardware']['device_name']}`，PyTorch "
        f"`{result['environment']['torch']}`。",
        "- 特征：BL1 静态稀疏特征；BL2 在 BL1 上增加严格 point-in-time 用户历史。",
        "- 校准：只使用 assessment 前一日数据的 sigmoid calibration。",
        "- 推断：2,000 次 paired user-cluster bootstrap。",
        "",
        "## 七日 pooled 指标",
        "",
        "| 模型 | AP | user-GAUC | Log Loss | Brier | ECE20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_id in ("BL0", "BL1", "BL2"):
        row = pooled[model_id]
        lines.append(
            f"| {model_id} | {row['average_precision']:.6f} | "
            f"{row['user_gauc_event_weighted']:.6f} | {row['log_loss']:.6f} | "
            f"{row['brier']:.6f} | {row['ece20_equal_width']:.6f} |"
        )
    lines += [
        "",
        "BL2−BL1："
        f"`ΔAP={relative['delta_average_precision']:+.6f}`、"
        f"`Δuser-GAUC={relative['delta_user_gauc_event_weighted']:+.6f}`、"
        f"`ΔLog Loss={relative['delta_log_loss']:+.6f}`、"
        f"`ΔBrier={relative['delta_brier']:+.6f}`。",
        "",
        "## 逐日结果",
        "",
        "| 日期 | 模型 | AP | user-GAUC | Log Loss | Brier |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["daily_metrics"]:
        lines.append(
            f"| {row['origin']} | {row['model_id']} | {row['average_precision']:.6f} | "
            f"{row['user_gauc_event_weighted']:.6f} | {row['log_loss']:.6f} | "
            f"{row['brier']:.6f} |"
        )
    lines += [
        "",
        "## 门禁判定",
        "",
        f"- 绝对概率质量门：`{probability['passed']}`。",
    ]
    for row in probability["per_model"]:
        lines.append(
            f"  - {row['model_id']}：pooled ΔLog Loss vs BL0="
            f"`{row['pooled_log_loss_minus_BL0']:+.6f}`，pooled ΔBrier="
            f"`{row['pooled_brier_minus_BL0']:+.6f}`，逐日非劣="
            f"`{row['nonworse_log_loss_days']}/7`、`{row['nonworse_brier_days']}/7`，"
            f"passed=`{row['passed']}`。"
        )
    lines += [
        f"- 相对历史增量门：`{relative['passed']}`；AP 正向日 "
        f"`{relative['positive_average_precision_days']}/7`，user-GAUC 非负日 "
        f"`{relative['nonnegative_user_gauc_days']}/7`。",
        f"- BL2−BL1 ΔAP 用户聚类 bootstrap 95% CI："
        f"`[{relative['average_precision_ci_lower']:+.6f}, "
        f"{relative['average_precision_ci_upper']:+.6f}]`。",
        "",
        "## Bootstrap 不确定性",
        "",
        "| 对比 | 指标 | 点估计 | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for row in result["bootstrap"]:
        lines.append(
            f"| {row['contrast']} | {row['metric']} | {row['point_estimate']:+.6f} | "
            f"[{row['ci95_lower']:+.6f}, {row['ci95_upper']:+.6f}] |"
        )
    lines += [
        "",
        "## 历史深度切片（描述性）",
        "",
        "| 切片 | 行数 | prevalence | BL1 AP | BL2 AP | ΔAP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["slice_metrics"]:
        if row["scope"] != "pooled_7_days" or row["BL1_metric_average_precision"] is None:
            continue
        lines.append(
            f"| {row['slice']} | {row['rows']:,} | {row['prevalence']:.6f} | "
            f"{row['BL1_metric_average_precision']:.6f} | "
            f"{row['BL2_metric_average_precision']:.6f} | "
            f"{row['paired_delta_average_precision']:+.6f} |"
        )
    lines += [
        "",
        "## 可发布与不可发布",
        "",
        "- 可发布：该冻结 Train-only GPU 协议下的 BL0/BL1/BL2 指标、逐日稳定性、"
        "校准、bootstrap 和切片结果。",
        "- 不可发布：这是 canonical v003、独立确认、Validation/测试泛化、线上因果收益，"
        "或据此已经批准 Gold/序列模型。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract_path = args.contract.resolve()
    experiment = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not experiment["authorization"]["execution_authorized"]:
        raise RuntimeError("GPU experiment contract is not authorized")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run outside a device-restricted sandbox")
    device = torch.device("cuda:0")

    source_contract, source_contract_sha = canonical.load_contract()
    expected_source_sha = experiment["source_protocol"][
        "canonical_contract_sha256_at_creation"
    ]
    if source_contract_sha != expected_source_sha:
        raise RuntimeError(
            f"source contract hash changed: expected {expected_source_sha}, "
            f"observed {source_contract_sha}"
        )
    canonical.verify_implementation_hashes(source_contract)
    inputs = canonical.verify_inputs(source_contract)
    input_path = PROJECT_ROOT / experiment["input"]["path"]
    if input_path.stat().st_size != int(experiment["input"]["expected_size_bytes"]):
        raise RuntimeError("input size mismatch")
    if sha256_file(input_path) != experiment["input"]["expected_sha256"]:
        raise RuntimeError("input SHA-256 mismatch")

    execution = experiment["execution"]
    started = time.perf_counter()
    frame = canonical.Frame.read(input_path)
    population = canonical.verify_population(source_contract, frame)
    splits = canonical.build_splits(source_contract, frame)

    daily_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    prediction_tables: list[pa.Table] = []
    pooled_labels: list[np.ndarray] = []
    pooled_users: list[np.ndarray] = []
    pooled_prior_batch: list[np.ndarray] = []
    pooled_predictions: dict[str, list[np.ndarray]] = {"BL0": [], "BL1": [], "BL2": []}

    for split in splits:
        print(f"[{time.strftime('%H:%M:%S')}] build origin {split.origin}", flush=True)
        preprocessing_started = time.perf_counter()
        origin = canonical.build_origin_matrices(frame, split)
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        labels = origin.labels["assessment"]
        users = origin.users["assessment"]
        predictions: dict[str, np.ndarray] = {
            "BL0": np.full(labels.size, split.bl0_probability, dtype=np.float64)
        }
        raw_scores: dict[str, np.ndarray] = {}

        for model_id in ("BL1", "BL2"):
            coefficient, intercept, training = gpu.train_sparse_logistic(
                origin.matrices["fit"][model_id],
                origin.labels["fit"],
                device=device,
                steps=int(execution["steps"]),
                learning_rate=float(execution["learning_rate"]),
                alpha=float(execution["alpha"]),
            )
            raw_calibration, calibration_scoring_seconds = gpu.score_sparse(
                origin.matrices["calibration"][model_id], coefficient, intercept, device
            )
            raw_assessment, assessment_scoring_seconds = gpu.score_sparse(
                origin.matrices["assessment"][model_id], coefficient, intercept, device
            )
            calibrator = repair.fit_previous_day_sigmoid(
                raw_calibration,
                origin.labels["calibration"],
                user_id=origin.users["calibration"],
            )
            probability = calibrator.apply(raw_assessment)
            repair.assert_calibration_monotone(raw_assessment, probability)
            predictions[model_id] = probability
            raw_scores[model_id] = raw_assessment
            training_rows.append(
                {
                    "origin": split.origin,
                    "model_id": model_id,
                    "fit_rows": int(origin.labels["fit"].size),
                    "feature_columns": int(origin.matrices["fit"][model_id].shape[1]),
                    "nonzeros": int(origin.matrices["fit"][model_id].nnz),
                    "preprocessing_seconds_shared_origin": preprocessing_seconds,
                    "gpu_training_seconds": training["elapsed_seconds"],
                    "gpu_calibration_scoring_seconds": calibration_scoring_seconds,
                    "gpu_assessment_scoring_seconds": assessment_scoring_seconds,
                    "initial_objective": training["initial_objective"],
                    "final_objective": training["final_objective"],
                    "coefficient_l2_norm": training["coefficient_l2_norm"],
                    "intercept": training["intercept"],
                    "peak_cuda_memory_bytes": training["peak_cuda_memory_bytes"],
                }
            )
            calibration_rows.append(
                {
                    "origin": split.origin,
                    "model_id": model_id,
                    "calibration_rows": calibrator.fit_rows,
                    "calibration_users": calibrator.fit_users,
                    "calibration_positives": calibrator.fit_positives,
                    "calibration_intercept": calibrator.intercept,
                    "calibration_slope": calibrator.slope,
                    "n_iter": calibrator.n_iter,
                }
            )

        origin_metric: dict[str, dict[str, Any]] = {}
        for model_id in ("BL0", "BL1", "BL2"):
            point = point_metrics(labels, predictions[model_id], users)
            origin_metric[model_id] = point
            daily_rows.append(
                {
                    "origin": split.origin,
                    "model_id": model_id,
                    **point,
                    "BL0_log_loss": None,
                    "BL0_brier": None,
                }
            )
            probability_rows.append(
                {
                    "scope": split.origin,
                    "model_id": model_id,
                    **distribution(predictions[model_id]),
                }
            )
        for row in daily_rows[-3:]:
            row["BL0_log_loss"] = origin_metric["BL0"]["log_loss"]
            row["BL0_brier"] = origin_metric["BL0"]["brier"]

        assessment_index = split.assessment_index
        prediction_tables.append(
            pa.table(
                {
                    "source_table": frame.columns["source_table"][assessment_index],
                    "source_row_number": frame.columns["source_row_number"][assessment_index],
                    "user_id": frame.columns["user_id"][assessment_index],
                    "video_id": frame.columns["video_id"][assessment_index],
                    "origin": np.repeat(split.origin, labels.size),
                    "long_view": labels,
                    "prior_batch_n": frame.columns["prior_batch_n"][assessment_index],
                    "p_BL0": predictions["BL0"],
                    "p_BL1": predictions["BL1"],
                    "p_BL2": predictions["BL2"],
                    "raw_score_BL1": raw_scores["BL1"],
                    "raw_score_BL2": raw_scores["BL2"],
                }
            )
        )
        pooled_labels.append(labels.copy())
        pooled_users.append(users.copy())
        pooled_prior_batch.append(frame.columns["prior_batch_n"][assessment_index].copy())
        for model_id in ("BL0", "BL1", "BL2"):
            pooled_predictions[model_id].append(predictions[model_id].copy())
        print(
            f"[{time.strftime('%H:%M:%S')}] origin {split.origin} complete: "
            f"BL1 AP={origin_metric['BL1']['average_precision']:.6f}, "
            f"BL2 AP={origin_metric['BL2']['average_precision']:.6f}",
            flush=True,
        )
        del origin, predictions, raw_scores
        gc.collect()
        torch.cuda.empty_cache()

    labels = np.concatenate(pooled_labels)
    users = np.concatenate(pooled_users)
    prior_batch = np.concatenate(pooled_prior_batch)
    predictions = {
        model_id: np.concatenate(parts) for model_id, parts in pooled_predictions.items()
    }
    pooled = {
        model_id: point_metrics(labels, probability, users)
        for model_id, probability in predictions.items()
    }
    for model_id, probability in predictions.items():
        probability_rows.append(
            {"scope": "pooled_7_days", "model_id": model_id, **distribution(probability)}
        )

    bootstrap_spec = experiment["evaluation"]["bootstrap"]
    user_universe = np.unique(users)
    if user_universe.size != int(bootstrap_spec["expected_users"]):
        raise RuntimeError("bootstrap user universe mismatch")
    multiplicities, multiplicity_sha = metrics.make_multiplicities(
        user_count=int(user_universe.size),
        replicates=int(bootstrap_spec["replicates"]),
        seed=int(bootstrap_spec["seed"]),
    )
    if multiplicity_sha != bootstrap_spec["expected_multiplicity_sha256"]:
        raise RuntimeError("bootstrap multiplicity digest mismatch")
    bootstrap_rows: list[dict[str, Any]] = []
    for baseline, candidate, contrast in (
        ("BL1", "BL2", "BL2_minus_BL1"),
        ("BL0", "BL1", "BL1_minus_BL0"),
    ):
        bootstrap_rows.extend(
            metrics.paired_user_cluster_bootstrap(
                labels,
                predictions[baseline],
                predictions[candidate],
                users,
                user_universe=user_universe,
                multiplicities=multiplicities,
                ap_block_size=8,
                epsilon=repair.METRIC_CLIP_LOW,
                contrast=contrast,
            )
        )

    p_gate = probability_gate(source_contract, daily_rows, pooled)
    h_gate = relative_gate(source_contract, daily_rows, pooled, bootstrap_rows)
    slices = canonical._slice_rows(
        "pooled_7_days", labels, users, prior_batch, predictions
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.concat_tables(prediction_tables),
        OUTPUT_DIR / "daily_predictions.parquet",
        compression="zstd",
    )
    write_csv(OUTPUT_DIR / "daily_metrics.csv", daily_rows)
    write_csv(
        OUTPUT_DIR / "pooled_metrics.csv",
        [{"model_id": model_id, **row} for model_id, row in pooled.items()],
    )
    write_csv(OUTPUT_DIR / "training_audit.csv", training_rows)
    write_csv(OUTPUT_DIR / "calibration_audit.csv", calibration_rows)
    write_csv(OUTPUT_DIR / "probability_distribution_audit.csv", probability_rows)
    write_csv(OUTPUT_DIR / "paired_user_cluster_bootstrap.csv", bootstrap_rows)
    write_csv(OUTPUT_DIR / "history_depth_slice_metrics.csv", slices)

    result: dict[str, Any] = {
        "status": "complete",
        "run_id": f"gate2b-gpu-v001-{time.strftime('%Y%m%d-%H%M%S')}",
        "generated_at": datetime.now().astimezone().isoformat(),
        "contract": {
            "path": str(contract_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(contract_path),
            "source_canonical_contract_sha256": source_contract_sha,
        },
        "input": {**inputs[0], "population": population},
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "torch": torch.__version__,
            "torch_compiled_cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
        "hardware": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "device_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        },
        "scope": {
            "origins": [split.origin for split in splits],
            "assessment_rows": int(labels.size),
            "assessment_users": int(user_universe.size),
            "full_frozen_rows_no_demo_limit": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "pooled_metrics": pooled,
        "daily_metrics": daily_rows,
        "probability_quality_gate": p_gate,
        "relative_history_gate": h_gate,
        "bootstrap": bootstrap_rows,
        "probability_distribution": probability_rows,
        "slice_metrics": slices,
        "claim_boundary": experiment["claim_boundary"],
    }
    report = render_report(result)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    write_json(OUTPUT_DIR / "run_manifest.json", result)

    artifact_paths = [
        OUTPUT_DIR / "daily_predictions.parquet",
        OUTPUT_DIR / "daily_metrics.csv",
        OUTPUT_DIR / "pooled_metrics.csv",
        OUTPUT_DIR / "training_audit.csv",
        OUTPUT_DIR / "calibration_audit.csv",
        OUTPUT_DIR / "probability_distribution_audit.csv",
        OUTPUT_DIR / "paired_user_cluster_bootstrap.csv",
        OUTPUT_DIR / "history_depth_slice_metrics.csv",
        OUTPUT_DIR / "run_manifest.json",
        REPORT_PATH,
    ]
    write_json(
        OUTPUT_DIR / "artifact_hash_manifest.json",
        {
            "run_id": result["run_id"],
            "hash_algorithm": "SHA-256",
            "artifacts": [
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in artifact_paths
            ],
        },
    )
    print(report)
    print(f"manifest={OUTPUT_DIR / 'run_manifest.json'}")
    print(f"report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
