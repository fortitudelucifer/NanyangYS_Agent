"""Train-only CUDA engineering demo for the Gate 2B frozen feature matrix.

This entry point is deliberately separate from the governed v003 release
runner.  It reuses the frozen input, temporal split builder and grouped feature
preprocessing, but replaces sklearn's CPU optimiser with a PyTorch sparse CUDA
optimiser for a bounded demonstration run.

The output is *not* a v003 release, is not eligible for a research checkpoint,
and must not be used to authorize Gold, Validation or sequence models.  Its
purpose is to prove that the local RTX GPU is genuinely used and to provide an
honest, reproducible timing/metric demo without changing the CPU canonical
contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_gate2b_probability_repair_v003 as canonical  # noqa: E402
from kuairand_longseq.evaluation import gate2b_metrics as metrics  # noqa: E402
from kuairand_longseq.models import gate2b_repair_v003 as repair  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "reports/generated/gate2b_gpu_demo"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/gate2b_gpu_demo.md"
SEED = 20260814


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Return a deterministic uint64 mixing function for bounded demo rows."""

    x = np.asarray(values, dtype=np.uint64).copy()
    x += np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def deterministic_subset(
    frame: canonical.Frame,
    index: np.ndarray,
    limit: int | None,
    *,
    salt: int,
) -> np.ndarray:
    """Choose stable identity-hash rows without looking at labels or features."""

    candidates = np.asarray(index, dtype=np.int64)
    if limit is None or limit <= 0 or candidates.size <= limit:
        return candidates.copy()
    row_number = frame.columns["source_row_number"][candidates].astype(np.uint64, copy=False)
    source_table = frame.columns["source_table"][candidates]
    _, table_code = np.unique(source_table, return_inverse=True)
    identity = row_number ^ (table_code.astype(np.uint64) << np.uint64(56))
    key = _splitmix64(identity ^ np.uint64(SEED + salt))
    selected_position = np.argpartition(key, int(limit) - 1)[: int(limit)]
    # Restore source-artifact order so downstream preprocessing remains stable.
    return np.sort(candidates[selected_position])


def scipy_csr_to_torch(matrix: sparse.csr_matrix, device: torch.device) -> torch.Tensor:
    """Copy a validated SciPy CSR matrix to a torch sparse CSR tensor."""

    csr = matrix.tocsr().astype(np.float32, copy=False)
    crow = torch.as_tensor(csr.indptr, dtype=torch.int64, device=device)
    col = torch.as_tensor(csr.indices, dtype=torch.int64, device=device)
    values = torch.as_tensor(csr.data, dtype=torch.float32, device=device)
    # PyTorch still labels sparse CSR as beta and emits a process-level warning
    # even when invariant policy is supplied explicitly.  The matrix has already
    # been canonicalised by SciPy; keep that library-status warning out of the
    # project's warnings-as-errors test suite.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"torch(\..*)?")
        warnings.filterwarnings(
            "ignore", message=r"Sparse invariant checks are implicitly disabled.*"
        )
        warnings.filterwarnings(
            "ignore", message=r"Sparse CSR tensor support is in beta state.*"
        )
        return torch.sparse_csr_tensor(
            crow,
            col,
            values,
            size=csr.shape,
            dtype=torch.float32,
            device=device,
            check_invariants=False,
        )


def _sparse_logits(matrix: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(matrix, weight[:, None]).squeeze(1) + bias


def train_sparse_logistic(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    *,
    device: torch.device,
    steps: int,
    learning_rate: float,
    alpha: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Fit a full-batch sparse logistic model on the requested torch device.

    Adam is intentionally recorded as a non-parity engineering optimiser.  The
    objective itself matches the v003 normalised objective:
    ``mean_log_loss + alpha / 2 * ||w||^2``; the intercept is not regularised.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0 or alpha <= 0.0:
        raise ValueError("learning rate and alpha must be positive")

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()

    tensor = scipy_csr_to_torch(matrix, device)
    target = torch.as_tensor(
        np.asarray(labels, dtype=np.float32), dtype=torch.float32, device=device
    )
    weight = torch.zeros(matrix.shape[1], dtype=torch.float32, device=device, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=float(learning_rate))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    losses: list[float] = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        score = _sparse_logits(tensor, weight, bias)
        objective = F.binary_cross_entropy_with_logits(score, target)
        objective = objective + 0.5 * float(alpha) * torch.dot(weight, weight)
        objective.backward()
        optimizer.step()
        losses.append(float(objective.detach().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    coefficient = weight.detach().cpu().numpy().astype(np.float64, copy=False)
    intercept = float(bias.detach().cpu())
    peak_memory = (
        int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
    )
    del tensor, target, weight, bias, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return coefficient, intercept, {
        "device": str(device),
        "steps": int(steps),
        "elapsed_seconds": float(elapsed),
        "seconds_per_step": float(elapsed / steps),
        "initial_objective": losses[0],
        "final_objective": losses[-1],
        "objective_reduction": losses[0] - losses[-1],
        "peak_cuda_memory_bytes": peak_memory,
        "coefficient_l2_norm": float(np.linalg.norm(coefficient)),
        "coefficient_absolute_maximum": float(np.max(np.abs(coefficient))),
        "intercept": intercept,
    }


def score_sparse(
    matrix: sparse.csr_matrix,
    coefficient: np.ndarray,
    intercept: float,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    tensor = scipy_csr_to_torch(matrix, device)
    weight = torch.as_tensor(coefficient, dtype=torch.float32, device=device)
    bias = torch.tensor(intercept, dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        score = _sparse_logits(tensor, weight, bias)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = score.cpu().numpy().astype(np.float64, copy=False)
    del tensor, weight, bias, score
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, float(elapsed)


def probability_metrics(
    labels: np.ndarray, probability: np.ndarray, users: np.ndarray
) -> dict[str, Any]:
    return metrics.point_metrics(
        labels,
        probability,
        users,
        epsilon=repair.METRIC_CLIP_LOW,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Gate 2B GPU engineering demo",
        "",
        "> **非正式实验结果。** 本运行仅证明 CUDA 工程路径与速度；它不是 v003 "
        "canonical release，不支持 Gold、Validation 或序列模型晋级。",
        "",
        "## 运行证据",
        "",
        f"- 设备：`{manifest['hardware']['device_name']}`",
        f"- PyTorch：`{manifest['environment']['torch']}`；编译 CUDA："
        f"`{manifest['environment']['torch_compiled_cuda']}`",
        f"- 冻结输入 SHA-256：`{manifest['input']['sha256']}`",
        f"- Origin：`{manifest['demo_scope']['origin']}`；fit/calibration/assessment 行数："
        f"`{manifest['demo_scope']['fit_rows']:,}` / "
        f"`{manifest['demo_scope']['calibration_rows']:,}` / "
        f"`{manifest['demo_scope']['assessment_rows']:,}`",
        "- CUDA 负责：稀疏线性模型优化、calibration/assessment raw-score 计算。",
        "- CPU 负责：Parquet 读取、冻结切分、OneHot/缩放、前一日 sigmoid calibration 和指标。",
        "",
        "## 模型结果（demo-only）",
        "",
        "| 模型 | GPU 训练秒 | AP | user-GAUC | Log Loss | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_id in ("BL1", "BL2"):
        row = manifest["models"][model_id]
        metric = row["calibrated_metrics"]
        lines.append(
            f"| {model_id} | {row['gpu_training']['elapsed_seconds']:.3f} | "
            f"{metric['average_precision']:.6f} | "
            f"{metric['user_gauc_event_weighted']:.6f} | "
            f"{metric['log_loss']:.6f} | {metric['brier']:.6f} |"
        )
    delta = manifest["demo_deltas"]["BL2_minus_BL1"]
    lines.extend(
        [
            "",
            "BL2−BL1："
            f"`ΔAP={delta['average_precision']:+.6f}`、"
            f"`Δuser-GAUC={delta['user_gauc_event_weighted']:+.6f}`、"
            f"`ΔLog Loss={delta['log_loss']:+.6f}`、"
            f"`ΔBrier={delta['brier']:+.6f}`。",
            "",
        ]
    )
    benchmark = manifest.get("benchmark")
    if benchmark:
        lines.extend(
            [
                "## 同矩阵 CPU/GPU 训练计时",
                "",
                f"- 模型：`{benchmark['model_id']}`；相同步数："
                f"`{benchmark['steps']}`。",
                f"- CPU：`{benchmark['cpu_seconds_per_step']:.6f}` 秒/步。",
                f"- GPU：`{benchmark['gpu_seconds_per_step']:.6f}` 秒/步。",
                f"- 训练核速度比：`{benchmark['speedup_gpu_over_cpu']:.2f}x`。",
                "",
            ]
        )
    lines.extend(
        [
            "## 解释边界",
            "",
            "- GPU demo 使用 `torch.optim.Adam`，不复现 sklearn `SGDClassifier` 的 "
            "adaptive schedule 与 averaged-SGD 语义。",
            "- 如果使用行数上限，行由 source identity 的固定哈希选择；没有读取标签来抽样，"
            "但仍不属于正式固定行协议。",
            "- 任何正式 v003 结论仍必须来自 CPU canonical runner 的全绿测试、最终哈希、"
            "独立复核、所有者批准和一次受管 release。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default="2022-04-14")
    parser.add_argument("--fit-limit", type=int, default=400_000)
    parser.add_argument("--calibration-limit", type=int, default=100_000)
    parser.add_argument("--assessment-limit", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument(
        "--benchmark-cpu",
        action="store_true",
        help="Run the same BL2 torch objective on CPU for an honest kernel speed comparison.",
    )
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run outside a device-restricted sandbox and verify nvidia-smi."
        )
    device = torch.device("cuda:0")
    torch.set_num_threads(int(args.cpu_threads))

    contract, contract_sha = canonical.load_contract()
    verified_inputs = canonical.verify_inputs(contract)
    input_path = PROJECT_ROOT / contract["input_allowlist"][0]["path"]

    load_started = time.perf_counter()
    frame = canonical.Frame.read(input_path)
    population = canonical.verify_population(contract, frame)
    splits = canonical.build_splits(contract, frame)
    split_lookup = {split.origin: split for split in splits}
    if args.origin not in split_lookup:
        raise ValueError(f"unknown origin {args.origin}; choices={sorted(split_lookup)}")
    original_split = split_lookup[args.origin]
    fit_index = deterministic_subset(frame, original_split.fit_index, args.fit_limit, salt=1)
    calibration_index = deterministic_subset(
        frame, original_split.calibration_index, args.calibration_limit, salt=2
    )
    assessment_index = deterministic_subset(
        frame, original_split.assessment_index, args.assessment_limit, salt=3
    )
    labels = frame.labels()
    demo_split = replace(
        original_split,
        fit_index=fit_index,
        calibration_index=calibration_index,
        assessment_index=assessment_index,
        fit_prevalence=float(labels[fit_index].mean(dtype=np.float64)),
        bl0_probability=float(labels[fit_index].mean(dtype=np.float64)),
    )
    data_load_seconds = time.perf_counter() - load_started

    preprocessing_started = time.perf_counter()
    origin = canonical.build_origin_matrices(frame, demo_split)
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    model_results: dict[str, Any] = {}
    for model_id in ("BL1", "BL2"):
        coefficient, intercept, training = train_sparse_logistic(
            origin.matrices["fit"][model_id],
            origin.labels["fit"],
            device=device,
            steps=int(args.steps),
            learning_rate=float(args.learning_rate),
            alpha=float(args.alpha),
        )
        raw_calibration, calibration_score_seconds = score_sparse(
            origin.matrices["calibration"][model_id], coefficient, intercept, device
        )
        raw_assessment, assessment_score_seconds = score_sparse(
            origin.matrices["assessment"][model_id], coefficient, intercept, device
        )
        calibrator = repair.fit_previous_day_sigmoid(
            raw_calibration,
            origin.labels["calibration"],
            user_id=origin.users["calibration"],
        )
        probability = calibrator.apply(raw_assessment)
        repair.assert_calibration_monotone(raw_assessment, probability)
        calibrated_metrics = probability_metrics(
            origin.labels["assessment"], probability, origin.users["assessment"]
        )
        uncalibrated_probability = 1.0 / (1.0 + np.exp(-np.clip(raw_assessment, -50, 50)))
        model_results[model_id] = {
            "matrix": {
                "rows": int(origin.matrices["fit"][model_id].shape[0]),
                "columns": int(origin.matrices["fit"][model_id].shape[1]),
                "nonzeros": int(origin.matrices["fit"][model_id].nnz),
            },
            "gpu_training": training,
            "gpu_scoring": {
                "calibration_seconds": calibration_score_seconds,
                "assessment_seconds": assessment_score_seconds,
            },
            "calibration": {
                "fit_rows": calibrator.fit_rows,
                "fit_users": calibrator.fit_users,
                "fit_positives": calibrator.fit_positives,
                "intercept": calibrator.intercept,
                "slope": calibrator.slope,
                "n_iter": calibrator.n_iter,
            },
            "uncalibrated_metrics": probability_metrics(
                origin.labels["assessment"],
                uncalibrated_probability,
                origin.users["assessment"],
            ),
            "calibrated_metrics": calibrated_metrics,
        }

    metric_names = (
        "average_precision",
        "user_gauc_event_weighted",
        "log_loss",
        "brier",
    )
    delta = {
        name: float(
            model_results["BL2"]["calibrated_metrics"][name]
            - model_results["BL1"]["calibrated_metrics"][name]
        )
        for name in metric_names
    }

    benchmark: dict[str, Any] | None = None
    if args.benchmark_cpu:
        _, _, cpu_training = train_sparse_logistic(
            origin.matrices["fit"]["BL2"],
            origin.labels["fit"],
            device=torch.device("cpu"),
            steps=int(args.steps),
            learning_rate=float(args.learning_rate),
            alpha=float(args.alpha),
        )
        gpu_step = float(model_results["BL2"]["gpu_training"]["seconds_per_step"])
        cpu_step = float(cpu_training["seconds_per_step"])
        benchmark = {
            "model_id": "BL2",
            "steps": int(args.steps),
            "cpu_threads": int(args.cpu_threads),
            "cpu_seconds_per_step": cpu_step,
            "gpu_seconds_per_step": gpu_step,
            "speedup_gpu_over_cpu": cpu_step / gpu_step,
            "comparison_scope": "same_sparse_matrix_same_torch_objective_same_steps",
        }

    manifest: dict[str, Any] = {
        "status": "complete",
        "demo_only": True,
        "checkpoint_eligible": False,
        "canonical_v003_release": False,
        "authorization_execution_authorized_was_not_changed": True,
        "generated_at": datetime.now().astimezone().isoformat(),
        "claim_boundary": (
            "GPU engineering timing and bounded Train-only demonstration only; "
            "not a scientific Gate 2B v003 result"
        ),
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "torch": torch.__version__,
            "torch_compiled_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "hardware": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "device_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        },
        "input": {
            "path": str(input_path.relative_to(PROJECT_ROOT)),
            "size_bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
            "contract_verification": verified_inputs,
            "population": population,
        },
        "source_contract": {
            "path": str(canonical.CONTRACT_PATH.relative_to(PROJECT_ROOT)),
            "sha256": contract_sha,
            "accelerator_declared_by_canonical_contract": contract["operational_budget"][
                "accelerator"
            ],
            "canonical_contract_modified": False,
        },
        "demo_scope": {
            "origin": args.origin,
            "selection": "deterministic_source_identity_hash_without_label_access",
            "fit_rows": int(fit_index.size),
            "calibration_rows": int(calibration_index.size),
            "assessment_rows": int(assessment_index.size),
            "fit_prevalence": demo_split.fit_prevalence,
            "full_origin_fit_rows": int(original_split.fit_index.size),
            "full_origin_calibration_rows": int(original_split.calibration_index.size),
            "full_origin_assessment_rows": int(original_split.assessment_index.size),
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "alpha": float(args.alpha),
            "optimizer": "torch.optim.Adam_noncanonical_demo",
            "objective": "mean_binary_log_loss_plus_alpha_over_2_times_weight_L2_squared",
        },
        "timing": {
            "parquet_load_population_and_split_seconds": data_load_seconds,
            "cpu_grouped_preprocessing_seconds": preprocessing_seconds,
        },
        "models": model_results,
        "demo_deltas": {"BL2_minus_BL1": delta},
        "benchmark": benchmark,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    report = render_report(manifest)
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"manifest={manifest_path}")
    print(f"report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
