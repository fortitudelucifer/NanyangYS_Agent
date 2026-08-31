#!/usr/bin/env python3
"""GPU target-domain paired retraining against frozen BL2 plus v011 calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import brentq
from scipy.special import expit


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_history_value_gpu_confirmation_v001 as v1  # noqa: E402
import run_history_value_adam_validation_v007 as v7  # noqa: E402
from kuairand_longseq.models import gate2b_repair_v003 as repair  # noqa: E402
from kuairand_longseq.models import history_value_gpu as gpu  # noqa: E402


CONTRACT_PATH = EXPERIMENT_ROOT / "contract_v012.yaml"
APPROVAL_PATH = EXPERIMENT_ROOT / "approval_v012.json"
SNAPSHOT_PATH = EXPERIMENT_ROOT / "predecessor_integrity_snapshot.json"
FEATURE_PATH = PROJECT_ROOT / "reports/generated/history_value_adam_random_v010/random_audit/features.parquet"
FROZEN_PICKLE = PROJECT_ROOT / "reports/generated/sealed_model_reconstruction_diagnostic_v010/reconstructed_frozen_models.pkl"
FROZEN_STATE = PROJECT_ROOT / "reports/generated/sealed_model_reconstruction_diagnostic_v010/reconstructed_frozen_model_state.npz"
OLD_PREDICTIONS = PROJECT_ROOT / "experiments/bl2_target_domain_calibration_v011/outputs/held_out_to_calibrator_test/predictions.parquet"
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
REPORT_PATH = EXPERIMENT_ROOT / "results_v012.md"

EXPECTED_SNAPSHOT_SHA256 = "641ea2b5e6e1e246526e7a7d99592854514093623e1a93368fb6f917a173ea1a"
AUTHORIZED_STAGES = [
    "preflight",
    "target_adaptation_train",
    "target_calibration",
    "model_selection",
    "freeze_selected_model",
    "final_temporal_replay_test",
]
CONFIG_IDS = ("C1_conservative", "C2_balanced", "C3_aggressive")
MODEL_IDS = ("NEW_BL1", "NEW_BL2")
FINAL_MODELS = ("TARGET_BL0", "OLD_BL2_PLUS_V011", "NEW_BL1", "NEW_BL2")
CONTRASTS = {
    "NEW_BL1_minus_TARGET_BL0": {"NEW_BL1": 1.0, "TARGET_BL0": -1.0},
    "NEW_BL2_minus_NEW_BL1": {"NEW_BL2": 1.0, "NEW_BL1": -1.0},
    "NEW_BL2_minus_OLD_BL2_PLUS_V011": {
        "NEW_BL2": 1.0,
        "OLD_BL2_PLUS_V011": -1.0,
    },
}


class ContractStop(RuntimeError):
    pass


@dataclass
class AdaptedFit:
    config_id: str
    model_id: str
    learning_rate: float
    steps: int
    tether_strength: float
    coefficient: np.ndarray
    intercept: float
    initial_objective: float
    final_objective: float
    terminal_gradient_norm: float
    coefficient_displacement_l2: float
    elapsed_seconds: float
    peak_cuda_memory_bytes: int
    objective_trace: list[dict[str, Any]]

    def audit_row(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "model_id": self.model_id,
            "learning_rate": self.learning_rate,
            "steps": self.steps,
            "tether_strength": self.tether_strength,
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "objective_improvement": self.initial_objective - self.final_objective,
            "terminal_gradient_norm": self.terminal_gradient_norm,
            "coefficient_displacement_l2": self.coefficient_displacement_l2,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_cuda_memory_bytes": self.peak_cuda_memory_bytes,
            "finite_parameters": bool(
                np.isfinite(self.coefficient).all() and np.isfinite(self.intercept)
            ),
        }


@dataclass(frozen=True)
class InterceptCalibrator:
    model_id: str
    slope: float
    intercept: float
    fit_rows: int
    fit_positives: int
    fit_prevalence: float
    score_evaluations: int

    def apply(self, raw: np.ndarray) -> np.ndarray:
        probability = expit(self.slope * np.asarray(raw, dtype=np.float64) + self.intercept)
        if not np.isfinite(probability).all():
            raise ContractStop(f"nonfinite calibrated probability: {self.model_id}")
        return probability

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": "M2_intercept_only",
            "slope": self.slope,
            "intercept": self.intercept,
            "fit_rows": self.fit_rows,
            "fit_positives": self.fit_positives,
            "fit_prevalence": self.fit_prevalence,
            "score_evaluations": self.score_evaluations,
        }


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> tuple[dict[str, Any], str]:
    contract = yaml.load(CONTRACT_PATH.read_text(encoding="utf-8"), Loader=v1.UniqueKeyLoader)
    if contract["contract_id"] != "bl2_target_domain_retraining_v012":
        raise ContractStop("wrong v012 contract id")
    observed_configs = tuple(row["config_id"] for row in contract["adaptation_candidates"])
    if observed_configs != CONFIG_IDS:
        raise ContractStop("v012 candidate order differs")
    if contract["authorization"]["authorized_stages_after_exact_hash_approval"] != AUTHORIZED_STAGES:
        raise ContractStop("v012 authorized stages differ")
    return contract, sha256_file(CONTRACT_PATH)


def verify_file(path: Path, *, size: int, digest: str, label: str) -> None:
    if not path.is_file() or path.stat().st_size != int(size):
        raise ContractStop(f"{label} missing or size mismatch")
    if sha256_file(path) != digest:
        raise ContractStop(f"{label} SHA mismatch")


def verify_manifest(path: Path) -> int:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for artifact in manifest.get("artifacts", []):
        item = Path(artifact["path"])
        if not item.is_absolute():
            item = PROJECT_ROOT / item
        verify_file(
            item,
            size=int(artifact["size_bytes"]),
            digest=artifact["sha256"],
            label=f"manifest artifact {item}",
        )
    return len(manifest.get("artifacts", []))


def verify_predecessors(contract: dict[str, Any]) -> dict[str, Any]:
    expected_snapshot = contract["predecessor_integrity"]
    verify_file(
        SNAPSHOT_PATH,
        size=int(expected_snapshot["snapshot_size_bytes"]),
        digest=expected_snapshot["snapshot_sha256"],
        label="v012 predecessor snapshot",
    )
    if expected_snapshot["snapshot_sha256"] != EXPECTED_SNAPSHOT_SHA256:
        raise ContractStop("v012 compiled snapshot SHA differs")
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    if snapshot["status"] != "all_declared_inputs_verified_before_v012_implementation":
        raise ContractStop("v012 predecessor snapshot status differs")
    by_role = {entry["role"]: entry for entry in snapshot["inputs"]}
    for entry in snapshot["inputs"]:
        verify_file(
            PROJECT_ROOT / entry["path"],
            size=int(entry["size_bytes"]),
            digest=entry["sha256"],
            label=f"v012 predecessor {entry['role']}",
        )
    if verify_manifest(PROJECT_ROOT / by_role["v010_random_manifest"]["path"]) != 18:
        raise ContractStop("v012 expected 18 v010 random artifacts")
    if verify_manifest(PROJECT_ROOT / by_role["v011_root_manifest"]["path"]) != 3:
        raise ContractStop("v012 expected 3 v011 root artifacts")
    v010 = json.loads((PROJECT_ROOT / by_role["v010_final_claim"]["path"]).read_text())
    if v010["terminal_interpretation"]["status"] != "history_supported_on_standard_and_random_under_frozen_Adam":
        raise ContractStop("v012 requires frozen v010 history conclusion")
    v011 = json.loads((PROJECT_ROOT / by_role["v011_final_decision"]["path"]).read_text())
    if v011["decision"]["scientific_status"] != "pass" or v011["selected_family"] != "M2_intercept_only":
        raise ContractStop("v012 requires frozen passing v011 M2")
    selected = json.loads((PROJECT_ROOT / by_role["v011_selected_calibrator"]["path"]).read_text())
    final = selected["final_calibrator"]
    if final["family"] != "M2_intercept_only":
        raise ContractStop("v012 v011 calibrator family differs")
    return {"snapshot": snapshot, "v010_status": v010["terminal_interpretation"]["status"], "v011": v011}


def verify_input_metadata(contract: dict[str, Any]) -> dict[str, Any]:
    metadata = pq.read_metadata(FEATURE_PATH)
    declared = contract["frozen_inputs"]["target_domain_features"]
    required = set(contract["frozen_inputs"]["required_feature_columns"])
    observed = set(metadata.schema.to_arrow_schema().names)
    if metadata.num_rows != int(declared["rows"]) or not required.issubset(observed):
        raise ContractStop("v012 feature metadata differs")
    old = pq.read_metadata(OLD_PREDICTIONS)
    if old.num_rows != int(contract["frozen_inputs"]["v011_old_predictions"]["rows"]):
        raise ContractStop("v012 old prediction metadata differs")
    return {
        "target_feature_rows": metadata.num_rows,
        "target_feature_columns": metadata.num_columns,
        "target_feature_sha256": sha256_file(FEATURE_PATH),
        "old_prediction_rows": old.num_rows,
        "old_prediction_sha256": sha256_file(OLD_PREDICTIONS),
    }


def verify_implementation(contract: dict[str, Any]) -> list[dict[str, str]]:
    records = []
    for entry in contract["implementation"]["result_producing_files"]:
        path = PROJECT_ROOT / entry["path"]
        observed = sha256_file(path)
        if observed != entry["sha256"]:
            raise ContractStop(f"v012 implementation SHA mismatch: {entry['path']}")
        records.append({"path": entry["path"], "sha256": observed})
    return records


def verify_environment(contract: dict[str, Any], *, require_cuda: bool) -> dict[str, Any]:
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pyarrow": pa.__version__,
        "torch": torch.__version__,
    }
    if observed != contract["environment"]["required_versions"]:
        raise ContractStop(f"v012 environment mismatch: {observed}")
    cuda = bool(torch.cuda.is_available())
    if require_cuda and not cuda:
        raise ContractStop("v012 release requires CUDA")
    device_name = torch.cuda.get_device_name(0) if cuda else None
    if require_cuda and device_name != contract["environment"]["required_gpu"]:
        raise ContractStop(f"v012 GPU differs: {device_name}")
    return {
        "versions": observed,
        "executable": sys.executable,
        "cuda_available": cuda,
        "compiled_cuda": torch.version.cuda,
        "device": device_name,
    }


def verify_approval(contract_sha: str) -> dict[str, Any]:
    if not APPROVAL_PATH.is_file():
        raise ContractStop("v012 exact-hash approval receipt is missing")
    receipt = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    required = {
        "contract_id": "bl2_target_domain_retraining_v012",
        "contract_sha256": contract_sha,
        "execution_authorized": True,
        "authorized_stages": AUTHORIZED_STAGES,
        "approved_by": "project_owner",
        "SGD_unfreeze_authorized": False,
        "test_refit_or_retry_authorized": False,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ContractStop(f"v012 approval mismatch: {key}")
    return receipt


def append_access(stage: str, status: str, contract_sha: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "stage_access_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "timestamp": datetime.now().astimezone().isoformat(),
            "stage": stage,
            "status": status,
            "contract_sha256": contract_sha,
        }, sort_keys=True) + "\n")


def split_index(dates: np.ndarray, start: str, end: str) -> np.ndarray:
    lo, hi = np.datetime64(start, "D"), np.datetime64(end, "D")
    return np.flatnonzero((dates >= lo) & (dates <= hi))


def split_audit(frame: Any, index: np.ndarray, expected: dict[str, Any], name: str) -> dict[str, Any]:
    labels = frame.labels()[index]
    users = frame.users()[index]
    identities = set(zip(
        frame.columns["source_table"][index].tolist(),
        frame.columns["source_row_number"][index].tolist(),
    ))
    if index.size != int(expected["rows"]) or int(labels.sum()) != int(expected["positives"]):
        raise ContractStop(f"v012 split count differs: {name}")
    if np.unique(users).size != int(expected["users"]) or len(identities) != index.size:
        raise ContractStop(f"v012 split identity differs: {name}")
    return {
        "split": name,
        "start": expected["start"],
        "end": expected["end"],
        "rows": int(index.size),
        "users": int(np.unique(users).size),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "unique_identities": len(identities),
    }


def load_frozen_model(contract: dict[str, Any]) -> Any:
    with FROZEN_PICKLE.open("rb") as handle:
        frozen = pickle.load(handle)
    if frozen.design.bl1_width != int(contract["frozen_design"]["BL1_width"]):
        raise ContractStop("v012 frozen BL1 width differs")
    if frozen.design.bl2_width != int(contract["frozen_design"]["BL2_width"]):
        raise ContractStop("v012 frozen BL2 width differs")
    state = np.load(FROZEN_STATE)
    for stream in ("ADAM_BL1", "ADAM_BL2"):
        if not np.array_equal(np.asarray(frozen.fits[stream].coefficient), state[f"{stream}_coefficient"]):
            raise ContractStop(f"v012 frozen coefficient differs: {stream}")
    return frozen


def tether_objective(
    matrix: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    initial_weight: torch.Tensor,
    tether_strength: float,
) -> torch.Tensor:
    raw = torch.sparse.mm(matrix, weight[:, None]).squeeze(1) + bias
    return F.binary_cross_entropy_with_logits(raw, labels) + (
        0.5 * float(tether_strength) * torch.dot(weight - initial_weight, weight - initial_weight)
    )


def fit_adapted(
    matrix: Any,
    labels: np.ndarray,
    initial_coefficient: np.ndarray,
    initial_intercept: float,
    config: dict[str, Any],
    model_id: str,
    device: torch.device,
) -> AdaptedFit:
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.cuda.empty_cache()
    x = gpu.scipy_csr_to_torch(matrix, device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    initial_weight = torch.as_tensor(initial_coefficient, dtype=torch.float32, device=device)
    weight = initial_weight.clone().detach().requires_grad_(True)
    bias = torch.tensor(float(initial_intercept), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam(
        [weight, bias],
        lr=float(config["learning_rate"]),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    checkpoints = set(int(value) for value in config["trace_checkpoints"])
    with torch.no_grad():
        initial = float(tether_objective(
            x, y, weight, bias, initial_weight, float(config["tether_strength"])
        ).cpu())
    trace = []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(1, int(config["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = tether_objective(
            x, y, weight, bias, initial_weight, float(config["tether_strength"])
        )
        if not bool(torch.isfinite(loss)):
            raise ContractStop(f"v012 nonfinite objective: {config['config_id']}/{model_id}/{step}")
        loss.backward()
        optimizer.step()
        if step in checkpoints:
            with torch.no_grad():
                value = tether_objective(
                    x, y, weight, bias, initial_weight, float(config["tether_strength"])
                )
            trace.append({"config_id": config["config_id"], "model_id": model_id, "step": step, "objective": float(value.cpu())})
    optimizer.zero_grad(set_to_none=True)
    final_loss = tether_objective(
        x, y, weight, bias, initial_weight, float(config["tether_strength"])
    )
    final_loss.backward()
    gradient_norm = float(torch.sqrt(torch.dot(weight.grad, weight.grad) + bias.grad * bias.grad).cpu())
    displacement = float(torch.linalg.vector_norm(weight.detach() - initial_weight).cpu())
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    coefficient = weight.detach().cpu().numpy().astype(np.float64, copy=False)
    intercept = float(bias.detach().cpu())
    peak = int(torch.cuda.max_memory_allocated())
    final = float(final_loss.detach().cpu())
    if final > initial + 1e-7 or not np.isfinite(coefficient).all() or not np.isfinite(intercept):
        raise ContractStop(f"v012 optimizer did not improve safely: {config['config_id']}/{model_id}")
    del x, y, initial_weight, weight, bias, optimizer, final_loss
    torch.cuda.empty_cache()
    return AdaptedFit(
        config_id=config["config_id"], model_id=model_id,
        learning_rate=float(config["learning_rate"]), steps=int(config["steps"]),
        tether_strength=float(config["tether_strength"]), coefficient=coefficient,
        intercept=intercept, initial_objective=initial, final_objective=final,
        terminal_gradient_norm=gradient_norm, coefficient_displacement_l2=displacement,
        elapsed_seconds=elapsed, peak_cuda_memory_bytes=peak, objective_trace=trace,
    )


def fit_intercept_calibrator(
    model_id: str, raw: np.ndarray, labels: np.ndarray, slope: float, contract: dict[str, Any]
) -> InterceptCalibrator:
    settings = contract["calibration"]
    target = float(np.mean(labels))
    evaluations = 0

    def score(intercept: float) -> float:
        nonlocal evaluations
        evaluations += 1
        return float(np.mean(expit(float(slope) * raw + intercept)) - target)

    intercept = brentq(
        score,
        float(settings["bracket"][0]),
        float(settings["bracket"][1]),
        xtol=float(settings["xtol"]),
        rtol=float(settings["rtol"]),
        maxiter=int(settings["max_iter"]),
        disp=True,
    )
    return InterceptCalibrator(
        model_id=model_id,
        slope=float(slope),
        intercept=float(intercept),
        fit_rows=int(labels.size),
        fit_positives=int(labels.sum()),
        fit_prevalence=target,
        score_evaluations=evaluations,
    )


def point_row(model_id: str, labels: np.ndarray, users: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    point = v1.metrics.point_metrics(labels, probability, users, epsilon=repair.METRIC_CLIP_LOW)
    return {
        "model_id": model_id,
        "rows": int(labels.size),
        "users": int(np.unique(users).size),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "mean_probability": float(np.mean(probability)),
        "mean_probability_minus_prevalence": float(np.mean(probability) - np.mean(labels)),
        **point,
    }


def probability_audit(model_id: str, probability: np.ndarray) -> dict[str, Any]:
    return {"model_id": model_id, **v1.probability_distribution(probability)}


def reliability_rows(model_id: str, labels: np.ndarray, probability: np.ndarray, bins: int = 20) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    index = np.maximum(index, 0)
    rows = []
    for bin_id in range(bins):
        mask = index == bin_id
        rows.append({
            "model_id": model_id,
            "bin_id": bin_id,
            "lower": float(edges[bin_id]),
            "upper": float(edges[bin_id + 1]),
            "rows": int(mask.sum()),
            "positives": int(labels[mask].sum()) if mask.any() else 0,
            "mean_probability": float(np.mean(probability[mask])) if mask.any() else None,
            "observed_rate": float(np.mean(labels[mask])) if mask.any() else None,
        })
    return rows


def metric_delta(left: dict[str, Any], right: dict[str, Any], metric: str) -> float:
    return float(left[metric]) - float(right[metric])


def candidate_selection(
    candidate_metrics: list[dict[str, Any]], contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexed = {(row["config_id"], row["model_id"]): row for row in candidate_metrics}
    rows = []
    gates = contract["selection_rule"]["eligibility_gates"]
    for config_id in CONFIG_IDS:
        bl0 = indexed[(config_id, "TARGET_BL0")]
        old = indexed[(config_id, "OLD_BL2_PLUS_V011")]
        bl1 = indexed[(config_id, "NEW_BL1")]
        bl2 = indexed[(config_id, "NEW_BL2")]
        row = {
            "config_id": config_id,
            "NEW_BL2_log_loss": bl2["log_loss"],
            "history_delta_AP": metric_delta(bl2, bl1, "average_precision"),
            "history_delta_event_gAUC": metric_delta(bl2, bl1, "user_gauc_event_weighted"),
            "history_delta_log_loss": metric_delta(bl2, bl1, "log_loss"),
            "history_delta_brier": metric_delta(bl2, bl1, "brier"),
            "static_delta_AP": metric_delta(bl1, bl0, "average_precision"),
            "new_minus_old_delta_AP": metric_delta(bl2, old, "average_precision"),
            "NEW_BL2_ECE20": bl2["ece20_equal_width"],
            "NEW_BL2_abs_mean_probability_gap": abs(bl2["mean_probability_minus_prevalence"]),
        }
        row["eligible"] = bool(
            row["history_delta_AP"] >= float(gates["minimum_history_delta_AP"])
            and row["history_delta_event_gAUC"] >= 0
            and row["history_delta_log_loss"] <= 0
            and row["history_delta_brier"] <= 0
            and row["static_delta_AP"] > 0
            and row["new_minus_old_delta_AP"] >= -float(gates["old_AP_noninferiority_margin"])
            and row["NEW_BL2_ECE20"] <= float(gates["maximum_ECE20"])
            and row["NEW_BL2_abs_mean_probability_gap"] <= float(gates["maximum_abs_mean_probability_gap"])
        )
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    pool = eligible if eligible else rows
    minimum = min(float(row["NEW_BL2_log_loss"]) for row in pool)
    tolerance = float(contract["selection_rule"]["simplicity_tie_tolerance_log_loss"])
    tied = [row for row in pool if float(row["NEW_BL2_log_loss"]) <= minimum + tolerance]
    order = {config_id: index for index, config_id in enumerate(CONFIG_IDS)}
    selected = min(tied, key=lambda row: order[row["config_id"]])
    decision = {
        "status": "selected_and_frozen_before_final_replay",
        "selected_config_id": selected["config_id"],
        "selected_was_eligible": bool(selected["eligible"]),
        "eligible_configs": [row["config_id"] for row in eligible],
        "minimum_pool_log_loss": minimum,
        "tie_tolerance": tolerance,
        "tied_configs": [row["config_id"] for row in tied],
        "fallback_used_because_no_eligible_candidate": not bool(eligible),
        "final_test_can_change_selection": False,
    }
    return rows, decision


def paired_bootstrap(
    labels: np.ndarray,
    users: np.ndarray,
    probabilities: dict[str, np.ndarray],
    contract: dict[str, Any],
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    user_universe = np.unique(users)
    replicates = int(contract["statistics"]["bootstrap_replicates"])
    seed = int(contract["statistics"]["bootstrap_seed"])
    multiplicities, multiplicity_sha = v1.metrics.make_multiplicities(
        user_count=user_universe.size, replicates=replicates, seed=seed
    )
    np.save(out / "bootstrap_multiplicity.npy", multiplicities, allow_pickle=False)
    row_user = v1.metrics._user_index(users, user_universe)
    user_rows = np.bincount(row_user, minlength=user_universe.size).astype(np.float64)
    denominator = multiplicities @ user_rows
    points = {
        model_id: v1.metrics.point_metrics(labels, probability, users, epsilon=repair.METRIC_CLIP_LOW)
        for model_id, probability in probabilities.items()
    }
    by_metric: dict[str, dict[str, np.ndarray]] = {
        metric: {} for metric in ("average_precision", "log_loss", "brier", "user_gauc_event_weighted")
    }
    for model_id, probability in probabilities.items():
        clipped = v1.metrics.clipped(probability, repair.METRIC_CLIP_LOW)
        by_metric["average_precision"][model_id] = v1.metrics.weighted_ap_replicates(
            labels, clipped, row_user, multiplicities, block_size=8, epsilon=repair.METRIC_CLIP_LOW
        )
        log_row = -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
        brier_row = np.square(clipped - labels)
        by_metric["log_loss"][model_id] = (
            multiplicities @ v1.metrics._per_user_sums(log_row, row_user, user_universe.size)
        ) / denominator
        by_metric["brier"][model_id] = (
            multiplicities @ v1.metrics._per_user_sums(brier_row, row_user, user_universe.size)
        ) / denominator
        components = v1.metrics.user_gauc_components(
            labels, clipped, users, user_universe=user_universe, epsilon=repair.METRIC_CLIP_LOW
        )
        eligible_rows = components.event_counts.astype(float) * components.eligible.astype(float)
        auc = np.nan_to_num(components.auc, nan=0.0)
        by_metric["user_gauc_event_weighted"][model_id] = (
            multiplicities @ (auc * eligible_rows)
        ) / (multiplicities @ eligible_rows)
    summaries = []
    replicate_rows = []
    for contrast, weights in CONTRASTS.items():
        for metric, values in by_metric.items():
            samples = sum(float(weight) * values[model_id] for model_id, weight in weights.items())
            valid = samples[np.isfinite(samples)]
            point = sum(float(weight) * points[model_id][metric] for model_id, weight in weights.items())
            summaries.append({
                "contrast": contrast,
                "metric": metric,
                "point_estimate": float(point),
                "bootstrap_replicates_requested": replicates,
                "effective_replicates": int(valid.size),
                "bootstrap_mean": float(valid.mean()),
                "bootstrap_se": float(valid.std(ddof=1)),
                "ci95_lower": float(np.quantile(valid, 0.025)),
                "ci95_upper": float(np.quantile(valid, 0.975)),
                "multiplicity_content_sha256": multiplicity_sha,
            })
            replicate_rows.extend(
                {"replicate": index, "contrast": contrast, "metric": metric, "value": float(value)}
                for index, value in enumerate(samples)
            )
    return summaries, replicate_rows


def bootstrap_lookup(rows: list[dict[str, Any]], contrast: str, metric: str) -> dict[str, Any]:
    for row in rows:
        if row["contrast"] == contrast and row["metric"] == metric:
            return row
    raise KeyError((contrast, metric))


def final_decision(
    points: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
    daily: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    selection: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    point = {row["model_id"]: row for row in points}
    audit = {row["model_id"]: row for row in audits}
    history_ap = bootstrap_lookup(bootstrap, "NEW_BL2_minus_NEW_BL1", "average_precision")
    history_gauc = bootstrap_lookup(bootstrap, "NEW_BL2_minus_NEW_BL1", "user_gauc_event_weighted")
    history_log = bootstrap_lookup(bootstrap, "NEW_BL2_minus_NEW_BL1", "log_loss")
    history_brier = bootstrap_lookup(bootstrap, "NEW_BL2_minus_NEW_BL1", "brier")
    static_ap = bootstrap_lookup(bootstrap, "NEW_BL1_minus_TARGET_BL0", "average_precision")
    static_gauc = bootstrap_lookup(bootstrap, "NEW_BL1_minus_TARGET_BL0", "user_gauc_event_weighted")
    static_log = bootstrap_lookup(bootstrap, "NEW_BL1_minus_TARGET_BL0", "log_loss")
    static_brier = bootstrap_lookup(bootstrap, "NEW_BL1_minus_TARGET_BL0", "brier")
    old_ap = bootstrap_lookup(bootstrap, "NEW_BL2_minus_OLD_BL2_PLUS_V011", "average_precision")
    old_log = bootstrap_lookup(bootstrap, "NEW_BL2_minus_OLD_BL2_PLUS_V011", "log_loss")
    old_brier = bootstrap_lookup(bootstrap, "NEW_BL2_minus_OLD_BL2_PLUS_V011", "brier")
    gates = contract["final_gates"]
    history_gate = bool(
        history_ap["point_estimate"] >= float(gates["minimum_history_delta_AP"])
        and history_ap["ci95_lower"] > 0
        and history_gauc["point_estimate"] >= 0
        and history_log["point_estimate"] <= 0
        and history_brier["point_estimate"] <= 0
    )
    static_gate = bool(
        static_ap["ci95_lower"] > 0
        and static_gauc["point_estimate"] >= 0
        and static_log["point_estimate"] <= 0
        and static_brier["point_estimate"] <= 0
    )
    probability_value_gate = bool(
        (old_log["ci95_upper"] < 0 or old_brier["ci95_upper"] < 0)
        and old_log["point_estimate"] <= 0
        and old_brier["point_estimate"] <= 0
    )
    ranking_noninferiority_gate = bool(
        old_ap["ci95_lower"] > -float(gates["old_AP_noninferiority_margin"])
    )
    new = point["NEW_BL2"]
    health = audit["NEW_BL2"]
    calibration_gate = bool(
        new["ece20_equal_width"] <= float(gates["maximum_ECE20"])
        and abs(new["mean_probability_minus_prevalence"]) <= float(gates["maximum_abs_mean_probability_gap"])
        and health["finite_share"] == 1.0
        and health["below_or_equal_1e_6_share"] <= float(gates["maximum_extreme_probability_share"])
        and health["above_or_equal_1_minus_1e_6_share"] <= float(gates["maximum_extreme_probability_share"])
    )
    by_day: dict[str, dict[str, dict[str, Any]]] = {}
    for row in daily:
        by_day.setdefault(row["event_date"], {})[row["model_id"]] = row
    positive_days = sum(
        models["NEW_BL2"]["average_precision"] > models["NEW_BL1"]["average_precision"]
        for models in by_day.values()
    )
    daily_probability_ok = all(
        abs(models["NEW_BL2"]["mean_probability_minus_prevalence"])
        <= float(gates["maximum_daily_abs_mean_probability_gap"])
        for models in by_day.values()
    )
    daily_gate = bool(
        positive_days >= int(gates["minimum_positive_history_AP_days"])
        and daily_probability_ok
    )
    all_gates = {
        "selection_eligibility_gate": bool(selection["selected_was_eligible"]),
        "history_value_gate": history_gate,
        "static_baseline_gate": static_gate,
        "retraining_probability_value_gate": probability_value_gate,
        "old_ranking_noninferiority_gate": ranking_noninferiority_gate,
        "calibration_health_gate": calibration_gate,
        "daily_stability_gate": daily_gate,
    }
    passed = all(all_gates.values())
    return {
        "scientific_status": "retraining_adds_value" if passed else "retain_frozen_BL2_plus_v011",
        "all_required_gates_passed": passed,
        "gates": all_gates,
        "positive_history_AP_days": positive_days,
        "required_positive_history_AP_days": int(gates["minimum_positive_history_AP_days"]),
        "changes_v010_history_conclusion": False,
        "changes_v011_calibration_conclusion": False,
        "evidence_level": "post_audit_target_adaptation_temporal_replay_not_pristine_confirmation",
    }


def render_report(
    digest: str,
    selection: dict[str, Any],
    selection_contrasts: list[dict[str, Any]],
    selected_fits: dict[str, AdaptedFit],
    selected_calibrators: dict[str, InterceptCalibrator],
    points: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
    decision: dict[str, Any],
    split_audits: list[dict[str, Any]],
) -> str:
    point = {row["model_id"]: row for row in points}
    lines = [
        "# BL2 目标域成对重训练结果 v012", "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        f"- 合同 SHA-256：`{digest}`",
        "- 设备：conda Kuai / CUDA / RTX 5070 Ti；优化器：GPU full-batch Adam。",
        "- SGD 继续冻结；v010/v011 制品未修改。",
        "- 证据等级：post-audit temporal replay，不是新的 pristine test。", "",
        "## 时间切分", "",
        "| split | rows | users | positives | prevalence |", "|---|---:|---:|---:|---:|",
    ]
    for row in split_audits:
        lines.append(f"| {row['split']} | {row['rows']:,} | {row['users']:,} | {row['positives']:,} | {row['prevalence']:.6f} |")
    lines.extend([
        "", "## Selection", "",
        f"选择配置：`{selection['selected_config_id']}`；selection 合格：`{selection['selected_was_eligible']}`；"
        f"fallback：`{selection['fallback_used_because_no_eligible_candidate']}`。", "",
        "| config | eligible | ΔAP BL2-BL1 | Δlogloss BL2-BL1 | ΔAP NEW-OLD | NEW_BL2 logloss |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in selection_contrasts:
        lines.append(
            f"| {row['config_id']} | {row['eligible']} | {row['history_delta_AP']:.6f} | "
            f"{row['history_delta_log_loss']:.6f} | {row['new_minus_old_delta_AP']:.6f} | "
            f"{row['NEW_BL2_log_loss']:.6f} |"
        )
    lines.extend(["", "最终冻结参数摘要：", ""])
    for model_id in MODEL_IDS:
        fit = selected_fits[model_id]
        cal = selected_calibrators[model_id]
        lines.append(
            f"- {model_id}: displacement L2=`{fit.coefficient_displacement_l2:.6f}`，"
            f"raw intercept=`{fit.intercept:.12g}`，calibration slope=`{cal.slope:.12g}`，"
            f"calibration intercept=`{cal.intercept:.12g}`。"
        )
    lines.extend([
        "", "## Final temporal replay", "",
        "| model | AP | event-gAUC | log-loss | Brier | ECE20 | mean p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for model_id in FINAL_MODELS:
        row = point[model_id]
        lines.append(
            f"| {model_id} | {row['average_precision']:.6f} | {row['user_gauc_event_weighted']:.6f} | "
            f"{row['log_loss']:.6f} | {row['brier']:.6f} | {row['ece20_equal_width']:.6f} | "
            f"{row['mean_probability']:.6f} |"
        )
    lines.extend(["", "| contrast / metric | point | 95% CI |", "|---|---:|---:|"])
    for contrast in CONTRASTS:
        for metric in ("average_precision", "user_gauc_event_weighted", "log_loss", "brier"):
            row = bootstrap_lookup(bootstrap, contrast, metric)
            lines.append(
                f"| {contrast} / {metric} | {row['point_estimate']:.6f} | "
                f"[{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}] |"
            )
    lines.extend(["", "## 决策", ""])
    for gate, value in decision["gates"].items():
        lines.append(f"- {gate}：{value}")
    lines.extend([
        f"- 最终状态：`{decision['scientific_status']}`", "",
        "若最终状态不是 `retraining_adds_value`，CPU Agent 应继续使用冻结 BL2 + v011 截距校准。",
        "任何结果都不推翻 v010 的历史排序结论，也不升级为新数据独立确认。", "",
    ])
    return "\n".join(lines)


def validate_only() -> None:
    contract, digest = load_contract()
    verify_predecessors(contract)
    verify_input_metadata(contract)
    verify_implementation(contract)
    verify_environment(contract, require_cuda=False)
    if APPROVAL_PATH.exists() or OUTPUT_ROOT.exists() or REPORT_PATH.exists():
        raise ContractStop("v012 approval/output exists before exact-hash release")
    print(f"V012_VALIDATE_ONLY_OK contract_sha256={digest}; split labels not opened; GPU training not started")


def run_experiment(approved_hash: str) -> None:
    contract, digest = load_contract()
    if digest != approved_hash:
        raise ContractStop(f"approved v012 contract SHA mismatch: observed {digest}")
    verify_approval(digest)
    predecessor = verify_predecessors(contract)
    metadata = verify_input_metadata(contract)
    implementation = verify_implementation(contract)
    environment = verify_environment(contract, require_cuda=True)
    if OUTPUT_ROOT.exists() or REPORT_PATH.exists():
        raise ContractStop("v012 refuses to overwrite prior outputs")
    OUTPUT_ROOT.mkdir(parents=True)
    append_access("preflight", "opened_without_split_labels", digest)
    preflight_dir = OUTPUT_ROOT / "preflight"
    preflight_dir.mkdir()

    frozen = load_frozen_model(contract)
    frame = v7.read_frame_with_time(FEATURE_PATH)
    dates = frame.dates()
    labels_all = frame.labels().astype(np.int8, copy=False)
    users_all = frame.users()
    split_indices = {}
    split_audits = []
    for name, spec in contract["temporal_splits"].items():
        index = split_index(dates, spec["start"], spec["end"])
        split_indices[name] = index
        split_audits.append(split_audit(frame, index, spec, name))
    union = np.concatenate(list(split_indices.values()))
    if union.size != frame.size or np.unique(union).size != frame.size:
        raise ContractStop("v012 temporal splits do not partition target rows")
    blocks = v1.raw_blocks(frame, np.arange(frame.size), frozen.design.prevalence)
    bl1_all, bl2_all = repair.transform_grouped(frozen.design, **blocks)
    repair.assert_column_prefix(bl1_all, bl2_all)
    if bl1_all.shape[1] != int(contract["frozen_design"]["BL1_width"]) or bl2_all.shape[1] != int(contract["frozen_design"]["BL2_width"]):
        raise ContractStop("v012 transformed matrix width differs")
    v1.write_json(preflight_dir / "integrity_and_data_audit.json", {
        "status": "pass", "contract_sha256": digest, "predecessor": predecessor,
        "metadata": metadata, "implementation": implementation, "environment": environment,
        "split_audits": split_audits,
        "matrix_audit": {"BL1_shape": list(bl1_all.shape), "BL1_nnz": bl1_all.nnz,
                         "BL2_shape": list(bl2_all.shape), "BL2_nnz": bl2_all.nnz},
    })
    v1.finalize_hashes(preflight_dir)
    append_access("preflight", "complete_hashed", digest)

    device = torch.device("cuda:0")
    train_index = split_indices["target_adaptation_train"]
    calibration_index = split_indices["target_calibration"]
    selection_index = split_indices["model_selection"]
    final_index = split_indices["final_temporal_replay_test"]
    matrices = {"NEW_BL1": bl1_all, "NEW_BL2": bl2_all}
    source_stream = {"NEW_BL1": "ADAM_BL1", "NEW_BL2": "ADAM_BL2"}

    append_access("target_adaptation_train", "opened_after_preflight_hash", digest)
    train_dir = OUTPUT_ROOT / "target_adaptation_train"
    train_dir.mkdir()
    fits: dict[tuple[str, str], AdaptedFit] = {}
    training_rows = []
    trace_rows = []
    for config in contract["adaptation_candidates"]:
        for model_id in MODEL_IDS:
            source_fit = frozen.fits[source_stream[model_id]]
            fit = fit_adapted(
                matrices[model_id][train_index], labels_all[train_index],
                np.asarray(source_fit.coefficient), float(source_fit.intercept),
                config, model_id, device,
            )
            fits[(config["config_id"], model_id)] = fit
            training_rows.append(fit.audit_row())
            trace_rows.extend(fit.objective_trace)
    state_payload = {}
    for (config_id, model_id), fit in fits.items():
        state_payload[f"{config_id}_{model_id}_coefficient"] = fit.coefficient
        state_payload[f"{config_id}_{model_id}_intercept"] = np.asarray([fit.intercept], dtype=np.float64)
    np.savez_compressed(train_dir / "candidate_model_states.npz", **state_payload)
    v1.write_csv(train_dir / "training_audit.csv", training_rows)
    v1.write_csv(train_dir / "objective_trace.csv", trace_rows)
    v1.write_json(train_dir / "split_audit.json", next(row for row in split_audits if row["split"] == "target_adaptation_train"))
    v1.finalize_hashes(train_dir)
    append_access("target_adaptation_train", "complete_hashed", digest)

    append_access("target_calibration", "opened_after_all_candidate_training_hashes", digest)
    calibration_dir = OUTPUT_ROOT / "target_calibration"
    calibration_dir.mkdir()
    calibrators: dict[tuple[str, str], InterceptCalibrator] = {}
    calibration_rows = []
    selection_probabilities: dict[tuple[str, str], np.ndarray] = {}
    for config in contract["adaptation_candidates"]:
        config_id = config["config_id"]
        for model_id in MODEL_IDS:
            fit = fits[(config_id, model_id)]
            raw_cal, _ = gpu.score(
                matrices[model_id][calibration_index], fit.coefficient, fit.intercept, device=device
            )
            slope = float(contract["calibration"]["frozen_slopes"][model_id])
            calibrator = fit_intercept_calibrator(
                model_id, raw_cal, labels_all[calibration_index], slope, contract
            )
            calibrators[(config_id, model_id)] = calibrator
            calibration_rows.append({"config_id": config_id, **calibrator.as_dict()})
            raw_selection, _ = gpu.score(
                matrices[model_id][selection_index], fit.coefficient, fit.intercept, device=device
            )
            selection_probabilities[(config_id, model_id)] = calibrator.apply(raw_selection)
    v1.write_csv(calibration_dir / "calibrator_parameters.csv", calibration_rows)
    v1.write_json(calibration_dir / "split_audit.json", next(row for row in split_audits if row["split"] == "target_calibration"))
    v1.finalize_hashes(calibration_dir)
    append_access("target_calibration", "complete_hashed", digest)

    old_table = pq.read_table(OLD_PREDICTIONS)
    old_source = old_table["source_table"].combine_chunks().to_numpy(zero_copy_only=False)
    old_row = old_table["source_row_number"].combine_chunks().to_numpy(zero_copy_only=False)
    old_mask = dates >= np.datetime64("2022-05-03", "D")
    if not (
        np.array_equal(old_source, frame.columns["source_table"][old_mask])
        and np.array_equal(old_row, frame.columns["source_row_number"][old_mask])
    ):
        raise ContractStop("v012 v011 comparison identities differ")
    old_probability_all = old_table["p_selected_calibrator"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    old_selection = old_probability_all[: selection_index.size]
    old_final = old_probability_all[selection_index.size :]
    if old_final.size != final_index.size:
        raise ContractStop("v012 old final comparison rows differ")
    train_calibration_index = np.concatenate([train_index, calibration_index])
    target_bl0 = float(labels_all[train_calibration_index].mean())
    expected_bl0 = float(contract["target_BL0"]["probability"])
    if abs(target_bl0 - expected_bl0) > 1e-15:
        raise ContractStop("v012 target BL0 differs")

    append_access("model_selection", "opened_after_training_and_calibration_hashes", digest)
    selection_dir = OUTPUT_ROOT / "model_selection"
    selection_dir.mkdir()
    selection_labels = labels_all[selection_index]
    selection_users = users_all[selection_index]
    candidate_metrics = []
    for config_id in CONFIG_IDS:
        probabilities = {
            "TARGET_BL0": np.full(selection_index.size, target_bl0),
            "OLD_BL2_PLUS_V011": old_selection,
            "NEW_BL1": selection_probabilities[(config_id, "NEW_BL1")],
            "NEW_BL2": selection_probabilities[(config_id, "NEW_BL2")],
        }
        for model_id, probability in probabilities.items():
            candidate_metrics.append({
                "config_id": config_id,
                **point_row(model_id, selection_labels, selection_users, probability),
            })
    selection_contrasts, selection = candidate_selection(candidate_metrics, contract)
    v1.write_csv(selection_dir / "candidate_metrics.csv", candidate_metrics)
    v1.write_csv(selection_dir / "candidate_contrasts_and_eligibility.csv", selection_contrasts)
    v1.write_json(selection_dir / "selection_decision.json", {
        **selection, "contract_sha256": digest, "final_test_opened": False,
    })
    v1.write_json(selection_dir / "split_audit.json", next(row for row in split_audits if row["split"] == "model_selection"))
    v1.finalize_hashes(selection_dir)
    append_access("model_selection", "selected_and_hashed_before_final_replay", digest)

    append_access("freeze_selected_model", "opened_before_final_replay", digest)
    frozen_dir = OUTPUT_ROOT / "frozen_selected_model"
    frozen_dir.mkdir()
    selected_id = selection["selected_config_id"]
    selected_fits = {model_id: fits[(selected_id, model_id)] for model_id in MODEL_IDS}
    selected_calibrators = {model_id: calibrators[(selected_id, model_id)] for model_id in MODEL_IDS}
    np.savez_compressed(
        frozen_dir / "selected_model_state.npz",
        **{
            f"{model_id}_coefficient": selected_fits[model_id].coefficient
            for model_id in MODEL_IDS
        },
        **{
            f"{model_id}_raw_intercept": np.asarray([selected_fits[model_id].intercept], dtype=np.float64)
            for model_id in MODEL_IDS
        },
        **{
            f"{model_id}_calibration_slope": np.asarray([selected_calibrators[model_id].slope], dtype=np.float64)
            for model_id in MODEL_IDS
        },
        **{
            f"{model_id}_calibration_intercept": np.asarray([selected_calibrators[model_id].intercept], dtype=np.float64)
            for model_id in MODEL_IDS
        },
    )
    v1.write_json(frozen_dir / "selected_model.json", {
        "contract_sha256": digest,
        "selected_config_id": selected_id,
        "selection": selection,
        "fits": {model_id: selected_fits[model_id].audit_row() for model_id in MODEL_IDS},
        "calibrators": {model_id: selected_calibrators[model_id].as_dict() for model_id in MODEL_IDS},
        "final_test_opened": False,
    })
    v1.finalize_hashes(frozen_dir)
    selected_state_sha = sha256_file(frozen_dir / "selected_model_state.npz")
    append_access("freeze_selected_model", "selected_state_frozen_and_hashed", digest)

    append_access("final_temporal_replay_test", "opened_after_selected_state_hash", digest)
    final_dir = OUTPUT_ROOT / "final_temporal_replay_test"
    final_dir.mkdir()
    final_labels = labels_all[final_index]
    final_users = users_all[final_index]
    raw_final = {}
    probability_final = {
        "TARGET_BL0": np.full(final_index.size, target_bl0),
        "OLD_BL2_PLUS_V011": old_final,
    }
    for model_id in MODEL_IDS:
        fit = selected_fits[model_id]
        raw, _ = gpu.score(
            matrices[model_id][final_index], fit.coefficient, fit.intercept, device=device
        )
        raw_final[model_id] = raw
        probability_final[model_id] = selected_calibrators[model_id].apply(raw)
    points = [
        point_row(model_id, final_labels, final_users, probability_final[model_id])
        for model_id in FINAL_MODELS
    ]
    audits = [probability_audit(model_id, probability_final[model_id]) for model_id in FINAL_MODELS]
    reliability = []
    for model_id in FINAL_MODELS:
        reliability.extend(reliability_rows(model_id, final_labels, probability_final[model_id]))
    daily = []
    final_dates = dates[final_index]
    for day in np.unique(final_dates):
        mask = final_dates == day
        for model_id in FINAL_MODELS:
            daily.append({
                "event_date": str(day),
                **point_row(model_id, final_labels[mask], final_users[mask], probability_final[model_id][mask]),
            })
    bootstrap, replicate_rows = paired_bootstrap(
        final_labels, final_users, probability_final, contract, final_dir
    )
    decision = final_decision(points, bootstrap, daily, audits, selection, contract)
    prediction_table = pa.table({
        "source_table": frame.columns["source_table"][final_index],
        "source_row_number": frame.columns["source_row_number"][final_index],
        "user_id": final_users,
        "event_date": final_dates,
        "long_view": final_labels,
        "p_TARGET_BL0": probability_final["TARGET_BL0"],
        "p_OLD_BL2_PLUS_V011": probability_final["OLD_BL2_PLUS_V011"],
        "raw_NEW_BL1": raw_final["NEW_BL1"],
        "p_NEW_BL1": probability_final["NEW_BL1"],
        "raw_NEW_BL2": raw_final["NEW_BL2"],
        "p_NEW_BL2": probability_final["NEW_BL2"],
    })
    pq.write_table(prediction_table, final_dir / "predictions.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(replicate_rows), final_dir / "bootstrap_replicates.parquet", compression="zstd")
    v1.write_csv(final_dir / "pooled_metrics.csv", points)
    v1.write_csv(final_dir / "paired_user_cluster_bootstrap.csv", bootstrap)
    v1.write_csv(final_dir / "daily_metrics.csv", daily)
    v1.write_csv(final_dir / "reliability_equal_width_20.csv", reliability)
    v1.write_csv(final_dir / "probability_distribution_audit.csv", audits)
    v1.write_json(final_dir / "split_audit.json", next(row for row in split_audits if row["split"] == "final_temporal_replay_test"))
    v1.write_json(final_dir / "stage_decision.json", decision)
    v1.write_json(final_dir / "run_manifest.json", {
        "status": "complete", "contract_sha256": digest,
        "selected_config_id": selected_id, "selected_model_state_sha256": selected_state_sha,
        "prediction_sha256": sha256_file(final_dir / "predictions.parquet"),
        "bootstrap_replicates_sha256": sha256_file(final_dir / "bootstrap_replicates.parquet"),
        "rows": int(final_index.size), "users": int(np.unique(final_users).size),
        "positives": int(final_labels.sum()), "decision": decision, "environment": environment,
    })
    v1.finalize_hashes(final_dir)
    append_access("final_temporal_replay_test", "complete_hashed_no_retry", digest)

    v1.write_json(OUTPUT_ROOT / "final_decision.json", {
        "contract_sha256": digest,
        "selected_config_id": selected_id,
        "selected_model_state_sha256": selected_state_sha,
        "decision": decision,
        "engineering_recommendation": (
            "deploy_NEW_BL2_with_v012_calibrator_pending_new_data_confirmation"
            if decision["scientific_status"] == "retraining_adds_value"
            else "retain_frozen_BL2_plus_v011_intercept_calibration"
        ),
    })
    REPORT_PATH.write_text(
        render_report(
            digest, selection, selection_contrasts, selected_fits, selected_calibrators,
            points, bootstrap, decision, split_audits,
        ),
        encoding="utf-8",
    )
    v1.write_json(OUTPUT_ROOT / "experiment_manifest.json", {
        "contract_sha256": digest,
        "report_path": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        "report_sha256": sha256_file(REPORT_PATH),
        "selected_config_id": selected_id,
        "selected_model_state_sha256": selected_state_sha,
        "substage_manifests": [
            {
                "path": str((directory / "artifact_hash_manifest.json").relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(directory / "artifact_hash_manifest.json"),
            }
            for directory in (preflight_dir, train_dir, calibration_dir, selection_dir, frozen_dir, final_dir)
        ],
    })
    v1.finalize_hashes(OUTPUT_ROOT)
    print(
        f"v012 complete; status={decision['scientific_status']}; "
        f"selected={selected_id}; report={REPORT_PATH}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--release-v012", action="store_true")
    parser.add_argument("--approved-contract-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_only:
        validate_only()
        return 0
    if not args.approved_contract_sha256:
        raise ContractStop("--approved-contract-sha256 is required")
    run_experiment(args.approved_contract_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
