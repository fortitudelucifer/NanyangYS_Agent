#!/usr/bin/env python3
"""Post-audit target-domain calibration of the frozen v010 Adam BL2 score."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn
import torch
import yaml
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_history_value_gpu_confirmation_v001 as v1  # noqa: E402


CONTRACT_PATH = EXPERIMENT_ROOT / "contract_v011.yaml"
APPROVAL_PATH = EXPERIMENT_ROOT / "approval_v011.json"
SNAPSHOT_PATH = EXPERIMENT_ROOT / "predecessor_integrity_snapshot.json"
INPUT_PATH = PROJECT_ROOT / "reports/generated/history_value_adam_random_v010/random_audit/predictions.parquet"
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
REPORT_PATH = EXPERIMENT_ROOT / "results_v011.md"
EXPECTED_INPUT_SHA256 = "48ae0793e554b1eb82d3ba75684915f471cf52f005039a6ad674075b4afffc31"
EXPECTED_SNAPSHOT_SHA256 = "1a2fd7f32226916bd0160aae7dc5622255e8ae4fd3df9eae46c81863feb4c94f"
AUTHORIZED_STAGES = [
    "preflight", "calibration_fit", "calibration_selection",
    "final_refit", "held_out_to_calibrator_test",
]
REQUIRED_COLUMNS = (
    "source_table", "source_row_number", "user_id", "event_date", "long_view",
    "raw_ADAM_BL2", "p_ADAM_BL2",
)
CANDIDATES = ("M1_prior_shift", "M2_intercept_only", "M3_platt")
SIMPLICITY = {name: index for index, name in enumerate(CANDIDATES)}


class ContractStop(RuntimeError):
    pass


@dataclass(frozen=True)
class CalibratorSpec:
    family: str
    intercept: float
    slope: float
    input_space: str
    fit_rows: int
    fit_positives: int
    fit_prevalence: float
    converged: bool
    n_iter: int
    convergence_warning_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "intercept": self.intercept,
            "slope": self.slope,
            "input_space": self.input_space,
            "fit_rows": self.fit_rows,
            "fit_positives": self.fit_positives,
            "fit_prevalence": self.fit_prevalence,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "convergence_warning_count": self.convergence_warning_count,
        }


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> tuple[dict[str, Any], str]:
    contract = yaml.load(CONTRACT_PATH.read_text(encoding="utf-8"), Loader=v1.UniqueKeyLoader)
    if contract["contract_id"] != "bl2_target_domain_calibration_v011":
        raise ContractStop("wrong v011 contract id")
    if tuple(contract["calibration_candidates"]["ordered_families"]) != CANDIDATES:
        raise ContractStop("v011 calibration candidate list differs")
    if contract["authorization"]["authorized_stages_after_exact_hash_approval"] != AUTHORIZED_STAGES:
        raise ContractStop("v011 authorized stages differ")
    return contract, sha256_file(CONTRACT_PATH)


def _verify_file(path: Path, *, expected_size: int, expected_sha: str, label: str) -> None:
    if not path.is_file() or path.stat().st_size != int(expected_size):
        raise ContractStop(f"{label} missing or size mismatch")
    if sha256_file(path) != expected_sha:
        raise ContractStop(f"{label} SHA mismatch")


def _verify_artifact_manifest(path: Path) -> int:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        item = Path(artifact["path"])
        if not item.is_absolute():
            item = PROJECT_ROOT / item
        _verify_file(
            item, expected_size=int(artifact["size_bytes"]),
            expected_sha=artifact["sha256"], label=f"manifest artifact {item}",
        )
    return len(manifest["artifacts"])


def verify_predecessor_integrity(contract: dict[str, Any]) -> dict[str, Any]:
    predecessor = contract["predecessor_integrity"]
    _verify_file(
        SNAPSHOT_PATH, expected_size=int(predecessor["snapshot_size_bytes"]),
        expected_sha=predecessor["snapshot_sha256"], label="v011 predecessor snapshot",
    )
    if predecessor["snapshot_sha256"] != EXPECTED_SNAPSHOT_SHA256:
        raise ContractStop("v011 compiled snapshot SHA differs")
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    if snapshot["status"] != "all_declared_predecessor_artifacts_verified":
        raise ContractStop("v011 predecessor snapshot status differs")
    expected_counts = predecessor["required_nested_artifact_counts"]
    for entry in snapshot["manifests"]:
        path = PROJECT_ROOT / entry["path"]
        _verify_file(
            path, expected_size=int(entry["size_bytes"]), expected_sha=entry["sha256"],
            label=f"v011 predecessor manifest {path.name}",
        )
        if path.name == "artifact_hash_manifest.json":
            count = _verify_artifact_manifest(path)
            relative = entry["path"]
            if relative in expected_counts and count != int(expected_counts[relative]):
                raise ContractStop(f"v011 predecessor artifact count differs: {relative}")
        elif path.name == "history_value_final_evidence_manifest_v010.json":
            count = _verify_artifact_manifest(path)
            if count != int(predecessor["final_evidence_artifact_count"]):
                raise ContractStop("v011 final evidence artifact count differs")
    calibration_input = snapshot["calibration_input"]
    _verify_file(
        INPUT_PATH, expected_size=int(calibration_input["size_bytes"]),
        expected_sha=calibration_input["sha256"], label="v011 frozen predictions input",
    )
    if calibration_input["sha256"] != EXPECTED_INPUT_SHA256:
        raise ContractStop("v011 compiled input SHA differs")
    decision_entry = snapshot["v010_final_decision"]
    decision_path = PROJECT_ROOT / decision_entry["path"]
    _verify_file(
        decision_path, expected_size=int(decision_entry["size_bytes"]),
        expected_sha=decision_entry["sha256"], label="v011 v010 final decision",
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision["terminal_interpretation"]["status"] != decision_entry["required_status"]:
        raise ContractStop("v011 requires the frozen v010 history conclusion")
    return snapshot


def verify_input_metadata(contract: dict[str, Any]) -> dict[str, Any]:
    metadata = pq.read_metadata(INPUT_PATH)
    schema_names = tuple(metadata.schema.to_arrow_schema().names)
    for column in REQUIRED_COLUMNS:
        if column not in schema_names:
            raise ContractStop(f"v011 input is missing {column}")
    declared = contract["frozen_input"]
    if metadata.num_rows != int(declared["rows"]):
        raise ContractStop("v011 input row count differs")
    return {
        "path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "size_bytes": INPUT_PATH.stat().st_size,
        "sha256": sha256_file(INPUT_PATH),
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "required_columns_present": list(REQUIRED_COLUMNS),
    }


def verify_implementation(contract: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for entry in contract["implementation"]["result_producing_files"]:
        path = PROJECT_ROOT / entry["path"]
        observed = sha256_file(path)
        if observed != entry["sha256"]:
            raise ContractStop(f"v011 implementation SHA mismatch: {entry['path']}")
        records.append({"path": entry["path"], "sha256": observed})
    return records


def verify_environment(contract: dict[str, Any]) -> dict[str, Any]:
    observed = {
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "pyarrow": pa.__version__, "torch": torch.__version__,
    }
    if observed != contract["environment"]["required_versions"]:
        raise ContractStop(f"v011 environment mismatch: {observed}")
    return {"versions": observed, "executable": sys.executable, "device": "CPU"}


def verify_approval(contract_sha: str) -> dict[str, Any]:
    if not APPROVAL_PATH.is_file():
        raise ContractStop("v011 exact-hash approval receipt is missing")
    receipt = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    required = {
        "contract_id": "bl2_target_domain_calibration_v011",
        "contract_sha256": contract_sha,
        "execution_authorized": True,
        "authorized_stages": AUTHORIZED_STAGES,
        "approved_by": "project_owner",
        "BL1_or_BL2_retraining_authorized": False,
        "held_out_test_refit_or_retry_authorized": False,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ContractStop(f"v011 approval mismatch: {key}")
    return receipt


def append_access(stage: str, status: str, contract_sha: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "stage_access_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "timestamp": datetime.now().astimezone().isoformat(), "stage": stage,
            "status": status, "contract_sha256": contract_sha,
        }, sort_keys=True) + "\n")


def _read_split(start: str, end: str) -> dict[str, np.ndarray]:
    table = pq.read_table(
        INPUT_PATH, columns=list(REQUIRED_COLUMNS),
        filters=[("event_date", ">=", date.fromisoformat(start)), ("event_date", "<=", date.fromisoformat(end))],
    )
    return {
        name: table[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in REQUIRED_COLUMNS
    }


def _split_audit(data: dict[str, np.ndarray], expected: dict[str, Any], label: str) -> dict[str, Any]:
    rows = int(data["long_view"].size)
    positives = int(np.asarray(data["long_view"], dtype=np.int8).sum())
    users = int(np.unique(data["user_id"]).size)
    identities = set(zip(data["source_table"].tolist(), data["source_row_number"].tolist()))
    if rows != int(expected["rows"]) or positives != int(expected["positives"]):
        raise ContractStop(f"v011 {label} rows or positives differ")
    if len(identities) != rows:
        raise ContractStop(f"v011 {label} identities are not unique")
    raw = np.asarray(data["raw_ADAM_BL2"], dtype=np.float64)
    probability = np.asarray(data["p_ADAM_BL2"], dtype=np.float64)
    if not np.isfinite(raw).all() or not np.isfinite(probability).all():
        raise ContractStop(f"v011 {label} has non-finite input score")
    return {
        "split": label, "rows": rows, "users": users, "positives": positives,
        "prevalence": positives / rows, "unique_identities": len(identities),
        "date_min": str(np.min(data["event_date"])), "date_max": str(np.max(data["event_date"])),
    }


def _combine(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([left[key], right[key]]) for key in REQUIRED_COLUMNS}


def _fit_intercept(raw: np.ndarray, labels: np.ndarray, slope: float, contract: dict[str, Any]) -> tuple[float, int]:
    settings = contract["calibration_candidates"]["M2_intercept_only"]
    target = float(np.mean(labels, dtype=np.float64))
    evaluations = 0

    def score(intercept: float) -> float:
        nonlocal evaluations
        evaluations += 1
        return float(np.mean(expit(slope * raw + intercept), dtype=np.float64) - target)

    intercept = brentq(
        score, float(settings["bracket"][0]), float(settings["bracket"][1]),
        xtol=float(settings["xtol"]), rtol=float(settings["rtol"]),
        maxiter=int(settings["max_iter"]), disp=True,
    )
    return float(intercept), evaluations


def fit_candidate(family: str, data: dict[str, np.ndarray], contract: dict[str, Any]) -> CalibratorSpec:
    raw = np.asarray(data["raw_ADAM_BL2"], dtype=np.float64)
    labels = np.asarray(data["long_view"], dtype=np.int8)
    rows, positives = labels.size, int(labels.sum())
    prevalence = positives / rows
    if family == "M1_prior_shift":
        source = float(contract["calibration_candidates"][family]["source_prevalence"])
        shift = float(logit(prevalence) - logit(source))
        return CalibratorSpec(family, shift, 1.0, "logit_frozen_probability", rows, positives, prevalence, True, 0, 0)
    if family == "M2_intercept_only":
        slope = float(contract["calibration_candidates"][family]["frozen_slope"])
        intercept, evaluations = _fit_intercept(raw, labels, slope, contract)
        return CalibratorSpec(family, intercept, slope, "raw_ADAM_BL2", rows, positives, prevalence, True, evaluations, 0)
    if family == "M3_platt":
        settings = contract["calibration_candidates"][family]
        model = LogisticRegression(
            solver="lbfgs", l1_ratio=0.0, C=float(settings["C"]), dual=False,
            tol=float(settings["tol"]), fit_intercept=True, intercept_scaling=1,
            class_weight=None, random_state=None, max_iter=int(settings["max_iter"]),
            verbose=0, warm_start=False, n_jobs=None,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(raw.reshape(-1, 1), labels)
        convergence_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        other_warnings = [item for item in caught if not issubclass(item.category, ConvergenceWarning)]
        if other_warnings:
            raise ContractStop("v011 Platt emitted a non-convergence warning")
        n_iter = int(np.asarray(model.n_iter_).ravel()[0])
        slope = float(np.asarray(model.coef_).ravel()[0])
        intercept = float(np.asarray(model.intercept_).ravel()[0])
        converged = convergence_warnings == 0 and n_iter < int(settings["max_iter"]) and slope > 0
        return CalibratorSpec(family, intercept, slope, "raw_ADAM_BL2", rows, positives, prevalence, converged, n_iter, convergence_warnings)
    raise ContractStop(f"unknown v011 calibration family: {family}")


def apply_candidate(spec: CalibratorSpec, data: dict[str, np.ndarray]) -> np.ndarray:
    raw = np.asarray(data["raw_ADAM_BL2"], dtype=np.float64)
    if spec.input_space == "logit_frozen_probability":
        old = np.asarray(data["p_ADAM_BL2"], dtype=np.float64)
        score = logit(np.clip(old, 1e-15, 1 - 1e-15))
    else:
        score = raw
    probability = expit(spec.slope * score + spec.intercept).astype(np.float64, copy=False)
    if not np.isfinite(probability).all() or np.min(probability) < 0 or np.max(probability) > 1:
        raise ContractStop(f"v011 invalid probability from {spec.family}")
    return probability


def stable_order_exact(raw: np.ndarray, probability: np.ndarray) -> bool:
    return bool(np.array_equal(
        np.argsort(np.asarray(raw, dtype=np.float64), kind="stable"),
        np.argsort(np.asarray(probability, dtype=np.float64), kind="stable"),
    ))


def point_row(model_id: str, data: dict[str, np.ndarray], probability: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(data["long_view"], dtype=np.int8)
    users = np.asarray(data["user_id"])
    point = v1.metrics.point_metrics(labels, probability, users, epsilon=v1.repair.METRIC_CLIP_LOW)
    return {
        "model_id": model_id, "rows": labels.size, "users": np.unique(users).size,
        "positives": int(labels.sum()), "prevalence": float(labels.mean()),
        "mean_probability": float(np.mean(probability)),
        "mean_probability_minus_prevalence": float(np.mean(probability) - np.mean(labels)),
        **point,
    }


def select_family(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ContractStop("v011 has no eligible calibration family")
    minimum = min(float(row["log_loss"]) for row in eligible)
    tolerance = float(contract["selection_rule"]["simplicity_tie_tolerance_log_loss"])
    tied = [row for row in eligible if float(row["log_loss"]) <= minimum + tolerance]
    selected = min(tied, key=lambda row: SIMPLICITY[row["family"]])
    return {
        "status": "family_selected_before_held_out_test",
        "selected_family": selected["family"], "minimum_selection_log_loss": minimum,
        "tie_tolerance": tolerance, "eligible_families": [row["family"] for row in eligible],
        "tied_families": [row["family"] for row in tied],
    }


def reliability_rows(model_id: str, labels: np.ndarray, probability: np.ndarray, bins: int = 20) -> list[dict[str, Any]]:
    index = np.minimum((probability * bins).astype(np.int64), bins - 1)
    rows: list[dict[str, Any]] = []
    for bin_id in range(bins):
        mask = index == bin_id
        count = int(mask.sum())
        rows.append({
            "model_id": model_id, "bin_id": bin_id,
            "lower": bin_id / bins, "upper": (bin_id + 1) / bins,
            "rows": count, "positives": int(labels[mask].sum()) if count else 0,
            "mean_probability": float(probability[mask].mean()) if count else None,
            "observed_rate": float(labels[mask].mean()) if count else None,
            "calibration_gap": float(probability[mask].mean() - labels[mask].mean()) if count else None,
        })
    return rows


def probability_audit(model_id: str, probability: np.ndarray) -> dict[str, Any]:
    q = np.quantile(probability, [0, .001, .01, .1, .5, .9, .99, .999, 1])
    return {
        "model_id": model_id, "rows": probability.size,
        "finite_share": float(np.mean(np.isfinite(probability))),
        "below_or_equal_1e_6_share": float(np.mean(probability <= 1e-6)),
        "above_or_equal_1_minus_1e_6_share": float(np.mean(probability >= 1 - 1e-6)),
        **dict(zip(("minimum", "p001", "p01", "p10", "median", "p90", "p99", "p999", "maximum"), map(float, q))),
    }


def paired_bootstrap(
    labels: np.ndarray, users: np.ndarray, original: np.ndarray, selected: np.ndarray,
    contract: dict[str, Any], out: Path,
) -> list[dict[str, Any]]:
    user_universe = np.unique(users)
    replicates = int(contract["statistics"]["bootstrap_replicates"])
    seed = int(contract["statistics"]["bootstrap_seed"])
    multiplicities, multiplicity_sha = v1.metrics.make_multiplicities(
        user_count=user_universe.size, replicates=replicates, seed=seed,
    )
    np.save(out / "bootstrap_multiplicity.npy", multiplicities, allow_pickle=False)
    if sha256_file(out / "bootstrap_multiplicity.npy") == "":
        raise AssertionError("unreachable")
    row_user = v1.metrics._user_index(users, user_universe)
    user_rows = np.bincount(row_user, minlength=user_universe.size).astype(np.float64)
    denominator = multiplicities @ user_rows
    points = {
        "original": v1.metrics.point_metrics(labels, original, users, epsilon=v1.repair.METRIC_CLIP_LOW),
        "selected": v1.metrics.point_metrics(labels, selected, users, epsilon=v1.repair.METRIC_CLIP_LOW),
    }
    values: dict[str, dict[str, np.ndarray]] = {
        "average_precision": {}, "log_loss": {}, "brier": {}, "user_gauc_event_weighted": {},
    }
    for name, probability in (("original", original), ("selected", selected)):
        clipped = v1.metrics.clipped(probability, v1.repair.METRIC_CLIP_LOW)
        values["average_precision"][name] = v1.metrics.weighted_ap_replicates(
            labels, clipped, row_user, multiplicities, block_size=8,
            epsilon=v1.repair.METRIC_CLIP_LOW,
        )
        log_row = -(labels * np.log(clipped) + (1-labels) * np.log1p(-clipped))
        brier_row = np.square(clipped-labels)
        values["log_loss"][name] = (
            multiplicities @ v1.metrics._per_user_sums(log_row, row_user, user_universe.size)
        ) / denominator
        values["brier"][name] = (
            multiplicities @ v1.metrics._per_user_sums(brier_row, row_user, user_universe.size)
        ) / denominator
        gauc = v1.metrics.user_gauc_components(
            labels, clipped, users, user_universe=user_universe,
            epsilon=v1.repair.METRIC_CLIP_LOW,
        )
        eligible_rows = gauc.event_counts.astype(float) * gauc.eligible.astype(float)
        auc = np.nan_to_num(gauc.auc, nan=0.0)
        values["user_gauc_event_weighted"][name] = (
            multiplicities @ (auc * eligible_rows)
        ) / (multiplicities @ eligible_rows)
    rows = []
    for metric, by_model in values.items():
        samples = by_model["selected"] - by_model["original"]
        valid = samples[np.isfinite(samples)]
        point = float(points["selected"][metric] - points["original"][metric])
        rows.append({
            "contrast": "selected_minus_original", "metric": metric,
            "point_estimate": point, "bootstrap_replicates_requested": replicates,
            "effective_replicates": valid.size, "bootstrap_mean": float(valid.mean()),
            "bootstrap_se": float(valid.std(ddof=1)),
            "ci95_lower": float(np.quantile(valid, .025)),
            "ci95_upper": float(np.quantile(valid, .975)),
            "multiplicity_content_sha256": multiplicity_sha,
        })
    return rows


def contrast_lookup(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    for row in rows:
        if row["metric"] == metric:
            return row
    raise KeyError(metric)


def render_report(
    contract_sha: str, selection_rows: list[dict[str, Any]], selection: dict[str, Any],
    final_spec: CalibratorSpec, point_rows: list[dict[str, Any]], contrasts: list[dict[str, Any]],
    decision: dict[str, Any], split_audits: list[dict[str, Any]],
) -> str:
    point = {row["model_id"]: row for row in point_rows}
    lines = [
        "# BL2 目标域后置校准结果 v011", "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        f"- 合同 SHA-256：`{contract_sha}`",
        "- 证据等级：post-audit、held-out-to-calibrator temporal test；不是新的 pristine test。",
        "- BL2、H2、Adam、特征和历史全部冻结；只拟合后置校准器。", "",
        "## 时间切分", "",
        "| split | rows | users | positives | prevalence |", "|---|---:|---:|---:|---:|",
    ]
    for row in split_audits:
        lines.append(f"| {row['split']} | {row['rows']:,} | {row['users']:,} | {row['positives']:,} | {row['prevalence']:.6f} |")
    lines.extend(["", "## 方法选择", "", "| family | log-loss | Brier | ECE20 | mean p | eligible |", "|---|---:|---:|---:|---:|---|"])
    for row in selection_rows:
        lines.append(
            f"| {row['family']} | {row['log_loss']:.6f} | {row['brier']:.6f} | "
            f"{row['ece20_equal_width']:.6f} | {row['mean_probability']:.6f} | {row['eligible']} |"
        )
    lines.extend([
        "", f"选择家族：`{selection['selected_family']}`；合并前两段后的最终参数："
        f"slope=`{final_spec.slope:.12g}`，intercept=`{final_spec.intercept:.12g}`。", "",
        "## 六天留出结果", "",
        "| model | log-loss | Brier | ECE20 | AP | event-gAUC | mean p |", "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for model_id in ("original_BL2", "selected_calibrator"):
        row = point[model_id]
        lines.append(
            f"| {model_id} | {row['log_loss']:.6f} | {row['brier']:.6f} | "
            f"{row['ece20_equal_width']:.6f} | {row['average_precision']:.6f} | "
            f"{row['user_gauc_event_weighted']:.6f} | {row['mean_probability']:.6f} |"
        )
    lines.extend(["", "| contrast | point | 95% CI |", "|---|---:|---:|"])
    for metric in ("log_loss", "brier", "average_precision", "user_gauc_event_weighted"):
        row = contrast_lookup(contrasts, metric)
        lines.append(f"| Δ{metric} | {row['point_estimate']:.6f} | [{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}] |")
    lines.extend([
        "", "## 决策", "",
        f"- log-loss gate：{decision['log_loss_gate']}",
        f"- Brier gate：{decision['brier_gate']}",
        f"- ECE gate：{decision['ece_gate']}",
        f"- mean probability gate：{decision['mean_probability_gate']}",
        f"- probability health gate：{decision['probability_health_gate']}",
        f"- ranking invariance gate：{decision['ranking_invariance_gate']}",
        f"- 最终状态：`{decision['scientific_status']}`", "",
        "该结果只评价后置概率校准；无论通过或失败，都不改变 v010 的 BL2 历史排序结论。", "",
    ])
    return "\n".join(lines)


def validate_only() -> None:
    contract, digest = load_contract()
    verify_predecessor_integrity(contract)
    verify_input_metadata(contract)
    verify_implementation(contract)
    verify_environment(contract)
    if OUTPUT_ROOT.exists() or REPORT_PATH.exists():
        raise ContractStop("v011 output already exists before execution")
    print(f"V011_VALIDATE_ONLY_OK contract_sha256={digest}; no calibration or test labels opened")


def run_experiment(approved_hash: str) -> None:
    contract, digest = load_contract()
    if digest != approved_hash:
        raise ContractStop(f"approved v011 contract SHA mismatch: observed {digest}")
    verify_approval(digest)
    snapshot = verify_predecessor_integrity(contract)
    input_metadata = verify_input_metadata(contract)
    implementation = verify_implementation(contract)
    environment = verify_environment(contract)
    if OUTPUT_ROOT.exists() or REPORT_PATH.exists():
        raise ContractStop("v011 refuses to overwrite prior outputs")
    OUTPUT_ROOT.mkdir(parents=True)

    append_access("preflight", "opened_without_calibration_or_test_labels", digest)
    preflight = OUTPUT_ROOT / "preflight"
    preflight.mkdir()
    v1.write_json(preflight / "integrity_verification.json", {
        "status": "pass", "contract_sha256": digest, "snapshot": snapshot,
        "input_metadata": input_metadata, "implementation": implementation,
        "environment": environment,
    })
    v1.finalize_hashes(preflight)
    append_access("preflight", "complete_hashed", digest)

    splits = contract["temporal_splits"]
    append_access("calibration_fit", "opened_before_selection_or_test", digest)
    fit = _read_split(splits["calibration_fit"]["start"], splits["calibration_fit"]["end"])
    fit_audit = _split_audit(fit, splits["calibration_fit"], "calibration_fit")
    fit_dir = OUTPUT_ROOT / "calibration_fit"
    fit_dir.mkdir()
    fitted = {family: fit_candidate(family, fit, contract) for family in CANDIDATES}
    if not all(spec.converged and spec.slope > 0 for spec in fitted.values()):
        raise ContractStop("v011 calibration-fit has an ineligible family")
    expected_initial_shift = float(
        contract["calibration_candidates"]["M1_prior_shift"]["selection_fit_expected_shift"]
    )
    if abs(fitted["M1_prior_shift"].intercept - expected_initial_shift) > 1e-12:
        raise ContractStop("v011 initial prior-shift parameter differs")
    v1.write_json(fit_dir / "split_audit.json", fit_audit)
    v1.write_json(fit_dir / "candidate_parameters.json", {
        "contract_sha256": digest,
        "candidates": [fitted[family].as_dict() for family in CANDIDATES],
        "held_out_test_opened": False,
    })
    v1.finalize_hashes(fit_dir)
    append_access("calibration_fit", "complete_hashed", digest)

    append_access("calibration_selection", "opened_after_fit_hash_before_test", digest)
    selection_data = _read_split(splits["calibration_selection"]["start"], splits["calibration_selection"]["end"])
    selection_audit = _split_audit(selection_data, splits["calibration_selection"], "calibration_selection")
    selection_rows = []
    raw_selection = np.asarray(selection_data["raw_ADAM_BL2"], dtype=np.float64)
    for family in CANDIDATES:
        probability = apply_candidate(fitted[family], selection_data)
        row = point_row(family, selection_data, probability)
        selection_rows.append({
            "family": family, **{key: value for key, value in row.items() if key != "model_id"},
            "converged": fitted[family].converged,
            "positive_slope": fitted[family].slope > 0,
            "stable_order_exact": stable_order_exact(raw_selection, probability),
            "eligible": bool(fitted[family].converged and fitted[family].slope > 0 and stable_order_exact(raw_selection, probability)),
        })
    selection = select_family(selection_rows, contract)
    selection_dir = OUTPUT_ROOT / "calibration_selection"
    selection_dir.mkdir()
    v1.write_json(selection_dir / "split_audit.json", selection_audit)
    v1.write_csv(selection_dir / "candidate_metrics.csv", selection_rows)
    v1.write_json(selection_dir / "family_selection.json", {
        **selection, "contract_sha256": digest, "held_out_test_opened": False,
    })
    v1.finalize_hashes(selection_dir)
    append_access("calibration_selection", "family_selected_and_hashed_before_test", digest)

    append_access("final_refit", "opened_before_test", digest)
    refit_data = _combine(fit, selection_data)
    final_spec = fit_candidate(selection["selected_family"], refit_data, contract)
    expected_refit = contract["final_refit"]
    if final_spec.fit_rows != int(expected_refit["rows"]) or final_spec.fit_positives != int(expected_refit["positives"]):
        raise ContractStop("v011 final refit population differs")
    if not final_spec.converged or final_spec.slope <= 0:
        raise ContractStop("v011 final refit is ineligible")
    if final_spec.family == "M1_prior_shift":
        expected_final_shift = float(
            contract["calibration_candidates"]["M1_prior_shift"]["final_refit_expected_shift"]
        )
        if abs(final_spec.intercept - expected_final_shift) > 1e-12:
            raise ContractStop("v011 final prior-shift parameter differs")
    refit_dir = OUTPUT_ROOT / "final_refit"
    refit_dir.mkdir()
    v1.write_json(refit_dir / "selected_calibrator.json", {
        "contract_sha256": digest, "selection": selection,
        "final_calibrator": final_spec.as_dict(), "held_out_test_opened": False,
    })
    v1.finalize_hashes(refit_dir)
    selected_calibrator_sha = sha256_file(refit_dir / "selected_calibrator.json")
    append_access("final_refit", "selected_parameters_frozen_and_hashed_before_test", digest)

    append_access("held_out_to_calibrator_test", "opened_after_final_refit_hash", digest)
    test_data = _read_split(splits["held_out_test"]["start"], splits["held_out_test"]["end"])
    test_audit = _split_audit(test_data, splits["held_out_test"], "held_out_test")
    labels = np.asarray(test_data["long_view"], dtype=np.int8)
    users = np.asarray(test_data["user_id"])
    raw = np.asarray(test_data["raw_ADAM_BL2"], dtype=np.float64)
    original = np.asarray(test_data["p_ADAM_BL2"], dtype=np.float64)
    selected_probability = apply_candidate(final_spec, test_data)
    test_dir = OUTPUT_ROOT / "held_out_to_calibrator_test"
    test_dir.mkdir()
    points = [
        point_row("original_BL2", test_data, original),
        point_row("selected_calibrator", test_data, selected_probability),
    ]
    contrasts = paired_bootstrap(labels, users, original, selected_probability, contract, test_dir)
    reliability = (
        reliability_rows("original_BL2", labels, original)
        + reliability_rows("selected_calibrator", labels, selected_probability)
    )
    distributions = [
        probability_audit("original_BL2", original),
        probability_audit("selected_calibrator", selected_probability),
    ]
    daily_rows = []
    for day in np.unique(test_data["event_date"]):
        mask = test_data["event_date"] == day
        subset = {key: value[mask] for key, value in test_data.items()}
        for model_id, probability in (
            ("original_BL2", original[mask]), ("selected_calibrator", selected_probability[mask]),
        ):
            daily_rows.append({"event_date": str(day), **point_row(model_id, subset, probability)})

    point = {row["model_id"]: row for row in points}
    selected_point = point["selected_calibrator"]
    selected_distribution = distributions[1]
    log_contrast = contrast_lookup(contrasts, "log_loss")
    brier_contrast = contrast_lookup(contrasts, "brier")
    ranking_metrics = ("average_precision", "roc_auc", "user_gauc_event_weighted")
    ranking_differences = {
        metric: float(selected_point[metric] - point["original_BL2"][metric])
        for metric in ranking_metrics
    }
    rank_tolerance = float(contract["gates"]["ranking_metric_absolute_tolerance"])
    log_gate = bool(log_contrast["point_estimate"] < 0 and log_contrast["ci95_upper"] < 0)
    brier_gate = bool(brier_contrast["point_estimate"] < 0 and brier_contrast["ci95_upper"] < 0)
    ece_gate = bool(selected_point["ece20_equal_width"] <= float(contract["gates"]["maximum_ECE20"]))
    mean_gate = bool(abs(selected_point["mean_probability_minus_prevalence"]) <= float(contract["gates"]["maximum_absolute_mean_probability_gap"]))
    health_gate = bool(
        selected_distribution["finite_share"] == 1.0
        and selected_distribution["below_or_equal_1e_6_share"] <= float(contract["gates"]["maximum_extreme_probability_share"])
        and selected_distribution["above_or_equal_1_minus_1e_6_share"] <= float(contract["gates"]["maximum_extreme_probability_share"])
    )
    ranking_gate = bool(
        stable_order_exact(raw, selected_probability)
        and all(abs(value) <= rank_tolerance for value in ranking_differences.values())
    )
    decision = {
        "stage": "held_out_to_calibrator_test",
        "scientific_status": "pass" if all((log_gate, brier_gate, ece_gate, mean_gate, health_gate, ranking_gate)) else "fail_or_mixed",
        "log_loss_gate": log_gate, "brier_gate": brier_gate, "ece_gate": ece_gate,
        "mean_probability_gate": mean_gate, "probability_health_gate": health_gate,
        "ranking_invariance_gate": ranking_gate, "ranking_metric_differences": ranking_differences,
        "stable_order_exact": stable_order_exact(raw, selected_probability),
        "evidence_level": "post_audit_held_out_to_calibrator_temporal_test",
        "changes_v010_history_ranking_conclusion": False,
    }
    prediction_table = pa.table({
        "source_table": test_data["source_table"],
        "source_row_number": test_data["source_row_number"],
        "user_id": test_data["user_id"], "event_date": test_data["event_date"],
        "long_view": labels, "raw_ADAM_BL2": raw,
        "p_original_BL2": original, "p_selected_calibrator": selected_probability,
    })
    pq.write_table(prediction_table, test_dir / "predictions.parquet", compression="zstd")
    v1.write_json(test_dir / "split_audit.json", test_audit)
    v1.write_csv(test_dir / "pooled_metrics.csv", points)
    v1.write_csv(test_dir / "paired_user_cluster_bootstrap.csv", contrasts)
    v1.write_csv(test_dir / "reliability_equal_width_20.csv", reliability)
    v1.write_csv(test_dir / "probability_distribution_audit.csv", distributions)
    v1.write_csv(test_dir / "daily_metrics.csv", daily_rows)
    v1.write_json(test_dir / "stage_decision.json", decision)
    v1.write_json(test_dir / "run_manifest.json", {
        "status": "complete", "contract_sha256": digest,
        "selected_calibrator_sha256": selected_calibrator_sha,
        "selected_family": final_spec.family, "rows": labels.size,
        "users": np.unique(users).size, "positives": int(labels.sum()),
        "prediction_sha256": sha256_file(test_dir / "predictions.parquet"),
        "decision": decision, "environment": environment,
    })
    v1.finalize_hashes(test_dir)
    append_access("held_out_to_calibrator_test", "complete_hashed", digest)

    v1.write_json(OUTPUT_ROOT / "final_decision.json", {
        "contract_sha256": digest, "selected_family": final_spec.family,
        "selected_calibrator_sha256": selected_calibrator_sha,
        "decision": decision,
        "deployment_status": (
            "engineering_calibrator_supported_pending_new_data_confirmation"
            if decision["scientific_status"] == "pass" else
            "engineering_calibrator_not_supported"
        ),
    })
    REPORT_PATH.write_text(
        render_report(digest, selection_rows, selection, final_spec, points, contrasts, decision,
                      [fit_audit, selection_audit, test_audit]),
        encoding="utf-8",
    )
    v1.write_json(OUTPUT_ROOT / "experiment_manifest.json", {
        "contract_sha256": digest,
        "report_path": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        "report_sha256": sha256_file(REPORT_PATH),
        "selected_family": final_spec.family,
        "selected_calibrator_sha256": selected_calibrator_sha,
        "substage_manifests": [
            {"path": str((directory / "artifact_hash_manifest.json").relative_to(PROJECT_ROOT)),
             "sha256": sha256_file(directory / "artifact_hash_manifest.json")}
            for directory in (preflight, fit_dir, selection_dir, refit_dir, test_dir)
        ],
    })
    v1.finalize_hashes(OUTPUT_ROOT)
    print(f"v011 calibration complete; status={decision['scientific_status']}; report={REPORT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--release-calibration", action="store_true")
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
