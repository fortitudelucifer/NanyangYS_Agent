#!/usr/bin/env python3
"""Run the final random audit from the hash-pinned sealed diagnostic model."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import run_history_value_gpu_confirmation_v001 as v1
import run_history_value_gpu_confirmation_v002 as v2
import run_history_value_adam_confirmation_v004 as v4
import run_history_value_adam_validation_v007 as v7
import run_history_value_adam_random_v009 as v9


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_adam_random_contract_v010.yaml"
BASE_CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_adam_random_contract_v009.yaml"
APPROVAL_PATH = PROJECT_ROOT / "configs/history_value_adam_random_approval_v010.json"
OUTPUT_ROOT = PROJECT_ROOT / "reports/generated/history_value_adam_random_v010"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/history_value_adam_random_results_v010.md"
V009_FAILURE = PROJECT_ROOT / "reports/generated/history_value_adam_random_v009/reconstruction_failure.json"
DIAGNOSTIC_ROOT = PROJECT_ROOT / "reports/generated/sealed_model_reconstruction_diagnostic_v010"
DIAGNOSTIC_AUDIT = DIAGNOSTIC_ROOT / "diagnostic_audit.json"
DIAGNOSTIC_MANIFEST = DIAGNOSTIC_ROOT / "artifact_hash_manifest.json"
DIAGNOSTIC_DIFFERENCES = DIAGNOSTIC_ROOT / "prediction_difference_distribution.csv"
DIAGNOSTIC_METRICS = DIAGNOSTIC_ROOT / "sealed_metric_reproduction.csv"
FROZEN_PICKLE = DIAGNOSTIC_ROOT / "reconstructed_frozen_models.pkl"
FROZEN_STATE = DIAGNOSTIC_ROOT / "reconstructed_frozen_model_state.npz"
ADEQUACY_PATH = DIAGNOSTIC_ROOT / "reconstructed_optimization_adequacy.csv"
CALIBRATION_PATH = DIAGNOSTIC_ROOT / "reconstructed_calibration_audit.csv"
EXPECTED_BASE_SHA256 = "b5af66d0087b16a5b7e248397cc745b4dc31cf7d0207b2842c589e8359e52c58"
EXPECTED_RANDOM_ROWS = 43027
AUTHORIZED_STAGES = ["preflight", "frozen_model_verification", "random_audit"]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_contract() -> tuple[dict[str, Any], str]:
    observed = v1.sha256_file(BASE_CONTRACT_PATH)
    if observed != EXPECTED_BASE_SHA256:
        raise v1.ContractStop("v010 base v009 contract hash mismatch")
    base, base_digest = v9.load_contract()
    if base_digest != observed:
        raise v1.ContractStop("v010 base merged digest mismatch")
    overlay = yaml.load(CONTRACT_PATH.read_text(encoding="utf-8"), Loader=v1.UniqueKeyLoader)
    if overlay["base_contract"]["sha256"] != observed:
        raise v1.ContractStop("v010 overlay does not pin v009")
    merged = v2.deep_merge(base, overlay)
    if int(merged["sequential_stage_protocol"]["stage_3_random_audit"]["expected_target_rows"]) != EXPECTED_RANDOM_ROWS:
        raise v1.ContractStop("v010 random target count differs")
    if merged["authorization"]["authorized_stage_scope_after_exact_hash_approval"] != AUTHORIZED_STAGES:
        raise v1.ContractStop("v010 authorized stage scope differs")
    return merged, v1.sha256_file(CONTRACT_PATH)


def verify_v009_stop(contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract["v009_stop_evidence"]
    if V009_FAILURE.stat().st_size != int(expected["size_bytes"]):
        raise v1.ContractStop("v010 v009 failure size mismatch")
    if v1.sha256_file(V009_FAILURE) != expected["sha256"]:
        raise v1.ContractStop("v010 v009 failure SHA mismatch")
    failure = json.loads(V009_FAILURE.read_text(encoding="utf-8"))
    if failure["contract_sha256"] != EXPECTED_BASE_SHA256:
        raise v1.ContractStop("v010 v009 failure contract mismatch")
    if failure["status"] != "stopped_before_any_random_data_access":
        raise v1.ContractStop("v010 v009 did not stop at the required boundary")
    if any(bool(value) for value in failure["access_and_output_boundary"].values()):
        raise v1.ContractStop("v010 v009 failure indicates random access")
    return failure


def _verify_diagnostic_manifest(contract: dict[str, Any]) -> int:
    expected = contract["sealed_reconstruction_diagnostic"]
    if v1.sha256_file(DIAGNOSTIC_MANIFEST) != expected["artifact_manifest_sha256"]:
        raise v1.ContractStop("v010 diagnostic manifest SHA mismatch")
    manifest = json.loads(DIAGNOSTIC_MANIFEST.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = Path(artifact["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file() or path.stat().st_size != int(artifact["size_bytes"]):
            raise v1.ContractStop(f"v010 missing diagnostic artifact: {path}")
        if v1.sha256_file(path) != artifact["sha256"]:
            raise v1.ContractStop(f"v010 diagnostic artifact SHA mismatch: {path.name}")
    return len(manifest["artifacts"])


def verify_diagnostic(contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract["sealed_reconstruction_diagnostic"]
    if _verify_diagnostic_manifest(contract) != int(expected["artifact_count"]):
        raise v1.ContractStop("v010 diagnostic artifact count mismatch")
    if v1.sha256_file(DIAGNOSTIC_AUDIT) != expected["audit_sha256"]:
        raise v1.ContractStop("v010 diagnostic audit SHA mismatch")
    audit = json.loads(DIAGNOSTIC_AUDIT.read_text(encoding="utf-8"))
    if audit["status"] != "complete_no_random_data_access" or audit["random_input_opened"]:
        raise v1.ContractStop("v010 diagnostic random-access boundary failed")
    if audit["source_v009_contract_sha256"] != EXPECTED_BASE_SHA256:
        raise v1.ContractStop("v010 diagnostic source contract mismatch")
    if not audit["sealed_target_identity_exact"]:
        raise v1.ContractStop("v010 diagnostic sealed identities differ")

    limits = expected["acceptance_limits_derived_before_random_access"]
    by_quantity = {row["quantity"]: row for row in audit["prediction_differences"]}
    if by_quantity["BL0_probability"]["maximum_absolute_difference"] != 0.0:
        raise v1.ContractStop("v010 diagnostic BL0 is not exact")
    for quantity in ("ADAM_BL1_probability", "ADAM_BL2_probability"):
        row = by_quantity[quantity]
        if float(row["maximum_absolute_difference"]) > float(limits["calibrated_probability_max_abs_difference"]):
            raise v1.ContractStop(f"v010 diagnostic probability drift failed: {quantity}")
        if float(row["pearson_correlation"]) < float(limits["minimum_pearson_correlation"]):
            raise v1.ContractStop(f"v010 diagnostic correlation failed: {quantity}")
    for quantity in ("ADAM_BL1_raw", "ADAM_BL2_raw"):
        if float(by_quantity[quantity]["maximum_absolute_difference"]) > float(limits["raw_score_max_abs_difference"]):
            raise v1.ContractStop(f"v010 diagnostic raw drift failed: {quantity}")
    if max(abs(float(row["difference"])) for row in audit["metric_reproduction"]) > float(limits["metric_max_abs_difference"]):
        raise v1.ContractStop("v010 diagnostic metric reproduction failed")

    adequacy = csv_rows(ADEQUACY_PATH)
    gpu_rows = [row for row in adequacy if row.get("stream")]
    if len(gpu_rows) != 2 or not all(row["adequacy_passed"] == "True" for row in gpu_rows):
        raise v1.ContractStop("v010 diagnostic Adam adequacy failed")
    calibration = csv_rows(CALIBRATION_PATH)
    if len(calibration) != 2 or {row["stream"] for row in calibration} != {"ADAM_BL1", "ADAM_BL2"}:
        raise v1.ContractStop("v010 diagnostic calibration rows differ")
    return audit


def load_and_verify_frozen_model(contract: dict[str, Any]) -> v1.FrozenModels:
    expected = contract["frozen_model_artifact"]
    for path, size_key, hash_key in (
        (FROZEN_PICKLE, "pickle_size_bytes", "pickle_sha256"),
        (FROZEN_STATE, "state_size_bytes", "state_sha256"),
    ):
        if path.stat().st_size != int(expected[size_key]) or v1.sha256_file(path) != expected[hash_key]:
            raise v1.ContractStop(f"v010 frozen model artifact mismatch: {path.name}")
    with FROZEN_PICKLE.open("rb") as handle:
        frozen = pickle.load(handle)
    if not isinstance(frozen, v1.FrozenModels):
        raise v1.ContractStop("v010 frozen model type differs")
    if set(frozen.fits) != {"ADAM_BL1", "ADAM_BL2"} or set(frozen.calibrators) != {"ADAM_BL1", "ADAM_BL2"}:
        raise v1.ContractStop("v010 frozen model streams differ")
    with np.load(FROZEN_STATE, allow_pickle=False) as state:
        if not np.array_equal(state["bl0_probability"], np.asarray([frozen.bl0], dtype=np.float64)):
            raise v1.ContractStop("v010 frozen BL0 state differs")
        if not np.array_equal(state["static_scaler_mean"], np.asarray(frozen.design.static_scaler.mean_, dtype=np.float64)):
            raise v1.ContractStop("v010 frozen static scaler differs")
        if not np.array_equal(state["h2_scaler_mean"], np.asarray(frozen.design.h2_scaler.mean_, dtype=np.float64)):
            raise v1.ContractStop("v010 frozen history scaler differs")
        for stream, fit in frozen.fits.items():
            if not np.array_equal(state[f"{stream}_coefficient"], np.asarray(fit.coefficient)):
                raise v1.ContractStop(f"v010 frozen coefficient differs: {stream}")
            calibrator = frozen.calibrators[stream]
            if not np.array_equal(state[f"{stream}_calibrator_intercept"], np.asarray([calibrator.intercept], dtype=np.float64)):
                raise v1.ContractStop(f"v010 frozen calibrator differs: {stream}")
    return frozen


def verify_approval_receipt(contract_sha: str) -> dict[str, Any]:
    if not APPROVAL_PATH.is_file():
        raise v1.ContractStop("v010 exact-hash approval receipt is missing")
    receipt = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    required = {
        "contract_id": "history_value_adam_random_v010",
        "contract_sha256": contract_sha,
        "execution_authorized": True,
        "authorized_stages": AUTHORIZED_STAGES,
        "automatic_ordered_transitions_authorized": False,
        "approved_by": "project_owner",
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise v1.ContractStop(f"v010 approval receipt mismatch: {key}")
    return receipt


def render_random_report(result: dict[str, Any], contract_sha: str, selected: dict[str, Any]) -> str:
    report = v9.render_random_report(result, contract_sha, selected)
    report = report.replace("random audit 最终结果 v009", "random audit 最终结果 v010")
    report = report.replace(
        "确定性重建 sealed fit/calibrator，并在 443 万 sealed 行上通过预测复现门后冻结。",
        "加载 SHA-256 固定的 sealed 诊断模型；该模型已在 4,431,299 个 sealed 行上验证数值等价。",
    )
    evidence = (
        "- 数值诊断：概率最大漂移 BL1=1.59293234436e-7、BL2=5.136831982e-7；"
        "sealed 指标最大复现误差=7.59325544664e-8；random 打开前模型已固定。\n"
    )
    return report.replace("- SGD：冻结，本结果不声称跨优化器稳健。\n", "- SGD：冻结，本结果不声称跨优化器稳健。\n" + evidence)


def install_overrides() -> None:
    v9.install_overrides()
    v1.CONTRACT_PATH = CONTRACT_PATH
    v1.APPROVAL_PATH = APPROVAL_PATH
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.REPORT_PATH = REPORT_PATH
    v1.load_contract = load_contract
    v1.verify_approval_receipt = verify_approval_receipt


def validate_only() -> None:
    contract, digest = load_contract()
    if contract["contract_id"] != "history_value_adam_random_v010":
        raise v1.ContractStop("wrong v010 contract id")
    v9.verify_sealed_prerequisite(contract)
    v9.verify_count_evidence(contract)
    verify_v009_stop(contract)
    verify_diagnostic(contract)
    load_and_verify_frozen_model(contract)
    v4.read_adam_evidence()
    v1.implementation_records(contract)
    v1.environment_manifest(contract, require_cuda=False)
    v1.progress(f"V010_VALIDATE_ONLY_OK contract_sha256={digest}; random data not opened")


def run_random(approved_hash: str) -> None:
    contract, digest = load_contract()
    if approved_hash != digest:
        raise v1.ContractStop(f"approved v010 contract hash mismatch: observed {digest}")
    verify_approval_receipt(digest)
    v9.verify_sealed_prerequisite(contract)
    v9.verify_count_evidence(contract)
    verify_v009_stop(contract)
    audit = verify_diagnostic(contract)
    v1.implementation_records(contract)
    environment = v1.environment_manifest(contract, require_cuda=True)
    device = torch.device("cuda:0")
    if OUTPUT_ROOT.exists():
        raise v1.ContractStop("v010 output root already exists; overwrite is forbidden")
    OUTPUT_ROOT.mkdir(parents=True)

    v1.append_access("sealed_prerequisite", "exact_v008_pass_and_18_artifact_hashes_verified", digest)
    v1.append_access("v009_stop", "verified_no_random_data_access", digest)
    v1.append_access("preflight", "opened", digest)
    selected = v4.preflight(contract, device)
    v1.append_access("preflight", "complete_hashed", digest)

    v1.append_access("frozen_model_verification", "opened_before_any_random_data_access", digest)
    frozen = load_and_verify_frozen_model(contract)
    verification_dir = OUTPUT_ROOT / "frozen_model_verification"
    verification_dir.mkdir()
    v1.write_json(verification_dir / "verification_audit.json", {
        "status": "pass_random_access_unlocked",
        "diagnostic_audit_sha256": v1.sha256_file(DIAGNOSTIC_AUDIT),
        "frozen_pickle_sha256": v1.sha256_file(FROZEN_PICKLE),
        "frozen_state_sha256": v1.sha256_file(FROZEN_STATE),
        "sealed_target_identity_exact": audit["sealed_target_identity_exact"],
        "diagnostic_random_input_opened": audit["random_input_opened"],
        "model_refit_or_recalibration_performed": False,
    })
    v1.finalize_hashes(verification_dir)
    v1.append_access("frozen_model_verification", "complete_hashed_no_refit", digest)

    v1.gpu.assert_stage_access("random_audit", {"preflight", "validation", "sealed_test"})
    v1.append_access("random_audit", "opened_after_exact_frozen_model_verification", digest)
    stage_dir = OUTPUT_ROOT / "random_audit"
    stage_dir.mkdir()
    feature_path, feature_info = v9.materialize_random_features(contract, stage_dir)
    frame = v7.read_frame_with_time(feature_path)
    target_index, predictions = v1.score_random_stage(frame, frozen, device=device)
    adequacy = [
        {**row, "stage": "random_audit", "source": "sealed_reconstruction_diagnostic_v010"}
        for row in csv_rows(ADEQUACY_PATH)
    ]
    calibration = [
        {**row, "stage": "random_audit", "source": "sealed_reconstruction_diagnostic_v010"}
        for row in csv_rows(CALIBRATION_PATH)
    ]
    result = v1.evaluate_and_freeze(
        frame, target_index, predictions, stage="random_audit", stage_dir=stage_dir,
        contract_sha=digest, feature_info=feature_info, adequacy_rows=adequacy,
        calibration_rows=calibration, environment=environment,
    )
    v1.append_access("random_audit", "complete_hashed", digest)
    random_pass = result["decision"]["scientific_status"] == "pass"
    terminal = {
        "status": (
            "history_supported_on_standard_and_random_under_frozen_Adam"
            if random_pass else
            "history_supported_on_standard_but_random_transport_not_supported"
        ),
        "Validation_status": "pass",
        "sealed_test_status": "pass",
        "random_audit_status": result["decision"]["scientific_status"],
        "optimizer_scope": "GPU_Adam_only",
        "optimizer_robustness_established": False,
        "SGD_status": "frozen_deferred",
        "random_fit_or_recalibration_performed": False,
    }
    v1.write_json(OUTPUT_ROOT / "final_claim_decision.json", {
        "contract_sha256": digest, "random_decision": result["decision"],
        "terminal_interpretation": terminal,
    })
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_random_report(result, digest, selected), encoding="utf-8")
    v1.finalize_hashes(OUTPUT_ROOT)
    v1.progress(f"v010 random audit complete; report={REPORT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--release-random", action="store_true")
    parser.add_argument("--approved-contract-sha256")
    return parser.parse_args()


def main() -> int:
    install_overrides()
    args = parse_args()
    if args.validate_only:
        validate_only()
        return 0
    if not args.approved_contract_sha256:
        raise v1.ContractStop("--approved-contract-sha256 is required")
    run_random(args.approved_contract_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
