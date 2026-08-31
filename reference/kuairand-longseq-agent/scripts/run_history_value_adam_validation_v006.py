#!/usr/bin/env python3
"""Run Validation with the corrected canonical-union target row count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

import run_history_value_gpu_confirmation_v001 as v1
import run_history_value_gpu_confirmation_v002 as v2
import run_history_value_adam_confirmation_v004 as v4
import run_history_value_adam_validation_v005 as v5


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_adam_validation_contract_v006.yaml"
BASE_CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_adam_validation_contract_v005.yaml"
APPROVAL_PATH = PROJECT_ROOT / "configs/history_value_adam_validation_approval_v006.json"
OUTPUT_ROOT = PROJECT_ROOT / "reports/generated/history_value_adam_validation_v006"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/history_value_adam_validation_results_v006.md"
V005_FAILURE_PATH = PROJECT_ROOT / "reports/generated/history_value_adam_validation_v005/validation_failure.json"
EXPECTED_BASE_SHA256 = "1b4931b9dafcca51eb565b14504269de121d77a420ba9dee72bf80adbdc1bbf8"
EXPECTED_TARGET_ROWS = 886452
AUTHORIZED_STAGES = ["preflight", "validation"]


def load_contract() -> tuple[dict[str, Any], str]:
    observed = v1.sha256_file(BASE_CONTRACT_PATH)
    if observed != EXPECTED_BASE_SHA256:
        raise v1.ContractStop("v006 base v005 contract hash mismatch")
    base, base_digest = v5.load_contract()
    if base_digest != observed:
        raise v1.ContractStop("v006 base merged digest mismatch")
    overlay = yaml.load(CONTRACT_PATH.read_text(encoding="utf-8"), Loader=v1.UniqueKeyLoader)
    if overlay["base_contract"]["sha256"] != observed:
        raise v1.ContractStop("v006 overlay does not pin v005")
    merged = v2.deep_merge(base, overlay)
    if int(merged["sequential_stage_protocol"]["stage_1_validation"]["expected_target_rows"]) != EXPECTED_TARGET_ROWS:
        raise v1.ContractStop("v006 corrected target count differs")
    if merged["authorization"]["authorized_stage_scope_after_exact_hash_approval"] != AUTHORIZED_STAGES:
        raise v1.ContractStop("v006 stage scope differs")
    if merged["authorization"]["sealed_test_access_authorized"] or merged["authorization"]["random_audit_access_authorized"]:
        raise v1.ContractStop("v006 later stages must remain locked")
    return merged, v1.sha256_file(CONTRACT_PATH)


def verify_failure_evidence(contract: dict[str, Any]) -> dict[str, Any]:
    evidence = json.loads(V005_FAILURE_PATH.read_text(encoding="utf-8"))
    diagnosis = evidence["diagnosis"]
    if evidence["contract_sha256"] != EXPECTED_BASE_SHA256:
        raise v1.ContractStop("v005 failure evidence contract mismatch")
    if evidence["access_and_output_boundary"]["GPU_model_fit_started"]:
        raise v1.ContractStop("v005 unexpectedly started a GPU model fit")
    if evidence["access_and_output_boundary"]["metrics_computed"]:
        raise v1.ContractStop("v005 unexpectedly computed metrics")
    if diagnosis["early_standard_silver_tab1_rows"] + diagnosis["additive_official_label_mismatch_tab1_rows"] != EXPECTED_TARGET_ROWS:
        raise v1.ContractStop("v005 count arithmetic mismatch")
    expected_sha = contract["successor_reason"]["v005_failure_record_expected_sha256"]
    observed_sha = v1.sha256_file(V005_FAILURE_PATH)
    if expected_sha != observed_sha:
        raise v1.ContractStop("v005 failure record SHA mismatch")
    return evidence


def verify_approval_receipt(contract_sha: str) -> dict[str, Any]:
    if not APPROVAL_PATH.is_file():
        raise v1.ContractStop("v006 exact-hash approval receipt is missing")
    receipt = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    required = {
        "contract_id": "history_value_adam_validation_v006",
        "contract_sha256": contract_sha,
        "execution_authorized": True,
        "authorized_stages": AUTHORIZED_STAGES,
        "automatic_ordered_transitions_authorized": False,
        "approved_by": "project_owner",
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise v1.ContractStop(f"v006 approval receipt mismatch: {key}")
    return receipt


def materialize_validation_features(
    contract: dict[str, Any], stage_dir: Path
) -> tuple[Path, dict[str, Any]]:
    path = stage_dir / "features.parquet"
    temp = stage_dir / ".features.building.parquet"
    if path.exists() or temp.exists():
        raise v1.ContractStop("v006 refuses to overwrite a feature artifact")
    keys = ["events_early_standard", "videos_basic", "label_formula_mismatch_rows"]
    verified = [v1.verify_input(contract, key) for key in keys]
    con = v1.duckdb.connect()
    v1.configure_duckdb(con, stage_dir / "duckdb_tmp")
    try:
        validation = v1.materialize_standard_features(
            con,
            early_path=v1.resolve_input(contract, "events_early_standard"),
            late_path=None,
            mismatch_path=v1.resolve_input(contract, "label_formula_mismatch_rows"),
            videos_path=v1.resolve_input(contract, "videos_basic"),
            output_path=temp,
            end_date="2022-04-21",
            target_start="2022-04-18",
            target_end="2022-04-21",
            expected_target_rows=EXPECTED_TARGET_ROWS,
        )
    finally:
        con.close()
    temp.replace(path)
    manifest = v1.feature_manifest(path, validation, verified)
    v1.write_json(stage_dir / "feature_manifest.json", manifest)
    return path, manifest


def install_overrides() -> None:
    v5.install_overrides()
    v1.CONTRACT_PATH = CONTRACT_PATH
    v1.APPROVAL_PATH = APPROVAL_PATH
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.REPORT_PATH = REPORT_PATH
    v1.load_contract = load_contract
    v1.verify_approval_receipt = verify_approval_receipt


def validate_only() -> None:
    contract, digest = load_contract()
    if contract["contract_id"] != "history_value_adam_validation_v006":
        raise v1.ContractStop("wrong v006 contract id")
    verify_failure_evidence(contract)
    v4.read_adam_evidence()
    v1.implementation_records(contract)
    v1.environment_manifest(contract, require_cuda=False)
    v1.progress(f"V006_VALIDATE_ONLY_OK contract_sha256={digest}; governed data not opened")


def run_validation(approved_hash: str) -> None:
    contract, digest = load_contract()
    if approved_hash != digest:
        raise v1.ContractStop(f"approved v006 contract hash mismatch: observed {digest}")
    verify_approval_receipt(digest)
    verify_failure_evidence(contract)
    v1.implementation_records(contract)
    environment = v1.environment_manifest(contract, require_cuda=True)
    device = torch.device("cuda:0")
    if OUTPUT_ROOT.exists():
        raise v1.ContractStop("v006 output root already exists; overwrite is forbidden")
    OUTPUT_ROOT.mkdir(parents=True)
    v1.append_access("preflight", "opened", digest)
    selected = v4.preflight(contract, device)
    v1.append_access("preflight", "complete_hashed", digest)

    v1.gpu.assert_stage_access("validation", {"preflight"})
    v1.append_access("validation", "opened_after_preflight_count_metadata_repair", digest)
    stage_dir = OUTPUT_ROOT / "validation"
    stage_dir.mkdir()
    feature_path, feature_info = materialize_validation_features(contract, stage_dir)
    frame = v1.read_frame(feature_path)
    target_index, predictions, _, adequacy, calibration = v4.fit_standard_stage(
        frame, stage="validation", fit_range=("2022-04-08", "2022-04-16"),
        calibration_date="2022-04-17", target_range=("2022-04-18", "2022-04-21"),
        selected=selected, device=device,
    )
    result = v1.evaluate_and_freeze(
        frame, target_index, predictions, stage="validation", stage_dir=stage_dir,
        contract_sha=digest, feature_info=feature_info, adequacy_rows=adequacy,
        calibration_rows=calibration, environment=environment,
    )
    v1.append_access("validation", "complete_hashed_stopped_for_project_owner_review", digest)
    checkpoint = {
        "status": "validation_complete_later_stages_locked",
        "contract_sha256": digest,
        "validation_decision": result["decision"],
        "sealed_test_accessed": False,
        "random_audit_accessed": False,
        "next_action": "project_owner_review_and_separate_contract_approval",
    }
    v1.write_json(OUTPUT_ROOT / "validation_checkpoint.json", checkpoint)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = v5.render_validation_report(result, digest, selected).replace(
        "Validation 阶段结果 v005", "Validation 阶段结果 v006", 1
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    v1.finalize_hashes(OUTPUT_ROOT)
    v1.progress(f"v006 Validation complete; sealed/random locked; report={REPORT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--release-validation", action="store_true")
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
    run_validation(args.approved_contract_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
