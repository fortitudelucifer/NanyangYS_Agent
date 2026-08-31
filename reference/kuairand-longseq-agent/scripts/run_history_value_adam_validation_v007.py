#!/usr/bin/env python3
"""Run Validation after loading the existing time_ms identity column."""

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
import run_history_value_adam_validation_v006 as v6


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_adam_validation_contract_v007.yaml"
BASE_CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_adam_validation_contract_v006.yaml"
APPROVAL_PATH = PROJECT_ROOT / "configs/history_value_adam_validation_approval_v007.json"
OUTPUT_ROOT = PROJECT_ROOT / "reports/generated/history_value_adam_validation_v007"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/history_value_adam_validation_results_v007.md"
V006_FAILURE_PATH = PROJECT_ROOT / "reports/generated/history_value_adam_validation_v006/validation_failure.json"
FEATURE_PATH = PROJECT_ROOT / "reports/generated/history_value_adam_validation_v006/validation/features.parquet"
FEATURE_MANIFEST_PATH = PROJECT_ROOT / "reports/generated/history_value_adam_validation_v006/validation/feature_manifest.json"
EXPECTED_BASE_SHA256 = "f5704cd236de1683abc1e5a11ef6f93e59aebf1d6fc1fa982058fbbcc983e7da"
EXPECTED_FEATURE_SHA256 = "8f900c5432feecc60d04f2ffa9cd1a6942678d8de60db8832b5046f1b152eade"
EXPECTED_FEATURE_MANIFEST_SHA256 = "da1c06d8d9d238659440ad612eb903ddb2db0415c505fa23e62cbd6ae6b7f482"
AUTHORIZED_STAGES = ["preflight", "validation"]


def load_contract() -> tuple[dict[str, Any], str]:
    observed = v1.sha256_file(BASE_CONTRACT_PATH)
    if observed != EXPECTED_BASE_SHA256:
        raise v1.ContractStop("v007 base v006 contract hash mismatch")
    base, base_digest = v6.load_contract()
    if base_digest != observed:
        raise v1.ContractStop("v007 base merged digest mismatch")
    overlay = yaml.load(CONTRACT_PATH.read_text(encoding="utf-8"), Loader=v1.UniqueKeyLoader)
    if overlay["base_contract"]["sha256"] != observed:
        raise v1.ContractStop("v007 overlay does not pin v006")
    merged = v2.deep_merge(base, overlay)
    if merged["authorization"]["authorized_stage_scope_after_exact_hash_approval"] != AUTHORIZED_STAGES:
        raise v1.ContractStop("v007 stage scope differs")
    if merged["authorization"]["sealed_test_access_authorized"] or merged["authorization"]["random_audit_access_authorized"]:
        raise v1.ContractStop("v007 later stages must remain locked")
    return merged, v1.sha256_file(CONTRACT_PATH)


def verify_repair_evidence(contract: dict[str, Any]) -> dict[str, Any]:
    evidence = json.loads(V006_FAILURE_PATH.read_text(encoding="utf-8"))
    if evidence["contract_sha256"] != EXPECTED_BASE_SHA256:
        raise v1.ContractStop("v006 failure evidence contract mismatch")
    boundary = evidence["access_and_output_boundary"]
    if boundary["predictions_materialized"] or boundary["metrics_computed"]:
        raise v1.ContractStop("v006 crossed the frozen prediction/metric boundary")
    expected_failure_sha = contract["successor_reason"]["v006_failure_record_expected_sha256"]
    if v1.sha256_file(V006_FAILURE_PATH) != expected_failure_sha:
        raise v1.ContractStop("v006 failure record SHA mismatch")
    reuse = contract["validated_feature_reuse"]
    if FEATURE_PATH.stat().st_size != int(reuse["size_bytes"]):
        raise v1.ContractStop("v006 reusable feature size mismatch")
    if v1.sha256_file(FEATURE_PATH) != EXPECTED_FEATURE_SHA256 or reuse["sha256"] != EXPECTED_FEATURE_SHA256:
        raise v1.ContractStop("v006 reusable feature SHA mismatch")
    if v1.sha256_file(FEATURE_MANIFEST_PATH) != EXPECTED_FEATURE_MANIFEST_SHA256:
        raise v1.ContractStop("v006 feature manifest SHA mismatch")
    schema = v1.pq.ParquetFile(FEATURE_PATH).schema_arrow.names
    required = list(v1.canonical.REQUIRED_COLUMNS) + ["time_ms"] + list(v1.EXTRA_COLUMNS)
    missing = sorted(set(required) - set(schema))
    if missing:
        raise v1.ContractStop(f"v007 feature schema missing columns: {missing}")
    return evidence


def verify_approval_receipt(contract_sha: str) -> dict[str, Any]:
    if not APPROVAL_PATH.is_file():
        raise v1.ContractStop("v007 exact-hash approval receipt is missing")
    receipt = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    required = {
        "contract_id": "history_value_adam_validation_v007",
        "contract_sha256": contract_sha,
        "execution_authorized": True,
        "authorized_stages": AUTHORIZED_STAGES,
        "automatic_ordered_transitions_authorized": False,
        "approved_by": "project_owner",
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise v1.ContractStop(f"v007 approval receipt mismatch: {key}")
    return receipt


def read_frame_with_time(path: Path) -> v1.canonical.Frame:
    columns = list(v1.canonical.REQUIRED_COLUMNS)
    columns.insert(columns.index("long_view"), "time_ms")
    columns.extend(v1.EXTRA_COLUMNS)
    if len(columns) != len(set(columns)):
        raise v1.ContractStop("v007 frame column list contains duplicates")
    table = v1.pq.read_table(path, columns=columns)
    return v1.canonical.Frame(columns={
        name: table.column(name).combine_chunks().to_numpy(zero_copy_only=False)
        for name in columns
    })


def install_overrides() -> None:
    v6.install_overrides()
    v1.CONTRACT_PATH = CONTRACT_PATH
    v1.APPROVAL_PATH = APPROVAL_PATH
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.REPORT_PATH = REPORT_PATH
    v1.load_contract = load_contract
    v1.verify_approval_receipt = verify_approval_receipt


def validate_only() -> None:
    contract, digest = load_contract()
    if contract["contract_id"] != "history_value_adam_validation_v007":
        raise v1.ContractStop("wrong v007 contract id")
    verify_repair_evidence(contract)
    v4.read_adam_evidence()
    v1.implementation_records(contract)
    v1.environment_manifest(contract, require_cuda=False)
    v1.progress(f"V007_VALIDATE_ONLY_OK contract_sha256={digest}; no new stage data opened")


def run_validation(approved_hash: str) -> None:
    contract, digest = load_contract()
    if approved_hash != digest:
        raise v1.ContractStop(f"approved v007 contract hash mismatch: observed {digest}")
    verify_approval_receipt(digest)
    verify_repair_evidence(contract)
    v1.implementation_records(contract)
    environment = v1.environment_manifest(contract, require_cuda=True)
    device = torch.device("cuda:0")
    if OUTPUT_ROOT.exists():
        raise v1.ContractStop("v007 output root already exists; overwrite is forbidden")
    OUTPUT_ROOT.mkdir(parents=True)
    v1.append_access("preflight", "opened", digest)
    selected = v4.preflight(contract, device)
    v1.append_access("preflight", "complete_hashed", digest)

    v1.gpu.assert_stage_access("validation", {"preflight"})
    v1.append_access("validation", "reopened_after_time_ms_loader_repair", digest)
    stage_dir = OUTPUT_ROOT / "validation"
    stage_dir.mkdir()
    prior_manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    feature_info = {
        "status": "reused_exact_v006_validated_feature_artifact",
        "path": str(FEATURE_PATH.relative_to(PROJECT_ROOT)),
        "size_bytes": FEATURE_PATH.stat().st_size,
        "sha256": EXPECTED_FEATURE_SHA256,
        "source_manifest_sha256": EXPECTED_FEATURE_MANIFEST_SHA256,
        "validation": prior_manifest["validation"],
        "verified_inputs": prior_manifest["verified_inputs"],
        "v007_only_change": "time_ms_loaded_into_in_memory_Frame",
    }
    v1.write_json(stage_dir / "feature_reuse_manifest.json", feature_info)
    frame = read_frame_with_time(FEATURE_PATH)
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
        "Validation 阶段结果 v005", "Validation 阶段结果 v007", 1
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    v1.finalize_hashes(OUTPUT_ROOT)
    v1.progress(f"v007 Validation complete; sealed/random locked; report={REPORT_PATH}")


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
