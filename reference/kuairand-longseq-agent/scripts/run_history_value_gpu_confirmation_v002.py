#!/usr/bin/env python3
"""Engineering successor to v001: only the reference max_iter is 2,000."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

import run_history_value_gpu_confirmation_v001 as v1
from kuairand_longseq.models import gate2b_repair_v003 as repair


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_gpu_confirmation_contract_v002.yaml"
BASE_CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_gpu_confirmation_contract_v001.yaml"
APPROVAL_PATH = PROJECT_ROOT / "configs/history_value_gpu_confirmation_approval_v002.json"
OUTPUT_ROOT = PROJECT_ROOT / "reports/generated/history_value_gpu_confirmation_v002"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/history_value_gpu_confirmation_results_v002.md"
EXPECTED_BASE_SHA256 = "71f024b86510643a58bababa97572489bbb50d2352fcab8e89a65e7b9b046d1d"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_contract() -> tuple[dict[str, Any], str]:
    observed = v1.sha256_file(BASE_CONTRACT_PATH)
    if observed != EXPECTED_BASE_SHA256:
        raise v1.ContractStop(
            f"v001 base contract hash mismatch: expected {EXPECTED_BASE_SHA256}, observed {observed}"
        )
    base = yaml.load(BASE_CONTRACT_PATH.read_text(encoding="utf-8"), Loader=v1.UniqueKeyLoader)
    overlay = yaml.load(CONTRACT_PATH.read_text(encoding="utf-8"), Loader=v1.UniqueKeyLoader)
    if overlay["base_contract"]["sha256"] != observed:
        raise v1.ContractStop("successor overlay does not pin the observed base contract")
    merged = deep_merge(base, overlay)
    if merged["optimizer_adequacy"]["reference_solver"]["max_iter"] != 2000:
        raise v1.ContractStop("v002 must set reference max_iter exactly to 2000")
    return merged, v1.sha256_file(CONTRACT_PATH)


def verify_approval_receipt(contract_sha: str) -> dict[str, Any]:
    if not APPROVAL_PATH.is_file():
        raise v1.ContractStop("v002 exact-hash approval receipt is missing")
    receipt = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    required = {
        "contract_id": "history_value_gpu_confirmation_v002",
        "contract_sha256": contract_sha,
        "execution_authorized": True,
        "automatic_ordered_transitions_authorized": True,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise v1.ContractStop(f"v002 approval receipt mismatch: {key}")
    if receipt.get("approved_by") != "project_owner":
        raise v1.ContractStop("v002 approval receipt must identify project_owner")
    return receipt


def reference_objectives(
    matrices: dict[str, Any], labels: Any, *, origin: str
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    objectives: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for model_id in ("BL1", "BL2"):
        v1.progress(f"{origin}: CPU reference {model_id} max_iter=2000")
        _, record, c_value = repair.fit_reference(
            matrices[model_id], labels, alpha=1e-4, max_iter=2000
        )
        if not record.converged:
            raise v1.ContractStop(f"v002 reference did not converge: {origin}/{model_id}")
        objectives[model_id] = record.objective
        rows.append({
            "origin": origin, "model_id": model_id, "reference_C": c_value,
            "successor_reference_max_iter": 2000, **record.as_row(),
        })
    return objectives, rows


def validate_only() -> None:
    contract, digest = load_contract()
    if contract["contract_id"] != "history_value_gpu_confirmation_v002":
        raise v1.ContractStop("wrong successor contract id")
    if tuple(contract["model_matrix"]["required_prediction_streams"]) != v1.STREAMS:
        raise v1.ContractStop("successor changed the five prediction streams")
    v1.implementation_records(contract)
    v1.environment_manifest(contract, require_cuda=False)
    v1.progress(
        f"V002_VALIDATE_ONLY_OK contract_sha256={digest}; governed data not opened"
    )


def render_report(results: dict[str, dict[str, Any]], contract_sha: str, selected: dict[str, Any]) -> str:
    return v1.render_report(results, contract_sha, selected).replace(
        "GPU 顺序确认实验 v001", "GPU 顺序确认实验 v002"
    ).replace(
        "history_value_gpu_confirmation_v001", "history_value_gpu_confirmation_v002"
    )


def install_successor_overrides() -> None:
    v1.CONTRACT_PATH = CONTRACT_PATH
    v1.APPROVAL_PATH = APPROVAL_PATH
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.REPORT_PATH = REPORT_PATH
    v1.load_contract = load_contract
    v1.verify_approval_receipt = verify_approval_receipt
    v1.reference_objectives = reference_objectives
    v1.validate_only = validate_only
    v1.render_report = render_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--release", action="store_true")
    parser.add_argument("--approved-contract-sha256")
    return parser.parse_args()


def main() -> int:
    install_successor_overrides()
    args = parse_args()
    if args.validate_only:
        validate_only()
        return 0
    if not args.approved_contract_sha256:
        raise v1.ContractStop("--approved-contract-sha256 is required for v002 release")
    v1.run_release(args.approved_contract_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
