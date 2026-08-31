#!/usr/bin/env python3
"""v003 successor: reuse adequate Adam evidence and extend plain SGD only."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import run_history_value_gpu_confirmation_v001 as v1
import run_history_value_gpu_confirmation_v002 as v2
from kuairand_longseq.models import gate2b_repair_v003 as repair
from kuairand_longseq.models import history_value_gpu as gpu


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_gpu_confirmation_contract_v003.yaml"
BASE_CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_gpu_confirmation_contract_v002.yaml"
APPROVAL_PATH = PROJECT_ROOT / "configs/history_value_gpu_confirmation_approval_v003.json"
OUTPUT_ROOT = PROJECT_ROOT / "reports/generated/history_value_gpu_confirmation_v003"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/history_value_gpu_confirmation_results_v003.md"
V002_ADEQUACY_PATH = PROJECT_ROOT / "reports/generated/history_value_gpu_confirmation_v002/preflight/optimization_adequacy.csv"
EXPECTED_BASE_SHA256 = "a11e75f43b2fe013e9ea0ee813953b46c224f8a5786e64a1c3bd25a43b37d09b"
EXPECTED_V002_ADEQUACY_SHA256 = "79867555a31a96f5ffe1be38930de69ad491d70fea5aed56008d7997d8cbcf16"
BASE_RENDER_REPORT = v1.render_report


def load_contract() -> tuple[dict[str, Any], str]:
    observed = v1.sha256_file(BASE_CONTRACT_PATH)
    if observed != EXPECTED_BASE_SHA256:
        raise v1.ContractStop("v003 base v002 contract hash mismatch")
    base, base_digest = v2.load_contract()
    if base_digest != observed:
        raise v1.ContractStop("v002 merged loader digest mismatch")
    overlay = yaml.load(CONTRACT_PATH.read_text(encoding="utf-8"), Loader=v1.UniqueKeyLoader)
    if overlay["base_contract"]["sha256"] != observed:
        raise v1.ContractStop("v003 overlay does not pin v002")
    merged = v2.deep_merge(base, overlay)
    if merged["GPU_optimizers"]["GPU_SGD"]["learning_rate_candidates"] != [1.0]:
        raise v1.ContractStop("v003 plain SGD learning rate must be exactly [1.0]")
    if merged["GPU_optimizers"]["GPU_SGD"]["candidate_step_checkpoints"] != [3000, 5000, 10000]:
        raise v1.ContractStop("v003 plain SGD checkpoints differ")
    return merged, v1.sha256_file(CONTRACT_PATH)


def verify_approval_receipt(contract_sha: str) -> dict[str, Any]:
    if not APPROVAL_PATH.is_file():
        raise v1.ContractStop("v003 exact-hash approval receipt is missing")
    receipt = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    required = {
        "contract_id": "history_value_gpu_confirmation_v003",
        "contract_sha256": contract_sha,
        "execution_authorized": True,
        "automatic_ordered_transitions_authorized": True,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise v1.ContractStop(f"v003 approval receipt mismatch: {key}")
    if receipt.get("approved_by") != "project_owner":
        raise v1.ContractStop("v003 approval receipt must identify project_owner")
    return receipt


def read_v002_adam_evidence() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if V002_ADEQUACY_PATH.stat().st_size != 23470:
        raise v1.ContractStop("v002 adequacy artifact size mismatch")
    if v1.sha256_file(V002_ADEQUACY_PATH) != EXPECTED_V002_ADEQUACY_SHA256:
        raise v1.ContractStop("v002 adequacy artifact SHA mismatch")
    with V002_ADEQUACY_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if row["optimizer"] == "ADAM" and int(row["steps"]) == 100
    ]
    if len(selected) != 6 or not all(row["adequacy_passed"] == "True" for row in selected):
        raise v1.ContractStop("v002 does not contain the required 6/6 Adam evidence")
    maximum_regret = max(float(row["objective_regret"]) for row in selected)
    return selected, {
        "source_path": str(V002_ADEQUACY_PATH.relative_to(PROJECT_ROOT)),
        "source_sha256": EXPECTED_V002_ADEQUACY_SHA256,
        "learning_rate": 0.03,
        "steps": 100,
        "passed_combinations": 6,
        "maximum_regret": maximum_regret,
    }


def preflight(contract: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = v1.OUTPUT_ROOT / "preflight"
    if out.exists():
        raise v1.ContractStop("v003 preflight output already exists")
    out.mkdir(parents=True)
    inherited_adam_rows, inherited_adam = read_v002_adam_evidence()
    declared = contract["implementation_status"]["optimizer_preflight_feature_artifact"]
    if v1.TRAIN_FEATURE.stat().st_size != int(declared["size_bytes"]) or v1.sha256_file(v1.TRAIN_FEATURE) != declared["sha256"]:
        raise v1.ContractStop("frozen Train feature hash mismatch")
    frame = v1.read_frame_with_defaults(v1.TRAIN_FEATURE)
    dates, labels = frame.dates(), frame.labels()
    adequacy_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    origins = list(contract["optimizer_adequacy"]["train_probe_origins"])
    for origin_text in origins:
        cutoff = np.datetime64(origin_text, "D") - np.timedelta64(1, "D")
        fit_index = np.flatnonzero(dates < cutoff)
        prevalence = float(labels[fit_index].mean())
        blocks = v1.raw_blocks(frame, fit_index, prevalence)
        _, bl1, bl2 = repair.fit_grouped_design(prevalence=prevalence, **blocks)
        matrices = {"BL1": bl1, "BL2": bl2}
        references, rows = v2.reference_objectives(
            matrices, labels[fit_index], origin=origin_text
        )
        reference_rows.extend(rows)
        for model_id in ("BL1", "BL2"):
            v1.progress(f"v003 preflight {origin_text} SGD lr=1 BL1/BL2 current={model_id}")
            fit = gpu.fit_trajectory(
                matrices[model_id], labels[fit_index], device=device,
                optimizer_name="SGD", learning_rate=1.0,
                checkpoints=[3000, 5000, 10000], alpha=1e-4,
            )
            for trace in fit.objective_trace:
                adequacy_rows.append({
                    "origin": origin_text, "optimizer": "SGD", "learning_rate": 1.0,
                    "steps": int(trace["step"]), "model_id": model_id,
                    "terminal_gradient_norm_at_max_checkpoint": fit.terminal_gradient_norm
                    if int(trace["step"]) == 10000 else None,
                    "trajectory_elapsed_seconds": fit.elapsed_seconds,
                    "trajectory_peak_cuda_memory_bytes": fit.peak_cuda_memory_bytes,
                    **gpu.adequacy(float(trace["objective"]), references[model_id], reference_converged=True),
                })
        del matrices, bl1, bl2, blocks
        gc.collect()
    candidates = []
    for steps in (3000, 5000, 10000):
        rows = [row for row in adequacy_rows if row["steps"] == steps]
        if len(rows) == 6 and all(row["adequacy_passed"] for row in rows):
            candidates.append((steps, max(float(row["objective_regret"]) for row in rows)))
    v1.write_csv(out / "optimization_adequacy.csv", adequacy_rows)
    v1.write_csv(out / "reference_solver_audit.csv", reference_rows)
    v1.write_json(out / "inherited_Adam_adequacy_evidence.json", {
        "manifest": inherited_adam, "rows": inherited_adam_rows,
    })
    if not candidates:
        raise v1.ContractStop("v003 plain SGD lr=1 has no adequate checkpoint")
    sgd_steps, sgd_regret = min(candidates)
    selected = {
        "status": "complete_frozen_before_validation",
        "ADAM": {
            "learning_rate": 0.03, "steps": 100,
            "maximum_regret": inherited_adam["maximum_regret"],
            "evidence": "hash_verified_v002_preflight",
        },
        "SGD": {
            "learning_rate": 1.0, "steps": sgd_steps,
            "maximum_regret": sgd_regret, "evidence": "v003_new_train_only_preflight",
        },
    }
    v1.write_json(v1.OUTPUT_ROOT / "selected_optimizer_configuration_manifest.json", selected)
    v1.finalize_hashes(out)
    return selected


def validate_only() -> None:
    contract, digest = load_contract()
    if contract["contract_id"] != "history_value_gpu_confirmation_v003":
        raise v1.ContractStop("wrong v003 contract id")
    read_v002_adam_evidence()
    v1.implementation_records(contract)
    v1.environment_manifest(contract, require_cuda=False)
    v1.progress(f"V003_VALIDATE_ONLY_OK contract_sha256={digest}; governed data not opened")


def render_report(results: dict[str, dict[str, Any]], contract_sha: str, selected: dict[str, Any]) -> str:
    return BASE_RENDER_REPORT(results, contract_sha, selected).replace(
        "GPU 顺序确认实验 v001", "GPU 顺序确认实验 v003"
    ).replace("history_value_gpu_confirmation_v001", "history_value_gpu_confirmation_v003")


def install_overrides() -> None:
    v1.CONTRACT_PATH = CONTRACT_PATH
    v1.APPROVAL_PATH = APPROVAL_PATH
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.REPORT_PATH = REPORT_PATH
    v1.load_contract = load_contract
    v1.verify_approval_receipt = verify_approval_receipt
    v1.reference_objectives = v2.reference_objectives
    v1.preflight = preflight
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
    install_overrides()
    args = parse_args()
    if args.validate_only:
        validate_only()
        return 0
    if not args.approved_contract_sha256:
        raise v1.ContractStop("--approved-contract-sha256 is required for v003 release")
    v1.run_release(args.approved_contract_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
