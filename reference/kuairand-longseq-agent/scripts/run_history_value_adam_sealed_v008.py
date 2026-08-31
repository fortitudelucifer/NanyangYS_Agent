#!/usr/bin/env python3
"""Run only the sealed standard-exposure stage after Validation passed."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

import run_history_value_gpu_confirmation_v001 as v1
import run_history_value_gpu_confirmation_v002 as v2
import run_history_value_adam_confirmation_v004 as v4
import run_history_value_adam_validation_v007 as v7


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_adam_sealed_contract_v008.yaml"
BASE_CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_adam_validation_contract_v007.yaml"
APPROVAL_PATH = PROJECT_ROOT / "configs/history_value_adam_sealed_approval_v008.json"
OUTPUT_ROOT = PROJECT_ROOT / "reports/generated/history_value_adam_sealed_v008"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/history_value_adam_sealed_results_v008.md"
VALIDATION_CHECKPOINT = PROJECT_ROOT / "reports/generated/history_value_adam_validation_v007/validation_checkpoint.json"
VALIDATION_DECISION = PROJECT_ROOT / "reports/generated/history_value_adam_validation_v007/validation/stage_decision.json"
VALIDATION_MANIFEST = PROJECT_ROOT / "reports/generated/history_value_adam_validation_v007/validation/artifact_hash_manifest.json"
COUNT_EVIDENCE_PATH = PROJECT_ROOT / "reports/analysis/sealed_target_count_precheck_2026-08-23.csv"
EXPECTED_BASE_SHA256 = "b6d908af14a29ad5b54454eb2ca30bb944e9891db4f4414308948638e6b6d897"
EXPECTED_TARGET_ROWS = 4431299
AUTHORIZED_STAGES = ["preflight", "sealed_test"]


def load_contract() -> tuple[dict[str, Any], str]:
    observed = v1.sha256_file(BASE_CONTRACT_PATH)
    if observed != EXPECTED_BASE_SHA256:
        raise v1.ContractStop("v008 base v007 contract hash mismatch")
    base, base_digest = v7.load_contract()
    if base_digest != observed:
        raise v1.ContractStop("v008 base merged digest mismatch")
    overlay = yaml.load(CONTRACT_PATH.read_text(encoding="utf-8"), Loader=v1.UniqueKeyLoader)
    if overlay["base_contract"]["sha256"] != observed:
        raise v1.ContractStop("v008 overlay does not pin v007")
    merged = v2.deep_merge(base, overlay)
    if int(merged["sequential_stage_protocol"]["stage_2_sealed_test"]["expected_target_rows"]) != EXPECTED_TARGET_ROWS:
        raise v1.ContractStop("v008 sealed target count differs")
    if merged["authorization"]["authorized_stage_scope_after_exact_hash_approval"] != AUTHORIZED_STAGES:
        raise v1.ContractStop("v008 authorized stage scope differs")
    if merged["authorization"]["random_audit_access_authorized"]:
        raise v1.ContractStop("v008 must not authorize random audit")
    return merged, v1.sha256_file(CONTRACT_PATH)


def verify_validation_prerequisite(contract: dict[str, Any]) -> dict[str, Any]:
    prereq = contract["validation_prerequisite"]
    expected = {
        VALIDATION_CHECKPOINT: prereq["checkpoint_sha256"],
        VALIDATION_DECISION: prereq["stage_decision_sha256"],
        VALIDATION_MANIFEST: prereq["stage_artifact_manifest_sha256"],
    }
    for path, digest in expected.items():
        if v1.sha256_file(path) != digest:
            raise v1.ContractStop(f"v008 Validation prerequisite SHA mismatch: {path.name}")
    checkpoint = json.loads(VALIDATION_CHECKPOINT.read_text(encoding="utf-8"))
    decision = json.loads(VALIDATION_DECISION.read_text(encoding="utf-8"))
    if checkpoint["contract_sha256"] != EXPECTED_BASE_SHA256:
        raise v1.ContractStop("v008 Validation checkpoint contract mismatch")
    if checkpoint["status"] != "validation_complete_later_stages_locked":
        raise v1.ContractStop("v008 Validation checkpoint status mismatch")
    if decision["scientific_status"] != prereq["required_scientific_status"]:
        raise v1.ContractStop("v008 requires the frozen Validation pass")
    if checkpoint["sealed_test_accessed"] or checkpoint["random_audit_accessed"]:
        raise v1.ContractStop("v008 Validation predecessor crossed later-stage boundary")
    manifest = json.loads(VALIDATION_MANIFEST.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = Path(artifact["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file() or path.stat().st_size != int(artifact["size_bytes"]):
            raise v1.ContractStop(f"v008 missing Validation artifact: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise v1.ContractStop(f"v008 Validation artifact SHA mismatch: {path}")
    return checkpoint


def verify_count_evidence(contract: dict[str, Any]) -> None:
    evidence = contract["sealed_target_count_repair"]
    if v1.sha256_file(COUNT_EVIDENCE_PATH) != evidence["count_evidence_sha256"]:
        raise v1.ContractStop("v008 sealed target count evidence SHA mismatch")
    if int(evidence["prior_main_table_only_count"]) + int(evidence["official_formula_mismatch_addback_rows"]) != EXPECTED_TARGET_ROWS:
        raise v1.ContractStop("v008 sealed target count arithmetic mismatch")


def verify_approval_receipt(contract_sha: str) -> dict[str, Any]:
    if not APPROVAL_PATH.is_file():
        raise v1.ContractStop("v008 exact-hash approval receipt is missing")
    receipt = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    required = {
        "contract_id": "history_value_adam_sealed_v008",
        "contract_sha256": contract_sha,
        "execution_authorized": True,
        "authorized_stages": AUTHORIZED_STAGES,
        "automatic_ordered_transitions_authorized": False,
        "approved_by": "project_owner",
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise v1.ContractStop(f"v008 approval receipt mismatch: {key}")
    return receipt


def materialize_sealed_features(
    contract: dict[str, Any], stage_dir: Path
) -> tuple[Path, dict[str, Any]]:
    path = stage_dir / "features.parquet"
    temp = stage_dir / ".features.building.parquet"
    if path.exists() or temp.exists():
        raise v1.ContractStop("v008 refuses to overwrite a feature artifact")
    keys = ["events_early_standard", "events_late_standard", "videos_basic", "label_formula_mismatch_rows"]
    verified = [v1.verify_input(contract, key) for key in keys]
    con = v1.duckdb.connect()
    v1.configure_duckdb(con, stage_dir / "duckdb_tmp")
    try:
        validation = v1.materialize_standard_features(
            con,
            early_path=v1.resolve_input(contract, "events_early_standard"),
            late_path=v1.resolve_input(contract, "events_late_standard"),
            mismatch_path=v1.resolve_input(contract, "label_formula_mismatch_rows"),
            videos_path=v1.resolve_input(contract, "videos_basic"),
            output_path=temp,
            end_date="2022-05-08",
            target_start="2022-04-22",
            target_end="2022-05-08",
            expected_target_rows=EXPECTED_TARGET_ROWS,
        )
    finally:
        con.close()
    temp.replace(path)
    manifest = v1.feature_manifest(path, validation, verified)
    v1.write_json(stage_dir / "feature_manifest.json", manifest)
    return path, manifest


def render_sealed_report(
    result: dict[str, Any], contract_sha: str, selected: dict[str, Any]
) -> str:
    pooled = {(row["scope"], row["model_id"]): row for row in result["pooled"]}
    all_rows = pooled[("all_target_rows", "BL0")]
    warm = pooled[("primary_warm_user", "BL0")]
    decision = result["decision"]
    lines = [
        "# 历史特征价值 GPU Adam：sealed test 阶段结果 v008", "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        f"- 合同 SHA-256：`{contract_sha}`",
        f"- Adam：lr={selected['ADAM']['learning_rate']}，steps={selected['ADAM']['steps']}。",
        "- 前置条件：v007 Validation 完整阶段判定为 pass，且全部哈希已复核。",
        "- 权限边界：只执行 sealed test；random audit 未打开。",
        "- SGD：冻结，本结果不声称跨优化器稳健。", "",
        "## 数据来源与问题", "",
        "目标为 KuaiRand 2022-04-22 至 05-08 的标准曝光 `tab=1` canonical union；"
        "训练重拟合窗为 04-08 至 04-20，04-21 仅做前一日概率校准。"
        "历史严格满足 `history_time_ms < target_time_ms`。", "",
        "问题是 Validation 支持的 ADAM_BL2−ADAM_BL1 历史增量，能否在更晚的 17 天窗口保持。", "",
        "## 样本", "",
        "| 范围 | 行数 | 用户数 | 正例数 | 正例率 |", "|---|---:|---:|---:|---:|",
        f"| 全部 sealed 目标 | {all_rows['rows']:,} | {all_rows['users']:,} | {all_rows['positives']:,} | {all_rows['prevalence']:.6f} |",
        f"| 主分析 warm users | {warm['rows']:,} | {warm['users']:,} | {warm['positives']:,} | {warm['prevalence']:.6f} |", "",
        "## 历史增量与统计区间", "",
        "| 指标 | 点估计 | 95% CI |", "|---|---:|---:|",
    ]
    for metric, label in (
        ("average_precision", "ΔAP"),
        ("user_gauc_event_weighted", "Δevent-gAUC"),
        ("log_loss", "Δlog-loss"),
        ("brier", "ΔBrier"),
    ):
        row = v1.metric_lookup(result, "sealed_test", "ADAM_BL2_minus_ADAM_BL1", metric)
        lines.append(f"| {label} | {row['point_estimate']:.6f} | [{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}] |")
    absolute = decision["absolute_probability_gate"]
    lines.extend([
        "", "## 预注册门与结论", "",
        f"- 静态 BL1 相对 BL0 门：{decision['static_baseline_gate']['ADAM']}。",
        f"- BL1/BL2 绝对概率门：BL1={absolute['ADAM_BL1']}，BL2={absolute['ADAM_BL2']}。",
        f"- 历史增量门：{decision['history_gate']['ADAM']}。",
        f"- 逐日 AP 正向：{decision['positive_AP_days']['ADAM']}/{decision['required_positive_AP_days']}。",
        f"- 概率健康门：{decision['probability_saturation_gate']}。",
        f"- sealed test 完整阶段判定：`{decision['scientific_status']}`。", "",
    ])
    if decision["scientific_status"] == "pass":
        lines.append("结论：Validation 支持的历史增量在更晚的 17 天 sealed 窗口持续成立；是否迁移到随机曝光仍未知。")
    else:
        lines.append("结论：历史增量未通过 sealed 完整阶段门，因此 Validation 效应没有在更晚的 17 天窗口稳定复现。")
    lines.extend([
        "", "这是离线预测证据，不证明因果或线上业务提升。random audit 是否运行需项目负责人另行决定。", "",
        "## 可作图文件", "",
        "sealed_test 目录提供逐行 predictions.parquet，以及 pooled、daily、contrast、2,000 次 bootstrap、"
        "历史深度切片和概率分布 CSV。", "",
    ])
    return "\n".join(lines)


def install_overrides() -> None:
    v7.install_overrides()
    v1.CONTRACT_PATH = CONTRACT_PATH
    v1.APPROVAL_PATH = APPROVAL_PATH
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.REPORT_PATH = REPORT_PATH
    v1.load_contract = load_contract
    v1.verify_approval_receipt = verify_approval_receipt


def validate_only() -> None:
    contract, digest = load_contract()
    if contract["contract_id"] != "history_value_adam_sealed_v008":
        raise v1.ContractStop("wrong v008 contract id")
    verify_validation_prerequisite(contract)
    verify_count_evidence(contract)
    v4.read_adam_evidence()
    v1.implementation_records(contract)
    v1.environment_manifest(contract, require_cuda=False)
    v1.progress(f"V008_VALIDATE_ONLY_OK contract_sha256={digest}; sealed model data not opened")


def run_sealed(approved_hash: str) -> None:
    contract, digest = load_contract()
    if approved_hash != digest:
        raise v1.ContractStop(f"approved v008 contract hash mismatch: observed {digest}")
    verify_approval_receipt(digest)
    verify_validation_prerequisite(contract)
    verify_count_evidence(contract)
    v1.implementation_records(contract)
    environment = v1.environment_manifest(contract, require_cuda=True)
    device = torch.device("cuda:0")
    if OUTPUT_ROOT.exists():
        raise v1.ContractStop("v008 output root already exists; overwrite is forbidden")
    OUTPUT_ROOT.mkdir(parents=True)
    v1.append_access("validation_prerequisite", "exact_v007_pass_and_17_artifact_hashes_verified", digest)
    v1.append_access("preflight", "opened", digest)
    selected = v4.preflight(contract, device)
    v1.append_access("preflight", "complete_hashed", digest)

    v1.gpu.assert_stage_access("sealed_test", {"preflight", "validation"})
    v1.append_access("sealed_test", "opened_after_exact_Validation_pass_prerequisite", digest)
    stage_dir = OUTPUT_ROOT / "sealed_test"
    stage_dir.mkdir()
    feature_path, feature_info = materialize_sealed_features(contract, stage_dir)
    frame = v7.read_frame_with_time(feature_path)
    target_index, predictions, _, adequacy, calibration = v4.fit_standard_stage(
        frame, stage="sealed_test", fit_range=("2022-04-08", "2022-04-20"),
        calibration_date="2022-04-21", target_range=("2022-04-22", "2022-05-08"),
        selected=selected, device=device,
    )
    result = v1.evaluate_and_freeze(
        frame, target_index, predictions, stage="sealed_test", stage_dir=stage_dir,
        contract_sha=digest, feature_info=feature_info, adequacy_rows=adequacy,
        calibration_rows=calibration, environment=environment,
    )
    v1.append_access("sealed_test", "complete_hashed_stopped_for_project_owner_review", digest)
    checkpoint = {
        "status": "sealed_complete_random_audit_locked",
        "contract_sha256": digest,
        "validation_prerequisite_contract_sha256": EXPECTED_BASE_SHA256,
        "sealed_decision": result["decision"],
        "random_audit_accessed": False,
        "next_action": "project_owner_review_and_separate_random_audit_decision",
    }
    v1.write_json(OUTPUT_ROOT / "sealed_checkpoint.json", checkpoint)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_sealed_report(result, digest, selected), encoding="utf-8")
    v1.finalize_hashes(OUTPUT_ROOT)
    v1.progress(f"v008 sealed test complete; random audit locked; report={REPORT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--release-sealed", action="store_true")
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
    run_sealed(args.approved_contract_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
