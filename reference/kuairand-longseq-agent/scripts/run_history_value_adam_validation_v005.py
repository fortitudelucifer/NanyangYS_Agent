#!/usr/bin/env python3
"""Run only the authorized Adam Validation stage and keep later stages locked."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

import run_history_value_gpu_confirmation_v001 as v1
import run_history_value_gpu_confirmation_v002 as v2
import run_history_value_adam_confirmation_v004 as v4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_adam_validation_contract_v005.yaml"
BASE_CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_adam_confirmation_contract_v004.yaml"
APPROVAL_PATH = PROJECT_ROOT / "configs/history_value_adam_validation_approval_v005.json"
OUTPUT_ROOT = PROJECT_ROOT / "reports/generated/history_value_adam_validation_v005"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/history_value_adam_validation_results_v005.md"
EXPECTED_BASE_SHA256 = "3c77577c0a5d53d66869baeaada41eab45329a91fc3911a113174e6e93d1295c"
AUTHORIZED_STAGES = ["preflight", "validation"]


def load_contract() -> tuple[dict[str, Any], str]:
    observed = v1.sha256_file(BASE_CONTRACT_PATH)
    if observed != EXPECTED_BASE_SHA256:
        raise v1.ContractStop("v005 base v004 contract hash mismatch")
    base, base_digest = v4.load_contract()
    if base_digest != observed:
        raise v1.ContractStop("v005 base merged digest mismatch")
    overlay = yaml.load(CONTRACT_PATH.read_text(encoding="utf-8"), Loader=v1.UniqueKeyLoader)
    if overlay["base_contract"]["sha256"] != observed:
        raise v1.ContractStop("v005 overlay does not pin v004")
    merged = v2.deep_merge(base, overlay)
    if merged["authorization"]["authorized_stage_scope_after_exact_hash_approval"] != AUTHORIZED_STAGES:
        raise v1.ContractStop("v005 authorized stage scope differs")
    if merged["authorization"]["sealed_test_access_authorized"]:
        raise v1.ContractStop("v005 must not authorize sealed test")
    if merged["authorization"]["random_audit_access_authorized"]:
        raise v1.ContractStop("v005 must not authorize random audit")
    return merged, v1.sha256_file(CONTRACT_PATH)


def verify_approval_receipt(contract_sha: str) -> dict[str, Any]:
    if not APPROVAL_PATH.is_file():
        raise v1.ContractStop("v005 exact-hash approval receipt is missing")
    receipt = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    required = {
        "contract_id": "history_value_adam_validation_v005",
        "contract_sha256": contract_sha,
        "execution_authorized": True,
        "authorized_stages": AUTHORIZED_STAGES,
        "automatic_ordered_transitions_authorized": False,
        "approved_by": "project_owner",
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise v1.ContractStop(f"v005 approval receipt mismatch: {key}")
    return receipt


def render_validation_report(
    result: dict[str, Any], contract_sha: str, selected: dict[str, Any]
) -> str:
    pooled = {(row["scope"], row["model_id"]): row for row in result["pooled"]}
    all_rows = pooled[("all_target_rows", "BL0")]
    warm = pooled[("primary_warm_user", "BL0")]
    decision = result["decision"]
    metrics = [
        ("average_precision", "ΔAP"),
        ("user_gauc_event_weighted", "Δevent-gAUC"),
        ("log_loss", "Δlog-loss"),
        ("brier", "ΔBrier"),
    ]
    lines = [
        "# 历史特征价值 GPU Adam：Validation 阶段结果 v005", "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        f"- 合同 SHA-256：`{contract_sha}`",
        f"- Adam：lr={selected['ADAM']['learning_rate']}，steps={selected['ADAM']['steps']}。",
        "- 权限边界：只执行并封存 Validation；sealed test 与 random audit 均未打开。",
        "- SGD：已冻结，本结果不声称跨优化器稳健。", "",
        "## 数据来源与问题", "",
        "数据来自 KuaiRand 正式 Silver early-standard 快照、官方 long_view 不一致回补隔离行和 videos_basic 静态元数据。"
        "目标为 2022-04-18 至 2022-04-21 的标准曝光 `tab=1`；训练窗为 2022-04-08 至 04-16，"
        "04-17 只用于前一日概率校准。历史严格满足 `history_time_ms < target_time_ms`。", "",
        "BL0 是冻结常数概率；ADAM_BL1 是静态特征基线；ADAM_BL2 是完全相同静态特征加 H2 严格历史。"
        "主问题是同一目标行上 ADAM_BL2 − ADAM_BL1 是否通过预注册增量门。", "",
        "## 样本", "",
        "| 范围 | 行数 | 用户数 | 正例数 | 正例率 |", "|---|---:|---:|---:|---:|",
        f"| 全部 Validation 目标 | {all_rows['rows']:,} | {all_rows['users']:,} | {all_rows['positives']:,} | {all_rows['prevalence']:.6f} |",
        f"| 主分析 warm users | {warm['rows']:,} | {warm['users']:,} | {warm['positives']:,} | {warm['prevalence']:.6f} |", "",
        "## 历史增量与统计区间", "",
        "2,000 次配对用户簇 bootstrap；正的 AP/gAUC 和负的 log-loss/Brier 表示 BL2 优于 BL1。", "",
        "| 指标 | 点估计 | 95% CI |", "|---|---:|---:|",
    ]
    for metric, label in metrics:
        row = v1.metric_lookup(result, "validation", "ADAM_BL2_minus_ADAM_BL1", metric)
        lines.append(
            f"| {label} | {row['point_estimate']:.6f} | [{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}] |"
        )
    absolute = decision["absolute_probability_gate"]
    lines.extend([
        "", "## 预注册门与结论", "",
        f"- 静态 BL1 相对 BL0 门：{decision['static_baseline_gate']['ADAM']}。",
        f"- BL1/BL2 绝对概率门：BL1={absolute['ADAM_BL1']}，BL2={absolute['ADAM_BL2']}。",
        f"- 历史增量门：{decision['history_gate']['ADAM']}。",
        f"- 逐日 AP 正向：{decision['positive_AP_days']['ADAM']}/{decision['required_positive_AP_days']}。",
        f"- 概率健康门：{decision['probability_saturation_gate']}。",
        f"- Validation 完整阶段判定：`{decision['scientific_status']}`。", "",
    ])
    if decision["scientific_status"] == "pass":
        lines.append("结论：Validation 支持冻结 Adam 协议下的历史增量；是否跨时间和迁移到随机曝光仍未知，需另行批准后续阶段。")
    else:
        lines.append("结论：Validation 未通过完整预注册门，广义历史增量在首个确认阶段被证伪；后续若运行只能刻画异质性，不能抹去本次失败。")
    lines.extend([
        "", "这是离线预测证据，不证明因果或线上业务提升。", "",
        "## 可作图文件", "",
        "Validation 目录包含逐行 predictions.parquet，以及 pooled、daily、contrast、2,000 次 bootstrap、"
        "历史深度切片和概率分布 CSV，可直接绘制置信区间、逐日趋势、分层效应与概率健康图。", "",
    ])
    return "\n".join(lines)


def install_overrides() -> None:
    v4.install_overrides()
    v1.CONTRACT_PATH = CONTRACT_PATH
    v1.APPROVAL_PATH = APPROVAL_PATH
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.REPORT_PATH = REPORT_PATH
    v1.load_contract = load_contract
    v1.verify_approval_receipt = verify_approval_receipt


def validate_only() -> None:
    contract, digest = load_contract()
    if contract["contract_id"] != "history_value_adam_validation_v005":
        raise v1.ContractStop("wrong v005 contract id")
    if tuple(contract["model_matrix"]["required_prediction_streams"]) != v4.STREAMS:
        raise v1.ContractStop("v005 prediction streams differ")
    if int(contract["bootstrap"]["replicates"]) != 2000:
        raise v1.ContractStop("v005 bootstrap count differs")
    v4.read_adam_evidence()
    v1.implementation_records(contract)
    v1.environment_manifest(contract, require_cuda=False)
    v1.progress(f"V005_VALIDATE_ONLY_OK contract_sha256={digest}; governed data not opened")


def run_validation(approved_hash: str) -> None:
    contract, digest = load_contract()
    if approved_hash != digest:
        raise v1.ContractStop(f"approved v005 contract hash mismatch: observed {digest}")
    verify_approval_receipt(digest)
    v1.implementation_records(contract)
    environment = v1.environment_manifest(contract, require_cuda=True)
    device = torch.device("cuda:0")
    if OUTPUT_ROOT.exists():
        raise v1.ContractStop("v005 output root already exists; overwrite is forbidden")
    OUTPUT_ROOT.mkdir(parents=True)
    v1.append_access("preflight", "opened", digest)
    selected = v4.preflight(contract, device)
    v1.append_access("preflight", "complete_hashed", digest)

    v1.gpu.assert_stage_access("validation", {"preflight"})
    v1.append_access("validation", "opened_after_preflight", digest)
    stage_dir = OUTPUT_ROOT / "validation"
    stage_dir.mkdir()
    feature_path, feature_info = v1.materialize_stage_features(contract, "validation", stage_dir)
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
    REPORT_PATH.write_text(render_validation_report(result, digest, selected), encoding="utf-8")
    v1.finalize_hashes(OUTPUT_ROOT)
    v1.progress(f"v005 Validation complete; sealed/random locked; report={REPORT_PATH}")


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
