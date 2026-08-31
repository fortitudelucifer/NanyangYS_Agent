#!/usr/bin/env python3
"""Run the final random-exposure audit under reconstructed frozen sealed Adam models."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
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
import run_history_value_adam_sealed_v008 as v8


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_adam_random_contract_v009.yaml"
BASE_CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_adam_sealed_contract_v008.yaml"
APPROVAL_PATH = PROJECT_ROOT / "configs/history_value_adam_random_approval_v009.json"
OUTPUT_ROOT = PROJECT_ROOT / "reports/generated/history_value_adam_random_v009"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/history_value_adam_random_results_v009.md"
SEALED_CHECKPOINT = PROJECT_ROOT / "reports/generated/history_value_adam_sealed_v008/sealed_checkpoint.json"
SEALED_DECISION = PROJECT_ROOT / "reports/generated/history_value_adam_sealed_v008/sealed_test/stage_decision.json"
SEALED_MANIFEST = PROJECT_ROOT / "reports/generated/history_value_adam_sealed_v008/sealed_test/artifact_hash_manifest.json"
SEALED_PREDICTIONS = PROJECT_ROOT / "reports/generated/history_value_adam_sealed_v008/sealed_test/predictions.parquet"
SEALED_FEATURES = PROJECT_ROOT / "reports/generated/history_value_adam_sealed_v008/sealed_test/features.parquet"
SEALED_FEATURE_MANIFEST = PROJECT_ROOT / "reports/generated/history_value_adam_sealed_v008/sealed_test/feature_manifest.json"
COUNT_EVIDENCE_PATH = PROJECT_ROOT / "reports/analysis/random_target_count_precheck_2026-08-23.csv"
EXPECTED_BASE_SHA256 = "b7b54a98ec02f0926a59fc3055097b82bdc9f7c5e5b5e7f1546beb3cdf8ef1c7"
EXPECTED_RANDOM_ROWS = 43027
AUTHORIZED_STAGES = ["preflight", "sealed_model_reconstruction", "random_audit"]


def load_contract() -> tuple[dict[str, Any], str]:
    observed = v1.sha256_file(BASE_CONTRACT_PATH)
    if observed != EXPECTED_BASE_SHA256:
        raise v1.ContractStop("v009 base v008 contract hash mismatch")
    base, base_digest = v8.load_contract()
    if base_digest != observed:
        raise v1.ContractStop("v009 base merged digest mismatch")
    overlay = yaml.load(CONTRACT_PATH.read_text(encoding="utf-8"), Loader=v1.UniqueKeyLoader)
    if overlay["base_contract"]["sha256"] != observed:
        raise v1.ContractStop("v009 overlay does not pin v008")
    merged = v2.deep_merge(base, overlay)
    if int(merged["sequential_stage_protocol"]["stage_3_random_audit"]["expected_target_rows"]) != EXPECTED_RANDOM_ROWS:
        raise v1.ContractStop("v009 random target count differs")
    if merged["authorization"]["authorized_stage_scope_after_exact_hash_approval"] != AUTHORIZED_STAGES:
        raise v1.ContractStop("v009 authorized stage scope differs")
    return merged, v1.sha256_file(CONTRACT_PATH)


def _verify_manifest(path: Path) -> int:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_absolute():
            artifact_path = PROJECT_ROOT / artifact_path
        if not artifact_path.is_file() or artifact_path.stat().st_size != int(artifact["size_bytes"]):
            raise v1.ContractStop(f"v009 missing predecessor artifact: {artifact_path}")
        if v1.sha256_file(artifact_path) != artifact["sha256"]:
            raise v1.ContractStop(f"v009 predecessor artifact SHA mismatch: {artifact_path}")
    return len(manifest["artifacts"])


def verify_sealed_prerequisite(contract: dict[str, Any]) -> dict[str, Any]:
    prereq = contract["sealed_prerequisite"]
    exact = {
        SEALED_CHECKPOINT: prereq["checkpoint_sha256"],
        SEALED_DECISION: prereq["stage_decision_sha256"],
        SEALED_MANIFEST: prereq["stage_artifact_manifest_sha256"],
        SEALED_PREDICTIONS: prereq["sealed_predictions_sha256"],
    }
    for path, digest in exact.items():
        if v1.sha256_file(path) != digest:
            raise v1.ContractStop(f"v009 sealed prerequisite SHA mismatch: {path.name}")
    checkpoint = json.loads(SEALED_CHECKPOINT.read_text(encoding="utf-8"))
    decision = json.loads(SEALED_DECISION.read_text(encoding="utf-8"))
    if checkpoint["contract_sha256"] != EXPECTED_BASE_SHA256:
        raise v1.ContractStop("v009 sealed checkpoint contract mismatch")
    if checkpoint["status"] != "sealed_complete_random_audit_locked":
        raise v1.ContractStop("v009 sealed checkpoint status mismatch")
    if decision["scientific_status"] != prereq["required_scientific_status"]:
        raise v1.ContractStop("v009 requires the frozen sealed pass")
    if checkpoint["random_audit_accessed"]:
        raise v1.ContractStop("v009 predecessor already accessed random audit")
    count = _verify_manifest(SEALED_MANIFEST)
    if count != 18:
        raise v1.ContractStop("v009 expected 18 sealed stage artifacts")
    reconstruction = contract["frozen_model_reconstruction"]
    if SEALED_FEATURES.stat().st_size != int(reconstruction["sealed_feature_size_bytes"]):
        raise v1.ContractStop("v009 sealed feature size mismatch")
    if v1.sha256_file(SEALED_FEATURES) != reconstruction["sealed_feature_sha256"]:
        raise v1.ContractStop("v009 sealed feature SHA mismatch")
    if v1.sha256_file(SEALED_FEATURE_MANIFEST) != reconstruction["sealed_feature_manifest_sha256"]:
        raise v1.ContractStop("v009 sealed feature manifest SHA mismatch")
    return checkpoint


def verify_count_evidence(contract: dict[str, Any]) -> None:
    evidence = contract["random_target_precheck"]
    if v1.sha256_file(COUNT_EVIDENCE_PATH) != evidence["evidence_sha256"]:
        raise v1.ContractStop("v009 random count evidence SHA mismatch")
    if int(evidence["silver_main_rows"]) + int(evidence["official_formula_mismatch_addback_rows"]) != EXPECTED_RANDOM_ROWS:
        raise v1.ContractStop("v009 random target count arithmetic mismatch")
    if int(evidence["standard_history_unseen_user_rows"]) != 0:
        raise v1.ContractStop("v009 random primary user coverage changed")


def verify_approval_receipt(contract_sha: str) -> dict[str, Any]:
    if not APPROVAL_PATH.is_file():
        raise v1.ContractStop("v009 exact-hash approval receipt is missing")
    receipt = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    required = {
        "contract_id": "history_value_adam_random_v009",
        "contract_sha256": contract_sha,
        "execution_authorized": True,
        "authorized_stages": AUTHORIZED_STAGES,
        "automatic_ordered_transitions_authorized": False,
        "approved_by": "project_owner",
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise v1.ContractStop(f"v009 approval receipt mismatch: {key}")
    return receipt


def _save_model_state(path: Path, frozen: v1.FrozenModels) -> None:
    payload: dict[str, np.ndarray] = {
        "bl0_probability": np.asarray([frozen.bl0], dtype=np.float64),
        "design_prevalence": np.asarray([frozen.design.prevalence], dtype=np.float64),
        "static_scaler_mean": np.asarray(frozen.design.static_scaler.mean_, dtype=np.float64),
        "static_scaler_scale": np.asarray(frozen.design.static_scaler.scale_, dtype=np.float64),
        "h2_scaler_mean": np.asarray(frozen.design.h2_scaler.mean_, dtype=np.float64),
        "h2_scaler_scale": np.asarray(frozen.design.h2_scaler.scale_, dtype=np.float64),
    }
    for index, categories in enumerate(frozen.design.encoder.categories_):
        payload[f"encoder_categories_{index}"] = np.asarray(categories)
    for stream, fit in frozen.fits.items():
        payload[f"{stream}_coefficient"] = np.asarray(fit.coefficient)
        payload[f"{stream}_intercept"] = np.asarray([fit.intercept], dtype=np.float64)
        calibrator = frozen.calibrators[stream]
        payload[f"{stream}_calibrator_intercept"] = np.asarray([calibrator.intercept], dtype=np.float64)
        payload[f"{stream}_calibrator_slope"] = np.asarray([calibrator.slope], dtype=np.float64)
    np.savez_compressed(path, **payload)


def reconstruct_frozen_models(
    contract: dict[str, Any], selected: dict[str, Any], device: torch.device
) -> tuple[v1.FrozenModels, list[dict[str, Any]], list[dict[str, Any]]]:
    out = OUTPUT_ROOT / "sealed_model_reconstruction"
    out.mkdir()
    sealed_frame = v7.read_frame_with_time(SEALED_FEATURES)
    target_index, predictions, frozen, adequacy, calibration = v4.fit_standard_stage(
        sealed_frame, stage="sealed_model_reconstruction",
        fit_range=("2022-04-08", "2022-04-20"), calibration_date="2022-04-21",
        target_range=("2022-04-22", "2022-05-08"), selected=selected, device=device,
    )
    required_columns = [
        "source_table", "source_row_number", "p_BL0", "p_ADAM_BL1", "p_ADAM_BL2",
        "raw_ADAM_BL1", "raw_ADAM_BL2",
    ]
    prior = v1.pq.read_table(SEALED_PREDICTIONS, columns=required_columns)
    source = prior.column("source_table").combine_chunks().to_numpy(zero_copy_only=False)
    row_number = prior.column("source_row_number").combine_chunks().to_numpy(zero_copy_only=False)
    identity_exact = bool(
        np.array_equal(source, sealed_frame.columns["source_table"][target_index])
        and np.array_equal(row_number, sealed_frame.columns["source_row_number"][target_index])
    )
    if not identity_exact:
        raise v1.ContractStop("v009 sealed reconstruction target identities differ")
    checks: list[dict[str, Any]] = []
    limits = contract["frozen_model_reconstruction"]["required_reproduction_checks"]
    mapping = {
        "BL0": ("p_BL0", "BL0", float(limits["BL0_probability_max_abs_difference"])),
        "ADAM_BL1_probability": ("p_ADAM_BL1", "ADAM_BL1", float(limits["calibrated_probability_max_abs_difference"])),
        "ADAM_BL2_probability": ("p_ADAM_BL2", "ADAM_BL2", float(limits["calibrated_probability_max_abs_difference"])),
        "ADAM_BL1_raw": ("raw_ADAM_BL1", "raw_ADAM_BL1", float(limits["raw_score_max_abs_difference"])),
        "ADAM_BL2_raw": ("raw_ADAM_BL2", "raw_ADAM_BL2", float(limits["raw_score_max_abs_difference"])),
    }
    for label, (prior_column, current_key, allowed) in mapping.items():
        old = prior.column(prior_column).combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
        new = np.asarray(predictions[current_key], dtype=np.float64)
        maximum = float(np.max(np.abs(old - new)))
        passed = bool(np.isfinite(maximum) and maximum <= allowed)
        checks.append({
            "quantity": label, "rows": old.size, "maximum_absolute_difference": maximum,
            "maximum_allowed_difference": allowed, "passed": passed,
        })
        if not passed:
            raise v1.ContractStop(f"v009 sealed model reconstruction mismatch: {label}")
    gpu_rows = [row for row in adequacy if row.get("stream")]
    if len(gpu_rows) != 2 or not all(bool(row["adequacy_passed"]) for row in gpu_rows):
        raise v1.ContractStop("v009 reconstructed Adam adequacy failed")
    state_path = out / "reconstructed_frozen_model_state.npz"
    _save_model_state(state_path, frozen)
    audit = {
        "status": "pass_random_access_unlocked",
        "sealed_contract_sha256": EXPECTED_BASE_SHA256,
        "sealed_predictions_sha256": contract["sealed_prerequisite"]["sealed_predictions_sha256"],
        "sealed_target_identity_exact": identity_exact,
        "comparisons": checks,
        "model_state_path": str(state_path.relative_to(PROJECT_ROOT)),
        "model_state_sha256": v1.sha256_file(state_path),
        "random_data_opened_before_this_audit": False,
    }
    v1.write_json(out / "reconstruction_audit.json", audit)
    v1.write_csv(out / "optimization_adequacy.csv", adequacy)
    v1.write_csv(out / "calibration_audit.csv", calibration)
    v1.finalize_hashes(out)
    del sealed_frame, predictions, prior, source, row_number
    gc.collect()
    return frozen, adequacy, calibration


def materialize_random_features(
    contract: dict[str, Any], stage_dir: Path
) -> tuple[Path, dict[str, Any]]:
    path = stage_dir / "features.parquet"
    temp = stage_dir / ".features.building.parquet"
    if path.exists() or temp.exists():
        raise v1.ContractStop("v009 refuses to overwrite a random feature artifact")
    keys = [
        "events_early_standard", "events_late_standard", "events_random",
        "videos_basic", "label_formula_mismatch_rows",
    ]
    verified = [v1.verify_input(contract, key) for key in keys]
    con = v1.duckdb.connect()
    v1.configure_duckdb(con, stage_dir / "duckdb_tmp")
    try:
        validation = v1.materialize_random_features(
            con,
            early_path=v1.resolve_input(contract, "events_early_standard"),
            late_path=v1.resolve_input(contract, "events_late_standard"),
            random_path=v1.resolve_input(contract, "events_random"),
            mismatch_path=v1.resolve_input(contract, "label_formula_mismatch_rows"),
            videos_path=v1.resolve_input(contract, "videos_basic"),
            output_path=temp, expected_target_rows=EXPECTED_RANDOM_ROWS,
        )
    finally:
        con.close()
    temp.replace(path)
    manifest = v1.feature_manifest(path, validation, verified)
    v1.write_json(stage_dir / "feature_manifest.json", manifest)
    return path, manifest


def render_random_report(
    result: dict[str, Any], contract_sha: str, selected: dict[str, Any]
) -> str:
    pooled = {(row["scope"], row["model_id"]): row for row in result["pooled"]}
    all_rows = pooled[("all_target_rows", "BL0")]
    warm = pooled[("primary_warm_user", "BL0")]
    decision = result["decision"]
    lines = [
        "# 历史特征价值 GPU Adam：random audit 最终结果 v009", "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        f"- 合同 SHA-256：`{contract_sha}`",
        f"- Adam：lr={selected['ADAM']['learning_rate']}，steps={selected['ADAM']['steps']}。",
        "- 前置证据：Validation 与 sealed test 均完整通过。",
        "- 模型来源：确定性重建 sealed fit/calibrator，并在 443 万 sealed 行上通过预测复现门后冻结。",
        "- SGD：冻结，本结果不声称跨优化器稳健。", "",
        "## 数据与迁移问题", "",
        "目标为 2022-04-22 至 05-08 的全部正式 random exposures；历史只来自标准曝光，"
        "random 标签从不更新历史，也不参与拟合、校准或阈值选择。", "",
        "random audit 只回答标准曝光上已复现的历史增量能否迁移到不同曝光分布。", "",
        "## 样本", "",
        "| 范围 | 行数 | 用户数 | 正例数 | 正例率 |", "|---|---:|---:|---:|---:|",
        f"| 全部 random 目标 | {all_rows['rows']:,} | {all_rows['users']:,} | {all_rows['positives']:,} | {all_rows['prevalence']:.6f} |",
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
        row = v1.metric_lookup(result, "random_audit", "ADAM_BL2_minus_ADAM_BL1", metric)
        lines.append(f"| {label} | {row['point_estimate']:.6f} | [{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}] |")
    lines.extend([
        "", "## 预注册迁移门与结论", "",
        f"- 历史迁移门：{decision['history_gate']['ADAM']}。",
        f"- 概率健康门：{decision['probability_saturation_gate']}。",
        f"- random audit 完整阶段判定：`{decision['scientific_status']}`。",
        "- random 的静态/绝对概率门只作描述，不参与迁移判定，因为曝光分布与正例率发生变化。", "",
    ])
    if decision["scientific_status"] == "pass":
        lines.append("最终结论：冻结 Adam 下的历史增量同时得到 Validation、sealed standard 和 random exposure 支持。")
    else:
        lines.append("最终结论：历史增量在 Validation 与 sealed 标准曝光上成立，但未通过 random 迁移门；结论限制在标准曝光。")
    lines.extend([
        "", "所有证据均为离线预测证据，不证明因果或线上业务提升；SGD 仍冻结。", "",
        "## 可作图文件", "",
        "random_audit 目录提供逐行 predictions.parquet，以及 pooled、daily、contrast、2,000 次 bootstrap、"
        "历史深度切片和概率分布 CSV。", "",
    ])
    return "\n".join(lines)


def install_overrides() -> None:
    v8.install_overrides()
    v1.CONTRACT_PATH = CONTRACT_PATH
    v1.APPROVAL_PATH = APPROVAL_PATH
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.REPORT_PATH = REPORT_PATH
    v1.load_contract = load_contract
    v1.verify_approval_receipt = verify_approval_receipt


def validate_only() -> None:
    contract, digest = load_contract()
    if contract["contract_id"] != "history_value_adam_random_v009":
        raise v1.ContractStop("wrong v009 contract id")
    verify_sealed_prerequisite(contract)
    verify_count_evidence(contract)
    v4.read_adam_evidence()
    v1.implementation_records(contract)
    v1.environment_manifest(contract, require_cuda=False)
    v1.progress(f"V009_VALIDATE_ONLY_OK contract_sha256={digest}; random model score data not opened")


def run_random(approved_hash: str) -> None:
    contract, digest = load_contract()
    if approved_hash != digest:
        raise v1.ContractStop(f"approved v009 contract hash mismatch: observed {digest}")
    verify_approval_receipt(digest)
    verify_sealed_prerequisite(contract)
    verify_count_evidence(contract)
    v1.implementation_records(contract)
    environment = v1.environment_manifest(contract, require_cuda=True)
    device = torch.device("cuda:0")
    if OUTPUT_ROOT.exists():
        raise v1.ContractStop("v009 output root already exists; overwrite is forbidden")
    OUTPUT_ROOT.mkdir(parents=True)
    v1.append_access("sealed_prerequisite", "exact_v008_pass_and_18_artifact_hashes_verified", digest)
    v1.append_access("preflight", "opened", digest)
    selected = v4.preflight(contract, device)
    v1.append_access("preflight", "complete_hashed", digest)

    v1.append_access("sealed_model_reconstruction", "opened_before_any_random_data_access", digest)
    frozen, adequacy, calibration = reconstruct_frozen_models(contract, selected, device)
    v1.append_access("sealed_model_reconstruction", "complete_hashed_prediction_reproduction_pass", digest)

    v1.gpu.assert_stage_access("random_audit", {"preflight", "validation", "sealed_test"})
    v1.append_access("random_audit", "opened_after_reconstruction_pass", digest)
    stage_dir = OUTPUT_ROOT / "random_audit"
    stage_dir.mkdir()
    feature_path, feature_info = materialize_random_features(contract, stage_dir)
    frame = v7.read_frame_with_time(feature_path)
    target_index, predictions = v1.score_random_stage(frame, frozen, device=device)
    copied_adequacy = [
        {**row, "stage": "random_audit", "source": "sealed_model_reconstruction"}
        for row in adequacy
    ]
    copied_calibration = [
        {**row, "stage": "random_audit", "source": "sealed_model_reconstruction"}
        for row in calibration
    ]
    result = v1.evaluate_and_freeze(
        frame, target_index, predictions, stage="random_audit", stage_dir=stage_dir,
        contract_sha=digest, feature_info=feature_info, adequacy_rows=copied_adequacy,
        calibration_rows=copied_calibration, environment=environment,
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
    }
    v1.write_json(OUTPUT_ROOT / "final_claim_decision.json", {
        "contract_sha256": digest, "random_decision": result["decision"],
        "terminal_interpretation": terminal,
    })
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_random_report(result, digest, selected), encoding="utf-8")
    v1.finalize_hashes(OUTPUT_ROOT)
    v1.progress(f"v009 random audit complete; report={REPORT_PATH}")


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
