#!/usr/bin/env python3
"""Run the Adam-only history-value confirmation after freezing SGD."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
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
CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_adam_confirmation_contract_v004.yaml"
BASE_CONTRACT_PATH = PROJECT_ROOT / "configs/history_value_gpu_confirmation_contract_v002.yaml"
APPROVAL_PATH = PROJECT_ROOT / "configs/history_value_adam_confirmation_approval_v004.json"
OUTPUT_ROOT = PROJECT_ROOT / "reports/generated/history_value_adam_confirmation_v004"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/history_value_adam_confirmation_results_v004.md"
V002_ADEQUACY_PATH = PROJECT_ROOT / "reports/generated/history_value_gpu_confirmation_v002/preflight/optimization_adequacy.csv"
EXPECTED_BASE_SHA256 = "a11e75f43b2fe013e9ea0ee813953b46c224f8a5786e64a1c3bd25a43b37d09b"
EXPECTED_ADEQUACY_SHA256 = "79867555a31a96f5ffe1be38930de69ad491d70fea5aed56008d7997d8cbcf16"
STREAMS = ("BL0", "ADAM_BL1", "ADAM_BL2")
LEARNED = STREAMS[1:]
CONTRASTS = {
    "ADAM_BL1_minus_BL0": {"ADAM_BL1": 1.0, "BL0": -1.0},
    "ADAM_BL2_minus_ADAM_BL1": {"ADAM_BL2": 1.0, "ADAM_BL1": -1.0},
}


def load_contract() -> tuple[dict[str, Any], str]:
    observed = v1.sha256_file(BASE_CONTRACT_PATH)
    if observed != EXPECTED_BASE_SHA256:
        raise v1.ContractStop("v004 base v002 contract hash mismatch")
    base, base_digest = v2.load_contract()
    if base_digest != observed:
        raise v1.ContractStop("v004 base merged digest mismatch")
    overlay = yaml.load(CONTRACT_PATH.read_text(encoding="utf-8"), Loader=v1.UniqueKeyLoader)
    if overlay["base_contract"]["sha256"] != observed:
        raise v1.ContractStop("v004 overlay does not pin v002")
    merged = v2.deep_merge(base, overlay)
    if tuple(merged["model_matrix"]["required_prediction_streams"]) != STREAMS:
        raise v1.ContractStop("v004 prediction streams differ")
    return merged, v1.sha256_file(CONTRACT_PATH)


def verify_approval_receipt(contract_sha: str) -> dict[str, Any]:
    if not APPROVAL_PATH.is_file():
        raise v1.ContractStop("v004 exact-hash approval receipt is missing")
    receipt = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    required = {
        "contract_id": "history_value_adam_confirmation_v004",
        "contract_sha256": contract_sha,
        "execution_authorized": True,
        "automatic_ordered_transitions_authorized": True,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise v1.ContractStop(f"v004 approval receipt mismatch: {key}")
    if receipt.get("approved_by") != "project_owner":
        raise v1.ContractStop("v004 approval receipt must identify project_owner")
    return receipt


def read_adam_evidence() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if V002_ADEQUACY_PATH.stat().st_size != 23470:
        raise v1.ContractStop("v002 adequacy evidence size mismatch")
    if v1.sha256_file(V002_ADEQUACY_PATH) != EXPECTED_ADEQUACY_SHA256:
        raise v1.ContractStop("v002 adequacy evidence SHA mismatch")
    with V002_ADEQUACY_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if row["optimizer"] == "ADAM" and int(row["steps"]) == 100
    ]
    if len(selected) != 6 or not all(row["adequacy_passed"] == "True" for row in selected):
        raise v1.ContractStop("Adam 100-step evidence is not 6/6 adequate")
    return selected, {
        "source_path": str(V002_ADEQUACY_PATH.relative_to(PROJECT_ROOT)),
        "source_sha256": EXPECTED_ADEQUACY_SHA256,
        "learning_rate": 0.03,
        "steps": 100,
        "passed_combinations": 6,
        "maximum_regret": max(float(row["objective_regret"]) for row in selected),
        "SGD_status": "frozen_not_a_v004_prediction_stream",
    }


def preflight(contract: dict[str, Any], device: torch.device) -> dict[str, Any]:
    del contract, device
    out = v1.OUTPUT_ROOT / "preflight"
    if out.exists():
        raise v1.ContractStop("v004 inherited preflight output already exists")
    out.mkdir(parents=True)
    rows, evidence = read_adam_evidence()
    selected = {
        "status": "complete_hash_verified_inherited_Adam_frozen_before_validation",
        "ADAM": {
            "learning_rate": 0.03,
            "steps": 100,
            "maximum_regret": evidence["maximum_regret"],
            "evidence": "hash_verified_v002_6_of_6_Train_only_objective_checks",
        },
        "SGD": {"status": "frozen_deferred_no_target_predictions"},
    }
    v1.write_json(out / "inherited_Adam_adequacy_evidence.json", {
        "manifest": evidence, "rows": rows,
    })
    v1.write_json(v1.OUTPUT_ROOT / "selected_optimizer_configuration_manifest.json", selected)
    v1.finalize_hashes(out)
    return selected


def fit_standard_stage(
    frame: v1.canonical.Frame,
    *,
    stage: str,
    fit_range: tuple[str, str],
    calibration_date: str,
    target_range: tuple[str, str],
    selected: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray], v1.FrozenModels, list[dict[str, Any]], list[dict[str, Any]]]:
    dates, labels = frame.dates(), frame.labels()
    fit_lo, fit_hi = (np.datetime64(value, "D") for value in fit_range)
    cal_date = np.datetime64(calibration_date, "D")
    target_lo, target_hi = (np.datetime64(value, "D") for value in target_range)
    fit_index = np.flatnonzero((dates >= fit_lo) & (dates <= fit_hi))
    cal_index = np.flatnonzero(dates == cal_date)
    target_index = np.flatnonzero((dates >= target_lo) & (dates <= target_hi))
    if np.unique(labels[cal_index]).size != 2:
        raise v1.ContractStop(f"{stage} calibration lacks both classes")
    split = v1.canonical.OriginSplit(
        origin=stage, fit_index=fit_index, calibration_index=cal_index,
        assessment_index=target_index, calibration_date=calibration_date,
        bl0_probability=float(labels[dates < target_lo].mean()),
        fit_prevalence=float(labels[fit_index].mean()),
    )
    origin = v1.canonical.build_origin_matrices(frame, split)
    references, reference_rows = v2.reference_objectives(
        origin.matrices["fit"], origin.labels["fit"], origin=stage
    )
    predictions: dict[str, np.ndarray] = {
        "BL0": np.full(target_index.size, split.bl0_probability, dtype=np.float64)
    }
    fits: dict[str, gpu.GpuFit] = {}
    calibrators: dict[str, repair.Calibrator] = {}
    adequacy_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    config = selected["ADAM"]
    for model_id in ("BL1", "BL2"):
        stream = f"ADAM_{model_id}"
        v1.progress(f"{stage}: fit {stream} on GPU")
        fit = gpu.fit_trajectory(
            origin.matrices["fit"][model_id], origin.labels["fit"], device=device,
            optimizer_name="ADAM", learning_rate=float(config["learning_rate"]),
            checkpoints=[int(config["steps"])], alpha=1e-4,
        )
        decision = gpu.adequacy(fit.objective, references[model_id], reference_converged=True)
        adequacy_rows.append({
            "stage": stage, "stream": stream, "model_id": model_id,
            "optimizer": "ADAM", "learning_rate": fit.learning_rate,
            "steps": fit.steps, "terminal_gradient_norm": fit.terminal_gradient_norm,
            "elapsed_seconds": fit.elapsed_seconds,
            "peak_cuda_memory_bytes": fit.peak_cuda_memory_bytes, **decision,
        })
        if not decision["adequacy_passed"]:
            raise v1.ContractStop(f"{stage} Adam adequacy failed: {stream}")
        raw_cal, _ = gpu.score(
            origin.matrices["calibration"][model_id], fit.coefficient, fit.intercept,
            device=device,
        )
        raw_target, _ = gpu.score(
            origin.matrices["assessment"][model_id], fit.coefficient, fit.intercept,
            device=device,
        )
        calibrator = repair.fit_previous_day_sigmoid(
            raw_cal, origin.labels["calibration"], user_id=origin.users["calibration"]
        )
        probability = calibrator.apply(raw_target)
        repair.assert_calibration_monotone(raw_target, probability)
        predictions[stream] = probability
        predictions[f"raw_{stream}"] = raw_target
        fits[stream] = fit
        calibrators[stream] = calibrator
        calibration_rows.append({
            "stage": stage, "stream": stream, "fit_rows": calibrator.fit_rows,
            "fit_users": calibrator.fit_users, "fit_positives": calibrator.fit_positives,
            "intercept": calibrator.intercept, "slope": calibrator.slope,
            "n_iter": calibrator.n_iter,
            "convergence_warning_count": calibrator.convergence_warning_count,
        })
    reference_rows = [{"stage": stage, **row} for row in reference_rows]
    frozen = v1.FrozenModels(origin.design, fits, calibrators, split.bl0_probability)
    return target_index, predictions, frozen, adequacy_rows + reference_rows, calibration_rows


def stage_decision(
    stage: str,
    pooled: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
    daily: list[dict[str, Any]],
    probability_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    point = {(row["scope"], row["model_id"]): row for row in pooled}
    boot = {(row["contrast"], row["metric"]): row for row in bootstrap}
    warm = "primary_warm_user"
    static_contrast = "ADAM_BL1_minus_BL0"
    history_contrast = "ADAM_BL2_minus_ADAM_BL1"
    static_pass = bool(
        boot[(static_contrast, "average_precision")]["ci95_lower"] > 0
        and boot[(static_contrast, "user_gauc_event_weighted")]["point_estimate"] >= 0
        and boot[(static_contrast, "log_loss")]["point_estimate"] <= 0
        and boot[(static_contrast, "brier")]["point_estimate"] <= 0
    )
    history_pass = bool(
        boot[(history_contrast, "average_precision")]["point_estimate"] >= .005
        and boot[(history_contrast, "average_precision")]["ci95_lower"] > 0
        and boot[(history_contrast, "user_gauc_event_weighted")]["point_estimate"] >= 0
        and boot[(history_contrast, "log_loss")]["point_estimate"] <= 0
        and boot[(history_contrast, "brier")]["point_estimate"] <= 0
    )
    absolute = {
        stream: bool(
            point[(warm, stream)]["log_loss"] <= point[(warm, "BL0")]["log_loss"] + 1e-10
            and point[(warm, stream)]["brier"] <= point[(warm, "BL0")]["brier"] + 1e-10
        )
        for stream in LEARNED
    }
    by_date: dict[str, dict[str, float]] = {}
    for row in daily:
        if row["population"] == warm and row["model_id"] in LEARNED:
            by_date.setdefault(row["event_date"], {})[row["model_id"]] = row["average_precision"]
    positive_days = sum(
        values["ADAM_BL2"] > values["ADAM_BL1"]
        for values in by_date.values() if len(values) == 2
    )
    saturation = all(
        row["below_or_equal_1e_6_share"] <= .05
        and row["above_or_equal_1_minus_1e_6_share"] <= .05
        for row in probability_rows
    )
    required_days = 0 if stage == "random_audit" else (3 if stage == "validation" else 12)
    if stage == "random_audit":
        passed = saturation and history_pass
    else:
        passed = saturation and static_pass and all(absolute.values()) and history_pass and positive_days >= required_days
    return {
        "stage": stage, "scientific_status": "pass" if passed else "fail_or_mixed",
        "scientific_failure_blocks_next_stage": False,
        "static_baseline_gate": {"ADAM": static_pass},
        "absolute_probability_gate": absolute,
        "history_gate": {"ADAM": history_pass},
        "positive_AP_days": {"ADAM": positive_days},
        "required_positive_AP_days": required_days,
        "probability_saturation_gate": saturation,
        "optimizer_robustness_established": False,
        "SGD_status": "frozen_deferred",
    }


def terminal_interpretation(stage_decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    passed = {
        stage: decision["scientific_status"] == "pass"
        for stage, decision in stage_decisions.items()
    }
    if passed["validation"] and passed["sealed_test"] and passed["random_audit"]:
        status = "supported_for_standard_and_random_under_frozen_Adam"
        conclusion = (
            "在冻结的 GPU Adam 离线协议下，历史增量在 Validation、sealed test "
            "和 random audit 三阶段均得到支持。"
        )
    elif not passed["validation"]:
        status = "falsified_at_validation"
        conclusion = (
            "Validation 未通过完整预注册门，广义历史增量在首个确认阶段被证伪；"
            "后续结果用于刻画异质性，不能抹去该失败。"
        )
    elif not passed["sealed_test"]:
        status = "not_sustained_on_sealed_test"
        conclusion = (
            "Validation 通过但 sealed test 未通过，历史增量没有跨更长时间窗保持稳定。"
        )
    else:
        status = "supported_on_standard_not_transportable_to_random"
        conclusion = (
            "Validation 与 sealed test 通过，但 random audit 未通过；"
            "结论限制在标准曝光，不支持向随机曝光迁移。"
        )
    return {
        "status": status,
        "conclusion": conclusion,
        "stage_pass": passed,
        "optimizer_scope": "GPU_Adam_only",
        "optimizer_robustness_established": False,
        "SGD_status": "frozen_deferred",
    }


def render_report(
    results: dict[str, dict[str, Any]], contract_sha: str, selected: dict[str, Any]
) -> str:
    lines = [
        "# 历史特征价值 GPU Adam 主实验 v004", "",
        f"- 生成时间：{datetime.now().astimezone().isoformat()}",
        f"- 合同 SHA-256：`{contract_sha}`",
        f"- Adam：lr={selected['ADAM']['learning_rate']}，steps={selected['ADAM']['steps']}。",
        "- SGD：已冻结，未生成任何后置预测；本报告不声称跨优化器稳健。", "",
        "## 数据来源与研究背景", "",
        "数据来自 KuaiRand 正式 Silver 快照：early standard、late standard、random exposures、"
        "仅回补官方 `long_view` 的公式不一致隔离行，以及 `videos_basic` 静态内容元数据。"
        "标准目标为 `tab=1`；random 目标为全部正式随机曝光。历史严格满足 "
        "`history_time_ms < target_time_ms`，且 random 标签从不回写历史。", "",
        "Validation 目标窗为 2022-04-18 至 2022-04-21；sealed test 为 2022-04-22 至 "
        "2022-05-08；random audit 使用正式随机曝光。restricted 与 random 窗口此前看过聚合"
        "标签/质量摘要，因此不是 pristine sealed data；此前未读取候选模型预测或 BL2-BL1 对比。", "",
        "唯一科学问题是：在完全相同的静态 BL1 上加入严格时点 H2 用户历史后，"
        "ADAM_BL2 是否相对 ADAM_BL1 改善 `long_view` 离线预测。BL0 是冻结常数概率参考，"
        "用于确认静态基线本身有效。", "",
        "## 冻结设计与统计规则", "",
        "- 三条逐行对齐预测流：BL0、ADAM_BL1、ADAM_BL2。",
        "- Adam 固定为 `lr=0.03, steps=100`；其 Train-only objective 充分性为 6/6 通过。",
        "- 主效应为同一目标行上的 ADAM_BL2 − ADAM_BL1；AP 与 gAUC 正值更好，log-loss 与 Brier 负值更好。",
        "- AP 最小效应阈值为 0.005，且其 95% CI 下界必须大于 0；gAUC 不得下降，log-loss 与 Brier 不得变差。",
        "- 区间来自 2,000 次配对用户簇 bootstrap；Validation/封存还要求分别至少 3/4、12/17 天 AP 方向为正。",
        "- 科学门失败仍继续后续阶段并保留结果；求解器、数据或概率健康无效则停止。", "",
        "## 样本概况", "",
        "| 阶段 | 全部目标行 | 全部用户 | 正例率 | 主分析 warm-user 行 | 主分析用户 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for stage in ("validation", "sealed_test", "random_audit"):
        pooled = {
            (row["scope"], row["model_id"]): row
            for row in results[stage]["pooled"]
        }
        all_rows = pooled[("all_target_rows", "BL0")]
        warm = pooled[("primary_warm_user", "BL0")]
        lines.append(
            f"| {stage} | {all_rows['rows']:,} | {all_rows['users']:,} | "
            f"{all_rows['prevalence']:.4f} | {warm['rows']:,} | {warm['users']:,} |"
        )
    lines.extend([
        "", "## 主要历史增量结果", "",
        "| 阶段 | ΔAP (95% CI) | Δevent-gAUC (95% CI) | Δlog-loss (95% CI) | ΔBrier (95% CI) | history gate |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for stage in ("validation", "sealed_test", "random_audit"):
        result = results[stage]
        contrast = "ADAM_BL2_minus_ADAM_BL1"
        ap = v1.metric_lookup(result, stage, contrast, "average_precision")
        gauc = v1.metric_lookup(result, stage, contrast, "user_gauc_event_weighted")
        logloss = v1.metric_lookup(result, stage, contrast, "log_loss")
        brier = v1.metric_lookup(result, stage, contrast, "brier")
        decision = result["decision"]
        lines.append(
            f"| {stage} | {ap['point_estimate']:.6f} [{ap['ci95_lower']:.6f}, {ap['ci95_upper']:.6f}] | "
            f"{gauc['point_estimate']:.6f} [{gauc['ci95_lower']:.6f}, {gauc['ci95_upper']:.6f}] | "
            f"{logloss['point_estimate']:.6f} [{logloss['ci95_lower']:.6f}, {logloss['ci95_upper']:.6f}] | "
            f"{brier['point_estimate']:.6f} [{brier['ci95_lower']:.6f}, {brier['ci95_upper']:.6f}] | "
            f"{decision['history_gate']['ADAM']} |"
        )
    lines.extend([
        "", "## 控制门与阶段判定", "",
        "| 阶段 | 静态 BL1 vs BL0 | BL1/BL2 绝对概率门 | 正向 AP 天数/要求 | 概率健康 | 完整阶段结论 |",
        "|---|---|---|---:|---|---|",
    ])
    for stage in ("validation", "sealed_test", "random_audit"):
        decision = results[stage]["decision"]
        absolute = decision["absolute_probability_gate"]
        lines.append(
            f"| {stage} | {decision['static_baseline_gate']['ADAM']} | "
            f"BL1={absolute['ADAM_BL1']}, BL2={absolute['ADAM_BL2']} | "
            f"{decision['positive_AP_days']['ADAM']}/{decision['required_positive_AP_days']} | "
            f"{decision['probability_saturation_gate']} | {decision['scientific_status']} |"
        )
    terminal = terminal_interpretation({
        stage: result["decision"] for stage, result in results.items()
    })
    lines.extend([
        "", "## 分析与结论", "", terminal["conclusion"], "",
        "终局结论使用完整阶段门，而不只看单一 AP 或 history gate；Validation 的失败不能被后置阶段反向救回。", "",
        "所有结果均为离线预测证据，不证明因果或线上业务提升。SGD 线路因 Train-only 求解不足被单独冻结，因此不能声称结论对优化器选择稳健。", "",
        "## 可作图文件", "",
        "每个阶段均提供 `pooled_metrics.csv`、`daily_metrics.csv`、`model_contrasts.csv`、"
        "`paired_user_cluster_bootstrap.csv`、`history_depth_slice_metrics.csv`、"
        "`probability_distribution_audit.csv` 和逐行 `predictions.parquet`。这些文件可直接绘制"
        "阶段对比图、逐日效应图、置信区间图、历史深度分层图和概率分布图。", "",
    ])
    return "\n".join(lines)


def run_release(approved_hash: str) -> None:
    v1.run_release(approved_hash)
    final_path = OUTPUT_ROOT / "final_claim_decision.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["terminal_interpretation"] = terminal_interpretation(final["stage_decisions"])
    final["report_path"] = str(REPORT_PATH.relative_to(PROJECT_ROOT))
    v1.write_json(final_path, final)
    v1.finalize_hashes(OUTPUT_ROOT)


def validate_only() -> None:
    contract, digest = load_contract()
    if contract["contract_id"] != "history_value_adam_confirmation_v004":
        raise v1.ContractStop("wrong v004 contract id")
    read_adam_evidence()
    v1.implementation_records(contract)
    v1.environment_manifest(contract, require_cuda=False)
    v1.progress(f"V004_VALIDATE_ONLY_OK contract_sha256={digest}; governed data not opened")


def install_overrides() -> None:
    v1.CONTRACT_PATH = CONTRACT_PATH
    v1.APPROVAL_PATH = APPROVAL_PATH
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.REPORT_PATH = REPORT_PATH
    v1.STREAMS = STREAMS
    v1.LEARNED = LEARNED
    v1.CONTRASTS = CONTRASTS
    v1.load_contract = load_contract
    v1.verify_approval_receipt = verify_approval_receipt
    v1.reference_objectives = v2.reference_objectives
    v1.preflight = preflight
    v1.fit_standard_stage = fit_standard_stage
    v1.stage_decision = stage_decision
    v1.render_report = render_report
    v1.validate_only = validate_only


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
        raise v1.ContractStop("--approved-contract-sha256 is required for v004 release")
    run_release(args.approved_contract_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
