from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = Path(__file__).resolve().parents[1] / "run_retraining_v012.py"
SPEC = importlib.util.spec_from_file_location("run_retraining_v012", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _contract() -> dict:
    return runner.load_contract()[0]


def _selection_metric(config_id: str, model_id: str, *, log_loss: float) -> dict:
    values = {
        "TARGET_BL0": dict(ap=0.10, gauc=0.50, brier=0.20, ece=0.0, gap=0.0),
        "OLD_BL2_PLUS_V011": dict(ap=0.20, gauc=0.60, brier=0.15, ece=0.01, gap=0.0),
        "NEW_BL1": dict(ap=0.15, gauc=0.55, brier=0.18, ece=0.01, gap=0.0),
        "NEW_BL2": dict(ap=0.21, gauc=0.62, brier=0.12, ece=0.01, gap=0.001),
    }[model_id]
    return {
        "config_id": config_id,
        "model_id": model_id,
        "average_precision": values["ap"],
        "user_gauc_event_weighted": values["gauc"],
        "log_loss": log_loss,
        "brier": values["brier"],
        "ece20_equal_width": values["ece"],
        "mean_probability_minus_prevalence": values["gap"],
    }


def test_contract_freezes_three_paired_configs_and_four_disjoint_splits() -> None:
    contract = _contract()
    assert tuple(row["config_id"] for row in contract["adaptation_candidates"]) == runner.CONFIG_IDS
    assert all(row["steps"] == 100 for row in contract["adaptation_candidates"])
    assert contract["scope"]["SGD_status"] == "frozen_deferred"
    assert sum(int(row["rows"]) for row in contract["temporal_splits"].values()) == 43_027
    assert [row["start"] for row in contract["temporal_splits"].values()] == [
        "2022-04-22", "2022-04-30", "2022-05-03", "2022-05-06"
    ]


def test_snapshot_hash_is_compiled_and_exact() -> None:
    digest = hashlib.sha256(runner.SNAPSHOT_PATH.read_bytes()).hexdigest()
    assert digest == runner.EXPECTED_SNAPSHOT_SHA256
    assert digest == _contract()["predecessor_integrity"]["snapshot_sha256"]


def test_intercept_calibrator_matches_target_mean_and_is_monotone() -> None:
    raw = np.linspace(-3, 3, 1000, dtype=np.float64)
    labels = np.zeros(1000, dtype=np.int8)
    labels[:137] = 1
    calibrator = runner.fit_intercept_calibrator(
        "NEW_BL2", raw, labels, 1.0169043468053687, _contract()
    )
    probability = calibrator.apply(raw)
    assert abs(float(probability.mean()) - 0.137) < 1e-12
    assert np.all(np.diff(probability) > 0)


def test_selection_tie_uses_more_conservative_configuration() -> None:
    metrics = []
    new_log_loss = {
        "C1_conservative": 0.20000,
        "C2_balanced": 0.19995,
        "C3_aggressive": 0.21000,
    }
    for config_id in runner.CONFIG_IDS:
        for model_id in runner.FINAL_MODELS:
            log_loss = {
                "TARGET_BL0": 0.30,
                "OLD_BL2_PLUS_V011": 0.25,
                "NEW_BL1": 0.28,
                "NEW_BL2": new_log_loss[config_id],
            }[model_id]
            metrics.append(_selection_metric(config_id, model_id, log_loss=log_loss))
    rows, decision = runner.candidate_selection(metrics, _contract())
    assert all(row["eligible"] for row in rows)
    assert decision["selected_config_id"] == "C1_conservative"
    assert decision["tied_configs"] == ["C1_conservative", "C2_balanced"]


def test_selection_failure_keeps_failure_and_uses_diagnostic_fallback() -> None:
    metrics = []
    for config_id, bl2_log_loss in zip(runner.CONFIG_IDS, (0.25, 0.20, 0.23)):
        for model_id in runner.FINAL_MODELS:
            log_loss = {
                "TARGET_BL0": 0.30,
                "OLD_BL2_PLUS_V011": 0.19,
                "NEW_BL1": 0.21,
                "NEW_BL2": bl2_log_loss,
            }[model_id]
            row = _selection_metric(config_id, model_id, log_loss=log_loss)
            if model_id == "NEW_BL2":
                row["average_precision"] = 0.14
            metrics.append(row)
    rows, decision = runner.candidate_selection(metrics, _contract())
    assert not any(row["eligible"] for row in rows)
    assert decision["selected_config_id"] == "C2_balanced"
    assert decision["fallback_used_because_no_eligible_candidate"] is True
    assert decision["selected_was_eligible"] is False


def _bootstrap_row(contrast: str, metric: str, point: float, low: float, high: float) -> dict:
    return {
        "contrast": contrast,
        "metric": metric,
        "point_estimate": point,
        "ci95_lower": low,
        "ci95_upper": high,
    }


def test_final_decision_requires_extra_value_and_preserves_old_conclusions() -> None:
    bootstrap = []
    values = {
        "NEW_BL1_minus_TARGET_BL0": {
            "average_precision": (0.10, 0.05, 0.15),
            "user_gauc_event_weighted": (0.02, 0.01, 0.03),
            "log_loss": (-0.08, -0.10, -0.06),
            "brier": (-0.04, -0.05, -0.03),
        },
        "NEW_BL2_minus_NEW_BL1": {
            "average_precision": (0.02, 0.01, 0.03),
            "user_gauc_event_weighted": (0.01, 0.00, 0.02),
            "log_loss": (-0.01, -0.02, -0.001),
            "brier": (-0.01, -0.02, -0.001),
        },
        "NEW_BL2_minus_OLD_BL2_PLUS_V011": {
            "average_precision": (0.001, -0.004, 0.006),
            "user_gauc_event_weighted": (0.001, -0.002, 0.004),
            "log_loss": (-0.01, -0.02, -0.005),
            "brier": (-0.001, -0.01, 0.004),
        },
    }
    for contrast, metrics in values.items():
        for metric, (point, low, high) in metrics.items():
            bootstrap.append(_bootstrap_row(contrast, metric, point, low, high))
    points = [
        {"model_id": model_id, "ece20_equal_width": 0.01,
         "mean_probability_minus_prevalence": 0.001}
        for model_id in runner.FINAL_MODELS
    ]
    audits = [
        {"model_id": model_id, "finite_share": 1.0,
         "below_or_equal_1e_6_share": 0.0,
         "above_or_equal_1_minus_1e_6_share": 0.0}
        for model_id in runner.FINAL_MODELS
    ]
    daily = []
    for day in ("2022-05-06", "2022-05-07", "2022-05-08"):
        daily.extend([
            {"event_date": day, "model_id": "NEW_BL1", "average_precision": 0.20,
             "mean_probability_minus_prevalence": 0.001},
            {"event_date": day, "model_id": "NEW_BL2", "average_precision": 0.22,
             "mean_probability_minus_prevalence": 0.001},
        ])
    decision = runner.final_decision(
        points, bootstrap, daily, audits,
        {"selected_was_eligible": True}, _contract()
    )
    assert decision["scientific_status"] == "retraining_adds_value"
    assert decision["all_required_gates_passed"] is True
    assert decision["changes_v010_history_conclusion"] is False
    assert decision["changes_v011_calibration_conclusion"] is False


def test_approval_and_outputs_absent_before_exact_hash_approval() -> None:
    assert not runner.APPROVAL_PATH.exists()
    assert not runner.OUTPUT_ROOT.exists()
    assert not runner.REPORT_PATH.exists()
