from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_history_value_adam_random_v009 as runner  # noqa: E402


def test_v009_has_exact_sealed_pass_prerequisite() -> None:
    contract, _ = runner.load_contract()
    checkpoint = runner.verify_sealed_prerequisite(contract)
    assert checkpoint["sealed_decision"]["scientific_status"] == "pass"
    assert checkpoint["random_audit_accessed"] is False


def test_v009_prechecks_exact_random_count_and_coverage() -> None:
    contract, _ = runner.load_contract()
    evidence = contract["random_target_precheck"]
    assert evidence["silver_main_rows"] == 42982
    assert evidence["official_formula_mismatch_addback_rows"] == 45
    assert evidence["canonical_union_expected_rows"] == runner.EXPECTED_RANDOM_ROWS == 43027
    assert evidence["standard_history_unseen_user_rows"] == 0
    assert evidence["videos_basic_missing_metadata_rows"] == 0
    runner.verify_count_evidence(contract)


def test_v009_requires_reconstruction_before_random_access() -> None:
    contract, _ = runner.load_contract()
    reconstruction = contract["frozen_model_reconstruction"]
    assert reconstruction["random_label_or_feature_access_before_reconstruction_pass"] == "forbidden"
    checks = reconstruction["required_reproduction_checks"]
    assert checks["sealed_target_identity_exact"] is True
    assert checks["raw_score_max_abs_difference"] == 0.000001
    assert checks["calibrated_probability_max_abs_difference"] == 0.000000001


def test_v009_preserves_frozen_adam_and_random_history_rules() -> None:
    contract, _ = runner.load_contract()
    assert tuple(contract["model_matrix"]["required_prediction_streams"]) == runner.v4.STREAMS
    assert contract["GPU_optimizers"]["GPU_ADAM"]["learning_rate"] == .03
    assert contract["GPU_optimizers"]["GPU_ADAM"]["selected_steps"] == 100
    assert contract["bootstrap"]["replicates"] == 2000
    stage = contract["sequential_stage_protocol"]["stage_3_random_audit"]
    assert stage["history_source"] == "standard_events_only"
    assert stage["random_fit_recalibration_or_threshold_selection"] == "forbidden"


def test_v009_approval_absent_before_exact_hash_approval() -> None:
    assert not runner.APPROVAL_PATH.exists()
    with pytest.raises(runner.v1.ContractStop, match="receipt is missing"):
        runner.verify_approval_receipt("not-approved")
