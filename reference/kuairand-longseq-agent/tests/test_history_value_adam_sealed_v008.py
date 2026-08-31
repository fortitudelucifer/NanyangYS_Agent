from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_history_value_adam_sealed_v008 as runner  # noqa: E402


def test_v008_has_exact_validation_pass_prerequisite() -> None:
    contract, _ = runner.load_contract()
    checkpoint = runner.verify_validation_prerequisite(contract)
    assert checkpoint["validation_decision"]["scientific_status"] == "pass"
    assert checkpoint["sealed_test_accessed"] is False
    assert checkpoint["random_audit_accessed"] is False


def test_v008_prechecks_correct_canonical_sealed_count() -> None:
    contract, _ = runner.load_contract()
    repair = contract["sealed_target_count_repair"]
    assert repair["prior_main_table_only_count"] == 4404167
    assert repair["official_formula_mismatch_addback_rows"] == 27132
    assert repair["canonical_union_expected_target_rows"] == runner.EXPECTED_TARGET_ROWS == 4431299
    runner.verify_count_evidence(contract)


def test_v008_authorizes_sealed_only_and_keeps_random_locked() -> None:
    contract, _ = runner.load_contract()
    authorization = contract["authorization"]
    assert authorization["authorized_stage_scope_after_exact_hash_approval"] == ["preflight", "sealed_test"]
    assert authorization["automatic_ordered_transitions_authorized"] is False
    assert authorization["random_audit_access_authorized"] is False


def test_v008_preserves_adam_model_and_statistical_gate() -> None:
    contract, _ = runner.load_contract()
    assert tuple(contract["model_matrix"]["required_prediction_streams"]) == runner.v4.STREAMS
    assert contract["GPU_optimizers"]["GPU_ADAM"]["learning_rate"] == .03
    assert contract["GPU_optimizers"]["GPU_ADAM"]["selected_steps"] == 100
    assert contract["bootstrap"]["replicates"] == 2000
    assert contract["stage_gates"]["sealed_test_gate"]["minimum_positive_AP_days"] == 12


def test_v008_approval_absent_before_exact_hash_approval() -> None:
    assert not runner.APPROVAL_PATH.exists()
    with pytest.raises(runner.v1.ContractStop, match="receipt is missing"):
        runner.verify_approval_receipt("not-approved")
