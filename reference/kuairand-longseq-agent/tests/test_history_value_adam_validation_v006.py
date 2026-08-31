from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_history_value_adam_validation_v006 as runner  # noqa: E402


def test_v006_repairs_only_canonical_target_count() -> None:
    contract, _ = runner.load_contract()
    assert contract["sequential_stage_protocol"]["stage_1_validation"]["expected_target_rows"] == 886452
    assert contract["data_governance"]["known_non_model_counts_for_preflight"]["Validation_target_silver_main_rows"] == 881035
    assert contract["data_governance"]["known_non_model_counts_for_preflight"]["Validation_target_official_mismatch_addback_rows"] == 5417


def test_v006_failure_evidence_proves_arithmetic_and_early_stop() -> None:
    contract, _ = runner.load_contract()
    evidence = runner.verify_failure_evidence(contract)
    assert evidence["observed_canonical_target_rows"] == 886452
    assert evidence["access_and_output_boundary"]["GPU_model_fit_started"] is False
    assert evidence["access_and_output_boundary"]["predictions_materialized"] is False
    assert evidence["access_and_output_boundary"]["metrics_computed"] is False


def test_v006_preserves_validation_only_authority_and_model() -> None:
    contract, _ = runner.load_contract()
    authorization = contract["authorization"]
    assert authorization["authorized_stage_scope_after_exact_hash_approval"] == ["preflight", "validation"]
    assert authorization["sealed_test_access_authorized"] is False
    assert authorization["random_audit_access_authorized"] is False
    assert tuple(contract["model_matrix"]["required_prediction_streams"]) == runner.v4.STREAMS
    assert contract["GPU_optimizers"]["GPU_ADAM"]["selected_steps"] == 100


def test_v006_approval_absent_before_exact_hash_approval() -> None:
    assert not runner.APPROVAL_PATH.exists()
    with pytest.raises(runner.v1.ContractStop, match="receipt is missing"):
        runner.verify_approval_receipt("not-approved")
