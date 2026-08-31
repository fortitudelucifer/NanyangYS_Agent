from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_history_value_adam_random_v010 as runner  # noqa: E402


def test_v010_pins_failed_v009_without_random_access() -> None:
    contract, _ = runner.load_contract()
    failure = runner.verify_v009_stop(contract)
    assert failure["failed_stage"] == "sealed_model_reconstruction_prediction_reproduction_gate"
    assert not any(failure["access_and_output_boundary"].values())


def test_v010_diagnostic_is_sealed_only_and_numerically_equivalent() -> None:
    contract, _ = runner.load_contract()
    audit = runner.verify_diagnostic(contract)
    assert audit["random_input_opened"] is False
    assert audit["sealed_target_identity_exact"] is True
    assert max(abs(row["difference"]) for row in audit["metric_reproduction"]) < 1e-6


def test_v010_loads_exact_frozen_model_without_refit() -> None:
    contract, _ = runner.load_contract()
    frozen = runner.load_and_verify_frozen_model(contract)
    assert set(frozen.fits) == {"ADAM_BL1", "ADAM_BL2"}
    assert set(frozen.calibrators) == {"ADAM_BL1", "ADAM_BL2"}


def test_v010_preserves_random_only_rules() -> None:
    contract, _ = runner.load_contract()
    stage = contract["sequential_stage_protocol"]["stage_3_random_audit"]
    assert stage["expected_target_rows"] == runner.EXPECTED_RANDOM_ROWS == 43027
    assert stage["history_source"] == "standard_events_only"
    assert stage["random_fit_recalibration_or_threshold_selection"] == "forbidden"
    assert stage["model_source"] == "exact_hash_pinned_sealed_diagnostic_pickle"
    assert contract["bootstrap"]["replicates"] == 2000


def test_v010_freezes_adam_and_sgd_scope() -> None:
    contract, _ = runner.load_contract()
    assert tuple(contract["model_matrix"]["required_prediction_streams"]) == runner.v4.STREAMS
    assert contract["GPU_optimizers"]["GPU_ADAM"]["learning_rate"] == .03
    assert contract["GPU_optimizers"]["GPU_ADAM"]["selected_steps"] == 100
    assert contract["successor_reason"]["SGD_status"] == "frozen_deferred"


def test_v010_approval_absent_before_exact_hash_approval() -> None:
    assert not runner.APPROVAL_PATH.exists()
    with pytest.raises(runner.v1.ContractStop, match="receipt is missing"):
        runner.verify_approval_receipt("not-approved")
