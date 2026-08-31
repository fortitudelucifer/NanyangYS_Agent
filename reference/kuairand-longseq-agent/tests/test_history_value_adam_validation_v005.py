from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_history_value_adam_validation_v005 as runner  # noqa: E402


def test_v005_authorizes_validation_only() -> None:
    contract, _ = runner.load_contract()
    authorization = contract["authorization"]
    assert authorization["authorized_stage_scope_after_exact_hash_approval"] == ["preflight", "validation"]
    assert authorization["sealed_test_access_authorized"] is False
    assert authorization["random_audit_access_authorized"] is False
    assert authorization["automatic_ordered_transitions_authorized"] is False


def test_v005_preserves_adam_design() -> None:
    contract, _ = runner.load_contract()
    assert tuple(contract["model_matrix"]["required_prediction_streams"]) == runner.v4.STREAMS
    assert contract["GPU_optimizers"]["GPU_ADAM"]["learning_rate"] == .03
    assert contract["GPU_optimizers"]["GPU_ADAM"]["selected_steps"] == 100
    assert contract["bootstrap"]["replicates"] == 2000


def test_v005_terminal_claim_is_interim() -> None:
    contract, _ = runner.load_contract()
    assert contract["terminal_interpretation"]["status"] == "validation_interim_only"
    assert "random_exposure_transport" in contract["terminal_interpretation"]["prohibited_claims"]


def test_v005_approval_absent_before_exact_hash_approval() -> None:
    assert not runner.APPROVAL_PATH.exists()
    with pytest.raises(runner.v1.ContractStop, match="receipt is missing"):
        runner.verify_approval_receipt("not-approved")
