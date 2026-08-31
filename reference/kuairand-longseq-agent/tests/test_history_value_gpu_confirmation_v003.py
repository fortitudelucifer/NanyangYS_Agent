from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_history_value_gpu_confirmation_v003 as runner  # noqa: E402


def test_v003_contract_inherits_science_and_extends_plain_sgd_only() -> None:
    contract, _ = runner.load_contract()
    assert contract["contract_id"] == "history_value_gpu_confirmation_v003"
    assert contract["model_matrix"]["required_prediction_streams"] == [
        "BL0", "ADAM_BL1", "ADAM_BL2", "SGD_BL1", "SGD_BL2"
    ]
    sgd = contract["GPU_optimizers"]["GPU_SGD"]
    assert sgd["learning_rate_candidates"] == [1.0]
    assert sgd["candidate_step_checkpoints"] == [3000, 5000, 10000]
    assert sgd["momentum"] == 0.0
    assert contract["stage_gates"]["common_history_gate_per_optimizer"]["pooled_average_precision_delta_minimum"] == .005


def test_v003_reuses_exact_six_of_six_adam_evidence() -> None:
    rows, manifest = runner.read_v002_adam_evidence()
    assert len(rows) == 6
    assert manifest["steps"] == 100
    assert manifest["learning_rate"] == .03


def test_v003_approval_absent_before_exact_hash_approval() -> None:
    assert not runner.APPROVAL_PATH.exists()
    with pytest.raises(runner.v1.ContractStop, match="receipt is missing"):
        runner.verify_approval_receipt("not-approved")
