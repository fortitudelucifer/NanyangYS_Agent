from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_history_value_adam_confirmation_v004 as runner  # noqa: E402


def test_v004_has_only_three_prediction_streams_and_frozen_sgd() -> None:
    contract, _ = runner.load_contract()
    assert tuple(contract["model_matrix"]["required_prediction_streams"]) == runner.STREAMS
    assert runner.STREAMS == ("BL0", "ADAM_BL1", "ADAM_BL2")
    assert contract["GPU_optimizers"]["GPU_SGD"]["status"] == "frozen_not_executed"
    assert contract["terminal_interpretation"]["required_limitation"].startswith("optimizer_robustness_not_established")


def test_v004_inherits_exact_adequate_Adam_checkpoint() -> None:
    rows, evidence = runner.read_adam_evidence()
    assert len(rows) == 6
    assert evidence["learning_rate"] == .03
    assert evidence["steps"] == 100
    assert evidence["maximum_regret"] < .002606


def test_v004_contrasts_exclude_sgd() -> None:
    assert set(runner.CONTRASTS) == {
        "ADAM_BL1_minus_BL0", "ADAM_BL2_minus_ADAM_BL1"
    }
    assert all("SGD" not in name for name in runner.CONTRASTS)


def test_v004_approval_absent_before_exact_hash_approval() -> None:
    assert not runner.APPROVAL_PATH.exists()
    with pytest.raises(runner.v1.ContractStop, match="receipt is missing"):
        runner.verify_approval_receipt("not-approved")


def test_terminal_interpretation_uses_complete_stage_status() -> None:
    decisions = {
        "validation": {"scientific_status": "fail_or_mixed"},
        "sealed_test": {"scientific_status": "pass"},
        "random_audit": {"scientific_status": "pass"},
    }
    result = runner.terminal_interpretation(decisions)
    assert result["status"] == "falsified_at_validation"
    assert result["optimizer_robustness_established"] is False
