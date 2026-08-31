from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_history_value_gpu_confirmation_v002 as runner  # noqa: E402


def test_successor_deep_merge_only_overrides_nested_value() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2]}
    merged = runner.deep_merge(base, {"a": {"x": 3}})
    assert merged == {"a": {"x": 3, "y": 2}, "b": [1, 2]}


def test_successor_contract_pins_base_and_reference_cap() -> None:
    contract, _ = runner.load_contract()
    assert contract["contract_id"] == "history_value_gpu_confirmation_v002"
    assert contract["base_contract"]["sha256"] == runner.EXPECTED_BASE_SHA256
    assert contract["optimizer_adequacy"]["reference_solver"]["max_iter"] == 2000
    assert contract["model_matrix"]["required_prediction_streams"] == [
        "BL0", "ADAM_BL1", "ADAM_BL2", "SGD_BL1", "SGD_BL2"
    ]


def test_successor_approval_receipt_matches_executed_v002_hash() -> None:
    assert runner.APPROVAL_PATH.exists()
    receipt = runner.verify_approval_receipt(
        "a11e75f43b2fe013e9ea0ee813953b46c224f8a5786e64a1c3bd25a43b37d09b"
    )
    assert receipt["approved_by"] == "project_owner"
    with pytest.raises(runner.v1.ContractStop, match="contract_sha256"):
        runner.verify_approval_receipt("not-the-approved-hash")
