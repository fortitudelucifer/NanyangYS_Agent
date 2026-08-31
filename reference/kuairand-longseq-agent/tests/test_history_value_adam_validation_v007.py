from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_history_value_adam_validation_v007 as runner  # noqa: E402


def test_v007_reuses_exact_validated_features_and_repairs_time_ms_only() -> None:
    contract, _ = runner.load_contract()
    reuse = contract["validated_feature_reuse"]
    assert reuse["sha256"] == runner.EXPECTED_FEATURE_SHA256
    assert reuse["validation_target_rows"] == 886452
    assert contract["successor_reason"]["permitted_change_only"] == "add_time_ms_to_in_memory_Frame_loading"


def test_v007_feature_schema_has_every_evaluation_identity() -> None:
    schema = set(runner.v1.pq.ParquetFile(runner.FEATURE_PATH).schema_arrow.names)
    required = {"source_table", "source_row_number", "user_id", "video_id", "event_date", "time_ms", "long_view", "prior_batch_n"}
    assert required <= schema


def test_v007_preserves_validation_only_authority_and_model() -> None:
    contract, _ = runner.load_contract()
    authorization = contract["authorization"]
    assert authorization["authorized_stage_scope_after_exact_hash_approval"] == ["preflight", "validation"]
    assert authorization["sealed_test_access_authorized"] is False
    assert authorization["random_audit_access_authorized"] is False
    assert tuple(contract["model_matrix"]["required_prediction_streams"]) == runner.v4.STREAMS
    assert contract["GPU_optimizers"]["GPU_ADAM"]["selected_steps"] == 100


def test_v007_approval_absent_before_exact_hash_approval() -> None:
    assert not runner.APPROVAL_PATH.exists()
    with pytest.raises(runner.v1.ContractStop, match="receipt is missing"):
        runner.verify_approval_receipt("not-approved")
