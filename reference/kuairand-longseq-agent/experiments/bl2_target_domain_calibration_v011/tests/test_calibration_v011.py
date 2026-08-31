from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
import run_calibration_v011 as runner  # noqa: E402


def synthetic_data() -> dict[str, np.ndarray]:
    raw = np.linspace(-3.0, 3.0, 400)
    probability = 1.0 / (1.0 + np.exp(-raw))
    labels = (np.arange(raw.size) % 7 == 0).astype(np.int8)
    return {
        "source_table": np.asarray(["synthetic"] * raw.size),
        "source_row_number": np.arange(raw.size),
        "user_id": np.arange(raw.size) % 40,
        "event_date": np.asarray([np.datetime64("2022-04-22")] * raw.size),
        "long_view": labels, "raw_ADAM_BL2": raw, "p_ADAM_BL2": probability,
    }


def test_predecessor_integrity_and_input_metadata() -> None:
    contract, _ = runner.load_contract()
    snapshot = runner.verify_predecessor_integrity(contract)
    metadata = runner.verify_input_metadata(contract)
    assert snapshot["calibration_input"]["rows"] == metadata["rows"] == 43027
    assert snapshot["v010_final_decision"]["required_status"] == "history_supported_on_standard_and_random_under_frozen_Adam"


def test_temporal_splits_are_exact_and_disjoint() -> None:
    contract, _ = runner.load_contract()
    splits = contract["temporal_splits"]
    assert splits["calibration_fit"] == {"start": "2022-04-22", "end": "2022-04-27", "rows": 8731, "positives": 686}
    assert splits["calibration_selection"] == {"start": "2022-04-28", "end": "2022-05-02", "rows": 10544, "positives": 872}
    assert splits["held_out_test"] == {"start": "2022-05-03", "end": "2022-05-08", "rows": 23752, "positives": 2063}
    assert sum(split["rows"] for split in splits.values()) == 43027


def test_all_candidate_families_are_monotone_on_synthetic_data() -> None:
    contract, _ = runner.load_contract()
    data = synthetic_data()
    for family in runner.CANDIDATES:
        spec = runner.fit_candidate(family, data, contract)
        probability = runner.apply_candidate(spec, data)
        assert spec.converged and spec.slope > 0
        assert runner.stable_order_exact(data["raw_ADAM_BL2"], probability)
        assert np.isfinite(probability).all()


def test_selection_tie_prefers_simpler_family() -> None:
    contract, _ = runner.load_contract()
    rows = [
        {"family": "M1_prior_shift", "log_loss": .20005, "eligible": True},
        {"family": "M2_intercept_only", "log_loss": .2, "eligible": True},
        {"family": "M3_platt", "log_loss": .19, "eligible": False},
    ]
    selection = runner.select_family(rows, contract)
    assert selection["selected_family"] == "M1_prior_shift"


def test_contract_forbids_model_retraining_and_test_retry() -> None:
    contract, _ = runner.load_contract()
    assert contract["scope"]["BL1_or_BL2_retraining"] == "forbidden"
    assert contract["scope"]["feature_recomputation"] == "forbidden"
    assert contract["scope"]["original_random_reopen"] == "forbidden"
    assert contract["held_out_test_protocol"]["maximum_metric_releases"] == 1
    assert contract["held_out_test_protocol"]["retry_after_metric_release"] == "forbidden"


def test_approval_and_outputs_absent_before_exact_hash_approval() -> None:
    assert not runner.APPROVAL_PATH.exists()
    assert not runner.OUTPUT_ROOT.exists()
    assert not runner.REPORT_PATH.exists()
    with pytest.raises(runner.ContractStop, match="receipt is missing"):
        runner.verify_approval("not-approved")
