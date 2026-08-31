from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


RUNNER_PATH = Path(__file__).resolve().parents[1] / "run_learning_curve_v013.py"
SPEC = importlib.util.spec_from_file_location("run_learning_curve_v013", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_contract_freezes_40_gpu_fits_and_no_sgd() -> None:
    contract, _ = runner.load_contract()
    runner.verify_contract_shape(contract)
    assert contract["execution_environment"]["maximum_GPU_fits"] == 40
    assert contract["scope"]["SGD_status"] == "frozen_deferred"
    assert contract["optimizer"]["name"] == "ADAM"


def test_nested_user_samples_are_deterministic_and_nested() -> None:
    users = np.arange(1, 101, dtype=np.int64)
    fractions = [0.10, 0.25, 0.50, 0.75, 1.0]
    first = runner.nested_user_prefixes(users, fractions, 20260814)
    second = runner.nested_user_prefixes(users, fractions, 20260814)
    previous: set[int] = set()
    for fraction in fractions:
        assert np.array_equal(first[fraction], second[fraction])
        current = set(map(int, first[fraction]))
        assert previous.issubset(current)
        previous = current
    assert len(first[0.10]) == 10
    assert len(first[1.0]) == 100


def test_sample_hash_depends_on_exact_user_identity() -> None:
    a = np.asarray([1, 2, 3], dtype=np.int64)
    b = np.asarray([1, 2, 4], dtype=np.int64)
    assert runner.sample_digest(a) == runner.sample_digest(a.copy())
    assert runner.sample_digest(a) != runner.sample_digest(b)


def test_bootstrap_summary_detects_positive_candidate_delta() -> None:
    values = np.asarray([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    row = runner.summary_row("p", "BL2_minus_BL1", "average_precision", 0.025, values)
    assert row["point_estimate"] > 0
    assert row["ci95_lower"] > 0
    assert row["effective_replicates"] == 4


def test_bootstrap_block_optimization_does_not_change_contract_statistics() -> None:
    contract, _ = runner.load_contract()
    assert runner.AP_BOOTSTRAP_BLOCK_SIZE == 64
    assert contract["statistics"]["bootstrap_replicates"] == 2000
    assert contract["statistics"]["bootstrap_seed"] == 20260814
    labels = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)
    probability = np.asarray([0.1, 0.9, 0.2, 0.8, 0.7, 0.3], dtype=np.float64)
    row_user = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32)
    multiplicities, _ = runner.metrics.make_multiplicities(user_count=3, replicates=40, seed=20260814)
    reference = runner.metrics.weighted_ap_replicates(
        labels, probability, row_user, multiplicities, block_size=2,
    )
    optimized = runner.metrics.weighted_ap_replicates(
        labels, probability, row_user, multiplicities, block_size=64,
    )
    assert np.allclose(reference, optimized, equal_nan=True)


def test_exact_hash_approval_matches_contract_and_completed_release() -> None:
    _, digest = runner.load_contract()
    receipt = runner.verify_approval(digest)
    assert receipt["execution_authorized"] is True
    assert receipt["contract_sha256"] == digest
    assert runner.OUTPUT_ROOT.exists()
    decision = runner.json.loads((runner.OUTPUT_ROOT / "final_decision.json").read_text())
    assert decision["scientific_status"] == "complete"
    assert decision["completed_curve_points"] == 20
    assert decision["failed_curve_points"] == 0
