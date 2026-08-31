"""No-data tests for the separate Gate 2B CUDA engineering demo."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts/run_gate2b_gpu_demo.py"
SPEC = importlib.util.spec_from_file_location("gate2b_gpu_demo", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


def test_deterministic_subset_is_stable_and_label_independent() -> None:
    frame = DEMO.canonical.Frame(
        columns={
            "source_row_number": np.arange(1000, dtype=np.int64),
            "source_table": np.asarray(["a"] * 500 + ["b"] * 500),
            "long_view": np.arange(1000, dtype=np.int64) % 2,
        }
    )
    candidates = np.arange(1000, dtype=np.int64)
    first = DEMO.deterministic_subset(frame, candidates, 100, salt=7)
    frame.columns["long_view"] = 1 - frame.columns["long_view"]
    second = DEMO.deterministic_subset(frame, candidates, 100, salt=7)
    assert np.array_equal(first, second)
    assert first.size == 100
    assert np.all(first[1:] > first[:-1])


def test_cpu_sparse_logistic_objective_decreases() -> None:
    rng = np.random.default_rng(20260814)
    dense = rng.normal(size=(500, 12)).astype(np.float32)
    matrix = sparse.csr_matrix(dense)
    label = (dense[:, 0] - 0.5 * dense[:, 1] > 0).astype(np.int8)
    _, _, record = DEMO.train_sparse_logistic(
        matrix,
        label,
        device=DEMO.torch.device("cpu"),
        steps=10,
        learning_rate=0.03,
        alpha=1e-4,
    )
    assert record["final_objective"] < record["initial_objective"]
    assert record["device"] == "cpu"

