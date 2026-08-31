import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kuairand_longseq.evaluation.gate2b_metrics import (  # noqa: E402
    EXPECTED_MULTIPLICITY_SHA256,
    binary_auc_midrank,
    ece_equal_width,
    make_multiplicities,
    paired_user_cluster_bootstrap,
    point_metrics,
)


class Gate2BMetricsTests(unittest.TestCase):
    def test_midrank_auc_matches_sklearn_with_ties(self):
        y = np.array([0, 1, 0, 1, 1, 0], dtype=np.int8)
        p = np.array([0.1, 0.8, 0.4, 0.4, 0.8, 0.2])
        self.assertAlmostEqual(binary_auc_midrank(y, p), roc_auc_score(y, p), places=14)

    def test_constant_score_average_precision_is_prevalence(self):
        y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
        p = np.full(y.size, 0.5)
        users = np.array([1, 1, 2, 2, 3, 3])
        metrics = point_metrics(y, p, users)
        self.assertAlmostEqual(metrics["average_precision"], y.mean(), places=14)
        self.assertAlmostEqual(metrics["average_precision"], average_precision_score(y, p), places=14)
        self.assertAlmostEqual(metrics["user_gauc_event_weighted"], 0.5, places=14)

    def test_ece_places_probability_one_in_last_bin(self):
        y = np.array([0, 1], dtype=np.int8)
        p = np.array([0.0, 1.0])
        _, rows = ece_equal_width(y, p, bins=20)
        self.assertEqual(rows[0]["rows"], 1)
        self.assertEqual(rows[-1]["rows"], 1)

    def test_frozen_bootstrap_digest(self):
        matrix, digest = make_multiplicities()
        self.assertEqual(matrix.shape, (2000, 950))
        self.assertEqual(digest, EXPECTED_MULTIPLICITY_SHA256)
        self.assertEqual(
            digest,
            hashlib.sha256(matrix.astype("<u2", copy=False).tobytes(order="C")).hexdigest(),
        )

    def test_identical_predictions_have_exact_zero_paired_delta(self):
        y = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int8)
        p = np.array([0.1, 0.8, 0.2, 0.7, 0.6, 0.3, 0.9, 0.4])
        users = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
        universe = np.array([1, 2, 3, 4], dtype=np.int64)
        matrix, _ = make_multiplicities(user_count=4, replicates=64, seed=20260814)
        rows = paired_user_cluster_bootstrap(
            y,
            p,
            p.copy(),
            users,
            user_universe=universe,
            multiplicities=matrix,
            ap_block_size=4,
        )
        for row in rows:
            self.assertEqual(row["point_estimate"], 0.0)
            self.assertEqual(row["bootstrap_se"], 0.0)
            self.assertEqual(row["ci95_lower"], 0.0)
            self.assertEqual(row["ci95_upper"], 0.0)


if __name__ == "__main__":
    unittest.main()
