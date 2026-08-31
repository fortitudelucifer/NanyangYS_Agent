"""Synthetic tests for the Gate 2B v003 probability-repair primitives.

Every test builds its own in-memory data.  No test reads Silver, quarantine,
the frozen Gate 2B feature artifact, Validation, the restricted test, or the
random table, so running this file cannot consume any protected evidence.
"""

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse
from sklearn.linear_model import LogisticRegression, SGDClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kuairand_longseq.models import gate2b_repair_v003 as repair  # noqa: E402

CONTRACT_PATH = PROJECT_ROOT / "configs/gate2b_probability_repair_contract_v003.yaml"


def load_contract() -> dict:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def synthetic_blocks(rows: int, seed: int = 7) -> dict[str, np.ndarray]:
    """A small point-in-time-shaped block set with real signal in H2."""

    rng = np.random.default_rng(seed)
    categorical = np.column_stack(
        [rng.integers(0, size, size=rows) for size in (5, 40, 12, 9, 3, 4, 3, 6, 7)]
    ).astype(np.int64)
    static_continuous = rng.normal(size=(rows, len(repair.STATIC_CONTINUOUS_FIELDS)))
    static_binary = rng.integers(0, 2, size=(rows, len(repair.STATIC_BINARY_FIELDS))).astype(
        np.float64
    )
    prior_event = rng.integers(0, 400, size=rows).astype(np.float64)
    latent = rng.beta(2.0, 4.0, size=rows)
    raw = {
        "prior_batch_n": rng.integers(0, 300, size=rows).astype(np.float64),
        "prior_event_n": prior_event,
        "prior_positive_n": np.floor(prior_event * latent),
        "last_user_gap_s": rng.integers(0, 90_000, size=rows).astype(np.float64),
    }
    for window in repair.H2_WINDOWS:
        event_n = rng.integers(0, window + 1, size=rows).astype(np.float64)
        raw[f"w{window}_event_n"] = event_n
        raw[f"w{window}_positive_n"] = np.floor(event_n * latent)
    labels = (rng.random(rows) < np.clip(0.12 + 0.62 * latent, 0.02, 0.95)).astype(np.int8)
    return {
        "categorical": categorical,
        "static_continuous": static_continuous,
        "static_binary": static_binary,
        "raw": raw,
        "labels": labels,
        "latent": latent,
    }


def build_design(rows: int = 900, seed: int = 7):
    blocks = synthetic_blocks(rows, seed=seed)
    prevalence = float(blocks["labels"].mean())
    h2_continuous, h2_binary = repair.derive_h2_blocks(blocks["raw"], prevalence)
    design, bl1, bl2 = repair.fit_grouped_design(
        categorical=blocks["categorical"],
        static_continuous=blocks["static_continuous"],
        static_binary=blocks["static_binary"],
        h2_continuous=h2_continuous,
        h2_binary=h2_binary,
        prevalence=prevalence,
    )
    return blocks, design, bl1, bl2


class PreprocessingTests(unittest.TestCase):
    def test_grouped_design_widths_and_prefix_invariant(self):
        blocks, design, bl1, bl2 = build_design()
        self.assertEqual(
            design.bl1_width,
            design.categorical_width
            + len(repair.STATIC_CONTINUOUS_FIELDS)
            + len(repair.STATIC_BINARY_FIELDS),
        )
        self.assertEqual(
            design.bl2_width,
            design.bl1_width + len(repair.H2_CONTINUOUS_FIELDS) + len(repair.H2_BINARY_FIELDS),
        )
        self.assertEqual(bl1.shape[1], design.bl1_width)
        self.assertEqual(bl2.shape[1], design.bl2_width)
        repair.assert_column_prefix(bl1, bl2)

    def test_binary_columns_are_passthrough_and_continuous_are_centered(self):
        blocks, design, bl1, bl2 = build_design()
        dense = np.asarray(bl2.todense())
        start = design.categorical_width
        continuous = dense[:, start : start + design.static_continuous_width]
        binary = dense[
            :,
            start
            + design.static_continuous_width : start
            + design.static_continuous_width
            + design.static_binary_width,
        ]
        # continuous block is standardised on the fit rows
        self.assertTrue(np.allclose(continuous.mean(axis=0), 0.0, atol=1e-4))
        self.assertTrue(np.allclose(continuous.std(axis=0), 1.0, atol=1e-3))
        # binary block is untouched
        self.assertTrue(np.isin(binary, (0.0, 1.0)).all())
        self.assertTrue(np.allclose(binary, blocks["static_binary"], atol=0))

    def test_transform_only_path_reuses_fitted_statistics(self):
        blocks, design, bl1_fit, bl2_fit = build_design()
        h2_continuous, h2_binary = repair.derive_h2_blocks(blocks["raw"], design.prevalence)
        bl1_again, bl2_again = repair.transform_grouped(
            design,
            categorical=blocks["categorical"],
            static_continuous=blocks["static_continuous"],
            static_binary=blocks["static_binary"],
            h2_continuous=h2_continuous,
            h2_binary=h2_binary,
        )
        self.assertEqual((bl1_again != bl1_fit).nnz, 0)
        self.assertEqual((bl2_again != bl2_fit).nnz, 0)

    def test_unseen_category_goes_to_infrequent_not_error(self):
        blocks, design, _, _ = build_design()
        h2_continuous, h2_binary = repair.derive_h2_blocks(blocks["raw"], design.prevalence)
        unseen = blocks["categorical"].copy()
        unseen[:, 1] = 10_000  # a category never observed on the fit rows
        bl1, bl2 = repair.transform_grouped(
            design,
            categorical=unseen,
            static_continuous=blocks["static_continuous"],
            static_binary=blocks["static_binary"],
            h2_continuous=h2_continuous,
            h2_binary=h2_binary,
        )
        self.assertEqual(bl2.shape[1], design.bl2_width)
        repair.assert_column_prefix(bl1, bl2)

    def test_continuous_clip_and_row_norm_checks(self):
        blocks, design, bl1, bl2 = build_design()
        stats = repair.numeric_hard_checks(bl2, "BL2")
        self.assertLessEqual(stats["absolute_maximum"], repair.CONTINUOUS_CLIP + 1e-6)
        self.assertLessEqual(stats["row_l2_max"], repair.MAX_ROW_L2_NORM)
        self.assertLessEqual(stats["row_l2_p99"], repair.MAX_P99_ROW_L2_NORM)

        # an extreme transform-time outlier must be clipped, not amplified
        h2_continuous, h2_binary = repair.derive_h2_blocks(blocks["raw"], design.prevalence)
        wild = blocks["static_continuous"].copy()
        wild[0, :] = 1e9
        _, bl2_wild = repair.transform_grouped(
            design,
            categorical=blocks["categorical"],
            static_continuous=wild,
            static_binary=blocks["static_binary"],
            h2_continuous=h2_continuous,
            h2_binary=h2_binary,
        )
        self.assertLessEqual(float(np.abs(bl2_wild.data).max()), repair.CONTINUOUS_CLIP + 1e-6)

    def test_row_norm_cap_is_enforced(self):
        wide = sparse.csr_matrix(np.full((10, 200), repair.CONTINUOUS_CLIP, dtype=np.float32))
        with self.assertRaises(repair.ContractViolation):
            repair.numeric_hard_checks(wide, "synthetic wide matrix")

    def test_binary_domain_violation_is_rejected(self):
        blocks = synthetic_blocks(200)
        prevalence = float(blocks["labels"].mean())
        h2_continuous, h2_binary = repair.derive_h2_blocks(blocks["raw"], prevalence)
        bad_binary = blocks["static_binary"].copy()
        bad_binary[0, 0] = 2.0
        with self.assertRaises(repair.ContractViolation):
            repair.fit_grouped_design(
                categorical=blocks["categorical"],
                static_continuous=blocks["static_continuous"],
                static_binary=bad_binary,
                h2_continuous=h2_continuous,
                h2_binary=h2_binary,
                prevalence=prevalence,
            )

    def test_h2_smoothing_uses_only_the_fit_prevalence(self):
        blocks = synthetic_blocks(300)
        zero_history = {name: np.zeros(3) for name in blocks["raw"]}
        continuous, binary = repair.derive_h2_blocks(zero_history, 0.31)
        rate_index = repair.H2_CONTINUOUS_FIELDS.index("smoothed_lifetime_long_view_rate")
        self.assertTrue(np.allclose(continuous[:, rate_index], 0.31))
        self.assertTrue(np.allclose(binary, 0.0))


class EstimatorTests(unittest.TestCase):
    def test_sgd_parameters_match_the_contract(self):
        contract = load_contract()
        declared = contract["models"]["shared_sparse_linear_estimator"]
        blocks, design, bl1, bl2 = build_design(rows=400)
        model, _ = repair.fit_sgd(bl2, blocks["labels"], alpha=1e-4, eta0=1e-3)
        params = model.get_params()
        self.assertEqual(params["loss"], declared["loss"])
        self.assertEqual(params["penalty"], declared["penalty"])
        self.assertEqual(params["learning_rate"], declared["learning_rate"])
        self.assertEqual(params["max_iter"], declared["max_iter"])
        self.assertEqual(params["tol"], declared["tol"])
        self.assertEqual(params["n_iter_no_change"], declared["n_iter_no_change"])
        self.assertEqual(params["random_state"], declared["random_state"])
        self.assertEqual(params["average"], declared["average"])
        self.assertEqual(params["early_stopping"], declared["early_stopping"])
        self.assertIsNone(params["class_weight"])
        self.assertFalse(params["warm_start"])
        self.assertEqual(params["l1_ratio"], declared["l1_ratio_inactive_but_explicit"])
        self.assertEqual(params["power_t"], declared["power_t_inactive_but_explicit"])
        self.assertEqual(params["epsilon"], declared["epsilon_inactive_but_explicit"])
        self.assertEqual(list(repair.SGD_ALPHA_VALUES), declared["alpha_values"])
        self.assertEqual(list(repair.SGD_ETA0_VALUES), declared["eta0_values"])

    def test_reference_C_mapping_reproduces_the_same_minimiser(self):
        blocks, design, bl1, bl2 = build_design(rows=600)
        y = blocks["labels"]
        alpha = 1e-3
        _, reference_record, C = repair.fit_reference(bl2, y, alpha=alpha)
        self.assertAlmostEqual(C, repair.reference_C(bl2.shape[0], alpha))
        self.assertAlmostEqual(1.0 / (bl2.shape[0] * C), alpha)

        # the reference must be at least as good as any SGD solution
        for eta0 in repair.SGD_ETA0_VALUES:
            _, sgd_record = repair.fit_sgd(bl2, y, alpha=alpha, eta0=eta0)
            self.assertGreaterEqual(
                sgd_record.objective,
                reference_record.objective - repair.REFERENCE_ABOVE_SGD_TOLERANCE,
                msg=f"reference should minimise the objective (eta0={eta0})",
            )

    def test_objective_matches_an_independent_computation(self):
        blocks, design, bl1, bl2 = build_design(rows=350)
        y = blocks["labels"].astype(np.float64)
        alpha = 1e-3
        model, record = repair.fit_sgd(bl2, blocks["labels"], alpha=alpha, eta0=1e-2)
        coefficient = np.asarray(model.coef_, dtype=np.float64).ravel()
        intercept = float(np.asarray(model.intercept_).ravel()[0])
        score = bl2 @ coefficient + intercept
        probability = 1.0 / (1.0 + np.exp(-score))
        manual = float(
            -np.mean(y * np.log(probability) + (1 - y) * np.log1p(-probability))
        ) + alpha * 0.5 * float(np.dot(coefficient, coefficient))
        self.assertAlmostEqual(record.objective, manual, places=10)

    def test_nonconverged_fit_is_flagged_and_never_silently_accepted(self):
        blocks, design, bl1, bl2 = build_design(rows=500)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            starved = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                learning_rate="adaptive",
                eta0=1e-3,
                max_iter=1,
                tol=1e-12,
                n_iter_no_change=5,
                random_state=repair.SEED,
                average=True,
            )
            starved.fit(bl2, blocks["labels"])
        n_iter = int(np.asarray(starved.n_iter_).ravel()[0])
        self.assertGreaterEqual(n_iter, starved.max_iter)
        self.assertTrue(any(issubclass(item.category, Warning) for item in caught))

        # a finite output must not override the non-convergence verdict
        healthy_model, healthy = repair.fit_sgd(bl2, blocks["labels"], alpha=1e-3, eta0=1e-2)
        self.assertTrue(np.isfinite(healthy.objective))
        self.assertEqual(healthy.converged, healthy.convergence_warning_count == 0 and healthy.n_iter < healthy.max_iter)

    def test_adequacy_gate_accepts_the_reference_and_rejects_a_starved_fit(self):
        blocks, design, bl1, bl2 = build_design(rows=700)
        y = blocks["labels"]
        alpha = 1e-3
        _, reference_record, _ = repair.fit_reference(bl2, y, alpha=alpha)

        good = repair.adequacy_decision(
            reference_record.objective, reference_record.objective, reference_converged=True
        )
        self.assertTrue(good["adequacy_passed"])
        self.assertAlmostEqual(good["objective_regret"], 0.0)

        starved = repair.adequacy_decision(
            reference_record.objective + 1.0,
            reference_record.objective,
            reference_converged=True,
        )
        self.assertFalse(starved["adequacy_passed"])
        self.assertGreater(starved["objective_regret"], starved["maximum_allowed_regret"])

        # a non-converged reference invalidates the comparison entirely
        unusable = repair.adequacy_decision(
            reference_record.objective, reference_record.objective, reference_converged=False
        )
        self.assertFalse(unusable["adequacy_passed"])

        # the reference sitting below the SGD objective is normal; sitting above
        # it beyond tolerance means the pairing is wrong
        inverted = repair.adequacy_decision(
            reference_record.objective - 1.0,
            reference_record.objective,
            reference_converged=True,
        )
        self.assertTrue(inverted["reference_above_SGD"])
        self.assertFalse(inverted["adequacy_passed"])

    def test_regret_tolerance_formula(self):
        self.assertEqual(repair.maximum_allowed_regret(0.0), repair.REGRET_ABSOLUTE_FLOOR)
        self.assertAlmostEqual(repair.maximum_allowed_regret(0.6), 0.003)
        self.assertAlmostEqual(repair.maximum_allowed_regret(-0.6), 0.003)


class CalibrationTests(unittest.TestCase):
    def _score_and_labels(self, rows: int = 4000, seed: int = 3):
        rng = np.random.default_rng(seed)
        score = rng.normal(scale=2.5, size=rows)
        probability = 1.0 / (1.0 + np.exp(-(0.8 * score - 0.7)))
        labels = (rng.random(rows) < probability).astype(np.int8)
        return score, labels

    def test_platt_scaling_is_order_preserving_and_improves_probability_quality(self):
        score, labels = self._score_and_labels()
        calibrator = repair.fit_previous_day_sigmoid(score, labels)
        self.assertGreater(calibrator.slope, 0.0)
        calibrated = calibrator.apply(score)
        repair.assert_calibration_monotone(score, calibrated)

        # calibration is monotone, so ranking is untouched
        from kuairand_longseq.evaluation.gate2b_metrics import binary_auc_midrank

        self.assertAlmostEqual(
            binary_auc_midrank(labels, score),
            binary_auc_midrank(labels, calibrated),
            places=12,
        )

        # a raw score read as a probability is badly calibrated; Platt fixes it
        naive = repair.metric_clip(score)
        base_rate = np.full_like(calibrated, labels.mean())
        def brier(p):
            return float(np.mean(np.square(np.asarray(p) - labels)))

        self.assertLess(brier(calibrated), brier(naive))
        self.assertLess(brier(calibrated), brier(base_rate))

    def test_calibration_rejects_degenerate_inputs(self):
        score, labels = self._score_and_labels(rows=500)
        with self.assertRaises(repair.ContractViolation):
            repair.fit_previous_day_sigmoid(score, np.ones_like(labels))
        with self.assertRaises(repair.ContractViolation):
            repair.fit_previous_day_sigmoid(np.zeros_like(score), labels)
        with self.assertRaises(repair.ContractViolation):
            repair.fit_previous_day_sigmoid(score[:10], labels)

    def test_negative_slope_is_rejected(self):
        score, labels = self._score_and_labels(rows=2000)
        with self.assertRaises(repair.ContractViolation):
            repair.fit_previous_day_sigmoid(-score, labels)

    def test_monotonicity_check_catches_a_corrupted_column(self):
        score, labels = self._score_and_labels(rows=800)
        calibrator = repair.fit_previous_day_sigmoid(score, labels)
        calibrated = calibrator.apply(score)
        corrupted = calibrated.copy()
        order = np.argsort(score, kind="mergesort")
        corrupted[order[0]], corrupted[order[-1]] = corrupted[order[-1]], corrupted[order[0]]
        with self.assertRaises(repair.ContractViolation):
            repair.assert_calibration_monotone(score, corrupted)

    def test_metric_clip_is_narrow_enough_to_preserve_ranking(self):
        self.assertEqual(repair.METRIC_CLIP_LOW, 1e-15)
        contract = load_contract()
        declared = contract["calibration"]["metric_only_numerical_clip"]
        self.assertAlmostEqual(declared[0], repair.METRIC_CLIP_LOW)
        self.assertAlmostEqual(declared[1], repair.METRIC_CLIP_HIGH)
        self.assertEqual(contract["calibration"]["probability_storage_clip"], "none")


class ContractBindingTests(unittest.TestCase):
    def test_constants_match_the_frozen_contract(self):
        contract = load_contract()
        self.assertEqual(contract["preprocessing"]["preprocessing_id"], repair.PREPROCESSING_ID)
        self.assertEqual(contract["calibration"]["calibration_id"], repair.CALIBRATION_ID)

        categorical = contract["preprocessing"]["categorical"]
        self.assertEqual(categorical["minimum_frequency"], repair.CATEGORICAL_MIN_FREQUENCY)
        self.assertEqual(
            categorical["maximum_categories_per_input_field"], repair.CATEGORICAL_MAX_CATEGORIES
        )
        continuous = contract["preprocessing"]["continuous_and_rate"]
        self.assertTrue(continuous["with_mean"])
        self.assertTrue(continuous["with_std"])
        self.assertEqual(continuous["post_transform_clip"], [-repair.CONTINUOUS_CLIP, repair.CONTINUOUS_CLIP])
        checks = contract["preprocessing"]["hard_numeric_checks_per_origin_and_bundle"]
        self.assertEqual(checks["maximum_row_l2_norm"], repair.MAX_ROW_L2_NORM)
        self.assertEqual(checks["maximum_p99_row_l2_norm"], repair.MAX_P99_ROW_L2_NORM)
        self.assertEqual(
            checks["transformed_continuous_absolute_maximum"], repair.CONTINUOUS_CLIP
        )

        adequacy = contract["optimization_adequacy"]["maximum_allowed_SGD_objective_regret"]
        self.assertEqual(adequacy["absolute_floor"], repair.REGRET_ABSOLUTE_FLOOR)
        self.assertEqual(adequacy["relative_fraction"], repair.REGRET_RELATIVE_FRACTION)
        self.assertEqual(
            contract["optimization_adequacy"]["reference_above_SGD_numerical_tolerance"],
            repair.REFERENCE_ABOVE_SGD_TOLERANCE,
        )

        reference = contract["models"]["diagnostic_reference_solver"]
        self.assertEqual(reference["max_iter"], repair.REFERENCE_MAX_ITER)
        self.assertEqual(reference["tol"], repair.REFERENCE_TOL)
        calibration_method = contract["calibration"]["method"]
        self.assertEqual(calibration_method["C"], repair.CALIBRATOR_C)
        self.assertEqual(calibration_method["max_iter"], repair.CALIBRATOR_MAX_ITER)
        self.assertEqual(calibration_method["tol"], repair.CALIBRATOR_TOL)

    def test_field_lists_match_the_contract(self):
        contract = load_contract()
        semantics = contract["feature_semantics"]
        self.assertEqual(list(repair.CATEGORICAL_FIELDS), semantics["categorical_fields"])
        self.assertEqual(
            list(repair.STATIC_CONTINUOUS_FIELDS), semantics["static_continuous_fields"]
        )
        self.assertEqual(list(repair.STATIC_BINARY_FIELDS), semantics["static_binary_fields"])
        self.assertEqual(
            list(repair.H2_CONTINUOUS_FIELDS),
            semantics["H2_derived_continuous_or_rate_fields"],
        )
        self.assertEqual(list(repair.H2_BINARY_FIELDS), semantics["H2_derived_binary_fields"])
        self.assertEqual(
            semantics["H2_derivation"]["smoothed_rate_prior_strength"],
            repair.H2_SMOOTHING_PRIOR_STRENGTH,
        )

    def test_paired_registry_matches_the_contract_exactly(self):
        contract = load_contract()
        declared = contract["search_and_selection"]["paired_configuration_registry"][
            "configurations"
        ]
        produced = repair.paired_configurations()
        self.assertEqual(len(produced), len(declared))
        for expected, actual in zip(declared, produced):
            self.assertEqual(expected["pair_id"], actual["pair_id"])
            self.assertAlmostEqual(expected["alpha"], actual["alpha"])
            self.assertAlmostEqual(expected["eta0"], actual["eta0"])

    def test_no_denylisted_predictor_appears_in_the_model_field_lists(self):
        contract = load_contract()
        denylist = {
            str(item) for item in contract["feature_semantics"]["hard_predictor_denylist"]
        }
        used = (
            set(repair.CATEGORICAL_FIELDS)
            | set(repair.STATIC_CONTINUOUS_FIELDS)
            | set(repair.STATIC_BINARY_FIELDS)
            | set(repair.H2_CONTINUOUS_FIELDS)
            | set(repair.H2_BINARY_FIELDS)
        )
        self.assertEqual(used & denylist, set())


if __name__ == "__main__":
    unittest.main()
