"""Synthetic tests for the Gate 2B v003 runner orchestration.

These exercise the decision logic that the contract makes load-bearing:
selection that may see BL1 only, fail-closed budget caps, split validation, and
the refusal to release without a versioned approval.  All data is built in
memory; nothing here reads the frozen feature artifact or any protected table.
"""

import csv
import hashlib
import importlib.util
import json
import sys
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml
from scipy import sparse
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "gate2b_repair_runner_v003",
    PROJECT_ROOT / "scripts/run_gate2b_probability_repair_v003.py",
)
runner = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# dataclasses resolve annotations through sys.modules, so register first
sys.modules[_spec.name] = runner
_spec.loader.exec_module(runner)

CONTRACT_PATH = PROJECT_ROOT / "configs/gate2b_probability_repair_contract_v003.yaml"


def load_contract() -> dict:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


class StrictSerializationTests(unittest.TestCase):
    def test_json_round_trip_is_rfc8259_strict_and_preserves_special_values(self):
        with TemporaryDirectory() as raw:
            path = Path(raw) / "strict.json"
            runner.write_json(
                path,
                {
                    "numpy_integer": np.int64(7),
                    "numpy_float": np.float32(0.25),
                    "positive_infinity": np.inf,
                    "negative_infinity": -np.inf,
                    "not_a_number": np.nan,
                    "dtype_class": np.float32,
                    "path": Path("synthetic") / "artifact.json",
                },
            )
            raw_text = path.read_text(encoding="utf-8")

            def reject_nonstandard_constant(token):
                raise AssertionError(f"non-RFC-8259 constant emitted: {token}")

            payload = json.loads(raw_text, parse_constant=reject_nonstandard_constant)
            self.assertEqual(payload["numpy_integer"], 7)
            self.assertAlmostEqual(payload["numpy_float"], 0.25)
            self.assertEqual(payload["positive_infinity"], "numpy_positive_infinity")
            self.assertEqual(payload["negative_infinity"], "numpy_negative_infinity")
            self.assertEqual(payload["not_a_number"], "numpy_nan")
            self.assertEqual(payload["dtype_class"], "numpy.float32")
            self.assertEqual(payload["path"], str(Path("synthetic") / "artifact.json"))

    def test_csv_round_trip_keeps_the_union_of_fields_from_every_row(self):
        with TemporaryDirectory() as raw:
            path = Path(raw) / "ledger.csv"
            runner.write_csv(
                path,
                [
                    {"fit_run_id": "first", "status": "complete", "later_only": None},
                    {"fit_run_id": "second", "status": "failed", "later_only": "evidence"},
                ],
            )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                list(rows[0]), ["fit_run_id", "status", "later_only"]
            )
            self.assertEqual(rows[0]["later_only"], "")
            self.assertEqual(rows[1]["later_only"], "evidence")
            self.assertEqual([row["fit_run_id"] for row in rows], ["first", "second"])


class SelectionTests(unittest.TestCase):
    """Contract: primary_shared_configuration_selection."""

    def _pooled(self, bl1: dict[str, dict[str, float]], bl2: dict[str, dict[str, float]]):
        pooled = {}
        for pair, values in bl1.items():
            pooled[("BL1", pair)] = values
        for pair, values in bl2.items():
            pooled[("BL2", pair)] = values
        return pooled

    def test_selection_follows_the_declared_metric_order(self):
        eligible = [
            {"pair_id": "A1em04_E1em03", "alpha": 1e-4, "eta0": 1e-3},
            {"pair_id": "A1em03_E1em02", "alpha": 1e-3, "eta0": 1e-2},
        ]
        pooled = self._pooled(
            bl1={
                "A1em04_E1em03": {
                    "log_loss": 0.60,
                    "brier": 0.21,
                    "average_precision": 0.33,
                    "user_gauc_event_weighted": 0.52,
                },
                # better log loss must win even with a worse average precision
                "A1em03_E1em02": {
                    "log_loss": 0.55,
                    "brier": 0.20,
                    "average_precision": 0.31,
                    "user_gauc_event_weighted": 0.50,
                },
            },
            bl2={},
        )
        chosen = runner._select_shared_configuration(eligible, pooled)
        self.assertEqual(chosen["pair_id"], "A1em03_E1em02")

    def test_selection_never_consults_BL2(self):
        eligible = [
            {"pair_id": "A1em04_E1em03", "alpha": 1e-4, "eta0": 1e-3},
            {"pair_id": "A1em03_E1em02", "alpha": 1e-3, "eta0": 1e-2},
        ]
        bl1 = {
            "A1em04_E1em03": {
                "log_loss": 0.55,
                "brier": 0.20,
                "average_precision": 0.31,
                "user_gauc_event_weighted": 0.50,
            },
            "A1em03_E1em02": {
                "log_loss": 0.60,
                "brier": 0.21,
                "average_precision": 0.33,
                "user_gauc_event_weighted": 0.52,
            },
        }
        # BL2 strongly prefers the other configuration; it must be ignored
        bl2_favouring_the_loser = {
            "A1em04_E1em03": {
                "log_loss": 0.90,
                "brier": 0.40,
                "average_precision": 0.10,
                "user_gauc_event_weighted": 0.40,
            },
            "A1em03_E1em02": {
                "log_loss": 0.10,
                "brier": 0.05,
                "average_precision": 0.90,
                "user_gauc_event_weighted": 0.90,
            },
        }
        chosen = runner._select_shared_configuration(
            eligible, self._pooled(bl1, bl2_favouring_the_loser)
        )
        self.assertEqual(chosen["pair_id"], "A1em04_E1em03")

    def test_exact_ties_fall_back_to_larger_alpha_then_smaller_eta0_then_pair_id(self):
        identical = {
            "log_loss": 0.6,
            "brier": 0.21,
            "average_precision": 0.33,
            "user_gauc_event_weighted": 0.52,
        }
        eligible = [
            {"pair_id": "A1em04_E1em03", "alpha": 1e-4, "eta0": 1e-3},
            {"pair_id": "A1em03_E1em03", "alpha": 1e-3, "eta0": 1e-3},
            {"pair_id": "A1em03_E1em02", "alpha": 1e-3, "eta0": 1e-2},
        ]
        pooled = self._pooled({item["pair_id"]: dict(identical) for item in eligible}, {})
        chosen = runner._select_shared_configuration(eligible, pooled)
        # larger alpha wins first, then the smaller eta0 among the survivors
        self.assertEqual(chosen["alpha"], 1e-3)
        self.assertEqual(chosen["eta0"], 1e-3)
        self.assertEqual(chosen["pair_id"], "A1em03_E1em03")

    def test_selection_is_deterministic_under_input_reordering(self):
        values = {
            "A1em04_E1em03": {
                "log_loss": 0.60,
                "brier": 0.21,
                "average_precision": 0.33,
                "user_gauc_event_weighted": 0.52,
            },
            "A1em03_E1em02": {
                "log_loss": 0.60,
                "brier": 0.21,
                "average_precision": 0.33,
                "user_gauc_event_weighted": 0.52,
            },
        }
        eligible = [
            {"pair_id": "A1em04_E1em03", "alpha": 1e-4, "eta0": 1e-3},
            {"pair_id": "A1em03_E1em02", "alpha": 1e-3, "eta0": 1e-2},
        ]
        first = runner._select_shared_configuration(list(eligible), self._pooled(values, {}))
        second = runner._select_shared_configuration(
            list(reversed(eligible)), self._pooled(values, {})
        )
        self.assertEqual(first["pair_id"], second["pair_id"])


class BudgetTests(unittest.TestCase):
    def _contract(self, **overrides):
        contract = load_contract()
        contract["operational_budget"].update(overrides)
        return contract

    def test_total_fit_operation_cap_is_enforced(self):
        budget = runner.Budget(contract=self._contract(maximum_total_fit_operations=2))
        budget.run("sgd", "a", lambda: None, stage="search")
        budget.run("sgd", "b", lambda: None, stage="search")
        with self.assertRaises(runner.ContractStop):
            budget.run("sgd", "c", lambda: None, stage="search")

    def test_per_fit_second_cap_is_enforced(self):
        import time as _time

        budget = runner.Budget(contract=self._contract(maximum_SGD_seconds_per_fit=0.001))
        with self.assertRaises(runner.ContractStop):
            budget.run("sgd", "slow", lambda: _time.sleep(0.05), stage="search")

    def test_failed_fits_still_consume_budget(self):
        budget = runner.Budget(contract=self._contract())

        def boom():
            raise ValueError("synthetic failure")

        with self.assertRaises(ValueError):
            budget.run("sgd", "boom", boom, stage="search")
        self.assertEqual(budget.counts["sgd"], 1)
        self.assertEqual(budget.stage_counts["search:sgd"], 1)
        self.assertEqual(len(budget.ledger), 1)
        self.assertTrue(budget.ledger[0]["status"].startswith("failed:"))

    def test_pre_fit_stage_cap_stops_without_executing_the_thunk(self):
        contract = self._contract(maximum_search_SGD_fit_runs=0)
        budget = runner.Budget(contract=contract)
        calls = {"count": 0}

        def must_not_run():
            calls["count"] += 1

        with self.assertRaises(runner.ContractStop) as caught:
            budget.run("sgd", "over_cap", must_not_run, stage="search")
        self.assertIn("next search:sgd fit would breach", str(caught.exception))
        self.assertEqual(calls["count"], 0)
        self.assertEqual(budget.counts, {})
        self.assertEqual(budget.stage_counts, {})
        self.assertEqual(budget.ledger, [])

    def test_post_fit_storage_cap_keeps_the_consumed_fit_in_the_ledger(self):
        contract = self._contract(maximum_artifact_storage_gb=0.0)
        budget = runner.Budget(contract=contract)
        with patch.object(runner, "managed_artifact_size_bytes", return_value=1):
            with self.assertRaises(runner.ContractStop) as caught:
                budget.run("sgd", "storage_overrun", lambda: "fit-result", stage="search")
        self.assertIn("resource cap breach after storage_overrun", str(caught.exception))
        self.assertIn("storage=", str(caught.exception))
        self.assertEqual(budget.counts["sgd"], 1)
        self.assertEqual(budget.stage_counts["search:sgd"], 1)
        self.assertEqual(len(budget.ledger), 1)
        self.assertEqual(budget.ledger[0]["fit_run_id"], "storage_overrun")

    def test_diagnostic_budget_uses_all_declared_slots_not_only_executed_fits(self):
        contract = self._contract()
        diagnostic = contract["probability_diagnostics"][
            "assessment_calibration_regression"
        ]
        self.assertEqual(diagnostic["declared_slots"], 24)
        self.assertEqual(
            diagnostic["declared_slots"],
            contract["operational_budget"][
                "maximum_diagnostic_calibration_regression_fits"
            ],
        )
        self.assertEqual(diagnostic["daily_origin_fits"] + diagnostic["pooled_fits"], 24)

    def test_budget_arithmetic_detects_a_broken_reconciliation(self):
        contract = self._contract(maximum_total_fit_operations=999)
        with self.assertRaises(runner.ContractStop):
            runner.verify_budget_arithmetic(contract)

    def test_budget_arithmetic_passes_on_the_real_contract(self):
        reconciliation = runner.verify_budget_arithmetic(load_contract())
        self.assertEqual(reconciliation["primary"], 76)
        self.assertEqual(reconciliation["reference"], 20)
        self.assertEqual(reconciliation["total"], 120)


class SplitValidationTests(unittest.TestCase):
    """Contract: temporal_protocol, verified against a synthetic frame."""

    def _frame(self, rows_per_day: int = 6, days: int = 4) -> runner.Frame:
        dates: list[np.datetime64] = []
        for offset in range(days):
            dates.extend([np.datetime64("2022-04-08", "D") + offset] * rows_per_day)
        size = len(dates)
        rng = np.random.default_rng(11)
        columns = {name: np.zeros(size, dtype=np.float64) for name in runner.REQUIRED_COLUMNS}
        columns["event_date"] = np.array(dates, dtype="datetime64[D]")
        columns["long_view"] = np.tile(np.array([0, 1], dtype=np.int64), size // 2)
        columns["user_id"] = rng.integers(0, 3, size=size).astype(np.int64)
        columns["source_row_number"] = np.arange(size, dtype=np.int64)
        columns["source_table"] = np.array(["early_standard"] * size)
        return runner.Frame(columns=columns)

    def _contract(self, expected_rows: int, expected_users: int, expected_positives: int) -> dict:
        return {
            "temporal_protocol": {
                "calibration_minimum_requirements": {
                    "minimum_rows": 1,
                    "minimum_users": 1,
                    "minimum_positives": 1,
                    "minimum_negatives": 1,
                },
                "origin_splits": [
                    {
                        "origin": "2022-04-11",
                        "estimator_fit_date_range_inclusive": ["2022-04-08", "2022-04-09"],
                        "calibration_date": "2022-04-10",
                        "calibration_expected": {
                            "rows": expected_rows,
                            "users": expected_users,
                            "positives": expected_positives,
                        },
                        "assessment_date": "2022-04-11",
                        "assessment_expected": {
                            "rows": 6,
                            "users": int(np.unique(self._frame().columns["user_id"][18:24]).size),
                            "positives": 3,
                        },
                    }
                ],
            }
        }

    def test_correct_counts_pass_and_groups_do_not_overlap(self):
        frame = self._frame()
        users = int(np.unique(frame.columns["user_id"][12:18]).size)
        contract = self._contract(6, users, 3)
        splits = runner.build_splits(contract, frame)
        self.assertEqual(len(splits), 1)
        split = splits[0]
        self.assertEqual(split.fit_index.size, 12)
        self.assertEqual(split.calibration_index.size, 6)
        self.assertEqual(split.assessment_index.size, 6)
        for left, right in (
            (split.fit_index, split.calibration_index),
            (split.fit_index, split.assessment_index),
            (split.calibration_index, split.assessment_index),
        ):
            self.assertEqual(np.intersect1d(left, right).size, 0)

    def test_a_single_wrong_expected_count_stops_before_any_fit(self):
        frame = self._frame()
        users = int(np.unique(frame.columns["user_id"][12:18]).size)
        contract = self._contract(7, users, 3)  # rows deliberately off by one
        with self.assertRaises(runner.ContractStop):
            runner.build_splits(contract, frame)

    def test_bl0_uses_every_row_strictly_before_the_origin(self):
        frame = self._frame()
        users = int(np.unique(frame.columns["user_id"][12:18]).size)
        split = runner.build_splits(self._contract(6, users, 3), frame)[0]
        before = frame.labels()[:18]
        self.assertAlmostEqual(split.bl0_probability, float(before.mean()))
        # BL0 spans the calibration day as well, which fit_prevalence does not
        self.assertAlmostEqual(split.fit_prevalence, float(frame.labels()[:12].mean()))


class ReleaseGuardTests(unittest.TestCase):
    def test_release_refuses_an_unapproved_hash(self):
        with self.assertRaises(runner.ContractStop) as caught:
            runner.release("0" * 64)
        self.assertIn("hash mismatch", str(caught.exception))

    def test_release_refuses_while_execution_is_unauthorized(self):
        contract, contract_hash = runner.load_contract()
        self.assertFalse(contract["authorization"]["execution_authorized"])
        with self.assertRaises(runner.ContractStop) as caught:
            runner.release(contract_hash)
        self.assertIn("execution_authorized", str(caught.exception))

    def test_main_returns_nonzero_on_a_stop(self):
        argv = sys.argv
        try:
            sys.argv = ["runner", "--release", "--approve-contract-hash", "0" * 64]
            self.assertEqual(runner.main(), 2)
        finally:
            sys.argv = argv

    def test_validate_only_never_requires_approval(self):
        argv = sys.argv
        try:
            sys.argv = ["runner", "--validate-only"]
            self.assertEqual(runner.main(), 0)
        finally:
            sys.argv = argv


class ImplementationIdentityTests(unittest.TestCase):
    def test_implementation_hash_mismatch_fails_closed_before_execution(self):
        contract = load_contract()
        contract["implementation_status"]["result_producing_implementation"]["files"][0][
            "sha256"
        ] = "0" * 64
        with self.assertRaises(runner.ContractStop) as caught:
            runner.verify_implementation_hashes(contract)
        message = str(caught.exception)
        self.assertIn("implementation SHA-256 mismatch", message)
        self.assertIn("scripts/run_gate2b_probability_repair_v003.py", message)


class NonconvergenceRoutingTests(unittest.TestCase):
    class _Model:
        def get_params(self, deep=True):
            return {"synthetic_nonconverged_model": True}

        def decision_function(self, matrix):
            return np.zeros(matrix.shape[0], dtype=np.float64)

        def predict_proba(self, matrix):
            return np.full((matrix.shape[0], 2), 0.5, dtype=np.float64)

    @staticmethod
    def _origin():
        matrix = sparse.csr_matrix(np.ones((6, 2), dtype=np.float64))
        split = SimpleNamespace(origin="2022-04-11", calibration_date="2022-04-10")
        return SimpleNamespace(
            split=split,
            matrices={
                "fit": {"BL1": matrix, "BL2": matrix},
                "calibration": {"BL1": matrix, "BL2": matrix},
                "assessment": {"BL1": matrix, "BL2": matrix},
            },
            labels={
                "fit": np.array([0, 1, 0, 1, 0, 1], dtype=np.int8),
                "calibration": np.array([0, 1, 0, 1, 0, 1], dtype=np.int8),
                "assessment": np.array([0, 1, 0, 1, 0, 1], dtype=np.int8),
            },
            users={
                "fit": np.arange(6),
                "calibration": np.arange(6),
                "assessment": np.arange(6),
            },
        )

    @staticmethod
    def _nonconverged_record():
        return runner.repair.FitRecord(
            converged=False,
            n_iter=runner.repair.SGD_MAX_ITER,
            max_iter=runner.repair.SGD_MAX_ITER,
            convergence_warning_count=1,
            coefficient_l1_norm=0.0,
            coefficient_l2_norm=0.0,
            coefficient_absolute_maximum=0.0,
            intercept=0.0,
            objective=0.7,
        )

    def test_search_nonconvergence_is_recorded_ineligible_and_does_not_stop_grid(self):
        budget = runner.Budget(contract=load_contract())
        book = runner.LedgerBook()
        with patch.object(
            runner.repair,
            "fit_sgd",
            return_value=(self._Model(), self._nonconverged_record()),
        ):
            component = runner.fit_component(
                budget,
                book,
                self._origin(),
                model_id="BL1",
                alpha=1e-4,
                eta0=1e-3,
                stage="search",
            )

        self.assertFalse(component.sgd_record.converged)
        self.assertEqual(component.failure_reason, "SGD_nonconvergence")
        self.assertEqual(book.convergence[-1]["status"], "nonconverged")
        self.assertEqual(budget.stage_counts["search:sgd"], 1)

    def test_frozen_daily_nonconvergence_has_its_own_immediate_terminal_state(self):
        budget = runner.Budget(contract=load_contract())
        book = runner.LedgerBook()
        with patch.object(
            runner.repair,
            "fit_sgd",
            return_value=(self._Model(), self._nonconverged_record()),
        ):
            with self.assertRaises(runner.TerminalStop) as caught:
                runner.fit_component(
                    budget,
                    book,
                    self._origin(),
                    model_id="BL1",
                    alpha=1e-4,
                    eta0=1e-3,
                    stage="frozen_daily",
                )

        self.assertEqual(
            caught.exception.state,
            "frozen_daily_component_convergence_failure",
        )
        self.assertEqual(
            caught.exception.exit_code,
            runner.EXIT_FROZEN_DAILY_COMPONENT_CONVERGENCE_FAILURE,
        )
        self.assertEqual(book.convergence[-1]["status"], "nonconverged")


class AssessmentCalibrationRegressionTests(unittest.TestCase):
    def test_primary_calibrator_does_not_read_assessment_AP(self):
        rng = np.random.default_rng(20260816)
        raw_calibration = np.linspace(-2.0, 2.0, 300, dtype=np.float64)
        calibration_probability = 1.0 / (1.0 + np.exp(-raw_calibration))
        calibration_labels = (
            rng.random(raw_calibration.size) < calibration_probability
        ).astype(np.int8)
        raw_assessment = np.linspace(-1.5, 1.5, 120, dtype=np.float64)
        origin = SimpleNamespace(
            split=SimpleNamespace(
                origin="2022-04-11", calibration_date="2022-04-10"
            ),
            labels={
                "calibration": calibration_labels,
                # Deliberately present but protected by the patched AP function.
                "assessment": np.tile(np.array([0, 1], dtype=np.int8), 60),
            },
            users={"calibration": np.arange(raw_calibration.size, dtype=np.int64)},
        )
        component = SimpleNamespace(
            origin="2022-04-11",
            model_id="BL1",
            pair_id="A1em04_E1em03",
            raw_calibration=raw_calibration,
            raw_assessment=raw_assessment,
        )
        budget = runner.Budget(contract=load_contract())
        book = runner.LedgerBook()
        with patch.object(
            runner,
            "average_precision_score",
            side_effect=AssertionError("assessment AP read inside calibrate_component"),
        ):
            outcome = runner.calibrate_component(
                budget, book, origin, component, stage="search"
            )

        self.assertTrue(outcome.eligible)
        self.assertIsNotNone(outcome.probability)
        self.assertEqual(budget.stage_counts["search:calibrator"], 1)
        self.assertEqual(book.calibration[-1]["status"], "complete")

    def test_BL0_is_not_fitted_even_when_pooled_daily_baselines_vary(self):
        budget = runner.Budget(contract=load_contract())
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
        # A pooled BL0 column may contain one constant per origin and therefore
        # vary across origins; the contract nevertheless declares every BL0
        # diagnostic slot N/A and attempted_fits=0.
        probability = np.array([0.2] * 4 + [0.4] * 4, dtype=np.float64)
        row = runner.assessment_calibration_regression(
            budget,
            labels,
            probability,
            scope="pooled_7_days",
            model_id="BL0",
        )
        self.assertEqual(row["status"], "not_applicable")
        self.assertTrue(row["reason"])
        self.assertIsNone(row["assessment_calibration_intercept"])
        self.assertIsNone(row["assessment_calibration_slope"])
        self.assertEqual(budget.counts.get("diagnostic", 0), 0)

    def test_unpenalized_sklearn_1_9_form_uses_infinite_C_without_warning(self):
        seen = {}

        def constructor(**kwargs):
            seen.update(kwargs)
            return LogisticRegression(**kwargs)

        logit = np.linspace(-2.0, 2.0, 200, dtype=np.float64)
        labels = (logit + np.sin(np.arange(logit.size)) * 0.1 > 0).astype(np.int8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch("sklearn.linear_model.LogisticRegression", side_effect=constructor):
                intercept, slope = runner._unconstrained_logit_regression(logit, labels)

        self.assertEqual(seen["C"], np.inf)
        self.assertEqual(seen["l1_ratio"], 0.0)
        self.assertEqual(seen["solver"], "lbfgs")
        self.assertFalse(caught)
        self.assertTrue(np.isfinite(intercept))
        self.assertTrue(np.isfinite(slope))


def _daily_fixture(
    *,
    bl1_log_loss: list[float],
    bl2_ap: list[float],
    bl1_ap: list[float],
    bl0_log_loss: float = 0.63,
) -> dict:
    """A minimal frozen-daily payload shaped like the real one."""

    origins = [f"2022-04-{day:02d}" for day in range(11, 18)]
    rng = np.random.default_rng(5)
    rows = 700
    labels = (rng.random(rows) < 0.32).astype(np.int8)
    users = rng.integers(0, 40, size=rows).astype(np.int64)
    base = np.full(rows, 0.32)
    better = np.clip(base + 0.35 * (labels - 0.32), 1e-6, 1 - 1e-6)
    daily_rows: list[dict] = []
    for position, origin in enumerate(origins):
        daily_rows.append(
            {
                "origin": origin,
                "model_id": "BL0",
                "log_loss": bl0_log_loss,
                "brier": 0.218,
                "average_precision": 0.32,
                "user_gauc_event_weighted": 0.5,
            }
        )
        daily_rows.append(
            {
                "origin": origin,
                "model_id": "BL1",
                "log_loss": bl1_log_loss[position],
                "brier": 0.210,
                "average_precision": bl1_ap[position],
                "user_gauc_event_weighted": 0.51,
            }
        )
        daily_rows.append(
            {
                "origin": origin,
                "model_id": "BL2",
                "log_loss": bl1_log_loss[position] - 0.01,
                "brier": 0.205,
                "average_precision": bl2_ap[position],
                "user_gauc_event_weighted": 0.53,
            }
        )
    per_origin = rows // len(origins)
    return {
        "daily_origins": origins,
        "daily_rows": daily_rows,
        "labels": labels,
        "users": users,
        "prior_batch": rng.integers(0, 800, size=rows).astype(np.int64),
        "predictions": {"BL0": base, "BL1": base.copy(), "BL2": better},
        "origin_sizes": [per_origin] * len(origins),
        "selected": {"pair_id": "A1em04_E1em03", "alpha": 1e-4, "eta0": 1e-3},
    }


class ProbabilityQualityGateTests(unittest.TestCase):
    """Contract: probability_quality_gate.daily_stability_requirements."""

    def _origins(self, daily: dict) -> dict:
        class _Split:
            def __init__(self, origin: str) -> None:
                self.origin = origin
                self.bl0_probability = 0.32

        class _Origin:
            def __init__(self, origin: str, labels: np.ndarray) -> None:
                self.split = _Split(origin)
                self.labels = {"assessment": labels}

        chunk = len(daily["labels"]) // len(daily["daily_origins"])
        return {
            origin: _Origin(origin, daily["labels"][i * chunk : (i + 1) * chunk])
            for i, origin in enumerate(daily["daily_origins"])
        }

    def test_seven_good_days_pass(self):
        daily = _daily_fixture(
            bl1_log_loss=[0.60] * 7, bl1_ap=[0.33] * 7, bl2_ap=[0.44] * 7
        )
        result = runner.probability_quality_gate(load_contract(), self._origins(daily), daily)
        for row in result["per_model"]:
            self.assertEqual(row["nonworse_log_loss_days"], 7)
        self.assertEqual(len(result["per_model"]), 2)

    def test_three_bad_days_fail_the_stability_requirement(self):
        # 4 non-worse days is below the declared minimum of 5
        daily = _daily_fixture(
            bl1_log_loss=[0.60, 0.60, 0.60, 0.60, 0.99, 0.99, 0.99],
            bl1_ap=[0.33] * 7,
            bl2_ap=[0.44] * 7,
        )
        result = runner.probability_quality_gate(load_contract(), self._origins(daily), daily)
        bl1 = next(row for row in result["per_model"] if row["model_id"] == "BL1")
        self.assertEqual(bl1["nonworse_log_loss_days"], 4)
        self.assertFalse(bl1["passed"])
        self.assertFalse(result["passed"])

    def test_exactly_five_non_worse_days_is_the_boundary(self):
        daily = _daily_fixture(
            bl1_log_loss=[0.60, 0.60, 0.60, 0.60, 0.60, 0.99, 0.99],
            bl1_ap=[0.33] * 7,
            bl2_ap=[0.44] * 7,
        )
        result = runner.probability_quality_gate(load_contract(), self._origins(daily), daily)
        bl1 = next(row for row in result["per_model"] if row["model_id"] == "BL1")
        self.assertEqual(bl1["nonworse_log_loss_days"], 5)


class SearchBL2GateTests(unittest.TestCase):
    @staticmethod
    def _passing_gate() -> dict:
        return {
            "optimization_adequate": True,
            "pooled_log_loss_minus_BL0": -0.01,
            "pooled_brier_minus_BL0": -0.01,
            "nonworse_log_loss_origins_vs_BL0": 2,
            "nonworse_brier_origins_vs_BL0": 2,
            "delta_average_precision": 0.01,
            "delta_user_gauc_event_weighted": 0.0,
            "delta_log_loss": -0.01,
            "delta_brier": -0.01,
            "positive_average_precision_origins": 2,
        }

    def test_same_origin_BL0_noninferiority_requires_two_of_three_origins(self):
        criteria = load_contract()["search_and_selection"][
            "primary_BL2_same_configuration_eligibility"
        ]
        gate = self._passing_gate()
        self.assertTrue(runner.primary_bl2_search_gate_passed(gate, criteria))

        gate["nonworse_log_loss_origins_vs_BL0"] = 1
        self.assertFalse(runner.primary_bl2_search_gate_passed(gate, criteria))

        gate = self._passing_gate()
        gate["nonworse_brier_origins_vs_BL0"] = 1
        self.assertFalse(runner.primary_bl2_search_gate_passed(gate, criteria))


class RelativeHistoryGateTests(unittest.TestCase):
    """Contract: relative_history_gate."""

    def _bootstrap(self, ci_lower: float) -> list[dict]:
        return [
            {
                "comparison_id": "BL2_minus_BL1",
                "metric": "average_precision",
                "point_estimate": 0.11,
                "CI_lower": ci_lower,
                "CI_upper": 0.12,
            }
        ]

    def test_gate_passes_when_every_requirement_holds(self):
        daily = _daily_fixture(
            bl1_log_loss=[0.60] * 7, bl1_ap=[0.33] * 7, bl2_ap=[0.44] * 7
        )
        result = runner.relative_history_gate(load_contract(), daily, self._bootstrap(0.10))
        self.assertEqual(result["positive_average_precision_days"], 7)
        self.assertTrue(result["passed"])

    def test_a_nonpositive_bootstrap_lower_bound_fails_the_gate(self):
        daily = _daily_fixture(
            bl1_log_loss=[0.60] * 7, bl1_ap=[0.33] * 7, bl2_ap=[0.44] * 7
        )
        result = runner.relative_history_gate(load_contract(), daily, self._bootstrap(-0.001))
        self.assertFalse(result["passed"])

    def test_three_negative_days_fail_the_daily_requirement(self):
        daily = _daily_fixture(
            bl1_log_loss=[0.60] * 7,
            bl1_ap=[0.33] * 7,
            bl2_ap=[0.44, 0.44, 0.44, 0.44, 0.30, 0.30, 0.30],
        )
        result = runner.relative_history_gate(load_contract(), daily, self._bootstrap(0.10))
        self.assertEqual(result["positive_average_precision_days"], 4)
        self.assertFalse(result["passed"])

    def test_missing_bootstrap_row_cannot_silently_pass(self):
        daily = _daily_fixture(
            bl1_log_loss=[0.60] * 7, bl1_ap=[0.33] * 7, bl2_ap=[0.44] * 7
        )
        result = runner.relative_history_gate(load_contract(), daily, [])
        self.assertTrue(np.isnan(result["average_precision_ci_lower"]))
        self.assertFalse(result["passed"])


class SliceTests(unittest.TestCase):
    """Contract: evaluation_slices."""

    def test_slice_bounds_match_the_contract_list(self):
        declared = [str(item) for item in load_contract()["evaluation_slices"]["slices"]]
        produced = [name for name, _, _ in runner.SLICE_BOUNDS]
        self.assertEqual(declared, produced)

    def test_masks_partition_history_depth_correctly(self):
        prior = np.array([0, 49, 50, 199, 200, 499, 500, 1200], dtype=np.int64)
        masks = dict(runner.slice_masks(prior))
        self.assertTrue(masks["all_assessment_rows"].all())
        self.assertEqual(masks["history_0_49"].sum(), 2)
        self.assertEqual(masks["history_50_199"].sum(), 2)
        self.assertEqual(masks["history_200_plus"].sum(), 4)
        self.assertEqual(masks["history_500_plus_exploratory"].sum(), 2)
        # the three primary depth bands are disjoint and cover every row
        primary = (
            masks["history_0_49"].astype(int)
            + masks["history_50_199"].astype(int)
            + masks["history_200_plus"].astype(int)
        )
        self.assertTrue((primary == 1).all())

    def test_slice_metrics_cover_both_declared_scopes(self):
        daily = _daily_fixture(bl1_log_loss=[0.6] * 7, bl1_ap=[0.33] * 7, bl2_ap=[0.44] * 7)
        daily["prior_batch"] = np.zeros(daily["labels"].shape, dtype=np.int64)
        rows = runner.slice_metrics(daily)

        # 5 slices on the pooled scope plus 5 on each of the 7 origins
        self.assertEqual(len(rows), 5 * (1 + len(daily["daily_origins"])))
        self.assertEqual({row["slice"] for row in rows}, {n for n, _, _ in runner.SLICE_BOUNDS})
        scopes = {row["scope"] for row in rows}
        self.assertIn("pooled_seven_days", scopes)
        self.assertEqual(scopes - {"pooled_seven_days"}, set(daily["daily_origins"]))

        for row in rows:
            self.assertEqual(row["gate_role"], "descriptive_only")
        populated = [row for row in rows if "BL1_metric_average_precision" in row]
        self.assertTrue(populated)
        for row in populated:
            self.assertIn("BL0_metric_average_precision", row)
            self.assertIn("paired_delta_average_precision", row)

        # every prior_batch is 0 here, so the deep-history slices must be empty
        for row in rows:
            if row["slice"] in ("history_200_plus", "history_500_plus_exploratory"):
                self.assertEqual(row["rows"], 0)

    def test_per_origin_slice_windows_must_tile_the_pooled_rows(self):
        daily = _daily_fixture(bl1_log_loss=[0.6] * 7, bl1_ap=[0.33] * 7, bl2_ap=[0.44] * 7)
        daily["origin_sizes"] = [1] * len(daily["daily_origins"])
        with self.assertRaises(runner.ContractStop):
            runner.slice_metrics(daily)

    def test_render_report_contains_at_least_one_real_slice_data_row(self):
        selected = {"pair_id": "A1em04_E1em03", "alpha": 1e-4, "eta0": 1e-3}
        search = {
            "selected": selected,
            "eligible_configuration_count": 4,
            "paired_robustness_configuration_count": 4,
            "paired_robustness_ineligible_pairs": {},
            "adequacy_rows": [],
            "pair_rows": [],
            "secondary_selected": None,
            "search_rule_rows": [],
            "pooled": {},
            "metric_rows": [],
        }
        pooled_rows = []
        for model_id in ("BL0", "BL1", "BL2"):
            pooled_rows.append(
                {
                    "origin": "pooled_7_days",
                    "model_id": model_id,
                    "uncalibrated_log_loss": 0.62,
                    "log_loss": 0.60,
                    "uncalibrated_brier": 0.22,
                    "brier": 0.21,
                    "uncalibrated_ECE20_equal_width": 0.03,
                    "ECE20_equal_width": 0.02,
                }
            )
        daily = {
            "daily_rows": pooled_rows,
            "calibration_regression_rows": [
                {"status": "complete"} for _ in range(16)
            ]
            + [{"status": "not_applicable"} for _ in range(8)],
            "adequacy_rows": [],
            "rule_rows": [],
        }
        probability_gate = {"passed": True, "per_model": []}
        history_gate = {
            "passed": True,
            "delta_average_precision": 0.01,
            "delta_user_gauc_event_weighted": 0.01,
            "delta_log_loss": -0.01,
            "delta_brier": -0.01,
            "positive_average_precision_days": 7,
            "total_days": 7,
            "average_precision_ci_lower": 0.001,
        }
        slices = [
            {
                "scope": "pooled_seven_days",
                "slice": "all_assessment_rows",
                "rows": 700,
                "prevalence": 0.32,
                "BL1_metric_average_precision": 0.34,
                "BL2_metric_average_precision": 0.44,
                "paired_delta_average_precision": 0.10,
            }
        ]
        with TemporaryDirectory() as raw:
            report = Path(raw) / "slice_report.md"
            runner.render_report(
                "f" * 64,
                search,
                daily,
                probability_gate,
                history_gate,
                slices,
                [],
                report,
            )
            text = report.read_text(encoding="utf-8")

        self.assertIn("## History-depth slices (descriptive only)", text)
        self.assertIn(
            "| pooled_seven_days | all_assessment_rows | 700 | 0.3200 | 0.340000 | 0.440000 | +0.100000 |",
            text,
        )


class ArtifactCoverageTests(unittest.TestCase):
    def test_runner_declares_every_contract_artifact(self):
        declared = {str(item) for item in load_contract()["required_outputs"]["artifacts"]}
        self.assertEqual(len(declared), 36)
        self.assertEqual(len(runner.ARTIFACT_NAMES), 36)
        self.assertEqual(declared - set(runner.ARTIFACT_NAMES), set())
        self.assertEqual(set(runner.ARTIFACT_NAMES) - declared, set())
        self.assertTrue(
            {
                "artifact_hash_manifest.json",
                "terminal_state.json",
                "search_artifact_freeze_manifest.json",
                "search_fixed_history_rule_diagnostics.csv",
            }
            <= declared
        )

    def test_ledger_schema_guard_rejects_an_incomplete_row(self):
        with self.assertRaises(runner.ContractStop):
            runner.require_fields([{"a": 1}], ["a", "b"], "synthetic ledger")
        with self.assertRaises(runner.ContractStop) as caught:
            runner.require_fields(
                [{"a": 1, "b": 2}, {"a": 3}], ["a", "b"], "synthetic ledger"
            )
        self.assertIn("row 1", str(caught.exception))
        runner.require_fields([{"a": 1, "b": 2}], ["a", "b"], "synthetic ledger")

    def test_search_freeze_manifest_hashes_fixed_rule_score_bytes(self):
        with TemporaryDirectory() as raw:
            output = Path(raw)
            schema = load_contract()["artifact_schemas"]["search_artifact_freeze_manifest"]
            predictions = [output / name for name in schema["exact_paths"]]
            for index, prediction in enumerate(predictions):
                prediction.write_bytes(f"synthetic-search-prediction-{index}".encode())
            scores = np.array([0.2, 0.4, 0.6], dtype=np.float64)
            frozen_scores = {
                (prediction.stem.removeprefix("search_predictions_origin_"), diagnostic_id):
                scores + index / 100.0
                for index, prediction in enumerate(predictions)
                for diagnostic_id in ("LIFETIME_SMOOTHED_RATE", "W10_SMOOTHED_RATE")
            }
            original = runner.OUTPUT_DIR
            runner.OUTPUT_DIR = output
            try:
                payload = runner.freeze_search_artifacts(
                    [prediction.name for prediction in predictions], frozen_scores
                )
            finally:
                runner.OUTPUT_DIR = original

            runner.require_fields(
                [payload], schema["required_fields"], "search freeze manifest"
            )
            runner.require_fields(
                payload["artifacts"],
                schema["artifact_entry_required_fields"],
                "search freeze artifacts",
            )
            self.assertEqual(payload["status"], schema["required_status"])
            self.assertEqual(payload["artifact_count"], schema["required_artifact_count"])
            self.assertEqual(
                sorted(row["path"] for row in payload["artifacts"]),
                sorted(schema["exact_paths"]),
            )
            for row in payload["artifacts"]:
                prediction = output / row["path"]
                self.assertEqual(row["size_bytes"], prediction.stat().st_size)
                self.assertEqual(row["sha256"], runner.sha256_file(prediction))

            self.assertEqual(len(payload["fixed_rule_scores"]), 6)
            rule = next(
                row
                for row in payload["fixed_rule_scores"]
                if row["origin"] == "2022-04-11"
                and row["diagnostic_id"] == "W10_SMOOTHED_RATE"
            )
            expected = hashlib.sha256(
                np.asarray(scores, dtype="<f8").tobytes(order="C")
            ).hexdigest()
            self.assertEqual(rule["sha256"], expected)
            self.assertEqual(rule["storage_column"], "fixed_rule_score_W10_SMOOTHED_RATE")

    def test_complete_hash_manifest_rejects_a_stale_required_artifact(self):
        with TemporaryDirectory() as raw:
            output = Path(raw)
            stale = output / "terminal_state.json"
            stale.write_text('{"state":"from_an_older_run"}', encoding="utf-8")
            original = (
                runner.OUTPUT_DIR,
                runner.REPORT_PATH,
                runner.FIGURE_PATH,
                runner.ACTIVE_RUN_STARTED,
                runner.ACTIVE_ARTIFACT_SNAPSHOT,
                runner.ACTIVE_RUN_ID,
                set(runner.ACTIVE_PRODUCED_ARTIFACTS),
            )
            runner.OUTPUT_DIR = output
            runner.REPORT_PATH = output / "report.md"
            runner.FIGURE_PATH = output / "figure.png"
            runner.ACTIVE_RUN_STARTED = True
            runner.ACTIVE_ARTIFACT_SNAPSHOT = runner.capture_managed_artifact_snapshot()
            runner.ACTIVE_RUN_ID = "synthetic-stale-artifact-run"
            runner.ACTIVE_PRODUCED_ARTIFACTS = set()
            contract = {
                "required_outputs": {
                    "artifacts": [
                        "terminal_state.json",
                        "artifact_hash_manifest.json",
                    ]
                }
            }
            try:
                with self.assertRaises(runner.ContractStop) as caught:
                    runner.write_artifact_hash_manifest(contract)
            finally:
                (
                    runner.OUTPUT_DIR,
                    runner.REPORT_PATH,
                    runner.FIGURE_PATH,
                    runner.ACTIVE_RUN_STARTED,
                    runner.ACTIVE_ARTIFACT_SNAPSHOT,
                    runner.ACTIVE_RUN_ID,
                    runner.ACTIVE_PRODUCED_ARTIFACTS,
                ) = original

            self.assertIn("not created or replaced by this release attempt", str(caught.exception))
            self.assertFalse((output / "artifact_hash_manifest.json").exists())

    def test_terminal_stop_writes_state_and_hashes_the_failure_checkpoint(self):
        with TemporaryDirectory() as raw:
            workspace = Path(raw)
            output = workspace / "out"
            original = (
                runner.OUTPUT_DIR,
                runner.REPORT_PATH,
                runner.FIGURE_PATH,
                runner.ACTIVE_BUDGET,
                runner.ACTIVE_BOOK,
                runner.ACTIVE_RUN_STARTED,
                runner.ACTIVE_ARTIFACT_SNAPSHOT,
                runner.ACTIVE_RUN_ID,
                set(runner.ACTIVE_PRODUCED_ARTIFACTS),
                sys.argv,
            )
            runner.OUTPUT_DIR = output
            runner.REPORT_PATH = workspace / "report.md"
            runner.FIGURE_PATH = workspace / "figure.png"
            runner.ACTIVE_BUDGET = None
            runner.ACTIVE_BOOK = None
            runner.ACTIVE_RUN_STARTED = True
            runner.ACTIVE_ARTIFACT_SNAPSHOT = runner.capture_managed_artifact_snapshot()
            runner.ACTIVE_RUN_ID = "synthetic-failure-run"
            runner.ACTIVE_PRODUCED_ARTIFACTS = set()
            sys.argv = ["runner", "--release", "--approve-contract-hash", "f" * 64]
            stop = runner.TerminalStop(
                "frozen_daily_optimization_adequacy_failure",
                runner.EXIT_CASE_D_FROZEN_DAILY_OPTIMIZATION_FAILURE,
                "synthetic terminal stop",
                stage="frozen_daily",
            )
            try:
                with patch.object(runner, "release", side_effect=stop):
                    code = runner.main()
            finally:
                (
                    runner.OUTPUT_DIR,
                    runner.REPORT_PATH,
                    runner.FIGURE_PATH,
                    runner.ACTIVE_BUDGET,
                    runner.ACTIVE_BOOK,
                    runner.ACTIVE_RUN_STARTED,
                    runner.ACTIVE_ARTIFACT_SNAPSHOT,
                    runner.ACTIVE_RUN_ID,
                    runner.ACTIVE_PRODUCED_ARTIFACTS,
                    sys.argv,
                ) = original

            self.assertEqual(code, runner.EXIT_CASE_D_FROZEN_DAILY_OPTIMIZATION_FAILURE)
            terminal_path = output / "terminal_state.json"
            run_manifest_path = output / "run_manifest.json"
            hash_path = output / "artifact_hash_manifest.json"
            self.assertTrue(terminal_path.is_file())
            self.assertTrue(run_manifest_path.is_file())
            self.assertTrue(hash_path.is_file())
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            self.assertEqual(
                terminal["state"], "frozen_daily_optimization_adequacy_failure"
            )
            self.assertEqual(terminal["stage"], "frozen_daily")
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            manifest = json.loads(hash_path.read_text(encoding="utf-8"))
            schemas = load_contract()["artifact_schemas"]
            self.assertFalse(
                set(schemas["terminal_state"]["required_fields"]) - set(terminal)
            )
            self.assertFalse(
                set(schemas["run_manifest"]["required_fields"]) - set(run_manifest)
            )
            self.assertFalse(
                set(schemas["artifact_hash_manifest"]["top_level_required_fields"])
                - set(manifest)
            )
            self.assertEqual(manifest["status"], "partial_fail_closed_checkpoint")
            self.assertEqual(
                {terminal["run_id"], run_manifest["run_id"], manifest["run_id"]},
                {"synthetic-failure-run"},
            )
            self.assertEqual(
                {
                    terminal["completion_scope"],
                    run_manifest["completion_scope"],
                    manifest["completion_scope"],
                },
                {"partial_failure"},
            )
            indexed = {row["path"]: row for row in manifest["artifacts"]}
            self.assertIn("terminal_state.json", indexed)
            self.assertIn("run_manifest.json", indexed)
            self.assertEqual(
                indexed["terminal_state.json"]["sha256"],
                runner.sha256_file(terminal_path),
            )


class DiagnosticTests(unittest.TestCase):
    def test_fixed_rules_have_no_fitted_parameters(self):
        contract = load_contract()
        declared = contract["models"]["fixed_nonfitted_history_diagnostics"]
        self.assertEqual(declared["common_rules"]["fitted_parameters"], "none")
        self.assertFalse(declared["common_rules"]["eligible_for_model_selection"])

        size = 5
        columns = {name: np.zeros(size) for name in runner.REQUIRED_COLUMNS}
        columns["prior_event_n"] = np.array([0, 10, 100, 0, 40], dtype=np.float64)
        columns["prior_positive_n"] = np.array([0, 5, 50, 0, 10], dtype=np.float64)
        columns["w10_event_n"] = np.array([0, 10, 10, 0, 8], dtype=np.float64)
        columns["w10_positive_n"] = np.array([0, 5, 9, 0, 2], dtype=np.float64)
        frame = runner.Frame(columns=columns)
        scores = runner.fixed_rule_scores(frame, np.arange(size), 0.32)
        # a user with no history falls back exactly to the fit prevalence
        self.assertAlmostEqual(scores["LIFETIME_SMOOTHED_RATE"][0], 0.32)
        self.assertAlmostEqual(scores["W10_SMOOTHED_RATE"][0], 0.32)
        self.assertAlmostEqual(scores["LIFETIME_SMOOTHED_RATE"][2], (50 + 20 * 0.32) / 120)

    def test_metric_epsilon_is_the_contract_value_not_the_v002_default(self):
        from kuairand_longseq.evaluation import gate2b_metrics as m
        from kuairand_longseq.models import gate2b_repair_v003 as repair

        self.assertEqual(m.PROBABILITY_EPSILON, 1e-7)
        self.assertEqual(repair.METRIC_CLIP_LOW, 1e-15)
        y = np.array([0, 1, 0, 1], dtype=np.int8)
        users = np.array([1, 1, 2, 2], dtype=np.int64)
        p = np.array([1e-12, 1.0 - 1e-12, 1e-12, 1.0 - 1e-12])
        wide = m.point_metrics(y, p, users)
        narrow = m.point_metrics(y, p, users, epsilon=repair.METRIC_CLIP_LOW)
        # the v002 default clip would materially misstate a confident model
        self.assertGreater(wide["log_loss"], narrow["log_loss"])


if __name__ == "__main__":
    unittest.main()
