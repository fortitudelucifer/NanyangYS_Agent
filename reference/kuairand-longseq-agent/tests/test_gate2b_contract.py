import hashlib
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def construct_unique(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_unique)


class Gate2BContractTests(unittest.TestCase):
    def test_scoped_authorization_and_budget(self):
        path = PROJECT_ROOT / "configs/gate2b_fixed_row_baseline_contract_v002.yaml"
        contract = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
        self.assertEqual(contract["status"], "approved_for_gate2b_train_only_fixed_row_fitting")
        denied = set(contract["authorization"]["does_not_authorize"])
        self.assertTrue({"silver_recleaning", "validation_access", "late_table_access", "random_table_access"} <= denied)
        self.assertEqual(contract["operational_budget"]["maximum_total_fit_runs"], 32)
        self.assertEqual(contract["evaluation"]["ece"]["role"], "descriptive_only_no_gate")
        self.assertEqual(len(contract["input_allowlist"]), 4)
        self.assertEqual(sum(bool(item["query_allowed"]) for item in contract["input_allowlist"]), 3)

    def test_probability_repair_contract_is_fail_closed_and_bounded(self):
        path = PROJECT_ROOT / "configs/gate2b_probability_repair_contract_v003.yaml"
        contract = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
        self.assertEqual(contract["version"], 3)
        self.assertEqual(contract["status"], "proposed_for_explicit_execution_approval_not_authorized_to_run")
        self.assertFalse(contract["authorization"]["execution_authorized"])
        self.assertFalse(
            contract["design_revision_001"][
                "v003_fit_or_scientific_metric_observed_before_revision"
            ]
        )
        # design_revision_003 records the runner as implemented; execution stays
        # gated on an explicit hash approval and an independent review.
        status = contract["implementation_status"]
        self.assertTrue(status["executable_v003_runner_exists"])
        self.assertFalse(status["contract_only_created"])
        self.assertTrue(
            all(
                value == "complete"
                for value in status["required_before_execution_approval_status"].values()
            )
        )
        self.assertEqual(status["independent_implementation_review"]["status"], "pending")
        self.assertTrue(
            status["independent_implementation_review"]["required_before_execution_approval"]
        )
        pinned = {item["path"] for item in status["result_producing_implementation"]["files"]}
        self.assertEqual(
            pinned,
            {
                "scripts/run_gate2b_probability_repair_v003.py",
                "src/kuairand_longseq/models/gate2b_repair_v003.py",
                "src/kuairand_longseq/evaluation/gate2b_metrics.py",
            },
        )
        for item in status["result_producing_implementation"]["files"]:
            actual = hashlib.sha256((PROJECT_ROOT / item["path"]).read_bytes()).hexdigest()
            self.assertEqual(actual, item["sha256"], msg=f"pinned hash stale for {item['path']}")
        self.assertIn(
            "independent_implementation_review_complete",
            contract["authorization"]["required_before_execution"],
        )
        self.assertFalse(
            contract["design_revision_003"][
                "v003_fit_or_scientific_metric_observed_before_revision"
            ]
        )
        self.assertEqual(len(contract["input_allowlist"]), 1)
        self.assertTrue(contract["input_allowlist"][0]["query_allowed"])
        self.assertEqual(
            contract["population"]["canonical_definition"],
            "early_standard_silver_union_exclusive_early_formula_mismatch_official_labels",
        )
        self.assertEqual(
            contract["population"]["canonical_sources"]["additive"]["expected_target_tab_rows"],
            14070,
        )
        splits = contract["temporal_protocol"]["origin_splits"]
        self.assertEqual(len(splits), 7)
        for split in splits:
            self.assertLess(split["estimator_fit_date_range_inclusive"][1], split["calibration_date"])
            self.assertLess(split["calibration_date"], split["assessment_date"])
            self.assertEqual(split["origin"], split["assessment_date"])
        estimator = contract["models"]["shared_sparse_linear_estimator"]
        self.assertEqual(estimator["learning_rate"], "adaptive")
        self.assertEqual(
            len(estimator["alpha_values"]) * len(estimator["eta0_values"]),
            estimator["configurations_per_model"],
        )
        reference = contract["models"]["diagnostic_reference_solver"]
        self.assertEqual(reference["total_fit_runs"], 20)
        self.assertEqual(
            reference["search_fit_runs"] + reference["frozen_daily_fit_runs"],
            reference["total_fit_runs"],
        )
        self.assertEqual(len(reference["frozen_daily_origins"]), 4)
        self.assertFalse(reference["frozen_daily_may_change_selection_or_hyperparameters"])
        self.assertFalse(reference["frozen_daily_may_be_used_for_model_selection"])
        self.assertEqual(
            reference["C_per_origin_and_alpha"],
            "1_divided_by_estimator_fit_rows_times_alpha",
        )
        selection = contract["search_and_selection"]
        phase_order = selection["search_phase_order"]
        self.assertLess(
            phase_order.index("materialize_all_search_assessment_metrics_in_one_batch"),
            phase_order.index("apply_BL1_eligibility_and_BL1_only_shared_configuration_selection"),
        )
        self.assertTrue(
            selection["primary_shared_configuration_selection"][
                "selection_uses_only_BL1_metrics_and_parameters"
            ]
        )
        self.assertTrue(
            selection["frozen_daily_backtest"][
                "alpha_and_eta0_identical_between_BL1_and_BL2"
            ]
        )
        self.assertEqual(len(selection["paired_configuration_registry"]["configurations"]), 4)
        budget = contract["operational_budget"]
        primary_fits = sum(
            budget[key]
            for key in (
                "maximum_search_SGD_fit_runs",
                "maximum_search_primary_calibrator_fit_runs",
                "maximum_daily_SGD_fit_runs",
                "maximum_daily_primary_calibrator_fit_runs",
            )
        )
        self.assertEqual(primary_fits, budget["maximum_primary_component_fit_runs"])
        self.assertEqual(
            primary_fits
            + budget["maximum_reference_solver_fit_runs"]
            + budget["maximum_diagnostic_calibration_regression_fits"],
            budget["maximum_total_fit_operations"],
        )
        self.assertEqual(budget["maximum_total_fit_operations"], 120)
        self.assertEqual(budget["maximum_daily_reference_solver_fit_runs"], 8)
        self.assertEqual(
            budget["maximum_search_reference_solver_fit_runs"]
            + budget["maximum_daily_reference_solver_fit_runs"],
            budget["maximum_reference_solver_fit_runs"],
        )
        self.assertEqual(contract["runtime_preflight"]["extra_fit_operations"], 0)
        self.assertEqual(
            contract["models"]["fixed_nonfitted_history_diagnostics"]["common_rules"][
                "total_evaluations"
            ],
            budget["maximum_nonfitted_history_rule_evaluations"],
        )
        self.assertEqual(
            contract["probability_diagnostics"]["assessment_calibration_regression"][
                "total_fits"
            ],
            budget["maximum_diagnostic_calibration_regression_fits"],
        )
        self.assertIn(
            "reference_regularized_training_objective",
            contract["artifact_schemas"]["reference_solver_ledger_required_fields"],
        )
        self.assertEqual(contract["calibration"]["fit_scope"], "calibration_date_rows_only")
        self.assertEqual(
            contract["calibration"]["primary_ranking_score"], "calibrated_probability"
        )
        self.assertEqual(
            contract["probability_quality_gate"]["scientific_noninferiority_margin"],
            {"log_loss": 0.0, "brier": 0.0},
        )
        self.assertEqual(
            contract["probability_diagnostics"]["pass_fail_role"]["ECE20"],
            "descriptive_only",
        )
        self.assertEqual(
            contract["discrimination_diagnostics"]["BL1_minus_BL0"]["superiority_gate"],
            "none",
        )
        self.assertEqual(
            contract["models"]["fixed_nonfitted_history_diagnostics"]["common_rules"][
                "hard_ROC_AUC_threshold"
            ],
            "forbidden",
        )
        self.assertEqual(
            contract["optimization_adequacy"]["reference_pairing_key"],
            ["model_id", "alpha", "origin"],
        )

    def test_runtime_preflight_projection_matches_declared_counts(self):
        path = PROJECT_ROOT / "configs/gate2b_probability_repair_contract_v003.yaml"
        contract = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
        proj = contract["runtime_preflight"]["projection"]
        budget = contract["operational_budget"]

        # remaining == total - preflight, for every fitted component type
        for total, pre, remaining in (
            ("total_SGD_fits", "preflight_SGD_fits", "remaining_SGD_fits"),
            (
                "total_primary_calibrator_fits",
                "preflight_primary_calibrator_fits",
                "remaining_primary_calibrator_fits",
            ),
            ("total_reference_fits", "preflight_reference_fits", "remaining_reference_fits"),
        ):
            self.assertEqual(proj[total] - proj[pre], proj[remaining], msg=remaining)

        # declared totals must agree with the operational budget caps
        self.assertEqual(
            proj["total_SGD_fits"],
            budget["maximum_search_SGD_fit_runs"] + budget["maximum_daily_SGD_fit_runs"],
        )
        self.assertEqual(
            proj["total_primary_calibrator_fits"],
            budget["maximum_search_primary_calibrator_fit_runs"]
            + budget["maximum_daily_primary_calibrator_fit_runs"],
        )
        self.assertEqual(
            proj["total_reference_fits"], budget["maximum_reference_solver_fit_runs"]
        )
        self.assertEqual(
            proj["diagnostic_calibration_regression_fits_using_primary_calibrator_time_proxy"],
            budget["maximum_diagnostic_calibration_regression_fits"],
        )

        # the projection formula must reference declared counts, never literals
        formula = proj["elapsed_projection_formula"]
        self.assertEqual(proj["literal_coefficients_in_the_projection_formula"], "forbidden")
        self.assertTrue(proj["projection_formula_coefficients_must_equal_declared_counts"])
        for name in (
            "remaining_SGD_fits",
            "remaining_primary_calibrator_fits",
            "diagnostic_calibration_regression_fits_using_primary_calibrator_time_proxy",
            "remaining_reference_fits",
            "safety_multiplier",
            "fixed_nonfit_reserve_seconds",
        ):
            self.assertIn(name, formula, msg=f"{name} missing from projection formula")
        for literal in (" 36", " 60", " 10", " 18", " 1.5", " 600"):
            self.assertNotIn(f"*{literal}", formula.replace("* ", "*"))
        self.assertEqual(proj["projected_total_elapsed_must_not_exceed_minutes"], 120)
        self.assertEqual(budget["maximum_elapsed_minutes"], 120)

    def test_frozen_daily_adequacy_is_fail_closed_and_outcomes_are_preregistered(self):
        path = PROJECT_ROOT / "configs/gate2b_probability_repair_contract_v003.yaml"
        contract = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)

        daily = contract["optimization_adequacy"]["frozen_daily_adequacy"]
        self.assertTrue(daily["applies_to_all_seven_frozen_daily_origins"])
        self.assertEqual(daily["failed_day_action"], "stop_release_before_metric_aggregation")
        self.assertEqual(daily["day_level_exclusion_or_reweighting"], "forbidden")
        self.assertEqual(daily["tolerance_change_for_daily_phase"], "forbidden")
        self.assertTrue(daily["runs_after_selection_and_cannot_change_selection"])

        backtest = contract["search_and_selection"]["frozen_daily_backtest"]
        self.assertEqual(backtest["reference_solver_fit_runs"], 8)
        self.assertTrue(backtest["optimization_adequacy_required_on_every_origin"])
        order = backtest["frozen_daily_phase_order"]
        self.assertLess(
            order.index("evaluate_frozen_daily_optimization_adequacy_on_all_seven_origins"),
            order.index(
                "only_then_fit_calibrators_and_materialize_daily_predictions_and_metrics"
            ),
        )
        self.assertEqual(
            len(backtest["reference_solver_origins"])
            + len(backtest["reference_solver_reused_search_origins"]),
            len(backtest["origins"]),
        )

        stops = contract["release_stop_conditions"]
        self.assertIn(
            "frozen_daily_optimization_adequacy_failure_on_any_of_the_seven_origins", stops
        )
        self.assertIn(
            "frozen_daily_origin_exclusion_or_reweighting_after_an_adequacy_failure", stops
        )

        outcomes = contract["preregistered_outcome_interpretation"]
        self.assertTrue(outcomes["registered_before_any_v003_fit_or_scientific_metric_read"])
        self.assertEqual(outcomes["post_hoc_reinterpretation_of_these_states"], "forbidden")
        cases = [key for key in outcomes if key.startswith("case_")]
        self.assertEqual(len(cases), 4)
        case_b = outcomes["case_B_static_baseline_is_uninformative_with_adequate_fit"]
        self.assertEqual(case_b["status"], "scientific_finding_not_a_defect")
        self.assertIn(
            "reopen_the_optimizer_search_on_the_basis_of_this_outcome",
            case_b["forbidden_action"],
        )
        self.assertIn(
            "reclassify_this_outcome_as_an_implementation_failure", case_b["forbidden_action"]
        )
        case_d = outcomes["case_D_frozen_daily_optimization_adequacy_failure"]
        self.assertEqual(case_d["scientific_conclusion"], "none")
        self.assertIn("exclude_the_failing_day_and_reaggregate", case_d["forbidden_action"])
        self.assertIn(
            "frozen_daily_optimization_adequacy_failure", contract["final_decision_states"]
        )
        self.assertFalse(
            contract["design_revision_002"][
                "v003_fit_or_scientific_metric_observed_before_revision"
            ]
        )

    def test_reference_solver_matches_sgd_regularization_objective(self):
        path = PROJECT_ROOT / "configs/gate2b_probability_repair_contract_v003.yaml"
        contract = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
        reference = contract["models"]["diagnostic_reference_solver"]
        n_rows = 1000
        alpha = reference["alpha_values"][0]
        C = 1.0 / (n_rows * alpha)
        self.assertAlmostEqual(1.0 / (n_rows * C), alpha)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = LogisticRegression(
                solver=reference["solver"],
                l1_ratio=reference["l1_ratio"],
                C=C,
                dual=reference["dual"],
                tol=reference["tol"],
                fit_intercept=reference["fit_intercept"],
                intercept_scaling=reference["intercept_scaling"],
                class_weight=reference["class_weight"],
                random_state=reference["random_state"],
                max_iter=reference["max_iter"],
                verbose=reference["verbose"],
                warm_start=reference["warm_start"],
                n_jobs=reference["n_jobs"],
            )
            model.fit(
                np.arange(20, dtype=np.float64).reshape(-1, 1),
                np.array([0] * 10 + [1] * 10),
            )
        self.assertFalse(caught)
        self.assertEqual(
            model.get_params()["penalty"],
            reference["expected_get_params_penalty_sentinel"],
        )


if __name__ == "__main__":
    unittest.main()
