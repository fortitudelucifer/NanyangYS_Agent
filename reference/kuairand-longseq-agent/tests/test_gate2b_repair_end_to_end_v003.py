"""End-to-end smoke test for the Gate 2B v003 runner.

Builds a small synthetic feature artifact and a synthetic contract derived from
the real one, then drives the complete release path: preflight, search, BL1-only
selection, the paired BL2 gate, the seven-day frozen backtest with its
fail-closed adequacy gate, the bootstrap, both terminal gates, and every declared
artifact.

Nothing here touches the frozen Gate 2B feature artifact, Silver, the
quarantine, Validation, the late table or the random table.
"""

import copy
import csv
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "gate2b_repair_runner_e2e",
    PROJECT_ROOT / "scripts/run_gate2b_probability_repair_v003.py",
)
runner = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = runner
_spec.loader.exec_module(runner)

DAYS = [f"2022-04-{day:02d}" for day in range(8, 18)]
# The smallest estimator-fit set here is two days.  Below roughly 4k fit rows the
# BL1 static bundle can still be improving at the frozen 100-iteration cap, which
# the runner correctly treats as non-convergence; the fixture is sized above that
# so the test exercises the protocol rather than a small-sample artefact.
ROWS_PER_DAY = 4000
USERS = 24


def build_feature_table(path: Path) -> dict:
    """A synthetic frame shaped like the real learning problem.

    Each user has a base propensity plus a per-user drift over the ten days.
    ``cat_user`` lets the static BL1 bundle learn the user average, while the
    strictly-prior H2 windows track the drifted current state, so BL2 can carry a
    genuine increment over BL1 rather than merely re-learning user identity.
    """

    rng = np.random.default_rng(20260815)
    total = len(DAYS) * ROWS_PER_DAY
    user_id = np.tile(np.arange(USERS, dtype=np.int64), total // USERS + 1)[:total]
    rng.shuffle(user_id)
    day_index = np.repeat(np.arange(len(DAYS), dtype=np.float64), ROWS_PER_DAY)

    base = rng.beta(2.0, 3.0, size=USERS)
    drift = rng.normal(scale=0.05, size=USERS)
    latent = np.clip(base[user_id] + drift[user_id] * day_index, 0.05, 0.95)
    labels = (rng.random(total) < latent).astype(np.int64)

    event_date = np.repeat(np.array(DAYS, dtype="datetime64[D]"), ROWS_PER_DAY)
    # lifetime history reflects the user average; the windows reflect the drift
    prior_event = rng.integers(80, 600, size=total).astype(np.int64)
    prior_positive = np.floor(prior_event * base[user_id]).astype(np.int64)
    columns: dict[str, np.ndarray] = {
        "source_table": np.full(total, "early_standard"),
        "source_row_number": np.arange(total, dtype=np.int64),
        "user_id": user_id,
        "video_id": rng.integers(0, 4000, size=total).astype(np.int64),
        "event_date": event_date,
        "long_view": labels,
        "prior_batch_n": rng.integers(0, 900, size=total).astype(np.int64),
        "prior_event_n": prior_event,
        "prior_positive_n": prior_positive,
        "last_user_gap_s": rng.integers(0, 90_000, size=total).astype(np.float32),
    }
    for window in (10, 50, 200):
        event_n = np.full(total, window, dtype=np.int64)
        columns[f"w{window}_event_n"] = event_n
        columns[f"w{window}_positive_n"] = np.floor(event_n * latent).astype(np.int64)
    for index, name in enumerate(runner.repair.CATEGORICAL_FIELDS):
        if name == "cat_user":
            columns[name] = user_id.copy()
        else:
            columns[name] = rng.integers(0, 5 + index * 3, size=total).astype(np.int64)
    for name in runner.repair.STATIC_CONTINUOUS_FIELDS:
        columns[name] = rng.normal(size=total).astype(np.float32)
    for name in runner.repair.STATIC_BINARY_FIELDS:
        columns[name] = rng.integers(0, 2, size=total).astype(np.float32)

    pq.write_table(pa.table(columns), path)
    return columns


def build_contract(feature_path: Path, columns: dict) -> dict:
    """Derive a synthetic contract from the real one, keeping its structure."""

    contract = copy.deepcopy(yaml.safe_load(runner.CONTRACT_PATH.read_text(encoding="utf-8")))
    contract["authorization"]["execution_authorized"] = True
    for name in contract["authorization"]["required_before_execution"]:
        if name == "actual_executable_budget_confirmed_not_to_exceed_the_frozen_planning_envelope":
            continue
        contract["authorization"]["required_before_execution_status"][name] = (
            "satisfied_by_synthetic_test_fixture"
        )
    # The fixture approves the exact implementation bytes imported by this test,
    # rather than inheriting stale pre-repair pins from the real fail-closed
    # contract.
    for item in contract["implementation_status"]["result_producing_implementation"][
        "files"
    ]:
        item["sha256"] = runner.sha256_file(PROJECT_ROOT / item["path"])
    contract["input_allowlist"] = [
        {
            "path": str(feature_path),
            "role": "synthetic_end_to_end_fixture",
            "expected_size_bytes": feature_path.stat().st_size,
            "expected_sha256": runner.sha256_file(feature_path),
            "query_allowed": True,
        }
    ]
    labels = columns["long_view"]
    contract["population"].update(
        {
            "expected_target_rows": int(labels.size),
            "expected_target_users": int(np.unique(columns["user_id"]).size),
            "expected_target_videos": int(np.unique(columns["video_id"]).size),
            "expected_target_positives": int(labels.sum()),
            "expected_all_target_identity_sha256": runner.identity_digest(
                columns["source_table"], columns["source_row_number"]
            ),
        }
    )

    dates = columns["event_date"].astype("datetime64[D]")
    splits = []
    for position in range(3, len(DAYS)):
        origin = DAYS[position]
        calibration = DAYS[position - 1]

        def counts(day: str) -> dict:
            mask = dates == np.datetime64(day, "D")
            return {
                "rows": int(mask.sum()),
                "users": int(np.unique(columns["user_id"][mask]).size),
                "positives": int(labels[mask].sum()),
            }

        splits.append(
            {
                "origin": origin,
                "estimator_fit_date_range_inclusive": [DAYS[0], DAYS[position - 2]],
                "calibration_date": calibration,
                "calibration_expected": counts(calibration),
                "assessment_date": origin,
                "assessment_expected": counts(origin),
            }
        )
    contract["temporal_protocol"]["origin_splits"] = splits
    contract["temporal_protocol"]["calibration_minimum_requirements"].update(
        {
            "minimum_rows": 10,
            "minimum_users": 2,
            "minimum_positives": 5,
            "minimum_negatives": 5,
        }
    )

    search_origins = [splits[0]["origin"], splits[3]["origin"], splits[-1]["origin"]]
    daily_origins = [item["origin"] for item in splits]
    selection = contract["search_and_selection"]
    selection["search_origins"] = search_origins
    selection["frozen_daily_backtest"]["origins"] = daily_origins
    selection["frozen_daily_backtest"]["reference_solver_origins"] = [
        origin for origin in daily_origins if origin not in search_origins
    ]
    selection["frozen_daily_backtest"]["reference_solver_reused_search_origins"] = search_origins
    contract["runtime_preflight"]["designated_existing_search_components"]["origin"] = search_origins[-1]

    reference_total = 12 + 2 * len(selection["frozen_daily_backtest"]["reference_solver_origins"])
    projection = contract["runtime_preflight"]["projection"]
    projection["total_reference_fits"] = reference_total
    projection["remaining_reference_fits"] = reference_total - 2
    budget = contract["operational_budget"]
    budget["maximum_search_reference_solver_fit_runs"] = 12
    budget["maximum_daily_reference_solver_fit_runs"] = reference_total - 12
    budget["maximum_reference_solver_fit_runs"] = reference_total
    budget["maximum_daily_SGD_fit_runs"] = 2 * len(daily_origins)
    budget["maximum_daily_primary_calibrator_fit_runs"] = 2 * len(daily_origins)
    primary = (
        budget["maximum_search_SGD_fit_runs"]
        + budget["maximum_search_primary_calibrator_fit_runs"]
        + budget["maximum_daily_SGD_fit_runs"]
        + budget["maximum_daily_primary_calibrator_fit_runs"]
    )
    budget["maximum_primary_component_fit_runs"] = primary
    budget["maximum_total_fit_operations"] = (
        primary + reference_total + budget["maximum_diagnostic_calibration_regression_fits"]
    )
    projection["total_SGD_fits"] = (
        budget["maximum_search_SGD_fit_runs"] + budget["maximum_daily_SGD_fit_runs"]
    )
    projection["remaining_SGD_fits"] = projection["total_SGD_fits"] - 2
    projection["total_primary_calibrator_fits"] = (
        budget["maximum_search_primary_calibrator_fit_runs"]
        + budget["maximum_daily_primary_calibrator_fit_runs"]
    )
    projection["remaining_primary_calibrator_fits"] = (
        projection["total_primary_calibrator_fits"] - 2
    )

    stability = contract["probability_quality_gate"]["daily_stability_requirements"]
    stability["total_days"] = len(daily_origins)
    contract["relative_history_gate"]["daily_requirements"]["total_days"] = len(daily_origins)
    # keep the digest guard active by restating it for the synthetic dimensions
    synthetic_users = int(np.unique(columns["user_id"]).size)
    contract["bootstrap"]["replicates"] = 200
    contract["bootstrap"]["expected_users"] = synthetic_users
    _, digest = runner.metrics.make_multiplicities(
        user_count=synthetic_users,
        replicates=200,
        seed=int(contract["bootstrap"]["seed"]),
    )
    contract["bootstrap"]["expected_multiplicity_matrix_sha256"] = digest
    return contract


class EndToEndTests(unittest.TestCase):
    def test_one_search_configuration_can_nonconverge_without_aborting_the_grid(self):
        class AfterSearch(RuntimeError):
            pass

        with TemporaryDirectory() as raw:
            workspace = Path(raw)
            feature_path = workspace / "synthetic_features.parquet"
            columns = build_feature_table(feature_path)
            contract = build_contract(feature_path, columns)
            designated_pair = contract["runtime_preflight"][
                "designated_existing_search_components"
            ]["pair_id"]
            target = next(
                pair
                for pair in runner.repair.paired_configurations()
                if pair["pair_id"] != designated_pair
            )

            contract_path = workspace / "synthetic_contract.yaml"
            contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
            contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
            output_dir = workspace / "out"
            original = (
                runner.CONTRACT_PATH,
                runner.OUTPUT_DIR,
                runner.REPORT_PATH,
                runner.FIGURE_PATH,
                runner.ACTIVE_BUDGET,
                runner.ACTIVE_BOOK,
                runner.ACTIVE_RUN_STARTED,
                runner.ACTIVE_ARTIFACT_SNAPSHOT,
                runner.ACTIVE_RUN_ID,
                set(runner.ACTIVE_PRODUCED_ARTIFACTS),
            )
            runner.CONTRACT_PATH = contract_path
            runner.OUTPUT_DIR = output_dir
            runner.REPORT_PATH = output_dir / "reports/analysis/gate2b_probability_repair_results_v003.md"
            runner.FIGURE_PATH = output_dir / "reports/figures/gate2b_probability_repair_results_v003.png"
            runner.ACTIVE_RUN_ID = None
            runner.ACTIVE_PRODUCED_ARTIFACTS = set()

            genuine_fit = runner.repair.fit_sgd
            genuine_daily = runner.frozen_daily_backtest
            genuine_AP_check = runner.verify_raw_calibrated_ap_equivalence
            reached_daily = {"value": False}
            AP_check_stages = []

            def one_bad_pair(x, y, *, alpha, eta0, **kwargs):
                model, record = genuine_fit(x, y, alpha=alpha, eta0=eta0, **kwargs)
                if np.isclose(alpha, target["alpha"]) and np.isclose(eta0, target["eta0"]):
                    record.converged = False
                    record.n_iter = record.max_iter
                    record.convergence_warning_count = max(
                        1, record.convergence_warning_count
                    )
                return model, record

            def stop_after_search(*args, **kwargs):
                reached_daily["value"] = True
                raise AfterSearch("search completed")

            def AP_check_after_freeze(*args, **kwargs):
                stage = kwargs["stage"]
                AP_check_stages.append(stage)
                if stage == "search":
                    self.assertTrue(
                        (output_dir / "search_artifact_freeze_manifest.json").is_file()
                    )
                return genuine_AP_check(*args, **kwargs)

            search_rule_evaluations = None
            try:
                runner.repair.fit_sgd = one_bad_pair
                runner.frozen_daily_backtest = stop_after_search
                runner.verify_raw_calibrated_ap_equivalence = AP_check_after_freeze
                with self.assertRaises(AfterSearch):
                    runner.release(contract_hash)
                search_rule_evaluations = runner.ACTIVE_BUDGET.nonfitted_rule_evaluations
            finally:
                runner.repair.fit_sgd = genuine_fit
                runner.frozen_daily_backtest = genuine_daily
                runner.verify_raw_calibrated_ap_equivalence = genuine_AP_check
                (
                    runner.CONTRACT_PATH,
                    runner.OUTPUT_DIR,
                    runner.REPORT_PATH,
                    runner.FIGURE_PATH,
                    runner.ACTIVE_BUDGET,
                    runner.ACTIVE_BOOK,
                    runner.ACTIVE_RUN_STARTED,
                    runner.ACTIVE_ARTIFACT_SNAPSHOT,
                    runner.ACTIVE_RUN_ID,
                    runner.ACTIVE_PRODUCED_ARTIFACTS,
                ) = original

            self.assertTrue(reached_daily["value"])
            self.assertTrue(AP_check_stages)
            self.assertEqual(set(AP_check_stages), {"search"})
            self.assertEqual(search_rule_evaluations, 6)
            with (output_dir / "search_trial_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                trial_rows = list(csv.DictReader(handle))
            target_rows = [row for row in trial_rows if row["pair_id"] == target["pair_id"]]
            self.assertEqual(len(target_rows), 6)
            self.assertTrue(all(row["eligible"] == "False" for row in target_rows))
            self.assertTrue(
                all("SGD_nonconverged" in row["ineligible_reason"] for row in target_rows)
            )
            selected = yaml.safe_load(
                (output_dir / "selected_models.json").read_text(encoding="utf-8")
            )
            self.assertNotEqual(selected["primary_shared_pair_id"], target["pair_id"])

            with (output_dir / "search_fixed_history_rule_diagnostics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                search_rule_rows = list(csv.DictReader(handle))
            self.assertEqual(len(search_rule_rows), 8)
            self.assertEqual(
                {row["origin_or_pooled"] for row in search_rule_rows},
                {
                    *contract["search_and_selection"]["search_origins"],
                    "pooled_3_origins",
                },
            )
            self.assertEqual(
                {row["diagnostic_id"] for row in search_rule_rows},
                {"LIFETIME_SMOOTHED_RATE", "W10_SMOOTHED_RATE"},
            )
            freeze = yaml.safe_load(
                (output_dir / "search_artifact_freeze_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(freeze["fixed_rule_scores"]), 6)
            self.assertTrue(all(row["sha256"] for row in freeze["fixed_rule_scores"]))

    def test_search_daily_fixed_rule_overlap_requires_bit_exact_raw_scores(self):
        with TemporaryDirectory() as raw:
            workspace = Path(raw)
            feature_path = workspace / "synthetic_features.parquet"
            columns = build_feature_table(feature_path)
            contract = build_contract(feature_path, columns)
            contract_path = workspace / "synthetic_contract.yaml"
            contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
            contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
            output_dir = workspace / "out"
            original = (
                runner.CONTRACT_PATH,
                runner.OUTPUT_DIR,
                runner.REPORT_PATH,
                runner.FIGURE_PATH,
                runner.ACTIVE_BUDGET,
                runner.ACTIVE_BOOK,
                runner.ACTIVE_RUN_STARTED,
                runner.ACTIVE_ARTIFACT_SNAPSHOT,
                runner.ACTIVE_RUN_ID,
                set(runner.ACTIVE_PRODUCED_ARTIFACTS),
            )
            runner.CONTRACT_PATH = contract_path
            runner.OUTPUT_DIR = output_dir
            runner.REPORT_PATH = output_dir / "report.md"
            runner.FIGURE_PATH = output_dir / "figure.png"
            runner.ACTIVE_RUN_ID = None
            runner.ACTIVE_PRODUCED_ARTIFACTS = set()
            genuine_daily = runner.frozen_daily_backtest
            genuine_adequacy = runner.repair.adequacy_decision
            genuine_AP_check = runner.verify_raw_calibrated_ap_equivalence
            adequacy_calls = {"count": 0}

            def counted_adequacy(*args, **kwargs):
                adequacy_calls["count"] += 1
                return genuine_adequacy(*args, **kwargs)

            def AP_check_after_stage_gate(*args, **kwargs):
                if kwargs["stage"] == "frozen_daily":
                    # 3 origins x 4 pairs x 2 models in search, then every
                    # 7 origins x 2 models in frozen daily, before any daily AP.
                    self.assertGreaterEqual(adequacy_calls["count"], 24 + 14)
                return genuine_AP_check(*args, **kwargs)

            def corrupt_one_frozen_score(*args, **kwargs):
                search = args[-1]
                key = sorted(search["frozen_rule_scores"])[0]
                corrupted = np.asarray(
                    search["frozen_rule_scores"][key], dtype=np.float64
                ).copy()
                corrupted[0] = np.nextafter(corrupted[0], np.inf)
                search["frozen_rule_scores"][key] = corrupted
                return genuine_daily(*args, **kwargs)

            try:
                runner.frozen_daily_backtest = corrupt_one_frozen_score
                runner.repair.adequacy_decision = counted_adequacy
                runner.verify_raw_calibrated_ap_equivalence = AP_check_after_stage_gate
                with self.assertRaises(runner.ContractStop) as caught:
                    runner.release(contract_hash)
            finally:
                runner.frozen_daily_backtest = genuine_daily
                runner.repair.adequacy_decision = genuine_adequacy
                runner.verify_raw_calibrated_ap_equivalence = genuine_AP_check
                (
                    runner.CONTRACT_PATH,
                    runner.OUTPUT_DIR,
                    runner.REPORT_PATH,
                    runner.FIGURE_PATH,
                    runner.ACTIVE_BUDGET,
                    runner.ACTIVE_BOOK,
                    runner.ACTIVE_RUN_STARTED,
                    runner.ACTIVE_ARTIFACT_SNAPSHOT,
                    runner.ACTIVE_RUN_ID,
                    runner.ACTIVE_PRODUCED_ARTIFACTS,
                ) = original

            self.assertIn("fixed-rule raw score mismatch", str(caught.exception))

    def test_all_search_sgd_nonconvergence_maps_to_exit_9(self):
        with TemporaryDirectory() as raw:
            workspace = Path(raw)
            feature_path = workspace / "synthetic_features.parquet"
            columns = build_feature_table(feature_path)
            contract = build_contract(feature_path, columns)
            contract_path = workspace / "synthetic_contract.yaml"
            contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
            contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
            output_dir = workspace / "out"

            genuine_fit = runner.repair.fit_sgd

            def every_sgd_is_ineligible(*args, **kwargs):
                model, record = genuine_fit(*args, **kwargs)
                record.converged = False
                record.n_iter = record.max_iter
                record.convergence_warning_count = max(
                    1, record.convergence_warning_count
                )
                return model, record

            original = (
                runner.CONTRACT_PATH,
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
            runner.CONTRACT_PATH = contract_path
            runner.OUTPUT_DIR = output_dir
            runner.REPORT_PATH = output_dir / "report.md"
            runner.FIGURE_PATH = output_dir / "figure.png"
            runner.ACTIVE_BUDGET = None
            runner.ACTIVE_BOOK = None
            runner.ACTIVE_RUN_STARTED = False
            runner.ACTIVE_ARTIFACT_SNAPSHOT = {}
            runner.ACTIVE_RUN_ID = None
            runner.ACTIVE_PRODUCED_ARTIFACTS = set()
            sys.argv = [
                "runner",
                "--release",
                "--approve-contract-hash",
                contract_hash,
            ]
            runner.repair.fit_sgd = every_sgd_is_ineligible
            try:
                code = runner.main()
            finally:
                runner.repair.fit_sgd = genuine_fit
                (
                    runner.CONTRACT_PATH,
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

            self.assertEqual(code, runner.EXIT_SEARCH_COMPONENT_CONVERGENCE_FAILURE)
            terminal = yaml.safe_load(
                (output_dir / "terminal_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(terminal["state"], "search_component_convergence_failure")
            self.assertEqual(terminal["exit_code"], 9)
            self.assertEqual(terminal["completion_scope"], "partial_failure")
            with (output_dir / "search_trial_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                trial_rows = list(csv.DictReader(handle))
            bl1_rows = [row for row in trial_rows if row["model_id"] == "BL1"]
            self.assertEqual(len(bl1_rows), 12)
            self.assertTrue(all(row["eligible"] == "False" for row in bl1_rows))
            self.assertTrue(
                all("SGD_nonconverged" in row["ineligible_reason"] for row in bl1_rows)
            )

    def test_late_daily_calibrator_failure_precedes_every_daily_assessment_metric(self):
        with TemporaryDirectory() as raw:
            workspace = Path(raw)
            feature_path = workspace / "synthetic_features.parquet"
            columns = build_feature_table(feature_path)
            contract = build_contract(feature_path, columns)
            contract_path = workspace / "synthetic_contract.yaml"
            contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
            contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
            output_dir = workspace / "out"

            state = {
                "current_stage": None,
                "daily_calibrator_calls": 0,
                "daily_started": False,
                "assessment_calls_after_daily_started": [],
            }
            genuine_calibrate = runner.calibrate_component
            genuine_fit_calibrator = runner.repair.fit_previous_day_sigmoid
            guarded_names = (
                "regret_diagnostics",
                "verify_raw_calibrated_ap_equivalence",
                "assessment_metrics",
                "contract_named_metrics",
                "uncalibrated_named_metrics",
                "raw_score_within_origin_metrics",
                "probability_distribution",
            )
            genuine_metrics = {name: getattr(runner, name) for name in guarded_names}

            def stage_aware_calibrate(*args, **kwargs):
                stage = kwargs["stage"]
                state["current_stage"] = stage
                if stage == "frozen_daily":
                    state["daily_started"] = True
                    state["daily_calibrator_calls"] += 1
                try:
                    return genuine_calibrate(*args, **kwargs)
                finally:
                    state["current_stage"] = None

            def fail_the_fourteenth_daily_calibrator(*args, **kwargs):
                if (
                    state["current_stage"] == "frozen_daily"
                    and state["daily_calibrator_calls"] == 14
                ):
                    raise runner.repair.ContractViolation(
                        "calibration did not converge inside the frozen cap"
                    )
                return genuine_fit_calibrator(*args, **kwargs)

            def guarded_metric(name):
                def call(*args, **kwargs):
                    if state["daily_started"]:
                        state["assessment_calls_after_daily_started"].append(name)
                    return genuine_metrics[name](*args, **kwargs)

                return call

            original = (
                runner.CONTRACT_PATH,
                runner.OUTPUT_DIR,
                runner.REPORT_PATH,
                runner.FIGURE_PATH,
                runner.ACTIVE_BUDGET,
                runner.ACTIVE_BOOK,
                runner.ACTIVE_RUN_STARTED,
                runner.ACTIVE_ARTIFACT_SNAPSHOT,
                runner.ACTIVE_RUN_ID,
                set(runner.ACTIVE_PRODUCED_ARTIFACTS),
            )
            runner.CONTRACT_PATH = contract_path
            runner.OUTPUT_DIR = output_dir
            runner.REPORT_PATH = output_dir / "report.md"
            runner.FIGURE_PATH = output_dir / "figure.png"
            runner.ACTIVE_RUN_ID = None
            runner.ACTIVE_PRODUCED_ARTIFACTS = set()
            runner.calibrate_component = stage_aware_calibrate
            runner.repair.fit_previous_day_sigmoid = fail_the_fourteenth_daily_calibrator
            for name in guarded_names:
                setattr(runner, name, guarded_metric(name))
            try:
                with self.assertRaises(runner.TerminalStop) as caught:
                    runner.release(contract_hash)
            finally:
                runner.calibrate_component = genuine_calibrate
                runner.repair.fit_previous_day_sigmoid = genuine_fit_calibrator
                for name, function in genuine_metrics.items():
                    setattr(runner, name, function)
                (
                    runner.CONTRACT_PATH,
                    runner.OUTPUT_DIR,
                    runner.REPORT_PATH,
                    runner.FIGURE_PATH,
                    runner.ACTIVE_BUDGET,
                    runner.ACTIVE_BOOK,
                    runner.ACTIVE_RUN_STARTED,
                    runner.ACTIVE_ARTIFACT_SNAPSHOT,
                    runner.ACTIVE_RUN_ID,
                    runner.ACTIVE_PRODUCED_ARTIFACTS,
                ) = original

            self.assertEqual(state["daily_calibrator_calls"], 14)
            self.assertEqual(state["assessment_calls_after_daily_started"], [])
            self.assertEqual(
                caught.exception.state, "frozen_daily_component_convergence_failure"
            )
            self.assertEqual(
                caught.exception.exit_code,
                runner.EXIT_FROZEN_DAILY_COMPONENT_CONVERGENCE_FAILURE,
            )
            self.assertEqual(caught.exception.stage, "frozen_daily")

    def test_full_release_path_produces_every_declared_artifact(self):
        with TemporaryDirectory() as raw:
            workspace = Path(raw)
            feature_path = workspace / "synthetic_features.parquet"
            columns = build_feature_table(feature_path)
            contract = build_contract(feature_path, columns)

            contract_path = workspace / "synthetic_contract.yaml"
            contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
            contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()

            output_dir = workspace / "out"
            # PROJECT_ROOT is left alone: joining it with an absolute fixture
            # path already yields the fixture path, and the run manifest hashes
            # the real source modules.
            original = (
                runner.CONTRACT_PATH,
                runner.OUTPUT_DIR,
                runner.REPORT_PATH,
                runner.FIGURE_PATH,
                runner.ACTIVE_BUDGET,
                runner.ACTIVE_BOOK,
                runner.ACTIVE_RUN_STARTED,
                runner.ACTIVE_ARTIFACT_SNAPSHOT,
                runner.ACTIVE_RUN_ID,
                set(runner.ACTIVE_PRODUCED_ARTIFACTS),
            )
            runner.CONTRACT_PATH = contract_path
            runner.OUTPUT_DIR = output_dir
            runner.REPORT_PATH = output_dir / "reports/analysis/gate2b_probability_repair_results_v003.md"
            runner.FIGURE_PATH = output_dir / "reports/figures/gate2b_probability_repair_results_v003.png"
            runner.ACTIVE_RUN_ID = None
            runner.ACTIVE_PRODUCED_ARTIFACTS = set()
            try:
                code = runner.release(contract_hash)
            finally:
                (
                    runner.CONTRACT_PATH,
                    runner.OUTPUT_DIR,
                    runner.REPORT_PATH,
                    runner.FIGURE_PATH,
                    runner.ACTIVE_BUDGET,
                    runner.ACTIVE_BOOK,
                    runner.ACTIVE_RUN_STARTED,
                    runner.ACTIVE_ARTIFACT_SNAPSHOT,
                    runner.ACTIVE_RUN_ID,
                    runner.ACTIVE_PRODUCED_ARTIFACTS,
                ) = original

            # 0 = pass, 3 = probability fail, 4 = relative history fail.
            # Any of these is a legitimate scientific terminal state; a crash is not.
            self.assertIn(code, (0, 3, 4))

            for name in runner.ARTIFACT_NAMES:
                if name.startswith("reports/"):
                    produced = output_dir / name
                elif name.startswith("search_predictions_origin_"):
                    continue  # origin names differ in the synthetic fixture
                else:
                    produced = output_dir / name
                self.assertTrue(produced.is_file(), msg=f"missing artifact {name}")
                self.assertGreater(produced.stat().st_size, 0, msg=f"empty artifact {name}")

            search_predictions = sorted(output_dir.glob("search_predictions_origin_*.parquet"))
            self.assertEqual(len(search_predictions), 3)

            manifest = yaml.safe_load((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
            schemas = contract["artifact_schemas"]
            self.assertFalse(
                set(schemas["run_manifest"]["required_fields"]) - set(manifest),
                msg="run_manifest.json is missing contract-required fields",
            )
            self.assertEqual(manifest["contract_sha256"], contract_hash)
            self.assertIn("probability_quality_gate_passed", manifest)
            self.assertIn("relative_history_gate_passed", manifest)
            self.assertEqual(manifest["diagnostic_calibration_regression_declared_slots"], 24)
            self.assertEqual(manifest["diagnostic_calibration_regression_executed_fits"], 16)
            self.assertEqual(manifest["nonfitted_history_rule_evaluations"], 20)
            self.assertEqual(len(manifest["artifact_inventory"]), 34)
            self.assertEqual(
                len({row["path"] for row in manifest["artifact_inventory"]}), 34
            )
            runner.require_fields(
                manifest["artifact_inventory"],
                schemas["run_manifest"]["full_completion_artifact_inventory"][
                    "entry_required_fields"
                ],
                "full run_manifest artifact inventory",
            )
            self.assertLessEqual(
                manifest["fit_operations"],
                contract["operational_budget"]["maximum_total_fit_operations"],
            )

            # D7: these are release-governance records, not merely information
            # duplicated in preprocessing_manifest.json.
            manifest_parameter_snapshots = manifest["get_params_snapshots"]
            runner.require_fields(
                manifest_parameter_snapshots,
                schemas["get_params_snapshots"]["entry_required_fields"],
                "run_manifest get_params snapshots",
            )
            required_component_families = set(
                schemas["get_params_snapshots"]["required_components"]
            )
            component_ids = [row["component_id"] for row in manifest_parameter_snapshots]
            self.assertEqual(len(component_ids), len(set(component_ids)))
            observed_component_families = {
                family
                for component_id in component_ids
                for family in required_component_families
                if component_id.startswith(f"{family}_")
            }
            self.assertEqual(observed_component_families, required_component_families)
            self.assertTrue(
                all(
                    any(
                        component_id.startswith(f"{family}_")
                        for family in required_component_families
                    )
                    for component_id in component_ids
                )
            )
            self.assertTrue(
                all(isinstance(row["get_params"], dict) for row in manifest_parameter_snapshots)
            )

            terminal = yaml.safe_load(
                (output_dir / "terminal_state.json").read_text(encoding="utf-8")
            )
            self.assertFalse(
                set(schemas["terminal_state"]["required_fields"]) - set(terminal),
                msg="terminal_state.json is missing contract-required fields",
            )
            self.assertEqual(manifest["terminal_state"], terminal["state"])
            self.assertEqual(manifest["exit_code"], terminal["exit_code"])

            preprocessing = yaml.safe_load(
                (output_dir / "preprocessing_manifest.json").read_text(encoding="utf-8")
            )
            parameter_snapshots = preprocessing["sklearn_get_params_snapshots"]
            self.assertEqual(
                set(parameter_snapshots),
                {
                    "preprocessing_by_origin",
                    "SGD_estimators",
                    "reference_solvers",
                    "primary_calibrators",
                    "assessment_calibration_regressions",
                },
            )
            self.assertEqual(len(parameter_snapshots["preprocessing_by_origin"]), 7)
            for name in (
                "SGD_estimators",
                "reference_solvers",
                "primary_calibrators",
                "assessment_calibration_regressions",
            ):
                self.assertTrue(parameter_snapshots[name], msg=f"empty snapshot group {name}")

            with (output_dir / "fixed_history_rule_diagnostics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                fixed_rule_rows = list(csv.DictReader(handle))
            self.assertEqual(len(fixed_rule_rows), 24)
            self.assertEqual(
                sum(row["stage"] == "search" for row in fixed_rule_rows), 8
            )
            self.assertEqual(
                sum(row["stage"] == "frozen_daily" for row in fixed_rule_rows), 16
            )

            # the primary comparison must be at one shared configuration
            selected = yaml.safe_load(
                (output_dir / "selected_primary_shared_configuration.json").read_text(
                    encoding="utf-8"
                )
            )
            registry = {item["pair_id"] for item in runner.repair.paired_configurations()}
            self.assertIn(selected["pair_id"], registry)

            # every daily origin carries an adequacy verdict
            audit = (output_dir / "optimization_objective_audit.csv").read_text(encoding="utf-8")
            for origin in contract["search_and_selection"]["frozen_daily_backtest"]["origins"]:
                self.assertIn(origin, audit)

            # the snapshot is a byte-exact copy of the contract that was run
            snapshot = (output_dir / "contract_snapshot.yaml").read_bytes()
            self.assertEqual(hashlib.sha256(snapshot).hexdigest(), contract_hash)

            # The 36th artifact is the self-excluding hash manifest, so it must
            # cover exactly the other 35 required artifacts and each digest must
            # match the finalized bytes.
            hash_manifest = yaml.safe_load(
                (output_dir / "artifact_hash_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(
                set(schemas["artifact_hash_manifest"]["top_level_required_fields"])
                - set(hash_manifest),
                msg="artifact_hash_manifest.json is missing contract-required fields",
            )
            self.assertEqual(
                hash_manifest["status"], "complete_all_required_artifacts_verified"
            )
            self.assertEqual(hash_manifest["required_artifact_count"], 36)
            self.assertEqual(hash_manifest["artifact_count"], 35)
            self.assertEqual(hash_manifest["missing_required_artifacts"], [])
            self.assertEqual(len(hash_manifest["artifacts"]), 35)
            runner.require_fields(
                hash_manifest["artifacts"],
                schemas["artifact_hash_manifest"]["artifact_entry_required_fields"],
                "artifact hash entries",
            )
            self.assertEqual(
                {terminal["run_id"], manifest["run_id"], hash_manifest["run_id"]},
                {terminal["run_id"]},
            )
            self.assertTrue(terminal["run_id"])
            self.assertEqual(
                {
                    terminal["completion_scope"],
                    manifest["completion_scope"],
                    hash_manifest["completion_scope"],
                },
                {"full"},
            )
            for row in hash_manifest["artifacts"]:
                if row["path"].startswith("reports/"):
                    artifact = output_dir / row["path"]
                else:
                    artifact = output_dir / row["path"]
                self.assertEqual(runner.sha256_file(artifact), row["sha256"])

            # B2 regression guard: the Markdown report must contain actual
            # slice values, not merely a header generated from wrong field names.
            report = (
                output_dir / "reports/analysis/gate2b_probability_repair_results_v003.md"
            ).read_text(encoding="utf-8")
            self.assertIn("| pooled_seven_days | all_assessment_rows |", report)
            self.assertRegex(
                report,
                r"\| pooled_seven_days \| all_assessment_rows \| \d+ \| [0-9.]+ \|",
            )
            self.assertIn("## Pooled uncalibrated versus calibrated probability diagnostics", report)
            self.assertIn(
                "## Reference objective gaps and assessment regret diagnostics (selected pair)",
                report,
            )
            self.assertIn("## Same-configuration sensitivity across all four pairs", report)
            self.assertIn(
                "Complete paired BL1/BL2 configurations available for optimizer-robustness",
                report,
            )

    def test_daily_adequacy_failure_stops_before_metric_aggregation(self):
        """A bad day must fail the release closed, not be excluded or reweighted."""

        with TemporaryDirectory() as raw:
            workspace = Path(raw)
            feature_path = workspace / "synthetic_features.parquet"
            columns = build_feature_table(feature_path)
            contract = build_contract(feature_path, columns)
            contract_path = workspace / "synthetic_contract.yaml"
            contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
            contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()

            # Let the search phase pass normally, then make the very first frozen
            # daily verdict inadequate.  The search phase issues one decision per
            # (origin, pair, model); everything after that belongs to the daily
            # phase, which is the branch under test.
            search_decisions = (
                len(contract["search_and_selection"]["search_origins"])
                * len(runner.repair.paired_configurations())
                * len(runner.MODEL_IDS)
            )
            genuine = runner.repair.adequacy_decision
            state = {"calls": 0}

            def failing_after_search(*args, **kwargs):
                decision = genuine(*args, **kwargs)
                state["calls"] += 1
                if state["calls"] > search_decisions:
                    decision["adequacy_passed"] = False
                return decision

            output_dir = workspace / "out"
            original = (
                runner.CONTRACT_PATH,
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
            runner.CONTRACT_PATH = contract_path
            runner.OUTPUT_DIR = output_dir
            runner.REPORT_PATH = output_dir / "reports/analysis/gate2b_probability_repair_results_v003.md"
            runner.FIGURE_PATH = output_dir / "reports/figures/gate2b_probability_repair_results_v003.png"
            runner.ACTIVE_BUDGET = None
            runner.ACTIVE_BOOK = None
            runner.ACTIVE_RUN_STARTED = False
            runner.ACTIVE_ARTIFACT_SNAPSHOT = {}
            runner.ACTIVE_RUN_ID = None
            runner.ACTIVE_PRODUCED_ARTIFACTS = set()
            sys.argv = [
                "runner",
                "--release",
                "--approve-contract-hash",
                contract_hash,
            ]
            runner.repair.adequacy_decision = failing_after_search
            try:
                code = runner.main()
            finally:
                runner.repair.adequacy_decision = genuine
                (
                    runner.CONTRACT_PATH,
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
            self.assertGreater(state["calls"], search_decisions)
            terminal = yaml.safe_load(
                (output_dir / "terminal_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                terminal["state"], "frozen_daily_optimization_adequacy_failure"
            )
            self.assertEqual(terminal["stage"], "frozen_daily")
            self.assertIn("frozen daily optimization adequacy failed", terminal["message"])
            self.assertIn(
                "Day-level exclusion or reweighting is forbidden", terminal["message"]
            )

            # C1/C15: the search checkpoint and all ledgers needed to audit the
            # stop remain available, and the partial hash manifest closes them.
            for artifact in (
                "search_artifact_freeze_manifest.json",
                "search_trial_manifest.csv",
                "search_metrics.csv",
                "paired_configuration_metrics.csv",
                "selected_models.json",
                "search_fixed_history_rule_diagnostics.csv",
                "optimization_objective_audit.csv",
                "convergence_ledger.csv",
                "reference_solver_ledger.csv",
                "calibration_fit_ledger.csv",
                "usage_ledger.csv",
                "terminal_state.json",
                "run_manifest.json",
                "artifact_hash_manifest.json",
            ):
                self.assertTrue((output_dir / artifact).is_file(), msg=f"missing {artifact}")
            self.assertEqual(
                len(list(output_dir.glob("search_predictions_origin_*.parquet"))), 3
            )

            failure_hashes = yaml.safe_load(
                (output_dir / "artifact_hash_manifest.json").read_text(encoding="utf-8")
            )
            failure_manifest = yaml.safe_load(
                (output_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            schemas = contract["artifact_schemas"]
            self.assertFalse(
                set(schemas["terminal_state"]["required_fields"]) - set(terminal)
            )
            self.assertFalse(
                set(schemas["run_manifest"]["required_fields"]) - set(failure_manifest)
            )
            self.assertFalse(
                set(schemas["artifact_hash_manifest"]["top_level_required_fields"])
                - set(failure_hashes)
            )
            self.assertEqual(failure_hashes["status"], "partial_fail_closed_checkpoint")
            self.assertEqual(
                failure_hashes["terminal_state"],
                "frozen_daily_optimization_adequacy_failure",
            )
            self.assertEqual(
                {
                    terminal["run_id"],
                    failure_manifest["run_id"],
                    failure_hashes["run_id"],
                },
                {terminal["run_id"]},
            )
            self.assertTrue(terminal["run_id"])
            self.assertEqual(
                {
                    terminal["completion_scope"],
                    failure_manifest["completion_scope"],
                    failure_hashes["completion_scope"],
                },
                {"partial_failure"},
            )
            self.assertEqual(failure_manifest["terminal_state"], terminal["state"])
            self.assertEqual(failure_manifest["exit_code"], terminal["exit_code"])
            failure_parameter_snapshots = failure_manifest["get_params_snapshots"]
            runner.require_fields(
                failure_parameter_snapshots,
                schemas["get_params_snapshots"]["entry_required_fields"],
                "partial run_manifest get_params snapshots",
            )
            required_component_families = set(
                schemas["get_params_snapshots"]["required_components"]
            )
            self.assertEqual(
                {
                    family
                    for row in failure_parameter_snapshots
                    for family in required_component_families
                    if row["component_id"].startswith(f"{family}_")
                },
                required_component_families,
            )
            runner.require_fields(
                failure_manifest["artifact_inventory"],
                schemas["run_manifest"]["full_completion_artifact_inventory"][
                    "entry_required_fields"
                ],
                "partial run_manifest artifact inventory",
            )
            self.assertEqual(failure_hashes["required_artifact_count"], 36)
            self.assertEqual(
                failure_hashes["artifact_count"], len(failure_hashes["artifacts"])
            )
            self.assertIn("daily_predictions.parquet", failure_hashes["missing_required_artifacts"])
            runner.require_fields(
                failure_hashes["artifacts"],
                schemas["artifact_hash_manifest"]["artifact_entry_required_fields"],
                "partial artifact hash entries",
            )
            indexed = {row["path"]: row for row in failure_hashes["artifacts"]}
            self.assertIn("terminal_state.json", indexed)
            self.assertIn("run_manifest.json", indexed)
            self.assertEqual(
                {row["path"] for row in failure_manifest["artifact_inventory"]},
                set(indexed) - {"run_manifest.json"},
            )
            self.assertEqual(
                indexed["terminal_state.json"]["sha256"],
                runner.sha256_file(output_dir / "terminal_state.json"),
            )
            # nothing downstream of the gate may have been written
            for artifact in (
                "pooled_and_slice_metrics.csv",
                "daily_metrics.csv",
                "daily_predictions.parquet",
                "paired_user_cluster_bootstrap.csv",
            ):
                self.assertFalse(
                    (output_dir / artifact).exists(), msg=f"{artifact} written after a stop"
                )


if __name__ == "__main__":
    unittest.main()
