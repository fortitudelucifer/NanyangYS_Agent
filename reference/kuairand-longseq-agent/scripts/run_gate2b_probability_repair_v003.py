"""Gate 2B v003 probability-repair runner.

Executes exactly the protocol declared in
``configs/gate2b_probability_repair_contract_v003.yaml``.

Two modes:

``--validate-only``
    Verify the contract, the single frozen input artifact, the seven temporal
    splits against their frozen expected counts, and the budget arithmetic.
    Performs no fit and reads no scientific metric.  Always permitted.

``--release``
    The full protocol.  Refused unless the contract itself carries
    ``authorization.execution_authorized: true`` and the caller passes the exact
    approved contract hash.  Both conditions are deliberate: approval is a
    versioned edit to the contract, not a command-line flag.

The runner never reads Silver, the quarantine, Validation, the late table or the
random table.  Its only data input is the declared frozen feature artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy import sparse
from sklearn.metrics import average_precision_score
import threadpoolctl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kuairand_longseq.evaluation import gate2b_metrics as metrics  # noqa: E402
from kuairand_longseq.models import gate2b_repair_v003 as repair  # noqa: E402

CONTRACT_PATH = PROJECT_ROOT / "configs/gate2b_probability_repair_contract_v003.yaml"
OUTPUT_DIR = PROJECT_ROOT / "reports/generated/gate2b_probability_repair_v003"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/gate2b_probability_repair_results_v003.md"
FIGURE_PATH = PROJECT_ROOT / "reports/figures/gate2b_probability_repair_results_v003.png"

IDENTITY_COLUMNS = (
    "source_table",
    "source_row_number",
    "user_id",
    "video_id",
    "event_date",
    "long_view",
)
H2_RAW_COLUMNS = (
    "prior_batch_n",
    "prior_event_n",
    "prior_positive_n",
    "last_user_gap_s",
    "w10_event_n",
    "w10_positive_n",
    "w50_event_n",
    "w50_positive_n",
    "w200_event_n",
    "w200_positive_n",
)
REQUIRED_COLUMNS = (
    list(IDENTITY_COLUMNS)
    + list(repair.CATEGORICAL_FIELDS)
    + list(repair.STATIC_CONTINUOUS_FIELDS)
    + list(repair.STATIC_BINARY_FIELDS)
    + list(H2_RAW_COLUMNS)
)

MODEL_IDS = ("BL1", "BL2")
BUNDLES = {"BL1": "S_ID_CONTENT_V1_with_PIT_GROUPED_SCALE_V2", "BL2": "H2_USER_STRICT_V1_with_PIT_GROUPED_SCALE_V2"}
RAW_SCORE_QUANTILES = (0.0, 0.001, 0.01, 0.10, 0.50, 0.90, 0.99, 0.999, 1.0)
SLICE_BOUNDS = (
    ("all_assessment_rows", None, None),
    ("history_0_49", 0, 49),
    ("history_50_199", 50, 199),
    ("history_200_plus", 200, None),
    ("history_500_plus_exploratory", 500, None),
)
USAGE_LEDGER_FIELDS = (
    "fit_run_id",
    "component_type",
    "stage",
    "elapsed_seconds",
    "peak_RAM_GB",
    "artifact_storage_GB",
    "thread_limit",
    "observed_threadpool_max",
    "observed_threadpools",
    "status",
)


class ContractStop(RuntimeError):
    """Raised for any declared release stop condition."""


class TerminalStop(ContractStop):
    """A preregistered terminal state with a stable, distinct CLI exit code."""

    def __init__(self, state: str, exit_code: int, message: str, *, stage: str) -> None:
        super().__init__(message)
        self.state = state
        self.exit_code = int(exit_code)
        self.stage = stage


EXIT_IMPLEMENTATION_OR_GOVERNANCE_FAILURE = 2
EXIT_ABSOLUTE_PROBABILITY_FAIL = 3
EXIT_RELATIVE_HISTORY_FAIL = 4
EXIT_CASE_C_SEARCH_OPTIMIZATION_FAILURE = 5
EXIT_CASE_D_FROZEN_DAILY_OPTIMIZATION_FAILURE = 6
EXIT_SEARCH_PROBABILITY_QUALITY_FAILURE = 7
EXIT_SEARCH_PRIMARY_PAIR_GATE_FAILURE = 8
EXIT_SEARCH_COMPONENT_CONVERGENCE_FAILURE = 9
EXIT_FROZEN_DAILY_COMPONENT_CONVERGENCE_FAILURE = 10
RESOURCE_POLL_SECONDS = 0.05
ACTIVE_BUDGET = None
ACTIVE_BOOK = None
ACTIVE_RUN_STARTED = False
ACTIVE_ARTIFACT_SNAPSHOT: dict[str, tuple[bool, int, int]] = {}
ACTIVE_RUN_ID: str | None = None
ACTIVE_PRODUCED_ARTIFACTS: set[str] = set()
ACTIVE_ASSESSMENT_GET_PARAMS: set[str] = set()


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: yaml.Loader, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ContractStop(f"duplicate contract key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def register_produced_artifact(path: Path) -> None:
    if not ACTIVE_RUN_STARTED:
        return
    resolved = path.resolve()
    for name in globals().get("ARTIFACT_NAMES", ()):
        if artifact_path(str(name)).resolve() == resolved:
            ACTIVE_PRODUCED_ARTIFACTS.add(str(name))
            return


def json_safe(value: Any) -> Any:
    """Convert payloads to strict RFC-8259 JSON without losing infinity semantics."""

    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "numpy_nan"
        return "numpy_positive_infinity" if value > 0 else "numpy_negative_infinity"
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def sibling_temp_path(path: Path) -> Path:
    token = ACTIVE_RUN_ID or "nonrun"
    return path.with_name(
        f".{path.stem}.{token}.{time.time_ns()}.tmp{path.suffix}"
    )


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = sibling_temp_path(path)
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    register_produced_artifact(path)


def write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = sibling_temp_path(path)
    try:
        pq.write_table(table, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    register_produced_artifact(path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(
            json_safe(payload), indent=2, sort_keys=True, allow_nan=False, default=str
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in columns})
    atomic_write_text(path, buffer.getvalue())


def write_csv_or_header(
    path: Path, rows: list[dict[str, Any]], required_fields: Iterable[str]
) -> None:
    """Write normal rows, or a nonempty schema-only checkpoint when none exist."""

    if rows:
        write_csv(path, rows)
        return
    atomic_write_text(path, ",".join(str(name) for name in required_fields) + "\n")


def require_fields(rows: list[dict[str, Any]], required: Iterable[str], label: str) -> None:
    """Contract: every declared ledger schema must be produced in full."""

    if not rows:
        raise ContractStop(f"{label} has no rows; required fields: {sorted(required)}")
    required_set = set(required)
    for index, row in enumerate(rows):
        missing = sorted(required_set - set(row))
        if missing:
            raise ContractStop(f"{label} row {index} is missing required fields: {missing}")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def managed_artifact_size_bytes() -> int:
    total = directory_size_bytes(OUTPUT_DIR)
    for path in (REPORT_PATH, FIGURE_PATH):
        if path.is_file() and not path.is_relative_to(OUTPUT_DIR):
            total += path.stat().st_size
    return total


class ResourceMonitor:
    """Poll process RSS while a fitted component runs."""

    def __init__(self, interval_seconds: float = RESOURCE_POLL_SECONDS) -> None:
        self.interval_seconds = float(interval_seconds)
        self.process = psutil.Process()
        self.peak_rss_bytes = int(self.process.memory_info().rss)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ResourceMonitor":
        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes, int(self.process.memory_info().rss)
                )

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, int(self.process.memory_info().rss))
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_ram_gb(self) -> float:
        return self.peak_rss_bytes / float(1024**3)


def verify_implementation_hashes(contract: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    declared = contract["implementation_status"]["result_producing_implementation"]["files"]
    for item in declared:
        path = PROJECT_ROOT / item["path"]
        observed = sha256_file(path)
        if observed != item["sha256"]:
            raise ContractStop(
                f"implementation SHA-256 mismatch for {item['path']}: "
                f"expected {item['sha256']}, observed {observed}"
            )
        records.append({"path": item["path"], "sha256": observed, "verified": True})
    return records


def identity_digest(source_table: np.ndarray, source_row_number: np.ndarray) -> str:
    pairs = sorted(
        zip(np.asarray(source_table).astype(str), np.asarray(source_row_number, dtype=np.int64)),
        key=lambda item: (item[0], int(item[1])),
    )
    digest = hashlib.sha256()
    for source, row_number in pairs:
        digest.update(f"{source}\t{int(row_number)}\n".encode("utf-8"))
    return digest.hexdigest()


def write_terminal_state(
    *,
    state: str,
    exit_code: int,
    stage: str,
    message: str,
    contract_hash: str,
    completion_scope: str,
    budget: "Budget | None" = None,
) -> None:
    if ACTIVE_RUN_ID is None:
        raise ContractStop("an active release run_id is required before terminal finalization")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        OUTPUT_DIR / "terminal_state.json",
        {
            "run_id": ACTIVE_RUN_ID,
            "state": state,
            "exit_code": int(exit_code),
            "stage": stage,
            "message": message,
            "contract_sha256": contract_hash,
            "active_release_started": bool(ACTIVE_RUN_STARTED),
            "completion_scope": completion_scope,
            "fit_operations": 0 if budget is None else sum(budget.counts.values()),
            "fit_operations_by_type": {} if budget is None else dict(sorted(budget.counts.items())),
            "fit_operations_by_stage": {}
            if budget is None
            else dict(sorted(budget.stage_counts.items())),
            "nonfitted_history_rule_evaluations": 0
            if budget is None
            else int(budget.nonfitted_rule_evaluations),
        },
    )


def alpha_slug(alpha: float) -> str:
    return f"a{alpha:g}".replace("-", "m").replace(".", "p")


# ---------------------------------------------------------------------------
# contract and input verification
# ---------------------------------------------------------------------------


def load_contract() -> tuple[dict[str, Any], str]:
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    return yaml.load(text, Loader=UniqueKeyLoader), sha256_file(CONTRACT_PATH)


def verify_inputs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Contract: ``input_allowlist`` / ``input_policy``."""

    records: list[dict[str, Any]] = []
    allowlist = contract["input_allowlist"]
    if len(allowlist) != 1:
        raise ContractStop("v003 declares exactly one frozen input artifact")
    for entry in allowlist:
        path = PROJECT_ROOT / entry["path"]
        if not path.is_file():
            raise ContractStop(f"declared input missing: {entry['path']}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != entry["expected_size_bytes"]:
            raise ContractStop(f"input size mismatch for {entry['path']}")
        if digest != entry["expected_sha256"]:
            raise ContractStop(f"input SHA-256 mismatch for {entry['path']}")
        records.append(
            {
                "path": entry["path"],
                "role": entry["role"],
                "size_bytes": size,
                "sha256": digest,
                "verified": True,
            }
        )
    return records


def verify_environment(contract: dict[str, Any]) -> dict[str, Any]:
    import duckdb
    import scipy
    import sklearn

    declared = contract["execution_environment"]["release_versions"]
    observed = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pa.__version__,
        "duckdb": duckdb.__version__,
        "threadpoolctl": threadpoolctl.__version__,
    }
    unknown = sorted(set(declared) - set(observed))
    if unknown:
        raise ContractStop(
            f"contract declares release versions the runner does not observe: {unknown}"
        )
    if contract["execution_environment"]["exact_version_match_required"]:
        for name, expected in declared.items():
            if str(observed.get(name)) != str(expected):
                raise ContractStop(
                    f"environment version mismatch for {name}: "
                    f"expected {expected}, observed {observed.get(name)}"
                )
    return observed


def verify_execution_preconditions(contract: dict[str, Any]) -> None:
    required = contract["authorization"]["required_before_execution"]
    statuses = contract["authorization"]["required_before_execution_status"]
    for name in required:
        status = str(statuses.get(name, "missing"))
        if name == "actual_executable_budget_confirmed_not_to_exceed_the_frozen_planning_envelope":
            if not status.startswith("deferred_to_the_runtime_preflight"):
                raise ContractStop(f"execution precondition {name} is not delegated to preflight")
        elif not status.startswith("satisfied"):
            raise ContractStop(f"execution precondition {name} is not satisfied: {status}")


def verify_budget_arithmetic(contract: dict[str, Any]) -> dict[str, Any]:
    """Contract: ``operational_budget`` plus ``runtime_preflight.projection``."""

    budget = contract["operational_budget"]
    projection = contract["runtime_preflight"]["projection"]
    maximum_threads = int(budget["maximum_cpu_threads"])
    if int(budget["SGD_estimator_threads"]) > maximum_threads:
        raise ContractStop("SGD thread limit exceeds the global CPU-thread cap")
    if int(budget["reference_solver_threads"]) > maximum_threads:
        raise ContractStop("reference-solver thread limit exceeds the global CPU-thread cap")
    primary = (
        budget["maximum_search_SGD_fit_runs"]
        + budget["maximum_search_primary_calibrator_fit_runs"]
        + budget["maximum_daily_SGD_fit_runs"]
        + budget["maximum_daily_primary_calibrator_fit_runs"]
    )
    if primary != budget["maximum_primary_component_fit_runs"]:
        raise ContractStop("primary component fit cap does not reconcile")
    reference = (
        budget["maximum_search_reference_solver_fit_runs"]
        + budget["maximum_daily_reference_solver_fit_runs"]
    )
    if reference != budget["maximum_reference_solver_fit_runs"]:
        raise ContractStop("reference solver fit cap does not reconcile")
    total = primary + reference + budget["maximum_diagnostic_calibration_regression_fits"]
    if total != budget["maximum_total_fit_operations"]:
        raise ContractStop("total fit operation cap does not reconcile")
    for total_key, pre_key, remaining_key in (
        ("total_SGD_fits", "preflight_SGD_fits", "remaining_SGD_fits"),
        (
            "total_primary_calibrator_fits",
            "preflight_primary_calibrator_fits",
            "remaining_primary_calibrator_fits",
        ),
        ("total_reference_fits", "preflight_reference_fits", "remaining_reference_fits"),
    ):
        if projection[total_key] - projection[pre_key] != projection[remaining_key]:
            raise ContractStop(f"preflight projection count mismatch for {remaining_key}")
    if projection["total_reference_fits"] != budget["maximum_reference_solver_fit_runs"]:
        raise ContractStop("preflight reference total disagrees with the budget cap")
    expected_sgd = budget["maximum_search_SGD_fit_runs"] + budget["maximum_daily_SGD_fit_runs"]
    expected_calibrators = (
        budget["maximum_search_primary_calibrator_fit_runs"]
        + budget["maximum_daily_primary_calibrator_fit_runs"]
    )
    if projection["total_SGD_fits"] != expected_sgd:
        raise ContractStop("preflight SGD total disagrees with the stage caps")
    if projection["total_primary_calibrator_fits"] != expected_calibrators:
        raise ContractStop("preflight calibrator total disagrees with the stage caps")
    if (
        projection["diagnostic_calibration_regression_fits_using_primary_calibrator_time_proxy"]
        != budget["maximum_diagnostic_calibration_regression_fits"]
    ):
        raise ContractStop("preflight diagnostic total disagrees with the diagnostic cap")
    return {"primary": primary, "reference": reference, "total": total}


# ---------------------------------------------------------------------------
# data + splits
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    columns: dict[str, np.ndarray]

    @classmethod
    def read(cls, path: Path) -> "Frame":
        table = pq.read_table(path, columns=REQUIRED_COLUMNS)
        missing = sorted(set(REQUIRED_COLUMNS) - set(table.column_names))
        if missing:
            raise ContractStop(f"frozen artifact is missing required columns: {missing}")
        columns = {
            name: table.column(name).combine_chunks().to_numpy(zero_copy_only=False)
            for name in REQUIRED_COLUMNS
        }
        return cls(columns=columns)

    @property
    def size(self) -> int:
        return int(next(iter(self.columns.values())).shape[0])

    def dates(self) -> np.ndarray:
        return self.columns["event_date"].astype("datetime64[D]", copy=False)

    def labels(self) -> np.ndarray:
        return self.columns["long_view"].astype(np.int8, copy=False)

    def users(self) -> np.ndarray:
        return self.columns["user_id"].astype(np.int64, copy=False)


def verify_population(contract: dict[str, Any], frame: Frame) -> dict[str, Any]:
    population = contract["population"]
    observed = {
        "rows": frame.size,
        "users": int(np.unique(frame.users()).size),
        "videos": int(np.unique(frame.columns["video_id"]).size),
        "positives": int(frame.labels().sum()),
        "identity_sha256": identity_digest(
            frame.columns["source_table"], frame.columns["source_row_number"]
        ),
    }
    expected = {
        "rows": int(population["expected_target_rows"]),
        "users": int(population["expected_target_users"]),
        "videos": int(population["expected_target_videos"]),
        "positives": int(population["expected_target_positives"]),
        "identity_sha256": str(population["expected_all_target_identity_sha256"]),
    }
    if observed != expected:
        raise ContractStop(f"population mismatch: expected {expected}, observed {observed}")
    return observed


@dataclass
class OriginSplit:
    origin: str
    fit_index: np.ndarray
    calibration_index: np.ndarray
    assessment_index: np.ndarray
    calibration_date: str
    bl0_probability: float
    fit_prevalence: float


def build_splits(contract: dict[str, Any], frame: Frame) -> list[OriginSplit]:
    """Contract: ``temporal_protocol``.

    Enforces the frozen expected counts and the three no-overlap rules before
    any fit is attempted.
    """

    dates = frame.dates()
    labels = frame.labels()
    users = frame.users()
    splits: list[OriginSplit] = []
    minimums = contract["temporal_protocol"]["calibration_minimum_requirements"]

    for entry in contract["temporal_protocol"]["origin_splits"]:
        origin = str(entry["origin"])
        fit_lo, fit_hi = (
            np.datetime64(value, "D") for value in entry["estimator_fit_date_range_inclusive"]
        )
        calibration_day = np.datetime64(str(entry["calibration_date"]), "D")
        assessment_day = np.datetime64(str(entry["assessment_date"]), "D")
        if not fit_hi < calibration_day < assessment_day:
            raise ContractStop(f"origin {origin} violates the frozen date ordering")
        if str(entry["assessment_date"]) != origin:
            raise ContractStop(f"origin {origin} assessment date must equal the origin")

        fit_index = np.flatnonzero((dates >= fit_lo) & (dates <= fit_hi))
        calibration_index = np.flatnonzero(dates == calibration_day)
        assessment_index = np.flatnonzero(dates == assessment_day)
        if (
            np.intersect1d(fit_index, calibration_index).size
            or np.intersect1d(fit_index, assessment_index).size
            or np.intersect1d(calibration_index, assessment_index).size
        ):
            raise ContractStop(f"origin {origin} has overlapping row groups")

        for label, index, expected in (
            ("calibration", calibration_index, entry["calibration_expected"]),
            ("assessment", assessment_index, entry["assessment_expected"]),
        ):
            observed = {
                "rows": int(index.size),
                "users": int(np.unique(users[index]).size),
                "positives": int(labels[index].sum()),
            }
            for key, value in expected.items():
                if observed[key] != int(value):
                    raise ContractStop(
                        f"origin {origin} {label} {key} mismatch: "
                        f"expected {value}, observed {observed[key]}"
                    )

        calibration_labels = labels[calibration_index]
        if np.unique(calibration_labels).size != 2:
            raise ContractStop(f"origin {origin} calibration rows lack both classes")
        positives = int(calibration_labels.sum())
        negatives = int(calibration_labels.size - positives)
        if (
            calibration_index.size < minimums["minimum_rows"]
            or np.unique(users[calibration_index]).size < minimums["minimum_users"]
            or positives < minimums["minimum_positives"]
            or negatives < minimums["minimum_negatives"]
        ):
            raise ContractStop(f"origin {origin} fails the calibration minimum requirements")

        before_origin = np.flatnonzero(dates < assessment_day)
        splits.append(
            OriginSplit(
                origin=origin,
                fit_index=fit_index,
                calibration_index=calibration_index,
                assessment_index=assessment_index,
                calibration_date=str(entry["calibration_date"]),
                bl0_probability=float(labels[before_origin].mean(dtype=np.float64)),
                fit_prevalence=float(labels[fit_index].mean(dtype=np.float64)),
            )
        )
    return splits


@dataclass
class OriginMatrices:
    split: OriginSplit
    design: repair.GroupedDesign
    matrices: dict[str, dict[str, sparse.csr_matrix]]
    labels: dict[str, np.ndarray]
    users: dict[str, np.ndarray]
    scaling_audit: list[dict[str, Any]]
    categorical_audit: list[dict[str, Any]]


def _group_blocks(frame: Frame, index: np.ndarray, prevalence: float) -> dict[str, np.ndarray]:
    categorical = np.column_stack(
        [frame.columns[name][index] for name in repair.CATEGORICAL_FIELDS]
    ).astype(np.int64, copy=False)
    static_continuous = np.column_stack(
        [frame.columns[name][index] for name in repair.STATIC_CONTINUOUS_FIELDS]
    ).astype(np.float64, copy=False)
    static_binary = np.column_stack(
        [frame.columns[name][index] for name in repair.STATIC_BINARY_FIELDS]
    ).astype(np.float64, copy=False)
    raw = {name: frame.columns[name][index] for name in H2_RAW_COLUMNS}
    h2_continuous, h2_binary = repair.derive_h2_blocks(raw, prevalence)
    return {
        "categorical": categorical,
        "static_continuous": static_continuous,
        "static_binary": static_binary,
        "h2_continuous": h2_continuous,
        "h2_binary": h2_binary,
    }


def build_origin_matrices(frame: Frame, split: OriginSplit) -> OriginMatrices:
    """Fit the design on estimator-fit rows only; transform the other groups."""

    prevalence = split.fit_prevalence
    fit_blocks = _group_blocks(frame, split.fit_index, prevalence)
    design, fit_bl1, fit_bl2 = repair.fit_grouped_design(prevalence=prevalence, **fit_blocks)

    matrices = {"fit": {"BL1": fit_bl1, "BL2": fit_bl2}}
    blocks_by_group = {"fit": fit_blocks}
    for name, index in (
        ("calibration", split.calibration_index),
        ("assessment", split.assessment_index),
    ):
        blocks = _group_blocks(frame, index, prevalence)
        blocks_by_group[name] = blocks
        bl1, bl2 = repair.transform_grouped(design, **blocks)
        matrices[name] = {"BL1": bl1, "BL2": bl2}

    audit: list[dict[str, Any]] = []
    categorical_audit: list[dict[str, Any]] = []
    for group, bundle_matrices in matrices.items():
        repair.assert_column_prefix(bundle_matrices["BL1"], bundle_matrices["BL2"])
        blocks = blocks_by_group[group]
        for row in repair.numeric_field_audit(
            design,
            split=group,
            static_continuous=blocks["static_continuous"],
            h2_continuous=blocks["h2_continuous"],
        ):
            audit.append({"record_type": "field", "origin": split.origin, **row})
        categorical_audit.extend(
            {
                "origin": split.origin,
                **row,
            }
            for row in repair.categorical_frequency_audit(
                design, split=group, categorical=blocks["categorical"]
            )
        )
        for model_id, matrix in bundle_matrices.items():
            stats = repair.numeric_hard_checks(matrix, f"{split.origin}/{group}/{model_id}")
            audit.append(
                {
                    "record_type": "matrix",
                    "origin": split.origin,
                    "split": group,
                    "model_id": model_id,
                    **stats,
                }
            )

    labels = frame.labels()
    users = frame.users()
    return OriginMatrices(
        split=split,
        design=design,
        matrices=matrices,
        labels={
            "fit": labels[split.fit_index],
            "calibration": labels[split.calibration_index],
            "assessment": labels[split.assessment_index],
        },
        users={
            "fit": users[split.fit_index],
            "calibration": users[split.calibration_index],
            "assessment": users[split.assessment_index],
        },
        scaling_audit=audit,
        categorical_audit=categorical_audit,
    )


# ---------------------------------------------------------------------------
# budget ledger
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    contract: dict[str, Any]
    started: float = field(default_factory=time.perf_counter)
    ledger: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    peak_ram_gb: float = 0.0
    nonfitted_rule_evaluations: int = 0
    release_monitor: ResourceMonitor | None = field(default=None, repr=False)

    SECOND_CAP_KEYS = {
        "sgd": "maximum_SGD_seconds_per_fit",
        "calibrator": "maximum_calibrator_seconds_per_fit",
        "reference": "maximum_reference_solver_seconds_per_fit",
        "diagnostic": "maximum_diagnostic_seconds_per_fit",
    }
    STAGE_CAP_KEYS = {
        ("search", "sgd"): "maximum_search_SGD_fit_runs",
        ("search", "calibrator"): "maximum_search_primary_calibrator_fit_runs",
        ("search", "reference"): "maximum_search_reference_solver_fit_runs",
        ("frozen_daily", "sgd"): "maximum_daily_SGD_fit_runs",
        ("frozen_daily", "calibrator"): "maximum_daily_primary_calibrator_fit_runs",
        ("frozen_daily", "reference"): "maximum_daily_reference_solver_fit_runs",
        ("diagnostic", "diagnostic"): "maximum_diagnostic_calibration_regression_fits",
    }

    def start_release_monitor(self) -> None:
        if self.release_monitor is not None:
            raise ContractStop("whole-release resource monitor is already active")
        self.release_monitor = ResourceMonitor()
        self.release_monitor.__enter__()

    def stop_release_monitor(self) -> None:
        if self.release_monitor is None:
            return
        monitor = self.release_monitor
        monitor.__exit__(None, None, None)
        self.peak_ram_gb = max(self.peak_ram_gb, monitor.peak_ram_gb)
        self.release_monitor = None

    def check_elapsed(self, stage: str) -> None:
        observed_ram_gb = psutil.Process().memory_info().rss / float(1024**3)
        if self.release_monitor is not None:
            observed_ram_gb = max(observed_ram_gb, self.release_monitor.peak_ram_gb)
        self.peak_ram_gb = max(self.peak_ram_gb, observed_ram_gb)
        ram_cap = float(self.contract["operational_budget"]["maximum_peak_ram_gb"])
        if self.peak_ram_gb > ram_cap:
            raise ContractStop(
                f"RAM cap breached during {stage}: {self.peak_ram_gb:.3f}GB > {ram_cap}GB"
            )
        minutes = self.elapsed_seconds / 60.0
        cap = float(self.contract["operational_budget"]["maximum_elapsed_minutes"])
        if minutes > cap:
            raise ContractStop(f"elapsed minute cap breached during {stage}: {minutes:.3f} > {cap}")

    def check_storage(self, stage: str) -> float:
        storage_gb = managed_artifact_size_bytes() / float(1024**3)
        cap = float(self.contract["operational_budget"]["maximum_artifact_storage_gb"])
        if storage_gb > cap:
            raise ContractStop(
                f"artifact storage cap breached during {stage}: {storage_gb:.6f}GB > {cap}GB"
            )
        return storage_gb

    def _check_counts(self, kind: str, stage: str) -> None:
        budget = self.contract["operational_budget"]
        stage_key = f"{stage}:{kind}"
        cap_key = self.STAGE_CAP_KEYS.get((stage, kind))
        if cap_key is None:
            raise ContractStop(f"no declared fit-count cap for {stage_key}")
        if self.stage_counts[stage_key] > int(budget[cap_key]):
            raise ContractStop(
                f"{stage_key} fit cap breached at {self.stage_counts[stage_key]} > {budget[cap_key]}"
            )
        primary = self.counts.get("sgd", 0) + self.counts.get("calibrator", 0)
        if primary > int(budget["maximum_primary_component_fit_runs"]):
            raise ContractStop("primary component fit cap breached")
        if self.counts.get("reference", 0) > int(budget["maximum_reference_solver_fit_runs"]):
            raise ContractStop("reference solver fit cap breached")
        if self.counts.get("diagnostic", 0) > int(
            budget["maximum_diagnostic_calibration_regression_fits"]
        ):
            raise ContractStop("diagnostic calibration regression fit cap breached")
        total = sum(self.counts.values())
        if total > int(budget["maximum_total_fit_operations"]):
            raise ContractStop(
                f"total fit operation cap breached at {total} > "
                f"{budget['maximum_total_fit_operations']}; counts by type "
                f"{dict(sorted(self.counts.items()))}"
            )

    def _check_next_count(self, kind: str, stage: str) -> None:
        """Fail before starting a fit that would exceed any declared cap."""

        budget = self.contract["operational_budget"]
        cap_key = self.STAGE_CAP_KEYS.get((stage, kind))
        if cap_key is None:
            raise ContractStop(f"no declared fit-count cap for {stage}:{kind}")
        stage_key = f"{stage}:{kind}"
        if self.stage_counts.get(stage_key, 0) + 1 > int(budget[cap_key]):
            raise ContractStop(f"next {stage_key} fit would breach {cap_key}")
        primary = self.counts.get("sgd", 0) + self.counts.get("calibrator", 0)
        if kind in {"sgd", "calibrator"} and primary + 1 > int(
            budget["maximum_primary_component_fit_runs"]
        ):
            raise ContractStop("next primary-component fit would breach its cap")
        if kind == "reference" and self.counts.get("reference", 0) + 1 > int(
            budget["maximum_reference_solver_fit_runs"]
        ):
            raise ContractStop("next reference fit would breach its cap")
        if kind == "diagnostic" and self.counts.get("diagnostic", 0) + 1 > int(
            budget["maximum_diagnostic_calibration_regression_fits"]
        ):
            raise ContractStop("next diagnostic fit would breach its cap")
        if sum(self.counts.values()) + 1 > int(budget["maximum_total_fit_operations"]):
            raise ContractStop("next fit would breach the total fit-operation cap")

    def record_nonfitted_rule_evaluations(self, count: int, *, stage: str) -> None:
        if count < 0:
            raise ContractStop("nonfitted rule evaluation count cannot be negative")
        cap = int(self.contract["operational_budget"]["maximum_nonfitted_history_rule_evaluations"])
        if self.nonfitted_rule_evaluations + int(count) > cap:
            raise ContractStop(
                f"next {stage} fixed-rule evaluations would breach the cap: "
                f"{self.nonfitted_rule_evaluations + int(count)} > {cap}"
            )
        self.nonfitted_rule_evaluations += int(count)
        self.check_elapsed(f"after_{stage}_fixed_rule_evaluations")

    def run(self, kind: str, fit_run_id: str, thunk: Callable[[], Any], **meta: Any) -> Any:
        budget = self.contract["operational_budget"]
        stage = str(meta.get("stage", ""))
        if not stage:
            raise ContractStop(f"fit {fit_run_id} has no declared stage")
        self.check_elapsed(f"before_{fit_run_id}")
        self._check_next_count(kind, stage)
        second_cap = float(budget[self.SECOND_CAP_KEYS[kind]])
        started = time.perf_counter()
        status = "complete"
        result: Any = None
        error: Exception | None = None
        if kind == "sgd":
            thread_limit = int(budget["SGD_estimator_threads"])
        elif kind == "reference":
            thread_limit = int(budget["reference_solver_threads"])
        else:
            thread_limit = int(budget["maximum_cpu_threads"])
        observed_threadpool_max = 0
        observed_threadpools: list[dict[str, Any]] = []
        with ResourceMonitor() as monitor, threadpoolctl.threadpool_limits(
            limits=thread_limit
        ):
            try:
                observed_threadpools = [
                    {
                        "prefix": str(item.get("prefix", "")),
                        "internal_api": str(item.get("internal_api", "")),
                        "user_api": str(item.get("user_api", "")),
                        "num_threads": int(item.get("num_threads", 0)),
                    }
                    for item in threadpoolctl.threadpool_info()
                ]
                observed_threadpool_max = max(
                    (item["num_threads"] for item in observed_threadpools), default=0
                )
                if observed_threadpool_max > thread_limit:
                    raise ContractStop(
                        f"thread pool cap not enforced for {fit_run_id}: "
                        f"observed {observed_threadpool_max} > {thread_limit}"
                    )
                result = thunk()
            except Exception as caught:  # failed fits still consume budget
                status = f"failed:{type(caught).__name__}"
                error = caught
        elapsed = time.perf_counter() - started
        self.peak_ram_gb = max(self.peak_ram_gb, monitor.peak_ram_gb)
        self.counts[kind] = self.counts.get(kind, 0) + 1
        stage_key = f"{stage}:{kind}"
        self.stage_counts[stage_key] = self.stage_counts.get(stage_key, 0) + 1
        storage_gb = managed_artifact_size_bytes() / float(1024**3)
        self.ledger.append(
            {
                "fit_run_id": fit_run_id,
                "component_type": kind,
                "stage": stage,
                "elapsed_seconds": elapsed,
                "peak_RAM_GB": monitor.peak_ram_gb,
                "artifact_storage_GB": storage_gb,
                "thread_limit": thread_limit,
                "observed_threadpool_max": observed_threadpool_max,
                "observed_threadpools": json.dumps(
                    observed_threadpools, sort_keys=True, separators=(",", ":")
                ),
                "status": status,
                **meta,
            }
        )
        self._check_counts(kind, stage)
        breaches: list[str] = []
        if storage_gb > float(budget["maximum_artifact_storage_gb"]):
            breaches.append(
                f"storage={storage_gb:.6f}GB>{budget['maximum_artifact_storage_gb']}GB"
            )
        if elapsed > second_cap:
            breaches.append(f"elapsed={elapsed:.3f}s>{second_cap}s")
        if monitor.peak_ram_gb > float(budget["maximum_peak_ram_gb"]):
            breaches.append(
                f"fit_peak_RAM={monitor.peak_ram_gb:.3f}GB>{budget['maximum_peak_ram_gb']}GB"
            )
        current_ram_gb = psutil.Process().memory_info().rss / float(1024**3)
        if self.release_monitor is not None:
            current_ram_gb = max(current_ram_gb, self.release_monitor.peak_ram_gb)
        self.peak_ram_gb = max(self.peak_ram_gb, current_ram_gb)
        if self.peak_ram_gb > float(budget["maximum_peak_ram_gb"]):
            breaches.append(
                f"release_peak_RAM={self.peak_ram_gb:.3f}GB>{budget['maximum_peak_ram_gb']}GB"
            )
        elapsed_minutes = self.elapsed_seconds / 60.0
        if elapsed_minutes > float(budget["maximum_elapsed_minutes"]):
            breaches.append(
                f"release_elapsed={elapsed_minutes:.3f}min>{budget['maximum_elapsed_minutes']}min"
            )
        if breaches:
            prefix = ""
            if error is not None:
                prefix = f"fit raised {type(error).__name__}: {error}; "
            raise ContractStop(
                f"{prefix}resource cap breach after {fit_run_id}: {'|'.join(breaches)}"
            ) from error
        if error is not None:
            raise error
        return result

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started

    def last_seconds(self) -> float:
        return float(self.ledger[-1]["elapsed_seconds"])


@dataclass
class LedgerBook:
    convergence: list[dict[str, Any]] = field(default_factory=list)
    reference: list[dict[str, Any]] = field(default_factory=list)
    calibration: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# fitted components
# ---------------------------------------------------------------------------


@dataclass
class Component:
    origin: str
    model_id: str
    pair_id: str
    alpha: float
    eta0: float
    stage: str
    sgd_record: repair.FitRecord
    raw_calibration: np.ndarray
    raw_assessment: np.ndarray
    uncalibrated_assessment: np.ndarray
    feature_columns: int
    estimator_fit_rows: int
    failure_reason: str = ""


def raw_scores(model: Any, matrix: sparse.csr_matrix) -> np.ndarray:
    score = np.asarray(model.decision_function(matrix), dtype=np.float64).ravel()
    if not np.isfinite(score).all():
        raise ContractStop("non-finite raw decision score")
    return score


def _quantile_text(values: np.ndarray) -> str:
    quantiles = np.quantile(np.asarray(values, dtype=np.float64), RAW_SCORE_QUANTILES)
    return "|".join(f"{value:.6g}" for value in quantiles)


def get_params_snapshot(model: Any) -> str:
    """Stable JSON rendering of the exact sklearn estimator parameters in use."""

    return json.dumps(
        json_safe(model.get_params(deep=True)), sort_keys=True, allow_nan=False, default=str
    )


def fit_component(
    budget: Budget,
    book: LedgerBook,
    origin: OriginMatrices,
    *,
    model_id: str,
    alpha: float,
    eta0: float,
    stage: str,
) -> Component:
    pair = repair.pair_id(alpha, eta0)
    fit_x = origin.matrices["fit"][model_id]
    fit_y = origin.labels["fit"]
    run_id = f"{stage}_{origin.split.origin}_{model_id}_{pair}"
    model, record = budget.run(
        "sgd",
        run_id,
        lambda: repair.fit_sgd(fit_x, fit_y, alpha=alpha, eta0=eta0),
        stage=stage,
        origin=origin.split.origin,
        model_id=model_id,
        pair_id=pair,
        alpha=alpha,
        eta0=eta0,
        estimator_fit_rows=int(fit_x.shape[0]),
        feature_columns=int(fit_x.shape[1]),
    )
    raw_assessment = raw_scores(model, origin.matrices["assessment"][model_id])
    uncalibrated = np.asarray(
        model.predict_proba(origin.matrices["assessment"][model_id])[:, 1], dtype=np.float64
    )
    book.convergence.append(
        {
            "fit_run_id": run_id,
            "fit_type": "SGD",
            "stage": stage,
            "origin": origin.split.origin,
            "model_id": model_id,
            "bundle": BUNDLES[model_id],
            "pair_id": pair,
            "alpha": alpha,
            "eta0": eta0,
            "estimator_fit_rows": int(fit_x.shape[0]),
            "feature_columns": int(fit_x.shape[1]),
            "max_iter": record.max_iter,
            "n_iter": record.n_iter,
            "convergence_warning_count": record.convergence_warning_count,
            "coefficient_L1_norm": record.coefficient_l1_norm,
            "coefficient_L2_norm": record.coefficient_l2_norm,
            "coefficient_absolute_maximum": record.coefficient_absolute_maximum,
            "intercept": record.intercept,
            "raw_score_quantiles": _quantile_text(raw_assessment),
            "get_params_snapshot": get_params_snapshot(model),
            "elapsed_seconds": budget.last_seconds(),
            "status": "converged" if record.converged else "nonconverged",
        }
    )
    failure_reason = "" if record.converged else "SGD_nonconvergence"
    if not record.converged and stage == "frozen_daily":
        raise TerminalStop(
            "frozen_daily_component_convergence_failure",
            EXIT_FROZEN_DAILY_COMPONENT_CONVERGENCE_FAILURE,
            f"SGD fit {run_id} did not converge inside the frozen iteration cap",
            stage="frozen_daily",
        )
    return Component(
        origin=origin.split.origin,
        model_id=model_id,
        pair_id=pair,
        alpha=alpha,
        eta0=eta0,
        stage=stage,
        sgd_record=record,
        raw_calibration=raw_scores(model, origin.matrices["calibration"][model_id]),
        raw_assessment=raw_assessment,
        uncalibrated_assessment=uncalibrated,
        feature_columns=int(fit_x.shape[1]),
        estimator_fit_rows=int(fit_x.shape[0]),
        failure_reason=failure_reason,
    )


def fit_reference_component(
    budget: Budget,
    book: LedgerBook,
    origin: OriginMatrices,
    *,
    model_id: str,
    alpha: float,
    stage: str,
) -> tuple[repair.FitRecord, float]:
    fit_x = origin.matrices["fit"][model_id]
    fit_y = origin.labels["fit"]
    run_id = f"{stage}_reference_{origin.split.origin}_{model_id}_{alpha_slug(alpha)}"
    model, record, C = budget.run(
        "reference",
        run_id,
        lambda: repair.fit_reference(fit_x, fit_y, alpha=alpha),
        stage=stage,
        origin=origin.split.origin,
        model_id=model_id,
        alpha=alpha,
        estimator_fit_rows=int(fit_x.shape[0]),
    )
    # kept only for the descriptive assessment regret diagnostics; the reference
    # solver is never a prediction candidate and never enters selection
    record.assessment_raw_score = raw_scores(
        model, origin.matrices["assessment"][model_id].astype(np.float64)
    )
    book.reference.append(
        {
            "fit_run_id": run_id,
            "phase": stage,
            "origin": origin.split.origin,
            "model_id": model_id,
            "alpha": alpha,
            "estimator_fit_rows": int(fit_x.shape[0]),
            "C": C,
            "solver": "lbfgs",
            "tol": repair.REFERENCE_TOL,
            "max_iter": record.max_iter,
            "n_iter": record.n_iter,
            "convergence_warning_count": record.convergence_warning_count,
            "reference_regularized_training_objective": record.objective,
            "get_params_snapshot": get_params_snapshot(model),
            "elapsed_seconds": budget.last_seconds(),
            "peak_RAM_GB": float(budget.ledger[-1]["peak_RAM_GB"]),
            "status": "converged" if record.converged else "nonconverged",
        }
    )
    if not record.converged and stage == "frozen_daily":
        raise TerminalStop(
            "frozen_daily_component_convergence_failure",
            EXIT_FROZEN_DAILY_COMPONENT_CONVERGENCE_FAILURE,
            f"reference solver {run_id} did not converge inside the frozen cap",
            stage="frozen_daily",
        )
    return record, C


@dataclass
class CalibrationOutcome:
    calibrator: repair.Calibrator | None
    probability: np.ndarray | None
    eligible: bool
    failure_reason: str = ""


def calibrate_component(
    budget: Budget,
    book: LedgerBook,
    origin: OriginMatrices,
    component: Component,
    *,
    stage: str,
) -> CalibrationOutcome:
    run_id = f"{stage}_calibrator_{component.origin}_{component.model_id}_{component.pair_id}"
    base_row = {
        "fit_run_id": run_id,
        "stage": stage,
        "origin": component.origin,
        "model_id": component.model_id,
        "pair_id": component.pair_id,
        "calibration_date": origin.split.calibration_date,
        "raw_score_source": "base_estimator_raw_decision_score",
        "solver": "lbfgs",
        "C": repair.CALIBRATOR_C,
        "l1_ratio": 0.0,
        "tol": repair.CALIBRATOR_TOL,
        "max_iter": repair.CALIBRATOR_MAX_ITER,
    }
    try:
        calibrator = budget.run(
            "calibrator",
            run_id,
            lambda: repair.fit_previous_day_sigmoid(
                component.raw_calibration,
                origin.labels["calibration"],
                user_id=origin.users["calibration"],
            ),
            stage=stage,
            origin=component.origin,
            model_id=component.model_id,
            pair_id=component.pair_id,
        )
    except ContractStop:
        raise
    except Exception as error:
        is_nonconvergence = (
            isinstance(error, repair.ContractViolation)
            and str(error) == "calibration did not converge inside the frozen cap"
        )
        book.calibration.append(
            {
                **base_row,
                "calibration_rows": int(origin.labels["calibration"].size),
                "calibration_users": int(np.unique(origin.users["calibration"]).size),
                "calibration_positives": int(origin.labels["calibration"].sum()),
                "calibration_prevalence": float(origin.labels["calibration"].mean()),
                "n_iter": None,
                "convergence_warning_count": None,
                "future_or_deprecation_warning_count": None,
                "calibration_intercept": None,
                "calibration_slope": None,
                "get_params_snapshot": None,
                "elapsed_seconds": budget.last_seconds(),
                "status": "nonconverged" if is_nonconvergence else f"failed:{type(error).__name__}",
            }
        )
        if stage == "search" and is_nonconvergence:
            return CalibrationOutcome(None, None, False, "calibrator_nonconverged")
        if stage == "frozen_daily" and is_nonconvergence:
            raise TerminalStop(
                "frozen_daily_component_convergence_failure",
                EXIT_FROZEN_DAILY_COMPONENT_CONVERGENCE_FAILURE,
                f"frozen daily calibrator {run_id} did not converge inside the frozen cap",
                stage=stage,
            ) from error
        raise TerminalStop(
            "implementation_or_governance_failure",
            EXIT_IMPLEMENTATION_OR_GOVERNANCE_FAILURE,
            f"{stage} calibrator {run_id} failed a non-routine contract check: {error}",
            stage=stage,
        ) from error
    calibrated = calibrator.apply(component.raw_assessment)
    repair.assert_calibration_monotone(component.raw_assessment, calibrated)
    book.calibration.append(
        {
            **base_row,
            "calibration_rows": calibrator.fit_rows,
            "calibration_users": calibrator.fit_users,
            "calibration_positives": calibrator.fit_positives,
            "calibration_prevalence": calibrator.fit_prevalence,
            "n_iter": calibrator.n_iter,
            "convergence_warning_count": calibrator.convergence_warning_count,
            "future_or_deprecation_warning_count": 0,
            "calibration_intercept": calibrator.intercept,
            "calibration_slope": calibrator.slope,
            "get_params_snapshot": get_params_snapshot(calibrator.model),
            "elapsed_seconds": budget.last_seconds(),
            "status": "complete",
        }
    )
    return CalibrationOutcome(calibrator, calibrated, True)


def verify_raw_calibrated_ap_equivalence(
    book: LedgerBook,
    y: np.ndarray,
    raw_score: np.ndarray,
    calibrated: np.ndarray,
    *,
    stage: str,
    origin_name: str,
    model_id: str,
    pair_id: str,
) -> None:
    """Read assessment labels only after the stage's predictions are frozen."""

    raw_ap = float(average_precision_score(y, raw_score))
    calibrated_ap = float(average_precision_score(y, calibrated))
    if abs(raw_ap - calibrated_ap) <= repair.CALIBRATION_AP_TOLERANCE:
        return
    run_id = f"{stage}_calibrator_{origin_name}_{model_id}_{pair_id}"
    for row in reversed(book.calibration):
        if row["fit_run_id"] == run_id:
            row["status"] = "failed:raw_calibrated_AP_mismatch"
            break
    raise TerminalStop(
        "implementation_or_governance_failure",
        EXIT_IMPLEMENTATION_OR_GOVERNANCE_FAILURE,
        f"raw/calibrated AP mismatch for {run_id}: {raw_ap} vs {calibrated_ap} "
        f"exceeds {repair.CALIBRATION_AP_TOLERANCE}",
        stage=stage,
    )


# ---------------------------------------------------------------------------
# metrics helpers
# ---------------------------------------------------------------------------


def assessment_metrics(
    y: np.ndarray, probability: np.ndarray, users: np.ndarray
) -> dict[str, float | int]:
    return metrics.point_metrics(y, probability, users, epsilon=repair.METRIC_CLIP_LOW)


def contract_named_metrics(
    y: np.ndarray, probability: np.ndarray, users: np.ndarray
) -> dict[str, float | int]:
    """Emit the metric names the contract declares, not the internal ones.

    Contract: ``probability_diagnostics.required_per_model_origin_and_pooled``.
    """

    point = assessment_metrics(y, probability, users)
    return {
        **point,
        "observed_prevalence": point["prevalence"],
        "ECE20_equal_width": point["ece20_equal_width"],
        "calibrated_average_precision": point["average_precision"],
        "calibrated_user_gauc_event_weighted": point["user_gauc_event_weighted"],
        "calibrated_user_gauc_user_equal": point["user_gauc_user_equal"],
    }


def uncalibrated_named_metrics(
    y: np.ndarray, probability: np.ndarray, users: np.ndarray
) -> dict[str, float | int]:
    point = assessment_metrics(y, probability, users)
    return {
        "uncalibrated_average_precision": point["average_precision"],
        "uncalibrated_log_loss": point["log_loss"],
        "uncalibrated_brier": point["brier"],
        "uncalibrated_ECE20_equal_width": point["ece20_equal_width"],
    }


def raw_score_within_origin_metrics(
    y: np.ndarray, raw_score: np.ndarray, users: np.ndarray
) -> dict[str, float]:
    """Contract: the three ``raw_score_*_within_origin`` diagnostics.

    Calibration is monotone, so these must agree with the calibrated ranking
    inside an origin; they are kept as an independent check of that invariant and
    are never comparable across origins.
    """

    gauc = metrics.user_gauc_components(
        y, _rank_positions(raw_score), users, epsilon=repair.METRIC_CLIP_LOW
    )
    return {
        "raw_score_average_precision_within_origin": float(
            metrics.point_metrics(
                y, _rank_positions(raw_score), users, epsilon=repair.METRIC_CLIP_LOW
            )["average_precision"]
        ),
        "raw_score_user_gauc_event_weighted_within_origin": gauc.event_weighted,
        "raw_score_user_gauc_user_equal_within_origin": gauc.user_equal,
    }


def _rank_positions(score: np.ndarray) -> np.ndarray:
    """Map an unbounded score onto (0, 1) order-preservingly for metric reuse."""

    values = np.asarray(score, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # average ranks for ties so the mapping stays a strict monotone transform
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    return ranks / (values.size + 1.0)


def assessment_calibration_regression(
    budget: Budget,
    y: np.ndarray,
    probability: np.ndarray,
    *,
    scope: str,
    model_id: str,
) -> dict[str, Any]:
    """Contract: ``probability_diagnostics.assessment_calibration_regression``.

    A linear-in-the-logit recalibration fit.  This is a descriptive diagnostic,
    not a claim that miscalibration is linear; ``calibration_bins.csv`` carries
    the twenty-bin nonparametric companion for any shape.
    """

    p = metrics.clipped(probability, repair.METRIC_CLIP_LOW)
    logit = np.log(p / (1.0 - p))
    diagnostic_get_params = get_params_snapshot(_unconstrained_logit_model())
    if ACTIVE_RUN_STARTED:
        ACTIVE_ASSESSMENT_GET_PARAMS.add(diagnostic_get_params)
    row: dict[str, Any] = {
        "scope": scope,
        "model_id": model_id,
        "rows": int(y.size),
        "input": "metric_clipped_logit_of_calibrated_probability",
        "role": "descriptive_only",
        "get_params_snapshot": diagnostic_get_params,
    }
    if model_id == "BL0":
        # BL0 emits one constant probability, so no function of any form is
        # attached to a base-estimator score. Across pooled origins its fitted
        # prevalence may vary, but the registered diagnostic remains N/A because
        # BL0 is the probability reference rather than a recalibratable model.
        row.update(
            {
                "status": "not_applicable",
                "reason": "constant_predictor_zero_variance_logit",
                "assessment_calibration_intercept": None,
                "assessment_calibration_slope": None,
            }
        )
        return row
    if float(np.var(logit)) <= 0.0:
        raise ContractStop(
            f"assessment calibration regression has zero-variance predictor for {scope}/{model_id}"
        )
    run_id = f"diagnostic_calibration_regression_{scope}_{model_id}"
    intercept, slope = budget.run(
        "diagnostic",
        run_id,
        lambda: _unconstrained_logit_regression(logit, y),
        stage="diagnostic",
        scope=scope,
        model_id=model_id,
    )
    row.update(
        {
            "status": "complete",
            "reason": "",
            "assessment_calibration_intercept": intercept,
            "assessment_calibration_slope": slope,
        }
    )
    return row


def _unconstrained_logit_regression(logit: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Descriptive recalibration fit with no sign constraint on the slope.

    The primary Platt calibrator requires a strictly positive slope because a
    negative one would invert the model's own ranking.  This diagnostic must not
    impose that: a slope below zero is a legitimate observation about the frozen
    model and has to be reportable rather than fatal.
    """

    model = _unconstrained_logit_model()
    import warnings as _warnings
    from sklearn.exceptions import ConvergenceWarning

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        model.fit(
            np.asarray(logit, dtype=np.float64).reshape(-1, 1),
            np.asarray(y, dtype=np.int8),
        )
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise ContractStop("assessment calibration regression did not converge")
    if any(
        issubclass(item.category, (FutureWarning, DeprecationWarning)) for item in caught
    ):
        raise ContractStop("assessment calibration regression emitted a compatibility warning")
    n_iter = int(np.asarray(model.n_iter_).ravel()[0])
    if n_iter >= repair.CALIBRATOR_MAX_ITER:
        raise ContractStop("assessment calibration regression reached its iteration cap")
    intercept = float(np.asarray(model.intercept_, dtype=np.float64).ravel()[0])
    slope = float(np.asarray(model.coef_, dtype=np.float64).ravel()[0])
    if not np.isfinite(intercept) or not np.isfinite(slope):
        raise ContractStop("assessment calibration regression produced non-finite parameters")
    return intercept, slope


def _unconstrained_logit_model() -> Any:
    """Construct the sklearn 1.9 unpenalized diagnostic estimator."""

    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        solver="lbfgs",
        l1_ratio=0.0,
        C=np.inf,
        dual=False,
        tol=repair.CALIBRATOR_TOL,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        max_iter=repair.CALIBRATOR_MAX_ITER,
        verbose=0,
        warm_start=False,
        n_jobs=None,
    )


def probability_distribution(probability: np.ndarray) -> dict[str, float]:
    p = np.asarray(probability, dtype=np.float64)
    quantiles = np.quantile(p, [0.001, 0.01, 0.10, 0.50, 0.90, 0.99, 0.999])
    return {
        "mean_prediction": float(p.mean()),
        "minimum": float(p.min()),
        "p001": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p10": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "p999": float(quantiles[6]),
        "maximum": float(p.max()),
        "share_probability_le_1e_6": float(np.mean(p <= 1e-6)),
        "share_probability_ge_1_minus_1e_6": float(np.mean(p >= 1.0 - 1e-6)),
        "share_probability_le_0_01": float(np.mean(p <= 0.01)),
        "share_probability_ge_0_99": float(np.mean(p >= 0.99)),
    }


def fixed_rule_scores(frame: Frame, index: np.ndarray, prevalence: float) -> dict[str, np.ndarray]:
    """Contract: ``models.fixed_nonfitted_history_diagnostics`` (no fitted parameters)."""

    prior_event = frame.columns["prior_event_n"][index].astype(np.float64)
    prior_positive = frame.columns["prior_positive_n"][index].astype(np.float64)
    w10_event = frame.columns["w10_event_n"][index].astype(np.float64)
    w10_positive = frame.columns["w10_positive_n"][index].astype(np.float64)
    strength = repair.H2_SMOOTHING_PRIOR_STRENGTH
    return {
        "LIFETIME_SMOOTHED_RATE": (prior_positive + strength * prevalence)
        / (prior_event + strength),
        "W10_SMOOTHED_RATE": (w10_positive + strength * prevalence) / (w10_event + strength),
    }


def fixed_rule_metric_rows(
    frame: Frame,
    origins: dict[str, OriginMatrices],
    origin_names: Iterable[str],
    *,
    stage: str,
    frozen_scores: dict[tuple[str, str], np.ndarray] | None = None,
    include_pooled: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pooled_scores: dict[str, list[np.ndarray]] = {
        "LIFETIME_SMOOTHED_RATE": [],
        "W10_SMOOTHED_RATE": [],
    }
    pooled_labels: list[np.ndarray] = []
    pooled_users: list[np.ndarray] = []
    for origin_name in origin_names:
        origin = origins[str(origin_name)]
        index = origin.split.assessment_index
        y = origin.labels["assessment"]
        users = origin.users["assessment"]
        scores = (
            fixed_rule_scores(frame, index, origin.split.fit_prevalence)
            if frozen_scores is None
            else {
                name: frozen_scores[(str(origin_name), name)]
                for name in ("LIFETIME_SMOOTHED_RATE", "W10_SMOOTHED_RATE")
            }
        )
        pooled_labels.append(y)
        pooled_users.append(users)
        for name, score in scores.items():
            pooled_scores[name].append(score)
            point = assessment_metrics(y, score, users)
            rows.append(
                {
                    "stage": stage,
                    "origin_or_pooled": str(origin_name),
                    "diagnostic_id": name,
                    "rows": point["rows"],
                    "users": point["users"],
                    "prevalence": point["prevalence"],
                    "average_precision": point["average_precision"],
                    "roc_auc": point["roc_auc"],
                    "user_gauc_event_weighted": point["user_gauc_event_weighted"],
                    "user_gauc_user_equal": point["user_gauc_user_equal"],
                }
            )
    if include_pooled:
        labels = np.concatenate(pooled_labels)
        users = np.concatenate(pooled_users)
        for name, parts in pooled_scores.items():
            point = assessment_metrics(labels, np.concatenate(parts), users)
            rows.append(
                {
                    "stage": stage,
                    "origin_or_pooled": "pooled_3_origins",
                    "diagnostic_id": name,
                    "rows": point["rows"],
                    "users": point["users"],
                    "prevalence": point["prevalence"],
                    "average_precision": point["average_precision"],
                    "roc_auc": point["roc_auc"],
                    "user_gauc_event_weighted": point["user_gauc_event_weighted"],
                    "user_gauc_user_equal": point["user_gauc_user_equal"],
                }
            )
    return rows


def enrich_fixed_rule_rows_with_primary_bl2(
    rows: list[dict[str, Any]],
    *,
    primary_pair_id: str,
    metric_lookup: Callable[[str, str], float],
) -> list[dict[str, Any]]:
    """Join the frozen rule diagnostics to the primary BL2 on identical scopes."""

    enriched: list[dict[str, Any]] = []
    field_map = {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "user_gauc_event_weighted": "user_gauc_event_weighted",
        "user_gauc_user_equal": "user_gauc_user_equal",
    }
    for row in rows:
        item = dict(row)
        item["primary_BL2_pair_id"] = primary_pair_id
        scope = str(item["origin_or_pooled"])
        for source_field, metric_name in field_map.items():
            primary_value = float(metric_lookup(scope, metric_name))
            item[f"primary_BL2_{source_field}"] = primary_value
            item[f"diagnostic_minus_primary_BL2_{source_field}"] = (
                float(item[source_field]) - primary_value
            )
        enriched.append(item)
    return enriched


def slice_masks(prior_batch: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Contract: ``evaluation_slices.slices`` on a fixed target-row set."""

    values = np.asarray(prior_batch, dtype=np.int64)
    masks: list[tuple[str, np.ndarray]] = []
    for name, low, high in SLICE_BOUNDS:
        if low is None:
            mask = np.ones(values.shape, dtype=bool)
        elif high is None:
            mask = values >= low
        else:
            mask = (values >= low) & (values <= high)
        masks.append((name, mask))
    return masks


def _origin_bl0_metric(origin: OriginMatrices, metric: str) -> float:
    y = origin.labels["assessment"]
    p = metrics.clipped(
        np.full(y.shape, origin.split.bl0_probability, dtype=np.float64), repair.METRIC_CLIP_LOW
    )
    if metric == "log_loss":
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))
    return float(np.mean(np.square(p - y)))


def _bl0_pooled_metrics(
    origins: dict[str, OriginMatrices], origin_names: Iterable[str]
) -> dict[str, float]:
    labels: list[np.ndarray] = []
    probability: list[np.ndarray] = []
    for name in origin_names:
        origin = origins[name]
        labels.append(origin.labels["assessment"])
        probability.append(
            np.full(origin.labels["assessment"].shape, origin.split.bl0_probability)
        )
    y = np.concatenate(labels)
    p = metrics.clipped(np.concatenate(probability), repair.METRIC_CLIP_LOW)
    return {
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
        "brier": float(np.mean(np.square(p - y))),
        "average_precision": float(y.mean()),
    }


def _origin_metric(
    rows: list[dict[str, Any]], origin: str, model_id: str, pair: str, metric: str
) -> float:
    for row in rows:
        if (
            row["stage"] == "search_origin"
            and row["origin"] == origin
            and row["model_id"] == model_id
            and row["pair_id"] == pair
        ):
            return float(row[metric])
    raise ContractStop(f"missing search metric row for {origin}/{model_id}/{pair}")


def _select_shared_configuration(
    eligible: list[dict[str, Any]], pooled: dict[tuple[str, str], dict[str, float | int]]
) -> dict[str, Any]:
    """Contract: ``primary_shared_configuration_selection``.

    Uses BL1 metrics and parameters only; BL2 deltas are never consulted.  At
    each key every configuration inside the floating tolerance is retained, then
    the next key is applied; ascending ``pair_id`` is the final fallback.
    """

    tolerance = 1e-10
    remaining = list(eligible)
    for metric, maximise in (
        ("log_loss", False),
        ("brier", False),
        ("average_precision", True),
        ("user_gauc_event_weighted", True),
    ):
        values = {
            item["pair_id"]: float(pooled[("BL1", item["pair_id"])][metric]) for item in remaining
        }
        best = max(values.values()) if maximise else min(values.values())
        remaining = [
            item for item in remaining if abs(values[item["pair_id"]] - best) <= tolerance
        ]
    if len(remaining) > 1:
        best_alpha = max(item["alpha"] for item in remaining)
        remaining = [item for item in remaining if item["alpha"] == best_alpha]
    if len(remaining) > 1:
        best_eta0 = min(item["eta0"] for item in remaining)
        remaining = [item for item in remaining if item["eta0"] == best_eta0]
    remaining.sort(key=lambda item: item["pair_id"])
    return remaining[0]


def _select_secondary_bl2(
    eligible: list[dict[str, Any]], pooled: dict[tuple[str, str], dict[str, float | int]]
) -> dict[str, Any] | None:
    if not eligible:
        return None
    remaining = list(eligible)
    tolerance = 1e-10
    for metric, maximise in (
        ("average_precision", True),
        ("user_gauc_event_weighted", True),
        ("log_loss", False),
        ("brier", False),
    ):
        values = {
            item["pair_id"]: float(pooled[("BL2", item["pair_id"])][metric])
            for item in remaining
        }
        best = max(values.values()) if maximise else min(values.values())
        remaining = [
            item for item in remaining if abs(values[item["pair_id"]] - best) <= tolerance
        ]
    best_alpha = max(item["alpha"] for item in remaining)
    remaining = [item for item in remaining if item["alpha"] == best_alpha]
    best_eta0 = min(item["eta0"] for item in remaining)
    remaining = [item for item in remaining if item["eta0"] == best_eta0]
    return sorted(remaining, key=lambda item: item["pair_id"])[0]


def paired_configuration_rows(
    registry: list[dict[str, Any]],
    search_origins: list[str],
    configuration_reasons: dict[tuple[str, str], list[str]],
    metric_rows: list[dict[str, Any]],
    pooled: dict[tuple[str, str], dict[str, float | int]],
    *,
    primary_pair_id: str | None = None,
) -> list[dict[str, Any]]:
    """Materialize all four same-configuration contrasts, including failures."""

    rows: list[dict[str, Any]] = []
    for pair in registry:
        pair_name = pair["pair_id"]
        pair_reasons = sorted(
            set(
                configuration_reasons[("BL1", pair_name)]
                + configuration_reasons[("BL2", pair_name)]
            )
        )
        for scope in [*search_origins, "pooled_3_origins"]:
            for metric_name in (
                "average_precision",
                "user_gauc_event_weighted",
                "log_loss",
                "brier",
            ):
                if pair_reasons:
                    bl1_value = bl2_value = delta = None
                elif scope == "pooled_3_origins":
                    bl1_value = pooled[("BL1", pair_name)][metric_name]
                    bl2_value = pooled[("BL2", pair_name)][metric_name]
                    delta = bl2_value - bl1_value
                else:
                    bl1_value = _origin_metric(metric_rows, scope, "BL1", pair_name, metric_name)
                    bl2_value = _origin_metric(metric_rows, scope, "BL2", pair_name, metric_name)
                    delta = bl2_value - bl1_value
                rows.append(
                    {
                        "comparison_id": "same_configuration_BL2_minus_BL1",
                        "origin_or_pooled": scope,
                        "pair_id": pair_name,
                        "alpha": pair["alpha"],
                        "eta0": pair["eta0"],
                        "metric": metric_name,
                        "BL1_metric": bl1_value,
                        "BL2_metric": bl2_value,
                        "BL2_minus_BL1_delta": delta,
                        "primary_shared_pair": pair_name == primary_pair_id,
                        "configuration_eligible": not pair_reasons,
                        "ineligible_reason": "|".join(pair_reasons),
                    }
                )
    return rows


def primary_bl2_search_gate_passed(gate: dict[str, Any], criteria: dict[str, Any]) -> bool:
    """Evaluate every registered same-configuration BL2 search requirement."""

    return bool(
        gate["optimization_adequate"]
        and gate["pooled_log_loss_minus_BL0"] <= 0
        and gate["pooled_brier_minus_BL0"] <= 0
        and gate["nonworse_log_loss_origins_vs_BL0"]
        >= criteria["minimum_nonworse_log_loss_origins_vs_BL0"]
        and gate["nonworse_brier_origins_vs_BL0"]
        >= criteria["minimum_nonworse_brier_origins_vs_BL0"]
        and gate["delta_average_precision"] > 0
        and gate["delta_user_gauc_event_weighted"] >= 0
        and gate["delta_log_loss"] <= 0
        and gate["delta_brier"] <= 0
        and gate["positive_average_precision_origins"]
        >= criteria["minimum_positive_average_precision_origins_vs_paired_BL1"]
    )


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------


def runtime_preflight(
    contract: dict[str, Any],
    budget: Budget,
    book: LedgerBook,
    origins: dict[str, OriginMatrices],
) -> tuple[dict[str, Any], dict[tuple[str, str], Component]]:
    """Contract: ``runtime_preflight``.  No scientific metric is read here."""

    spec = contract["runtime_preflight"]
    designated = spec["designated_existing_search_components"]
    origin = origins[str(designated["origin"])]
    registry = {item["pair_id"]: item for item in repair.paired_configurations()}
    pair = registry[str(designated["pair_id"])]

    # Every preflight component is a real search component and is reused, so the
    # preflight adds zero extra fit operations as the contract declares.
    cached: dict[str, Any] = {
        "components": {},
        "calibrators": {},
        "calibrated": {},
        "calibration_failures": {},
        "reference_records": {},
        "reference_C": {},
    }
    seconds: dict[str, list[float]] = {"sgd": [], "calibrator": [], "reference": []}
    component_ids: list[str] = []
    name = origin.split.origin

    for model_id in designated["SGD_models"]:
        component = fit_component(
            budget, book, origin, model_id=model_id, alpha=pair["alpha"],
            eta0=pair["eta0"], stage="search",
        )
        cached["components"][(name, model_id, pair["pair_id"])] = component
        seconds["sgd"].append(budget.last_seconds())
        component_ids.append(f"SGD_{model_id}")

    for model_id in designated["primary_calibrators"]:
        key = (name, model_id, pair["pair_id"])
        outcome = calibrate_component(
            budget, book, origin, cached["components"][key], stage="search"
        )
        if outcome.eligible:
            cached["calibrators"][key] = outcome.calibrator
            cached["calibrated"][key] = outcome.probability
        else:
            cached["calibration_failures"][key] = outcome.failure_reason
        seconds["calibrator"].append(float(book.calibration[-1]["elapsed_seconds"]))
        component_ids.append(f"calibrator_{model_id}")

    reference_alpha = float(designated["reference_alpha"])
    for model_id in designated["reference_solver_models"]:
        record, C = fit_reference_component(
            budget, book, origin, model_id=model_id, alpha=reference_alpha, stage="search"
        )
        cached["reference_records"][(name, model_id, reference_alpha)] = record
        cached["reference_C"][(name, model_id, reference_alpha)] = C
        seconds["reference"].append(budget.last_seconds())
        component_ids.append(f"reference_{model_id}")

    projection = spec["projection"]
    multiplier = float(projection["safety_multiplier"])
    calibrator_like = (
        projection["remaining_primary_calibrator_fits"]
        + projection["diagnostic_calibration_regression_fits_using_primary_calibrator_time_proxy"]
    )
    projected_seconds = (
        budget.elapsed_seconds
        + multiplier
        * (
            max(seconds["sgd"]) * projection["remaining_SGD_fits"]
            + max(seconds["calibrator"]) * calibrator_like
            + max(seconds["reference"]) * projection["remaining_reference_fits"]
        )
        + projection["fixed_nonfit_reserve_seconds"]
    )
    projected_minutes = projected_seconds / 60.0
    elapsed_cap = float(projection["projected_total_elapsed_must_not_exceed_minutes"])
    observed_payload_bytes = 0
    for component in cached["components"].values():
        observed_payload_bytes += (
            component.raw_calibration.nbytes
            + component.raw_assessment.nbytes
            + component.uncalibrated_assessment.nbytes
        )
    for probability in cached["calibrated"].values():
        observed_payload_bytes += probability.nbytes
    for record in cached["reference_records"].values():
        if record.assessment_raw_score is not None:
            observed_payload_bytes += record.assessment_raw_score.nbytes
    observed_artifact_gb = observed_payload_bytes / float(1024**3)
    artifact_scale = float(
        contract["operational_budget"]["maximum_total_fit_operations"]
        / designated["component_fit_operations"]
    )
    projected_artifact_gb = observed_artifact_gb * artifact_scale * multiplier
    observed_ram_gb = budget.peak_ram_gb
    projected_ram_gb = observed_ram_gb * multiplier
    ram_cap = float(projection["projected_peak_RAM_must_not_exceed_GB"])
    artifact_cap = float(projection["projected_artifact_storage_must_not_exceed_GB"])
    failures: list[str] = []
    if projected_minutes > elapsed_cap:
        failures.append("projected_elapsed_exceeds_cap")
    if projected_ram_gb > ram_cap:
        failures.append("projected_RAM_exceeds_cap")
    if projected_artifact_gb > artifact_cap:
        failures.append("projected_artifact_storage_exceeds_cap")
    passed = not failures
    payload = {
        "designated_component_ids": "|".join(component_ids),
        "observed_seconds_by_component_type": json.dumps(
            {kind: max(values) for kind, values in seconds.items()}
        ),
        "observed_peak_RAM_GB": observed_ram_gb,
        "observed_artifact_GB": observed_artifact_gb,
        "safety_multiplier": multiplier,
        "projected_total_elapsed_minutes": projected_minutes,
        "projected_peak_RAM_GB": projected_ram_gb,
        "projected_artifact_GB": projected_artifact_gb,
        "passed": passed,
        "stop_reason": "|".join(failures),
        "origin": origin.split.origin,
        "pair_id": pair["pair_id"],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "runtime_preflight.json", payload)
    if not passed:
        write_csv(OUTPUT_DIR / "usage_ledger.csv", budget.ledger)
        raise ContractStop("runtime preflight failed: " + "|".join(failures))
    return payload, cached


def search_phase(
    contract: dict[str, Any],
    budget: Budget,
    book: LedgerBook,
    frame: Frame,
    origins: dict[str, OriginMatrices],
    cached: dict[tuple[str, str], Component],
) -> dict[str, Any]:
    """Contract: ``search_and_selection.search_phase_order`` steps 2-8."""

    selection = contract["search_and_selection"]
    search_origins = [str(value) for value in selection["search_origins"]]
    registry = repair.paired_configurations()

    # seeded with the reused preflight outputs
    components: dict[tuple[str, str, str], Component] = dict(cached["components"])
    calibrated: dict[tuple[str, str, str], np.ndarray] = dict(cached["calibrated"])
    calibrators: dict[tuple[str, str, str], repair.Calibrator] = dict(cached["calibrators"])
    calibration_failures: dict[tuple[str, str, str], str] = dict(
        cached["calibration_failures"]
    )
    trial_rows: list[dict[str, Any]] = []

    for origin_name in search_origins:
        origin = origins[origin_name]
        for pair in registry:
            for model_id in MODEL_IDS:
                key = (origin_name, model_id, pair["pair_id"])
                if key not in components:
                    components[key] = fit_component(
                        budget,
                        book,
                        origin,
                        model_id=model_id,
                        alpha=pair["alpha"],
                        eta0=pair["eta0"],
                        stage="search",
                    )
                if key not in calibrated and key not in calibration_failures:
                    outcome = calibrate_component(
                        budget, book, origin, components[key], stage="search"
                    )
                    if outcome.eligible:
                        calibrators[key] = outcome.calibrator
                        calibrated[key] = outcome.probability
                    else:
                        calibration_failures[key] = outcome.failure_reason
                trial_rows.append(
                    {
                        "stage": "search",
                        "origin": origin_name,
                        "model_id": model_id,
                        "bundle": BUNDLES[model_id],
                        "pair_id": pair["pair_id"],
                        "alpha": pair["alpha"],
                        "eta0": pair["eta0"],
                        "estimator_fit_rows": components[key].estimator_fit_rows,
                        "feature_columns": components[key].feature_columns,
                        "reused_from_runtime_preflight": key in cached["components"],
                    }
                )

    reference_records: dict[tuple[str, str, float], repair.FitRecord] = dict(
        cached["reference_records"]
    )
    reference_C: dict[tuple[str, str, float], float] = dict(cached["reference_C"])
    for origin_name in search_origins:
        origin = origins[origin_name]
        for alpha in repair.SGD_ALPHA_VALUES:
            for model_id in MODEL_IDS:
                key = (origin_name, model_id, float(alpha))
                if key in reference_records:
                    continue
                record, C = fit_reference_component(
                    budget, book, origin, model_id=model_id, alpha=float(alpha), stage="search"
                )
                reference_records[key] = record
                reference_C[key] = C

    budget.record_nonfitted_rule_evaluations(2 * len(search_origins), stage="search")
    prediction_paths, frozen_rule_scores = _write_search_predictions(
        frame,
        origins,
        search_origins,
        components,
        calibrated,
        calibrators,
        reference_records,
        reference_C,
    )
    search_freeze_manifest = freeze_search_artifacts(prediction_paths, frozen_rule_scores)
    freeze_schema = contract["artifact_schemas"]["search_artifact_freeze_manifest"]
    require_fields(
        [search_freeze_manifest], freeze_schema["required_fields"],
        "search_artifact_freeze_manifest",
    )
    require_fields(
        list(search_freeze_manifest["artifacts"]),
        freeze_schema["artifact_entry_required_fields"],
        "search_artifact_freeze_manifest.artifacts",
    )
    if int(search_freeze_manifest["artifact_count"]) != int(
        freeze_schema["required_artifact_count"]
    ):
        raise ContractStop("search freeze manifest does not contain exactly three artifacts")
    if sorted(str(item["path"]) for item in search_freeze_manifest["artifacts"]) != sorted(
        str(item) for item in freeze_schema["exact_paths"]
    ):
        raise ContractStop("search freeze manifest paths do not match the frozen contract")
    frozen_search_tables = read_frozen_search_tables(search_freeze_manifest)
    for origin_name, table in frozen_search_tables.items():
        required_prediction_columns: set[str] = set()
        for pair in registry:
            for model_id in MODEL_IDS:
                prefix = f"{model_id}_{pair['pair_id']}"
                required_prediction_columns.update(
                    {
                        f"raw_decision_score_{prefix}",
                        f"uncalibrated_probability_{prefix}",
                        f"calibrated_probability_{prefix}",
                    }
                )
        for alpha in repair.SGD_ALPHA_VALUES:
            for model_id in MODEL_IDS:
                slug = alpha_slug(float(alpha))
                required_prediction_columns.update(
                    {
                        f"reference_raw_decision_score_{model_id}_{slug}",
                        f"reference_model_id_{model_id}_{slug}",
                        f"reference_alpha_{model_id}_{slug}",
                        f"reference_C_{model_id}_{slug}",
                    }
                )
        missing_prediction_columns = sorted(
            required_prediction_columns - set(table.column_names)
        )
        if missing_prediction_columns:
            raise ContractStop(
                f"frozen search prediction columns incomplete for {origin_name}: "
                f"{missing_prediction_columns}"
            )
    frozen_rule_scores = {
        (origin, diagnostic_id): np.asarray(
            arrow_numpy(table, f"fixed_rule_score_{diagnostic_id}"), dtype=np.float64
        )
        for origin, table in frozen_search_tables.items()
        for diagnostic_id in ("LIFETIME_SMOOTHED_RATE", "W10_SMOOTHED_RATE")
    }
    search_rule_rows = frozen_search_fixed_rule_metric_rows(
        search_freeze_manifest, frozen_search_tables
    )
    require_fields(
        search_rule_rows,
        contract["artifact_schemas"][
            "search_fixed_history_rule_diagnostics_required_fields"
        ],
        "search_fixed_history_rule_diagnostics",
    )
    write_csv(OUTPUT_DIR / "search_fixed_history_rule_diagnostics.csv", search_rule_rows)

    adequacy_rows: list[dict[str, Any]] = []
    adequate: dict[tuple[str, str], bool] = {}
    origin_adequate: dict[tuple[str, str, str], bool] = {}
    for (origin_name, model_id, pair_name), component in sorted(components.items()):
        reference = reference_records[(origin_name, model_id, float(component.alpha))]
        decision = repair.adequacy_decision(
            component.sgd_record.objective,
            reference.objective,
            reference_converged=reference.converged,
        )
        decision["SGD_converged"] = component.sgd_record.converged
        decision["adequacy_passed"] = bool(
            component.sgd_record.converged and decision["adequacy_passed"]
        )
        adequacy_rows.append(
            {
                "phase": "search",
                "origin": origin_name,
                "model_id": model_id,
                "pair_id": pair_name,
                "alpha": component.alpha,
                "eta0": component.eta0,
                "reference_fit_source": "search_reference_fit",
                **decision,
            }
        )
        key = (model_id, pair_name)
        origin_key = (origin_name, model_id, pair_name)
        origin_adequate[origin_key] = bool(decision["adequacy_passed"])
        adequate[key] = adequate.get(key, True) and origin_adequate[origin_key]

    # Training-objective decisions above are complete before any candidate
    # assessment diagnostic is materialized.  These diagnostics are appended in
    # one batch and are never consulted by the adequacy decision.
    for row in adequacy_rows:
        component = components[(row["origin"], row["model_id"], row["pair_id"])]
        reference = reference_records[(row["origin"], row["model_id"], float(row["alpha"]))]
        row.update(regret_diagnostics(component, reference, origins[row["origin"]]))

    for component_key in sorted(calibrated):
        origin_name, model_id, pair_id = component_key
        frozen_table = frozen_search_tables[origin_name]
        prefix = f"{model_id}_{pair_id}"
        verify_raw_calibrated_ap_equivalence(
            book,
            np.asarray(arrow_numpy(frozen_table, "long_view"), dtype=np.int8),
            np.asarray(
                arrow_numpy(frozen_table, f"raw_decision_score_{prefix}"),
                dtype=np.float64,
            ),
            np.asarray(
                arrow_numpy(frozen_table, f"calibrated_probability_{prefix}"),
                dtype=np.float64,
            ),
            stage="search",
            origin_name=origin_name,
            model_id=model_id,
            pair_id=pair_id,
        )

    configuration_reasons: dict[tuple[str, str], list[str]] = {}
    for model_id in MODEL_IDS:
        for pair in registry:
            key = (model_id, pair["pair_id"])
            reasons: list[str] = []
            for origin_name in search_origins:
                component_key = (origin_name, model_id, pair["pair_id"])
                if not components[component_key].sgd_record.converged:
                    reasons.append(f"{origin_name}:SGD_nonconverged")
                reference = reference_records[
                    (origin_name, model_id, float(pair["alpha"]))
                ]
                if not reference.converged:
                    reasons.append(f"{origin_name}:reference_nonconverged")
                if component_key in calibration_failures:
                    reasons.append(f"{origin_name}:{calibration_failures[component_key]}")
                if (
                    components[component_key].sgd_record.converged
                    and reference.converged
                    and not origin_adequate.get(component_key, False)
                ):
                    reasons.append(f"{origin_name}:optimization_inadequate")
            configuration_reasons[key] = sorted(set(reasons))

    for row in trial_rows:
        component_key = (row["origin"], row["model_id"], row["pair_id"])
        config_key = (row["model_id"], row["pair_id"])
        row["fit_converged"] = components[component_key].sgd_record.converged
        row["calibrator_converged"] = component_key in calibrated
        row["optimization_adequate"] = origin_adequate.get(component_key, False)
        row["eligible"] = not configuration_reasons[config_key]
        row["ineligible_reason"] = "|".join(configuration_reasons[config_key])

    metric_rows: list[dict[str, Any]] = []
    pooled: dict[tuple[str, str], dict[str, float | int]] = {}
    for model_id in MODEL_IDS:
        for pair in registry:
            key = (model_id, pair["pair_id"])
            if configuration_reasons[key]:
                continue
            parts: dict[str, list[np.ndarray]] = {"p": [], "y": [], "u": []}
            for origin_name in search_origins:
                frozen_table = frozen_search_tables[origin_name]
                prefix = f"{model_id}_{pair['pair_id']}"
                probability = np.asarray(
                    arrow_numpy(frozen_table, f"calibrated_probability_{prefix}"),
                    dtype=np.float64,
                )
                labels = np.asarray(arrow_numpy(frozen_table, "long_view"), dtype=np.int8)
                users = np.asarray(arrow_numpy(frozen_table, "user_id"), dtype=np.int64)
                metric_rows.append(
                    {
                        "stage": "search_origin",
                        "origin": origin_name,
                        "model_id": model_id,
                        "pair_id": pair["pair_id"],
                        "alpha": pair["alpha"],
                        "eta0": pair["eta0"],
                        **assessment_metrics(labels, probability, users),
                    }
                )
                parts["p"].append(probability)
                parts["y"].append(labels)
                parts["u"].append(users)
            pooled[key] = assessment_metrics(
                np.concatenate(parts["y"]), np.concatenate(parts["p"]), np.concatenate(parts["u"])
            )
            metric_rows.append(
                {
                    "stage": "search_pooled",
                    "origin": "pooled_3_origins",
                    "model_id": model_id,
                    "pair_id": pair["pair_id"],
                    "alpha": pair["alpha"],
                    "eta0": pair["eta0"],
                    **pooled[key],
                }
            )

    bl0_pooled = _bl0_pooled_metrics(origins, search_origins)
    pair_rows = paired_configuration_rows(
        registry, search_origins, configuration_reasons, metric_rows, pooled
    )
    paired_robustness_configuration_count = sum(
        1
        for pair in registry
        if not configuration_reasons[("BL1", pair["pair_id"])]
        and not configuration_reasons[("BL2", pair["pair_id"])]
    )
    paired_robustness_ineligible = [
        {
            "pair_id": pair["pair_id"],
            "reasons": sorted(
                set(
                    configuration_reasons[("BL1", pair["pair_id"])]
                    + configuration_reasons[("BL2", pair["pair_id"])]
                )
            ),
        }
        for pair in registry
        if configuration_reasons[("BL1", pair["pair_id"])]
        or configuration_reasons[("BL2", pair["pair_id"])]
    ]

    eligible: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    criteria = selection["BL1_shared_configuration_eligibility"]
    for pair in registry:
        key = ("BL1", pair["pair_id"])
        if configuration_reasons[key]:
            gate = {
                "pair_id": pair["pair_id"],
                "alpha": pair["alpha"],
                "eta0": pair["eta0"],
                "optimization_adequate": False,
                "pooled_log_loss_minus_BL0": None,
                "pooled_brier_minus_BL0": None,
                "nonworse_log_loss_origins": 0,
                "nonworse_brier_origins": 0,
                "eligible": False,
                "ineligible_reasons": "|".join(configuration_reasons[key]),
            }
            eligibility_rows.append(gate)
            continue
        row = pooled[key]
        log_loss_days = sum(
            1
            for origin_name in search_origins
            if _origin_metric(metric_rows, origin_name, "BL1", pair["pair_id"], "log_loss")
            <= _origin_bl0_metric(origins[origin_name], "log_loss")
        )
        brier_days = sum(
            1
            for origin_name in search_origins
            if _origin_metric(metric_rows, origin_name, "BL1", pair["pair_id"], "brier")
            <= _origin_bl0_metric(origins[origin_name], "brier")
        )
        gate_reasons: list[str] = []
        if not adequate.get(key, False):
            gate_reasons.append("optimization_inadequate")
        if row["log_loss"] - bl0_pooled["log_loss"] > 0:
            gate_reasons.append("pooled_log_loss_above_BL0")
        if row["brier"] - bl0_pooled["brier"] > 0:
            gate_reasons.append("pooled_brier_above_BL0")
        if log_loss_days < criteria["minimum_nonworse_log_loss_origins"]:
            gate_reasons.append("insufficient_nonworse_log_loss_origins")
        if brier_days < criteria["minimum_nonworse_brier_origins"]:
            gate_reasons.append("insufficient_nonworse_brier_origins")
        gate = {
            "pair_id": pair["pair_id"],
            "alpha": pair["alpha"],
            "eta0": pair["eta0"],
            "optimization_adequate": adequate.get(key, False),
            "pooled_log_loss_minus_BL0": row["log_loss"] - bl0_pooled["log_loss"],
            "pooled_brier_minus_BL0": row["brier"] - bl0_pooled["brier"],
            "nonworse_log_loss_origins": log_loss_days,
            "nonworse_brier_origins": brier_days,
            "ineligible_reasons": "|".join(gate_reasons),
        }
        gate["eligible"] = not gate_reasons
        eligibility_rows.append(gate)
        if gate["eligible"]:
            eligible.append(gate)

    bl1_eligibility_by_pair = {row["pair_id"]: row for row in eligibility_rows}
    for row in trial_rows:
        if row["model_id"] == "BL1":
            eligibility = bl1_eligibility_by_pair[row["pair_id"]]
            row["eligible"] = bool(eligibility["eligible"])
            row["ineligible_reason"] = str(eligibility.get("ineligible_reasons", ""))

    if not eligible:
        persist_search_checkpoint(
            budget=budget,
            book=book,
            trial_rows=trial_rows,
            adequacy_rows=adequacy_rows,
            metric_rows=metric_rows,
            pair_rows=pair_rows,
            selected_payload={
                "eligible_configuration_count": 0,
                "paired_robustness_configuration_count": paired_robustness_configuration_count,
                "paired_robustness_ineligible_pairs": paired_robustness_ineligible,
                "bl1_eligibility": eligibility_rows,
            },
        )
        component_eligible_count = sum(
            1
            for pair in registry
            if not any(
                reason.endswith(("SGD_nonconverged", "calibrator_nonconverged"))
                for reason in configuration_reasons[("BL1", pair["pair_id"])]
            )
        )
        optimization_adequate_count = sum(
            1
            for pair in registry
            if adequate.get(("BL1", pair["pair_id"]), False)
            and not any(
                reason.endswith(("SGD_nonconverged", "calibrator_nonconverged"))
                for reason in configuration_reasons[("BL1", pair["pair_id"])]
            )
        )
        if component_eligible_count == 0:
            raise TerminalStop(
                "search_component_convergence_failure",
                EXIT_SEARCH_COMPONENT_CONVERGENCE_FAILURE,
                "all BL1 configurations were exhausted by base-estimator or primary-"
                "calibrator nonconvergence",
                stage="search",
            )
        if optimization_adequate_count == 0:
            raise TerminalStop(
                "search_optimization_adequacy_failure",
                EXIT_CASE_C_SEARCH_OPTIMIZATION_FAILURE,
                "no optimization-adequate BL1 configuration remained after search fitting",
                stage="search",
            )
        raise TerminalStop(
            "search_probability_quality_failure",
            EXIT_SEARCH_PROBABILITY_QUALITY_FAILURE,
            "optimization-adequate BL1 configurations exist but none passes the search probability gate",
            stage="search",
        )

    selected = _select_shared_configuration(eligible, pooled)
    for row in pair_rows:
        row["primary_shared_pair"] = row["pair_id"] == selected["pair_id"]

    bl2_key = ("BL2", selected["pair_id"])
    bl1_key = ("BL1", selected["pair_id"])
    if configuration_reasons[bl2_key]:
        persist_search_checkpoint(
            budget=budget,
            book=book,
            trial_rows=trial_rows,
            adequacy_rows=adequacy_rows,
            metric_rows=metric_rows,
            pair_rows=pair_rows,
            selected_payload={
                "primary_shared_pair_id": selected["pair_id"],
                "eligible_configuration_count": len(eligible),
                "paired_robustness_configuration_count": paired_robustness_configuration_count,
                "paired_robustness_ineligible_pairs": paired_robustness_ineligible,
                "bl1_eligibility": eligibility_rows,
                "bl2_ineligible_reason": "|".join(configuration_reasons[bl2_key]),
            },
        )
        bl2_component_failure = any(
            reason.endswith(("SGD_nonconverged", "calibrator_nonconverged"))
            for reason in configuration_reasons[bl2_key]
        )
        if not bl2_component_failure:
            raise TerminalStop(
                "search_optimization_adequacy_failure",
                EXIT_CASE_C_SEARCH_OPTIMIZATION_FAILURE,
                "the BL1-selected same-configuration BL2 is not optimization adequate",
                stage="search",
            )
        raise TerminalStop(
            "search_component_convergence_failure",
            EXIT_SEARCH_COMPONENT_CONVERGENCE_FAILURE,
            "the BL1-selected same-configuration BL2 has an ineligible search calibrator",
            stage="search",
        )
    bl2_log_loss_days = sum(
        1
        for origin_name in search_origins
        if _origin_metric(metric_rows, origin_name, "BL2", selected["pair_id"], "log_loss")
        <= _origin_bl0_metric(origins[origin_name], "log_loss")
    )
    bl2_brier_days = sum(
        1
        for origin_name in search_origins
        if _origin_metric(metric_rows, origin_name, "BL2", selected["pair_id"], "brier")
        <= _origin_bl0_metric(origins[origin_name], "brier")
    )
    bl2_gate = {
        "pair_id": selected["pair_id"],
        "optimization_adequate": adequate.get(bl2_key, False),
        "pooled_log_loss_minus_BL0": pooled[bl2_key]["log_loss"] - bl0_pooled["log_loss"],
        "pooled_brier_minus_BL0": pooled[bl2_key]["brier"] - bl0_pooled["brier"],
        "delta_average_precision": pooled[bl2_key]["average_precision"]
        - pooled[bl1_key]["average_precision"],
        "delta_user_gauc_event_weighted": pooled[bl2_key]["user_gauc_event_weighted"]
        - pooled[bl1_key]["user_gauc_event_weighted"],
        "delta_log_loss": pooled[bl2_key]["log_loss"] - pooled[bl1_key]["log_loss"],
        "delta_brier": pooled[bl2_key]["brier"] - pooled[bl1_key]["brier"],
        "nonworse_log_loss_origins_vs_BL0": bl2_log_loss_days,
        "nonworse_brier_origins_vs_BL0": bl2_brier_days,
    }
    positive_ap_origins = sum(
        1
        for origin_name in search_origins
        if _origin_metric(metric_rows, origin_name, "BL2", selected["pair_id"], "average_precision")
        > _origin_metric(metric_rows, origin_name, "BL1", selected["pair_id"], "average_precision")
    )
    bl2_gate["positive_average_precision_origins"] = positive_ap_origins
    bl2_gate["passed"] = primary_bl2_search_gate_passed(
        bl2_gate, selection["primary_BL2_same_configuration_eligibility"]
    )
    if not bl2_gate["passed"]:
        failed_requirements = [
            name
            for name, passed in (
                ("optimization_inadequate", bl2_gate["optimization_adequate"]),
                ("pooled_log_loss_above_BL0", bl2_gate["pooled_log_loss_minus_BL0"] <= 0),
                ("pooled_brier_above_BL0", bl2_gate["pooled_brier_minus_BL0"] <= 0),
                (
                    "insufficient_nonworse_log_loss_origins",
                    bl2_log_loss_days
                    >= selection["primary_BL2_same_configuration_eligibility"][
                        "minimum_nonworse_log_loss_origins_vs_BL0"
                    ],
                ),
                (
                    "insufficient_nonworse_brier_origins",
                    bl2_brier_days
                    >= selection["primary_BL2_same_configuration_eligibility"][
                        "minimum_nonworse_brier_origins_vs_BL0"
                    ],
                ),
                ("nonpositive_delta_average_precision", bl2_gate["delta_average_precision"] > 0),
                (
                    "negative_delta_user_gauc_event_weighted",
                    bl2_gate["delta_user_gauc_event_weighted"] >= 0,
                ),
                ("positive_delta_log_loss", bl2_gate["delta_log_loss"] <= 0),
                ("positive_delta_brier", bl2_gate["delta_brier"] <= 0),
                (
                    "insufficient_positive_average_precision_origins",
                    positive_ap_origins
                    >= selection["primary_BL2_same_configuration_eligibility"][
                        "minimum_positive_average_precision_origins_vs_paired_BL1"
                    ],
                ),
            )
            if not passed
        ]
        bl2_gate["ineligible_reasons"] = "|".join(failed_requirements)
        for row in trial_rows:
            if row["model_id"] == "BL2" and row["pair_id"] == selected["pair_id"]:
                row["eligible"] = False
                row["ineligible_reason"] = bl2_gate["ineligible_reasons"]
        persist_search_checkpoint(
            budget=budget,
            book=book,
            trial_rows=trial_rows,
            adequacy_rows=adequacy_rows,
            metric_rows=metric_rows,
            pair_rows=pair_rows,
            selected_payload={
                "primary_shared_pair_id": selected["pair_id"],
                "eligible_configuration_count": len(eligible),
                "paired_robustness_configuration_count": paired_robustness_configuration_count,
                "paired_robustness_ineligible_pairs": paired_robustness_ineligible,
                "bl1_eligibility": eligibility_rows,
                "bl2_primary_gate": bl2_gate,
            },
        )
        raise TerminalStop(
            "search_primary_pair_gate_failure",
            EXIT_SEARCH_PRIMARY_PAIR_GATE_FAILURE,
            "primary same-configuration BL2 search gate failed",
            stage="search",
        )

    def primary_search_bl2_metric(scope: str, metric_name: str) -> float:
        if scope == "pooled_3_origins":
            return float(pooled[("BL2", selected["pair_id"])][metric_name])
        return _origin_metric(
            metric_rows, scope, "BL2", selected["pair_id"], metric_name
        )

    search_rule_rows = enrich_fixed_rule_rows_with_primary_bl2(
        search_rule_rows,
        primary_pair_id=selected["pair_id"],
        metric_lookup=primary_search_bl2_metric,
    )

    secondary_candidates: list[dict[str, Any]] = []
    for pair in registry:
        pair_name = pair["pair_id"]
        key = ("BL2", pair_name)
        if configuration_reasons[key]:
            continue
        log_days = sum(
            _origin_metric(metric_rows, origin_name, "BL2", pair_name, "log_loss")
            <= _origin_bl0_metric(origins[origin_name], "log_loss")
            for origin_name in search_origins
        )
        brier_days = sum(
            _origin_metric(metric_rows, origin_name, "BL2", pair_name, "brier")
            <= _origin_bl0_metric(origins[origin_name], "brier")
            for origin_name in search_origins
        )
        if (
            pooled[key]["log_loss"] - bl0_pooled["log_loss"] <= 0
            and pooled[key]["brier"] - bl0_pooled["brier"] <= 0
            and log_days
            >= selection["primary_BL2_same_configuration_eligibility"][
                "minimum_nonworse_log_loss_origins_vs_BL0"
            ]
            and brier_days
            >= selection["primary_BL2_same_configuration_eligibility"][
                "minimum_nonworse_brier_origins_vs_BL0"
            ]
        ):
            secondary_candidates.append(pair)
    secondary_selected = _select_secondary_bl2(secondary_candidates, pooled)
    if secondary_selected is not None:
        secondary_pair = secondary_selected["pair_id"]
        primary_pair = selected["pair_id"]
        for scope in [*search_origins, "pooled_3_origins"]:
            for metric_name in (
                "average_precision",
                "user_gauc_event_weighted",
                "log_loss",
                "brier",
            ):
                if scope == "pooled_3_origins":
                    bl1_value = pooled[("BL1", primary_pair)][metric_name]
                    bl2_value = pooled[("BL2", secondary_pair)][metric_name]
                else:
                    bl1_value = _origin_metric(
                        metric_rows, scope, "BL1", primary_pair, metric_name
                    )
                    bl2_value = _origin_metric(
                        metric_rows, scope, "BL2", secondary_pair, metric_name
                    )
                pair_rows.append(
                    {
                        "comparison_id": "secondary_independently_optimized_BL2_vs_primary_BL1",
                        "origin_or_pooled": scope,
                        "pair_id": secondary_pair,
                        "BL1_pair_id": primary_pair,
                        "BL2_pair_id": secondary_pair,
                        "alpha": secondary_selected["alpha"],
                        "eta0": secondary_selected["eta0"],
                        "metric": metric_name,
                        "BL1_metric": bl1_value,
                        "BL2_metric": bl2_value,
                        "BL2_minus_BL1_delta": bl2_value - bl1_value,
                        "primary_shared_pair": False,
                        "configuration_eligible": True,
                        "ineligible_reason": "",
                        "role": "engineering_diagnostic_only_not_primary_H2_evidence",
                    }
                )

    secondary_ids = {item["pair_id"] for item in secondary_candidates}
    for row in trial_rows:
        if row["model_id"] == "BL1":
            eligibility = bl1_eligibility_by_pair[row["pair_id"]]
            row["eligible"] = bool(eligibility["eligible"])
            row["ineligible_reason"] = str(eligibility.get("ineligible_reasons", ""))
        else:
            row["eligible"] = row["pair_id"] in secondary_ids
            if not row["eligible"] and not row["ineligible_reason"]:
                row["ineligible_reason"] = "BL2_absolute_probability_gate_failed"

    persist_search_checkpoint(
        budget=budget,
        book=book,
        trial_rows=trial_rows,
        adequacy_rows=adequacy_rows,
        metric_rows=metric_rows,
        pair_rows=pair_rows,
        selected_payload={
            "primary_shared_pair_id": selected["pair_id"],
            "BL1": {"bundle": BUNDLES["BL1"], **selected},
            "BL2": {"bundle": BUNDLES["BL2"], **selected},
            "eligible_configuration_count": len(eligible),
            "paired_robustness_configuration_count": paired_robustness_configuration_count,
            "paired_robustness_ineligible_pairs": paired_robustness_ineligible,
            "secondary_BL2": secondary_selected,
            "secondary_eligible_configuration_count": len(secondary_candidates),
            "bl1_eligibility": eligibility_rows,
            "bl2_primary_gate": bl2_gate,
        },
    )

    return {
        "components": components,
        "calibrated": calibrated,
        "adequacy_rows": adequacy_rows,
        "reference_records": reference_records,
        "reference_C": reference_C,
        "metric_rows": metric_rows,
        "pooled": pooled,
        "bl0_pooled": bl0_pooled,
        "selected": selected,
        "bl1_eligibility_rows": eligibility_rows,
        "bl2_gate": bl2_gate,
        "pair_rows": pair_rows,
        "trial_rows": trial_rows,
        "prediction_paths": prediction_paths,
        "search_freeze_manifest": search_freeze_manifest,
        "search_rule_rows": search_rule_rows,
        "frozen_rule_scores": frozen_rule_scores,
        "configuration_reasons": configuration_reasons,
        "eligible_configuration_count": len(eligible),
        "paired_robustness_configuration_count": paired_robustness_configuration_count,
        "paired_robustness_ineligible_pairs": paired_robustness_ineligible,
        "secondary_selected": secondary_selected,
        "secondary_eligible_configuration_count": len(secondary_candidates),
    }


def _write_search_predictions(
    frame: Frame,
    origins: dict[str, OriginMatrices],
    search_origins: list[str],
    components: dict[tuple[str, str, str], Component],
    calibrated: dict[tuple[str, str, str], np.ndarray],
    calibrators: dict[tuple[str, str, str], repair.Calibrator],
    reference_records: dict[tuple[str, str, float], repair.FitRecord],
    reference_C: dict[tuple[str, str, float], float],
) -> tuple[list[str], dict[tuple[str, str], np.ndarray]]:
    """Contract: ``artifact_schemas.prediction_artifacts`` (search scope)."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    frozen_rule_scores: dict[tuple[str, str], np.ndarray] = {}
    for origin_name in search_origins:
        origin = origins[origin_name]
        index = origin.split.assessment_index
        table: dict[str, np.ndarray] = {
            "source_table": frame.columns["source_table"][index],
            "source_row_number": frame.columns["source_row_number"][index],
            "user_id": frame.columns["user_id"][index],
            "video_id": frame.columns["video_id"][index],
            "origin": np.full(index.shape, origin_name),
            "long_view": origin.labels["assessment"],
            "prior_batch_n": frame.columns["prior_batch_n"][index],
            "p_BL0": np.full(index.shape, origin.split.bl0_probability),
        }
        for diagnostic_id, score in fixed_rule_scores(
            frame, index, origin.split.fit_prevalence
        ).items():
            table[f"fixed_rule_score_{diagnostic_id}"] = score
            frozen_rule_scores[(origin_name, diagnostic_id)] = score
        for (o_name, model_id, pair_name), component in components.items():
            if o_name != origin_name:
                continue
            prefix = f"{model_id}_{pair_name}"
            calibrator = calibrators.get((o_name, model_id, pair_name))
            table[f"pair_id_{prefix}"] = np.full(index.shape, pair_name)
            table[f"alpha_{prefix}"] = np.full(index.shape, component.alpha)
            table[f"eta0_{prefix}"] = np.full(index.shape, component.eta0)
            table[f"raw_decision_score_{prefix}"] = component.raw_assessment
            table[f"uncalibrated_probability_{prefix}"] = component.uncalibrated_assessment
            if calibrator is None:
                # A1: search nonconvergence makes this configuration ineligible
                # but does not abort the frozen grid.  Arrow null is the only
                # permitted representation: NaN would violate the global
                # non-finite probability prohibition and could be mistaken for
                # a numeric prediction by downstream code.
                table[f"calibrated_probability_{prefix}"] = pa.nulls(
                    index.size, type=pa.float64()
                )
                table[f"calibration_intercept_{prefix}"] = pa.nulls(
                    index.size, type=pa.float64()
                )
                table[f"calibration_slope_{prefix}"] = pa.nulls(
                    index.size, type=pa.float64()
                )
            else:
                table[f"calibrated_probability_{prefix}"] = calibrated[
                    (o_name, model_id, pair_name)
                ]
                table[f"calibration_intercept_{prefix}"] = np.full(
                    index.shape, calibrator.intercept
                )
                table[f"calibration_slope_{prefix}"] = np.full(index.shape, calibrator.slope)
        for (o_name, model_id, alpha), C in reference_C.items():
            if o_name != origin_name:
                continue
            record = reference_records[(o_name, model_id, alpha)]
            if record.assessment_raw_score is None:
                raise ContractStop(f"reference raw score missing for {o_name}/{model_id}/{alpha}")
            slug = alpha_slug(alpha)
            table[f"reference_raw_decision_score_{model_id}_{slug}"] = record.assessment_raw_score
            table[f"reference_model_id_{model_id}_{slug}"] = np.full(index.shape, model_id)
            table[f"reference_C_{model_id}_{slug}"] = np.full(index.shape, C)
            table[f"reference_alpha_{model_id}_{slug}"] = np.full(index.shape, alpha)
        path = OUTPUT_DIR / f"search_predictions_origin_{origin_name}.parquet"
        write_parquet(path, pa.table(table))
        written.append(path.name)
    return written, frozen_rule_scores


def freeze_search_artifacts(
    prediction_paths: list[str],
    frozen_rule_scores: dict[tuple[str, str], np.ndarray],
) -> dict[str, Any]:
    paths = [OUTPUT_DIR / name for name in prediction_paths]
    artifacts = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    payload = {
        "status": "frozen_before_any_derived_search_rule_or_candidate_assessment_metric",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "fixed_rule_scores": [
            {
                "origin": origin,
                "diagnostic_id": diagnostic_id,
                "rows": int(np.asarray(score).size),
                "dtype": "little_endian_float64",
                "sha256": hashlib.sha256(
                    np.asarray(score, dtype="<f8").tobytes(order="C")
                ).hexdigest(),
                "storage_column": f"fixed_rule_score_{diagnostic_id}",
            }
            for (origin, diagnostic_id), score in sorted(frozen_rule_scores.items())
        ],
    }
    write_json(OUTPUT_DIR / "search_artifact_freeze_manifest.json", payload)
    return payload


def read_frozen_search_tables(
    freeze_manifest: dict[str, Any],
) -> dict[str, pa.Table]:
    """Re-read hash-frozen search Parquets before deriving assessment metrics."""

    required_status = "frozen_before_any_derived_search_rule_or_candidate_assessment_metric"
    if freeze_manifest.get("status") != required_status:
        raise ContractStop("search freeze manifest has the wrong status")
    artifacts = list(freeze_manifest.get("artifacts", []))
    if int(freeze_manifest.get("artifact_count", -1)) != len(artifacts):
        raise ContractStop("search freeze manifest artifact_count does not reconcile")
    if len({str(item.get("path")) for item in artifacts}) != len(artifacts):
        raise ContractStop("search freeze manifest paths are not unique")

    required_columns = {
        "source_table",
        "source_row_number",
        "user_id",
        "video_id",
        "origin",
        "long_view",
        "fixed_rule_score_LIFETIME_SMOOTHED_RATE",
        "fixed_rule_score_W10_SMOOTHED_RATE",
    }
    tables: dict[str, pa.Table] = {}
    for entry in artifacts:
        name = str(entry["path"])
        path = OUTPUT_DIR / name
        if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]):
            raise ContractStop(f"frozen search artifact size mismatch: {name}")
        if sha256_file(path) != str(entry["sha256"]):
            raise ContractStop(f"frozen search artifact SHA-256 mismatch: {name}")
        table = pq.read_table(path)
        missing = sorted(required_columns - set(table.column_names))
        if missing:
            raise ContractStop(f"frozen search artifact {name} is missing columns: {missing}")
        origin = name.removeprefix("search_predictions_origin_").removesuffix(".parquet")
        if origin in tables:
            raise ContractStop(f"duplicate frozen search origin: {origin}")
        if table.num_rows <= 0:
            raise ContractStop(f"frozen search artifact has no rows: {name}")
        tables[origin] = table
        if path.stat().st_size != int(entry["size_bytes"]) or sha256_file(path) != str(
            entry["sha256"]
        ):
            raise ContractStop(f"frozen search artifact changed while being read: {name}")
    rule_entries = list(freeze_manifest.get("fixed_rule_scores", []))
    if len(rule_entries) != 2 * len(tables):
        raise ContractStop("search freeze manifest fixed-rule vector count does not reconcile")
    for entry in rule_entries:
        origin = str(entry["origin"])
        diagnostic_id = str(entry["diagnostic_id"])
        table = tables.get(origin)
        if table is None:
            raise ContractStop(f"fixed-rule freeze entry has unknown origin {origin}")
        score = np.asarray(
            arrow_numpy(table, f"fixed_rule_score_{diagnostic_id}"), dtype="<f8"
        )
        observed = hashlib.sha256(score.tobytes(order="C")).hexdigest()
        if int(entry["rows"]) != int(score.size) or observed != str(entry["sha256"]):
            raise ContractStop(
                f"fixed-rule vector hash mismatch after Parquet readback: {origin}/{diagnostic_id}"
            )
    return tables


def arrow_numpy(table: pa.Table, column: str) -> np.ndarray:
    if column not in table.column_names:
        raise ContractStop(f"frozen search table is missing required column {column}")
    return table.column(column).combine_chunks().to_numpy(zero_copy_only=False)


def frozen_search_fixed_rule_metric_rows(
    freeze_manifest: dict[str, Any], frozen_tables: dict[str, pa.Table]
) -> list[dict[str, Any]]:
    """Derive the registered 6+2 rule rows only from frozen Parquet columns."""

    entry_by_origin = {
        str(entry["path"])
        .removeprefix("search_predictions_origin_")
        .removesuffix(".parquet"): entry
        for entry in freeze_manifest["artifacts"]
    }
    rows: list[dict[str, Any]] = []
    pooled_labels: list[np.ndarray] = []
    pooled_users: list[np.ndarray] = []
    pooled_scores: dict[str, list[np.ndarray]] = {
        "LIFETIME_SMOOTHED_RATE": [],
        "W10_SMOOTHED_RATE": [],
    }
    for origin in sorted(frozen_tables):
        table = frozen_tables[origin]
        y = np.asarray(arrow_numpy(table, "long_view"), dtype=np.int8)
        users = np.asarray(arrow_numpy(table, "user_id"), dtype=np.int64)
        pooled_labels.append(y)
        pooled_users.append(users)
        source = str(entry_by_origin[origin]["sha256"])
        for diagnostic_id in pooled_scores:
            score = np.asarray(
                arrow_numpy(table, f"fixed_rule_score_{diagnostic_id}"), dtype=np.float64
            )
            pooled_scores[diagnostic_id].append(score)
            point = assessment_metrics(y, score, users)
            rows.append(
                {
                    "stage": "search",
                    "origin_or_pooled": origin,
                    "diagnostic_id": diagnostic_id,
                    "rows": point["rows"],
                    "users": point["users"],
                    "prevalence": point["prevalence"],
                    "average_precision": point["average_precision"],
                    "roc_auc": point["roc_auc"],
                    "user_gauc_event_weighted": point["user_gauc_event_weighted"],
                    "user_gauc_user_equal": point["user_gauc_user_equal"],
                    "frozen_search_prediction_artifact_SHA256_source": source,
                }
            )
    pooled_source = json.dumps(
        [
            {"path": str(entry["path"]), "sha256": str(entry["sha256"])}
            for entry in sorted(freeze_manifest["artifacts"], key=lambda item: str(item["path"]))
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    labels = np.concatenate(pooled_labels)
    users = np.concatenate(pooled_users)
    for diagnostic_id, score_parts in pooled_scores.items():
        point = assessment_metrics(labels, np.concatenate(score_parts), users)
        rows.append(
            {
                "stage": "search",
                "origin_or_pooled": "pooled_3_origins",
                "diagnostic_id": diagnostic_id,
                "rows": point["rows"],
                "users": point["users"],
                "prevalence": point["prevalence"],
                "average_precision": point["average_precision"],
                "roc_auc": point["roc_auc"],
                "user_gauc_event_weighted": point["user_gauc_event_weighted"],
                "user_gauc_user_equal": point["user_gauc_user_equal"],
                "frozen_search_prediction_artifact_SHA256_source": pooled_source,
            }
        )
    return rows


def persist_search_checkpoint(
    *,
    budget: Budget,
    book: LedgerBook,
    trial_rows: list[dict[str, Any]],
    adequacy_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]] | None = None,
    selected_payload: dict[str, Any] | None = None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = yaml.load(CONTRACT_PATH.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    schemas = contract["artifact_schemas"]
    write_csv(OUTPUT_DIR / "search_trial_manifest.csv", trial_rows)
    write_csv_or_header(
        OUTPUT_DIR / "convergence_ledger.csv",
        book.convergence,
        schemas["convergence_ledger_required_fields"],
    )
    write_csv_or_header(
        OUTPUT_DIR / "reference_solver_ledger.csv",
        book.reference,
        schemas["reference_solver_ledger_required_fields"],
    )
    write_csv_or_header(
        OUTPUT_DIR / "calibration_fit_ledger.csv",
        book.calibration,
        schemas["calibration_fit_ledger_required_fields"],
    )
    write_csv_or_header(
        OUTPUT_DIR / "usage_ledger.csv",
        budget.ledger,
        USAGE_LEDGER_FIELDS,
    )
    write_csv_or_header(
        OUTPUT_DIR / "optimization_objective_audit.csv",
        adequacy_rows,
        schemas["optimization_objective_audit_required_fields"],
    )
    write_csv_or_header(
        OUTPUT_DIR / "search_metrics.csv",
        metric_rows,
        [
            "stage",
            "origin",
            "model_id",
            "pair_id",
            "alpha",
            "eta0",
            "rows",
            "users",
            "positives",
            "prevalence",
            "average_precision",
            "roc_auc",
            "user_gauc_event_weighted",
            "user_gauc_user_equal",
            "log_loss",
            "brier",
            "ece20_equal_width",
        ],
    )
    write_csv_or_header(
        OUTPUT_DIR / "paired_configuration_metrics.csv",
        pair_rows or [],
        schemas["paired_configuration_metrics_required_fields"],
    )
    if selected_payload is not None:
        write_json(OUTPUT_DIR / "selected_models.json", selected_payload)
    budget.check_elapsed("persist_search_checkpoint")
    budget.check_storage("persist_search_checkpoint")


def frozen_daily_backtest(
    contract: dict[str, Any],
    budget: Budget,
    book: LedgerBook,
    frame: Frame,
    origins: dict[str, OriginMatrices],
    search: dict[str, Any],
) -> dict[str, Any]:
    """Contract: ``frozen_daily_backtest.frozen_daily_phase_order``."""

    backtest = contract["search_and_selection"]["frozen_daily_backtest"]
    daily_origins = [str(value) for value in backtest["origins"]]
    reference_origins = {str(value) for value in backtest["reference_solver_origins"]}
    selected = search["selected"]
    alpha = float(selected["alpha"])
    eta0 = float(selected["eta0"])

    components: dict[tuple[str, str], Component] = {}
    for origin_name in daily_origins:
        for model_id in MODEL_IDS:
            components[(origin_name, model_id)] = fit_component(
                budget,
                book,
                origins[origin_name],
                model_id=model_id,
                alpha=alpha,
                eta0=eta0,
                stage="frozen_daily",
            )

    reference_records: dict[tuple[str, str], repair.FitRecord] = {}
    reference_source: dict[tuple[str, str], str] = {}
    for origin_name in daily_origins:
        for model_id in MODEL_IDS:
            key = (origin_name, model_id)
            if origin_name in reference_origins:
                record, _ = fit_reference_component(
                    budget,
                    book,
                    origins[origin_name],
                    model_id=model_id,
                    alpha=alpha,
                    stage="frozen_daily",
                )
                reference_source[key] = "frozen_daily_reference_fit"
            else:
                record = search["reference_records"][(origin_name, model_id, alpha)]
                reference_source[key] = "reused_search_reference_fit"
            reference_records[key] = record

    adequacy_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for origin_name in daily_origins:
        for model_id in MODEL_IDS:
            key = (origin_name, model_id)
            decision = repair.adequacy_decision(
                components[key].sgd_record.objective,
                reference_records[key].objective,
                reference_converged=reference_records[key].converged,
            )
            adequacy_rows.append(
                {
                    "phase": "frozen_daily",
                    "origin": origin_name,
                    "model_id": model_id,
                    "pair_id": selected["pair_id"],
                    "alpha": alpha,
                    "eta0": eta0,
                    "reference_fit_source": reference_source[key],
                    **decision,
                    "assessment_AP_regret_diagnostic": None,
                    "assessment_ROC_AUC_regret_diagnostic": None,
                    "assessment_user_GAUC_regret_diagnostic": None,
                    "coefficient_cosine_similarity": None,
                    "coefficient_L2_norm_ratio": None,
                }
            )
            if not decision["adequacy_passed"]:
                failures.append(f"{origin_name}/{model_id}")
    if failures:
        # Preserve the failed daily adequacy evidence before the fail-closed
        # terminal state. Search artifacts were already checkpointed.
        write_csv(
            OUTPUT_DIR / "optimization_objective_audit.csv",
            search["adequacy_rows"] + adequacy_rows,
        )
        write_csv(OUTPUT_DIR / "convergence_ledger.csv", book.convergence)
        write_csv(OUTPUT_DIR / "reference_solver_ledger.csv", book.reference)
        write_csv(OUTPUT_DIR / "calibration_fit_ledger.csv", book.calibration)
        write_csv(OUTPUT_DIR / "usage_ledger.csv", budget.ledger)
        budget.check_elapsed("frozen_daily_adequacy_failure_checkpoint")
        budget.check_storage("frozen_daily_adequacy_failure_checkpoint")
        raise TerminalStop(
            "frozen_daily_optimization_adequacy_failure",
            EXIT_CASE_D_FROZEN_DAILY_OPTIMIZATION_FAILURE,
            "frozen daily optimization adequacy failed on "
            f"{sorted(failures)}; stopping before metric aggregation. "
            "Day-level exclusion or reweighting is forbidden.",
            stage="frozen_daily",
        )

    # Complete every one of the 14 frozen daily primary calibrator fits before
    # reading any daily assessment metric.  A late-day component failure thus
    # cannot leave a partially aggregated scientific result.
    daily_calibrators: dict[tuple[str, str], repair.Calibrator] = {}
    daily_probabilities: dict[tuple[str, str], np.ndarray] = {}
    for origin_name in daily_origins:
        origin = origins[origin_name]
        for model_id in MODEL_IDS:
            component = components[(origin_name, model_id)]
            outcome = calibrate_component(
                budget, book, origin, component, stage="frozen_daily"
            )
            if not outcome.eligible or outcome.calibrator is None or outcome.probability is None:
                raise TerminalStop(
                    "implementation_or_governance_failure",
                    EXIT_IMPLEMENTATION_OR_GOVERNANCE_FAILURE,
                    f"frozen daily calibration unexpectedly ineligible for {origin_name}/{model_id}",
                    stage="frozen_daily",
                )
            daily_calibrators[(origin_name, model_id)] = outcome.calibrator
            daily_probabilities[(origin_name, model_id)] = outcome.probability

    budget.record_nonfitted_rule_evaluations(
        2 * len(daily_origins), stage="frozen_daily"
    )

    # Only now may the assessment-only regret and monotonic-AP diagnostics be
    # materialized; neither feeds the already-completed adequacy decision.
    adequacy_by_key = {
        (str(row["origin"]), str(row["model_id"])): row for row in adequacy_rows
    }
    for origin_name in daily_origins:
        origin = origins[origin_name]
        for model_id in MODEL_IDS:
            key = (origin_name, model_id)
            adequacy_by_key[key].update(
                regret_diagnostics(components[key], reference_records[key], origin)
            )
            verify_raw_calibrated_ap_equivalence(
                book,
                origin.labels["assessment"],
                components[key].raw_assessment,
                daily_probabilities[key],
                stage="frozen_daily",
                origin_name=origin_name,
                model_id=model_id,
                pair_id=selected["pair_id"],
            )

    daily_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []
    predictions: dict[str, list[np.ndarray]] = {"BL0": [], "BL1": [], "BL2": []}
    label_parts: list[np.ndarray] = []
    user_parts: list[np.ndarray] = []
    prior_parts: list[np.ndarray] = []
    identity: dict[str, list[np.ndarray]] = {
        name: [] for name in ("source_table", "source_row_number", "user_id", "video_id", "origin")
    }
    raw_parts: dict[str, list[np.ndarray]] = {"BL1": [], "BL2": []}
    calibrator_parts: dict[str, list[np.ndarray]] = {"BL1": [], "BL2": []}
    slope_parts: dict[str, list[np.ndarray]] = {"BL1": [], "BL2": []}
    uncalibrated_parts: dict[str, list[np.ndarray]] = {"BL1": [], "BL2": []}
    rule_parts: dict[str, list[np.ndarray]] = {}

    for origin_name in daily_origins:
        origin = origins[origin_name]
        index = origin.split.assessment_index
        y = origin.labels["assessment"]
        users = origin.users["assessment"]
        label_parts.append(y)
        user_parts.append(users)
        prior_parts.append(frame.columns["prior_batch_n"][index])
        for name in identity:
            identity[name].append(
                np.full(index.shape, origin_name)
                if name == "origin"
                else frame.columns[name][index]
            )
        bl0 = np.full(y.shape, origin.split.bl0_probability, dtype=np.float64)
        predictions["BL0"].append(bl0)
        daily_rows.append(
            {
                "origin": origin_name,
                "model_id": "BL0",
                **contract_named_metrics(y, bl0, users),
                **uncalibrated_named_metrics(y, bl0, users),
                **probability_distribution(bl0),
                **{
                    name: None
                    for name in (
                        "raw_score_average_precision_within_origin",
                        "raw_score_user_gauc_event_weighted_within_origin",
                        "raw_score_user_gauc_user_equal_within_origin",
                    )
                },
            }
        )
        distribution_rows.append(
            {"origin": origin_name, "model_id": "BL0", **probability_distribution(bl0)}
        )
        for model_id in MODEL_IDS:
            component = components[(origin_name, model_id)]
            calibrator = daily_calibrators[(origin_name, model_id)]
            probability = daily_probabilities[(origin_name, model_id)]
            predictions[model_id].append(probability)
            raw_parts[model_id].append(component.raw_assessment)
            uncalibrated_parts[model_id].append(component.uncalibrated_assessment)
            calibrator_parts[model_id].append(np.full(y.shape, calibrator.intercept))
            slope_parts[model_id].append(np.full(y.shape, calibrator.slope))
            daily_rows.append(
                {
                    "origin": origin_name,
                    "model_id": model_id,
                    "pair_id": selected["pair_id"],
                    **contract_named_metrics(y, probability, users),
                    **uncalibrated_named_metrics(
                        y, component.uncalibrated_assessment, users
                    ),
                    **probability_distribution(probability),
                    **raw_score_within_origin_metrics(y, component.raw_assessment, users),
                }
            )
            distribution_rows.append(
                {
                    "origin": origin_name,
                    "model_id": model_id,
                    **probability_distribution(probability),
                }
            )
        for name, score in fixed_rule_scores(
            frame, index, origin.split.fit_prevalence
        ).items():
            frozen_search_score = search["frozen_rule_scores"].get((origin_name, name))
            if frozen_search_score is not None and not np.array_equal(
                np.asarray(score, dtype=np.float64),
                np.asarray(frozen_search_score, dtype=np.float64),
            ):
                raise ContractStop(
                    f"search/frozen-daily fixed-rule raw score mismatch for {origin_name}/{name}"
                )
            rule_parts.setdefault(name, []).append(score)
            point = assessment_metrics(y, score, users)
            rule_rows.append(
                {
                    "stage": "frozen_daily",
                    "origin_or_pooled": origin_name,
                    "diagnostic_id": name,
                    "rows": point["rows"],
                    "users": point["users"],
                    "prevalence": point["prevalence"],
                    "average_precision": point["average_precision"],
                    "roc_auc": point["roc_auc"],
                    "user_gauc_event_weighted": point["user_gauc_event_weighted"],
                    "user_gauc_user_equal": point["user_gauc_user_equal"],
                }
            )

    labels = np.concatenate(label_parts)
    users = np.concatenate(user_parts)
    prior_batch = np.concatenate(prior_parts)
    pooled_predictions = {name: np.concatenate(parts) for name, parts in predictions.items()}
    pooled_uncalibrated = {
        "BL0": pooled_predictions["BL0"],
        "BL1": np.concatenate(uncalibrated_parts["BL1"]),
        "BL2": np.concatenate(uncalibrated_parts["BL2"]),
    }
    for name, parts in rule_parts.items():
        point = assessment_metrics(labels, np.concatenate(parts), users)
        rule_rows.append(
            {
                "stage": "frozen_daily",
                "origin_or_pooled": "pooled_7_days",
                "diagnostic_id": name,
                "rows": point["rows"],
                "users": point["users"],
                "prevalence": point["prevalence"],
                "average_precision": point["average_precision"],
                "roc_auc": point["roc_auc"],
                "user_gauc_event_weighted": point["user_gauc_event_weighted"],
                "user_gauc_user_equal": point["user_gauc_user_equal"],
            }
        )

    for model_id in ("BL0", "BL1", "BL2"):
        pooled_row = {
            "origin": "pooled_7_days",
            "model_id": model_id,
            **contract_named_metrics(labels, pooled_predictions[model_id], users),
            **uncalibrated_named_metrics(labels, pooled_uncalibrated[model_id], users),
            **probability_distribution(pooled_predictions[model_id]),
            "raw_score_average_precision_within_origin": None,
            "raw_score_user_gauc_event_weighted_within_origin": None,
            "raw_score_user_gauc_user_equal_within_origin": None,
        }
        if model_id != "BL0":
            pooled_row["pair_id"] = selected["pair_id"]
        daily_rows.append(pooled_row)
        distribution_rows.append(
            {
                "origin": "pooled_7_days",
                "model_id": model_id,
                **probability_distribution(pooled_predictions[model_id]),
            }
        )

    def primary_daily_bl2_metric(scope: str, metric_name: str) -> float:
        for metric_row in daily_rows:
            if metric_row["origin"] == scope and metric_row["model_id"] == "BL2":
                return float(metric_row[metric_name])
        raise ContractStop(f"missing primary BL2 daily metric for {scope}/{metric_name}")

    rule_rows = enrich_fixed_rule_rows_with_primary_bl2(
        rule_rows,
        primary_pair_id=selected["pair_id"],
        metric_lookup=primary_daily_bl2_metric,
    )

    search_rule_lookup = {
        (row["origin_or_pooled"], row["diagnostic_id"]): row
        for row in search["search_rule_rows"]
        if row["origin_or_pooled"] != "pooled_3_origins"
    }
    daily_rule_lookup = {
        (row["origin_or_pooled"], row["diagnostic_id"]): row
        for row in rule_rows
        if row["origin_or_pooled"] != "pooled_7_days"
    }
    for key, search_row in search_rule_lookup.items():
        daily_row = daily_rule_lookup.get(key)
        if daily_row is None:
            raise ContractStop(f"missing frozen-daily fixed-rule diagnostic for {key}")
        for metric_name in (
            "rows",
            "users",
            "prevalence",
            "average_precision",
            "roc_auc",
            "user_gauc_event_weighted",
            "user_gauc_user_equal",
        ):
            if daily_row[metric_name] != search_row[metric_name]:
                raise ContractStop(
                    f"search/frozen-daily fixed-rule diagnostic mismatch for {key}/{metric_name}"
                )

    daily_predictions = {
        **{name: np.concatenate(parts) for name, parts in identity.items()},
        "long_view": labels,
        "prior_batch_n": prior_batch,
        "p_BL0": pooled_predictions["BL0"],
        "pair_id": np.full(labels.shape, selected["pair_id"]),
        "alpha": np.full(labels.shape, alpha),
        "eta0": np.full(labels.shape, eta0),
    }
    for model_id in MODEL_IDS:
        daily_predictions[f"raw_decision_score_{model_id}"] = np.concatenate(raw_parts[model_id])
        daily_predictions[f"uncalibrated_probability_{model_id}"] = np.concatenate(
            uncalibrated_parts[model_id]
        )
        daily_predictions[f"calibrated_probability_{model_id}"] = pooled_predictions[model_id]
        daily_predictions[f"calibration_intercept_{model_id}"] = np.concatenate(
            calibrator_parts[model_id]
        )
        daily_predictions[f"calibration_slope_{model_id}"] = np.concatenate(slope_parts[model_id])

    # 21 per-origin + 3 pooled declared; BL0 is not applicable on both scopes, so
    # 16 are actually executed and the remaining 8 are reported as N/A.
    calibration_regression_rows: list[dict[str, Any]] = []
    for position, origin_name in enumerate(daily_origins):
        y_origin = origins[origin_name].labels["assessment"]
        for model_id in ("BL0", "BL1", "BL2"):
            # predictions[model_id] is one array per origin in daily_origins order
            probability = predictions[model_id][position]
            if probability.shape != y_origin.shape:
                raise ContractStop(
                    f"daily prediction block for {origin_name}/{model_id} is misaligned"
                )
            calibration_regression_rows.append(
                assessment_calibration_regression(
                    budget, y_origin, probability, scope=origin_name, model_id=model_id
                )
            )
    for model_id in ("BL0", "BL1", "BL2"):
        calibration_regression_rows.append(
            assessment_calibration_regression(
                budget,
                labels,
                pooled_predictions[model_id],
                scope="pooled_7_days",
                model_id=model_id,
            )
        )

    return {
        "selected": selected,
        "daily_origins": daily_origins,
        "adequacy_rows": adequacy_rows,
        "daily_rows": daily_rows,
        "calibration_regression_rows": calibration_regression_rows,
        "distribution_rows": distribution_rows,
        "rule_rows": rule_rows,
        "labels": labels,
        "users": users,
        "prior_batch": prior_batch,
        "predictions": pooled_predictions,
        "origin_sizes": [int(part.size) for part in label_parts],
        "daily_prediction_table": daily_predictions,
    }


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def _calibration_regression_lookup(
    daily: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Attach the per-origin assessment calibration intercept and slope."""

    for entry in daily["calibration_regression_rows"]:
        if entry["scope"] == row["origin"] and entry["model_id"] == row["model_id"]:
            return {
                "assessment_calibration_intercept": entry["assessment_calibration_intercept"],
                "assessment_calibration_slope": entry["assessment_calibration_slope"],
                "assessment_calibration_status": entry["status"],
            }
    raise ContractStop(
        f"no assessment calibration regression row for {row['origin']}/{row['model_id']}"
    )


def regret_diagnostics(
    component: Component, reference: repair.FitRecord, origin: OriginMatrices
) -> dict[str, float | None]:
    """Contract: the three ``assessment_*_regret_diagnostic`` fields.

    Explicitly listed under ``assessment_diagnostics_not_used_for_eligibility``:
    they describe how far the SGD solution sits from the reference on assessment
    rows, and must never feed the adequacy verdict, which is training-objective
    only.
    """

    if reference.assessment_raw_score is None:
        return {
            "assessment_AP_regret_diagnostic": None,
            "assessment_ROC_AUC_regret_diagnostic": None,
            "assessment_user_GAUC_regret_diagnostic": None,
            "coefficient_cosine_similarity": None,
            "coefficient_L2_norm_ratio": None,
        }
    y = origin.labels["assessment"]
    users = origin.users["assessment"]
    sgd = raw_score_within_origin_metrics(y, component.raw_assessment, users)
    ref = raw_score_within_origin_metrics(y, reference.assessment_raw_score, users)
    sgd_coefficient = component.sgd_record.coefficient
    reference_coefficient = reference.coefficient
    coefficient_cosine: float | None = None
    coefficient_norm_ratio: float | None = None
    if sgd_coefficient is not None and reference_coefficient is not None:
        sgd_norm = float(np.linalg.norm(sgd_coefficient))
        reference_norm = float(np.linalg.norm(reference_coefficient))
        if sgd_norm > 0.0 and reference_norm > 0.0:
            coefficient_cosine = float(
                np.dot(sgd_coefficient, reference_coefficient) / (sgd_norm * reference_norm)
            )
            coefficient_norm_ratio = sgd_norm / reference_norm
    return {
        "assessment_AP_regret_diagnostic": ref["raw_score_average_precision_within_origin"]
        - sgd["raw_score_average_precision_within_origin"],
        "assessment_ROC_AUC_regret_diagnostic": metrics.binary_auc_midrank(
            y, reference.assessment_raw_score
        )
        - metrics.binary_auc_midrank(y, component.raw_assessment),
        "assessment_user_GAUC_regret_diagnostic": ref[
            "raw_score_user_gauc_event_weighted_within_origin"
        ]
        - sgd["raw_score_user_gauc_event_weighted_within_origin"],
        "coefficient_cosine_similarity": coefficient_cosine,
        "coefficient_L2_norm_ratio": coefficient_norm_ratio,
    }


def _daily_value(rows: list[dict[str, Any]], origin: str, model_id: str, metric: str) -> float:
    for row in rows:
        if row["origin"] == origin and row["model_id"] == model_id:
            return float(row[metric])
    raise ContractStop(f"missing daily metric {metric} for {origin}/{model_id}")


def probability_quality_gate(
    contract: dict[str, Any], origins: dict[str, OriginMatrices], daily: dict[str, Any]
) -> dict[str, Any]:
    """Contract: ``probability_quality_gate`` (mandatory absolute sanity)."""

    spec = contract["probability_quality_gate"]
    tolerance = float(spec["numerical_comparison_tolerance"])
    stability = spec["daily_stability_requirements"]
    pooled_bl0 = _bl0_pooled_metrics(origins, daily["daily_origins"])
    results: dict[str, Any] = {"per_model": [], "passed": True}

    for model_id in spec["required_for_each_of"]:
        pooled = assessment_metrics(
            daily["labels"], daily["predictions"][model_id], daily["users"]
        )
        log_loss_days = sum(
            1
            for origin_name in daily["daily_origins"]
            if _daily_value(daily["daily_rows"], origin_name, model_id, "log_loss")
            <= _daily_value(daily["daily_rows"], origin_name, "BL0", "log_loss") + tolerance
        )
        brier_days = sum(
            1
            for origin_name in daily["daily_origins"]
            if _daily_value(daily["daily_rows"], origin_name, model_id, "brier")
            <= _daily_value(daily["daily_rows"], origin_name, "BL0", "brier") + tolerance
        )
        # Reported diagnostic only: the same day counts taken against the pooled
        # BL0 instead of each day's own BL0. Never a gate - a gate must have one
        # definition, or pass and fail both become defensible after the fact.
        pooled_baseline_log_loss_days = sum(
            1
            for origin_name in daily["daily_origins"]
            if _daily_value(daily["daily_rows"], origin_name, model_id, "log_loss")
            <= pooled_bl0["log_loss"] + tolerance
        )
        pooled_baseline_brier_days = sum(
            1
            for origin_name in daily["daily_origins"]
            if _daily_value(daily["daily_rows"], origin_name, model_id, "brier")
            <= pooled_bl0["brier"] + tolerance
        )
        row = {
            "model_id": model_id,
            "pooled_log_loss_minus_BL0": pooled["log_loss"] - pooled_bl0["log_loss"],
            "pooled_brier_minus_BL0": pooled["brier"] - pooled_bl0["brier"],
            "nonworse_log_loss_days": log_loss_days,
            "nonworse_brier_days": brier_days,
            "total_days": int(stability["total_days"]),
            "nonworse_baseline": "same_day_BL0",
            "diagnostic_pooled_baseline_log_loss_days": pooled_baseline_log_loss_days,
            "diagnostic_pooled_baseline_brier_days": pooled_baseline_brier_days,
            "diagnostic_pooled_baseline_role": "reported_not_a_gate",
        }
        row["passed"] = bool(
            row["pooled_log_loss_minus_BL0"] <= tolerance
            and row["pooled_brier_minus_BL0"] <= tolerance
            and log_loss_days >= stability["minimum_nonworse_log_loss_days"]
            and brier_days >= stability["minimum_nonworse_brier_days"]
        )
        results["per_model"].append(row)
        results["passed"] = results["passed"] and row["passed"]
    return results


def relative_history_gate(
    contract: dict[str, Any], daily: dict[str, Any], bootstrap_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Contract: ``relative_history_gate`` (frozen BL2 minus frozen BL1)."""

    spec = contract["relative_history_gate"]
    requirements = spec["daily_requirements"]
    pooled_bl1 = assessment_metrics(daily["labels"], daily["predictions"]["BL1"], daily["users"])
    pooled_bl2 = assessment_metrics(daily["labels"], daily["predictions"]["BL2"], daily["users"])
    positive_ap_days = sum(
        1
        for origin_name in daily["daily_origins"]
        if _daily_value(daily["daily_rows"], origin_name, "BL2", "average_precision")
        > _daily_value(daily["daily_rows"], origin_name, "BL1", "average_precision")
    )
    nonnegative_gauc_days = sum(
        1
        for origin_name in daily["daily_origins"]
        if _daily_value(daily["daily_rows"], origin_name, "BL2", "user_gauc_event_weighted")
        >= _daily_value(daily["daily_rows"], origin_name, "BL1", "user_gauc_event_weighted")
    )
    ci_matches = [
        float(row["CI_lower"])
        for row in bootstrap_rows
        if row["comparison_id"] == "BL2_minus_BL1"
        and row["metric"] == "average_precision"
    ]
    if len(ci_matches) != 1 or not np.isfinite(ci_matches[0]):
        raise ContractStop("paired BL2-minus-BL1 average-precision CI is missing or non-finite")
    ci_lower = ci_matches[0]
    result = {
        "pair_id": daily["selected"]["pair_id"],
        "delta_average_precision": pooled_bl2["average_precision"] - pooled_bl1["average_precision"],
        "delta_user_gauc_event_weighted": pooled_bl2["user_gauc_event_weighted"]
        - pooled_bl1["user_gauc_event_weighted"],
        "delta_log_loss": pooled_bl2["log_loss"] - pooled_bl1["log_loss"],
        "delta_brier": pooled_bl2["brier"] - pooled_bl1["brier"],
        "positive_average_precision_days": positive_ap_days,
        "nonnegative_user_gauc_days": nonnegative_gauc_days,
        "total_days": int(requirements["total_days"]),
        "average_precision_ci_lower": ci_lower,
    }
    result["passed"] = bool(
        result["delta_average_precision"] > 0
        and result["delta_user_gauc_event_weighted"] >= 0
        and result["delta_log_loss"] <= 0
        and result["delta_brier"] <= 0
        and positive_ap_days >= requirements["minimum_positive_average_precision_days"]
        and nonnegative_gauc_days >= requirements["minimum_nonnegative_user_gauc_days"]
        and np.isfinite(ci_lower)
        and ci_lower > 0
    )
    return result


def _slice_rows(
    scope: str,
    labels: np.ndarray,
    users: np.ndarray,
    prior_batch: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, mask in slice_masks(prior_batch):
        count = int(mask.sum())
        y = labels[mask]
        u = users[mask]
        base: dict[str, Any] = {
            "scope": scope,
            "slice": name,
            "rows": count,
            "users": int(np.unique(u).size),
            "positives": int(y.sum()),
            "prevalence": None if count == 0 else float(y.mean()),
            "gate_role": "descriptive_only",
        }
        if count == 0 or np.unique(y).size != 2:
            for model_id in ("BL0", "BL1", "BL2"):
                for metric_name in (
                    "average_precision",
                    "log_loss",
                    "brier",
                    "user_gauc_event_weighted",
                ):
                    base[f"{model_id}_metric_{metric_name}"] = None
            base["paired_delta_average_precision"] = None
            base["paired_delta_log_loss"] = None
            base["note"] = (
                "empty_slice_reported_for_coverage" if count == 0 else "single_class_slice"
            )
            rows.append(base)
            continue
        entry: dict[str, Any] = {
            **base,
            "note": "evaluated",
        }
        for model_id in ("BL0", "BL1", "BL2"):
            point = assessment_metrics(y, predictions[model_id][mask], u)
            entry[f"{model_id}_metric_average_precision"] = point["average_precision"]
            entry[f"{model_id}_metric_log_loss"] = point["log_loss"]
            entry[f"{model_id}_metric_brier"] = point["brier"]
            entry[f"{model_id}_metric_user_gauc_event_weighted"] = point[
                "user_gauc_event_weighted"
            ]
        entry["paired_delta_average_precision"] = (
            entry["BL2_metric_average_precision"] - entry["BL1_metric_average_precision"]
        )
        entry["paired_delta_log_loss"] = (
            entry["BL2_metric_log_loss"] - entry["BL1_metric_log_loss"]
        )
        rows.append(entry)
    return rows


def slice_metrics(daily: dict[str, Any]) -> list[dict[str, Any]]:
    """Contract: ``evaluation_slices`` on both declared scopes.

    Slices are ``descriptive_only``, so retaining the pooled and the per-origin
    view together costs nothing and forecloses nothing.  Every row carries a
    ``scope`` column; cross-slice raw average-precision comparison stays
    forbidden and is not produced here.
    """

    rows = _slice_rows(
        "pooled_seven_days",
        daily["labels"],
        daily["users"],
        daily["prior_batch"],
        daily["predictions"],
    )
    offset = 0
    for position, origin_name in enumerate(daily["daily_origins"]):
        size = int(daily["origin_sizes"][position])
        window = slice(offset, offset + size)
        rows.extend(
            _slice_rows(
                origin_name,
                daily["labels"][window],
                daily["users"][window],
                daily["prior_batch"][window],
                {name: values[window] for name, values in daily["predictions"].items()},
            )
        )
        offset += size
    if offset != daily["labels"].size:
        raise ContractStop("per-origin slice windows do not cover the pooled daily rows")
    return rows


def paired_bootstrap(
    contract: dict[str, Any], daily: dict[str, Any]
) -> list[dict[str, Any]]:
    """Contract: ``bootstrap`` plus the BL1-minus-BL0 discrimination diagnostic."""

    spec = contract["bootstrap"]
    universe = np.unique(daily["users"])
    # Fail closed on both counts. A user-count mismatch previously skipped the
    # digest check entirely, which silently voided the reproducibility pin.
    if universe.size != int(spec["expected_users"]):
        raise ContractStop(
            f"bootstrap user universe is {universe.size} but the contract declares "
            f"{spec['expected_users']}; the frozen multiplicity digest does not apply"
        )
    multiplicities, digest = metrics.make_multiplicities(
        user_count=universe.size,
        replicates=int(spec["replicates"]),
        seed=int(spec["seed"]),
    )
    if digest != spec["expected_multiplicity_matrix_sha256"]:
        raise ContractStop("bootstrap multiplicity digest mismatch")
    gauc = metrics.user_gauc_components(
        daily["labels"],
        daily["predictions"]["BL1"],
        daily["users"],
        user_universe=universe,
        epsilon=repair.METRIC_CLIP_LOW,
    )
    rows: list[dict[str, Any]] = []
    for contrast, baseline, candidate in (
        ("BL2_minus_BL1", "BL1", "BL2"),
        ("BL1_minus_BL0", "BL0", "BL1"),
    ):
        for summary in metrics.paired_user_cluster_bootstrap(
            daily["labels"],
            daily["predictions"][baseline],
            daily["predictions"][candidate],
            daily["users"],
            user_universe=universe,
            multiplicities=multiplicities,
            epsilon=repair.METRIC_CLIP_LOW,
            contrast=contrast,
        ):
            rows.append(
                {
                    "comparison_id": contrast,
                    "metric": summary["metric"],
                    "point_estimate": summary["point_estimate"],
                    "replicate_count": summary["effective_replicates"],
                    "seed": int(spec["seed"]),
                    "CI_lower": summary["ci95_lower"],
                    "CI_upper": summary["ci95_upper"],
                    "eligible_users": gauc.eligible_users,
                    "eligible_rows": gauc.eligible_rows,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def render_figure(daily_rows: list[dict[str, Any]], origins: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for axis, metric, title in zip(
        axes,
        ("average_precision", "log_loss", "brier"),
        ("Average precision", "Log loss", "Brier"),
    ):
        for model_id in ("BL0", "BL1", "BL2"):
            values = [_daily_value(daily_rows, origin, model_id, metric) for origin in origins]
            axis.plot(origins, values, marker="o", label=model_id)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.3)
    axes[0].legend()
    figure.suptitle("Gate 2B v003 frozen daily backtest (Train-only, developmental)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = sibling_temp_path(path)
    try:
        figure.savefig(temporary, dpi=140)
        plt.close(figure)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    register_produced_artifact(path)


def render_report(
    contract_hash: str,
    search: dict[str, Any],
    daily: dict[str, Any],
    probability_gate: dict[str, Any],
    history_gate: dict[str, Any],
    slices: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    selected = search["selected"]
    def optional_float(value: Any, format_spec: str = "+.6f") -> str:
        return "N/A" if value is None else format(float(value), format_spec)

    if not probability_gate["passed"]:
        state = "absolute_probability_fail"
    elif not history_gate["passed"]:
        state = "absolute_probability_pass_but_relative_history_fail"
    else:
        state = "pass"
    lines = [
        "# Gate 2B v003 probability repair results",
        "",
        f"> Contract SHA-256: `{contract_hash}`",
        "> Scope: post-hoc, Train-only, developmental repair. Not a preregistered confirmatory test.",
        "> All dates are source-Train dates; calibration uses the previous completed Train date.",
        "",
        f"**Terminal state: `{state}`**",
        "",
        "## Selected shared configuration",
        "",
        f"- `pair_id` = `{selected['pair_id']}` (alpha={selected['alpha']}, eta0={selected['eta0']})",
        "- Selected on BL1 probability quality only; BL2 deltas were not consulted.",
        "- BL1 and BL2 share this configuration, so the increment is not confounded with regularisation.",
        f"- Eligible BL1 configurations: {search['eligible_configuration_count']}/4.",
        "- Complete paired BL1/BL2 configurations available for optimizer-robustness "
        f"diagnostics: {search['paired_robustness_configuration_count']}/4.",
        "",
        "## Absolute probability quality gate (versus BL0)",
        "",
        "| model | pooled ΔLogLoss | pooled ΔBrier | non-worse LogLoss days | non-worse Brier days | passed |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in probability_gate["per_model"]:
        lines.append(
            f"| {row['model_id']} | {row['pooled_log_loss_minus_BL0']:+.6f} | "
            f"{row['pooled_brier_minus_BL0']:+.6f} | {row['nonworse_log_loss_days']}/{row['total_days']} | "
            f"{row['nonworse_brier_days']}/{row['total_days']} | {row['passed']} |"
        )
    lines += [
        "",
        "## Pooled uncalibrated versus calibrated probability diagnostics",
        "",
        "| model | raw-sigmoid LogLoss | calibrated LogLoss | raw-sigmoid Brier | calibrated Brier | raw-sigmoid ECE20 | calibrated ECE20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id in ("BL0", "BL1", "BL2"):
        lines.append(
            f"| {model_id} | "
            f"{_daily_value(daily['daily_rows'], 'pooled_7_days', model_id, 'uncalibrated_log_loss'):.6f} | "
            f"{_daily_value(daily['daily_rows'], 'pooled_7_days', model_id, 'log_loss'):.6f} | "
            f"{_daily_value(daily['daily_rows'], 'pooled_7_days', model_id, 'uncalibrated_brier'):.6f} | "
            f"{_daily_value(daily['daily_rows'], 'pooled_7_days', model_id, 'brier'):.6f} | "
            f"{_daily_value(daily['daily_rows'], 'pooled_7_days', model_id, 'uncalibrated_ECE20_equal_width'):.6f} | "
            f"{_daily_value(daily['daily_rows'], 'pooled_7_days', model_id, 'ECE20_equal_width'):.6f} |"
        )
    diagnostic_executed = sum(
        row["status"] == "complete" for row in daily["calibration_regression_rows"]
    )
    diagnostic_not_applicable = sum(
        row["status"] == "not_applicable" for row in daily["calibration_regression_rows"]
    )
    lines += [
        "",
        "- Assessment calibration regression accounting: 24 declared slots; "
        f"{diagnostic_executed} executed unpenalized fits; "
        f"{diagnostic_not_applicable} BL0 slots recorded N/A and not counted as fits.",
    ]
    lines += [
        "",
        "## Relative history gate (frozen BL2 minus frozen BL1, same configuration)",
        "",
        f"- ΔAverage precision: {history_gate['delta_average_precision']:+.6f}",
        f"- Δuser-GAUC (event weighted): {history_gate['delta_user_gauc_event_weighted']:+.6f}",
        f"- ΔLog loss: {history_gate['delta_log_loss']:+.6f}",
        f"- ΔBrier: {history_gate['delta_brier']:+.6f}",
        f"- Positive AP days: {history_gate['positive_average_precision_days']}/{history_gate['total_days']}",
        f"- Paired user-cluster bootstrap ΔAP 95% CI lower bound: {history_gate['average_precision_ci_lower']:+.6f}",
        f"- Passed: {history_gate['passed']}",
        "",
        "## BL1 minus BL0 discrimination diagnostic (no hard gate)",
        "",
    ]
    paired_count = int(search["paired_robustness_configuration_count"])
    if paired_count < 4:
        lines.append(
            "- Optimizer-configuration robustness is based on only "
            f"{paired_count} complete paired configuration(s), not all four."
        )
        for item in search["paired_robustness_ineligible_pairs"]:
            lines.append(
                f"  - `{item['pair_id']}` ineligible: `{'|'.join(item['reasons'])}`"
            )
    if paired_count == 1:
        lines.append(
            "- `cross_optimizer_configuration_robustness_cannot_be_assessed`: "
            "跨优化器配置的稳健性无法评估。"
        )
    for row in bootstrap_rows:
        if row["comparison_id"] == "BL1_minus_BL0" and row["metric"] == "average_precision":
            lines.append(
                f"- ΔAP = {row['point_estimate']:+.6f}, "
                f"95% CI [{row['CI_lower']:+.6f}, {row['CI_upper']:+.6f}]"
            )
            if float(row["CI_lower"]) > 0:
                lines.append(
                    "- Registered reading (case A): the static bundle demonstrates positive "
                    "marginal ranking signal beyond BL0."
                )
            elif float(row["CI_upper"]) < 0:
                lines.append(
                    "- Registered reading (case B2): the static bundle is distinguishably worse "
                    "than BL0 under this frozen fitting and calibration protocol; this is not the "
                    "same as an effect indistinguishable from zero."
                )
            else:
                lines.append(
                    "- Registered reading (case B): the static bundle has not demonstrated "
                    "discrimination beyond BL0. With the optimization adequacy gate passed this is "
                    "a finding about the feature set, not an implementation defect, and the "
                    "increment must not be described as sitting on top of an effective static "
                    "baseline."
                )
    lines += [
        "",
        "## Reference objective gaps and assessment regret diagnostics (selected pair)",
        "",
        "| phase | origin | model | reference source | objective regret | allowed regret | AP regret | ROC-AUC regret | user-GAUC regret | coefficient cosine | coefficient L2 ratio | passed |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in search["adequacy_rows"] + daily["adequacy_rows"]:
        if row["pair_id"] != selected["pair_id"]:
            continue
        lines.append(
            f"| {row['phase']} | {row['origin']} | {row['model_id']} | "
            f"{row['reference_fit_source']} | {row['objective_regret']:+.8f} | "
            f"{row['maximum_allowed_regret']:.8f} | "
            f"{float(row['assessment_AP_regret_diagnostic']):+.6f} | "
            f"{float(row['assessment_ROC_AUC_regret_diagnostic']):+.6f} | "
            f"{float(row['assessment_user_GAUC_regret_diagnostic']):+.6f} | "
            f"{optional_float(row['coefficient_cosine_similarity'], '.6f')} | "
            f"{optional_float(row['coefficient_L2_norm_ratio'], '.6f')} | "
            f"{row['adequacy_passed']} |"
        )
    lines += [
        "",
        "## Same-configuration sensitivity across all four pairs",
        "",
        "| pair | origin/scope | metric | BL2−BL1 | eligible | ineligible reason |",
        "|---|---|---|---:|---|---|",
    ]
    for row in search["pair_rows"]:
        if row.get("comparison_id") != "same_configuration_BL2_minus_BL1":
            continue
        delta = row["BL2_minus_BL1_delta"]
        delta_text = "N/A" if delta is None else f"{float(delta):+.6f}"
        lines.append(
            f"| {row['pair_id']} | {row['origin_or_pooled']} | {row['metric']} | "
            f"{delta_text} | {row['configuration_eligible']} | "
            f"{row.get('ineligible_reason', '')} |"
        )
    secondary = search.get("secondary_selected")
    lines += ["", "## Independently optimized BL2 engineering diagnostic", ""]
    if secondary is None:
        lines.append("- No absolute-safe independently optimized BL2 configuration was eligible.")
    else:
        lines.append(
            f"- Secondary BL2 pair: `{secondary['pair_id']}`; this comparison is engineering-only "
            "and cannot support or override the primary H2 gate."
        )
        for row in search["pair_rows"]:
            if (
                row.get("comparison_id")
                == "secondary_independently_optimized_BL2_vs_primary_BL1"
                and row["origin_or_pooled"] == "pooled_3_origins"
            ):
                lines.append(
                    f"- {row['metric']}: BL2−primary-BL1 = "
                    f"{float(row['BL2_minus_BL1_delta']):+.6f}"
                )
    lines += [
        "",
        "## Fixed lifetime/W10 history-rule diagnostics",
        "",
        "| stage | origin/scope | rule | metric | rule value | primary BL2 | rule−BL2 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in search["search_rule_rows"] + daily["rule_rows"]:
        scope = row["origin_or_pooled"]
        for field, label in (
            ("average_precision", "average precision"),
            ("roc_auc", "ROC-AUC"),
            ("user_gauc_event_weighted", "user-GAUC event-weighted"),
            ("user_gauc_user_equal", "user-GAUC user-equal"),
        ):
            lines.append(
                f"| {row['stage']} | {scope} | {row['diagnostic_id']} | {label} | "
                f"{row[field]:.6f} | {row[f'primary_BL2_{field}']:.6f} | "
                f"{row[f'diagnostic_minus_primary_BL2_{field}']:+.6f} |"
            )
    lines += ["", "## History-depth slices (descriptive only)", "",
              "| scope | slice | rows | prevalence | BL1 AP | BL2 AP | ΔAP |", "|---|---|---:|---:|---:|---:|---:|"]
    slice_data_rows = 0
    for row in slices:
        if row["BL1_metric_average_precision"] is None:
            lines.append(
                f"| {row['scope']} | {row['slice']} | {row['rows']} | N/A | N/A | N/A | N/A |"
            )
            continue
        slice_data_rows += 1
        lines.append(
            f"| {row['scope']} | {row['slice']} | {row['rows']} | {row['prevalence']:.4f} | "
            f"{row['BL1_metric_average_precision']:.6f} | "
            f"{row['BL2_metric_average_precision']:.6f} | "
            f"{row['paired_delta_average_precision']:+.6f} |"
        )
    if slice_data_rows == 0:
        raise ContractStop("report slice table has no data rows")
    lines += [
        "",
        "## Claim boundary",
        "",
        "- This contract was designed after the v002 metrics were observed.",
        "- Permitted: a post-hoc Train-only probability baseline repair under the v003 contract.",
        "- Forbidden: any preregistered success claim, independent confirmation, Validation "
        "generalisation, restricted-test or random performance claim, causal or business lift, "
        "Gold dataset claim, or sequence-model claim.",
        "",
    ]
    atomic_write_text(path, "\n".join(lines))


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def validate_only() -> int:
    contract, contract_hash = load_contract()
    log(f"contract {CONTRACT_PATH.name} sha256={contract_hash}")
    implementation = verify_implementation_hashes(contract)
    log(f"implementation hashes verified: {len(implementation)} files")
    inputs = verify_inputs(contract)
    log(f"input verified: {inputs[0]['path']} ({inputs[0]['size_bytes']} bytes)")
    environment = verify_environment(contract)
    log(f"environment matches release_versions: {environment}")
    reconciliation = verify_budget_arithmetic(contract)
    log(
        "budget reconciles: "
        f"{reconciliation['primary']} primary + {reconciliation['reference']} reference "
        f"+ {contract['operational_budget']['maximum_diagnostic_calibration_regression_fits']} "
        f"diagnostic = {reconciliation['total']}"
    )
    frame = Frame.read(PROJECT_ROOT / contract["input_allowlist"][0]["path"])
    population = verify_population(contract, frame)
    log(
        "population verified: "
        f"{population['rows']} rows / {population['users']} users / "
        f"{population['videos']} videos / {population['positives']} positives"
    )
    splits = build_splits(contract, frame)
    log(f"all {len(splits)} temporal splits match their frozen expected counts")
    declared = {str(item) for item in contract["required_outputs"]["artifacts"]}
    produced = set(ARTIFACT_NAMES)
    missing = sorted(declared - produced)
    if missing:
        raise ContractStop(f"runner does not produce declared artifacts: {missing}")
    log(f"runner covers all {len(declared)} declared artifacts")
    log(f"execution_authorized = {contract['authorization']['execution_authorized']}")
    log("validate-only complete; no fit was performed and no metric was read")
    return 0


ARTIFACT_NAMES = (
    "contract_snapshot.yaml",
    "input_verification.json",
    "target_row_manifests.csv",
    "temporal_split_manifest.csv",
    "preprocessing_manifest.json",
    "numeric_scaling_audit.csv",
    "categorical_frequency_audit.csv",
    "runtime_preflight.json",
    "search_artifact_freeze_manifest.json",
    "search_fixed_history_rule_diagnostics.csv",
    "search_trial_manifest.csv",
    "convergence_ledger.csv",
    "reference_solver_ledger.csv",
    "optimization_objective_audit.csv",
    "calibration_fit_ledger.csv",
    "usage_ledger.csv",
    "search_predictions_origin_2022-04-11.parquet",
    "search_predictions_origin_2022-04-14.parquet",
    "search_predictions_origin_2022-04-17.parquet",
    "search_metrics.csv",
    "paired_configuration_metrics.csv",
    "fixed_history_rule_diagnostics.csv",
    "selected_models.json",
    "selected_primary_shared_configuration.json",
    "daily_predictions.parquet",
    "daily_metrics.csv",
    "pooled_and_slice_metrics.csv",
    "probability_distribution_audit.csv",
    "calibration_bins.csv",
    "calibration_intercept_slope.csv",
    "paired_user_cluster_bootstrap.csv",
    "terminal_state.json",
    "run_manifest.json",
    "artifact_hash_manifest.json",
    "reports/analysis/gate2b_probability_repair_results_v003.md",
    "reports/figures/gate2b_probability_repair_results_v003.png",
)


def artifact_path(name: str) -> Path:
    if name == "reports/analysis/gate2b_probability_repair_results_v003.md":
        return REPORT_PATH
    if name == "reports/figures/gate2b_probability_repair_results_v003.png":
        return FIGURE_PATH
    return OUTPUT_DIR / name


def capture_managed_artifact_snapshot() -> dict[str, tuple[bool, int, int]]:
    """Capture pre-run fingerprints so a failure manifest cannot absorb stale files."""

    snapshot: dict[str, tuple[bool, int, int]] = {}
    for name in ARTIFACT_NAMES:
        path = artifact_path(name)
        if path.is_file():
            stat = path.stat()
            snapshot[name] = (True, int(stat.st_size), int(stat.st_mtime_ns))
        else:
            snapshot[name] = (False, 0, 0)
    return snapshot


def managed_artifact_changed(name: str) -> bool:
    path = artifact_path(name)
    if not path.is_file():
        return False
    stat = path.stat()
    current = (True, int(stat.st_size), int(stat.st_mtime_ns))
    return current != ACTIVE_ARTIFACT_SNAPSHOT.get(name, (False, 0, 0))


def artifact_entries(names: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in names:
        path = artifact_path(str(name))
        if not path.is_file():
            raise ContractStop(f"required artifact missing before finalization: {name}")
        size = path.stat().st_size
        if size <= 0:
            raise ContractStop(f"required artifact is empty before finalization: {name}")
        rows.append({"path": str(name), "size_bytes": size, "sha256": sha256_file(path)})
    return rows


def code_hashes() -> dict[str, str]:
    return {
        "runner": sha256_file(Path(__file__)),
        "repair_module": sha256_file(
            PROJECT_ROOT / "src/kuairand_longseq/models/gate2b_repair_v003.py"
        ),
        "metrics_module": sha256_file(
            PROJECT_ROOT / "src/kuairand_longseq/evaluation/gate2b_metrics.py"
        ),
    }


def threadpool_execution_summary(budget: "Budget | None") -> dict[str, Any]:
    rows = [] if budget is None else budget.ledger
    by_component: dict[str, int] = {}
    for row in rows:
        component = str(row.get("component_type", "unknown"))
        by_component[component] = max(
            by_component.get(component, 0), int(row.get("observed_threadpool_max", 0))
        )
    return {
        "controller": "threadpoolctl",
        "threadpoolctl_version": threadpoolctl.__version__,
        "fit_operations_observed": len(rows),
        "maximum_observed_threads": max(by_component.values(), default=0),
        "maximum_observed_threads_by_component": dict(sorted(by_component.items())),
        "all_recorded_operations_within_declared_limit": all(
            int(row.get("observed_threadpool_max", 0)) <= int(row.get("thread_limit", -1))
            for row in rows
        ),
    }


def collect_get_params_snapshots(
    book: "LedgerBook | None", daily: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Collect exact instantiated estimator snapshots for the run manifest."""

    groups: list[tuple[str, str, Iterable[Any]]] = []
    if book is not None:
        groups.extend(
            [
                (
                    "shared_SGD_estimator",
                    "sklearn.linear_model.SGDClassifier",
                    (row.get("get_params_snapshot") for row in book.convergence),
                ),
                (
                    "diagnostic_reference_solver",
                    "sklearn.linear_model.LogisticRegression",
                    (row.get("get_params_snapshot") for row in book.reference),
                ),
                (
                    "primary_previous_day_sigmoid_calibrator",
                    "sklearn.linear_model.LogisticRegression",
                    (row.get("get_params_snapshot") for row in book.calibration),
                ),
            ]
        )
    if daily is not None:
        assessment_values: Iterable[Any] = (
            row.get("get_params_snapshot")
            for row in daily.get("calibration_regression_rows", [])
        )
    else:
        assessment_values = sorted(ACTIVE_ASSESSMENT_GET_PARAMS)
    if daily is not None or ACTIVE_ASSESSMENT_GET_PARAMS:
        groups.append(
            (
                "assessment_calibration_regression",
                "sklearn.linear_model.LogisticRegression",
                assessment_values,
            )
        )

    snapshots: list[dict[str, Any]] = []
    for component, sklearn_class, values in groups:
        unique = sorted({str(value) for value in values if value not in (None, "")})
        for index, value in enumerate(unique, start=1):
            try:
                params = json.loads(value)
            except json.JSONDecodeError as error:
                raise ContractStop(
                    f"invalid strict get_params snapshot for {component}: {error}"
                ) from error
            snapshots.append(
                {
                    "component_id": f"{component}_{index:03d}",
                    "sklearn_class": sklearn_class,
                    "get_params": params,
                }
            )
    return snapshots


def validate_terminal_mapping(
    contract: dict[str, Any], state: str, exit_code: int, completion_scope: str
) -> None:
    registry = contract["terminal_state_registry"]
    if state not in registry:
        raise ContractStop(f"unregistered terminal state: {state}")
    declared = registry[state]
    if int(declared["exit_code"]) != int(exit_code):
        raise ContractStop(f"terminal exit code mismatch for {state}")
    if str(declared["completion_scope"]) != completion_scope:
        raise ContractStop(f"terminal completion scope mismatch for {state}")


def verify_daily_probability_rows(contract: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Fail closed on every concrete field represented by the contract shorthand."""

    required: list[str] = []
    for name in contract["probability_diagnostics"]["required_per_model_origin_and_pooled"]:
        if name == "minimum_p001_p01_p10_median_p90_p99_p999_maximum":
            required.extend(
                ["minimum", "p001", "p01", "p10", "median", "p90", "p99", "p999", "maximum"]
            )
        else:
            required.append(str(name))
    required.extend(
        [
            "uncalibrated_average_precision",
            "uncalibrated_log_loss",
            "uncalibrated_brier",
            "uncalibrated_ECE20_equal_width",
        ]
    )
    require_fields(rows, required, "daily probability diagnostics")
    for row in rows:
        model_id = str(row["model_id"])
        raw_fields = (
            "raw_score_average_precision_within_origin",
            "raw_score_user_gauc_event_weighted_within_origin",
            "raw_score_user_gauc_user_equal_within_origin",
        )
        if model_id == "BL0":
            if any(row[name] is not None for name in raw_fields):
                raise ContractStop("BL0 raw-score diagnostics must be recorded as not applicable/null")
        elif row["origin"] != "pooled_7_days":
            if any(row[name] is None or not np.isfinite(float(row[name])) for name in raw_fields):
                raise ContractStop(f"{model_id} within-origin raw-score diagnostics are incomplete")
        for name in (
            "minimum",
            "p001",
            "p01",
            "p10",
            "median",
            "p90",
            "p99",
            "p999",
            "maximum",
        ):
            if not np.isfinite(float(row[name])):
                raise ContractStop(f"daily probability diagnostic {name} is not finite")


def write_artifact_hash_manifest(
    contract: dict[str, Any], budget: "Budget | None" = None
) -> list[dict[str, Any]]:
    """Atomically commit the complete hash manifest as the final release write."""

    if ACTIVE_RUN_ID is None:
        raise ContractStop("full artifact finalization requires an active run_id")
    names = [
        str(name)
        for name in contract["required_outputs"]["artifacts"]
        if Path(str(name)).name != "artifact_hash_manifest.json"
    ]
    if len(names) != len(set(names)):
        raise ContractStop("required artifact paths are not unique")
    stale = [
        name
        for name in names
        if name not in ACTIVE_PRODUCED_ARTIFACTS or not managed_artifact_changed(name)
    ]
    if stale:
        raise ContractStop(
            "required artifacts were not created or replaced by this release attempt: "
            f"{stale}"
        )
    rows = artifact_entries(names)
    if budget is not None:
        budget.check_elapsed("after_first_full_artifact_hash_pass")
        budget.check_storage("after_first_full_artifact_hash_pass")
    # Verify every entry again before the atomic manifest write.  The write
    # itself is the completion commit; no failure-capable action follows it.
    for row in rows:
        path = artifact_path(row["path"])
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise ContractStop(f"artifact changed during final hash verification: {row['path']}")
    if budget is not None:
        budget.check_elapsed("after_second_full_artifact_hash_pass")
        budget.check_storage("after_second_full_artifact_hash_pass")
    write_json(
        OUTPUT_DIR / "artifact_hash_manifest.json",
        {
            "run_id": ACTIVE_RUN_ID,
            "completion_scope": "full",
            "status": "complete_all_required_artifacts_verified",
            "hash_algorithm": "SHA-256",
            "self_hash_status": "excluded_to_avoid_self_reference",
            "required_artifact_count": len(contract["required_outputs"]["artifacts"]),
            "artifact_count": len(rows),
            "missing_required_artifacts": [],
            "artifacts": rows,
        },
    )
    return rows


def write_failure_artifact_hash_manifest(state: str) -> list[dict[str, Any]]:
    """Hash only managed artifacts created or changed by this release attempt."""

    if ACTIVE_RUN_ID is None:
        raise ContractStop("failure artifact finalization requires an active run_id")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(
        name
        for name in ACTIVE_PRODUCED_ARTIFACTS
        if name in ARTIFACT_NAMES
        and name != "artifact_hash_manifest.json"
        and managed_artifact_changed(name)
        and artifact_path(name).is_file()
        and artifact_path(name).stat().st_size > 0
    )
    rows = artifact_entries(names)
    for row in rows:
        path = artifact_path(row["path"])
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise ContractStop(f"failure artifact changed during hash verification: {row['path']}")
    present_after_commit = set(names) | {"artifact_hash_manifest.json"}
    missing = sorted(set(ARTIFACT_NAMES) - present_after_commit)
    write_json(
        OUTPUT_DIR / "artifact_hash_manifest.json",
        {
            "run_id": ACTIVE_RUN_ID,
            "completion_scope": "partial_failure",
            "status": "partial_fail_closed_checkpoint",
            "terminal_state": state,
            "hash_algorithm": "SHA-256",
            "self_hash_status": "excluded_to_avoid_self_reference",
            "coverage": "current_run_produced_path_registry",
            "required_artifact_count": len(ARTIFACT_NAMES),
            "artifact_count": len(rows),
            "missing_required_artifacts": missing,
            "artifacts": rows,
        },
    )
    return rows


def _write_artifacts(
    contract: dict[str, Any],
    contract_hash: str,
    budget: Budget,
    book: LedgerBook,
    frame: Frame,
    origins: dict[str, OriginMatrices],
    preflight: dict[str, Any],
    search: dict[str, Any],
    daily: dict[str, Any],
    probability_gate: dict[str, Any],
    history_gate: dict[str, Any],
    slices: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
) -> None:
    schemas = contract["artifact_schemas"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    atomic_write_text(
        OUTPUT_DIR / "contract_snapshot.yaml", CONTRACT_PATH.read_text(encoding="utf-8")
    )
    write_json(OUTPUT_DIR / "input_verification.json", verify_inputs(contract))

    write_csv(
        OUTPUT_DIR / "target_row_manifests.csv",
        [
            {
                "origin": origin.split.origin,
                "group": group,
                "rows": int(index.size),
                "users": int(np.unique(frame.users()[index]).size),
                "positives": int(frame.labels()[index].sum()),
                "identity_sha256": identity_digest(
                    frame.columns["source_table"][index],
                    frame.columns["source_row_number"][index],
                ),
            }
            for origin in origins.values()
            for group, index in (
                ("fit", origin.split.fit_index),
                ("calibration", origin.split.calibration_index),
                ("assessment", origin.split.assessment_index),
            )
        ],
    )
    write_csv(
        OUTPUT_DIR / "temporal_split_manifest.csv",
        [
            {
                "origin": origin.split.origin,
                "calibration_date": origin.split.calibration_date,
                "fit_rows": int(origin.split.fit_index.size),
                "calibration_rows": int(origin.split.calibration_index.size),
                "assessment_rows": int(origin.split.assessment_index.size),
                "fit_prevalence": origin.split.fit_prevalence,
                "bl0_probability": origin.split.bl0_probability,
            }
            for origin in origins.values()
        ],
    )
    write_json(
        OUTPUT_DIR / "preprocessing_manifest.json",
        {
            "preprocessing_id": repair.PREPROCESSING_ID,
            "calibration_id": repair.CALIBRATION_ID,
            "continuous_clip": repair.CONTINUOUS_CLIP,
            "categorical_minimum_frequency": repair.CATEGORICAL_MIN_FREQUENCY,
            "categorical_maximum_categories": repair.CATEGORICAL_MAX_CATEGORIES,
            "per_origin": {
                name: {
                    "bl1_width": origin.design.bl1_width,
                    "bl2_width": origin.design.bl2_width,
                    **origin.design.diagnostics,
                }
                for name, origin in origins.items()
            },
            "sklearn_get_params_snapshots": {
                "preprocessing_by_origin": {
                    name: {
                        "one_hot_encoder": origin.design.encoder.get_params(deep=True),
                        "static_standard_scaler": origin.design.static_scaler.get_params(deep=True),
                        "h2_standard_scaler": origin.design.h2_scaler.get_params(deep=True),
                    }
                    for name, origin in origins.items()
                },
                "SGD_estimators": [
                    json.loads(value)
                    for value in sorted(
                        {row["get_params_snapshot"] for row in book.convergence}
                    )
                ],
                "reference_solvers": [
                    json.loads(value)
                    for value in sorted({row["get_params_snapshot"] for row in book.reference})
                ],
                "primary_calibrators": [
                    json.loads(value)
                    for value in sorted(
                        {
                            row["get_params_snapshot"]
                            for row in book.calibration
                            if row.get("get_params_snapshot") is not None
                        }
                    )
                ],
                "assessment_calibration_regressions": [
                    json.loads(value)
                    for value in sorted(
                        {
                            row["get_params_snapshot"]
                            for row in daily["calibration_regression_rows"]
                        }
                    )
                ],
            },
        },
    )
    write_csv(
        OUTPUT_DIR / "numeric_scaling_audit.csv",
        [row for origin in origins.values() for row in origin.scaling_audit],
    )
    write_csv(
        OUTPUT_DIR / "categorical_frequency_audit.csv",
        [row for origin in origins.values() for row in origin.categorical_audit],
    )
    write_json(OUTPUT_DIR / "runtime_preflight.json", preflight)
    require_fields([preflight], schemas["runtime_preflight_required_fields"], "runtime_preflight")

    write_csv(OUTPUT_DIR / "search_trial_manifest.csv", search["trial_rows"])
    require_fields(book.convergence, schemas["convergence_ledger_required_fields"], "convergence_ledger")
    write_csv(OUTPUT_DIR / "convergence_ledger.csv", book.convergence)
    require_fields(book.reference, schemas["reference_solver_ledger_required_fields"], "reference_solver_ledger")
    write_csv(OUTPUT_DIR / "reference_solver_ledger.csv", book.reference)
    objective_rows = search["adequacy_rows"] + daily["adequacy_rows"]
    require_fields(
        objective_rows,
        schemas["optimization_objective_audit_required_fields"],
        "optimization_objective_audit",
    )
    write_csv(OUTPUT_DIR / "optimization_objective_audit.csv", objective_rows)
    require_fields(book.calibration, schemas["calibration_fit_ledger_required_fields"], "calibration_fit_ledger")
    write_csv(OUTPUT_DIR / "calibration_fit_ledger.csv", book.calibration)
    write_csv(OUTPUT_DIR / "usage_ledger.csv", budget.ledger)

    write_csv(OUTPUT_DIR / "search_metrics.csv", search["metric_rows"])
    require_fields(
        search["pair_rows"], schemas["paired_configuration_metrics_required_fields"],
        "paired_configuration_metrics",
    )
    write_csv(OUTPUT_DIR / "paired_configuration_metrics.csv", search["pair_rows"])
    require_fields(
        search["search_rule_rows"] + daily["rule_rows"],
        schemas["fixed_history_rule_diagnostics_required_fields"],
        "fixed_history_rule_diagnostics",
    )
    write_csv(
        OUTPUT_DIR / "fixed_history_rule_diagnostics.csv",
        search["search_rule_rows"] + daily["rule_rows"],
    )

    write_json(
        OUTPUT_DIR / "selected_models.json",
        {
            "primary_shared_pair_id": search["selected"]["pair_id"],
            "BL1": {"bundle": BUNDLES["BL1"], **search["selected"]},
            "BL2": {"bundle": BUNDLES["BL2"], **search["selected"]},
            "eligible_configuration_count": search["eligible_configuration_count"],
            "paired_robustness_configuration_count": search[
                "paired_robustness_configuration_count"
            ],
            "paired_robustness_ineligible_pairs": search[
                "paired_robustness_ineligible_pairs"
            ],
            "secondary_BL2": search["secondary_selected"],
            "secondary_eligible_configuration_count": search[
                "secondary_eligible_configuration_count"
            ],
            "bl1_eligibility": search["bl1_eligibility_rows"],
            "bl2_primary_gate": search["bl2_gate"],
        },
    )
    write_json(OUTPUT_DIR / "selected_primary_shared_configuration.json", search["selected"])

    daily_prediction_path = OUTPUT_DIR / "daily_predictions.parquet"
    write_parquet(daily_prediction_path, pa.table(daily["daily_prediction_table"]))
    # Contract: probability_diagnostics.required_per_model_origin_and_pooled and
    # evaluation_slices.required_context are checked too, not only artifact_schemas.
    enriched_daily = [
        {**row, **_calibration_regression_lookup(daily, row)} for row in daily["daily_rows"]
    ]
    verify_daily_probability_rows(contract, enriched_daily)
    write_csv(OUTPUT_DIR / "daily_metrics.csv", enriched_daily)
    if len(slices) != int(contract["evaluation_slices"]["expected_rows"]):
        raise ContractStop(
            f"pooled_and_slice_metrics has {len(slices)} rows, expected "
            f"{contract['evaluation_slices']['expected_rows']}"
        )
    require_fields(
        slices,
        [
            "rows",
            "users",
            "positives",
            "prevalence",
            "scope",
            "slice",
            "BL0_metric_average_precision",
            "BL1_metric_average_precision",
            "BL2_metric_average_precision",
            "paired_delta_average_precision",
        ],
        "pooled_and_slice_metrics",
    )
    write_csv(OUTPUT_DIR / "pooled_and_slice_metrics.csv", slices)
    write_csv(OUTPUT_DIR / "probability_distribution_audit.csv", daily["distribution_rows"])

    calibration_bins: list[dict[str, Any]] = []
    for model_id in ("BL0", "BL1", "BL2"):
        _, bins = metrics.ece_equal_width(
            daily["labels"], daily["predictions"][model_id], bins=20,
            epsilon=repair.METRIC_CLIP_LOW,
        )
        calibration_bins.extend({"model_id": model_id, **row} for row in bins)
    write_csv(OUTPUT_DIR / "calibration_bins.csv", calibration_bins)
    # the assessment recalibration intercept and slope; the Platt calibrator fits
    # already have their own ledger in calibration_fit_ledger.csv
    write_csv(
        OUTPUT_DIR / "calibration_intercept_slope.csv", daily["calibration_regression_rows"]
    )
    require_fields(
        bootstrap_rows, schemas["paired_user_cluster_bootstrap_required_fields"],
        "paired_user_cluster_bootstrap",
    )
    write_csv(OUTPUT_DIR / "paired_user_cluster_bootstrap.csv", bootstrap_rows)

    render_figure(daily["daily_rows"], daily["daily_origins"], FIGURE_PATH)
    render_report(
        contract_hash, search, daily, probability_gate, history_gate, slices, bootstrap_rows,
        REPORT_PATH,
    )
    budget.stop_release_monitor()
    budget.check_elapsed("after_report_and_figure")
    budget.check_storage("after_report_and_figure")

    if not probability_gate["passed"]:
        terminal_state = "absolute_probability_fail"
        terminal_code = EXIT_ABSOLUTE_PROBABILITY_FAIL
    elif not history_gate["passed"]:
        terminal_state = "absolute_probability_pass_but_relative_history_fail"
        terminal_code = EXIT_RELATIVE_HISTORY_FAIL
    else:
        terminal_state = "pass"
        terminal_code = 0
    validate_terminal_mapping(contract, terminal_state, terminal_code, "full")
    write_terminal_state(
        state=terminal_state,
        exit_code=terminal_code,
        stage="final",
        message="final gates evaluated",
        contract_hash=contract_hash,
        completion_scope="full",
        budget=budget,
    )
    inventory_names = [
        str(name)
        for name in contract["required_outputs"]["artifacts"]
        if Path(str(name)).name not in {"run_manifest.json", "artifact_hash_manifest.json"}
    ]
    stale_inventory = [
        name
        for name in inventory_names
        if name not in ACTIVE_PRODUCED_ARTIFACTS or not managed_artifact_changed(name)
    ]
    if stale_inventory:
        raise ContractStop(
            "full run manifest inventory contains non-current artifacts: "
            f"{stale_inventory}"
        )
    artifact_inventory = artifact_entries(inventory_names)
    get_params_snapshots = collect_get_params_snapshots(book, daily)
    required_snapshot_groups = set(
        contract["artifact_schemas"]["get_params_snapshots"]["required_components"]
    )
    observed_snapshot_groups = {
        next(
            group
            for group in required_snapshot_groups
            if str(item["component_id"]).startswith(f"{group}_")
        )
        for item in get_params_snapshots
        if any(
            str(item["component_id"]).startswith(f"{group}_")
            for group in required_snapshot_groups
        )
    }
    if observed_snapshot_groups != required_snapshot_groups:
        raise ContractStop(
            "run manifest get_params snapshots are incomplete: "
            f"missing {sorted(required_snapshot_groups - observed_snapshot_groups)}"
        )

    def run_manifest_payload() -> dict[str, Any]:
        return {
            "run_id": ACTIVE_RUN_ID,
            "status": "complete_run_payload_final_hash_is_separate_authority",
            "contract_sha256": contract_hash,
            "elapsed_seconds": budget.elapsed_seconds,
            "fit_operations": sum(budget.counts.values()),
            "fit_operations_by_type": budget.counts,
            "fit_operations_by_stage": budget.stage_counts,
            "nonfitted_history_rule_evaluations": budget.nonfitted_rule_evaluations,
            "diagnostic_calibration_regression_declared_slots": contract[
                "probability_diagnostics"
            ]["assessment_calibration_regression"]["declared_slots"],
            "diagnostic_calibration_regression_executed_fits": budget.counts.get(
                "diagnostic", 0
            ),
            "peak_RAM_GB": budget.peak_ram_gb,
            "selected_pair_id": search["selected"]["pair_id"],
            "probability_quality_gate_passed": probability_gate["passed"],
            "relative_history_gate_passed": history_gate["passed"],
            "terminal_state": terminal_state,
            "exit_code": terminal_code,
            "completion_scope": "full",
            "complete_artifact_hash_manifest": True,
            "inherited_upstream_guarantees": contract["inherited_upstream_guarantees"],
            "environment": verify_environment(contract),
            "threadpool_execution": threadpool_execution_summary(budget),
            "code_sha256": code_hashes(),
            "get_params_snapshots": get_params_snapshots,
            "artifact_inventory": artifact_inventory,
            "artifact_inventory_exclusions": {
                "run_manifest.json": "self_reference",
                "artifact_hash_manifest.json": "written_after_run_manifest_and_covers_run_manifest",
            },
            "artifact_hash_manifest": "artifact_hash_manifest.json",
        }

    # Include the draft manifest in the final resource check. If this check
    # fails, the unified failure route rewrites terminal_state and emits a
    # partial hash manifest; no complete-state hash claim survives.
    write_json(OUTPUT_DIR / "run_manifest.json", run_manifest_payload())
    budget.check_elapsed("after_run_manifest_before_final_hash")
    budget.check_storage("after_run_manifest_before_final_hash")
    write_json(OUTPUT_DIR / "run_manifest.json", run_manifest_payload())
    write_artifact_hash_manifest(contract, budget)


def release(approved_hash: str) -> int:
    global ACTIVE_BUDGET, ACTIVE_BOOK, ACTIVE_RUN_STARTED, ACTIVE_ARTIFACT_SNAPSHOT
    global ACTIVE_RUN_ID, ACTIVE_PRODUCED_ARTIFACTS, ACTIVE_ASSESSMENT_GET_PARAMS
    ACTIVE_BUDGET = None
    ACTIVE_BOOK = None
    ACTIVE_RUN_STARTED = False
    ACTIVE_ARTIFACT_SNAPSHOT = {}
    ACTIVE_RUN_ID = None
    ACTIVE_PRODUCED_ARTIFACTS = set()
    ACTIVE_ASSESSMENT_GET_PARAMS = set()
    contract, contract_hash = load_contract()
    if contract_hash != approved_hash:
        raise ContractStop(
            "contract hash mismatch: refusing to run against an unapproved contract"
        )
    if not contract["authorization"]["execution_authorized"]:
        raise ContractStop(
            "authorization.execution_authorized is false; approval is a versioned edit to the "
            "contract, not a command-line flag"
        )
    verify_execution_preconditions(contract)
    verify_implementation_hashes(contract)
    verify_inputs(contract)
    verify_environment(contract)
    verify_budget_arithmetic(contract)

    budget = Budget(contract=contract)
    book = LedgerBook()
    ACTIVE_BUDGET = budget
    ACTIVE_BOOK = book
    ACTIVE_ARTIFACT_SNAPSHOT = capture_managed_artifact_snapshot()
    ACTIVE_RUN_ID = (
        f"gate2b-v003-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000:06d}"
    )
    ACTIVE_PRODUCED_ARTIFACTS = set()
    ACTIVE_ASSESSMENT_GET_PARAMS = set()
    ACTIVE_RUN_STARTED = True
    budget.start_release_monitor()
    frame = Frame.read(PROJECT_ROOT / contract["input_allowlist"][0]["path"])
    verify_population(contract, frame)
    splits = build_splits(contract, frame)
    budget.check_elapsed("after_input_population_and_split_verification")
    origins = {split.origin: build_origin_matrices(frame, split) for split in splits}
    budget.check_elapsed("after_origin_matrix_build")
    budget.check_storage("after_origin_matrix_build")
    log(f"built point-in-time designs for {len(origins)} origins")

    preflight, cached = runtime_preflight(contract, budget, book, origins)
    budget.check_elapsed("after_runtime_preflight")
    budget.check_storage("after_runtime_preflight")
    log(f"runtime preflight projects {preflight['projected_total_elapsed_minutes']:.1f} minutes")

    search = search_phase(contract, budget, book, frame, origins, cached)
    budget.check_elapsed("after_search_phase")
    budget.check_storage("after_search_phase")
    log(f"selected shared configuration {search['selected']['pair_id']}")

    daily = frozen_daily_backtest(contract, budget, book, frame, origins, search)
    budget.check_elapsed("after_frozen_daily_backtest")
    budget.check_storage("after_frozen_daily_backtest")
    budget.check_elapsed("before_paired_bootstrap")
    bootstrap_rows = paired_bootstrap(contract, daily)
    budget.check_elapsed("after_paired_bootstrap")
    probability_gate = probability_quality_gate(contract, origins, daily)
    history_gate = relative_history_gate(contract, daily, bootstrap_rows)
    slices = slice_metrics(daily)
    budget.check_elapsed("after_gates_and_slices")
    budget.check_storage("after_gates_and_slices")
    log(
        f"probability gate passed={probability_gate['passed']} "
        f"relative history gate passed={history_gate['passed']}"
    )

    _write_artifacts(
        contract, contract_hash, budget, book, frame, origins, preflight, search, daily,
        probability_gate, history_gate, slices, bootstrap_rows,
    )
    ACTIVE_RUN_STARTED = False
    log(f"release complete in {budget.elapsed_seconds/60:.1f} minutes")
    if not probability_gate["passed"]:
        log("STATE: absolute_probability_fail - stop at the baseline layer")
        return 3
    if not history_gate["passed"]:
        log("STATE: absolute_probability_pass_but_relative_history_fail")
        return 4
    log("STATE: pass - eligible to request separate Gold and next stage approval only")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--release", action="store_true")
    parser.add_argument(
        "--approve-contract-hash",
        default="",
        help="exact SHA-256 of the approved contract; required with --release",
    )
    return parser.parse_args()


def persist_failure_boundary(
    *, state: str, exit_code: int, stage: str, message: str, contract_hash: str
) -> None:
    """Persist the auditable partial state for an active release attempt."""

    if not ACTIVE_RUN_STARTED:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if CONTRACT_PATH.is_file():
        atomic_write_text(
            OUTPUT_DIR / "contract_snapshot.yaml", CONTRACT_PATH.read_text(encoding="utf-8")
        )
    budget = ACTIVE_BUDGET
    book = ACTIVE_BOOK
    if budget is not None:
        budget.stop_release_monitor()
    contract = yaml.load(CONTRACT_PATH.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    schemas = contract["artifact_schemas"]
    write_csv_or_header(
        OUTPUT_DIR / "usage_ledger.csv",
        [] if budget is None else budget.ledger,
        USAGE_LEDGER_FIELDS,
    )
    write_csv_or_header(
        OUTPUT_DIR / "convergence_ledger.csv",
        [] if book is None else book.convergence,
        schemas["convergence_ledger_required_fields"],
    )
    write_csv_or_header(
        OUTPUT_DIR / "reference_solver_ledger.csv",
        [] if book is None else book.reference,
        schemas["reference_solver_ledger_required_fields"],
    )
    write_csv_or_header(
        OUTPUT_DIR / "calibration_fit_ledger.csv",
        [] if book is None else book.calibration,
        schemas["calibration_fit_ledger_required_fields"],
    )
    validate_terminal_mapping(contract, state, exit_code, "partial_failure")
    write_terminal_state(
        state=state,
        exit_code=exit_code,
        stage=stage,
        message=message,
        contract_hash=contract_hash,
        completion_scope="partial_failure",
        budget=budget,
    )
    inventory_names = sorted(
        name
        for name in ACTIVE_PRODUCED_ARTIFACTS
        if name in ARTIFACT_NAMES
        and name not in {"run_manifest.json", "artifact_hash_manifest.json"}
        and managed_artifact_changed(name)
        and artifact_path(name).is_file()
        and artifact_path(name).stat().st_size > 0
    )
    artifact_inventory = artifact_entries(inventory_names)
    write_json(
        OUTPUT_DIR / "run_manifest.json",
        {
            "run_id": ACTIVE_RUN_ID,
            "status": "failed_during_or_before_finalization",
            "contract_sha256": contract_hash,
            "terminal_state": state,
            "exit_code": int(exit_code),
            "completion_scope": "partial_failure",
            "message": message,
            "fit_operations": 0 if budget is None else sum(budget.counts.values()),
            "fit_operations_by_type": {} if budget is None else dict(budget.counts),
            "fit_operations_by_stage": {} if budget is None else dict(budget.stage_counts),
            "inherited_upstream_guarantees": contract["inherited_upstream_guarantees"],
            "environment": verify_environment(contract),
            "threadpool_execution": threadpool_execution_summary(budget),
            "code_sha256": code_hashes(),
            "get_params_snapshots": collect_get_params_snapshots(book),
            "artifact_inventory": artifact_inventory,
            "artifact_hash_manifest": "artifact_hash_manifest.json",
        },
    )
    write_failure_artifact_hash_manifest(state)


def main() -> int:
    args = parse_args()
    try:
        if args.validate_only:
            return validate_only()
        if not args.approve_contract_hash:
            raise ContractStop("--release requires --approve-contract-hash")
        return release(args.approve_contract_hash)
    except TerminalStop as stop:
        contract_hash = sha256_file(CONTRACT_PATH) if CONTRACT_PATH.is_file() else "unavailable"
        persist_failure_boundary(
            state=stop.state,
            exit_code=stop.exit_code,
            stage=stop.stage,
            message=str(stop),
            contract_hash=contract_hash,
        )
        log(f"STATE {stop.state}: {stop}")
        return stop.exit_code
    except ContractStop as stop:
        contract_hash = sha256_file(CONTRACT_PATH) if CONTRACT_PATH.is_file() else "unavailable"
        persist_failure_boundary(
            state="implementation_or_governance_failure",
            exit_code=EXIT_IMPLEMENTATION_OR_GOVERNANCE_FAILURE,
            stage="runtime",
            message=str(stop),
            contract_hash=contract_hash,
        )
        log(f"STOP: {stop}")
        return EXIT_IMPLEMENTATION_OR_GOVERNANCE_FAILURE
    except Exception as error:  # every implementation defect uses the same fail-closed route
        contract_hash = sha256_file(CONTRACT_PATH) if CONTRACT_PATH.is_file() else "unavailable"
        message = f"{type(error).__name__}: {error}"
        persist_failure_boundary(
            state="implementation_or_governance_failure",
            exit_code=EXIT_IMPLEMENTATION_OR_GOVERNANCE_FAILURE,
            stage="runtime",
            message=message,
            contract_hash=contract_hash,
        )
        log(f"UNEXPECTED STOP: {message}")
        return EXIT_IMPLEMENTATION_OR_GOVERNANCE_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
