"""Gate 2B v003 probability-repair primitives.

This module implements exactly what
``configs/gate2b_probability_repair_contract_v003.yaml`` declares and nothing
more.  It never discovers inputs, never reads Silver or quarantine, and never
selects a model: orchestration, ordering and gate decisions belong to the
runner.

Contract sections implemented here:

* ``preprocessing``           -> :data:`PREPROCESSING_ID` / :func:`fit_grouped_design`
* ``models.shared_sparse_linear_estimator`` -> :func:`fit_sgd`
* ``models.diagnostic_reference_solver``    -> :func:`fit_reference`
* ``optimization_adequacy``   -> :func:`regularized_objective` / :func:`adequacy_decision`
* ``calibration``             -> :data:`CALIBRATION_ID` / :func:`fit_previous_day_sigmoid`
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PREPROCESSING_ID = "PIT_GROUPED_SCALE_V2"
CALIBRATION_ID = "PREVIOUS_DAY_SIGMOID_V1"

SEED = 20260814

# --- preprocessing constants (contract: preprocessing) -----------------------
CATEGORICAL_MIN_FREQUENCY = 20
CATEGORICAL_MAX_CATEGORIES = 4096
CONTINUOUS_CLIP = 10.0
MAX_ROW_L2_NORM = 50.0
MAX_P99_ROW_L2_NORM = 25.0

# --- estimator constants (contract: models.shared_sparse_linear_estimator) ---
SGD_ALPHA_VALUES = (1e-4, 1e-3)
SGD_ETA0_VALUES = (1e-3, 1e-2)
SGD_MAX_ITER = 100
SGD_TOL = 1e-4
SGD_N_ITER_NO_CHANGE = 5

# --- reference solver (contract: models.diagnostic_reference_solver) ---------
REFERENCE_MAX_ITER = 500
REFERENCE_TOL = 1e-8

# --- adequacy gate (contract: optimization_adequacy) ------------------------
REGRET_ABSOLUTE_FLOOR = 1e-4
REGRET_RELATIVE_FRACTION = 0.005
REFERENCE_ABOVE_SGD_TOLERANCE = 1e-10

# --- calibration (contract: calibration) ------------------------------------
CALIBRATOR_C = 1.0
CALIBRATOR_MAX_ITER = 1000
CALIBRATOR_TOL = 1e-10
CALIBRATION_AP_TOLERANCE = 1e-10

# Metric-only numerical clip.  The contract is explicit that this is numerical
# safety and never a model repair, so it is applied at metric time only and the
# stored calibrated probability is never clipped.
METRIC_CLIP_LOW = 1e-15
METRIC_CLIP_HIGH = 1.0 - 1e-15


CATEGORICAL_FIELDS = (
    "cat_user",
    "cat_video",
    "cat_author",
    "cat_music",
    "cat_video_type",
    "cat_upload_type",
    "cat_music_type",
    "cat_tag_combo",
    "cat_duration_bucket",
)

STATIC_CONTINUOUS_FIELDS = (
    "static_log_duration",
    "static_log_upload_age",
    "static_log_width",
    "static_log_height",
    "static_aspect",
)

STATIC_BINARY_FIELDS = (
    "static_duration_valid",
    "static_upload_age_valid",
    "static_upload_future",
    "static_geometry_valid",
    "static_tag_missing",
)

H2_CONTINUOUS_FIELDS = (
    "log1p_prior_batch_count",
    "log1p_prior_event_count",
    "log1p_prior_positive_count",
    "smoothed_lifetime_long_view_rate",
    "log1p_last_user_gap_seconds",
    "log1p_w10_event_count",
    "log1p_w10_positive_count",
    "smoothed_w10_long_view_rate",
    "log1p_w50_event_count",
    "log1p_w50_positive_count",
    "smoothed_w50_long_view_rate",
    "log1p_w200_event_count",
    "log1p_w200_positive_count",
    "smoothed_w200_long_view_rate",
)

H2_BINARY_FIELDS = (
    "has_history",
    "w10_full_window_mask",
    "w50_full_window_mask",
    "w200_full_window_mask",
)

H2_SMOOTHING_PRIOR_STRENGTH = 20.0
H2_WINDOWS = (10, 50, 200)


class ContractViolation(RuntimeError):
    """Raised whenever a frozen contract invariant fails."""


# ---------------------------------------------------------------------------
# H2 derivation (contract: feature_semantics.H2_derivation)
# ---------------------------------------------------------------------------


def derive_h2_blocks(
    raw: dict[str, np.ndarray], prevalence: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(continuous_block, binary_block)`` in declared field order.

    ``prevalence`` is the estimator-fit target prevalence and is the only
    smoothing prior permitted by the contract.  Calibration and assessment
    labels never influence it.
    """

    def column(name: str) -> np.ndarray:
        return np.asarray(raw[name], dtype=np.float64)

    prior_batch = column("prior_batch_n")
    prior_event = column("prior_event_n")
    prior_positive = column("prior_positive_n")
    gap = column("last_user_gap_s")

    def log1p_nonnegative(values: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(values, 0.0))

    def smoothed(positive: np.ndarray, total: np.ndarray) -> np.ndarray:
        prior = H2_SMOOTHING_PRIOR_STRENGTH
        return (positive + prior * prevalence) / (total + prior)

    continuous = [
        log1p_nonnegative(prior_batch),
        log1p_nonnegative(prior_event),
        log1p_nonnegative(prior_positive),
        smoothed(prior_positive, prior_event),
        log1p_nonnegative(gap),
    ]
    binary = [(prior_event > 0).astype(np.float64)]
    for window in H2_WINDOWS:
        event_n = column(f"w{window}_event_n")
        positive_n = column(f"w{window}_positive_n")
        continuous.extend(
            [
                log1p_nonnegative(event_n),
                log1p_nonnegative(positive_n),
                smoothed(positive_n, event_n),
            ]
        )
        binary.append((prior_batch >= window).astype(np.float64))

    continuous_block = np.column_stack(continuous)
    binary_block = np.column_stack(binary)
    if continuous_block.shape[1] != len(H2_CONTINUOUS_FIELDS):
        raise ContractViolation("H2 continuous width does not match the declared field list")
    if binary_block.shape[1] != len(H2_BINARY_FIELDS):
        raise ContractViolation("H2 binary width does not match the declared field list")
    return continuous_block, binary_block


# ---------------------------------------------------------------------------
# Grouped preprocessing (contract: preprocessing / PIT_GROUPED_SCALE_V2)
# ---------------------------------------------------------------------------


@dataclass
class GroupedDesign:
    """Fitted on estimator-fit rows only; calibration/assessment transform only."""

    encoder: OneHotEncoder
    static_scaler: StandardScaler
    h2_scaler: StandardScaler
    prevalence: float
    categorical_width: int
    static_continuous_width: int
    static_binary_width: int
    h2_continuous_width: int
    h2_binary_width: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def bl1_width(self) -> int:
        return self.categorical_width + self.static_continuous_width + self.static_binary_width

    @property
    def bl2_width(self) -> int:
        return self.bl1_width + self.h2_continuous_width + self.h2_binary_width


def _check_binary_domain(block: np.ndarray, label: str) -> None:
    if block.size and not np.isin(block, (0.0, 1.0)).all():
        raise ContractViolation(f"{label} contains values outside the binary domain")


def _check_finite(block: np.ndarray, label: str) -> None:
    if not np.isfinite(block).all():
        raise ContractViolation(f"{label} contains non-finite values")


def _scale_and_clip(scaler: StandardScaler, block: np.ndarray, *, fit: bool) -> np.ndarray:
    values = np.asarray(block, dtype=np.float64)
    transformed = scaler.fit_transform(values) if fit else scaler.transform(values)
    clipped = np.clip(transformed, -CONTINUOUS_CLIP, CONTINUOUS_CLIP)
    return np.asarray(clipped, dtype=np.float32)


def _distribution(prefix: str, values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_minimum": float(np.min(array)),
        f"{prefix}_p01": float(np.quantile(array, 0.01)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p99": float(np.quantile(array, 0.99)),
        f"{prefix}_maximum": float(np.max(array)),
    }


def numeric_field_audit(
    design: "GroupedDesign",
    *,
    split: str,
    static_continuous: np.ndarray,
    h2_continuous: np.ndarray,
) -> list[dict[str, Any]]:
    """Return every contract-declared per-field scaling diagnostic."""

    rows: list[dict[str, Any]] = []
    for group, names, raw, scaler in (
        ("static_continuous", STATIC_CONTINUOUS_FIELDS, static_continuous, design.static_scaler),
        ("H2_continuous_or_rate", H2_CONTINUOUS_FIELDS, h2_continuous, design.h2_scaler),
    ):
        raw64 = np.asarray(raw, dtype=np.float64)
        transformed = np.clip(
            scaler.transform(raw64), -CONTINUOUS_CLIP, CONTINUOUS_CLIP
        ).astype(np.float32)
        for index, name in enumerate(names):
            rows.append(
                {
                    "split": split,
                    "feature_group": group,
                    "field": name,
                    **_distribution("raw", raw64[:, index]),
                    **_distribution("transformed", transformed[:, index]),
                    "scaler_mean": float(scaler.mean_[index]),
                    "scaler_scale": float(scaler.scale_[index]),
                    "zero_variance": bool(float(scaler.var_[index]) == 0.0),
                }
            )
    return rows


def categorical_frequency_audit(
    design: "GroupedDesign", *, split: str, categorical: np.ndarray
) -> list[dict[str, Any]]:
    """Audit observed/dedicated/infrequent/unknown category coverage per field."""

    values = np.asarray(categorical, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    infrequent = getattr(design.encoder, "infrequent_categories_", None)
    for index, name in enumerate(CATEGORICAL_FIELDS):
        observed_values = np.asarray(design.encoder.categories_[index])
        infrequent_values = (
            np.asarray(infrequent[index])
            if infrequent is not None and infrequent[index] is not None
            else np.asarray([], dtype=observed_values.dtype)
        )
        column = values[:, index]
        unknown_mask = ~np.isin(column, observed_values)
        infrequent_mask = np.isin(column, infrequent_values)
        dedicated_mask = ~(unknown_mask | infrequent_mask)
        rows.append(
            {
                "split": split,
                "field": name,
                "rows": int(column.size),
                "observed_category_count": int(observed_values.size),
                "dedicated_category_count": int(observed_values.size - infrequent_values.size),
                "infrequent_category_count": int(infrequent_values.size),
                "infrequent_event_share": float(np.mean(infrequent_mask)),
                "dedicated_event_coverage": float(np.mean(dedicated_mask)),
                "unknown_event_share": float(np.mean(unknown_mask)),
            }
        )
    return rows


def fit_grouped_design(
    *,
    categorical: np.ndarray,
    static_continuous: np.ndarray,
    static_binary: np.ndarray,
    h2_continuous: np.ndarray,
    h2_binary: np.ndarray,
    prevalence: float,
) -> tuple[GroupedDesign, sparse.csr_matrix, sparse.csr_matrix]:
    """Fit the design on estimator-fit rows and return ``(design, bl1, bl2)``."""

    _check_finite(static_continuous, "static continuous fit block")
    _check_finite(h2_continuous, "H2 continuous fit block")
    _check_binary_domain(static_binary, "static binary fit block")
    _check_binary_domain(h2_binary, "H2 binary fit block")

    encoder = OneHotEncoder(
        categories="auto",
        drop=None,
        handle_unknown="infrequent_if_exist",
        min_frequency=CATEGORICAL_MIN_FREQUENCY,
        max_categories=CATEGORICAL_MAX_CATEGORIES,
        sparse_output=True,
        dtype=np.float32,
        feature_name_combiner="concat",
    )
    encoded = encoder.fit_transform(np.asarray(categorical, dtype=np.int64)).tocsr()

    static_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    h2_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    static_num = _scale_and_clip(static_scaler, static_continuous, fit=True)
    h2_num = _scale_and_clip(h2_scaler, h2_continuous, fit=True)

    design = GroupedDesign(
        encoder=encoder,
        static_scaler=static_scaler,
        h2_scaler=h2_scaler,
        prevalence=float(prevalence),
        categorical_width=encoded.shape[1],
        static_continuous_width=static_num.shape[1],
        static_binary_width=np.asarray(static_binary).shape[1],
        h2_continuous_width=h2_num.shape[1],
        h2_binary_width=np.asarray(h2_binary).shape[1],
    )
    bl1, bl2 = _assemble(
        encoded=encoded,
        static_num=static_num,
        static_binary=static_binary,
        h2_num=h2_num,
        h2_binary=h2_binary,
        design=design,
    )
    infrequent = getattr(encoder, "infrequent_categories_", None)
    infrequent_counts = [
        0 if infrequent is None or values is None else int(len(values))
        for values in ([None] * len(encoder.categories_) if infrequent is None else infrequent)
    ]
    design.diagnostics = {
        "categorical_output_width": design.categorical_width,
        "infrequent_categories_present": any(count > 0 for count in infrequent_counts),
        "observed_categories_per_field": [int(len(cats)) for cats in encoder.categories_],
        "dedicated_categories_per_field": [
            int(len(cats) - infrequent_count)
            for cats, infrequent_count in zip(encoder.categories_, infrequent_counts)
        ],
        "infrequent_categories_per_field": infrequent_counts,
        "static_scaler_zero_variance_columns": int(
            np.count_nonzero(static_scaler.var_ == 0.0)
        ),
        "h2_scaler_zero_variance_columns": int(np.count_nonzero(h2_scaler.var_ == 0.0)),
    }
    return design, bl1, bl2


def transform_grouped(
    design: GroupedDesign,
    *,
    categorical: np.ndarray,
    static_continuous: np.ndarray,
    static_binary: np.ndarray,
    h2_continuous: np.ndarray,
    h2_binary: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Transform-only path for calibration and assessment rows."""

    _check_finite(static_continuous, "static continuous transform block")
    _check_finite(h2_continuous, "H2 continuous transform block")
    _check_binary_domain(static_binary, "static binary transform block")
    _check_binary_domain(h2_binary, "H2 binary transform block")

    encoded = design.encoder.transform(np.asarray(categorical, dtype=np.int64)).tocsr()
    static_num = _scale_and_clip(design.static_scaler, static_continuous, fit=False)
    h2_num = _scale_and_clip(design.h2_scaler, h2_continuous, fit=False)
    return _assemble(
        encoded=encoded,
        static_num=static_num,
        static_binary=static_binary,
        h2_num=h2_num,
        h2_binary=h2_binary,
        design=design,
    )


def _assemble(
    *,
    encoded: sparse.csr_matrix,
    static_num: np.ndarray,
    static_binary: np.ndarray,
    h2_num: np.ndarray,
    h2_binary: np.ndarray,
    design: GroupedDesign,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    static_bin = np.asarray(static_binary, dtype=np.float32)
    h2_bin = np.asarray(h2_binary, dtype=np.float32)
    bl1 = sparse.hstack(
        [encoded, sparse.csr_matrix(static_num), sparse.csr_matrix(static_bin)],
        format="csr",
        dtype=np.float32,
    )
    bl2 = sparse.hstack(
        [bl1, sparse.csr_matrix(h2_num), sparse.csr_matrix(h2_bin)],
        format="csr",
        dtype=np.float32,
    )
    if bl1.shape[1] != design.bl1_width or bl2.shape[1] != design.bl2_width:
        raise ContractViolation("assembled matrix width does not match the fitted design")
    return bl1.tocsr(), bl2.tocsr()


def assert_column_prefix(bl1: sparse.csr_matrix, bl2: sparse.csr_matrix) -> None:
    """Contract: ``BL1_static_matrix_must_be_an_exact_column_prefix_of_BL2``."""

    if bl1.shape[0] != bl2.shape[0] or bl1.shape[1] > bl2.shape[1]:
        raise ContractViolation("BL1 cannot be a column prefix of BL2")
    head = bl2.tocsc()[:, : bl1.shape[1]].tocsr()
    if (head != bl1.tocsr()).nnz != 0:
        raise ContractViolation("BL1 is not an exact column prefix of BL2")


def numeric_hard_checks(matrix: sparse.csr_matrix, label: str) -> dict[str, float | int]:
    """Contract: ``preprocessing.hard_numeric_checks_per_origin_and_bundle``."""

    data = matrix.data
    if data.size and not np.isfinite(data).all():
        raise ContractViolation(f"{label} contains non-finite values")
    absolute_max = float(np.abs(data).max()) if data.size else 0.0
    if absolute_max > CONTINUOUS_CLIP:
        raise ContractViolation(
            f"{label} exceeds the transformed continuous absolute maximum"
        )
    squared = matrix.multiply(matrix)
    row_norm = np.sqrt(np.asarray(squared.sum(axis=1)).ravel())
    maximum = float(row_norm.max()) if row_norm.size else 0.0
    p99 = float(np.quantile(row_norm, 0.99)) if row_norm.size else 0.0
    if maximum > MAX_ROW_L2_NORM:
        raise ContractViolation(f"{label} maximum row L2 norm {maximum} exceeds the cap")
    if p99 > MAX_P99_ROW_L2_NORM:
        raise ContractViolation(f"{label} p99 row L2 norm {p99} exceeds the cap")
    return {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "nonzeros": int(matrix.nnz),
        "density": float(matrix.nnz / max(matrix.shape[0] * matrix.shape[1], 1)),
        "absolute_maximum": absolute_max,
        "row_l2_p50": float(np.quantile(row_norm, 0.50)) if row_norm.size else 0.0,
        "row_l2_p90": float(np.quantile(row_norm, 0.90)) if row_norm.size else 0.0,
        "row_l2_p99": p99,
        "row_l2_p999": float(np.quantile(row_norm, 0.999)) if row_norm.size else 0.0,
        "row_l2_max": maximum,
    }


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


@dataclass
class FitRecord:
    converged: bool
    n_iter: int
    max_iter: int
    convergence_warning_count: int
    coefficient_l1_norm: float
    coefficient_l2_norm: float
    coefficient_absolute_maximum: float
    intercept: float
    objective: float
    # Populated only for the reference solver, and only so the runner can report
    # the descriptive assessment regret diagnostics. The reference solver is never
    # a prediction candidate and never enters selection.
    assessment_raw_score: np.ndarray | None = None
    # Retained only for descriptive SGD-vs-reference coefficient diagnostics.
    # It is never serialized into prediction artifacts or used for eligibility.
    coefficient: np.ndarray | None = field(default=None, repr=False)

    def as_row(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "n_iter": self.n_iter,
            "max_iter": self.max_iter,
            "convergence_warning_count": self.convergence_warning_count,
            "coefficient_l1_norm": self.coefficient_l1_norm,
            "coefficient_l2_norm": self.coefficient_l2_norm,
            "coefficient_absolute_maximum": self.coefficient_absolute_maximum,
            "intercept": self.intercept,
            "regularized_training_objective": self.objective,
        }


def _coefficients(model: Any) -> tuple[np.ndarray, float]:
    coefficient = np.asarray(model.coef_, dtype=np.float64).ravel()
    intercept = float(np.asarray(model.intercept_, dtype=np.float64).ravel()[0])
    if not np.isfinite(coefficient).all() or not np.isfinite(intercept):
        raise ContractViolation("non-finite coefficients or intercept")
    return coefficient, intercept


def regularized_objective(
    x: sparse.csr_matrix, y: np.ndarray, coefficient: np.ndarray, intercept: float, alpha: float
) -> float:
    """Mean binary log loss from the raw score plus ``alpha/2 * ||w||^2``.

    The intercept is not regularized, matching both ``SGDClassifier`` and
    ``LogisticRegression``.  Evaluated in float64 on estimator-fit rows only.
    """

    labels = np.asarray(y, dtype=np.float64)
    score = np.asarray(x @ np.asarray(coefficient, dtype=np.float64), dtype=np.float64) + float(
        intercept
    )
    signed = (2.0 * labels - 1.0) * score
    mean_loss = float(np.mean(np.logaddexp(0.0, -signed), dtype=np.float64))
    penalty = float(alpha) * 0.5 * float(np.dot(coefficient, coefficient))
    return mean_loss + penalty


def _fit_record(
    model: Any,
    x: sparse.csr_matrix,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    warning_count: int,
) -> FitRecord:
    coefficient, intercept = _coefficients(model)
    n_iter = int(np.asarray(model.n_iter_).ravel()[0])
    return FitRecord(
        converged=warning_count == 0 and n_iter < max_iter,
        n_iter=n_iter,
        max_iter=int(max_iter),
        convergence_warning_count=int(warning_count),
        coefficient_l1_norm=float(np.abs(coefficient).sum()),
        coefficient_l2_norm=float(np.sqrt(np.dot(coefficient, coefficient))),
        coefficient_absolute_maximum=float(np.abs(coefficient).max()) if coefficient.size else 0.0,
        intercept=intercept,
        objective=regularized_objective(x, y, coefficient, intercept, alpha),
        coefficient=coefficient.copy(),
    )


def _count_convergence_warnings(caught: Iterable[warnings.WarningMessage]) -> int:
    return sum(1 for item in caught if issubclass(item.category, ConvergenceWarning))


def fit_sgd(
    x: sparse.csr_matrix,
    y: np.ndarray,
    *,
    alpha: float,
    eta0: float,
    seed: int = SEED,
    max_iter: int = SGD_MAX_ITER,
) -> tuple[SGDClassifier, FitRecord]:
    """Contract: ``models.shared_sparse_linear_estimator``.

    Every parameter the contract names is passed explicitly; no library default
    is inherited silently.  ``learning_rate='adaptive'`` with an explicit
    ``eta0`` is the v003 repair of the v002 ``optimal`` schedule, which coupled
    the step size to ``alpha``.
    """

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(alpha),
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=int(max_iter),
        tol=SGD_TOL,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        n_jobs=1,
        random_state=int(seed),
        learning_rate="adaptive",
        eta0=float(eta0),
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=SGD_N_ITER_NO_CHANGE,
        class_weight=None,
        warm_start=False,
        average=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, np.asarray(y, dtype=np.int8))
    record = _fit_record(model, x, y, alpha, max_iter, _count_convergence_warnings(caught))
    return model, record


def reference_C(estimator_fit_rows: int, alpha: float) -> float:
    """``C = 1 / (estimator_fit_rows * alpha)``.

    Derivation: ``SGDClassifier`` minimises ``mean_loss + alpha/2 * ||w||^2``
    while ``LogisticRegression`` minimises ``1/2 * ||w||^2 + C * sum_loss``,
    i.e. ``mean_loss + 1/(2*n*C) * ||w||^2`` after normalising by ``n``.
    Matching the two gives ``alpha = 1 / (n * C)``.
    """

    if estimator_fit_rows <= 0 or alpha <= 0:
        raise ContractViolation("reference C mapping requires positive rows and alpha")
    return 1.0 / (float(estimator_fit_rows) * float(alpha))


def fit_reference(
    x: sparse.csr_matrix,
    y: np.ndarray,
    *,
    alpha: float,
    max_iter: int = REFERENCE_MAX_ITER,
) -> tuple[LogisticRegression, FitRecord, float]:
    """Contract: ``models.diagnostic_reference_solver``.

    Not selectable, not a prediction candidate: it exists only to bound the
    distance between the SGD solution and the optimum of the same objective.
    """

    rows = int(x.shape[0])
    C = reference_C(rows, alpha)
    model = LogisticRegression(
        solver="lbfgs",
        l1_ratio=0.0,
        C=C,
        dual=False,
        tol=REFERENCE_TOL,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        max_iter=int(max_iter),
        verbose=0,
        warm_start=False,
        n_jobs=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x.astype(np.float64), np.asarray(y, dtype=np.int8))
    record = _fit_record(model, x, y, alpha, max_iter, _count_convergence_warnings(caught))
    return model, record, C


def maximum_allowed_regret(reference_objective: float) -> float:
    return max(
        REGRET_ABSOLUTE_FLOOR, REGRET_RELATIVE_FRACTION * abs(float(reference_objective))
    )


def adequacy_decision(
    sgd_objective: float, reference_objective: float, *, reference_converged: bool
) -> dict[str, Any]:
    """Contract: ``optimization_adequacy``.

    The reference must converge first and must not sit above the SGD objective
    beyond numerical tolerance; otherwise the comparison is meaningless and the
    configuration is not adequate.
    """

    regret = float(sgd_objective) - float(reference_objective)
    allowed = maximum_allowed_regret(reference_objective)
    reference_above = regret < -REFERENCE_ABOVE_SGD_TOLERANCE
    passed = bool(reference_converged) and not reference_above and regret <= allowed
    return {
        "SGD_regularized_training_objective": float(sgd_objective),
        "reference_regularized_training_objective": float(reference_objective),
        "objective_regret": regret,
        "maximum_allowed_regret": allowed,
        "reference_converged": bool(reference_converged),
        "reference_above_SGD": reference_above,
        "adequacy_passed": passed,
    }


# ---------------------------------------------------------------------------
# Calibration (contract: calibration / PREVIOUS_DAY_SIGMOID_V1)
# ---------------------------------------------------------------------------


@dataclass
class Calibrator:
    model: LogisticRegression
    intercept: float
    slope: float
    n_iter: int
    convergence_warning_count: int
    fit_rows: int
    fit_users: int
    fit_positives: int
    fit_prevalence: float

    def apply(self, raw_score: np.ndarray) -> np.ndarray:
        score = np.asarray(raw_score, dtype=np.float64).reshape(-1, 1)
        probability = self.model.predict_proba(score)[:, 1].astype(np.float64, copy=False)
        if not np.isfinite(probability).all():
            raise ContractViolation("calibrated probability is not finite")
        return probability


def fit_previous_day_sigmoid(
    raw_score: np.ndarray,
    y: np.ndarray,
    *,
    user_id: np.ndarray | None = None,
    max_iter: int = CALIBRATOR_MAX_ITER,
) -> Calibrator:
    """Platt scaling on one raw decision score from the previous calendar day."""

    score = np.asarray(raw_score, dtype=np.float64).ravel()
    labels = np.asarray(y, dtype=np.int8).ravel()
    if score.shape != labels.shape or score.size == 0:
        raise ContractViolation("calibration inputs must align and be non-empty")
    if np.unique(labels).size != 2:
        raise ContractViolation("calibration requires both label classes")
    if not np.isfinite(score).all():
        raise ContractViolation("calibration raw scores must be finite")
    if float(np.var(score)) <= 0.0:
        raise ContractViolation("calibration raw score variance must be strictly positive")

    model = LogisticRegression(
        solver="lbfgs",
        l1_ratio=0.0,
        C=CALIBRATOR_C,
        dual=False,
        tol=CALIBRATOR_TOL,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        max_iter=int(max_iter),
        verbose=0,
        warm_start=False,
        n_jobs=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(score.reshape(-1, 1), labels)
    warning_count = _count_convergence_warnings(caught)
    other = sum(
        1
        for item in caught
        if issubclass(item.category, (FutureWarning, DeprecationWarning))
    )
    if other:
        raise ContractViolation("calibration emitted future or deprecation warnings")
    n_iter = int(np.asarray(model.n_iter_).ravel()[0])
    slope = float(np.asarray(model.coef_, dtype=np.float64).ravel()[0])
    intercept = float(np.asarray(model.intercept_, dtype=np.float64).ravel()[0])
    if warning_count or n_iter >= max_iter:
        raise ContractViolation("calibration did not converge inside the frozen cap")
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ContractViolation("calibration produced non-finite parameters")
    if slope <= 0.0:
        raise ContractViolation("calibration slope must be strictly positive")
    users = int(np.unique(user_id).size) if user_id is not None else 0
    positives = int(labels.sum())
    return Calibrator(
        model=model,
        intercept=intercept,
        slope=slope,
        n_iter=n_iter,
        convergence_warning_count=warning_count,
        fit_rows=int(labels.size),
        fit_users=users,
        fit_positives=positives,
        fit_prevalence=float(labels.mean()),
    )


def assert_calibration_monotone(
    raw_score: np.ndarray, calibrated: np.ndarray, *, tolerance: float = 1e-10
) -> None:
    """Contract: ``calibration.monotonicity_check``.

    A strictly positive Platt slope is order preserving, so any inversion means
    the calibrated column was not produced by the stored calibrator.
    """

    score = np.asarray(raw_score, dtype=np.float64)
    probability = np.asarray(calibrated, dtype=np.float64)
    if score.shape != probability.shape:
        raise ContractViolation("monotonicity check arrays must align")
    order = np.argsort(score, kind="mergesort")
    ordered = probability[order]
    if np.any(np.diff(ordered) < -tolerance):
        raise ContractViolation("calibrated probability order does not match raw score order")


def metric_clip(probability: np.ndarray) -> np.ndarray:
    """Numerical safety only; never applied to the stored probability column."""

    return np.clip(np.asarray(probability, dtype=np.float64), METRIC_CLIP_LOW, METRIC_CLIP_HIGH)


def _decimal_slug(value: float, prefix: str) -> str:
    exponent = int(round(-np.log10(float(value))))
    if not np.isclose(float(value), 10.0**-exponent, rtol=0, atol=0):
        raise ContractViolation(f"{prefix} value {value} is not an exact negative power of ten")
    return f"{prefix}1em{exponent:02d}"


def pair_id(alpha: float, eta0: float) -> str:
    return f"{_decimal_slug(alpha, 'A')}_{_decimal_slug(eta0, 'E')}"


def paired_configurations() -> list[dict[str, Any]]:
    """Contract: ``search_and_selection.paired_configuration_registry``.

    Order is frozen: alpha outer, eta0 inner, both in declared order.  The
    contract uses ascending ``pair_id`` as the final deterministic tie-break, so
    this list must stay reproducible.
    """

    return [
        {"pair_id": pair_id(alpha, eta0), "alpha": float(alpha), "eta0": float(eta0)}
        for alpha in SGD_ALPHA_VALUES
        for eta0 in SGD_ETA0_VALUES
    ]
