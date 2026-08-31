"""Metrics and paired user-cluster uncertainty for Gate 2B.

All functions operate on already frozen, aligned prediction rows.  They never
discover or read project data on their own.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


PROBABILITY_EPSILON = 1e-7
BOOTSTRAP_SEED = 20260814
BOOTSTRAP_REPLICATES = 2000
AP_BOOTSTRAP_BLOCK_SIZE = 8
EXPECTED_MULTIPLICITY_SHA256 = (
    "582d0ef006da0f61fb753fa6d15d6ee801f7fce5f820fc1c92a376268112d972"
)


def clipped(probability: np.ndarray, epsilon: float = PROBABILITY_EPSILON) -> np.ndarray:
    """Numerical-safety clip only.

    ``epsilon`` defaults to the v002 release value so previously published
    numbers stay bit-identical.  The v003 contract declares a much narrower
    ``metric_only_numerical_clip`` of ``[1e-15, 1-1e-15]`` and passes it
    explicitly; a wide clip would flatten calibrated probabilities and could
    break the raw-versus-calibrated ranking equivalence that contract checks.
    """

    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("probabilities must be a finite one-dimensional array")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("probability epsilon must lie strictly inside (0, 0.5)")
    return np.clip(values, epsilon, 1.0 - epsilon)


def ece_equal_width(
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: int = 20,
    *,
    epsilon: float = PROBABILITY_EPSILON,
) -> tuple[float, list[dict[str, float | int]]]:
    y = np.asarray(y_true, dtype=np.int8)
    p = clipped(probability, epsilon)
    if y.shape != p.shape or bins <= 0:
        raise ValueError("invalid ECE inputs")
    bin_index = np.minimum((p * bins).astype(np.int64), bins - 1)
    rows: list[dict[str, float | int]] = []
    total = y.size
    ece = 0.0
    for index in range(bins):
        mask = bin_index == index
        count = int(mask.sum())
        mean_probability = float(p[mask].mean()) if count else float("nan")
        observed_rate = float(y[mask].mean()) if count else float("nan")
        contribution = 0.0 if count == 0 else count / total * abs(mean_probability - observed_rate)
        ece += contribution
        rows.append(
            {
                "bin_index": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "rows": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_gap": (
                    float("nan") if count == 0 else abs(mean_probability - observed_rate)
                ),
                "weighted_contribution": contribution,
            }
        )
    return float(ece), rows


def binary_auc_midrank(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(score, dtype=np.float64)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_score = s[order]
    sorted_y = y[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    ends = np.r_[starts[1:], y.size]
    positive_rank_sum = 0.0
    for start, end in zip(starts, ends):
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * float(sorted_y[start:end].sum())
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


@dataclass(frozen=True)
class UserGaucComponents:
    user_ids: np.ndarray
    event_counts: np.ndarray
    eligible: np.ndarray
    auc: np.ndarray

    @property
    def eligible_users(self) -> int:
        return int(self.eligible.sum())

    @property
    def eligible_rows(self) -> int:
        return int(self.event_counts[self.eligible].sum())

    @property
    def event_weighted(self) -> float:
        weights = self.event_counts[self.eligible].astype(np.float64)
        return float(np.average(self.auc[self.eligible], weights=weights))

    @property
    def user_equal(self) -> float:
        return float(self.auc[self.eligible].mean())


def user_gauc_components(
    y_true: np.ndarray,
    probability: np.ndarray,
    user_id: np.ndarray,
    *,
    user_universe: np.ndarray | None = None,
    epsilon: float = PROBABILITY_EPSILON,
) -> UserGaucComponents:
    y = np.asarray(y_true, dtype=np.int8)
    p = clipped(probability, epsilon)
    users = np.asarray(user_id, dtype=np.int64)
    if y.shape != p.shape or y.shape != users.shape:
        raise ValueError("user-GAUC arrays must align")
    universe = np.unique(users) if user_universe is None else np.asarray(user_universe, dtype=np.int64)
    if np.any(universe[1:] <= universe[:-1]):
        raise ValueError("user universe must be strictly increasing")
    lookup = {int(value): index for index, value in enumerate(universe)}
    event_counts = np.zeros(universe.size, dtype=np.int64)
    eligible = np.zeros(universe.size, dtype=bool)
    auc = np.full(universe.size, np.nan, dtype=np.float64)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], users.size]
    for start, end in zip(starts, ends):
        indices = order[start:end]
        index = lookup.get(int(sorted_users[start]))
        if index is None:
            raise ValueError("observed user absent from frozen universe")
        event_counts[index] = end - start
        labels = y[indices]
        if labels.min() != labels.max():
            eligible[index] = True
            auc[index] = binary_auc_midrank(labels, p[indices])
    return UserGaucComponents(universe, event_counts, eligible, auc)


def point_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    user_id: np.ndarray,
    *,
    epsilon: float = PROBABILITY_EPSILON,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int8)
    p = clipped(probability, epsilon)
    if y.shape != p.shape or y.size == 0 or np.unique(y).size != 2:
        raise ValueError("point metrics require aligned nonempty two-class arrays")
    gauc = user_gauc_components(y, p, user_id, epsilon=epsilon)
    ece, _ = ece_equal_width(y, p, bins=20, epsilon=epsilon)
    return {
        "rows": int(y.size),
        "users": int(np.unique(user_id).size),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(np.mean(np.square(p - y), dtype=np.float64)),
        "ece20_equal_width": ece,
        "user_gauc_event_weighted": gauc.event_weighted,
        "user_gauc_user_equal": gauc.user_equal,
        "user_gauc_eligible_users": gauc.eligible_users,
        "user_gauc_eligible_rows": gauc.eligible_rows,
        "user_gauc_eligible_user_fraction": gauc.eligible_users / np.unique(user_id).size,
        "user_gauc_eligible_row_fraction": gauc.eligible_rows / y.size,
    }


def make_multiplicities(
    user_count: int = 950,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[np.ndarray, str]:
    generator = np.random.Generator(np.random.PCG64(seed))
    sampled = generator.integers(0, user_count, size=(replicates, user_count), dtype=np.int32)
    multiplicities = np.zeros((replicates, user_count), dtype=np.uint16)
    for replicate in range(replicates):
        multiplicities[replicate] = np.bincount(
            sampled[replicate], minlength=user_count
        ).astype(np.uint16)
    digest = hashlib.sha256(
        multiplicities.astype("<u2", copy=False).tobytes(order="C")
    ).hexdigest()
    return multiplicities, digest


def _user_index(user_id: np.ndarray, user_universe: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(user_universe, np.asarray(user_id, dtype=np.int64))
    if np.any(positions >= user_universe.size) or np.any(user_universe[positions] != user_id):
        raise ValueError("prediction rows contain users outside frozen universe")
    return positions.astype(np.int32, copy=False)


def weighted_ap_replicates(
    y_true: np.ndarray,
    probability: np.ndarray,
    row_user_index: np.ndarray,
    multiplicities: np.ndarray,
    *,
    block_size: int = AP_BOOTSTRAP_BLOCK_SIZE,
    epsilon: float = PROBABILITY_EPSILON,
) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.int8)
    p = clipped(probability, epsilon)
    user_index = np.asarray(row_user_index, dtype=np.int32)
    if y.shape != p.shape or y.shape != user_index.shape:
        raise ValueError("weighted AP arrays must align")
    order = np.argsort(-p, kind="mergesort")
    score = p[order]
    labels = y[order].astype(np.float64)
    sorted_user_index = user_index[order]
    starts = np.r_[0, np.flatnonzero(score[1:] != score[:-1]) + 1]
    output = np.full(multiplicities.shape[0], np.nan, dtype=np.float64)
    for first in range(0, multiplicities.shape[0], block_size):
        last = min(first + block_size, multiplicities.shape[0])
        weights = multiplicities[first:last, sorted_user_index].astype(np.float64, copy=False)
        group_weight = np.add.reduceat(weights, starts, axis=1)
        group_positive = np.add.reduceat(weights * labels[None, :], starts, axis=1)
        cumulative_weight = np.cumsum(group_weight, axis=1)
        cumulative_positive = np.cumsum(group_positive, axis=1)
        total_positive = cumulative_positive[:, -1]
        valid = total_positive > 0
        precision = np.divide(
            cumulative_positive,
            cumulative_weight,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_weight > 0,
        )
        values = np.full(last - first, np.nan, dtype=np.float64)
        values[valid] = (
            np.sum(precision[valid] * group_positive[valid], axis=1) / total_positive[valid]
        )
        output[first:last] = values
    return output


def _per_user_sums(
    values: np.ndarray, row_user_index: np.ndarray, user_count: int
) -> np.ndarray:
    return np.bincount(
        row_user_index, weights=np.asarray(values, dtype=np.float64), minlength=user_count
    ).astype(np.float64)


def _summary(
    metric: str,
    point: float,
    values: np.ndarray,
    contrast: str = "BL2_minus_BL1",
) -> dict[str, float | int | str]:
    valid = values[np.isfinite(values)]
    return {
        "metric": metric,
        "contrast": contrast,
        "point_estimate": float(point),
        "bootstrap_replicates_requested": int(values.size),
        "effective_replicates": int(valid.size),
        "bootstrap_mean": float(valid.mean()),
        "bootstrap_se": float(valid.std(ddof=1)),
        "ci95_lower": float(np.quantile(valid, 0.025)),
        "ci95_upper": float(np.quantile(valid, 0.975)),
    }


def paired_user_cluster_bootstrap(
    y_true: np.ndarray,
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
    user_id: np.ndarray,
    *,
    user_universe: np.ndarray,
    multiplicities: np.ndarray,
    ap_block_size: int = AP_BOOTSTRAP_BLOCK_SIZE,
    epsilon: float = PROBABILITY_EPSILON,
    contrast: str = "BL2_minus_BL1",
) -> list[dict[str, float | int | str]]:
    y = np.asarray(y_true, dtype=np.int8)
    baseline = clipped(baseline_probability, epsilon)
    candidate = clipped(candidate_probability, epsilon)
    users = np.asarray(user_id, dtype=np.int64)
    universe = np.asarray(user_universe, dtype=np.int64)
    if not (y.shape == baseline.shape == candidate.shape == users.shape):
        raise ValueError("paired bootstrap arrays must align")
    if multiplicities.shape[1] != universe.size:
        raise ValueError("multiplicity matrix and user universe differ")
    row_user_index = _user_index(users, universe)
    user_count = universe.size
    user_rows = np.bincount(row_user_index, minlength=user_count).astype(np.float64)
    replicate_rows = multiplicities @ user_rows

    point_baseline = point_metrics(y, baseline, users, epsilon=epsilon)
    point_candidate = point_metrics(y, candidate, users, epsilon=epsilon)
    summaries: list[dict[str, float | int | str]] = []

    ap_baseline = weighted_ap_replicates(
        y, baseline, row_user_index, multiplicities, block_size=ap_block_size, epsilon=epsilon
    )
    ap_candidate = weighted_ap_replicates(
        y, candidate, row_user_index, multiplicities, block_size=ap_block_size, epsilon=epsilon
    )
    summaries.append(
        _summary(
            "average_precision",
            float(point_candidate["average_precision"] - point_baseline["average_precision"]),
            ap_candidate - ap_baseline,
            contrast,
        )
    )

    for metric, baseline_row_loss, candidate_row_loss in [
        (
            "log_loss",
            -(y * np.log(baseline) + (1 - y) * np.log1p(-baseline)),
            -(y * np.log(candidate) + (1 - y) * np.log1p(-candidate)),
        ),
        ("brier", np.square(baseline - y), np.square(candidate - y)),
    ]:
        baseline_sum = _per_user_sums(baseline_row_loss, row_user_index, user_count)
        candidate_sum = _per_user_sums(candidate_row_loss, row_user_index, user_count)
        delta = (multiplicities @ (candidate_sum - baseline_sum)) / replicate_rows
        summaries.append(
            _summary(
                metric,
                float(point_candidate[metric] - point_baseline[metric]),
                delta,
                contrast,
            )
        )

    baseline_gauc = user_gauc_components(y, baseline, users, user_universe=universe, epsilon=epsilon)
    candidate_gauc = user_gauc_components(y, candidate, users, user_universe=universe, epsilon=epsilon)
    if not np.array_equal(baseline_gauc.eligible, candidate_gauc.eligible):
        raise RuntimeError("paired user-GAUC eligibility differs")
    eligible = baseline_gauc.eligible.astype(np.float64)
    eligible_rows = baseline_gauc.event_counts.astype(np.float64) * eligible
    delta_auc = np.nan_to_num(candidate_gauc.auc - baseline_gauc.auc, nan=0.0)
    event_denominator = multiplicities @ eligible_rows
    user_denominator = multiplicities @ eligible
    event_delta = (multiplicities @ (delta_auc * eligible_rows)) / event_denominator
    user_delta = (multiplicities @ (delta_auc * eligible)) / user_denominator
    summaries.append(
        _summary(
            "user_gauc_event_weighted",
            float(
                point_candidate["user_gauc_event_weighted"]
                - point_baseline["user_gauc_event_weighted"]
            ),
            event_delta,
            contrast,
        )
    )
    summaries.append(
        _summary(
            "user_gauc_user_equal",
            float(point_candidate["user_gauc_user_equal"] - point_baseline["user_gauc_user_equal"]),
            user_delta,
            contrast,
        )
    )
    return summaries
