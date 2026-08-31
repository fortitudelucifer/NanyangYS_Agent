"""Deterministic sparse linear baselines for the Gate 2B Train-only study.

This module deliberately knows nothing about late, Validation, random, Gold, or
restricted-test data. It consumes a previously materialized, point-in-time-safe
feature parquet and fits only the frozen BL1/BL2 SGD logistic configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


IDENTITY_COLUMNS = [
    "source_table",
    "source_row_number",
    "user_id",
    "video_id",
    "event_date",
    "time_ms",
    "long_view",
]

CATEGORICAL_COLUMNS = [
    "cat_user",
    "cat_video",
    "cat_author",
    "cat_music",
    "cat_video_type",
    "cat_upload_type",
    "cat_music_type",
    "cat_tag_combo",
    "cat_duration_bucket",
]

STATIC_NUMERIC_COLUMNS = [
    "static_log_duration",
    "static_duration_valid",
    "static_log_upload_age",
    "static_upload_age_valid",
    "static_upload_future",
    "static_log_width",
    "static_log_height",
    "static_aspect",
    "static_geometry_valid",
    "static_tag_missing",
]

H2_RAW_COLUMNS = [
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
]

ENTITY_PREFIXES = ["video", "author", "tag", "duration"]
REQUIRED_COLUMNS = (
    IDENTITY_COLUMNS
    + CATEGORICAL_COLUMNS
    + STATIC_NUMERIC_COLUMNS
    + H2_RAW_COLUMNS
)


def _to_numpy(table: pa.Table, name: str) -> np.ndarray:
    return table.column(name).combine_chunks().to_numpy(zero_copy_only=False)


@dataclass
class FeatureArrays:
    table: pa.Table
    columns: dict[str, np.ndarray]

    @classmethod
    def read(cls, path: Path) -> "FeatureArrays":
        table = pq.read_table(path, columns=REQUIRED_COLUMNS)
        missing = sorted(set(REQUIRED_COLUMNS) - set(table.column_names))
        if missing:
            raise RuntimeError(f"feature artifact missing required columns: {missing}")
        columns = {name: _to_numpy(table, name) for name in REQUIRED_COLUMNS}
        return cls(table=table, columns=columns)

    @property
    def size(self) -> int:
        return self.table.num_rows

    def dates(self) -> np.ndarray:
        return self.columns["event_date"].astype("datetime64[D]", copy=False)

    def labels(self) -> np.ndarray:
        return self.columns["long_view"].astype(np.int8, copy=False)

    def user_ids(self) -> np.ndarray:
        return self.columns["user_id"].astype(np.int64, copy=False)

    def take_arrow(self, indices: np.ndarray, names: Iterable[str]) -> pa.Table:
        return self.table.select(list(names)).take(pa.array(indices, type=pa.int64()))


@dataclass
class OriginMatrices:
    train_indices: np.ndarray
    assess_indices: np.ndarray
    y_train: np.ndarray
    y_assess: np.ndarray
    p0: float
    static_train: sparse.csr_matrix
    static_assess: sparse.csr_matrix
    h2_train: sparse.csr_matrix
    h2_assess: sparse.csr_matrix
    h3_train: sparse.csr_matrix | None
    h3_assess: sparse.csr_matrix | None
    encoder_categories: list[int]
    static_numeric_dim: int
    h2_numeric_dim: int
    h3_numeric_dim: int


def _matrix(columns: dict[str, np.ndarray], names: list[str], idx: np.ndarray) -> np.ndarray:
    return np.column_stack([columns[name][idx] for name in names])


def _finite_float32(values: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError(f"non-finite values in {label}")
    return result


def _history_h2(columns: dict[str, np.ndarray], idx: np.ndarray, p0: float) -> np.ndarray:
    prior_batch = np.asarray(columns["prior_batch_n"][idx], dtype=np.float64)
    prior_n = np.asarray(columns["prior_event_n"][idx], dtype=np.float64)
    prior_pos = np.asarray(columns["prior_positive_n"][idx], dtype=np.float64)
    gap = np.asarray(columns["last_user_gap_s"][idx], dtype=np.float64)
    pieces = [
        np.log1p(prior_batch),
        np.log1p(prior_n),
        np.log1p(prior_pos),
        (prior_pos + 20.0 * p0) / (prior_n + 20.0),
        (prior_n > 0).astype(np.float64),
        np.log1p(np.maximum(gap, 0.0)),
    ]
    for window in (10, 50, 200):
        event_n = np.asarray(columns[f"w{window}_event_n"][idx], dtype=np.float64)
        positive_n = np.asarray(columns[f"w{window}_positive_n"][idx], dtype=np.float64)
        pieces.extend(
            [
                np.log1p(event_n),
                np.log1p(positive_n),
                (positive_n + 20.0 * p0) / (event_n + 20.0),
                (prior_batch >= window).astype(np.float64),
            ]
        )
    return _finite_float32(np.column_stack(pieces), "H2 history features")


def _history_h3(columns: dict[str, np.ndarray], idx: np.ndarray, p0: float) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for prefix in ENTITY_PREFIXES:
        event_n = np.asarray(columns[f"{prefix}_prior_n"][idx], dtype=np.float64)
        positive_n = np.asarray(columns[f"{prefix}_prior_positive_n"][idx], dtype=np.float64)
        gap = np.asarray(columns[f"{prefix}_last_gap_s"][idx], dtype=np.float64)
        pieces.extend(
            [
                np.log1p(event_n),
                np.log1p(positive_n),
                (positive_n + 10.0 * p0) / (event_n + 10.0),
                (event_n > 0).astype(np.float64),
                np.log1p(np.maximum(gap, 0.0)),
            ]
        )
    return _finite_float32(np.column_stack(pieces), "H3 interaction features")


def prepare_origin_matrices(
    data: FeatureArrays,
    origin: str,
    *,
    include_h3: bool,
    train_limit: int | None = None,
    assess_limit: int | None = None,
) -> OriginMatrices:
    origin_day = np.datetime64(origin, "D")
    dates = data.dates()
    train_indices = np.flatnonzero(dates < origin_day)
    assess_indices = np.flatnonzero(dates == origin_day)
    if train_limit is not None and train_indices.size > train_limit:
        train_indices = train_indices[-train_limit:]
    if assess_limit is not None and assess_indices.size > assess_limit:
        assess_indices = assess_indices[:assess_limit]
    if train_indices.size == 0 or assess_indices.size == 0:
        raise RuntimeError(f"empty train or assessment split for origin {origin}")
    if dates[train_indices].max() >= origin_day or np.any(dates[assess_indices] != origin_day):
        raise RuntimeError(f"date isolation failed for origin {origin}")

    labels = data.labels()
    y_train = np.asarray(labels[train_indices], dtype=np.int8)
    y_assess = np.asarray(labels[assess_indices], dtype=np.int8)
    if np.unique(y_train).size != 2 or np.unique(y_assess).size != 2:
        raise RuntimeError(f"both classes required for origin {origin}")
    p0 = float(y_train.mean(dtype=np.float64))

    cat_train = _matrix(data.columns, CATEGORICAL_COLUMNS, train_indices).astype(np.int64, copy=False)
    cat_assess = _matrix(data.columns, CATEGORICAL_COLUMNS, assess_indices).astype(np.int64, copy=False)
    encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=20,
        max_categories=4096,
        sparse_output=True,
        dtype=np.float32,
    )
    encoded_train = encoder.fit_transform(cat_train).tocsr()
    encoded_assess = encoder.transform(cat_assess).tocsr()

    static_train_raw = _finite_float32(
        _matrix(data.columns, STATIC_NUMERIC_COLUMNS, train_indices), "static train features"
    )
    static_assess_raw = _finite_float32(
        _matrix(data.columns, STATIC_NUMERIC_COLUMNS, assess_indices), "static assessment features"
    )
    static_scaler = StandardScaler(with_mean=False, copy=False)
    static_train_num = sparse.csr_matrix(static_scaler.fit_transform(static_train_raw), dtype=np.float32)
    static_assess_num = sparse.csr_matrix(static_scaler.transform(static_assess_raw), dtype=np.float32)
    static_train = sparse.hstack([encoded_train, static_train_num], format="csr", dtype=np.float32)
    static_assess = sparse.hstack([encoded_assess, static_assess_num], format="csr", dtype=np.float32)

    h2_train_raw = _history_h2(data.columns, train_indices, p0)
    h2_assess_raw = _history_h2(data.columns, assess_indices, p0)
    h2_scaler = StandardScaler(with_mean=False, copy=False)
    h2_train_num = sparse.csr_matrix(h2_scaler.fit_transform(h2_train_raw), dtype=np.float32)
    h2_assess_num = sparse.csr_matrix(h2_scaler.transform(h2_assess_raw), dtype=np.float32)
    h2_train = sparse.hstack([static_train, h2_train_num], format="csr", dtype=np.float32)
    h2_assess = sparse.hstack([static_assess, h2_assess_num], format="csr", dtype=np.float32)

    h3_train = None
    h3_assess = None
    h3_dim = 0
    if include_h3:
        raise RuntimeError("H3 is deferred by operational_amendment_001 and cannot run in this stage")

    return OriginMatrices(
        train_indices=train_indices,
        assess_indices=assess_indices,
        y_train=y_train,
        y_assess=y_assess,
        p0=p0,
        static_train=static_train,
        static_assess=static_assess,
        h2_train=h2_train,
        h2_assess=h2_assess,
        h3_train=h3_train,
        h3_assess=h3_assess,
        encoder_categories=[len(categories) for categories in encoder.categories_],
        static_numeric_dim=len(STATIC_NUMERIC_COLUMNS),
        h2_numeric_dim=h2_train_raw.shape[1],
        h3_numeric_dim=h3_dim,
    )


def fit_predict_sgd(
    train_x: sparse.csr_matrix,
    y_train: np.ndarray,
    assess_x: sparse.csr_matrix,
    *,
    alpha: float,
    seed: int = 20260814,
    max_iter: int = 15,
) -> tuple[np.ndarray, SGDClassifier]:
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(alpha),
        max_iter=int(max_iter),
        tol=None,
        shuffle=True,
        random_state=int(seed),
        early_stopping=False,
        average=True,
        class_weight=None,
        n_jobs=1,
    )
    model.fit(train_x, y_train)
    probabilities = model.predict_proba(assess_x)[:, 1].astype(np.float64, copy=False)
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    if probabilities.shape[0] != assess_x.shape[0] or not np.isfinite(probabilities).all():
        raise RuntimeError("invalid SGD probability output")
    return probabilities, model
