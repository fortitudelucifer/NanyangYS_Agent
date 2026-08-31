"""Rebuild Train-only Gate 2 design evidence for KuaiRand-1K v002.

This script has no configurable data path and no recursive discovery.  It reads
only the four explicitly allowlisted canonical Train inputs (``users`` is
hash-verified but is not needed in the SQL) plus one existing generated CSV used
as a deterministic cross-check.  It never builds Gold and never reads late,
random, Validation, restricted-test, or statistic-feature data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports/generated/gate2_train_design_v002"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/gate2_train_design_evidence_v002.md"
SCRIPT_PATH = Path(__file__).resolve()
TRAIN_START = "2022-04-08"
TRAIN_END = "2022-04-17"
TARGET_ROWS = 2_399_844
TARGET_USERS = 950
TARGET_POSITIVES = 765_417
BOOTSTRAP_SEED = 20_260_814
BOOTSTRAP_REPLICATES = 2_000
THREADS = 1

INPUTS = {
    "early_train": {
        "path": PROJECT_ROOT / "data/silver/events_early_standard.parquet",
        "size_bytes": 143_161_102,
        "sha256": "b9f954eebae4c01e616dc1cca01d5428650e334a83cdaa174618cc77494ad0a0",
        "query_use": "canonical_base",
    },
    "formula_mismatch": {
        "path": PROJECT_ROOT / "data/quarantine/label_formula_mismatch_rows.parquet",
        "size_bytes": 2_776_464,
        "sha256": "1779f544dd0be64ccbd0a4e4107746045750e25af5a189070465b385e5ce843b",
        "query_use": "filtered_canonical_additive",
    },
    "users": {
        "path": PROJECT_ROOT / "data/silver/users.parquet",
        "size_bytes": 30_450,
        "sha256": "044b473ec7158ef726484cf282cd08ec2f4b72e529d85b3dc3a1b128f88aa842",
        "query_use": "hash_verification_only_not_needed_for_this_design_evidence",
    },
    "videos_basic": {
        "path": PROJECT_ROOT / "data/silver/videos_basic.parquet",
        "size_bytes": 77_851_324,
        "sha256": "c0184fd2df1356deeca1faebfc23451bb7269f3d00d4213d8b8150d6134ace1f",
        "query_use": "candidate_mapping_audit",
    },
}

CROSSCHECK = {
    "path": PROJECT_ROOT / "reports/generated/train_association_v002/history_date_coverage.csv",
    "size_bytes": 854,
    "sha256": "04684bbdb221d926f3394128b24e19958dccdb6ace06db7239e38389dbda4eca",
}

COHORTS = [
    ("all_rows_masked", 0, "10|50|200", "primary_population; use_available_history_with_mask"),
    ("history_10_plus", 10, "10", "full_10_window_diagnostic"),
    ("history_50_plus", 50, "10|50", "short_history_fixed_row_ablation"),
    ("history_200_plus", 200, "10|50|200", "long_sequence_primary_fixed_row_ablation"),
    ("history_500_plus", 500, "10|50|200|500", "extended_exploratory_gate"),
]

PICTURE_NAME_PATTERN = "(?i)(picture|photo|album)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        action="store_true",
        help="Run the only supported checkpoint mode. Inputs and Train scope remain fixed.",
    )
    args = parser.parse_args()
    if not args.release:
        parser.error("explicit --release is required; no alternate data scope is supported")
    return args


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_size = path.stat().st_size
    observed_sha = sha256_file(path)
    if observed_size != expected_size:
        raise RuntimeError(f"size mismatch for {path}: {observed_size} != {expected_size}")
    if observed_sha != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed_sha} != {expected_sha256}")
    return {
        "path": relative(path),
        "expected_size_bytes": expected_size,
        "observed_size_bytes": observed_size,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha,
        "size_verified": True,
        "sha256_verified": True,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fetch_dicts(con: duckdb.DuckDBPyConnection, sql: str, columns: list[str]) -> list[dict[str, Any]]:
    return [dict(zip(columns, row)) for row in con.execute(sql).fetchall()]


def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    early = sql_path(INPUTS["early_train"]["path"])
    mismatch = sql_path(INPUTS["formula_mismatch"]["path"])
    videos = sql_path(INPUTS["videos_basic"]["path"])
    con.execute(
        f"""
        CREATE TEMP TABLE canonical_all AS
        SELECT source_table, source_row_number, user_id, video_id, event_date,
               time_ms, tab, long_view, duration_ms
        FROM read_parquet({early})
        WHERE event_date BETWEEN DATE '{TRAIN_START}' AND DATE '{TRAIN_END}'
        UNION ALL
        SELECT source_table, source_row_number, user_id, video_id, event_date,
               time_ms, tab, long_view, duration_ms
        FROM read_parquet({mismatch})
        WHERE source_table = 'early_standard'
          AND exclusion_reason = 'LONG_VIEW_FORMULA_MISMATCH'
          AND event_date BETWEEN DATE '{TRAIN_START}' AND DATE '{TRAIN_END}'
        """
    )
    con.execute("CREATE TEMP TABLE target AS SELECT * FROM canonical_all WHERE tab = 1")
    con.execute(
        """
        CREATE TEMP TABLE history_batch AS
        SELECT user_id, time_ms, count(*)::BIGINT AS batch_event_n
        FROM canonical_all
        GROUP BY user_id, time_ms
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE history_state AS
        SELECT user_id, time_ms,
               row_number() OVER (PARTITION BY user_id ORDER BY time_ms) - 1
                   AS prior_history_batch_count
        FROM history_batch
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE scored AS
        SELECT t.*,
               coalesce(h.prior_history_batch_count, 0)::BIGINT
                   AS prior_history_batch_count,
               v.upload_type, v.video_type, v.video_duration,
               v.server_width, v.server_height
        FROM target t
        LEFT JOIN history_state h USING (user_id, time_ms)
        LEFT JOIN read_parquet({videos}) v USING (video_id)
        """
    )


def validate_tables(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    row = con.execute(
        """
        SELECT count(*) AS target_rows,
               count(DISTINCT (source_table, source_row_number)) AS unique_source_identities,
               count(DISTINCT user_id) AS users,
               count(DISTINCT video_id) AS videos,
               sum(long_view) AS positives,
               min(event_date) AS date_min,
               max(event_date) AS date_max,
               sum((tab <> 1 OR tab IS NULL)::INT) AS non_tab1_rows,
               sum((long_view NOT IN (0, 1) OR long_view IS NULL)::INT) AS invalid_labels,
               sum((upload_type IS NULL)::INT) AS missing_video_joins
        FROM scored
        """
    ).fetchone()
    values = {
        "target_rows": row[0],
        "unique_source_identities": row[1],
        "users": row[2],
        "videos": row[3],
        "positives": row[4],
        "date_min": row[5].isoformat(),
        "date_max": row[6].isoformat(),
        "non_tab1_rows": row[7],
        "invalid_labels": row[8],
        "missing_video_joins": row[9],
    }
    expected = {
        "target_rows": TARGET_ROWS,
        "unique_source_identities": TARGET_ROWS,
        "users": TARGET_USERS,
        "videos": 974_550,
        "positives": TARGET_POSITIVES,
        "date_min": TRAIN_START,
        "date_max": TRAIN_END,
        "non_tab1_rows": 0,
        "invalid_labels": 0,
        "missing_video_joins": 0,
    }
    if values != expected:
        raise RuntimeError(f"canonical invariant mismatch: {values!r} != {expected!r}")
    return values


def burn_in_rows(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for excluded_days, start in enumerate(["2022-04-08", "2022-04-09", "2022-04-10", "2022-04-11"]):
        result = con.execute(
            f"""
            WITH keep AS (SELECT * FROM scored WHERE event_date >= DATE '{start}'),
            user_class AS (
                SELECT user_id, count(*) AS n, sum(long_view) AS y
                FROM keep GROUP BY user_id
            ),
            daily AS (SELECT event_date, avg(long_view) AS rate FROM keep GROUP BY event_date)
            SELECT count(*), count(DISTINCT user_id), count(DISTINCT video_id),
                   sum(long_view), avg(long_view),
                   sum((prior_history_batch_count >= 10)::INT),
                   avg((prior_history_batch_count >= 10)::INT),
                   sum((prior_history_batch_count >= 50)::INT),
                   avg((prior_history_batch_count >= 50)::INT),
                   sum((prior_history_batch_count >= 200)::INT),
                   avg((prior_history_batch_count >= 200)::INT),
                   sum((prior_history_batch_count >= 500)::INT),
                   avg((prior_history_batch_count >= 500)::INT),
                   (SELECT count(*) FROM user_class WHERE y > 0 AND y < n),
                   (SELECT coalesce(sum(n), 0) FROM user_class WHERE y > 0 AND y < n),
                   (SELECT min(rate) FROM daily), (SELECT max(rate) FROM daily)
            FROM keep
            """
        ).fetchone()
        rows.append(
            {
                "policy_id": f"B{excluded_days}",
                "target_start_date": start,
                "excluded_calendar_days": excluded_days,
                "role": (
                    "canonical_all_rows_population"
                    if excluded_days == 0
                    else "frozen_rolling_origin_assessment_target_only"
                    if excluded_days == 3
                    else "diagnostic_sensitivity_not_primary_filter"
                ),
                "target_rows": result[0],
                "retained_row_share": result[0] / TARGET_ROWS,
                "users": result[1],
                "videos": result[2],
                "positives": result[3],
                "label_rate": result[4],
                "history_10_plus_rows": result[5],
                "history_10_plus_share": result[6],
                "history_50_plus_rows": result[7],
                "history_50_plus_share": result[8],
                "history_200_plus_rows": result[9],
                "history_200_plus_share": result[10],
                "history_500_plus_rows": result[11],
                "history_500_plus_share": result[12],
                "both_label_users": result[13],
                "both_label_user_rows": result[14],
                "min_daily_label_rate": result[15],
                "max_daily_label_rate": result[16],
            }
        )
    return rows


def identity_digests(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Hash sorted source identities in one streaming pass for all nested cohorts."""
    hashers = {cohort_id: hashlib.sha256() for cohort_id, *_ in COHORTS}
    cursor = con.execute(
        """
        SELECT source_table, source_row_number, prior_history_batch_count
        FROM scored
        ORDER BY source_table, source_row_number
        """
    )
    while True:
        batch = cursor.fetchmany(100_000)
        if not batch:
            break
        for source_table, source_row_number, history_count in batch:
            payload = f"{source_table}\t{source_row_number}\n".encode("utf-8")
            for cohort_id, minimum, *_ in COHORTS:
                if history_count >= minimum:
                    hashers[cohort_id].update(payload)
    return {key: value.hexdigest() for key, value in hashers.items()}


def fixed_row_evidence(
    con: duckdb.DuckDBPyConnection, digests: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    by_date: list[dict[str, Any]] = []
    total_by_date = dict(con.execute("SELECT event_date, count(*) FROM scored GROUP BY event_date").fetchall())
    for cohort_id, minimum, windows, role in COHORTS:
        where = "TRUE" if minimum == 0 else f"prior_history_batch_count >= {minimum}"
        result = con.execute(
            f"""
            WITH keep AS (SELECT * FROM scored WHERE {where}),
            user_class AS (
                SELECT user_id, count(*) AS n, sum(long_view) AS y
                FROM keep GROUP BY user_id
            ),
            time_group AS (
                SELECT user_id, time_ms, count(*) AS n, sum(long_view) AS y
                FROM keep GROUP BY user_id, time_ms
            )
            SELECT count(*), count(DISTINCT user_id), count(DISTINCT video_id),
                   sum(long_view), avg(long_view),
                   count(DISTINCT (user_id, time_ms)),
                   (SELECT count(*) FROM time_group WHERE y > 0 AND y < n),
                   (SELECT count(*) FROM user_class WHERE y > 0 AND y < n),
                   (SELECT coalesce(sum(n), 0) FROM user_class WHERE y > 0 AND y < n),
                   min(event_date), max(event_date)
            FROM keep
            """
        ).fetchone()
        summary.append(
            {
                "cohort_id": cohort_id,
                "minimum_prior_completed_batches": minimum,
                "reproducible_filter": "TRUE" if minimum == 0 else f"prior_history_batch_count >= {minimum}",
                "windows_compared": windows,
                "design_role": role,
                "short_history_policy": (
                    "use_all_available_with_mask" if minimum == 0 else "not_applicable_full_window_cohort"
                ),
                "identity_key": "source_table|source_row_number",
                "identity_sha256": digests[cohort_id],
                "target_rows": result[0],
                "row_share": result[0] / TARGET_ROWS,
                "users": result[1],
                "videos": result[2],
                "positives": result[3],
                "negatives": result[0] - result[3],
                "label_rate": result[4],
                "user_timestamp_groups": result[5],
                "mixed_label_user_timestamp_groups": result[6],
                "both_label_users": result[7],
                "both_label_user_rows": result[8],
                "both_label_user_row_share": result[8] / result[0],
                "earliest_date": result[9].isoformat(),
                "latest_date": result[10].isoformat(),
                "manifest_role": "logical_source_identity_prototype_not_gold_sample_id_manifest",
            }
        )
        daily = con.execute(
            f"""
            SELECT event_date, count(*), count(DISTINCT user_id),
                   sum(long_view), avg(long_view)
            FROM scored WHERE {where}
            GROUP BY event_date ORDER BY event_date
            """
        ).fetchall()
        for event_date, count, users, positives, label_rate in daily:
            by_date.append(
                {
                    "cohort_id": cohort_id,
                    "event_date": event_date.isoformat(),
                    "rows": count,
                    "share_of_same_date_rows": count / total_by_date[event_date],
                    "users": users,
                    "positives": positives,
                    "label_rate": label_rate,
                }
            )
    return summary, by_date


def crosscheck_release_coverage(con: duckdb.DuckDBPyConnection) -> float:
    release: dict[str, dict[str, str]] = {}
    with CROSSCHECK["path"].open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            release[row["event_date"]] = row
    maximum = 0.0
    for day in range(8, 18):
        event_date = f"2022-04-{day:02d}"
        for threshold, column in [
            (50, "share_history_50_plus"),
            (200, "share_history_200_plus"),
            (500, "share_history_500_plus"),
        ]:
            observed = con.execute(
                f"""
                SELECT avg((prior_history_batch_count >= {threshold})::INT)
                FROM scored WHERE event_date = DATE '{event_date}'
                """
            ).fetchone()[0]
            maximum = max(maximum, abs(observed - float(release[event_date][column])))
    if maximum >= 1e-15:
        raise RuntimeError(f"release coverage cross-check failed: max abs diff={maximum}")
    return maximum


def mapping_evidence(
    con: duckdb.DuckDBPyConnection,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    valid_duration = "video_duration IS NOT NULL AND isfinite(video_duration) AND video_duration > 0"
    event_signal = "duration_ms IS NULL OR duration_ms <= 0"
    video_signal = f"NOT ({valid_duration})"
    category_columns = [
        "upload_type", "candidate_picture_name_flag", "target_rows", "row_share", "videos", "users",
        "event_duration_signal_rows", "event_duration_signal_share", "video_duration_signal_rows",
        "video_duration_signal_share", "diagnostic_name_vs_event_duration_disagreement_rows",
        "diagnostic_name_vs_event_duration_disagreement_rate", "normal_video_type_share",
        "geometry_complete_share",
    ]
    categories = fetch_dicts(
        con,
        f"""
        WITH mapped AS (
            SELECT coalesce(upload_type, 'UNKNOWN') AS upload_type,
                   regexp_matches(coalesce(upload_type, ''), '{PICTURE_NAME_PATTERN}') AS name_flag,
                   ({event_signal}) AS event_signal,
                   ({video_signal}) AS video_signal,
                   user_id, video_id, video_type, server_width, server_height
            FROM scored
        )
        SELECT upload_type, bool_or(name_flag), count(*), count(*) / {TARGET_ROWS}.0,
               count(DISTINCT video_id), count(DISTINCT user_id),
               sum(event_signal::INT), avg(event_signal::INT),
               sum(video_signal::INT), avg(video_signal::INT),
               sum((name_flag <> event_signal)::INT), avg((name_flag <> event_signal)::INT),
               avg((video_type = 'NORMAL')::INT),
               avg((server_width > 0 AND server_height > 0)::INT)
        FROM mapped GROUP BY upload_type ORDER BY count(*) DESC, upload_type
        """,
        category_columns,
    )
    contingency: list[dict[str, Any]] = []
    contingency_columns = [
        "audit_basis", "candidate_picture_name_flag", "audit_signal_flag", "rows",
        "row_share", "share_within_candidate_or_signal_union",
    ]
    for basis, signal in [
        ("event_duration_missing_or_nonpositive", event_signal),
        ("videos_basic_duration_missing_or_nonpositive", video_signal),
    ]:
        contingency.extend(
            fetch_dicts(
                con,
                f"""
                WITH mapped AS (
                    SELECT regexp_matches(coalesce(upload_type, ''), '{PICTURE_NAME_PATTERN}') AS name_flag,
                           ({signal}) AS signal
                    FROM scored
                ), union_size AS (SELECT sum((name_flag OR signal)::INT) AS n FROM mapped)
                SELECT '{basis}', name_flag, signal, count(*), count(*) / {TARGET_ROWS}.0,
                       CASE WHEN name_flag OR signal
                            THEN count(*) / (SELECT n FROM union_size)::DOUBLE END
                FROM mapped GROUP BY name_flag, signal
                ORDER BY name_flag DESC, signal DESC
                """,
                contingency_columns,
            )
        )
    summary_columns = [
        "candidate_rows", "candidate_videos", "candidate_users", "event_duration_signal_rows",
        "candidate_and_event_signal_rows", "event_diagnostic_disagreement_rows",
        "candidate_or_event_signal_rows", "video_duration_signal_rows", "candidate_and_video_signal_rows",
        "video_diagnostic_disagreement_rows", "candidate_or_video_signal_rows",
        "candidate_normal_video_type_share",
    ]
    summary = fetch_dicts(
        con,
        f"""
        WITH mapped AS (
            SELECT regexp_matches(coalesce(upload_type, ''), '{PICTURE_NAME_PATTERN}') AS name_flag,
                   ({event_signal}) AS event_signal, ({video_signal}) AS video_signal,
                   user_id, video_id, video_type
            FROM scored
        )
        SELECT sum(name_flag::INT), count(DISTINCT CASE WHEN name_flag THEN video_id END),
               count(DISTINCT CASE WHEN name_flag THEN user_id END),
               sum(event_signal::INT), sum((name_flag AND event_signal)::INT),
               sum((name_flag <> event_signal)::INT), sum((name_flag OR event_signal)::INT),
               sum(video_signal::INT), sum((name_flag AND video_signal)::INT),
               sum((name_flag <> video_signal)::INT), sum((name_flag OR video_signal)::INT),
               avg(CASE WHEN name_flag THEN (video_type = 'NORMAL')::INT END)
        FROM mapped
        """,
        summary_columns,
    )[0]
    summary.update(
        {
            "candidate_row_share": summary["candidate_rows"] / TARGET_ROWS,
            "event_signal_capture": summary["candidate_and_event_signal_rows"] / summary["event_duration_signal_rows"],
            "event_signal_candidate_agreement": summary["candidate_and_event_signal_rows"] / summary["candidate_rows"],
            "event_signal_jaccard": summary["candidate_and_event_signal_rows"] / summary["candidate_or_event_signal_rows"],
            "event_diagnostic_disagreement_share_of_union": summary["event_diagnostic_disagreement_rows"] / summary["candidate_or_event_signal_rows"],
            "video_signal_capture": summary["candidate_and_video_signal_rows"] / summary["video_duration_signal_rows"],
            "video_signal_candidate_agreement": summary["candidate_and_video_signal_rows"] / summary["candidate_rows"],
            "video_signal_jaccard": summary["candidate_and_video_signal_rows"] / summary["candidate_or_video_signal_rows"],
            "video_diagnostic_disagreement_share_of_union": summary["video_diagnostic_disagreement_rows"] / summary["candidate_or_video_signal_rows"],
            "status": "diagnostic_only_not_freezable_as_true_modality",
        }
    )
    proxy_columns = ["proxy_level", "rows", "row_share", "users", "videos", "positives", "negatives", "label_rate", "mapping_rule", "status"]
    proxy = fetch_dicts(
        con,
        f"""
        WITH mapped AS (
            SELECT *, regexp_matches(coalesce(upload_type, ''), '{PICTURE_NAME_PATTERN}') AS name_flag,
                   ({valid_duration}) AS valid_duration
            FROM scored
        ), classified AS (
            SELECT *, CASE
                WHEN name_flag AND NOT valid_duration THEN 'picture_like'
                WHEN NOT name_flag AND valid_duration THEN 'video_like'
                ELSE 'unknown' END AS proxy_level
            FROM mapped
        )
        SELECT proxy_level, count(*), count(*) / {TARGET_ROWS}.0,
               count(DISTINCT user_id), count(DISTINCT video_id), sum(long_view),
               count(*) - sum(long_view), avg(long_view),
               CASE proxy_level
                 WHEN 'picture_like' THEN 'name_flag AND videos_basic_duration_missing_or_nonpositive'
                 WHEN 'video_like' THEN 'NOT name_flag AND videos_basic_duration_positive_finite'
                 ELSE 'all_remaining_conflicts_or_uncertain' END,
               'candidate_proxy_v1_diagnostic_only_not_official_modality'
        FROM classified GROUP BY proxy_level
        ORDER BY CASE proxy_level WHEN 'video_like' THEN 1 WHEN 'picture_like' THEN 2 ELSE 3 END
        """,
        proxy_columns,
    )
    unknown_columns = ["unknown_reason", "rows", "row_share", "users", "positives", "negatives"]
    unknown = fetch_dicts(
        con,
        f"""
        WITH mapped AS (
            SELECT *, regexp_matches(coalesce(upload_type, ''), '{PICTURE_NAME_PATTERN}') AS name_flag,
                   ({valid_duration}) AS valid_duration
            FROM scored
        )
        SELECT CASE
                 WHEN name_flag AND valid_duration THEN 'picture_name_but_positive_video_duration'
                 WHEN NOT name_flag AND NOT valid_duration THEN 'nonpicture_name_but_missing_or_nonpositive_video_duration'
               END,
               count(*), count(*) / {TARGET_ROWS}.0, count(DISTINCT user_id),
               sum(long_view), count(*) - sum(long_view)
        FROM mapped
        WHERE name_flag = valid_duration
        GROUP BY 1 ORDER BY count(*) DESC
        """,
        unknown_columns,
    )
    return categories, contingency, [summary], proxy, unknown


def cluster_and_bootstrap(
    con: duckdb.DuckDBPyConnection,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    user_ids = [row[0] for row in con.execute("SELECT DISTINCT user_id FROM scored ORDER BY user_id").fetchall()]
    if len(user_ids) != TARGET_USERS:
        raise RuntimeError(f"unexpected bootstrap user universe: {len(user_ids)}")
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    universe_hash = hashlib.sha256("".join(f"{user_id}\n" for user_id in user_ids).encode("utf-8")).hexdigest()
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    sampled_indices = generator.integers(
        0, TARGET_USERS, size=(BOOTSTRAP_REPLICATES, TARGET_USERS), dtype=np.int32
    )
    multiplicities = np.zeros((BOOTSTRAP_REPLICATES, TARGET_USERS), dtype=np.uint16)
    for replicate in range(BOOTSTRAP_REPLICATES):
        multiplicities[replicate] = np.bincount(
            sampled_indices[replicate], minlength=TARGET_USERS
        ).astype(np.uint16)
    multiplicity_hash = hashlib.sha256(multiplicities.astype("<u2", copy=False).tobytes(order="C")).hexdigest()
    concentration: list[dict[str, Any]] = []
    bootstrap: list[dict[str, Any]] = []
    for cohort_id, minimum, *_ in COHORTS:
        where = "TRUE" if minimum == 0 else f"prior_history_batch_count >= {minimum}"
        per_user = con.execute(
            f"""
            SELECT user_id, count(*)::DOUBLE, sum(long_view)::DOUBLE
            FROM scored WHERE {where} GROUP BY user_id ORDER BY user_id
            """
        ).fetchall()
        events = np.zeros(TARGET_USERS, dtype=np.float64)
        positives = np.zeros(TARGET_USERS, dtype=np.float64)
        present = np.zeros(TARGET_USERS, dtype=np.float64)
        for user_id, count, positive in per_user:
            index = user_index[user_id]
            events[index] = count
            positives[index] = positive
            present[index] = 1.0
        cohort_events = events[present == 1]
        cohort_positives = positives[present == 1]
        rates = np.divide(positives, events, out=np.zeros_like(positives), where=events > 0)
        both = (positives > 0) & (positives < events)
        sorted_events = np.sort(cohort_events)[::-1]
        users = len(cohort_events)
        point_micro = positives.sum() / events.sum()
        naive_se = math.sqrt(point_micro * (1 - point_micro) / events.sum())
        concentration.append(
            {
                "cohort_id": cohort_id,
                "users": users,
                "events": int(events.sum()),
                "both_label_users": int(both.sum()),
                "both_label_user_fraction": float(both.sum() / users),
                "both_label_user_event_share": float(events[both].sum() / events.sum()),
                "min_events_per_user": int(cohort_events.min()),
                "p25_events_per_user": float(np.quantile(cohort_events, 0.25)),
                "median_events_per_user": float(np.median(cohort_events)),
                "p75_events_per_user": float(np.quantile(cohort_events, 0.75)),
                "p90_events_per_user": float(np.quantile(cohort_events, 0.90)),
                "p99_events_per_user": float(np.quantile(cohort_events, 0.99)),
                "max_events_per_user": int(cohort_events.max()),
                "top_10pct_users_event_share": float(sorted_events[: math.ceil(0.10 * users)].sum() / events.sum()),
                "top_25pct_users_event_share": float(sorted_events[: math.ceil(0.25 * users)].sum() / events.sum()),
                "naive_iid_event_rate_se": naive_se,
            }
        )
        replicate_event_n = multiplicities @ events
        replicate_positive_n = multiplicities @ positives
        replicate_user_n = multiplicities @ present
        valid = (replicate_event_n > 0) & (replicate_user_n > 0)
        micro = replicate_positive_n[valid] / replicate_event_n[valid]
        macro = (multiplicities[valid] @ rates) / replicate_user_n[valid]
        for estimand, values, point, iid_se in [
            ("event_micro_label_rate", micro, point_micro, naive_se),
            ("user_macro_mean_label_rate", macro, float(rates.sum() / present.sum()), None),
        ]:
            bootstrap_se = float(values.std(ddof=1))
            bootstrap.append(
                {
                    "cohort_id": cohort_id,
                    "estimand": estimand,
                    "point_estimate": point,
                    "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
                    "effective_replicates": len(values),
                    "seed": BOOTSTRAP_SEED,
                    "cluster_unit": "user_id",
                    "shared_resample_plan": True,
                    "user_universe_size": TARGET_USERS,
                    "multiplicity_matrix_sha256": multiplicity_hash,
                    "bootstrap_se": bootstrap_se,
                    "ci95_lower": float(np.quantile(values, 0.025)),
                    "ci95_upper": float(np.quantile(values, 0.975)),
                    "naive_iid_se": "" if iid_se is None else iid_se,
                    "se_inflation_vs_iid": "" if iid_se is None else bootstrap_se / iid_se,
                    "role": "shared_cluster_resample_plumbing_smoke_test_not_model_contrast",
                }
            )
    plan = {
        "cluster_unit": "user_id",
        "user_order": "ascending_user_id_from_all_rows_target_universe",
        "user_universe_size": TARGET_USERS,
        "user_universe_sha256": universe_hash,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "bit_generator": "numpy.PCG64",
        "draw_shape": [BOOTSTRAP_REPLICATES, TARGET_USERS],
        "multiplicity_dtype_for_hash": "little_endian_uint16",
        "multiplicity_matrix_sha256": multiplicity_hash,
        "shared_across_all_cohorts_estimands_and_future_paired_models": True,
        "ci_method_for_smoke_test": "percentile_2.5_97.5",
        "status": "resample_plumbing_freeze_ready_model_contrast_variance_not_yet_observed",
    }
    return concentration, bootstrap, plan


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def integer(value: int) -> str:
    return f"{value:,}"


def render_report(
    burn: list[dict[str, Any]],
    fixed: list[dict[str, Any]],
    mapping_summary: dict[str, Any],
    proxy: list[dict[str, Any]],
    unknown: list[dict[str, Any]],
    cluster: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
    bootstrap_plan: dict[str, Any],
    invariants: dict[str, Any],
    crosscheck_max_diff: float,
) -> str:
    burn_lines = "\n".join(
        f"| {row['policy_id']} | {row['target_start_date']} | {integer(row['target_rows'])} | "
        f"{percent(row['retained_row_share'])} | {percent(row['history_50_plus_share'])} | "
        f"{percent(row['history_200_plus_share'])} | {percent(row['history_500_plus_share'])} | "
        f"{integer(row['users'])} | {integer(row['both_label_users'])} | {percent(row['label_rate'])} |"
        for row in burn
    )
    fixed_lines = "\n".join(
        f"| {row['cohort_id']} | {row['windows_compared'].replace('|', ' / ')} | {integer(row['target_rows'])} | "
        f"{percent(row['row_share'])} | {integer(row['users'])} | {integer(row['both_label_users'])} | "
        f"{percent(row['label_rate'])} | `{row['identity_sha256'][:12]}...` |"
        for row in fixed
    )
    proxy_by_level = {row["proxy_level"]: row for row in proxy}
    proxy_lines = "\n".join(
        f"| {level} | {integer(proxy_by_level[level]['rows'])} | {percent(proxy_by_level[level]['row_share'])} | "
        f"{integer(proxy_by_level[level]['users'])} | {integer(proxy_by_level[level]['positives'])} | "
        f"{integer(proxy_by_level[level]['negatives'])} | {percent(proxy_by_level[level]['label_rate'])} |"
        for level in ["video_like", "picture_like", "unknown"]
    )
    unknown_lines = "\n".join(
        f"| {row['unknown_reason']} | {integer(row['rows'])} | {percent(row['row_share'])} | {integer(row['users'])} |"
        for row in unknown
    )
    cluster_by_id = {row["cohort_id"]: row for row in cluster}
    boot_by_key = {(row["cohort_id"], row["estimand"]): row for row in bootstrap}
    cluster_lines = "\n".join(
        f"| {cohort_id} | {integer(cluster_by_id[cohort_id]['users'])} | "
        f"{percent(cluster_by_id[cohort_id]['top_10pct_users_event_share'])} | "
        f"{percent(cluster_by_id[cohort_id]['top_25pct_users_event_share'])} | "
        f"{integer(cluster_by_id[cohort_id]['max_events_per_user'])} | "
        f"{boot_by_key[(cohort_id, 'event_micro_label_rate')]['bootstrap_se']:.5f} | "
        f"{boot_by_key[(cohort_id, 'event_micro_label_rate')]['se_inflation_vs_iid']:.1f}x |"
        for cohort_id in ["all_rows_masked", "history_50_plus", "history_200_plus", "history_500_plus"]
    )
    return f"""# Gate 2 Train-only 设计证据 v002

> 状态：**设计证据已生成；不代表 Gate 2 已通过，也不是模型训练结论**  
> 数据边界：canonical source-Train，{TRAIN_START} 至 {TRAIN_END}；目标 `tab=1`；历史使用同范围全 tab，严格 `history_time < target_time`  
> 禁止访问：late、random、Validation、restricted test、`video_features_statistic_1k.csv`；未重清洗 Silver，未构建 Gold

## 1. 本轮回答什么

本轮只为四个待定设计提供可复算证据：calendar burn-in 的覆盖代价、10/50/200 固定目标行、`picture-like` 候选代理的映射审计、以及 user-cluster bootstrap 的最小实现。canonical 目标仍为 {integer(invariants['target_rows'])} 行、{integer(invariants['users'])} 个用户、{integer(invariants['positives'])} 个正例；源身份键唯一。既有 release 覆盖表的最大绝对复算差为 `{crosscheck_max_diff:.1e}`。

## 2. Calendar burn-in：只作敏感性和 rolling-origin 设计证据

主目标人群继续使用 **all rows + available-history mask**。B1–B2 只是敏感性候选；B3 只定义未来 Train rolling-origin 的 assessment target 日期，不定义新的 canonical 人群。

| policy | 目标起点 | 保留行 | 行保留率 | history≥50 | history≥200 | history≥500 | 用户 | 双标签用户 | 正例率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{burn_lines}

B3（从 04-11 开始）把 `history≥200` 覆盖从 72.03% 提高到 88.43%，但若把它误当成人群过滤就会丢失 758,746 行（31.62%）和前三个日历日；保留期每日正例率仍覆盖 29.90%–34.88%。因此 coverage 不能证明 B3 更“正确”，也不能消除时间漂移。

- **协议决定**：canonical 主人群不按 calendar burn-in 删除目标行，短历史用显式 mask；04-08 至 04-10 冻结为历史积累期，04-11 至 04-17 冻结为未来七个逐日 Train assessment origins。该选择只依据 Train 覆盖和需要七个日回测点，不使用标签率优化。
- **不可声称**：04-11 是更“正确”的主人群起点、全局资格阈值或已被模型增量验证的最优日期；B0–B2 继续作为覆盖/敏感性证据。

## 3. 10/50/200 固定行可用性

| cohort | 比较窗口 | 目标行 | 覆盖 | 用户 | 双标签用户 | 正例率 | 源身份 digest |
|---|---|---:|---:|---:|---:|---:|---|
{fixed_lines}

`history_200_plus` 仍有 1,728,489 行、758 个用户，可以承担 10/50/200 的主长序列消融；但它在 04-08 的日内覆盖仅 5.97%，到 04-17 才达到 94.21%，所以任何窗口增量都必须逐日报告，不能只看池化指标。`history_500_plus` 仅保留 452 个用户，继续限定为 exploratory 是合理的。

每个 cohort 已记录过滤式和按 `(source_table, source_row_number)` 排序流式计算的 SHA-256。但这些 digest 是 **logical/source-identity prototype**，不是 Gold `sample_id` manifest。

- **可冻结**：all-row masked 主设计；`history≥50` 的 10/50 固定行消融；`history≥200` 的 10/50/200 固定行消融；500 仅 exploratory；比较内不得按模型删行。
- **仍待完成**：Gold builder 生成正式 target-row manifests、`sample_id`、特征矩阵 denylist 检查及其最终哈希。

## 4. `candidate_mapping_audit`：不是官方内容模态

名称候选只使用大小写不敏感 token `picture|photo|album`，命中的值为 `LongPicture`、`PictureSet`、`PictureCopy`、`PhotoCopy`、`FlashPhoto`、`OriginPicture`、`LocalIntelligenceAlbum`。它覆盖 {integer(mapping_summary['candidate_rows'])} 行（{percent(mapping_summary['candidate_row_share'])}）、{integer(mapping_summary['candidate_users'])} 个用户。

与 event-duration audit signal 的交集为 {integer(mapping_summary['candidate_and_event_signal_rows'])} 行，Jaccard={mapping_summary['event_signal_jaccard']:.3f}；union 内诊断不一致为 {integer(mapping_summary['event_diagnostic_disagreement_rows'])} 行（{percent(mapping_summary['event_diagnostic_disagreement_share_of_union'])}）。与 `videos_basic.video_duration` signal 的 Jaccard={mapping_summary['video_signal_jaccard']:.3f}，union 内不一致为 {percent(mapping_summary['video_diagnostic_disagreement_share_of_union'])}。这些只是两个非权威 signal 的一致性，不是 accuracy。

保守候选 `proxy_v1` 定义如下：

- `picture_like = name_flag AND videos_basic duration missing/nonpositive`；
- `video_like = NOT name_flag AND videos_basic duration positive finite`；
- 其余全部为 `unknown`。

| proxy_v1 | 行 | 覆盖 | 用户 | 正例 | 负例 | 正例率（仅结果描述） |
|---|---:|---:|---:|---:|---:|---:|
{proxy_lines}

`unknown` 的来源：

| 原因 | 行 | 覆盖 | 用户 |
|---|---:|---:|---:|
{unknown_lines}

正负例计数只用于审计异质性，**没有参与映射规则**。`video_type` 和几何完整性也没有提供官方语义确认。

- **可冻结**：`content_modality_proxy_v1` 的 diagnostic-only 规则、unknown 默认和三切片审计输出。
- **不可冻结**：把 `proxy_v1` 称为真实模态、用作正式 predictor，或据此把主人群改为 video-only。未来未见值仍自动进入 unknown。

## 5. 用户聚类不确定性最小方案

| cohort | 用户 | top 10% 用户行占比 | top 25% | 单用户最大行 | cluster SE（正例率烟测） | 相对 naive-iid SE |
|---|---:|---:|---:|---:|---:|---:|
{cluster_lines}

all-row 主人群中 top 10% 用户贡献 35.85% 事件，逐行独立假设把 event-micro 正例率 SE 低估约 27.8 倍。该数值不是模型性能不确定性，但足以否决 row-iid 推断。

最小实现为：在升序 950 用户全集上，用 `numpy.PCG64(seed={BOOTSTRAP_SEED})` 生成同一个 `{BOOTSTRAP_REPLICATES} × 950` 用户 multiplicity plan；所有 cohort、estimand、baseline/candidate 共用。矩阵 digest 为 `{bootstrap_plan['multiplicity_matrix_sha256']}`。每次抽中用户时携带该用户在固定 cohort 的全部行；比较模型时复算 PR-AUC、Log Loss、Brier 和 user-GAUC 的**配对差**，不对两个模型各自独立 bootstrap。当前 CSV 只是标签率 plumbing smoke test，{BOOTSTRAP_REPLICATES} 次均有效。

- **可冻结**：cluster=`user_id`、共享 multiplicities、固定用户顺序、同一 replicate 内所有模型和指标配对。
- **不可冻结**：Train 模型差的方差必须等固定 baseline/candidate 配对预测；MDE 必须进一步等待获批的 Validation 配对预测。SESOI 和非劣界也不能由本次 label-rate smoke test 推出。

## 6. Gate 2 当前结论

| 项目 | 状态 |
|---|---|
| canonical Train 边界与唯一源身份 | 可冻结 |
| all-row + mask 主人群 | 可冻结 |
| Calendar burn-in | canonical 全行保留；B3 冻结为 assessment-only 起点，B1/B2 为 sensitivity |
| 10/50/200 cohort 逻辑和 prototype digests | 可冻结逻辑；正式 Gold manifests 待建 |
| 500 窗口 | 可冻结为 exploratory-only |
| `proxy_v1` | diagnostic-only 规则已冻结；真实模态与 predictor 声明禁止 |
| user-cluster 共享重采样设计 | 可冻结 plumbing；模型差异边界未冻结 |

因此，本轮推进了 Gate 2 的**设计证据**，但尚不能宣布 Gate 2 通过。下一步应先冻结相同 fixed rows 上的 baseline/candidate 协议，再在获得正式训练批准后产生 Train rolling-origin 配对预测，以估计 Train 原型差值和逐日稳定性；Validation MDE 仍必须等待获批的配对 Validation 预测。Gold build 也需单独批准。

## 7. 可复算产物

- `scripts/analyze_gate2_train_design_v002.py`
- `reports/generated/gate2_train_design_v002/burn_in_tradeoff.csv`
- `reports/generated/gate2_train_design_v002/fixed_row_cohort_summary.csv`
- `reports/generated/gate2_train_design_v002/fixed_row_by_date.csv`
- `reports/generated/gate2_train_design_v002/candidate_mapping_audit.csv`
- `reports/generated/gate2_train_design_v002/candidate_mapping_contingency.csv`
- `reports/generated/gate2_train_design_v002/candidate_mapping_summary.csv`
- `reports/generated/gate2_train_design_v002/candidate_proxy_v1_summary.csv`
- `reports/generated/gate2_train_design_v002/candidate_proxy_v1_unknown_audit.csv`
- `reports/generated/gate2_train_design_v002/user_cluster_summary.csv`
- `reports/generated/gate2_train_design_v002/cluster_bootstrap_label_rate.csv`
- `reports/generated/gate2_train_design_v002/design_manifest.json`
"""


def main() -> None:
    parse_args()
    started = time.perf_counter()
    verified_inputs = []
    for role, spec in INPUTS.items():
        verified = verify_file(spec["path"], spec["size_bytes"], spec["sha256"])
        verified.update({"role": role, "query_use": spec["query_use"]})
        verified_inputs.append(verified)
    verified_crosscheck = verify_file(
        CROSSCHECK["path"], CROSSCHECK["size_bytes"], CROSSCHECK["sha256"]
    )
    con = duckdb.connect()
    con.execute(f"SET threads={THREADS}")
    create_tables(con)
    invariants = validate_tables(con)
    burn = burn_in_rows(con)
    digests = identity_digests(con)
    fixed, fixed_by_date = fixed_row_evidence(con, digests)
    crosscheck_max_diff = crosscheck_release_coverage(con)
    mapping, contingency, mapping_summary, proxy, unknown = mapping_evidence(con)
    cluster, bootstrap, bootstrap_plan = cluster_and_bootstrap(con)
    con.close()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_rows = {
        "burn_in_tradeoff.csv": burn,
        "fixed_row_cohort_summary.csv": fixed,
        "fixed_row_by_date.csv": fixed_by_date,
        "candidate_mapping_audit.csv": mapping,
        "candidate_mapping_contingency.csv": contingency,
        "candidate_mapping_summary.csv": mapping_summary,
        "candidate_proxy_v1_summary.csv": proxy,
        "candidate_proxy_v1_unknown_audit.csv": unknown,
        "user_cluster_summary.csv": cluster,
        "cluster_bootstrap_label_rate.csv": bootstrap,
    }
    output_paths: list[Path] = []
    for name, rows in output_rows.items():
        path = OUTPUT_DIR / name
        write_csv(path, rows)
        output_paths.append(path)
    report = render_report(
        burn, fixed, mapping_summary[0], proxy, unknown, cluster, bootstrap,
        bootstrap_plan, invariants, crosscheck_max_diff,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")
    output_paths.append(REPORT_PATH)
    script_hash = sha256_file(SCRIPT_PATH)
    manifest = {
        "analysis_id": "gate2_train_design_evidence_v002",
        "analysis_date": "2026-08-14",
        "run_mode": "release",
        "status": "complete_train_only_design_evidence_not_gate2_approval",
        "checkpoint_eligible_artifact": True,
        "gate2_approved": False,
        "input_data_files": verified_inputs,
        "existing_generated_crosscheck": verified_crosscheck,
        "scope": {
            "target_date_range": [TRAIN_START, TRAIN_END],
            "target_tab": 1,
            "history_tabs": "all",
            "history_time_rule": "strictly_less_than_target_time",
            "calendar_burn_in": {
                "history_accumulation_range": ["2022-04-08", "2022-04-10"],
                "rolling_origin_assessment_target_range": ["2022-04-11", "2022-04-17"],
                "canonical_train_row_exclusion": False,
                "role": "frozen_protocol_decision_no_training_authorization",
            },
            "canonical_additive_filter": {
                "source_table": "early_standard",
                "exclusion_reason": "LONG_VIEW_FORMULA_MISMATCH",
            },
            "late_access": False,
            "random_access": False,
            "validation_access": False,
            "restricted_test_access": False,
            "statistic_file_access": False,
            "silver_recleaned": False,
            "gold_built": False,
            "formal_model_training": False,
        },
        "invariants": {**invariants, "release_history_coverage_max_abs_diff": crosscheck_max_diff},
        "fixed_row_source_identity_prototypes": [
            {
                "cohort_id": row["cohort_id"],
                "filter": row["reproducible_filter"],
                "identity_key": ["source_table", "source_row_number"],
                "identity_digest_encoding": "UTF-8 source_table\\tdecimal_source_row_number\\n sorted by source_table,source_row_number",
                "identity_sha256": row["identity_sha256"],
                "rows": row["target_rows"],
                "users": row["users"],
                "positives": row["positives"],
                "negatives": row["negatives"],
                "role": "logical_source_identity_prototype_not_gold_sample_id_manifest",
            }
            for row in fixed
        ],
        "candidate_mapping": {
            "name_rule": "case_insensitive upload_type token picture|photo|album",
            "proxy_v1": {
                "picture_like": "name_flag AND videos_basic duration missing/nonpositive",
                "video_like": "NOT name_flag AND videos_basic duration positive finite",
                "unknown": "all remaining conflicts or uncertain rows",
            },
            "status": "diagnostic_proxy_v1_frozen_not_official_modality_predictor_use_forbidden",
            "label_used_in_mapping_definition": False,
        },
        "bootstrap_plan": bootstrap_plan,
        "environment": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threads": THREADS,
            "accelerator_used": False,
        },
        "script": {
            "path": relative(SCRIPT_PATH),
            "size_bytes": SCRIPT_PATH.stat().st_size,
            "sha256": script_hash,
        },
        "outputs": [
            {"path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in output_paths
        ],
        "notes": [
            "No recursive glob or configurable data path is used.",
            "All four allowlisted inputs receive full size and SHA-256 verification; users.parquet is not queried.",
            "Burn-in candidates are sensitivity and rolling-origin evidence, not primary target filters.",
            "Candidate proxy v1 is diagnostic-only and is not an official true-modality mapping.",
            "Bootstrap output is a label-rate implementation smoke test, not a paired model contrast.",
            "The manifest intentionally excludes its own hash.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = OUTPUT_DIR / "design_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "target_rows": invariants["target_rows"],
                "outputs": len(output_paths) + 1,
                "elapsed_seconds": manifest["elapsed_seconds"],
                "manifest": relative(manifest_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
