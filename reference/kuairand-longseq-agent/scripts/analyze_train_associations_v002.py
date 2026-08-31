from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated" / "train_association_v002"
RELEASE_REPORT_PATH = PROJECT_ROOT / "reports" / "analysis" / "train_association_report_v002.md"
QUICK_OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated" / "train_association_v002_quick"
QUICK_REPORT_PATH = QUICK_OUTPUT_DIR / "train_association_quick_report.md"
QUICK_DEFAULT_THREADS = min(8, max(1, os.cpu_count() or 1))

TRAIN_START = "2022-04-08"
TRAIN_END = "2022-04-17"
EXPECTED_ALL_TAB_ROWS = 3_671_408
EXPECTED_TARGET_ROWS = 2_399_844
EXPECTED_TARGET_USERS = 950
EXPECTED_TARGET_VIDEOS = 974_550
EXPECTED_TARGET_POSITIVES = 765_417
EXPECTED_ADDITIVE_ALL_ROWS = 16_990
EXPECTED_ADDITIVE_ALL_POSITIVES = 16_989
EXPECTED_ADDITIVE_TARGET_ROWS = 14_070
EXPECTED_ADDITIVE_TARGET_POSITIVES = 14_069

INPUTS = {
    "early_train": PROJECT_ROOT / "data" / "silver" / "events_early_standard.parquet",
    "formula_mismatch": PROJECT_ROOT
    / "data"
    / "quarantine"
    / "label_formula_mismatch_rows.parquet",
    "users": PROJECT_ROOT / "data" / "silver" / "users.parquet",
    "videos_basic": PROJECT_ROOT / "data" / "silver" / "videos_basic.parquet",
}
METADATA_FILES = {
    "silver_run_manifest": PROJECT_ROOT / "data" / "manifests" / "silver_run_manifest.json",
    "silver_output_manifest": PROJECT_ROOT
    / "data"
    / "manifests"
    / "silver_output_manifest.json",
    "experiment_contract": PROJECT_ROOT / "configs" / "experiment_v002_proposal.yaml",
    "feature_availability": PROJECT_ROOT / "configs" / "feature_availability_v002.yaml",
    "hypothesis_registry": PROJECT_ROOT / "configs" / "hypothesis_registry_v002.yaml",
}


@dataclass(frozen=True)
class RunConfig:
    mode: str
    threads: int
    output_dir: Path
    report_path: Path
    full_input_sha256: bool
    render_chart: bool
    checkpoint_eligible: bool

    @property
    def verification_level(self) -> str:
        return "full_sha256" if self.full_input_sha256 else "manifest_membership_and_size"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("threads must be >= 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Run KuaiRand v002 Train-only descriptive associations."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Fast, non-checkpoint iteration: 8 CPU threads by default, size checks, isolated outputs.",
    )
    mode.add_argument(
        "--release",
        action="store_true",
        help="Checkpoint-eligible release: one CPU thread, full input SHA-256, chart and canonical report.",
    )
    parser.add_argument(
        "--threads",
        type=positive_int,
        help="Quick-mode DuckDB thread count; defaults to min(8, logical CPUs).",
    )
    parser.add_argument(
        "--with-chart",
        action="store_true",
        help="Also render a chart in quick mode; release always renders it.",
    )
    args = parser.parse_args(argv)

    if args.release:
        if args.threads is not None:
            parser.error("--threads is only available with --quick; release is fixed to one thread")
        return RunConfig(
            mode="release",
            threads=1,
            output_dir=RELEASE_OUTPUT_DIR,
            report_path=RELEASE_REPORT_PATH,
            full_input_sha256=True,
            render_chart=True,
            checkpoint_eligible=True,
        )

    return RunConfig(
        mode="quick",
        threads=args.threads or QUICK_DEFAULT_THREADS,
        output_dir=QUICK_OUTPUT_DIR,
        report_path=QUICK_REPORT_PATH,
        full_input_sha256=False,
        render_chart=args.with_chart,
        checkpoint_eligible=False,
    )


def accelerator_probe() -> dict[str, Any]:
    cudf_available = importlib.util.find_spec("cudf") is not None
    cudf_polars_available = importlib.util.find_spec("cudf_polars") is not None
    return {
        "selected_backend": "duckdb_cpu",
        "accelerator_used": False,
        "gpu_used": False,
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "cudf_available": cudf_available,
        "cudf_polars_available": cudf_polars_available,
        "gpu_dataframe_backend_available": cudf_available and cudf_polars_available,
        "decision": "current SQL pipeline runs on DuckDB CPU; no transparent GPU claim",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sql_literal(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def rows_as_dicts(cursor: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def query_dicts(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    return rows_as_dicts(con.execute(sql))


def one_row(con: duckdb.DuckDBPyConnection, sql: str) -> dict[str, Any]:
    rows = query_dicts(con, sql)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row, received {len(rows)}")
    return rows[0]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def assert_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{name}: expected {expected!r}, observed {actual!r}")


def verify_inputs(full_sha256: bool) -> list[dict[str, Any]]:
    for path in [*INPUTS.values(), *METADATA_FILES.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(METADATA_FILES["silver_output_manifest"].read_text(encoding="utf-8"))
    assert_equal(manifest.get("run_id"), "silver-20260814-155536", "formal Silver run_id")
    entries = {str(Path(entry["path"]).resolve()).casefold(): entry for entry in manifest["files"]}

    verified: list[dict[str, Any]] = []
    for role, path in INPUTS.items():
        key = str(path.resolve()).casefold()
        if key not in entries:
            raise RuntimeError(f"Input is absent from formal output manifest: {path}")
        entry = entries[key]
        actual_size = path.stat().st_size
        assert_equal(actual_size, entry["size_bytes"], f"{role} size")
        actual_hash = sha256_file(path) if full_sha256 else None
        if full_sha256:
            assert_equal(actual_hash, entry["sha256"], f"{role} SHA-256")
        verified.append(
            {
                "role": role,
                "path": relative(path),
                "size_bytes": actual_size,
                "expected_sha256": entry["sha256"],
                "observed_sha256": actual_hash,
                "formal_manifest_entry_found": True,
                "size_verified_this_run": True,
                "sha256_verified_this_run": full_sha256,
            }
        )
    return verified


def create_analysis_tables(con: duckdb.DuckDBPyConnection) -> None:
    early = sql_literal(INPUTS["early_train"])
    mismatch = sql_literal(INPUTS["formula_mismatch"])
    users = sql_literal(INPUTS["users"])
    videos = sql_literal(INPUTS["videos_basic"])

    con.execute(
        f"""
        CREATE TEMP TABLE canonical_all AS
        SELECT
            source_table, source_row_number, user_id, video_id, event_date,
            time_ms, tab, long_view, duration_ms, exclusion_reason
        FROM read_parquet({early})
        WHERE event_date BETWEEN DATE '{TRAIN_START}' AND DATE '{TRAIN_END}'
        UNION ALL
        SELECT
            source_table, source_row_number, user_id, video_id, event_date,
            time_ms, tab, long_view, duration_ms, exclusion_reason
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
        SELECT
            user_id,
            time_ms,
            count(*)::BIGINT AS batch_event_n,
            sum(long_view)::BIGINT AS batch_positive_n
        FROM canonical_all
        GROUP BY user_id, time_ms
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE history_state AS
        SELECT
            user_id,
            time_ms,
            row_number() OVER (
                PARTITION BY user_id ORDER BY time_ms
            ) - 1 AS prior_history_batch_count,
            sum(batch_event_n) OVER (
                PARTITION BY user_id ORDER BY time_ms
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS prior_event_n,
            sum(batch_positive_n) OVER (
                PARTITION BY user_id ORDER BY time_ms
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS prior_positive_n,
            sum(batch_event_n) OVER (
                PARTITION BY user_id ORDER BY time_ms
                ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
            ) AS recent10_event_n,
            sum(batch_positive_n) OVER (
                PARTITION BY user_id ORDER BY time_ms
                ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
            ) AS recent10_positive_n
        FROM history_batch
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE scored AS
        SELECT
            t.source_table,
            t.source_row_number,
            t.user_id,
            t.video_id,
            t.event_date,
            t.time_ms,
            t.long_view,
            t.duration_ms AS event_duration_ms,
            coalesce(h.prior_history_batch_count, 0)::BIGINT AS prior_history_batch_count,
            h.prior_event_n,
            h.prior_positive_n,
            CASE WHEN h.prior_event_n > 0
                 THEN h.prior_positive_n::DOUBLE / h.prior_event_n END AS prior_user_lv_rate,
            h.recent10_event_n,
            h.recent10_positive_n,
            CASE WHEN h.recent10_event_n > 0
                 THEN h.recent10_positive_n::DOUBLE / h.recent10_event_n END AS recent10_lv_rate,
            u.user_active_degree,
            v.video_duration,
            v.upload_type,
            v.upload_dt,
            CASE WHEN v.upload_dt IS NOT NULL AND v.upload_dt <= t.event_date
                 THEN date_diff('day', v.upload_dt, t.event_date)::DOUBLE END AS upload_age_days,
            (u.user_id IS NULL) AS user_join_missing,
            (v.video_id IS NULL) AS video_join_missing
        FROM target t
        LEFT JOIN read_parquet({users}) u USING (user_id)
        LEFT JOIN read_parquet({videos}) v USING (video_id)
        LEFT JOIN history_state h USING (user_id, time_ms)
        """
    )


def validate_analysis_tables(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    counts = one_row(
        con,
        """
        SELECT
            (SELECT count(*) FROM canonical_all) AS all_tab_rows,
            (SELECT count(*) FROM target) AS target_rows,
            (SELECT count(DISTINCT user_id) FROM target) AS target_users,
            (SELECT count(DISTINCT video_id) FROM target) AS target_videos,
            (SELECT sum(long_view) FROM target) AS target_positives,
            (SELECT count(*) FROM canonical_all
             WHERE exclusion_reason='LONG_VIEW_FORMULA_MISMATCH') AS additive_all_rows,
            (SELECT sum(long_view) FROM canonical_all
             WHERE exclusion_reason='LONG_VIEW_FORMULA_MISMATCH') AS additive_all_positives,
            (SELECT count(*) FROM target
             WHERE exclusion_reason='LONG_VIEW_FORMULA_MISMATCH') AS additive_target_rows,
            (SELECT sum(long_view) FROM target
             WHERE exclusion_reason='LONG_VIEW_FORMULA_MISMATCH') AS additive_target_positives,
            (SELECT count(*) FROM (
                SELECT source_table, source_row_number
                FROM canonical_all GROUP BY ALL HAVING count(*) > 1
             )) AS duplicate_source_identities,
            (SELECT count(*) FROM target WHERE tab <> 1 OR tab IS NULL) AS non_target_tab_rows,
            (SELECT count(*) FROM target
             WHERE event_date < DATE '2022-04-08' OR event_date > DATE '2022-04-17') AS out_of_range_rows,
            (SELECT count(*) FROM target WHERE long_view NOT IN (0,1) OR long_view IS NULL) AS invalid_labels,
            (SELECT sum(user_join_missing::INT) FROM scored) AS missing_user_joins,
            (SELECT sum(video_join_missing::INT) FROM scored) AS missing_video_joins
        """,
    )
    expected = {
        "all_tab_rows": EXPECTED_ALL_TAB_ROWS,
        "target_rows": EXPECTED_TARGET_ROWS,
        "target_users": EXPECTED_TARGET_USERS,
        "target_videos": EXPECTED_TARGET_VIDEOS,
        "target_positives": EXPECTED_TARGET_POSITIVES,
        "additive_all_rows": EXPECTED_ADDITIVE_ALL_ROWS,
        "additive_all_positives": EXPECTED_ADDITIVE_ALL_POSITIVES,
        "additive_target_rows": EXPECTED_ADDITIVE_TARGET_ROWS,
        "additive_target_positives": EXPECTED_ADDITIVE_TARGET_POSITIVES,
        "duplicate_source_identities": 0,
        "non_target_tab_rows": 0,
        "out_of_range_rows": 0,
        "invalid_labels": 0,
        "missing_user_joins": 0,
        "missing_video_joins": 0,
    }
    for key, expected_value in expected.items():
        assert_equal(counts[key], expected_value, key)
    return counts


def canonical_summary(con: duckdb.DuckDBPyConnection, checks: dict[str, Any]) -> list[dict[str, Any]]:
    extra = one_row(
        con,
        """
        WITH per_user AS (
            SELECT user_id, avg(long_view)::DOUBLE AS label_rate
            FROM target GROUP BY user_id
        ), target_batch AS (
            SELECT user_id, time_ms, count(*) AS n,
                   sum(long_view) AS positives
            FROM target GROUP BY user_id, time_ms
        ), all_tab_batch AS (
            SELECT user_id, time_ms, count(DISTINCT tab) AS tab_n
            FROM canonical_all GROUP BY user_id, time_ms
        )
        SELECT
            avg(long_view)::DOUBLE AS event_micro_label_rate,
            (SELECT avg(label_rate) FROM per_user) AS user_macro_label_rate,
            (SELECT count(*) FROM target_batch) AS timestamp_groups,
            (SELECT count(*) FROM target_batch WHERE positives > 0 AND positives < n) AS mixed_timestamp_groups,
            (SELECT coalesce(sum(n),0) FROM target_batch
             WHERE positives > 0 AND positives < n) AS mixed_timestamp_rows,
            (SELECT count(*) FROM target_batch WHERE positives=0 OR positives=n) AS homogeneous_timestamp_groups,
            (SELECT coalesce(sum(n),0) FROM target_batch
             WHERE positives=0 OR positives=n) AS homogeneous_timestamp_rows,
            (SELECT coalesce(sum(positives * (n-positives)),0) FROM target_batch
             WHERE positives > 0 AND positives < n) AS timestamp_discordant_pairs,
            (SELECT max(n) FROM target_batch) AS max_timestamp_group_size,
            (SELECT count(*) FROM all_tab_batch WHERE tab_n > 1) AS all_cross_tab_timestamp_groups,
            (SELECT count(*)
             FROM target_batch t
             JOIN all_tab_batch a USING (user_id, time_ms)
             WHERE a.tab_n > 1) AS target_cross_tab_timestamp_groups
        FROM target
        """,
    )
    ordered = [
        ("all_tab_train_rows", checks["all_tab_rows"]),
        ("target_tab1_rows", checks["target_rows"]),
        ("target_users", checks["target_users"]),
        ("target_videos", checks["target_videos"]),
        ("target_positives", checks["target_positives"]),
        ("event_micro_label_rate", extra["event_micro_label_rate"]),
        ("user_macro_label_rate", extra["user_macro_label_rate"]),
        ("additive_mismatch_all_tab_rows", checks["additive_all_rows"]),
        ("additive_mismatch_all_tab_positives", checks["additive_all_positives"]),
        ("additive_mismatch_target_rows", checks["additive_target_rows"]),
        ("additive_mismatch_target_positives", checks["additive_target_positives"]),
        ("timestamp_groups", extra["timestamp_groups"]),
        ("mixed_timestamp_groups", extra["mixed_timestamp_groups"]),
        ("mixed_timestamp_rows", extra["mixed_timestamp_rows"]),
        ("homogeneous_timestamp_groups", extra["homogeneous_timestamp_groups"]),
        ("homogeneous_timestamp_rows", extra["homogeneous_timestamp_rows"]),
        ("timestamp_discordant_pairs", extra["timestamp_discordant_pairs"]),
        ("max_timestamp_group_size", extra["max_timestamp_group_size"]),
        ("all_cross_tab_timestamp_groups", extra["all_cross_tab_timestamp_groups"]),
        ("target_cross_tab_timestamp_groups", extra["target_cross_tab_timestamp_groups"]),
        ("missing_user_joins", checks["missing_user_joins"]),
        ("missing_video_joins", checks["missing_video_joins"]),
    ]
    return [{"metric": key, "value": value} for key, value in ordered]


def daily_summary(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    return query_dicts(
        con,
        """
        SELECT
            event_date::VARCHAR AS event_date,
            count(*)::BIGINT AS rows,
            count(DISTINCT user_id)::BIGINT AS users,
            count(DISTINCT video_id)::BIGINT AS videos,
            sum(long_view)::BIGINT AS positives,
            avg(long_view)::DOUBLE AS label_rate
        FROM scored GROUP BY event_date ORDER BY event_date
        """,
    )


def history_summaries(
    con: duckdb.DuckDBPyConnection,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case = """
        CASE
          WHEN prior_history_batch_count = 0 THEN '0'
          WHEN prior_history_batch_count < 10 THEN '1-9'
          WHEN prior_history_batch_count < 50 THEN '10-49'
          WHEN prior_history_batch_count < 200 THEN '50-199'
          WHEN prior_history_batch_count < 500 THEN '200-499'
          ELSE '500+'
        END
    """
    ordering = "CASE history_depth WHEN '0' THEN 1 WHEN '1-9' THEN 2 WHEN '10-49' THEN 3 WHEN '50-199' THEN 4 WHEN '200-499' THEN 5 ELSE 6 END"
    depth = query_dicts(
        con,
        f"""
        SELECT * FROM (
            SELECT
                {case} AS history_depth,
                count(*)::BIGINT AS rows,
                count(DISTINCT user_id)::BIGINT AS users,
                sum(long_view)::BIGINT AS positives,
                avg(long_view)::DOUBLE AS label_rate
            FROM scored GROUP BY 1
        ) ORDER BY {ordering}
        """,
    )
    coverage = query_dicts(
        con,
        """
        SELECT
            event_date::VARCHAR AS event_date,
            count(*)::BIGINT AS rows,
            avg((prior_history_batch_count >= 50)::INT)::DOUBLE AS share_history_50_plus,
            avg((prior_history_batch_count >= 200)::INT)::DOUBLE AS share_history_200_plus,
            avg((prior_history_batch_count >= 500)::INT)::DOUBLE AS share_history_500_plus
        FROM scored GROUP BY event_date ORDER BY event_date
        """,
    )
    return depth, coverage


def dimension_summary(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    duration_case = """
        CASE
          WHEN video_duration IS NULL OR video_duration <= 0 THEN 'UNKNOWN_OR_NONPOSITIVE'
          WHEN video_duration < 10000 THEN '0-10s'
          WHEN video_duration < 20000 THEN '10-20s'
          WHEN video_duration < 40000 THEN '20-40s'
          WHEN video_duration < 60000 THEN '40-60s'
          WHEN video_duration < 120000 THEN '60-120s'
          ELSE '120s+'
        END
    """
    rows = query_dicts(
        con,
        f"""
        SELECT 'duration_bucket' AS dimension, {duration_case} AS level,
               count(*)::BIGINT AS rows, count(DISTINCT user_id)::BIGINT AS users,
               sum(long_view)::BIGINT AS positives, avg(long_view)::DOUBLE AS label_rate,
               'videos_basic.video_duration; descriptive only' AS note
        FROM scored GROUP BY 2
        UNION ALL
        SELECT 'upload_type', coalesce(upload_type,'UNKNOWN'),
               count(*)::BIGINT, count(DISTINCT user_id)::BIGINT,
               sum(long_view)::BIGINT, avg(long_view)::DOUBLE,
               'source category; not an approved modality mapping'
        FROM scored GROUP BY 2
        UNION ALL
        SELECT 'user_active_degree', coalesce(user_active_degree,'UNKNOWN'),
               count(*)::BIGINT, count(DISTINCT user_id)::BIGINT,
               sum(long_view)::BIGINT, avg(long_view)::DOUBLE,
               'diagnostic only: snapshot cutoff is unproven'
        FROM scored GROUP BY 2
        """,
    )
    dimension_order = {"duration_bucket": 1, "upload_type": 2, "user_active_degree": 3}
    duration_order = {
        "UNKNOWN_OR_NONPOSITIVE": 0,
        "0-10s": 1,
        "10-20s": 2,
        "20-40s": 3,
        "40-60s": 4,
        "60-120s": 5,
        "120s+": 6,
    }
    rows.sort(
        key=lambda row: (
            dimension_order[row["dimension"]],
            duration_order.get(row["level"], -int(row["rows"])),
            row["level"],
        )
    )
    return rows


def duration_audit(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    return one_row(
        con,
        """
        SELECT
            sum((event_duration_ms IS NULL OR event_duration_ms <= 0)::INT)::BIGINT AS event_nonpositive_rows,
            sum((video_duration IS NULL OR video_duration <= 0)::INT)::BIGINT AS video_nonpositive_rows,
            sum((event_duration_ms > 0 AND video_duration > 0)::INT)::BIGINT AS both_positive_rows,
            sum((event_duration_ms > 0 AND video_duration > 0
                 AND abs(event_duration_ms-video_duration) > 0.5)::INT)::BIGINT AS positive_disagreement_rows,
            sum((event_duration_ms <= 0 AND video_duration > 0)::INT)::BIGINT AS event_nonpositive_video_positive_rows,
            sum((upload_dt IS NOT NULL AND upload_dt > event_date)::INT)::BIGINT AS future_upload_date_rows
        FROM scored
        """,
    )


def auc_metrics_for_feature(
    con: duckdb.DuckDBPyConnection,
    feature_name: str,
    score_expression: str,
    eligibility_expression: str,
    interpretation: str,
) -> list[dict[str, Any]]:
    con.execute("DROP TABLE IF EXISTS feature_score")
    con.execute(
        f"""
        CREATE TEMP TABLE feature_score AS
        SELECT user_id, time_ms, long_view::INTEGER AS y,
               ({score_expression})::DOUBLE AS score
        FROM scored
        WHERE {eligibility_expression}
          AND isfinite(({score_expression})::DOUBLE)
        """
    )
    total = one_row(
        con,
        "SELECT count(*)::BIGINT AS rows, sum(y)::BIGINT AS positives FROM feature_score",
    )
    if total["positives"] in (0, total["rows"]):
        raise RuntimeError(f"Feature {feature_name} has no two-class AUC population")

    pooled = one_row(
        con,
        """
        WITH ranked AS (
          SELECT y,
                 rank() OVER (ORDER BY score)
                 + (count(*) OVER (PARTITION BY score)-1)/2.0 AS average_rank
          FROM feature_score
        ), agg AS (
          SELECT count(*)::DOUBLE AS n,
                 sum(y)::DOUBLE AS n_pos,
                 sum(CASE WHEN y=1 THEN average_rank ELSE 0 END)::DOUBLE AS positive_rank_sum
          FROM ranked
        )
        SELECT
          (positive_rank_sum - n_pos*(n_pos+1)/2.0) / (n_pos*(n-n_pos)) AS auc
        FROM agg
        """,
    )
    user = one_row(
        con,
        """
        WITH ranked AS (
          SELECT user_id, y,
                 rank() OVER (PARTITION BY user_id ORDER BY score)
                 + (count(*) OVER (PARTITION BY user_id,score)-1)/2.0 AS average_rank
          FROM feature_score
        ), grouped AS (
          SELECT user_id, count(*)::DOUBLE AS n, sum(y)::DOUBLE AS n_pos,
                 sum(CASE WHEN y=1 THEN average_rank ELSE 0 END)::DOUBLE AS positive_rank_sum
          FROM ranked GROUP BY user_id
        ), eligible AS (
          SELECT *, n-n_pos AS n_neg,
                 (positive_rank_sum - n_pos*(n_pos+1)/2.0)/(n_pos*(n-n_pos)) AS auc
          FROM grouped WHERE n_pos > 0 AND n_pos < n
        )
        SELECT count(*)::BIGINT AS groups, sum(n)::BIGINT AS rows,
               sum(n_pos)::BIGINT AS positives, sum(n_neg)::BIGINT AS negatives,
               sum(auc*n)/sum(n) AS event_weighted_auc,
               avg(auc) AS group_equal_auc,
               sum(auc*n_pos*n_neg)/sum(n_pos*n_neg) AS pair_weighted_auc
        FROM eligible
        """,
    )
    timestamp = one_row(
        con,
        """
        WITH ranked AS (
          SELECT user_id, time_ms, y,
                 rank() OVER (PARTITION BY user_id,time_ms ORDER BY score)
                 + (count(*) OVER (PARTITION BY user_id,time_ms,score)-1)/2.0 AS average_rank
          FROM feature_score
        ), grouped AS (
          SELECT user_id, time_ms, count(*)::DOUBLE AS n, sum(y)::DOUBLE AS n_pos,
                 sum(CASE WHEN y=1 THEN average_rank ELSE 0 END)::DOUBLE AS positive_rank_sum
          FROM ranked GROUP BY user_id,time_ms
        ), eligible AS (
          SELECT *, n-n_pos AS n_neg,
                 (positive_rank_sum - n_pos*(n_pos+1)/2.0)/(n_pos*(n-n_pos)) AS auc
          FROM grouped WHERE n_pos > 0 AND n_pos < n
        )
        SELECT count(*)::BIGINT AS groups, coalesce(sum(n),0)::BIGINT AS rows,
               coalesce(sum(n_pos),0)::BIGINT AS positives,
               coalesce(sum(n_neg),0)::BIGINT AS negatives,
               sum(auc*n_pos*n_neg)/sum(n_pos*n_neg) AS pair_weighted_auc,
               avg(auc) AS group_equal_auc
        FROM eligible
        """,
    )
    coverage = total["rows"] / EXPECTED_TARGET_ROWS
    common = {
        "feature": feature_name,
        "scored_rows": total["rows"],
        "coverage": coverage,
        "interpretation": interpretation,
    }
    return [
        {
            **common,
            "metric": "pooled_auc",
            "value": pooled["auc"],
            "eligible_groups": 1,
            "eligible_rows": total["rows"],
            "weighting": "all scored events",
        },
        {
            **common,
            "metric": "user_gauc_event_weighted",
            "value": user["event_weighted_auc"],
            "eligible_groups": user["groups"],
            "eligible_rows": user["rows"],
            "weighting": "eligible-user AUC weighted by scored events",
        },
        {
            **common,
            "metric": "user_auc_group_equal",
            "value": user["group_equal_auc"],
            "eligible_groups": user["groups"],
            "eligible_rows": user["rows"],
            "weighting": "eligible users equally weighted",
        },
        {
            **common,
            "metric": "user_auc_pair_weighted",
            "value": user["pair_weighted_auc"],
            "eligible_groups": user["groups"],
            "eligible_rows": user["rows"],
            "weighting": "eligible users weighted by positive-negative pairs",
        },
        {
            **common,
            "metric": "within_timestamp_pair_auc",
            "value": timestamp["pair_weighted_auc"],
            "eligible_groups": timestamp["groups"],
            "eligible_rows": timestamp["rows"],
            "weighting": "mixed-label timestamp groups; discordant-pair weighted",
        },
        {
            **common,
            "metric": "within_timestamp_group_equal_auc",
            "value": timestamp["group_equal_auc"],
            "eligible_groups": timestamp["groups"],
            "eligible_rows": timestamp["rows"],
            "weighting": "mixed-label timestamp groups equally weighted",
        },
    ]


def metric_summary(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    specs = [
        (
            "prior_user_lv_rate",
            "prior_user_lv_rate",
            "prior_user_lv_rate IS NOT NULL",
            "strict prior all-tab cumulative event rate; constant inside a user-timestamp batch",
        ),
        (
            "recent10_batch_lv_rate",
            "recent10_lv_rate",
            "recent10_lv_rate IS NOT NULL",
            "strict prior ten completed timestamp batches; constant inside a user-timestamp batch",
        ),
        (
            "prior_history_batch_count",
            "prior_history_batch_count",
            "prior_history_batch_count IS NOT NULL",
            "observed history depth; strongly confounded with calendar time and activity",
        ),
        (
            "video_duration_seconds",
            "video_duration/1000.0",
            "video_duration > 0",
            "raw monotone score only; nonlinear buckets and interactions remain possible",
        ),
        (
            "upload_age_days",
            "upload_age_days",
            "upload_age_days IS NOT NULL AND upload_age_days >= 0",
            "intrinsic upload-age score where upload date is not after the event date",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for feature, expression, eligibility, interpretation in specs:
        rows.extend(
            auc_metrics_for_feature(con, feature, expression, eligibility, interpretation)
        )
    return rows


def fmt_int(value: Any) -> str:
    return f"{int(value):,}"


def fmt_rate(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def fmt_auc(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{float(value):.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_chart(
    daily: list[dict[str, Any]],
    history: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    dimensions: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    colors = {"pooled": "#4C78A8", "user": "#F58518", "timestamp": "#54A24B"}

    ax = axes[0, 0]
    dates = [row["event_date"][5:] for row in daily]
    rates = [100 * row["label_rate"] for row in daily]
    ax.plot(dates, rates, marker="o", linewidth=2.4, color="#4C78A8")
    ax.set_title("A. Daily official long_view rate")
    ax.set_ylabel("Positive rate (%)")
    ax.tick_params(axis="x", rotation=35)

    ax = axes[0, 1]
    depth_names = [row["history_depth"] for row in history]
    depth_rates = [100 * row["label_rate"] for row in history]
    ax.bar(depth_names, depth_rates, color="#ECA82C")
    ax.set_title("B. Rate by observed prior history depth")
    ax.set_ylabel("Positive rate (%)")
    ax.text(
        0.98,
        0.97,
        "Descriptive only: depth co-moves with date/activity",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )

    ax = axes[1, 0]
    feature_order = [
        "prior_user_lv_rate",
        "recent10_batch_lv_rate",
        "prior_history_batch_count",
        "video_duration_seconds",
        "upload_age_days",
    ]
    metric_lookup = {(row["feature"], row["metric"]): row["value"] for row in metrics}
    x = list(range(len(feature_order)))
    width = 0.24
    series = [
        ("Pooled", "pooled_auc", colors["pooled"], -width),
        ("User GAUC", "user_gauc_event_weighted", colors["user"], 0),
        ("Within-time", "within_timestamp_pair_auc", colors["timestamp"], width),
    ]
    for label, metric, color, offset in series:
        values = [metric_lookup.get((feature, metric), math.nan) for feature in feature_order]
        ax.bar([value + offset for value in x], values, width=width, label=label, color=color)
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.set_ylim(0.35, 0.76)
    ax.set_xticks(x)
    ax.set_xticklabels(["prior rate", "recent-10", "history depth", "duration", "upload age"], rotation=25, ha="right")
    ax.set_title("C. Same score, three association lenses")
    ax.set_ylabel("Descriptive AUC")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    duration = [row for row in dimensions if row["dimension"] == "duration_bucket"]
    duration_labels = [
        "unknown/nonpos" if row["level"] == "UNKNOWN_OR_NONPOSITIVE" else row["level"]
        for row in duration
    ]
    ax.bar(duration_labels, [100 * row["label_rate"] for row in duration], color="#B279A2")
    ax.set_title("D. Rate by videos_basic duration state")
    ax.set_ylabel("Positive rate (%)")
    ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "KuaiRand-1K v002 — Train-only descriptive associations (not model evaluation)",
        fontsize=15,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor="white")
    plt.close(fig)


def write_report(
    summary: list[dict[str, Any]],
    daily: list[dict[str, Any]],
    history: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    dimensions: list[dict[str, Any]],
    duration: dict[str, Any],
    config: RunConfig,
    artifact_paths: list[Path],
) -> None:
    values = {row["metric"]: row["value"] for row in summary}
    metric_lookup = {(row["feature"], row["metric"]): row for row in metrics}
    metric_rows = []
    for feature in [
        "prior_user_lv_rate",
        "recent10_batch_lv_rate",
        "prior_history_batch_count",
        "video_duration_seconds",
        "upload_age_days",
    ]:
        pooled = metric_lookup[(feature, "pooled_auc")]
        user = metric_lookup[(feature, "user_gauc_event_weighted")]
        timestamp = metric_lookup[(feature, "within_timestamp_pair_auc")]
        metric_rows.append(
            [
                feature,
                fmt_auc(pooled["value"]),
                fmt_auc(user["value"]),
                fmt_auc(timestamp["value"]),
                fmt_rate(pooled["coverage"]),
                fmt_int(user["eligible_groups"]),
                fmt_int(timestamp["eligible_groups"]),
            ]
        )

    duration_rows = [row for row in dimensions if row["dimension"] == "duration_bucket"]
    activity_rows = [row for row in dimensions if row["dimension"] == "user_active_degree"]
    upload_rows = sorted(
        [row for row in dimensions if row["dimension"] == "upload_type"],
        key=lambda row: row["rows"],
        reverse=True,
    )[:10]

    if config.mode == "release":
        title = "# KuaiRand-1K v002 Train-only 描述性关联报告"
        status_line = "首轮确定性 Train-only release 已完成；不是正式模型训练、Validation 或测试报告"
        mode_line = "release：完整输入 SHA-256、单线程确定性执行、正式图表与 canonical manifest"
        command = "python scripts/analyze_train_associations_v002.py --release"
    else:
        title = "# KuaiRand-1K v002 Train-only QUICK 描述性关联报告"
        status_line = "QUICK 迭代已完成；不可作为 checkpoint、Gate、正式训练或 Validation 证据"
        mode_line = (
            f"quick：{config.threads} 个 CPU 线程、仅 manifest 成员与文件大小校验、独立输出；"
            "未执行大文件 SHA-256"
        )
        chart_flag = " --with-chart" if config.render_chart else ""
        command = (
            "python scripts/analyze_train_associations_v002.py "
            f"--quick --threads {config.threads}{chart_flag}"
        )
    artifact_lines = "\n".join(f"- {relative(path)}" for path in artifact_paths)

    report = f"""{title}

> 状态：{status_line}  
> 运行档位：{mode_line}  
> 数据边界：canonical source-Train，2022-04-08 至 2022-04-17；目标 tab=1；历史使用同范围全 tab 且严格 time < target time  
> 标签：官方 long_view；正式 Silver 与同源 exclusive formula-mismatch 行做 UNION ALL  
> 正式 Silver run：silver-20260814-155536

## 1. 这次实际做了什么

本轮把 Silver 的事件、用户键和视频键组织成一个只读的研究测量面，检查入口规模、维表连接、时间覆盖、严格先验历史、内容元数据以及三种不同关联口径。没有重清洗 Silver，没有构建 Gold，没有训练候选模型，也没有读取 late、random、Validation 或 statistic 文件。

三种口径不能互换：

- pooled AUC 描述跨全部事件的分离，可能主要来自不同用户的基础率；
- user-GAUC 描述同一用户内部随时间/内容变化的分离；
- within-timestamp pair AUC 只比较同一 user_id、time_ms 的混合标签候选，不能称为真实 impression/slate。

## 2. Canonical Train 入口与硬校验

{markdown_table(
        ["项目", "结果"],
        [
            ["全 tab 历史事件", fmt_int(values['all_tab_train_rows'])],
            ["tab=1 目标事件", fmt_int(values['target_tab1_rows'])],
            ["用户 / 视频", f"{fmt_int(values['target_users'])} / {fmt_int(values['target_videos'])}"],
            ["正例 / event-micro 率", f"{fmt_int(values['target_positives'])} / {fmt_rate(values['event_micro_label_rate'])}"],
            ["user-macro 平均率", fmt_rate(values['user_macro_label_rate'])],
            ["加回 mismatch（目标 tab=1）", f"{fmt_int(values['additive_mismatch_target_rows'])} 行；{fmt_int(values['additive_mismatch_target_positives'])} 正例"],
            ["user / video 缺失连接", f"{fmt_int(values['missing_user_joins'])} / {fmt_int(values['missing_video_joins'])}"],
        ],
    )}

入口断言全部通过：source identity 无重复、日期和 tab 无越界、标签为二元、用户和视频连接缺失均为 0。event-micro 率与 user-macro 平均率相差 {fmt_rate(values['user_macro_label_rate'] - values['event_micro_label_rate'])}，说明活跃用户权重会显著改变总体结论。

## 3. 日期与历史覆盖

{markdown_table(
        ["日期", "事件", "用户", "正例率"],
        [[row['event_date'], fmt_int(row['rows']), fmt_int(row['users']), fmt_rate(row['label_rate'])] for row in daily],
    )}

10 天内标签率并不恒定；因此后续基线必须做 Train rolling-origin，并显式控制日期。以下历史深度只是观察到的 04-08 起始窗口，不是用户的完整生命周期历史。

{markdown_table(
        ["严格先验批次数", "事件", "用户", "正例率"],
        [[row['history_depth'], fmt_int(row['rows']), fmt_int(row['users']), fmt_rate(row['label_rate'])] for row in history],
    )}

历史越深时原始正例率下降，不能解释成“长历史导致偏好变弱”：历史深度同时随日期推进，并与用户活跃度和曝光量共同变化。每日覆盖详见 generated CSV；calendar burn-in 仍未冻结。

## 4. 同一分数在三种口径下的关联

{markdown_table(
        ["分数", "pooled AUC", "user-GAUC", "同时间戳 pair AUC", "覆盖", "可算用户", "可算时间戳组"],
        metric_rows,
    )}

解释边界：

1. 累积用户历史率在 pooled 口径很强，但 user-GAUC 接近随机，说明大部分分离来自用户基础率。recent-10 历史在用户内更有信息，但它在同一时间戳内是常数，因此不能区分该批候选。
2. history depth 与日期/活跃度共变，它的单变量 AUC 不能作为历史窗口选择依据。
3. 原始时长和上传年龄只是单调分数；接近或低于 0.5 不等于字段无用，仍可能存在非线性、交互或校准价值。
4. 任何单变量 AUC 均不用于硬筛字段。真正的增量要在固定目标行、冻结基线和 rolling-origin 比较中判断。

同时间戳诊断覆盖 {fmt_int(values['mixed_timestamp_groups'])} 个混合标签组、{fmt_int(values['mixed_timestamp_rows'])} 行；另有 {fmt_int(values['homogeneous_timestamp_groups'])} 个同质组不会进入 pair AUC。在 tab=1 目标组中，有 {fmt_int(values['target_cross_tab_timestamp_groups'])} 个 user-time 组在全 tab 数据里还包含其他 tab，再次说明 user_id+time_ms 不是已验证的曝光 slate。

## 5. 内容和用户维度的异质性

### 5.1 视频时长状态

{markdown_table(
        ["时长状态", "事件", "用户", "正例率"],
        [[row['level'], fmt_int(row['rows']), fmt_int(row['users']), fmt_rate(row['label_rate'])] for row in duration_rows],
    )}

事件时长非正/缺失为 {fmt_int(duration['event_nonpositive_rows'])} 行，videos_basic 时长非正/缺失为 {fmt_int(duration['video_nonpositive_rows'])} 行；两者均为正时只有 {fmt_int(duration['positive_disagreement_rows'])} 行差异超过 0.5 ms。后续应预先选定 videos_basic.video_duration 为候选输入，event duration 只做一致性审计，不能静默 coalesce。

### 5.2 主要 upload_type

{markdown_table(
        ["upload_type", "事件", "用户", "正例率"],
        [[row['level'], fmt_int(row['rows']), fmt_int(row['users']), fmt_rate(row['label_rate'])] for row in upload_rows],
    )}

upload_type 的差异支持继续审查 content modality proxy，但不能直接把名称映射成真实模态；mapping 与 unknown 规则仍待批准。

### 5.3 源用户活跃度（仅诊断）

{markdown_table(
        ["user_active_degree", "事件", "用户", "正例率"],
        [[row['level'], fmt_int(row['rows']), fmt_int(row['users']), fmt_rate(row['label_rate'])] for row in activity_rows],
    )}

这些用户快照字段的采集时点尚未证明，所以只能用于覆盖/混杂诊断，不能据本表直接进入 point-in-time 特征 allowlist。

## 6. 对下一轮建模的直接约束

- RQ1 必须区分“总体概率校准”与“用户内/候选内判别”；只看 pooled PR-AUC 或 pooled AUC 不够。
- RQ2 应优先构造严格先验的 user-by-author、user-by-tag、user-by-content 交互；纯用户状态不可能在同一 user-time 内排序。
- 10/50/200 历史窗口必须在同一 target-row manifest 上消融，并同时控制日期与历史深度；本报告不冻结 burn-in。
- 先完成 compute/search budget、modality proxy 审批、固定行基线设计和不确定性实现，再批准 Gold 小样或正式训练。
- 当前结果是 source-Train 内的描述性假设生成证据，不是特征有效性结论、因果结论、Validation 证据或线上收益估计。

## 7. 可复现产物

{artifact_lines}

复算命令：

    {command}
"""
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(report, encoding="utf-8")


def output_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    run_started = time.perf_counter()
    stage_seconds: dict[str, float] = {}

    def timed(name: str, function: Any) -> Any:
        started = time.perf_counter()
        result = function()
        stage_seconds[name] = time.perf_counter() - started
        return result

    verified_inputs = timed(
        "verify_inputs", lambda: verify_inputs(config.full_input_sha256)
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads={config.threads}")
    con.execute("SET preserve_insertion_order=false")
    timed("create_analysis_tables", lambda: create_analysis_tables(con))
    checks = timed("validate_analysis_tables", lambda: validate_analysis_tables(con))

    summary = timed("canonical_summary", lambda: canonical_summary(con, checks))
    daily = timed("daily_summary", lambda: daily_summary(con))
    history, coverage = timed("history_summaries", lambda: history_summaries(con))
    dimensions = timed("dimension_summary", lambda: dimension_summary(con))
    duration = timed("duration_audit", lambda: duration_audit(con))
    metrics = timed("metric_summary", lambda: metric_summary(con))

    csv_outputs = {
        "canonical_summary": config.output_dir / "canonical_summary.csv",
        "daily_summary": config.output_dir / "daily_summary.csv",
        "history_depth_summary": config.output_dir / "history_depth_summary.csv",
        "history_date_coverage": config.output_dir / "history_date_coverage.csv",
        "metric_summary": config.output_dir / "metric_summary.csv",
        "dimension_summary": config.output_dir / "dimension_summary.csv",
        "duration_audit": config.output_dir / "duration_audit.csv",
    }

    def write_csv_outputs() -> None:
        write_csv(csv_outputs["canonical_summary"], summary)
        write_csv(csv_outputs["daily_summary"], daily)
        write_csv(csv_outputs["history_depth_summary"], history)
        write_csv(csv_outputs["history_date_coverage"], coverage)
        write_csv(csv_outputs["metric_summary"], metrics)
        write_csv(csv_outputs["dimension_summary"], dimensions)
        write_csv(csv_outputs["duration_audit"], [duration])

    timed("write_csv", write_csv_outputs)

    chart_path = config.output_dir / "train_association_overview.png"
    chart_written = False
    if config.render_chart:
        timed(
            "render_chart",
            lambda: render_chart(daily, history, metrics, dimensions, chart_path),
        )
        chart_written = True

    manifest_path = config.output_dir / "analysis_manifest.json"
    artifact_paths = [
        Path(__file__).resolve(),
        *csv_outputs.values(),
        *([chart_path] if chart_written else []),
        config.report_path,
        manifest_path,
    ]
    timed(
        "write_report",
        lambda: write_report(
            summary,
            daily,
            history,
            coverage,
            metrics,
            dimensions,
            duration,
            config,
            artifact_paths,
        ),
    )

    output_paths = [
        *csv_outputs.values(),
        *([chart_path] if chart_written else []),
        config.report_path,
    ]
    metadata_hashes = [
        {"role": role, "path": relative(path), "sha256": sha256_file(path)}
        for role, path in METADATA_FILES.items()
    ]
    accelerator = accelerator_probe()
    elapsed_before_manifest = time.perf_counter() - run_started
    manifest = {
        "analysis_id": (
            "kuairand_train_association_v002"
            if config.mode == "release"
            else "kuairand_train_association_v002_quick"
        ),
        "status": (
            "complete_train_only_descriptive_release"
            if config.mode == "release"
            else "quick_iteration_complete_nonrelease"
        ),
        "run_mode": config.mode,
        "checkpoint_eligible": config.checkpoint_eligible,
        "verification_level": config.verification_level,
        "deterministic_release": config.mode == "release",
        "analysis_date": "2026-08-14",
        "formal_silver_run_id": "silver-20260814-155536",
        "scope": {
            "target": "tab=1 canonical source-Train events",
            "target_date_range": [TRAIN_START, TRAIN_END],
            "history": "canonical source-Train all tabs with history_time < target_time",
            "label": "official long_view",
            "formal_model_training": False,
            "validation_access": False,
            "late_access": False,
            "random_access": False,
            "statistic_file_access": False,
            "silver_recleaned": False,
        },
        "input_data_files": verified_inputs,
        "metadata_files": metadata_hashes,
        "script": output_record(Path(__file__).resolve()),
        "environment": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "platform": platform.platform(),
            "threads": config.threads,
            "accelerator_backend": accelerator["selected_backend"],
            "accelerator_used": accelerator["accelerator_used"],
        },
        "accelerator": accelerator,
        "execution_policy": {
            "input_sha256_verified": config.full_input_sha256,
            "chart_generated": chart_written,
            "canonical_release_paths_updated": config.mode == "release",
            "quick_outputs_isolated": config.mode == "quick",
            "parallel_best_effort": config.mode == "quick",
        },
        "invariants": checks,
        "outputs": [output_record(path) for path in output_paths],
        "notes": [
            "No recursive glob is used.",
            "AUC values are descriptive and are not hard feature filters.",
            "within_timestamp_pair_auc is not named impression or slate GAUC.",
            "Quick mode is never checkpoint eligible and never overwrites release paths.",
            "checkpoint_eligible applies to this analysis artifact only; it does not approve a research Gate.",
            "The manifest intentionally excludes its own hash.",
        ],
    }
    if config.mode == "quick":
        manifest["performance"] = {
            "elapsed_seconds_before_manifest": elapsed_before_manifest,
            "stage_seconds": stage_seconds,
        }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    con.close()
    elapsed_seconds = time.perf_counter() - run_started
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": config.mode,
                "checkpoint_eligible": config.checkpoint_eligible,
                "threads": config.threads,
                "accelerator_backend": accelerator["selected_backend"],
                "gpu_used": accelerator["gpu_used"],
                "elapsed_seconds": round(elapsed_seconds, 3),
                "target_rows": checks["target_rows"],
                "report": relative(config.report_path),
                "manifest": relative(manifest_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
