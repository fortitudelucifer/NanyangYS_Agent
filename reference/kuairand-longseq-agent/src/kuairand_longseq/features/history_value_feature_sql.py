"""Point-in-time feature materialization for the history-value confirmation.

The standard builder allows standard outcomes to update later standard history.
The random builder is deliberately separate: its state tables contain standard
events only, so a random label cannot enter its own or another random row's H2
features by construction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import duckdb


SELECTED_COLUMNS = """
  source_table, source_row_number, user_id, video_id, event_date, time_ms, long_view,
  cat_user, cat_video, cat_author, cat_music, cat_video_type, cat_upload_type,
  cat_music_type, cat_tag_combo, cat_duration_bucket,
  static_log_duration, static_duration_valid, static_log_upload_age,
  static_upload_age_valid, static_upload_future, static_log_width,
  static_log_height, static_aspect, static_geometry_valid, static_tag_missing,
  prior_batch_n, prior_event_n, prior_positive_n, last_user_gap_s,
  w10_event_n, w10_positive_n, w50_event_n, w50_positive_n,
  w200_event_n, w200_positive_n, warm_user, warm_video, behavior_cold_video
"""


def _path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _hash_category(expression: str) -> str:
    return f"CAST(hash({expression}) % 9223372036854775807 AS BIGINT)"


def _static_select(alias: str = "e") -> str:
    video_type = _hash_category("coalesce(v.video_type, '__MISSING_VIDEO_TYPE__')")
    upload_type = _hash_category("coalesce(v.upload_type, '__MISSING_UPLOAD_TYPE__')")
    tag = _hash_category("coalesce(v.tag_clean, '__MISSING_TAG__')")
    return f"""
        {alias}.*,
        v.video_id IS NULL AS video_join_missing,
        coalesce({alias}.user_id, -1)::BIGINT AS cat_user,
        coalesce({alias}.video_id, -1)::BIGINT AS cat_video,
        coalesce(v.author_id, -1)::BIGINT AS cat_author,
        coalesce(v.music_id, -1)::BIGINT AS cat_music,
        {video_type} AS cat_video_type,
        {upload_type} AS cat_upload_type,
        coalesce(CAST(v.music_type AS BIGINT), -1)::BIGINT AS cat_music_type,
        {tag} AS cat_tag_combo,
        CASE
          WHEN v.video_duration IS NULL OR NOT isfinite(v.video_duration)
               OR v.video_duration <= 0 THEN 0
          WHEN v.video_duration < 10000 THEN 1
          WHEN v.video_duration < 20000 THEN 2
          WHEN v.video_duration < 40000 THEN 3
          WHEN v.video_duration < 60000 THEN 4
          WHEN v.video_duration < 120000 THEN 5
          WHEN v.video_duration < 300000 THEN 6
          ELSE 7
        END::BIGINT AS cat_duration_bucket,
        CASE WHEN v.video_duration IS NOT NULL AND isfinite(v.video_duration)
                       AND v.video_duration > 0
             THEN ln(1.0 + v.video_duration / 1000.0) ELSE 0.0 END::FLOAT
             AS static_log_duration,
        (v.video_duration IS NOT NULL AND isfinite(v.video_duration)
             AND v.video_duration > 0)::INT::FLOAT AS static_duration_valid,
        CASE WHEN v.upload_dt IS NOT NULL AND v.upload_dt <= {alias}.event_date
             THEN ln(1.0 + date_diff('day', v.upload_dt, {alias}.event_date))
             ELSE 0.0 END::FLOAT AS static_log_upload_age,
        (v.upload_dt IS NOT NULL AND v.upload_dt <= {alias}.event_date)::INT::FLOAT
             AS static_upload_age_valid,
        (v.upload_dt IS NOT NULL AND v.upload_dt > {alias}.event_date)::INT::FLOAT
             AS static_upload_future,
        CASE WHEN v.server_width IS NOT NULL AND isfinite(v.server_width)
                       AND v.server_width > 0
             THEN ln(1.0 + v.server_width) ELSE 0.0 END::FLOAT AS static_log_width,
        CASE WHEN v.server_height IS NOT NULL AND isfinite(v.server_height)
                       AND v.server_height > 0
             THEN ln(1.0 + v.server_height) ELSE 0.0 END::FLOAT AS static_log_height,
        CASE WHEN v.server_width IS NOT NULL AND isfinite(v.server_width)
                       AND v.server_width > 0
                       AND v.server_height IS NOT NULL AND isfinite(v.server_height)
                       AND v.server_height > 0
             THEN greatest(0.1, least(10.0, v.server_width / v.server_height))
             ELSE 0.0 END::FLOAT AS static_aspect,
        (v.server_width IS NOT NULL AND isfinite(v.server_width) AND v.server_width > 0
             AND v.server_height IS NOT NULL AND isfinite(v.server_height)
             AND v.server_height > 0)::INT::FLOAT AS static_geometry_valid,
        coalesce(v.tag_missing, true)::INT::FLOAT AS static_tag_missing
    """


def _standard_union_sql(
    *, early_path: Path, late_path: Path | None, mismatch_path: Path, end_date: str
) -> str:
    sources = [
        f"""SELECT source_table, source_row_number, user_id, video_id, event_date,
                    time_ms, tab, long_view
             FROM read_parquet({_path(early_path)})
             WHERE event_date BETWEEN DATE '2022-04-08' AND DATE '{end_date}'"""
    ]
    allowed = ["'early_standard'"]
    if late_path is not None:
        sources.append(
            f"""SELECT source_table, source_row_number, user_id, video_id, event_date,
                        time_ms, tab, long_view
                 FROM read_parquet({_path(late_path)})
                 WHERE event_date BETWEEN DATE '2022-04-08' AND DATE '{end_date}'"""
        )
        allowed.append("'late_standard'")
    sources.append(
        f"""SELECT source_table, source_row_number, user_id, video_id, event_date,
                    time_ms, tab, long_view
             FROM read_parquet({_path(mismatch_path)})
             WHERE source_table IN ({', '.join(allowed)})
               AND exclusion_reason = 'LONG_VIEW_FORMULA_MISMATCH'
               AND event_date BETWEEN DATE '2022-04-08' AND DATE '{end_date}'"""
    )
    return "\nUNION ALL\n".join(sources)


def _standard_state_sql() -> str:
    return """
    CREATE TEMP TABLE user_batch AS
    SELECT user_id, time_ms,
           count(*)::BIGINT AS batch_event_n,
           sum(long_view)::BIGINT AS batch_positive_n
    FROM event_meta
    GROUP BY user_id, time_ms;

    CREATE TEMP TABLE user_state AS
    SELECT user_id, time_ms,
           row_number() OVER (PARTITION BY user_id ORDER BY time_ms) - 1
               AS prior_batch_n,
           coalesce(sum(batch_event_n) OVER (
               PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0)::BIGINT
               AS prior_event_n,
           coalesce(sum(batch_positive_n) OVER (
               PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0)::BIGINT
               AS prior_positive_n,
           lag(time_ms) OVER (PARTITION BY user_id ORDER BY time_ms) AS prior_time_ms,
           coalesce(sum(batch_event_n) OVER (
               PARTITION BY user_id ORDER BY time_ms ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS w10_event_n,
           coalesce(sum(batch_positive_n) OVER (
               PARTITION BY user_id ORDER BY time_ms ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS w10_positive_n,
           coalesce(sum(batch_event_n) OVER (
               PARTITION BY user_id ORDER BY time_ms ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS w50_event_n,
           coalesce(sum(batch_positive_n) OVER (
               PARTITION BY user_id ORDER BY time_ms ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS w50_positive_n,
           coalesce(sum(batch_event_n) OVER (
               PARTITION BY user_id ORDER BY time_ms ROWS BETWEEN 200 PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS w200_event_n,
           coalesce(sum(batch_positive_n) OVER (
               PARTITION BY user_id ORDER BY time_ms ROWS BETWEEN 200 PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS w200_positive_n
    FROM user_batch;
    """


def _validate_features(
    con: duckdb.DuckDBPyConnection,
    *,
    target_start: str,
    target_end: str,
    expected_target_rows: int,
) -> dict[str, int]:
    row = con.execute(
        f"""
        SELECT count(*),
               sum((event_date BETWEEN DATE '{target_start}' AND DATE '{target_end}')::INT),
               count(DISTINCT (source_table, source_row_number)),
               sum(video_join_missing::INT),
               sum((audit_user_prior_time_ms >= time_ms)::INT),
               sum((w10_event_n > w50_event_n OR w50_event_n > w200_event_n
                    OR w200_event_n > prior_event_n)::INT),
               sum((prior_positive_n > prior_event_n
                    OR w10_positive_n > w10_event_n
                    OR w50_positive_n > w50_event_n
                    OR w200_positive_n > w200_event_n)::INT)
        FROM features_with_audit
        """
    ).fetchone()
    keys = (
        "feature_rows", "target_rows", "unique_identities", "missing_video_joins",
        "pit_violations", "window_monotonicity_violations", "positive_count_violations",
    )
    result = dict(zip(keys, map(int, row)))
    if result["target_rows"] != int(expected_target_rows):
        raise RuntimeError(
            f"target row mismatch: expected {expected_target_rows}, observed {result['target_rows']}"
        )
    if result["feature_rows"] != result["unique_identities"]:
        raise RuntimeError("feature identities are not unique")
    for key in (
        "missing_video_joins", "pit_violations", "window_monotonicity_violations",
        "positive_count_violations",
    ):
        if result[key]:
            raise RuntimeError(f"feature invariant failed: {key}={result[key]}")
    return result


def materialize_standard_features(
    con: duckdb.DuckDBPyConnection,
    *,
    early_path: Path,
    late_path: Path | None,
    mismatch_path: Path,
    videos_path: Path,
    output_path: Path,
    end_date: str,
    target_start: str,
    target_end: str,
    expected_target_rows: int,
) -> dict[str, Any]:
    """Materialize standard tab=1 rows with rolling all-tab standard history."""

    union = _standard_union_sql(
        early_path=early_path, late_path=late_path, mismatch_path=mismatch_path,
        end_date=end_date,
    )
    setup = f"""
    CREATE TEMP TABLE canonical_standard AS {union};
    CREATE UNIQUE INDEX canonical_identity
      ON canonical_standard(source_table, source_row_number);
    CREATE TEMP TABLE user_first AS
      SELECT user_id, min(event_date) AS first_date FROM canonical_standard GROUP BY user_id;
    CREATE TEMP TABLE video_first AS
      SELECT video_id, min(event_date) AS first_date FROM canonical_standard GROUP BY video_id;
    CREATE TEMP TABLE event_meta AS
    SELECT {_static_select('e')}
    FROM canonical_standard e LEFT JOIN read_parquet({_path(videos_path)}) v USING (video_id);
    {_standard_state_sql()}
    CREATE TEMP TABLE features_with_audit AS
    SELECT t.*, u.prior_batch_n::BIGINT AS prior_batch_n,
           u.prior_event_n, u.prior_positive_n,
           CASE WHEN u.prior_time_ms IS NULL THEN 0.0
                ELSE greatest(0.0, (t.time_ms-u.prior_time_ms)/1000.0) END::FLOAT
                AS last_user_gap_s,
           u.w10_event_n, u.w10_positive_n, u.w50_event_n, u.w50_positive_n,
           u.w200_event_n, u.w200_positive_n,
           (uf.first_date < DATE '2022-04-22')::BOOLEAN AS warm_user,
           (vf.first_date < DATE '{target_start}')::BOOLEAN AS warm_video,
           (vf.first_date >= DATE '{target_start}')::BOOLEAN AS behavior_cold_video,
           u.prior_time_ms AS audit_user_prior_time_ms
    FROM event_meta t
    JOIN user_state u USING (user_id, time_ms)
    JOIN user_first uf USING (user_id)
    JOIN video_first vf USING (video_id)
    WHERE t.tab=1;
    """
    con.execute(setup)
    validation = _validate_features(
        con, target_start=target_start, target_end=target_end,
        expected_target_rows=expected_target_rows,
    )
    copy = f"""
    COPY (SELECT {SELECTED_COLUMNS} FROM features_with_audit
          ORDER BY event_date,time_ms,user_id,source_table,source_row_number)
    TO {_path(output_path)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);
    """
    con.execute(copy)
    validation["feature_sql_sha256"] = hashlib.sha256((setup + copy).encode()).hexdigest()
    validation["random_labels_can_update_history"] = False
    return validation


def materialize_random_features(
    con: duckdb.DuckDBPyConnection,
    *,
    early_path: Path,
    late_path: Path,
    random_path: Path,
    mismatch_path: Path,
    videos_path: Path,
    output_path: Path,
    expected_target_rows: int,
) -> dict[str, Any]:
    """Materialize random targets using post-batch state from standard events only."""

    standard_union = _standard_union_sql(
        early_path=early_path, late_path=late_path, mismatch_path=mismatch_path,
        end_date="2022-05-08",
    )
    random_union = f"""
      SELECT source_table, source_row_number, user_id, video_id, event_date,
             time_ms, tab, long_view
      FROM read_parquet({_path(random_path)})
      WHERE event_date BETWEEN DATE '2022-04-22' AND DATE '2022-05-08'
      UNION ALL
      SELECT source_table, source_row_number, user_id, video_id, event_date,
             time_ms, tab, long_view
      FROM read_parquet({_path(mismatch_path)})
      WHERE source_table='random'
        AND exclusion_reason='LONG_VIEW_FORMULA_MISMATCH'
        AND event_date BETWEEN DATE '2022-04-22' AND DATE '2022-05-08'
    """
    setup = f"""
    CREATE TEMP TABLE canonical_standard AS {standard_union};
    CREATE UNIQUE INDEX standard_identity
      ON canonical_standard(source_table, source_row_number);
    CREATE TEMP TABLE canonical_random AS {random_union};
    CREATE UNIQUE INDEX random_identity
      ON canonical_random(source_table, source_row_number);
    CREATE TEMP TABLE user_first AS
      SELECT user_id, min(event_date) AS first_date FROM canonical_standard GROUP BY user_id;
    CREATE TEMP TABLE video_first AS
      SELECT video_id, min(event_date) AS first_date FROM canonical_standard GROUP BY video_id;
    CREATE TEMP TABLE user_batch AS
      SELECT user_id,time_ms,count(*)::BIGINT batch_event_n,
             sum(long_view)::BIGINT batch_positive_n
      FROM canonical_standard GROUP BY user_id,time_ms;
    CREATE TEMP TABLE user_post_state AS
      SELECT user_id,time_ms,
             row_number() OVER (PARTITION BY user_id ORDER BY time_ms)::BIGINT prior_batch_n,
             sum(batch_event_n) OVER (PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)::BIGINT prior_event_n,
             sum(batch_positive_n) OVER (PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)::BIGINT prior_positive_n,
             sum(batch_event_n) OVER (PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)::BIGINT w10_event_n,
             sum(batch_positive_n) OVER (PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)::BIGINT w10_positive_n,
             sum(batch_event_n) OVER (PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN 49 PRECEDING AND CURRENT ROW)::BIGINT w50_event_n,
             sum(batch_positive_n) OVER (PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN 49 PRECEDING AND CURRENT ROW)::BIGINT w50_positive_n,
             sum(batch_event_n) OVER (PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN 199 PRECEDING AND CURRENT ROW)::BIGINT w200_event_n,
             sum(batch_positive_n) OVER (PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN 199 PRECEDING AND CURRENT ROW)::BIGINT w200_positive_n
      FROM user_batch;
    CREATE TEMP TABLE random_meta AS
      SELECT {_static_select('e')}
      FROM canonical_random e LEFT JOIN read_parquet({_path(videos_path)}) v USING(video_id);
    CREATE TEMP TABLE features_with_audit AS
    SELECT r.*,
           coalesce(s.prior_batch_n,0)::BIGINT AS prior_batch_n,
           coalesce(s.prior_event_n,0)::BIGINT AS prior_event_n,
           coalesce(s.prior_positive_n,0)::BIGINT AS prior_positive_n,
           CASE WHEN s.time_ms IS NULL THEN 0.0
                ELSE greatest(0.0,(r.time_ms-s.time_ms)/1000.0) END::FLOAT last_user_gap_s,
           coalesce(s.w10_event_n,0)::BIGINT w10_event_n,
           coalesce(s.w10_positive_n,0)::BIGINT w10_positive_n,
           coalesce(s.w50_event_n,0)::BIGINT w50_event_n,
           coalesce(s.w50_positive_n,0)::BIGINT w50_positive_n,
           coalesce(s.w200_event_n,0)::BIGINT w200_event_n,
           coalesce(s.w200_positive_n,0)::BIGINT w200_positive_n,
           coalesce(uf.first_date < DATE '2022-04-22', false)::BOOLEAN warm_user,
           coalesce(vf.first_date < DATE '2022-04-22', false)::BOOLEAN warm_video,
           coalesce(vf.first_date >= DATE '2022-04-22', true)::BOOLEAN behavior_cold_video,
           s.time_ms AS audit_user_prior_time_ms
    FROM random_meta r
    ASOF LEFT JOIN user_post_state s
      ON r.user_id=s.user_id AND r.time_ms>s.time_ms
    LEFT JOIN user_first uf ON r.user_id=uf.user_id
    LEFT JOIN video_first vf ON r.video_id=vf.video_id;
    """
    con.execute(setup)
    validation = _validate_features(
        con, target_start="2022-04-22", target_end="2022-05-08",
        expected_target_rows=expected_target_rows,
    )
    copy = f"""
    COPY (SELECT {SELECTED_COLUMNS} FROM features_with_audit
          ORDER BY event_date,time_ms,user_id,source_table,source_row_number)
    TO {_path(output_path)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);
    """
    con.execute(copy)
    validation["feature_sql_sha256"] = hashlib.sha256((setup + copy).encode()).hexdigest()
    validation["history_source"] = "canonical_standard_only"
    validation["random_labels_can_update_history"] = False
    return validation
