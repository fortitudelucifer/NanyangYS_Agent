"""DuckDB feature materialization for the scoped Gate 2B Train-only study."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import duckdb


EXPECTED = {
    "all_tab_rows": 3_671_408,
    "target_rows": 2_399_844,
    "target_users": 950,
    "target_videos": 974_550,
    "target_positives": 765_417,
}


def _sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _hash_category(expression: str) -> str:
    return f"CAST(hash({expression}) % 9223372036854775807 AS BIGINT)"


def _entity_sql(prefix: str, category_column: str, validity: str) -> str:
    return f"""
    CREATE TEMP TABLE {prefix}_batch AS
    SELECT user_id, time_ms, {category_column} AS entity_key,
           count(*)::BIGINT AS event_n,
           sum(long_view)::BIGINT AS positive_n
    FROM event_meta
    WHERE {validity}
    GROUP BY user_id, time_ms, {category_column};

    CREATE TEMP TABLE {prefix}_state AS
    SELECT user_id, time_ms, entity_key,
           coalesce(sum(event_n) OVER (
               PARTITION BY user_id, entity_key ORDER BY time_ms
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS prior_n,
           coalesce(sum(positive_n) OVER (
               PARTITION BY user_id, entity_key ORDER BY time_ms
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS prior_positive_n,
           lag(time_ms) OVER (
               PARTITION BY user_id, entity_key ORDER BY time_ms
           ) AS prior_time_ms
    FROM {prefix}_batch;
    """


def materialize_gate2b_features(
    con: duckdb.DuckDBPyConnection,
    *,
    early_path: Path,
    mismatch_path: Path,
    videos_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build the fixed Train target feature matrix and fail closed on invariants."""

    early = _sql_path(early_path)
    mismatch = _sql_path(mismatch_path)
    videos = _sql_path(videos_path)
    output = _sql_path(output_path)
    cat_video_type = _hash_category("coalesce(v.video_type, '__MISSING_VIDEO_TYPE__')")
    cat_upload_type = _hash_category("coalesce(v.upload_type, '__MISSING_UPLOAD_TYPE__')")
    cat_tag = _hash_category("coalesce(v.tag_clean, '__MISSING_TAG__')")

    setup_sql = f"""
    CREATE TEMP TABLE canonical_all AS
    SELECT source_table, source_row_number, user_id, video_id, event_date,
           time_ms, tab, long_view
    FROM read_parquet({early})
    WHERE event_date BETWEEN DATE '2022-04-08' AND DATE '2022-04-17'
    UNION ALL
    SELECT source_table, source_row_number, user_id, video_id, event_date,
           time_ms, tab, long_view
    FROM read_parquet({mismatch})
    WHERE source_table = 'early_standard'
      AND exclusion_reason = 'LONG_VIEW_FORMULA_MISMATCH'
      AND event_date BETWEEN DATE '2022-04-08' AND DATE '2022-04-17';

    CREATE TEMP TABLE event_meta AS
    SELECT
        e.*,
        v.video_id IS NULL AS video_join_missing,
        coalesce(e.user_id, -1)::BIGINT AS cat_user,
        coalesce(e.video_id, -1)::BIGINT AS cat_video,
        coalesce(v.author_id, -1)::BIGINT AS cat_author,
        coalesce(v.music_id, -1)::BIGINT AS cat_music,
        {cat_video_type} AS cat_video_type,
        {cat_upload_type} AS cat_upload_type,
        coalesce(CAST(v.music_type AS BIGINT), -1)::BIGINT AS cat_music_type,
        {cat_tag} AS cat_tag_combo,
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
        CASE WHEN v.upload_dt IS NOT NULL AND v.upload_dt <= e.event_date
             THEN ln(1.0 + date_diff('day', v.upload_dt, e.event_date))
             ELSE 0.0 END::FLOAT AS static_log_upload_age,
        (v.upload_dt IS NOT NULL AND v.upload_dt <= e.event_date)::INT::FLOAT
             AS static_upload_age_valid,
        (v.upload_dt IS NOT NULL AND v.upload_dt > e.event_date)::INT::FLOAT
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
    FROM canonical_all e
    LEFT JOIN read_parquet({videos}) v USING (video_id);

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
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS prior_event_n,
           coalesce(sum(batch_positive_n) OVER (
               PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS prior_positive_n,
           lag(time_ms) OVER (PARTITION BY user_id ORDER BY time_ms) AS prior_time_ms,
           coalesce(sum(batch_event_n) OVER (
               PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS w10_event_n,
           coalesce(sum(batch_positive_n) OVER (
               PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS w10_positive_n,
           coalesce(sum(batch_event_n) OVER (
               PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS w50_event_n,
           coalesce(sum(batch_positive_n) OVER (
               PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS w50_positive_n,
           coalesce(sum(batch_event_n) OVER (
               PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN 200 PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS w200_event_n,
           coalesce(sum(batch_positive_n) OVER (
               PARTITION BY user_id ORDER BY time_ms
               ROWS BETWEEN 200 PRECEDING AND 1 PRECEDING
           ), 0)::BIGINT AS w200_positive_n
    FROM user_batch;
    """
    con.execute(setup_sql)

    entity_statements: list[str] = []

    final_sql = """
    CREATE TEMP TABLE features_with_audit AS
    SELECT
        t.source_table, t.source_row_number, t.user_id, t.video_id,
        t.event_date, t.time_ms, t.long_view,
        t.video_join_missing,
        t.cat_user, t.cat_video, t.cat_author, t.cat_music,
        t.cat_video_type, t.cat_upload_type, t.cat_music_type,
        t.cat_tag_combo, t.cat_duration_bucket,
        t.static_log_duration, t.static_duration_valid,
        t.static_log_upload_age, t.static_upload_age_valid,
        t.static_upload_future, t.static_log_width, t.static_log_height,
        t.static_aspect, t.static_geometry_valid, t.static_tag_missing,
        u.prior_batch_n::BIGINT AS prior_batch_n,
        u.prior_event_n, u.prior_positive_n,
        CASE WHEN u.prior_time_ms IS NULL THEN 0.0
             ELSE greatest(0.0, (t.time_ms - u.prior_time_ms) / 1000.0) END::FLOAT
             AS last_user_gap_s,
        u.w10_event_n, u.w10_positive_n,
        u.w50_event_n, u.w50_positive_n,
        u.w200_event_n, u.w200_positive_n,
        u.prior_time_ms AS audit_user_prior_time_ms
    FROM event_meta t
    JOIN user_state u USING (user_id, time_ms)
    WHERE t.tab = 1;
    """
    con.execute(final_sql)

    validation_row = con.execute(
        """
        SELECT
          (SELECT count(*) FROM canonical_all),
          count(*),
          count(DISTINCT (source_table, source_row_number)),
          count(DISTINCT user_id),
          count(DISTINCT video_id),
          sum(long_view),
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
    keys = [
        "all_tab_rows",
        "target_rows",
        "unique_target_identities",
        "target_users",
        "target_videos",
        "target_positives",
        "missing_video_joins",
        "user_pit_violations",
        "window_monotonicity_violations",
        "positive_count_violations",
    ]
    validation = dict(zip(keys, map(int, validation_row)))
    for key, expected in EXPECTED.items():
        if validation[key] != expected:
            raise RuntimeError(f"{key}: expected {expected}, observed {validation[key]}")
    if validation["unique_target_identities"] != EXPECTED["target_rows"]:
        raise RuntimeError("target identity is not unique")
    for key in [
        "missing_video_joins",
        "user_pit_violations",
        "window_monotonicity_violations",
        "positive_count_violations",
    ]:
        if validation[key] != 0:
            raise RuntimeError(f"feature invariant failed: {key}={validation[key]}")

    batch_variation = con.execute(
        """
        SELECT count(*) FROM (
          SELECT user_id, time_ms
          FROM features_with_audit
          GROUP BY user_id, time_ms
          HAVING min(prior_batch_n) <> max(prior_batch_n)
              OR min(prior_event_n) <> max(prior_event_n)
              OR min(prior_positive_n) <> max(prior_positive_n)
              OR min(w10_event_n) <> max(w10_event_n)
              OR min(w50_event_n) <> max(w50_event_n)
              OR min(w200_event_n) <> max(w200_event_n)
        )
        """
    ).fetchone()[0]
    validation["user_feature_batch_variation_groups"] = int(batch_variation)
    if batch_variation:
        raise RuntimeError("user-only features vary inside a timestamp batch")

    selected_columns = """
      source_table, source_row_number, user_id, video_id, event_date, time_ms, long_view,
      cat_user, cat_video, cat_author, cat_music, cat_video_type, cat_upload_type,
      cat_music_type, cat_tag_combo, cat_duration_bucket,
      static_log_duration, static_duration_valid, static_log_upload_age,
      static_upload_age_valid, static_upload_future, static_log_width,
      static_log_height, static_aspect, static_geometry_valid, static_tag_missing,
      prior_batch_n, prior_event_n, prior_positive_n, last_user_gap_s,
      w10_event_n, w10_positive_n, w50_event_n, w50_positive_n,
      w200_event_n, w200_positive_n
    """
    copy_sql = f"""
    COPY (
      SELECT {selected_columns}
      FROM features_with_audit
      ORDER BY event_date, time_ms, user_id, source_table, source_row_number
    ) TO {output} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);
    """
    con.execute(copy_sql)
    validation["feature_sql_sha256"] = hashlib.sha256(
        (setup_sql + "".join(entity_statements) + final_sql + copy_sql).encode("utf-8")
    ).hexdigest()
    return validation
