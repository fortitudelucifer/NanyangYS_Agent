from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import yaml


CLEANING_VERSION = "v0.1.0"

EVENT_SCHEMA = {
    "user_id": "BIGINT",
    "video_id": "BIGINT",
    "date": "BIGINT",
    "hourmin": "BIGINT",
    "time_ms": "BIGINT",
    "is_click": "BIGINT",
    "is_like": "BIGINT",
    "is_follow": "BIGINT",
    "is_comment": "BIGINT",
    "is_forward": "BIGINT",
    "is_hate": "BIGINT",
    "long_view": "BIGINT",
    "play_time_ms": "BIGINT",
    "duration_ms": "BIGINT",
    "profile_stay_time": "BIGINT",
    "comment_stay_time": "BIGINT",
    "is_profile_enter": "BIGINT",
    "is_rand": "BIGINT",
    "tab": "BIGINT",
}

USER_SCHEMA = {
    "user_id": "BIGINT",
    "user_active_degree": "VARCHAR",
    "is_lowactive_period": "BIGINT",
    "is_live_streamer": "BIGINT",
    "is_video_author": "BIGINT",
    "follow_user_num": "BIGINT",
    "follow_user_num_range": "VARCHAR",
    "fans_user_num": "BIGINT",
    "fans_user_num_range": "VARCHAR",
    "friend_user_num": "BIGINT",
    "friend_user_num_range": "VARCHAR",
    "register_days": "BIGINT",
    "register_days_range": "VARCHAR",
    **{f"onehot_feat{i}": "DOUBLE" for i in range(18)},
}

VIDEO_SCHEMA = {
    "video_id": "BIGINT",
    "author_id": "BIGINT",
    "video_type": "VARCHAR",
    "upload_dt": "DATE",
    "upload_type": "VARCHAR",
    "visible_status": "DOUBLE",
    "video_duration": "DOUBLE",
    "server_width": "DOUBLE",
    "server_height": "DOUBLE",
    "music_id": "BIGINT",
    "music_type": "DOUBLE",
    "tag": "VARCHAR",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def qliteral(value: str | Path) -> str:
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def schema_literal(schema: dict[str, str]) -> str:
    entries = ", ".join(f"{qliteral(k)}: {qliteral(v)}" for k, v in schema.items())
    return "{" + entries + "}"


def csv_scan(path: Path, schema: dict[str, str]) -> str:
    return (
        f"read_csv({qliteral(path)}, header=true, auto_detect=false, "
        f"columns={schema_literal(schema)}, nullstr='')"
    )


def stable_row_hash(columns: Iterable[str]) -> str:
    fields = ", ".join(f"c{i} := {qident(c)}" for i, c in enumerate(columns))
    return f"md5(to_json(struct_pack({fields})))"


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def resolve_project_path(project_root: Path, configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def assert_raw_outside_generated(raw_paths: Iterable[Path], generated_paths: Iterable[Path]) -> None:
    generated = [p.resolve() for p in generated_paths]
    for raw in raw_paths:
        resolved = raw.resolve()
        for target in generated:
            if resolved == target or target in resolved.parents:
                raise RuntimeError(f"Raw input is inside a generated-data directory: {resolved}")


def header_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def validate_inputs(project_root: Path) -> dict[str, Any]:
    config_dir = project_root / "configs"
    paths_cfg = load_yaml(config_dir / "data_paths.yaml")
    contract = load_yaml(config_dir / "data_contract.yaml")
    rules = load_yaml(config_dir / "cleaning_rules.yaml")
    experiment = load_yaml(config_dir / "experiment_v001.yaml")

    if rules.get("status") != "approved_for_silver_build":
        raise RuntimeError("cleaning_rules.yaml is not approved_for_silver_build")
    if experiment.get("status") != "approved_for_silver_build":
        raise RuntimeError("experiment_v001.yaml is not approved_for_silver_build")
    if not contract.get("raw_is_immutable"):
        raise RuntimeError("data_contract.yaml must declare raw_is_immutable: true")

    raw_paths = {
        name: resolve_project_path(project_root, configured)
        for name, configured in paths_cfg["raw"].items()
        if name != "root"
    }
    generated_paths = {
        name: resolve_project_path(project_root, configured)
        for name, configured in paths_cfg["generated"].items()
    }
    missing = [str(path) for path in raw_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing raw files: " + ", ".join(missing))
    assert_raw_outside_generated(raw_paths.values(), generated_paths.values())

    event_sources = ["early_standard", "late_standard", "random"]
    required = list(contract["required_event_columns"])
    for source in event_sources:
        columns = header_columns(raw_paths[source])
        absent = [column for column in required if column not in columns]
        if absent:
            raise ValueError(f"{source} is missing required columns: {absent}")

    if header_columns(raw_paths["users"]) != list(USER_SCHEMA):
        raise ValueError("user_features_1k.csv header differs from the frozen contract")
    if header_columns(raw_paths["videos_basic"]) != list(VIDEO_SCHEMA):
        raise ValueError("video_features_basic_1k.csv header differs from the frozen contract")

    return {
        "paths": paths_cfg,
        "contract": contract,
        "rules": rules,
        "experiment": experiment,
        "raw_paths": raw_paths,
        "generated_paths": generated_paths,
    }


def configure_duckdb(con: duckdb.DuckDBPyConnection, temp_dir: Path) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory={qliteral(temp_dir)}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET threads={max(1, min(8, os.cpu_count() or 4))}")


def copy_query(con: duckdb.DuckDBPyConnection, query: str, destination: Path, fmt: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        options = "FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000"
    elif fmt == "csv":
        options = "FORMAT CSV, HEADER true"
    else:
        raise ValueError(fmt)
    con.execute(f"COPY ({query}) TO {qliteral(destination)} ({options})")


def scalar(con: duckdb.DuckDBPyConnection, query: str) -> Any:
    return con.execute(query).fetchone()[0]


def event_work_sql(
    ranked_table: str,
    expected_rand: int,
    date_min: int,
    date_max: int,
) -> str:
    binary_cols = [
        "is_click", "is_like", "is_follow", "is_comment", "is_forward",
        "is_hate", "long_view", "is_profile_enter", "is_rand",
    ]
    invalid_binary = " OR ".join(
        f"({qident(c)} IS NULL OR {qident(c)} NOT IN (0, 1))" for c in binary_cols
    )
    invalid_domain = f"""
        ({invalid_binary})
        OR user_id IS NULL OR user_id < 0
        OR video_id IS NULL OR video_id < 0
        OR time_ms IS NULL OR time_ms <= 0
        OR date IS NULL OR date < {date_min} OR date > {date_max}
        OR hourmin IS NULL OR hourmin < 0 OR hourmin > 2359 OR hourmin % 100 >= 60
        OR tab IS NULL OR tab < 0 OR tab > 4
        OR is_rand IS NULL OR is_rand <> {expected_rand}
        OR play_time_ms IS NULL OR play_time_ms < 0
    """
    invalid_reasons = f"""
        concat_ws('|',
          CASE WHEN ({invalid_binary}) THEN 'INVALID_BINARY_VALUE' END,
          CASE WHEN user_id IS NULL OR user_id < 0 THEN 'INVALID_USER_ID' END,
          CASE WHEN video_id IS NULL OR video_id < 0 THEN 'INVALID_VIDEO_ID' END,
          CASE WHEN time_ms IS NULL OR time_ms <= 0 THEN 'INVALID_TIME_MS' END,
          CASE WHEN date IS NULL OR date < {date_min} OR date > {date_max} THEN 'INVALID_DATE' END,
          CASE WHEN hourmin IS NULL OR hourmin < 0 OR hourmin > 2359 OR hourmin % 100 >= 60 THEN 'INVALID_HOURMIN' END,
          CASE WHEN tab IS NULL OR tab < 0 OR tab > 4 THEN 'INVALID_TAB' END,
          CASE WHEN is_rand IS NULL OR is_rand <> {expected_rand} THEN 'POLICY_FLAG_MISMATCH' END,
          CASE WHEN play_time_ms IS NULL OR play_time_ms < 0 THEN 'INVALID_PLAY_TIME' END
        )
    """
    return f"""
      WITH canonical AS (
        SELECT * FROM {ranked_table} WHERE exact_duplicate_rank = 1
      ), batch_annotated AS (
        SELECT *,
          count(*) OVER (PARTITION BY user_id, video_id, time_ms) AS conflict_group_size,
          count(DISTINCT video_id) OVER (PARTITION BY user_id, time_ms) AS timestamp_batch_video_count
        FROM canonical
      ), flags AS (
        SELECT *,
          try_strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE AS event_date,
          ({invalid_domain}) AS invalid_domain,
          {invalid_reasons} AS invalid_reason_codes,
          (duration_ms IS NULL OR duration_ms <= 0) AS duration_missing_or_nonpositive,
          (duration_ms > 0 AND play_time_ms >= 0 AND play_time_ms > duration_ms) AS play_time_greater_than_duration,
          (timestamp_batch_video_count > 1) AS same_timestamp_multiple_videos,
          CASE
            WHEN duration_ms > 0 AND play_time_ms >= 0
            THEN CASE WHEN duration_ms <= 18000 THEN (play_time_ms >= duration_ms)::BIGINT
                      ELSE (play_time_ms >= 18000)::BIGINT END
          END AS expected_long_view,
          CASE
            WHEN duration_ms > 0 AND play_time_ms >= 0 AND long_view IN (0, 1)
            THEN long_view <> CASE WHEN duration_ms <= 18000 THEN (play_time_ms >= duration_ms)::BIGINT
                                   ELSE (play_time_ms >= 18000)::BIGINT END
            ELSE false
          END AS long_view_formula_mismatch,
          CASE WHEN duration_ms > 0 AND play_time_ms >= 0
               THEN play_time_ms::DOUBLE / duration_ms END AS raw_watch_ratio,
          CASE WHEN duration_ms > 0 AND play_time_ms >= 0
               THEN least(play_time_ms::DOUBLE / duration_ms, 1.0) END AS capped_watch_ratio,
          CASE WHEN user_id IS NOT NULL AND video_id IS NOT NULL AND time_ms IS NOT NULL
                    AND conflict_group_size > 1
               THEN md5(concat_ws(':', CAST(user_id AS VARCHAR), CAST(video_id AS VARCHAR), CAST(time_ms AS VARCHAR)))
          END AS conflict_group_id,
          CASE WHEN user_id IS NOT NULL AND time_ms IS NOT NULL
               THEN md5(concat_ws(':', CAST(user_id AS VARCHAR), CAST(time_ms AS VARCHAR)))
          END AS timestamp_batch_id
        FROM batch_annotated
      ), classified AS (
        SELECT *,
          CASE
            WHEN conflict_group_id IS NOT NULL THEN 'CONFLICTING_EVENT_GROUP'
            WHEN invalid_domain THEN 'INVALID_DOMAIN'
            WHEN long_view_formula_mismatch THEN 'LONG_VIEW_FORMULA_MISMATCH'
          END AS exclusion_reason
        FROM flags
      )
      SELECT *,
        CASE
          WHEN exclusion_reason IS NOT NULL THEN 'quarantined'
          WHEN duration_missing_or_nonpositive OR play_time_greater_than_duration
               OR exact_duplicate_count > 1 THEN 'valid_with_warning'
          ELSE 'valid'
        END AS record_status,
        concat_ws('|',
          CASE WHEN exact_duplicate_count > 1 THEN 'EXACT_DUPLICATE_CANONICAL' END,
          CASE WHEN duration_missing_or_nonpositive THEN 'DURATION_MISSING_OR_NONPOSITIVE' END,
          CASE WHEN play_time_greater_than_duration THEN 'PLAY_TIME_GREATER_THAN_DURATION' END,
          CASE WHEN same_timestamp_multiple_videos THEN 'SAME_TIMESTAMP_MULTIPLE_VIDEOS' END,
          CASE WHEN long_view_formula_mismatch THEN 'LONG_VIEW_FORMULA_MISMATCH' END
        ) AS quality_flags,
        {qliteral(CLEANING_VERSION)} AS cleaning_version
      FROM classified
    """


def create_ranked_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    source_table: str,
    source_path: Path,
    schema: dict[str, str],
) -> None:
    columns = list(schema)
    partition = ", ".join(qident(c) for c in columns)
    row_hash = stable_row_hash(columns)
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE {table_name} AS
      WITH numbered AS (
        SELECT row_number() OVER ()::BIGINT AS source_row_number, *
        FROM {csv_scan(source_path, schema)}
      )
      SELECT *,
        count(*) OVER (PARTITION BY {partition})::BIGINT AS exact_duplicate_count,
        row_number() OVER (PARTITION BY {partition} ORDER BY source_row_number)::BIGINT AS exact_duplicate_rank,
        min(source_row_number) OVER (PARTITION BY {partition})::BIGINT AS canonical_source_row_number,
        CASE WHEN count(*) OVER (PARTITION BY {partition}) > 1 THEN {row_hash} END AS duplicate_group_id,
        {qliteral(source_table)} AS source_table
      FROM numbered
    """)


def duplicate_summaries(con: duckdb.DuckDBPyConnection, table_name: str) -> list[dict[str, Any]]:
    rows = con.execute(f"""
      SELECT source_table, duplicate_group_id,
             max(exact_duplicate_count)::BIGINT AS raw_rows,
             1::BIGINT AS canonical_rows,
             max(exact_duplicate_count)::BIGINT - 1 AS duplicate_copy_rows,
             min(canonical_source_row_number)::BIGINT AS canonical_source_row_number
      FROM {table_name}
      WHERE exact_duplicate_count > 1
      GROUP BY source_table, duplicate_group_id
      ORDER BY source_table, canonical_source_row_number
    """).fetchall()
    names = ["source_table", "duplicate_group_id", "raw_rows", "canonical_rows", "duplicate_copy_rows", "canonical_source_row_number"]
    return [dict(zip(names, row)) for row in rows]


def export_duplicate_copies(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    raw_columns: list[str],
    destination: Path,
) -> None:
    payload = stable_row_hash(raw_columns)
    fields = ", ".join(f"c{i} := {qident(c)}" for i, c in enumerate(raw_columns))
    copy_query(con, f"""
      SELECT source_table, source_row_number, canonical_source_row_number,
             duplicate_group_id, {payload} AS payload_hash,
             to_json(struct_pack({fields})) AS payload_json
      FROM {table_name}
      WHERE exact_duplicate_rank > 1
      ORDER BY source_row_number
    """, destination, "parquet")


def process_event(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    source_path: Path,
    event_contract: dict[str, Any],
    stage: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = "ranked_event"
    work = "event_work"
    create_ranked_table(con, ranked, source_table, source_path, EVENT_SCHEMA)
    duplicate_groups = duplicate_summaries(con, ranked)
    export_duplicate_copies(con, ranked, list(EVENT_SCHEMA), stage / "work" / f"duplicate_copies_{source_table}.parquet")

    date_min = int(event_contract["expected_date_min"].replace("-", ""))
    date_max = int(event_contract["expected_date_max"].replace("-", ""))
    expected_rand = int(event_contract["expected_is_rand"])
    con.execute(f"CREATE OR REPLACE TEMP TABLE {work} AS " + event_work_sql(ranked, expected_rand, date_min, date_max))

    copy_query(
        con,
        f"SELECT * FROM {work} WHERE exclusion_reason IS NULL ORDER BY user_id, time_ms, video_id, source_row_number",
        stage / "silver" / f"events_{source_table}.parquet",
        "parquet",
    )
    for reason, filename in [
        ("CONFLICTING_EVENT_GROUP", "conflict"),
        ("INVALID_DOMAIN", "invalid"),
        ("LONG_VIEW_FORMULA_MISMATCH", "mismatch"),
    ]:
        copy_query(
            con,
            f"SELECT * FROM {work} WHERE exclusion_reason = {qliteral(reason)} ORDER BY source_row_number",
            stage / "work" / f"{filename}_{source_table}.parquet",
            "parquet",
        )

    raw_rows = int(scalar(con, f"SELECT count(*) FROM {ranked}"))
    canonical_rows = int(scalar(con, f"SELECT count(*) FROM {work}"))
    duplicate_copy_rows = raw_rows - canonical_rows
    reason_counts = dict(con.execute(f"""
      SELECT exclusion_reason, count(*)::BIGINT
      FROM {work}
      WHERE exclusion_reason IS NOT NULL
      GROUP BY exclusion_reason
    """).fetchall())
    silver_rows = int(scalar(con, f"SELECT count(*) FROM {work} WHERE exclusion_reason IS NULL"))
    reconciliation = {
        "source_table": source_table,
        "raw_rows": raw_rows,
        "canonical_rows": canonical_rows,
        "duplicate_copy_rows": duplicate_copy_rows,
        "conflicting_rows": int(reason_counts.get("CONFLICTING_EVENT_GROUP", 0)),
        "invalid_domain_rows": int(reason_counts.get("INVALID_DOMAIN", 0)),
        "label_formula_mismatch_rows": int(reason_counts.get("LONG_VIEW_FORMULA_MISMATCH", 0)),
        "silver_rows": silver_rows,
    }
    reconciliation["reconciled"] = (
        raw_rows
        == duplicate_copy_rows
        + reconciliation["conflicting_rows"]
        + reconciliation["invalid_domain_rows"]
        + reconciliation["label_formula_mismatch_rows"]
        + silver_rows
    )

    label_rates: list[dict[str, Any]] = []
    for label in ["long_view", "is_like", "is_hate"]:
        raw = con.execute(f"""
          SELECT count(*)::BIGINT,
                 count(*) FILTER (WHERE {qident(label)} IN (0,1))::BIGINT,
                 avg({qident(label)}::DOUBLE) FILTER (WHERE {qident(label)} IN (0,1))
          FROM {ranked}
        """).fetchone()
        canonical = con.execute(f"""
          SELECT count(*)::BIGINT,
                 count(*) FILTER (WHERE {qident(label)} IN (0,1))::BIGINT,
                 avg({qident(label)}::DOUBLE) FILTER (WHERE {qident(label)} IN (0,1))
          FROM {work}
        """).fetchone()
        silver = con.execute(f"""
          SELECT count(*)::BIGINT,
                 count(*) FILTER (WHERE {qident(label)} IN (0,1))::BIGINT,
                 avg({qident(label)}::DOUBLE) FILTER (WHERE {qident(label)} IN (0,1))
          FROM {work} WHERE exclusion_reason IS NULL
        """).fetchone()
        label_rates.append({
            "source_table": source_table,
            "label": label,
            "raw_rows": raw[0], "raw_valid_label_rows": raw[1], "raw_rate": raw[2],
            "canonical_rows": canonical[0], "canonical_valid_label_rows": canonical[1], "canonical_rate": canonical[2],
            "silver_rows": silver[0], "silver_valid_label_rows": silver[1], "silver_rate": silver[2],
            "silver_minus_raw_rate": None if raw[2] is None or silver[2] is None else silver[2] - raw[2],
        })

    rule_queries = {
        "exact_duplicate_rows": f"SELECT count(*) FROM {ranked} WHERE exact_duplicate_rank > 1",
        "same_user_video_time_with_conflicting_payload": f"SELECT count(*) FROM {work} WHERE conflict_group_id IS NOT NULL",
        "invalid_domain_row": f"SELECT count(*) FROM {work} WHERE invalid_domain",
        "label_formula_inconsistency": f"SELECT count(*) FROM {work} WHERE long_view_formula_mismatch",
        "missing_or_nonpositive_duration": f"SELECT count(*) FROM {work} WHERE duration_missing_or_nonpositive",
        "play_time_greater_than_duration": f"SELECT count(*) FROM {work} WHERE play_time_greater_than_duration",
        "same_user_time_multiple_videos": f"SELECT count(*) FROM {work} WHERE same_timestamp_multiple_videos",
    }
    quality = [
        {"source_table": source_table, "rule": rule, "affected_rows": int(scalar(con, query))}
        for rule, query in rule_queries.items()
    ]
    con.execute(f"DROP TABLE {work}")
    con.execute(f"DROP TABLE {ranked}")
    return reconciliation, duplicate_groups, label_rates, quality


def process_dimension(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    source_path: Path,
    schema: dict[str, str],
    entity_key: str,
    silver_name: str,
    stage: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = "ranked_dimension"
    create_ranked_table(con, ranked, source_table, source_path, schema)
    duplicate_groups = duplicate_summaries(con, ranked)
    export_duplicate_copies(con, ranked, list(schema), stage / "work" / f"duplicate_copies_{source_table}.parquet")
    columns = ", ".join(qident(c) for c in schema)
    payload_fields = ", ".join(f"c{i} := {qident(c)}" for i, c in enumerate(schema))
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE dimension_work AS
      WITH canonical AS (
        SELECT * FROM {ranked} WHERE exact_duplicate_rank = 1
      ), keyed AS (
        SELECT *, count(*) OVER (PARTITION BY {qident(entity_key)}) AS entity_key_count
        FROM canonical
      )
      SELECT *,
        CASE WHEN {qident(entity_key)} IS NULL OR entity_key_count > 1 THEN 'DIMENSION_KEY_CONFLICT' END AS exclusion_reason,
        to_json(struct_pack({payload_fields})) AS payload_json
      FROM keyed
    """)
    if source_table == "users":
        derived = """
          CASE
            WHEN is_live_streamer = -124 THEN 'UNKNOWN_SENTINEL'
            WHEN is_live_streamer = 1 THEN 'CONFIRMED_LIVE'
            WHEN is_live_streamer = 0 THEN 'CONFIRMED_NOT_LIVE'
            WHEN is_live_streamer IS NULL THEN 'MISSING'
            ELSE 'UNEXPECTED_VALUE'
          END AS is_live_streamer_state,
          (is_live_streamer = -124) AS is_live_streamer_unknown_sentinel,
          CASE WHEN is_live_streamer = -124 OR is_live_streamer IS NULL OR is_live_streamer NOT IN (-124,0,1)
               THEN 'valid_with_warning' ELSE 'valid' END AS record_status,
          concat_ws('|',
            CASE WHEN exact_duplicate_count > 1 THEN 'EXACT_DUPLICATE_CANONICAL' END,
            CASE WHEN is_live_streamer = -124 THEN 'IS_LIVE_STREAMER_UNKNOWN_SENTINEL' END,
            CASE WHEN is_live_streamer IS NULL THEN 'IS_LIVE_STREAMER_MISSING' END,
            CASE WHEN is_live_streamer NOT IN (-124,0,1) THEN 'IS_LIVE_STREAMER_UNEXPECTED' END
          ) AS quality_flags
        """
        quality_specs = {
            "is_live_streamer_sentinel": "is_live_streamer = -124",
            "is_live_streamer_missing": "is_live_streamer IS NULL",
        }
    else:
        derived = """
          CASE WHEN tag IS NULL OR trim(tag) = '' THEN 'UNKNOWN_TAG' ELSE tag END AS tag_clean,
          (tag IS NULL OR trim(tag) = '') AS tag_missing,
          (upload_dt IS NULL) AS upload_dt_missing,
          (video_duration IS NULL OR video_duration <= 0) AS video_duration_missing_or_nonpositive,
          CASE WHEN tag IS NULL OR trim(tag) = '' OR upload_dt IS NULL
                     OR video_duration IS NULL OR video_duration <= 0 OR exact_duplicate_count > 1
               THEN 'valid_with_warning' ELSE 'valid' END AS record_status,
          concat_ws('|',
            CASE WHEN exact_duplicate_count > 1 THEN 'EXACT_DUPLICATE_CANONICAL' END,
            CASE WHEN tag IS NULL OR trim(tag) = '' THEN 'TAG_MISSING' END,
            CASE WHEN upload_dt IS NULL THEN 'UPLOAD_DT_MISSING' END,
            CASE WHEN video_duration IS NULL OR video_duration <= 0 THEN 'VIDEO_DURATION_MISSING_OR_NONPOSITIVE' END
          ) AS quality_flags
        """
        quality_specs = {
            "missing_tag": "tag IS NULL OR trim(tag) = ''",
            "missing_upload_date": "upload_dt IS NULL",
            "missing_or_nonpositive_video_duration": "video_duration IS NULL OR video_duration <= 0",
        }
    copy_query(con, f"""
      SELECT {columns}, source_row_number, duplicate_group_id, {derived},
             {qliteral(source_table)} AS source_table, {qliteral(CLEANING_VERSION)} AS cleaning_version
      FROM dimension_work
      WHERE exclusion_reason IS NULL
      ORDER BY {qident(entity_key)}, source_row_number
    """, stage / "silver" / silver_name, "parquet")
    copy_query(con, f"""
      SELECT {qliteral(source_table)} AS source_table, {qliteral(entity_key)} AS entity_key_name,
             CAST({qident(entity_key)} AS VARCHAR) AS entity_key_value,
             source_row_number, payload_json, exclusion_reason,
             {qliteral(CLEANING_VERSION)} AS cleaning_version
      FROM dimension_work
      WHERE exclusion_reason IS NOT NULL
    """, stage / "work" / f"dimension_conflicts_{source_table}.parquet", "parquet")

    raw_rows = int(scalar(con, f"SELECT count(*) FROM {ranked}"))
    canonical_rows = int(scalar(con, "SELECT count(*) FROM dimension_work"))
    conflicts = int(scalar(con, "SELECT count(*) FROM dimension_work WHERE exclusion_reason IS NOT NULL"))
    silver_rows = canonical_rows - conflicts
    reconciliation = {
        "source_table": source_table,
        "raw_rows": raw_rows,
        "canonical_rows": canonical_rows,
        "duplicate_copy_rows": raw_rows - canonical_rows,
        "conflicting_rows": conflicts,
        "invalid_domain_rows": 0,
        "label_formula_mismatch_rows": 0,
        "silver_rows": silver_rows,
        "reconciled": raw_rows == (raw_rows - canonical_rows) + conflicts + silver_rows,
    }
    quality = [{"source_table": source_table, "rule": "exact_duplicate_rows", "affected_rows": raw_rows - canonical_rows}]
    quality.extend(
        {"source_table": source_table, "rule": rule, "affected_rows": int(scalar(con, f"SELECT count(*) FROM dimension_work WHERE {predicate}"))}
        for rule, predicate in quality_specs.items()
    )
    con.execute("DROP TABLE dimension_work")
    con.execute(f"DROP TABLE {ranked}")
    return reconciliation, duplicate_groups, [], quality


def merge_parquets(con: duckdb.DuckDBPyConnection, inputs: list[Path], output: Path) -> None:
    paths = "[" + ", ".join(qliteral(p) for p in inputs) + "]"
    copy_query(con, f"SELECT * FROM read_parquet({paths}, union_by_name=true)", output, "parquet")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_flag_dictionary(con: duckdb.DuckDBPyConnection, output: Path) -> None:
    rows = [
        ("EXACT_DUPLICATE_CANONICAL", "all", "Canonical row retained from an exact-duplicate group", "warning"),
        ("DURATION_MISSING_OR_NONPOSITIVE", "events", "duration_ms is missing or non-positive; exclude from duration-dependent features", "warning"),
        ("PLAY_TIME_GREATER_THAN_DURATION", "events", "Raw play time exceeds duration; raw value retained and capped ratio added", "warning"),
        ("SAME_TIMESTAMP_MULTIPLE_VIDEOS", "events", "Multiple videos share one user timestamp batch; no within-batch order is inferred", "information"),
        ("LONG_VIEW_FORMULA_MISMATCH", "events", "Official label differs from the public duration/play-time formula", "quarantine"),
        ("IS_LIVE_STREAMER_UNKNOWN_SENTINEL", "users", "Raw value -124 mapped to an explicit unknown category", "warning"),
        ("TAG_MISSING", "videos", "Missing tag mapped to UNKNOWN_TAG in tag_clean", "warning"),
        ("UPLOAD_DT_MISSING", "videos", "Upload date is missing and retained as null", "warning"),
        ("VIDEO_DURATION_MISSING_OR_NONPOSITIVE", "videos", "Video basic duration is missing or non-positive", "warning"),
    ]
    con.execute("CREATE OR REPLACE TEMP TABLE flag_dictionary(flag_code VARCHAR, applies_to VARCHAR, meaning VARCHAR, severity VARCHAR)")
    con.executemany("INSERT INTO flag_dictionary VALUES (?, ?, ?, ?)", rows)
    copy_query(con, "SELECT * FROM flag_dictionary ORDER BY applies_to, flag_code", output, "parquet")
    con.execute("DROP TABLE flag_dictionary")


def markdown_report(
    run_id: str,
    reconciliations: list[dict[str, Any]],
    quality: list[dict[str, Any]],
    label_rates: list[dict[str, Any]],
    gates: dict[str, Any],
) -> str:
    lines = [
        "# KuaiRand-1K Silver 数据清洗报告",
        "",
        f"- 运行 ID：`{run_id}`",
        f"- 清洗版本：`{CLEANING_VERSION}`",
        "- 原始 CSV：只读，未覆盖、未移动",
        "- 事后统计表：未进入 Silver 特征层",
        "",
        "## 行数对账",
        "",
        "| 数据源 | 原始行 | canonical 行 | 重复副本 | 冲突隔离 | 非法域隔离 | 公式疑点隔离 | Silver 行 | 对账 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in reconciliations:
        lines.append(
            f"| {row['source_table']} | {row['raw_rows']:,} | {row['canonical_rows']:,} | "
            f"{row['duplicate_copy_rows']:,} | {row['conflicting_rows']:,} | {row['invalid_domain_rows']:,} | "
            f"{row['label_formula_mismatch_rows']:,} | {row['silver_rows']:,} | "
            f"{'PASS' if row['reconciled'] else 'FAIL'} |"
        )
    lines.extend(["", "## 主要质量规则", "", "| 数据源 | 规则 | 影响行数 |", "|---|---|---:|"])
    for row in quality:
        lines.append(f"| {row['source_table']} | {row['rule']} | {row['affected_rows']:,} |")
    lines.extend(["", "## 标签率变化", "", "| 数据源 | 标签 | 原始率 | canonical 率 | Silver 率 | Silver-原始（百分点） |", "|---|---|---:|---:|---:|---:|"])
    for row in label_rates:
        def pct(value: Any) -> str:
            return "NA" if value is None else f"{100 * float(value):.4f}%"
        delta = row["silver_minus_raw_rate"]
        lines.append(
            f"| {row['source_table']} | {row['label']} | {pct(row['raw_rate'])} | "
            f"{pct(row['canonical_rate'])} | {pct(row['silver_rate'])} | "
            f"{'NA' if delta is None else f'{100 * float(delta):+.4f}'} |"
        )
    lines.extend(["", "## 验收门禁", "", "| 门禁 | 结果 | 证据 |", "|---|---|---|"])
    for name, result in gates.items():
        lines.append(f"| {name} | {'PASS' if result['passed'] else 'FAIL'} | {result['detail']} |")
    lines.extend([
        "",
        "## 表述边界",
        "",
        "Silver 只代表按已批准确定性规则得到的可追溯数据层，不代表标签一定正确，也不代表模型已经有效。"
        "`long_view` 公式不一致行被称为公式一致性疑点，而不是直接断言官方标签错误。"
        "同一时间戳内的多个视频被保留为批次，后续只能使用严格更早时间戳的反馈。",
        "",
    ])
    return "\n".join(lines)


def publish(stage: Path, project_root: Path) -> list[Path]:
    mappings = {
        stage / "silver": project_root / "data" / "silver",
        stage / "quarantine": project_root / "data" / "quarantine",
        stage / "manifests": project_root / "data" / "manifests",
        stage / "reports": project_root / "reports" / "generated",
    }
    files = [p for source in mappings for p in source.rglob("*") if p.is_file()]
    collisions = []
    for source_root, target_root in mappings.items():
        for source in source_root.rglob("*"):
            if source.is_file() and (target_root / source.relative_to(source_root)).exists():
                collisions.append(str(target_root / source.relative_to(source_root)))
    if collisions:
        raise FileExistsError("Refusing to overwrite existing Silver artifacts: " + ", ".join(collisions))
    published: list[Path] = []
    for source_root, target_root in mappings.items():
        target_root.mkdir(parents=True, exist_ok=True)
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            destination = target_root / source.relative_to(source_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            published.append(destination)
    return published


def run(project_root: Path, validate_only: bool = False) -> int:
    project_root = project_root.resolve()
    context = validate_inputs(project_root)
    if validate_only:
        print(json.dumps({
            "status": "validated",
            "project_root": str(project_root),
            "raw_files": {k: str(v) for k, v in context["raw_paths"].items()},
            "cleaning_status": context["rules"].get("status"),
            "experiment_status": context["experiment"].get("status"),
        }, ensure_ascii=False, indent=2))
        return 0

    run_id = datetime.now().strftime("silver-%Y%m%d-%H%M%S")
    stage = project_root / "artifacts" / ".silver_build_tmp" / run_id
    for name in ["silver", "quarantine", "manifests", "reports", "work", "duckdb-temp"]:
        (stage / name).mkdir(parents=True, exist_ok=False)

    raw_stats = {
        name: {"path": str(path), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for name, path in context["raw_paths"].items()
    }
    started_at = utc_now()
    con = duckdb.connect(str(stage / "work" / "silver_build.duckdb"))
    configure_duckdb(con, stage / "duckdb-temp")
    reconciliations: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    label_rates: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    try:
        for source in ["early_standard", "late_standard", "random"]:
            result = process_event(
                con,
                source,
                context["raw_paths"][source],
                context["contract"]["event_tables"][source],
                stage,
            )
            reconciliation, duplicates, rates, rules = result
            reconciliations.append(reconciliation)
            duplicate_groups.extend(duplicates)
            label_rates.extend(rates)
            quality.extend(rules)
            print(json.dumps({"stage": "event_complete", **reconciliation}, ensure_ascii=False), flush=True)

        for args in [
            ("users", context["raw_paths"]["users"], USER_SCHEMA, "user_id", "users.parquet"),
            ("videos_basic", context["raw_paths"]["videos_basic"], VIDEO_SCHEMA, "video_id", "videos_basic.parquet"),
        ]:
            reconciliation, duplicates, rates, rules = process_dimension(con, *args, stage)
            reconciliations.append(reconciliation)
            duplicate_groups.extend(duplicates)
            label_rates.extend(rates)
            quality.extend(rules)
            print(json.dumps({"stage": "dimension_complete", **reconciliation}, ensure_ascii=False), flush=True)

        work = stage / "work"
        merge_parquets(con, sorted(work.glob("conflict_*.parquet")), stage / "quarantine" / "conflicting_event_groups.parquet")
        merge_parquets(con, sorted(work.glob("invalid_*.parquet")), stage / "quarantine" / "invalid_domain_rows.parquet")
        merge_parquets(con, sorted(work.glob("mismatch_*.parquet")), stage / "quarantine" / "label_formula_mismatch_rows.parquet")
        merge_parquets(con, sorted(work.glob("dimension_conflicts_*.parquet")), stage / "quarantine" / "dimension_conflicts.parquet")
        merge_parquets(con, sorted(work.glob("duplicate_copies_*.parquet")), stage / "manifests" / "duplicate_copies.parquet")
        write_flag_dictionary(con, stage / "silver" / "quality_flags_dictionary.parquet")

        write_rows(stage / "manifests" / "row_reconciliation.csv", reconciliations)
        write_rows(stage / "manifests" / "duplicate_audit.csv", duplicate_groups)
        write_rows(stage / "manifests" / "label_rate_before_after.csv", label_rates)
        write_rows(stage / "manifests" / "quality_rule_summary.csv", quality)

        event_outputs = [
            stage / "silver" / "events_early_standard.parquet",
            stage / "silver" / "events_late_standard.parquet",
            stage / "silver" / "events_random.parquet",
        ]
        binary_checks = []
        for path in event_outputs:
            for column in context["contract"]["binary_columns"]:
                binary_checks.append(
                    int(scalar(con, f"SELECT count(*) FROM read_parquet({qliteral(path)}) WHERE {qident(column)} IS NULL OR {qident(column)} NOT IN (0,1)"))
                )
        join_user_missing = int(scalar(con, f"""
          SELECT count(*) FROM read_parquet({qliteral(event_outputs)}) e
          LEFT JOIN read_parquet({qliteral(stage / 'silver' / 'users.parquet')}) u USING (user_id)
          WHERE u.user_id IS NULL
        """))
        join_video_missing = int(scalar(con, f"""
          SELECT count(*) FROM read_parquet({qliteral(event_outputs)}) e
          LEFT JOIN read_parquet({qliteral(stage / 'silver' / 'videos_basic.parquet')}) v USING (video_id)
          WHERE v.video_id IS NULL
        """))
        gates = {
            "row_reconciliation": {
                "passed": all(row["reconciled"] for row in reconciliations),
                "detail": "all source tables balance",
            },
            "silver_binary_domain": {
                "passed": sum(binary_checks) == 0,
                "detail": f"invalid binary cells={sum(binary_checks)}",
            },
            "user_join_coverage": {
                "passed": join_user_missing == 0,
                "detail": f"missing event-user joins={join_user_missing}",
            },
            "video_join_coverage": {
                "passed": join_video_missing == 0,
                "detail": f"missing event-video joins={join_video_missing}",
            },
            "post_hoc_statistic_excluded": {
                "passed": not any("statistic" in p.name for p in (stage / "silver").glob("*")),
                "detail": "video_features_statistic_1k.csv was not materialized or joined",
            },
        }
        if not all(gate["passed"] for gate in gates.values()):
            raise RuntimeError("One or more Silver acceptance gates failed: " + json.dumps(gates, ensure_ascii=False))

        raw_manifest = []
        for name, path in context["raw_paths"].items():
            current = path.stat()
            before = raw_stats[name]
            unchanged = current.st_size == before["size_bytes"] and current.st_mtime_ns == before["mtime_ns"]
            if not unchanged:
                raise RuntimeError(f"Raw file changed during the run: {path}")
            raw_manifest.append({
                "source_name": name,
                "path": str(path),
                "size_bytes": current.st_size,
                "mtime_ns": current.st_mtime_ns,
                "sha256": file_sha256(path),
                "used_in_silver": name != "videos_statistic",
            })
        (stage / "manifests" / "raw_file_manifest.json").write_text(
            json.dumps(raw_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        report = markdown_report(run_id, reconciliations, quality, label_rates, gates)
        (stage / "reports" / "silver_cleaning_report.md").write_text(report, encoding="utf-8")
        con.close()
        for disposable in [stage / "work", stage / "duckdb-temp"]:
            shutil.rmtree(disposable, ignore_errors=True)

        config_files = [
            project_root / "configs" / "data_paths.yaml",
            project_root / "configs" / "data_contract.yaml",
            project_root / "configs" / "cleaning_rules.yaml",
            project_root / "configs" / "experiment_v001.yaml",
        ]
        manifest = {
            "run_id": run_id,
            "status": "complete",
            "cleaning_version": CLEANING_VERSION,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "project_root": str(project_root),
            "python": sys.version,
            "platform": platform.platform(),
            "duckdb_version": duckdb.__version__,
            "config_sha256": {p.name: file_sha256(p) for p in config_files},
            "config_bundle_sha256": json_hash({p.name: file_sha256(p) for p in config_files}),
            "code_sha256": file_sha256(Path(__file__)),
            "gates": gates,
            "row_reconciliation": reconciliations,
        }
        (stage / "manifests" / "silver_run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        published = publish(stage, project_root)
        shutil.rmtree(stage, ignore_errors=True)
        output_manifest = {
            "run_id": run_id,
            "files": [
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
                for path in published
            ],
        }
        output_manifest_path = project_root / "data" / "manifests" / "silver_output_manifest.json"
        output_manifest_path.write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "complete", "run_id": run_id, "published_files": len(published) + 1}, ensure_ascii=False), flush=True)
        return 0
    except Exception:
        try:
            con.close()
        except Exception:
            pass
        failure = {"run_id": run_id, "status": "failed", "started_at_utc": started_at, "failed_at_utc": utc_now()}
        (stage / "failure_manifest.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the audited KuaiRand-1K Silver layer")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    return run(args.project_root, validate_only=args.validate_only)


if __name__ == "__main__":
    raise SystemExit(main())
