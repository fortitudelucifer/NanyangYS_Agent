"""Compatibility fixes kept separate because the current Codex task has a stale cwd."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import silver_cleaning as impl


_original_qliteral = impl.qliteral
_original_event_work_sql = impl.event_work_sql


def qliteral(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_original_qliteral(Path(item)) for item in value) + "]"
    return _original_qliteral(value)


def event_work_sql(ranked_table: str, expected_rand: int, date_min: int, date_max: int) -> str:
    sql = _original_event_work_sql(ranked_table, expected_rand, date_min, date_max)
    predicate = f"date IS NULL OR date < {date_min} OR date > {date_max}"
    calendar_check = "try_strptime(CAST(date AS VARCHAR), '%Y%m%d') IS NULL"
    sql = sql.replace(
        f"OR {predicate}\n        OR hourmin",
        f"OR {predicate}\n        OR {calendar_check}\n        OR hourmin",
    )
    sql = sql.replace(
        f"CASE WHEN {predicate} THEN 'INVALID_DATE' END,",
        f"CASE WHEN {predicate} OR {calendar_check} THEN 'INVALID_DATE' END,",
    )
    return sql


impl.qliteral = qliteral
impl.event_work_sql = event_work_sql

main = impl.main
run = impl.run
EVENT_SCHEMA = impl.EVENT_SCHEMA
create_ranked_table = impl.create_ranked_table

