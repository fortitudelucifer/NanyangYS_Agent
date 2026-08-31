"""KuaiRand-1K tab-domain correction layered over the reviewed Silver builder."""

from __future__ import annotations

from . import silver_cleaning_compat as compat


impl = compat.impl
_previous_event_work_sql = impl.event_work_sql


def event_work_sql(ranked_table: str, expected_rand: int, date_min: int, date_max: int) -> str:
    sql = _previous_event_work_sql(ranked_table, expected_rand, date_min, date_max)
    return sql.replace("tab < 0 OR tab > 4", "tab < 0 OR tab > 14")


impl.event_work_sql = event_work_sql

main = impl.main
run = impl.run

