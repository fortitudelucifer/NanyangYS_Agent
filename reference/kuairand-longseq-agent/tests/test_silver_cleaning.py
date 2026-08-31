import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kuairand_longseq.data import silver_cleaning_compat as cleaning  # noqa: E402


def test_sql_quoting_and_path_lists() -> None:
    assert cleaning.impl.qident('a"b') == '"a""b"'
    assert cleaning.impl.qliteral(Path("a'b.csv")) == "'a''b.csv'"
    assert cleaning.impl.qliteral([Path("a.parquet"), Path("b.parquet")]) == "['a.parquet', 'b.parquet']"


def test_event_classification_sql(tmp_path: Path) -> None:
    source = tmp_path / "events.csv"
    source.write_text(
        ",".join(cleaning.EVENT_SCHEMA) + "\n"
        "1,10,20220408,930,1000,1,0,0,0,0,0,1,20000,20000,0,0,0,0,1\n"
        "1,10,20220408,930,1000,1,0,0,0,0,0,1,20000,20000,0,0,0,0,1\n"
        "1,11,20220408,930,1000,0,0,0,0,0,0,0,0,-1,0,0,0,0,1\n"
        "1,12,20220431,930,1001,0,0,0,0,0,0,0,0,10000,0,0,0,0,1\n",
        encoding="utf-8",
    )
    con = duckdb.connect()
    cleaning.create_ranked_table(con, "ranked_event", "early_standard", source, cleaning.EVENT_SCHEMA)
    con.execute(
        "CREATE TEMP TABLE event_work AS "
        + cleaning.impl.event_work_sql("ranked_event", 0, 20220408, 20220421)
    )
    assert con.execute("SELECT count(*) FROM ranked_event WHERE exact_duplicate_rank > 1").fetchone()[0] == 1
    rows = con.execute(
        "SELECT video_id, exclusion_reason, duration_missing_or_nonpositive "
        "FROM event_work ORDER BY video_id"
    ).fetchall()
    assert rows == [
        (10, None, False),
        (11, None, True),
        (12, "INVALID_DOMAIN", False),
    ]
