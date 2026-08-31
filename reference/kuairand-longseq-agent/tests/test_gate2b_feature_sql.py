import sys
import unittest
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kuairand_longseq.features.gate2b_feature_sql import _entity_sql  # noqa: E402


USER_STATE_SQL = """
CREATE TEMP TABLE user_batch AS
SELECT user_id, time_ms, count(*)::BIGINT batch_event_n,
       sum(long_view)::BIGINT batch_positive_n
FROM events GROUP BY user_id, time_ms;
CREATE TEMP TABLE user_state AS
SELECT user_id,time_ms,
       row_number() OVER(PARTITION BY user_id ORDER BY time_ms)-1 prior_batch_n,
       coalesce(sum(batch_event_n) OVER(PARTITION BY user_id ORDER BY time_ms
         ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0)::BIGINT prior_event_n,
       coalesce(sum(batch_positive_n) OVER(PARTITION BY user_id ORDER BY time_ms
         ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0)::BIGINT prior_positive_n,
       coalesce(sum(batch_event_n) OVER(PARTITION BY user_id ORDER BY time_ms
         ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),0)::BIGINT w10_event_n,
       coalesce(sum(batch_event_n) OVER(PARTITION BY user_id ORDER BY time_ms
         ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING),0)::BIGINT w50_event_n,
       coalesce(sum(batch_event_n) OVER(PARTITION BY user_id ORDER BY time_ms
         ROWS BETWEEN 200 PRECEDING AND 1 PRECEDING),0)::BIGINT w200_event_n
FROM user_batch;
"""


class Gate2BFeatureSqlTests(unittest.TestCase):
    def _connection(self, first_batch_labels=(0, 1)):
        con = duckdb.connect()
        con.execute(
            "CREATE TABLE events(user_id BIGINT,time_ms BIGINT,long_view INTEGER,cat_author BIGINT)"
        )
        rows = [
            (1, 100, first_batch_labels[0], 9),
            (1, 100, first_batch_labels[1], 9),
            (1, 200, 1, 9),
            (1, 300, 0, 8),
        ]
        con.executemany("INSERT INTO events VALUES (?,?,?,?)", rows)
        con.execute(USER_STATE_SQL)
        return con

    def test_same_timestamp_batch_is_excluded_as_a_whole(self):
        con = self._connection()
        state = con.execute(
            "SELECT prior_batch_n,prior_event_n,prior_positive_n FROM user_state WHERE time_ms=100"
        ).fetchone()
        self.assertEqual(state, (0, 0, 0))
        next_state = con.execute(
            "SELECT prior_batch_n,prior_event_n,prior_positive_n FROM user_state WHERE time_ms=200"
        ).fetchone()
        self.assertEqual(next_state, (1, 2, 1))
        con.close()

    def test_current_batch_label_flip_does_not_change_current_features(self):
        first = self._connection((0, 1))
        second = self._connection((1, 1))
        current_first = first.execute("SELECT * EXCLUDE(user_id,time_ms) FROM user_state WHERE time_ms=100").fetchone()
        current_second = second.execute("SELECT * EXCLUDE(user_id,time_ms) FROM user_state WHERE time_ms=100").fetchone()
        self.assertEqual(current_first, current_second)
        future_first = first.execute("SELECT prior_positive_n FROM user_state WHERE time_ms=200").fetchone()[0]
        future_second = second.execute("SELECT prior_positive_n FROM user_state WHERE time_ms=200").fetchone()[0]
        self.assertNotEqual(future_first, future_second)
        first.close()
        second.close()

    def test_windows_are_monotone_and_entity_state_is_strict(self):
        con = self._connection()
        violations = con.execute(
            "SELECT count(*) FROM user_state WHERE w10_event_n>w50_event_n OR w50_event_n>w200_event_n OR w200_event_n>prior_event_n"
        ).fetchone()[0]
        self.assertEqual(violations, 0)
        con.execute("CREATE TEMP TABLE event_meta AS SELECT * FROM events")
        con.execute(_entity_sql("author", "cat_author", "cat_author <> -1"))
        at_100 = con.execute(
            "SELECT prior_n,prior_positive_n,prior_time_ms FROM author_state WHERE time_ms=100"
        ).fetchone()
        self.assertEqual(at_100, (0, 0, None))
        at_200 = con.execute(
            "SELECT prior_n,prior_positive_n,prior_time_ms FROM author_state WHERE time_ms=200"
        ).fetchone()
        self.assertEqual(at_200, (2, 1, 100))
        con.close()


if __name__ == "__main__":
    unittest.main()
