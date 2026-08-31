from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse

from kuairand_longseq.features.history_value_feature_sql import (
    materialize_random_features,
    materialize_standard_features,
)
from kuairand_longseq.models import history_value_gpu as gpu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_history_value_gpu_confirmation_v001 as runner  # noqa: E402


EVENT_FIELDS = {
    "source_table": pa.string(), "source_row_number": pa.int64(),
    "user_id": pa.int64(), "video_id": pa.int64(), "event_date": pa.date32(),
    "time_ms": pa.int64(), "tab": pa.int64(), "long_view": pa.int8(),
}


def write_events(path: Path, rows: list[dict[str, object]], *, mismatch: bool = False) -> None:
    fields = dict(EVENT_FIELDS)
    if mismatch:
        fields["exclusion_reason"] = pa.string()
    table = pa.table({
        name: pa.array([
            date.fromisoformat(str(row.get(name))) if name == "event_date" else row.get(name)
            for row in rows
        ], type=dtype)
        for name, dtype in fields.items()
    })
    pq.write_table(table, path)


def write_videos(path: Path) -> None:
    pq.write_table(pa.table({
        "video_id": pa.array([10, 11, 12], type=pa.int64()),
        "author_id": pa.array([1, 1, 2], type=pa.int64()),
        "music_id": pa.array([1, 2, 3], type=pa.int64()),
        "video_type": ["NORMAL", "NORMAL", "NORMAL"],
        "upload_type": ["LongImport", "LongImport", "LongImport"],
        "music_type": pa.array([1, 1, 1], type=pa.int64()),
        "tag_clean": ["a", "b", "c"],
        "tag_missing": [False, False, False],
        "video_duration": [20_000.0, 30_000.0, 40_000.0],
        "upload_dt": pa.array([date(2022, 4, 1)] * 3, type=pa.date32()),
        "server_width": [1080.0] * 3, "server_height": [1920.0] * 3,
    }), path)


def event(source: str, row: int, video: int, day: str, time_ms: int, label: int, tab: int = 1) -> dict[str, object]:
    return {
        "source_table": source, "source_row_number": row, "user_id": 1,
        "video_id": video, "event_date": day, "time_ms": time_ms,
        "tab": tab, "long_view": label,
    }


def test_standard_history_is_strict_at_timestamp_batch(tmp_path: Path) -> None:
    early, mismatch, videos, output = (tmp_path / name for name in ("early.parquet", "mismatch.parquet", "videos.parquet", "out.parquet"))
    write_events(early, [
        event("early_standard", 1, 10, "2022-04-17", 100, 1),
        event("early_standard", 2, 11, "2022-04-18", 200, 1),
        event("early_standard", 3, 12, "2022-04-18", 200, 0),
    ])
    write_events(mismatch, [], mismatch=True)
    write_videos(videos)
    con = duckdb.connect()
    result = materialize_standard_features(
        con, early_path=early, late_path=None, mismatch_path=mismatch,
        videos_path=videos, output_path=output, end_date="2022-04-18",
        target_start="2022-04-18", target_end="2022-04-18", expected_target_rows=2,
    )
    con.close()
    table = pq.read_table(output).to_pydict()
    target = [index for index, day in enumerate(table["event_date"]) if str(day) == "2022-04-18"]
    assert [table["prior_event_n"][index] for index in target] == [1, 1]
    assert [table["prior_positive_n"][index] for index in target] == [1, 1]
    assert result["pit_violations"] == 0


def test_random_labels_never_update_random_history(tmp_path: Path) -> None:
    early, late, random, mismatch, videos, output = (tmp_path / name for name in ("early.parquet", "late.parquet", "random.parquet", "mismatch.parquet", "videos.parquet", "out.parquet"))
    write_events(early, [event("early_standard", 1, 10, "2022-04-21", 100, 1)])
    write_events(late, [event("late_standard", 1, 11, "2022-04-22", 300, 0)])
    write_events(random, [
        event("random", 1, 11, "2022-04-22", 200, 0, tab=0),
        event("random", 2, 12, "2022-04-22", 250, 1, tab=1),
    ])
    write_events(mismatch, [], mismatch=True)
    write_videos(videos)
    con = duckdb.connect()
    result = materialize_random_features(
        con, early_path=early, late_path=late, random_path=random,
        mismatch_path=mismatch, videos_path=videos, output_path=output,
        expected_target_rows=2,
    )
    con.close()
    table = pq.read_table(output).to_pydict()
    assert table["prior_event_n"] == [1, 1]
    assert table["prior_positive_n"] == [1, 1]
    assert result["random_labels_can_update_history"] is False


def test_gpu_trajectory_refuses_cpu() -> None:
    matrix = sparse.csr_matrix(np.eye(2, dtype=np.float32))
    with pytest.raises(RuntimeError, match="forbids non-CUDA"):
        gpu.fit_trajectory(
            matrix, np.array([0, 1]), device=__import__("torch").device("cpu"),
            optimizer_name="ADAM", learning_rate=.03, checkpoints=[2], alpha=1e-4,
        )


def test_adequacy_formula() -> None:
    passing = gpu.adequacy(.501, .5, reference_converged=True)
    failing = gpu.adequacy(.504, .5, reference_converged=True)
    assert passing["adequacy_passed"] is True
    assert failing["adequacy_passed"] is False


def test_stage_access_fails_closed_out_of_order() -> None:
    gpu.assert_stage_access("validation", {"preflight"})
    with pytest.raises(RuntimeError, match="requires validation"):
        gpu.assert_stage_access("sealed_test", {"preflight"})
    with pytest.raises(RuntimeError, match="requires sealed_test"):
        gpu.assert_stage_access("random_audit", {"preflight", "validation"})


def test_artifact_hash_manifest_covers_files(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two", encoding="utf-8")
    runner.finalize_hashes(tmp_path)
    payload = json.loads((tmp_path / "artifact_hash_manifest.json").read_text())
    assert {Path(row["path"]).name for row in payload["artifacts"]} == {"one.txt", "two.txt"}
    for row in payload["artifacts"]:
        assert row["sha256"] == runner.sha256_file(tmp_path / Path(row["path"]).name)


def test_release_requires_exact_hash_approval_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = tmp_path / "approval.json"
    monkeypatch.setattr(runner, "APPROVAL_PATH", receipt)
    with pytest.raises(runner.ContractStop, match="receipt is missing"):
        runner.verify_approval_receipt("abc")
    receipt.write_text(json.dumps({
        "contract_id": "history_value_gpu_confirmation_v001",
        "contract_sha256": "wrong",
        "execution_authorized": True,
        "automatic_ordered_transitions_authorized": True,
        "approved_by": "project_owner",
    }), encoding="utf-8")
    with pytest.raises(runner.ContractStop, match="contract_sha256"):
        runner.verify_approval_receipt("abc")
