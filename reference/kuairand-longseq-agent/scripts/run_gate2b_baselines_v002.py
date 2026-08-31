"""Run the scoped Gate 2B fixed-row Train-only baseline study.

The script is fail-closed: it uses four explicit input paths, queries only three
of them, verifies release hashes, never discovers files recursively, and stops
before daily fitting when the frozen BL2 search gate fails.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
import yaml
from matplotlib import pyplot as plt
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kuairand_longseq.evaluation.gate2b_metrics import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_MULTIPLICITY_SHA256,
    ece_equal_width,
    make_multiplicities,
    paired_user_cluster_bootstrap,
    point_metrics,
)
from kuairand_longseq.features.gate2b_feature_sql import (  # noqa: E402
    materialize_gate2b_features,
)
from kuairand_longseq.models.gate2b_baselines import (  # noqa: E402
    FeatureArrays,
    fit_predict_sgd,
    prepare_origin_matrices,
)

matplotlib.use("Agg")

CONTRACT_PATH = PROJECT_ROOT / "configs/gate2b_fixed_row_baseline_contract_v002.yaml"
RELEASE_DIR = PROJECT_ROOT / "reports/generated/gate2b_baselines_v002"
QUICK_DIR = PROJECT_ROOT / "reports/generated/gate2b_baselines_v002_quick"
REPORT_PATH = PROJECT_ROOT / "reports/analysis/gate2b_baseline_results_v002.md"
FIGURE_PATH = PROJECT_ROOT / "reports/figures/gate2b_baseline_results_v002.png"
CHECKPOINT_PATH = PROJECT_ROOT / "reports/gate2b_checkpoint_2026-08-15.md"

SEARCH_ORIGINS = ["2022-04-11", "2022-04-14", "2022-04-17"]
DAILY_ORIGINS = [f"2022-04-{day:02d}" for day in range(11, 18)]
ALPHAS = [1e-6, 1e-5, 1e-4]
SEED = 20260814
MAX_ITER = 15
MAX_FITS = 32

INPUT_PATHS = {
    "early_train": PROJECT_ROOT / "data/silver/events_early_standard.parquet",
    "formula_mismatch": PROJECT_ROOT / "data/quarantine/label_formula_mismatch_rows.parquet",
    "videos_basic": PROJECT_ROOT / "data/silver/videos_basic.parquet",
    "users_hash_only": PROJECT_ROOT / "data/silver/users.parquet",
}

OUTPUT_IDENTITY_COLUMNS = [
    "source_table",
    "source_row_number",
    "user_id",
    "video_id",
    "event_date",
    "time_ms",
    "long_view",
    "prior_batch_n",
]


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: yaml.Loader, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def progress(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_contract() -> tuple[dict[str, Any], str]:
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    contract = yaml.load(text, Loader=UniqueKeyLoader)
    if contract["status"] != "approved_for_gate2b_train_only_fixed_row_fitting":
        raise RuntimeError("Gate 2B scoped fitting is not authorized by the executable contract")
    if contract["authorization"]["does_not_authorize"] != [
        "silver_recleaning",
        "gold_build_or_gold_sample_id_claim",
        "general_formal_model_training",
        "sequence_or_neural_models",
        "validation_access",
        "late_table_access",
        "restricted_test_access",
        "random_table_access",
        "video_statistic_feature_access",
    ]:
        raise RuntimeError("authorization boundary changed unexpectedly")
    return contract, hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_inputs(contract: dict[str, Any], full_sha256: bool) -> list[dict[str, Any]]:
    expected_by_path = {entry["path"]: entry for entry in contract["input_allowlist"]}
    verified: list[dict[str, Any]] = []
    for role, path in INPUT_PATHS.items():
        key = relative(path)
        entry = expected_by_path.get(key)
        if entry is None:
            raise RuntimeError(f"input absent from explicit contract allowlist: {key}")
        observed_size = path.stat().st_size
        if observed_size != entry["expected_size_bytes"]:
            raise RuntimeError(f"input size mismatch for {key}")
        observed_hash = sha256_file(path) if full_sha256 else None
        if full_sha256 and observed_hash != entry["expected_sha256"]:
            raise RuntimeError(f"input SHA-256 mismatch for {key}")
        verified.append(
            {
                "role": role,
                "path": key,
                "query_allowed": bool(entry["query_allowed"]),
                "observed_size_bytes": observed_size,
                "expected_sha256": entry["expected_sha256"],
                "observed_sha256": observed_hash,
                "size_verified": True,
                "sha256_verified": full_sha256,
            }
        )
    return verified


def alpha_slug(alpha: float) -> str:
    return f"{alpha:.0e}".replace("-", "m").replace("+", "p")


def search_trials(alphas: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        rows.append(
            {
                "trial_id": f"BL1_S_ID_CONTENT_V1_A{alpha_slug(alpha)}",
                "prediction_column": f"p_bl1_a{alpha_slug(alpha)}",
                "model_id": "BL1",
                "bundle": "S_ID_CONTENT_V1",
                "alpha": alpha,
                "seed": SEED,
                "max_iter": MAX_ITER,
            }
        )
    for bundle, short in [("H2_USER_STRICT_V1", "h2")]:
        for alpha in alphas:
            rows.append(
                {
                    "trial_id": f"BL2_{bundle}_A{alpha_slug(alpha)}",
                    "prediction_column": f"p_bl2_{short}_a{alpha_slug(alpha)}",
                    "model_id": "BL2",
                    "bundle": bundle,
                    "alpha": alpha,
                    "seed": SEED,
                    "max_iter": MAX_ITER,
                }
            )
    return rows


def prediction_table(
    data: FeatureArrays,
    indices: np.ndarray,
    probability_columns: dict[str, np.ndarray],
    metadata: dict[str, str],
) -> pa.Table:
    arrays: dict[str, Any] = {name: data.columns[name][indices] for name in OUTPUT_IDENTITY_COLUMNS}
    for name, values in probability_columns.items():
        array = np.asarray(values, dtype=np.float64)
        if array.size != indices.size or not np.isfinite(array).all():
            raise RuntimeError(f"invalid prediction column: {name}")
        arrays[name] = array
    table = pa.table(arrays)
    encoded = {key.encode(): value.encode() for key, value in metadata.items()}
    return table.replace_schema_metadata(encoded)


def metrics_row(
    stage: str,
    origin: str,
    trial: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "origin": origin,
        "trial_id": trial["trial_id"],
        "model_id": trial["model_id"],
        "bundle": trial["bundle"],
        "alpha": trial["alpha"],
        **metrics,
    }


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any], name: str) -> float:
    return float(candidate[name]) - float(baseline[name])


def fit_one(
    ledger: list[dict[str, Any]],
    *,
    stage: str,
    origin: str,
    trial: dict[str, Any],
    train_x: Any,
    y_train: np.ndarray,
    assess_x: Any,
    started: float,
) -> np.ndarray:
    if len(ledger) >= MAX_FITS:
        raise RuntimeError("Gate 2B fit-run budget would be exceeded")
    if (time.perf_counter() - started) / 60.0 >= 60.0:
        raise RuntimeError("Gate 2B elapsed-time budget reached before next fit")
    fit_start = time.perf_counter()
    record: dict[str, Any] = {
        "fit_run_id": f"{stage}_{origin}_{trial['trial_id']}",
        "stage": stage,
        "origin": origin,
        "model_id": trial["model_id"],
        "bundle": trial["bundle"],
        "alpha": trial["alpha"],
        "seed": SEED,
        "train_rows": int(train_x.shape[0]),
        "assessment_rows": int(assess_x.shape[0]),
        "feature_columns": int(train_x.shape[1]),
        "cpu_threads_contract": 8,
        "gpu_used": False,
        "status": "running",
    }
    try:
        with threadpool_limits(limits=1):
            probability, model = fit_predict_sgd(
                train_x,
                y_train,
                assess_x,
                alpha=float(trial["alpha"]),
                seed=SEED,
                max_iter=MAX_ITER,
            )
        record.update(
            {
                "status": "complete",
                "elapsed_seconds": time.perf_counter() - fit_start,
                "epochs": int(model.n_iter_),
                "coefficient_l2_norm": float(np.linalg.norm(model.coef_)),
                "metric_materialized_during_fit": False,
            }
        )
        ledger.append(record)
        return probability
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - fit_start,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "metric_materialized_during_fit": False,
            }
        )
        ledger.append(record)
        raise


def identity_digest(con: duckdb.DuckDBPyConnection, feature_path: Path, where: str) -> str:
    cursor = con.execute(
        f"""
        SELECT source_table, source_row_number
        FROM read_parquet(?) WHERE {where}
        ORDER BY source_table, source_row_number
        """,
        [str(feature_path)],
    )
    digest = hashlib.sha256()
    while rows := cursor.fetchmany(100_000):
        for source_table, source_row_number in rows:
            digest.update(f"{source_table}\t{source_row_number}\n".encode("utf-8"))
    return digest.hexdigest()


def target_manifests(con: duckdb.DuckDBPyConnection, feature_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cohort, where in [("all_canonical", "TRUE")] + [
        (origin, f"event_date = DATE '{origin}'") for origin in DAILY_ORIGINS
    ]:
        count, users, positives = con.execute(
            f"SELECT count(*), count(DISTINCT user_id), sum(long_view) FROM read_parquet(?) WHERE {where}",
            [str(feature_path)],
        ).fetchone()
        rows.append(
            {
                "cohort": cohort,
                "filter": where,
                "identity_key": "source_table|source_row_number",
                "rows": int(count),
                "users": int(users),
                "positives": int(positives),
                "identity_sha256": identity_digest(con, feature_path, where),
                "role": "train_only_source_identity_not_gold_sample_id",
            }
        )
    if rows[0]["identity_sha256"] != "bf058e955f3f8dddaaf2686e6ff2e2a0516bcb6a94d28bc0716b78b0bde0ae0f":
        raise RuntimeError("all-target source identity digest mismatch")
    return rows


def materialize_search_metrics(
    output_dir: Path,
    origins: list[str],
    trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, pa.Table]]:
    rows: list[dict[str, Any]] = []
    pooled_tables: list[pa.Table] = []
    by_origin_table: dict[str, pa.Table] = {}
    for origin in origins:
        path = output_dir / f"search_predictions_origin_{origin}.parquet"
        table = pq.read_table(path)
        by_origin_table[origin] = table
        pooled_tables.append(table)
        y = table["long_view"].to_numpy().astype(np.int8)
        users = table["user_id"].to_numpy().astype(np.int64)
        p0_trial = {
            "trial_id": "BL0_ORIGIN_PREVALENCE",
            "model_id": "BL0",
            "bundle": "GLOBAL_CONSTANT",
            "alpha": "",
        }
        rows.append(metrics_row("search_origin", origin, p0_trial, point_metrics(y, table["p_bl0"].to_numpy(), users)))
        for trial in trials:
            rows.append(
                metrics_row(
                    "search_origin",
                    origin,
                    trial,
                    point_metrics(y, table[trial["prediction_column"]].to_numpy(), users),
                )
            )
    pooled = pa.concat_tables(pooled_tables)
    y = pooled["long_view"].to_numpy().astype(np.int8)
    users = pooled["user_id"].to_numpy().astype(np.int64)
    pooled_by_trial: dict[str, dict[str, Any]] = {}
    for trial in trials:
        metrics = point_metrics(y, pooled[trial["prediction_column"]].to_numpy(), users)
        pooled_by_trial[trial["trial_id"]] = metrics
        rows.append(metrics_row("search_pooled", "pooled_3_origins", trial, metrics))
    return rows, pooled_by_trial, by_origin_table


def select_models(
    trials: list[dict[str, Any]],
    pooled: dict[str, dict[str, Any]],
    by_origin_table: dict[str, pa.Table],
) -> dict[str, Any]:
    bl1_trials = [trial for trial in trials if trial["model_id"] == "BL1"]
    bl1 = max(
        bl1_trials,
        key=lambda trial: (
            pooled[trial["trial_id"]]["average_precision"],
            pooled[trial["trial_id"]]["user_gauc_event_weighted"],
            -pooled[trial["trial_id"]]["log_loss"],
            -pooled[trial["trial_id"]]["brier"],
            trial["alpha"],
        ),
    )
    baseline_metrics = pooled[bl1["trial_id"]]
    candidates: list[dict[str, Any]] = []
    for trial in [item for item in trials if item["model_id"] == "BL2"]:
        candidate_metrics = pooled[trial["trial_id"]]
        positive_origins = 0
        origin_deltas: dict[str, float] = {}
        for origin, table in by_origin_table.items():
            y = table["long_view"].to_numpy().astype(np.int8)
            candidate_ap = point_metrics(
                y,
                table[trial["prediction_column"]].to_numpy(),
                table["user_id"].to_numpy(),
            )["average_precision"]
            baseline_ap = point_metrics(
                y,
                table[bl1["prediction_column"]].to_numpy(),
                table["user_id"].to_numpy(),
            )["average_precision"]
            delta = float(candidate_ap - baseline_ap)
            origin_deltas[origin] = delta
            positive_origins += int(delta > 0)
        gate = {
            "positive_ap_origins": positive_origins,
            "delta_average_precision": metric_delta(candidate_metrics, baseline_metrics, "average_precision"),
            "delta_user_gauc_event_weighted": metric_delta(
                candidate_metrics, baseline_metrics, "user_gauc_event_weighted"
            ),
            "delta_log_loss": metric_delta(candidate_metrics, baseline_metrics, "log_loss"),
            "delta_brier": metric_delta(candidate_metrics, baseline_metrics, "brier"),
            "origin_delta_average_precision": origin_deltas,
        }
        gate["passed"] = bool(
            positive_origins >= 2
            and gate["delta_average_precision"] > 0
            and gate["delta_user_gauc_event_weighted"] >= 0
            and gate["delta_log_loss"] <= 0
            and gate["delta_brier"] <= 0
        )
        candidates.append({"trial": trial, "gate": gate})
    passing = [item for item in candidates if item["gate"]["passed"]]
    selected_bl2 = None
    if passing:
        selected_bl2 = max(
            passing,
            key=lambda item: (
                pooled[item["trial"]["trial_id"]]["average_precision"],
                pooled[item["trial"]["trial_id"]]["user_gauc_event_weighted"],
                -pooled[item["trial"]["trial_id"]]["log_loss"],
                -pooled[item["trial"]["trial_id"]]["brier"],
                item["trial"]["bundle"] == "H2_USER_STRICT_V1",
                item["trial"]["alpha"],
            ),
        )
    return {
        "BL1": bl1,
        "BL1_pooled_metrics": baseline_metrics,
        "BL2_candidates": candidates,
        "BL2_search_gate_passed": selected_bl2 is not None,
        "BL2": None if selected_bl2 is None else selected_bl2["trial"],
        "BL2_gate": None if selected_bl2 is None else selected_bl2["gate"],
    }


def slice_masks(prior_batch: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("all_assessment_rows", np.ones(prior_batch.size, dtype=bool)),
        ("history_0_49", prior_batch < 50),
        ("history_50_199", (prior_batch >= 50) & (prior_batch < 200)),
        ("history_200_plus", prior_batch >= 200),
        ("history_500_plus_exploratory", prior_batch >= 500),
    ]


def daily_and_slice_metrics(table: pa.Table) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    y_all = table["long_view"].to_numpy().astype(np.int8)
    users_all = table["user_id"].to_numpy().astype(np.int64)
    dates = table["event_date"].to_numpy().astype("datetime64[D]")
    prior = table["prior_batch_n"].to_numpy().astype(np.int64)
    models = {"BL0": "p_bl0", "BL1": "p_bl1", "BL2": "p_bl2"}
    daily_rows: list[dict[str, Any]] = []
    for origin in DAILY_ORIGINS:
        mask = dates == np.datetime64(origin, "D")
        for model, column in models.items():
            daily_rows.append(
                {
                    "origin": origin,
                    "model_id": model,
                    **point_metrics(y_all[mask], table[column].to_numpy()[mask], users_all[mask]),
                }
            )
    slice_rows: list[dict[str, Any]] = []
    for slice_id, mask in slice_masks(prior):
        if int(mask.sum()) == 0:
            continue
        model_metrics: dict[str, dict[str, Any]] = {}
        for model, column in models.items():
            metrics = point_metrics(y_all[mask], table[column].to_numpy()[mask], users_all[mask])
            model_metrics[model] = metrics
            row = {"slice": slice_id, "model_id": model, **metrics}
            if model == "BL2":
                for metric in [
                    "average_precision",
                    "user_gauc_event_weighted",
                    "log_loss",
                    "brier",
                    "ece20_equal_width",
                ]:
                    row[f"delta_vs_BL1_{metric}"] = float(
                        metrics[metric] - model_metrics["BL1"][metric]
                    )
            slice_rows.append(row)
    daily_bl1 = {row["origin"]: row for row in daily_rows if row["model_id"] == "BL1"}
    daily_bl2 = {row["origin"]: row for row in daily_rows if row["model_id"] == "BL2"}
    positive_days = sum(
        daily_bl2[origin]["average_precision"] > daily_bl1[origin]["average_precision"]
        for origin in DAILY_ORIGINS
    )
    pooled_bl1 = next(row for row in slice_rows if row["slice"] == "all_assessment_rows" and row["model_id"] == "BL1")
    pooled_bl2 = next(row for row in slice_rows if row["slice"] == "all_assessment_rows" and row["model_id"] == "BL2")
    gate = {
        "positive_average_precision_days": int(positive_days),
        "delta_average_precision": metric_delta(pooled_bl2, pooled_bl1, "average_precision"),
        "delta_user_gauc_event_weighted": metric_delta(
            pooled_bl2, pooled_bl1, "user_gauc_event_weighted"
        ),
        "delta_log_loss": metric_delta(pooled_bl2, pooled_bl1, "log_loss"),
        "delta_brier": metric_delta(pooled_bl2, pooled_bl1, "brier"),
    }
    gate["passed"] = bool(
        positive_days >= 5
        and gate["delta_average_precision"] > 0
        and gate["delta_user_gauc_event_weighted"] >= 0
        and gate["delta_log_loss"] <= 0
        and gate["delta_brier"] <= 0
    )
    return daily_rows, slice_rows, gate


def calibration_rows(table: pa.Table) -> list[dict[str, Any]]:
    y = table["long_view"].to_numpy().astype(np.int8)
    rows: list[dict[str, Any]] = []
    for model, column in {"BL0": "p_bl0", "BL1": "p_bl1", "BL2": "p_bl2"}.items():
        ece, bins = ece_equal_width(y, table[column].to_numpy(), bins=20)
        for row in bins:
            rows.append({"model_id": model, "ece20": ece, **row})
    return rows


def render_figure(daily_rows: list[dict[str, Any]], slice_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_model = {
        model: [row for row in daily_rows if row["model_id"] == model]
        for model in ["BL0", "BL1", "BL2"]
    }
    colors = {"BL0": "#9ca3af", "BL1": "#2563eb", "BL2": "#dc2626"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.arange(7)
    labels = [origin[-5:] for origin in DAILY_ORIGINS]
    for model, rows in by_model.items():
        axes[0, 0].plot(x, [row["average_precision"] for row in rows], marker="o", label=model, color=colors[model])
    axes[0, 0].set_title("A. Daily Train rolling-origin average precision")
    axes[0, 0].set_xticks(x, labels, rotation=30)
    axes[0, 0].set_ylabel("Average precision")
    axes[0, 0].legend()

    delta_ap = [
        by_model["BL2"][index]["average_precision"] - by_model["BL1"][index]["average_precision"]
        for index in range(7)
    ]
    axes[0, 1].bar(x, delta_ap, color=["#16a34a" if value > 0 else "#dc2626" for value in delta_ap])
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("B. Daily increment: BL2 minus BL1")
    axes[0, 1].set_xticks(x, labels, rotation=30)
    axes[0, 1].set_ylabel("Delta average precision")

    pooled = {
        row["model_id"]: row
        for row in slice_rows
        if row["slice"] == "all_assessment_rows"
    }
    model_x = np.arange(3)
    axes[1, 0].bar(model_x - 0.18, [pooled[m]["average_precision"] for m in ["BL0", "BL1", "BL2"]], 0.36, label="AP")
    axes[1, 0].bar(model_x + 0.18, [pooled[m]["user_gauc_event_weighted"] for m in ["BL0", "BL1", "BL2"]], 0.36, label="user-GAUC")
    axes[1, 0].set_xticks(model_x, ["BL0", "BL1", "BL2"])
    axes[1, 0].set_title("C. Pooled discrimination (same rows)")
    axes[1, 0].legend()

    axes[1, 1].bar(model_x - 0.18, [pooled[m]["log_loss"] for m in ["BL0", "BL1", "BL2"]], 0.36, label="Log Loss")
    axes[1, 1].bar(model_x + 0.18, [pooled[m]["brier"] for m in ["BL0", "BL1", "BL2"]], 0.36, label="Brier")
    axes[1, 1].set_xticks(model_x, ["BL0", "BL1", "BL2"])
    axes[1, 1].set_title("D. Pooled probability error (lower is better)")
    axes[1, 1].legend()
    fig.suptitle("KuaiRand-1K Gate 2B — Train-only fixed-row baseline evidence", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def render_report(
    *,
    selected: dict[str, Any],
    daily_gate: dict[str, Any],
    slice_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    fit_count: int,
    elapsed_seconds: float,
) -> str:
    pooled = {
        row["model_id"]: row
        for row in slice_rows
        if row["slice"] == "all_assessment_rows"
    }
    ci = {row["metric"]: row for row in bootstrap_rows}
    rows = []
    for model in ["BL0", "BL1", "BL2"]:
        value = pooled[model]
        rows.append(
            f"| {model} | {fmt(value['average_precision'])} | {fmt(value['user_gauc_event_weighted'])} | "
            f"{fmt(value['log_loss'])} | {fmt(value['brier'])} | {fmt(value['ece20_equal_width'])} |"
        )
    decision = "通过" if daily_gate["passed"] else "未通过"
    sequence = "可提交独立的序列模型阶段审批" if daily_gate["passed"] else "停止序列升级并复核历史特征稳定性"
    return f"""# Gate 2B Train-only 固定行基线结果（2026-08-15）

> 结论：七日 Train rolling-origin 稳定性门禁 **{decision}**。下一动作是：**{sequence}**。  
> 边界：这不是 Gold、Validation 或测试集结果，也不是因果或线上效果证明。

![Gate 2B 结果概览](../figures/gate2b_baseline_results_v002.png)

## 做了什么

- 数据：canonical source-Train 2022-04-08 至 04-17；目标 `tab=1`，历史使用全 tab；所有历史满足 `history_time_ms < target_time_ms`。
- 固定行：04-11 至 04-17 共 7 个逐日 assessment origin，BL0/BL1/BL2 在每一天使用完全相同的 source identity 和标签。
- BL0：仅使用 origin 之前目标行的正例率；BL1：静态 ID/内容元数据的稀疏线性基线；BL2：在 BL1 上增加严格用户历史。高基数 user×content 历史因 pre-metric quick 性能止损而延后独立评估。
- 选择：只在预注册的 04-11、04-14、04-17 三个搜索 origin 内选择 alpha 与 BL2 bundle；冻结后再重新拟合七日回测。
- 不确定性：950 用户、2,000 次共享 PCG64 user-cluster bootstrap；只解释为 Train-only 设计证据。

## 冻结模型

- BL1：`{selected['BL1']['trial_id']}`
- BL2：`{selected['BL2']['trial_id']}`
- 实际拟合：{fit_count} / 32；总耗时 {elapsed_seconds:.1f} 秒。

## 同一批 04-11 至 04-17 目标行的 pooled 指标

| 模型 | Average precision | user-GAUC（事件加权） | Log Loss | Brier | ECE20（描述） |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

BL2 − BL1：ΔAP={fmt(daily_gate['delta_average_precision'])}，Δuser-GAUC={fmt(daily_gate['delta_user_gauc_event_weighted'])}，ΔLog Loss={fmt(daily_gate['delta_log_loss'])}，ΔBrier={fmt(daily_gate['delta_brier'])}；7 天中 {daily_gate['positive_average_precision_days']} 天 ΔAP>0。

## 配对用户聚类不确定性

| 对比指标（BL2−BL1） | 点估计 | 95% percentile CI | Train-only 解读 |
|---|---:|---:|---|
| Average precision | {fmt(ci['average_precision']['point_estimate'])} | [{fmt(ci['average_precision']['ci95_lower'])}, {fmt(ci['average_precision']['ci95_upper'])}] | 聚合判别增量 |
| user-GAUC | {fmt(ci['user_gauc_event_weighted']['point_estimate'])} | [{fmt(ci['user_gauc_event_weighted']['ci95_lower'])}, {fmt(ci['user_gauc_event_weighted']['ci95_upper'])}] | 用户内判别增量 |
| Log Loss | {fmt(ci['log_loss']['point_estimate'])} | [{fmt(ci['log_loss']['ci95_lower'])}, {fmt(ci['log_loss']['ci95_upper'])}] | 小于 0 更好 |
| Brier | {fmt(ci['brier']['point_estimate'])} | [{fmt(ci['brier']['ci95_lower'])}, {fmt(ci['brier']['ci95_upper'])}] | 小于 0 更好 |

## 当前只能得出的结论

1. 可以判断严格历史统计在 source-Train 固定行上是否带来稳定的预测增量；不能据此宣称对 Validation/test 泛化。
2. ECE20 只报告校准形态，不参与 Gate 2B 的通过判定；正式 SESOI 与非劣边界仍须在第一次读取 Validation 指标前冻结。
3. 即使门禁通过，Gold 构建、序列模型和 Validation 访问仍需分别审批；若门禁失败，不得通过改超参数后反复重跑来追逐正结果。
"""


def build_run_manifest(
    output_dir: Path,
    *,
    mode: str,
    input_verification: list[dict[str, Any]],
    contract_sha: str,
    feature_validation: dict[str, Any],
    selected: dict[str, Any] | None,
    fit_count: int,
    elapsed_seconds: float,
    status: str,
    gates: dict[str, Any],
) -> dict[str, Any]:
    managed: list[dict[str, Any]] = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            managed.append(
                {"path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    for path in [REPORT_PATH, FIGURE_PATH, CHECKPOINT_PATH]:
        if path.exists():
            managed.append(
                {"path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    return {
        "run_id": f"gate2b-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "status": status,
        "mode": mode,
        "checkpoint_eligible": mode == "release" and status == "complete_train_only_gate2b",
        "scope": {
            "source_train_only": True,
            "silver_recleaned": False,
            "gold_built": False,
            "validation_accessed": False,
            "late_accessed": False,
            "random_accessed": False,
            "statistic_file_accessed": False,
            "sequence_or_neural_models_fit": False,
        },
        "contract": {"path": relative(CONTRACT_PATH), "sha256": contract_sha},
        "inputs": input_verification,
        "feature_validation": feature_validation,
        "selected_models": selected,
        "fit_runs": fit_count,
        "gates": gates,
        "environment": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "scikit_learn": sklearn.__version__,
            "threads": 8,
            "accelerator_used": False,
            "platform": platform.platform(),
        },
        "elapsed_seconds": elapsed_seconds,
        "script": {
            "path": relative(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "outputs": managed,
    }


def validate_only() -> None:
    contract, digest = load_contract()
    trials = search_trials(ALPHAS)
    assert len(trials) == 6
    assert len([trial for trial in trials if trial["model_id"] == "BL1"]) == 3
    assert len([trial for trial in trials if trial["model_id"] == "BL2"]) == 3
    multiplicities, observed = make_multiplicities()
    assert multiplicities.shape == (2000, 950)
    if observed != EXPECTED_MULTIPLICITY_SHA256:
        raise RuntimeError("frozen bootstrap digest mismatch")
    if contract["operational_budget"]["maximum_total_fit_runs"] != MAX_FITS:
        raise RuntimeError("code and contract fit budgets differ")
    progress(f"VALIDATE_ONLY_OK contract_sha256={digest} bootstrap_sha256={observed}")


def run(mode: str, reuse_features: bool) -> None:
    started = time.perf_counter()
    contract, contract_sha = load_contract()
    full_sha = mode == "release"
    output_dir = RELEASE_DIR if mode == "release" else QUICK_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"verifying four explicit inputs ({'full SHA-256' if full_sha else 'size only'})")
    input_verification = verify_inputs(contract, full_sha)
    trials = search_trials(ALPHAS if mode == "release" else [1e-5])
    origins = SEARCH_ORIGINS if mode == "release" else [SEARCH_ORIGINS[0]]
    trial_manifest_path = output_dir / "search_trial_manifest.csv"
    write_csv(
        trial_manifest_path,
        [
            {
                **trial,
                "search_origins": "|".join(origins),
                "contract_sha256": contract_sha,
                "frozen_before_fit": True,
            }
            for trial in trials
        ],
    )
    trial_manifest_sha = sha256_file(trial_manifest_path)

    feature_path = output_dir / "gate2b_feature_matrix.parquet"
    feature_manifest_path = output_dir / "feature_manifest.json"
    feature_validation: dict[str, Any]
    if reuse_features and feature_path.exists() and feature_manifest_path.exists():
        previous = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != contract_sha:
            raise RuntimeError("cached feature contract hash differs")
        observed_feature_sha = sha256_file(feature_path)
        if observed_feature_sha != previous.get("feature_sha256"):
            raise RuntimeError("cached feature artifact hash differs")
        feature_validation = previous["validation"]
        progress("reusing hash-verified feature artifact")
    else:
        temp_feature = output_dir / "gate2b_feature_matrix.building.parquet"
        if temp_feature.exists():
            temp_feature.unlink()
        temp_dir = output_dir / "duckdb_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        con.execute("SET threads=8")
        con.execute("SET memory_limit='12GB'")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET temp_directory=?", [str(temp_dir)])
        progress("materializing strict point-in-time Train feature matrix")
        feature_validation = materialize_gate2b_features(
            con,
            early_path=INPUT_PATHS["early_train"],
            mismatch_path=INPUT_PATHS["formula_mismatch"],
            videos_path=INPUT_PATHS["videos_basic"],
            output_path=temp_feature,
        )
        con.close()
        temp_feature.replace(feature_path)
        write_json(
            feature_manifest_path,
            {
                "status": "complete_point_in_time_train_only",
                "contract_sha256": contract_sha,
                "validation": feature_validation,
                "feature_path": relative(feature_path),
                "feature_size_bytes": feature_path.stat().st_size,
                "feature_sha256": sha256_file(feature_path),
                "silver_recleaned": False,
                "gold_built": False,
            },
        )
        progress(f"feature matrix complete: {feature_validation['target_rows']:,} rows")

    con = duckdb.connect()
    manifests = target_manifests(con, feature_path)
    con.close()
    write_csv(output_dir / "target_row_manifests.csv", manifests)
    progress("fixed source-identity manifests verified")

    data = FeatureArrays.read(feature_path)
    ledger: list[dict[str, Any]] = []
    for origin in origins:
        progress(f"search origin {origin}: preparing fold-train-only encoder/scalers")
        matrices = prepare_origin_matrices(
            data,
            origin,
            include_h3=False,
            train_limit=250_000 if mode == "quick" else None,
            assess_limit=50_000 if mode == "quick" else None,
        )
        probabilities: dict[str, np.ndarray] = {"p_bl0": np.full(matrices.y_assess.size, matrices.p0)}
        for trial in trials:
            if trial["model_id"] == "BL1":
                train_x, assess_x = matrices.static_train, matrices.static_assess
            elif trial["bundle"] == "H2_USER_STRICT_V1":
                train_x, assess_x = matrices.h2_train, matrices.h2_assess
            else:
                raise RuntimeError("unregistered BL2 feature bundle")
            probabilities[trial["prediction_column"]] = fit_one(
                ledger,
                stage="search",
                origin=origin,
                trial=trial,
                train_x=train_x,
                y_train=matrices.y_train,
                assess_x=assess_x,
                started=started,
            )
        table = prediction_table(
            data,
            matrices.assess_indices,
            probabilities,
            {
                "stage": "search",
                "origin": origin,
                "contract_sha256": contract_sha,
                "trial_manifest_sha256": trial_manifest_sha,
            },
        )
        path = output_dir / f"search_predictions_origin_{origin}.parquet"
        pq.write_table(table, path, compression="zstd", row_group_size=100_000)
        progress(f"search origin {origin}: wrote {table.num_rows:,} aligned predictions")
        del matrices, probabilities, table
        gc.collect()

    search_metric_rows, pooled, by_origin = materialize_search_metrics(output_dir, origins, trials)
    write_csv(output_dir / "search_metrics.csv", search_metric_rows)
    if mode == "quick":
        write_csv(output_dir / "usage_ledger.csv", ledger)
        manifest = build_run_manifest(
            output_dir,
            mode=mode,
            input_verification=input_verification,
            contract_sha=contract_sha,
            feature_validation=feature_validation,
            selected=None,
            fit_count=len(ledger),
            elapsed_seconds=time.perf_counter() - started,
            status="complete_quick_smoke_not_checkpoint_eligible",
            gates={"search_selection_run": False},
        )
        write_json(output_dir / "run_manifest.json", manifest)
        progress("quick smoke complete; no research gate changed")
        return

    selected = select_models(trials, pooled, by_origin)
    write_json(output_dir / "selected_models.json", selected)
    if not selected["BL2_search_gate_passed"]:
        write_csv(output_dir / "usage_ledger.csv", ledger)
        manifest = build_run_manifest(
            output_dir,
            mode=mode,
            input_verification=input_verification,
            contract_sha=contract_sha,
            feature_validation=feature_validation,
            selected=selected,
            fit_count=len(ledger),
            elapsed_seconds=time.perf_counter() - started,
            status="stopped_at_frozen_BL2_search_gate",
            gates={"BL2_search": False, "daily_backtest_run": False},
        )
        write_json(output_dir / "run_manifest.json", manifest)
        progress("BL2 search gate failed; stopped before daily fitting as preregistered")
        return

    progress(f"search gate passed; frozen BL1={selected['BL1']['trial_id']} BL2={selected['BL2']['trial_id']}")
    daily_tables: list[pa.Table] = []
    selected_bl1 = selected["BL1"]
    selected_bl2 = selected["BL2"]
    for origin in DAILY_ORIGINS:
        progress(f"daily origin {origin}: fitting frozen BL1 and BL2")
        matrices = prepare_origin_matrices(data, origin, include_h3=False)
        p_bl1 = fit_one(
            ledger,
            stage="daily_frozen",
            origin=origin,
            trial=selected_bl1,
            train_x=matrices.static_train,
            y_train=matrices.y_train,
            assess_x=matrices.static_assess,
            started=started,
        )
        train_x, assess_x = matrices.h2_train, matrices.h2_assess
        p_bl2 = fit_one(
            ledger,
            stage="daily_frozen",
            origin=origin,
            trial=selected_bl2,
            train_x=train_x,
            y_train=matrices.y_train,
            assess_x=assess_x,
            started=started,
        )
        table = prediction_table(
            data,
            matrices.assess_indices,
            {
                "p_bl0": np.full(matrices.y_assess.size, matrices.p0),
                "p_bl1": p_bl1,
                "p_bl2": p_bl2,
            },
            {
                "stage": "daily_frozen",
                "origin": origin,
                "contract_sha256": contract_sha,
                "BL1_trial_id": selected_bl1["trial_id"],
                "BL2_trial_id": selected_bl2["trial_id"],
            },
        )
        daily_tables.append(table)
        del matrices, p_bl1, p_bl2, table
        gc.collect()
    daily_table = pa.concat_tables(daily_tables)
    daily_path = output_dir / "daily_predictions.parquet"
    pq.write_table(daily_table, daily_path, compression="zstd", row_group_size=100_000)
    del daily_tables
    progress(f"daily prediction artifact frozen: {daily_table.num_rows:,} rows")

    reloaded_daily = pq.read_table(daily_path)
    daily_rows, slice_rows, daily_gate = daily_and_slice_metrics(reloaded_daily)
    write_csv(output_dir / "daily_metrics.csv", daily_rows)
    write_csv(output_dir / "pooled_and_slice_metrics.csv", slice_rows)
    write_csv(output_dir / "calibration_bins.csv", calibration_rows(reloaded_daily))

    full_users = np.unique(data.columns["user_id"].astype(np.int64))
    if full_users.size != 950:
        raise RuntimeError("frozen bootstrap user universe must contain 950 users")
    multiplicities, multiplicity_sha = make_multiplicities(
        user_count=950, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED
    )
    if multiplicity_sha != EXPECTED_MULTIPLICITY_SHA256:
        raise RuntimeError("bootstrap multiplicity digest mismatch")
    progress("running 2,000 paired user-cluster bootstrap replicates on frozen predictions")
    bootstrap_rows = paired_user_cluster_bootstrap(
        reloaded_daily["long_view"].to_numpy().astype(np.int8),
        reloaded_daily["p_bl1"].to_numpy(),
        reloaded_daily["p_bl2"].to_numpy(),
        reloaded_daily["user_id"].to_numpy().astype(np.int64),
        user_universe=full_users,
        multiplicities=multiplicities,
        ap_block_size=8,
    )
    for row in bootstrap_rows:
        row.update(
            {
                "seed": BOOTSTRAP_SEED,
                "cluster": "user_id",
                "multiplicity_matrix_sha256": multiplicity_sha,
                "role": "train_only_design_uncertainty_not_validation_MDE",
            }
        )
    write_csv(output_dir / "paired_user_cluster_bootstrap.csv", bootstrap_rows)
    write_csv(output_dir / "usage_ledger.csv", ledger)
    render_figure(daily_rows, slice_rows, FIGURE_PATH)
    elapsed = time.perf_counter() - started
    report = render_report(
        selected=selected,
        daily_gate=daily_gate,
        slice_rows=slice_rows,
        bootstrap_rows=bootstrap_rows,
        fit_count=len(ledger),
        elapsed_seconds=elapsed,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    checkpoint = f"""# Gate 2B checkpoint（2026-08-15）

> 状态：Train-only BL0/BL1/BL2 固定行回测已完成；七日稳定性门禁：**{'通过' if daily_gate['passed'] else '未通过'}**。

- 结果报告：[`reports/analysis/gate2b_baseline_results_v002.md`](analysis/gate2b_baseline_results_v002.md)
- 可执行合同：[`configs/gate2b_fixed_row_baseline_contract_v002.yaml`](../configs/gate2b_fixed_row_baseline_contract_v002.yaml)
- 受管产物：[`reports/generated/gate2b_baselines_v002/run_manifest.json`](generated/gate2b_baselines_v002/run_manifest.json)
- 边界：未重洗 Silver；未构建 Gold；未访问 Validation、late、random、statistic 或 restricted test；未拟合序列/神经模型。
- 下一步：{'提交 Gold 与序列研究的独立门禁审查，不自动执行。' if daily_gate['passed'] else '停止模型升级，审查历史特征与时间稳定性，不追逐式重调。'}
"""
    CHECKPOINT_PATH.write_text(checkpoint, encoding="utf-8")
    manifest = build_run_manifest(
        output_dir,
        mode=mode,
        input_verification=input_verification,
        contract_sha=contract_sha,
        feature_validation=feature_validation,
        selected=selected,
        fit_count=len(ledger),
        elapsed_seconds=elapsed,
        status="complete_train_only_gate2b",
        gates={"BL2_search": True, "daily_stability": daily_gate},
    )
    write_json(output_dir / "run_manifest.json", manifest)
    progress(f"release complete; daily gate={'PASS' if daily_gate['passed'] else 'FAIL'}; elapsed={elapsed:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="isolated smoke run; cannot change a gate")
    mode.add_argument("--release", action="store_true", help="full deterministic checkpoint run")
    mode.add_argument("--validate-only", action="store_true", help="validate frozen contract and bootstrap plan")
    parser.add_argument("--reuse-features", action="store_true", help="reuse only a hash-verified matching feature artifact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        validate_only()
        return
    run("quick" if args.quick else "release", args.reuse_features)


if __name__ == "__main__":
    main()
