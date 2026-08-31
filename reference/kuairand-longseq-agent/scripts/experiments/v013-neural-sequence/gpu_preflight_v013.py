"""Synthetic CUDA memory/throughput preflight for v013 neural geometry.

The runner never reads Gold, Silver, target labels, or protected partitions.  It
is engineering evidence only and cannot authorize or substitute for a formal
v013 training run.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
import yaml


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kuairand_longseq.models.neural_sequence_v013 import (  # noqa: E402
    HISTORY_LENGTH_BY_VARIANT,
    SCIENTIFIC_VARIANTS,
    SyntheticV013Config,
    SyntheticV013StressModel,
    make_synthetic_batch,
    parameter_count,
)


EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "neural_sequence_candidate_model_v013"
)
CONTRACT_PATH = EXPERIMENT_DIR / "contract_v013.yaml"
FORMAL_APPROVAL_PATH = EXPERIMENT_DIR / "approval_v013.json"
SYNTHETIC_APPROVAL_PATH = (
    EXPERIMENT_DIR / "synthetic_gpu_preflight_authorization_v013.json"
)
RESULT_PREFIX = "V013_SYNTHETIC_WORKER_RESULT="


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected an object in {path}")
    return payload


def validate_governing_state() -> dict[str, Any]:
    """Verify that formal execution stays closed and synthetic scope is explicit."""

    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    formal = load_json(FORMAL_APPROVAL_PATH)
    synthetic = load_json(SYNTHETIC_APPROVAL_PATH)
    if not isinstance(contract, dict):
        raise RuntimeError("v013 contract must parse as a mapping")
    observed_hash = sha256_file(CONTRACT_PATH)
    if contract.get("contract_id") != "neural_sequence_candidate_model_v013":
        raise RuntimeError("unexpected v013 contract_id")
    authorization = contract.get("authorization", {})
    if authorization.get("execution_authorized") is not False:
        raise RuntimeError("synthetic preflight requires formal execution to stay closed")
    if formal.get("neural_training_authorized") is not False:
        raise RuntimeError("formal approval unexpectedly permits neural training")
    if synthetic.get("synthetic_gpu_preflight_authorized") is not True:
        raise RuntimeError("synthetic GPU preflight is not authorized")
    if synthetic.get("scientific_neural_training_authorized") is not False:
        raise RuntimeError("synthetic authorization must not permit scientific training")
    if synthetic.get("observed_draft_contract_sha256") != observed_hash:
        raise RuntimeError("synthetic authorization does not match the observed draft")

    registered = tuple(
        row["model_id"]
        for row in contract.get("model_registry", [])
        if row.get("model_id") in SCIENTIFIC_VARIANTS
    )
    if registered != SCIENTIFIC_VARIANTS:
        raise RuntimeError("synthetic model registry differs from the v013 contract")
    dimensions = contract["neural_architecture"][
        "proposed_single_configuration_before_execution"
    ]
    expected_dimensions = {
        "categorical_embedding_dim": 32,
        "event_embedding_dim": 64,
        "attention_hidden_dim": 64,
        "fusion_hidden_dims": [256, 128, 64],
        "dropout": 0.10,
    }
    for key, expected in expected_dimensions.items():
        if dimensions.get(key) != expected:
            raise RuntimeError(f"contract dimension {key} is not {expected!r}")
    if contract["neural_architecture"]["HIER500_definition"].get(
        "full_length_transformer"
    ) != "forbidden_in_v013":
        raise RuntimeError("HIER500 full-length Transformer prohibition is missing")
    if contract["training_randomness"].get("neural_seed_values") != [
        20260824,
        20260825,
        20260826,
        20260827,
        20260828,
    ]:
        raise RuntimeError("v013 seed registry changed")
    return {
        "contract_sha256": observed_hash,
        "formal_execution_authorized": False,
        "synthetic_gpu_preflight_authorized": True,
        "evidence_level": "engineering_synthetic_only",
        "registered_variants": list(registered),
    }


def config_from_args(args: argparse.Namespace) -> SyntheticV013Config:
    return SyntheticV013Config(
        author_vocab_size=args.author_vocab_size,
        music_vocab_size=args.music_vocab_size,
        tag_vocab_size=args.tag_vocab_size,
        upload_type_vocab_size=args.upload_type_vocab_size,
        scene_vocab_size=args.scene_vocab_size,
    )


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; transparent CPU fallback is forbidden")
    if args.variant not in SCIENTIFIC_VARIANTS:
        raise RuntimeError(f"unknown v013 variant: {args.variant}")
    if args.batch_size <= 0 or args.steps <= 0:
        raise RuntimeError("batch size and steps must be positive")
    if not 0.0 < args.headroom_ratio < 1.0:
        raise RuntimeError("headroom ratio must be between zero and one")

    config = config_from_args(args)
    config.validate()
    device = torch.device("cuda", args.device_index)
    properties = torch.cuda.get_device_properties(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
    peak_reset_succeeded = True
    started = time.perf_counter()
    model: SyntheticV013StressModel | None = None
    optimizer: torch.optim.Optimizer | None = None
    batch: dict[str, torch.Tensor] | None = None

    try:
        model = SyntheticV013StressModel(args.variant, config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
            foreach=False,
        )
        batch = make_synthetic_batch(
            args.variant, args.batch_size, config, device=device, seed=args.seed
        )
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except RuntimeError:
            peak_reset_succeeded = False

        autocast_enabled = args.precision == "bf16"
        autocast_dtype = torch.bfloat16
        torch.cuda.synchronize(device)
        training_started = time.perf_counter()
        terminal_loss = float("nan")
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                logits = model(batch)
                loss = F.binary_cross_entropy_with_logits(logits, batch["target"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("synthetic preflight produced a nonfinite loss")
            loss.backward()
            optimizer.step()
            terminal_loss = float(loss.detach().cpu())
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - training_started
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        total_memory = int(properties.total_memory)
        headroom_passed = peak_reserved <= int(total_memory * args.headroom_ratio)
        return {
            "status": "pass" if headroom_passed else "insufficient_headroom",
            "oom": False,
            "headroom_passed": bool(headroom_passed),
            "variant": args.variant,
            "history_length": HISTORY_LENGTH_BY_VARIANT[args.variant],
            "batch_size": args.batch_size,
            "steps": args.steps,
            "precision": args.precision,
            "seed": args.seed,
            "optimizer": "AdamW_engineering_only_lr_1e-3_wd_1e-2_foreach_false",
            "terminal_loss": terminal_loss,
            "elapsed_training_seconds": elapsed,
            "seconds_per_step": elapsed / args.steps,
            "samples_per_second": args.batch_size * args.steps / elapsed,
            "parameter_count": parameter_count(model),
            "parameter_bytes_fp32": parameter_count(model) * 4,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "total_cuda_memory_bytes": total_memory,
            "peak_reserved_fraction": peak_reserved / total_memory,
            "required_headroom_fraction": 1.0 - args.headroom_ratio,
            "peak_memory_reset_succeeded": peak_reset_succeeded,
            "device": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "synthetic_vocabularies": {
                "author": config.author_vocab_size,
                "music": config.music_vocab_size,
                "tag": config.tag_vocab_size,
                "upload_type": config.upload_type_vocab_size,
                "scene": config.scene_vocab_size,
            },
            "scientific_result": False,
        }
    except torch.OutOfMemoryError as exc:
        total_memory = int(properties.total_memory)
        return {
            "status": "oom",
            "oom": True,
            "headroom_passed": False,
            "variant": args.variant,
            "history_length": HISTORY_LENGTH_BY_VARIANT[args.variant],
            "batch_size": args.batch_size,
            "steps": args.steps,
            "precision": args.precision,
            "seed": args.seed,
            "elapsed_until_failure_seconds": time.perf_counter() - started,
            "total_cuda_memory_bytes": total_memory,
            "device": properties.name,
            "error": str(exc),
            "scientific_result": False,
        }
    finally:
        del batch, optimizer, model
        torch.cuda.empty_cache()


def worker_command(args: argparse.Namespace, variant: str, batch_size: int) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--worker",
        "--variant",
        variant,
        "--batch-size",
        str(batch_size),
        "--steps",
        str(args.steps),
        "--precision",
        args.precision,
        "--seed",
        str(args.seed),
        "--device-index",
        str(args.device_index),
        "--headroom-ratio",
        str(args.headroom_ratio),
    ]
    for flag, value in (
        ("--author-vocab-size", args.author_vocab_size),
        ("--music-vocab-size", args.music_vocab_size),
        ("--tag-vocab-size", args.tag_vocab_size),
        ("--upload-type-vocab-size", args.upload_type_vocab_size),
        ("--scene-vocab-size", args.scene_vocab_size),
    ):
        command.extend((flag, str(value)))
    return command


def parse_worker_output(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    return {
        "status": "worker_error",
        "oom": False,
        "headroom_passed": False,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "scientific_result": False,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    columns = (
        "status",
        "variant",
        "history_length",
        "batch_size",
        "steps",
        "precision",
        "headroom_passed",
        "oom",
        "parameter_count",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "total_cuda_memory_bytes",
        "peak_reserved_fraction",
        "seconds_per_step",
        "samples_per_second",
        "device",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def finalize_hashes(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_hash_manifest.json":
            rows.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        output_dir / "artifact_hash_manifest.json",
        {
            "manifest_scope": "synthetic_gpu_engineering_preflight_v013",
            "artifacts": rows,
        },
    )


def environment_payload() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    payload: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        payload.update(
            {
                "device": properties.name,
                "device_count": torch.cuda.device_count(),
                "compute_capability": [properties.major, properties.minor],
                "total_cuda_memory_bytes": int(properties.total_memory),
            }
        )
    return payload


def run_parent(args: argparse.Namespace) -> int:
    governing = validate_governing_state()
    if args.validate_only:
        print(json.dumps(governing, indent=2, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; transparent CPU fallback is forbidden")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else EXPERIMENT_DIR / "outputs" / f"gpu-preflight-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "environment.json", environment_payload())
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (output_dir / "environment_lock.txt").write_text(
        freeze.stdout, encoding="utf-8"
    )

    results: list[dict[str, Any]] = []
    worker_logs: list[dict[str, Any]] = []
    for variant in args.variants:
        for batch_size in args.batch_sizes:
            command = worker_command(args, variant, batch_size)
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=args.worker_timeout_seconds,
            )
            result = parse_worker_output(completed)
            result.setdefault("variant", variant)
            result.setdefault("batch_size", batch_size)
            results.append(result)
            worker_logs.append(
                {
                    "variant": variant,
                    "batch_size": batch_size,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )

    write_json(output_dir / "stress_results.json", results)
    write_json(output_dir / "worker_logs.json", worker_logs)
    write_results_csv(output_dir / "stress_results.csv", results)
    focus_variant = "HIER500" if "HIER500" in args.variants else args.variants[-1]
    safe_batches = [
        int(row["batch_size"])
        for row in results
        if row.get("variant") == focus_variant and row.get("headroom_passed") is True
    ]
    recommended = max(safe_batches) if safe_batches else None
    decision = {
        "decision_type": "synthetic_gpu_engineering_preflight_only",
        "scientific_result": False,
        "formal_training_device_qualified": False,
        "focus_variant": focus_variant,
        "largest_tested_batch_with_required_headroom": recommended,
        "sufficient_for_current_implementation_and_pilot_prep": recommended is not None,
        "desktop_5070ti_needed_now": False if recommended is not None else None,
        "hardware_recommendation": (
            "keep_using_current_5060_for_implementation_and_pilot_prep"
            if recommended is not None
            else "undetermined_without_a_safe_baseline_in_this_sweep"
        ),
        "remaining_gate": (
            "repeat against the frozen Gold feature cardinalities, actual data loader, "
            "and frozen training configuration before formal device qualification"
        ),
    }
    write_json(output_dir / "hardware_decision.json", decision)
    write_json(
        output_dir / "run_manifest.json",
        {
            "run_id": output_dir.name,
            "created_at": datetime.now().astimezone().isoformat(),
            "evidence_level": "engineering_synthetic_only",
            "formal_execution_authorized": False,
            "governing_state": governing,
            "script_sha256": sha256_file(SCRIPT_PATH),
            "model_geometry_sha256": sha256_file(
                PROJECT_ROOT
                / "src"
                / "kuairand_longseq"
                / "models"
                / "neural_sequence_v013.py"
            ),
            "synthetic_authorization_sha256": sha256_file(SYNTHETIC_APPROVAL_PATH),
            "variants": args.variants,
            "batch_sizes": args.batch_sizes,
            "steps": args.steps,
            "precision": args.precision,
            "seed": args.seed,
            "headroom_ratio": args.headroom_ratio,
            "output_dir": str(output_dir),
        },
    )
    finalize_hashes(output_dir)
    print(json.dumps({"output_dir": str(output_dir), "decision": decision}, indent=2))
    return 0 if recommended is not None else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variants", nargs="+", choices=SCIENTIFIC_VARIANTS, default=["HIER500"])
    parser.add_argument("--variant", choices=SCIENTIFIC_VARIANTS, default="HIER500")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--headroom-ratio", type=float, default=0.85)
    parser.add_argument("--worker-timeout-seconds", type=int, default=600)
    parser.add_argument("--author-vocab-size", type=int, default=500_000)
    parser.add_argument("--music-vocab-size", type=int, default=1_000_000)
    parser.add_argument("--tag-vocab-size", type=int, default=100_000)
    parser.add_argument("--upload-type-vocab-size", type=int, default=128)
    parser.add_argument("--scene-vocab-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_governing_state()
    if args.worker:
        result = run_worker(args)
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
