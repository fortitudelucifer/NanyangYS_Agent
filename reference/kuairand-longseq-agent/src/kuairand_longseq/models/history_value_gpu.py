"""Full-batch CUDA optimizers for the frozen history-value experiment."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse


SEED = 20260814
STAGE_PREDECESSOR = {
    "validation": "preflight",
    "sealed_test": "validation",
    "random_audit": "sealed_test",
}


def assert_stage_access(requested: str, completed_hashed: set[str]) -> None:
    """Fail closed when a governed split is requested out of contract order."""

    if requested not in STAGE_PREDECESSOR:
        raise RuntimeError(f"unknown governed stage: {requested}")
    predecessor = STAGE_PREDECESSOR[requested]
    if predecessor not in completed_hashed:
        raise RuntimeError(
            f"out-of-order access: {requested} requires {predecessor} complete and hashed"
        )


@dataclass
class GpuFit:
    optimizer: str
    learning_rate: float
    steps: int
    coefficient: np.ndarray
    intercept: float
    objective: float
    terminal_gradient_norm: float
    elapsed_seconds: float
    peak_cuda_memory_bytes: int
    objective_trace: list[dict[str, float | int]]


def scipy_csr_to_torch(matrix: sparse.csr_matrix, device: torch.device) -> torch.Tensor:
    csr = matrix.tocsr().astype(np.float32, copy=False)
    crow = torch.as_tensor(csr.indptr, dtype=torch.int64, device=device)
    col = torch.as_tensor(csr.indices, dtype=torch.int64, device=device)
    value = torch.as_tensor(csr.data, dtype=torch.float32, device=device)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Sparse CSR tensor support is in beta.*")
        warnings.filterwarnings(
            "ignore", message=r"Sparse invariant checks are implicitly disabled.*"
        )
        return torch.sparse_csr_tensor(
            crow, col, value, size=csr.shape, dtype=torch.float32,
            device=device, check_invariants=False,
        )


def _objective(
    matrix: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    score = torch.sparse.mm(matrix, weight[:, None]).squeeze(1) + bias
    return F.binary_cross_entropy_with_logits(score, labels) + (
        0.5 * float(alpha) * torch.dot(weight, weight)
    )


def _make_optimizer(
    name: str, parameters: Iterable[torch.Tensor], learning_rate: float
) -> torch.optim.Optimizer:
    if name == "ADAM":
        return torch.optim.Adam(
            parameters, lr=float(learning_rate), betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0.0, amsgrad=False, foreach=False, maximize=False,
            capturable=False, differentiable=False, fused=False,
        )
    if name == "SGD":
        return torch.optim.SGD(
            parameters, lr=float(learning_rate), momentum=0.0, dampening=0.0,
            weight_decay=0.0, nesterov=False, maximize=False, foreach=False,
            differentiable=False, fused=False,
        )
    raise ValueError(f"unsupported GPU optimizer: {name}")


def fit_trajectory(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    *,
    device: torch.device,
    optimizer_name: str,
    learning_rate: float,
    checkpoints: Iterable[int],
    alpha: float,
) -> GpuFit:
    """Fit one trajectory and record post-update objectives at all checkpoints."""

    requested = tuple(sorted({int(value) for value in checkpoints}))
    if not requested or requested[0] <= 0:
        raise ValueError("positive checkpoints are required")
    y = np.asarray(labels, dtype=np.float32)
    if y.ndim != 1 or y.size != matrix.shape[0] or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("binary labels must align with the sparse matrix")
    if device.type != "cuda":
        raise RuntimeError("history confirmation forbids non-CUDA optimizer execution")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.cuda.empty_cache()
    # This CUDA 12.8 / RTX 50-series build rejects resetPeakMemoryStats despite
    # accepting ordinary tensor work. Keep the process-level peak, which is an
    # upper bound and therefore conservative for the resource contract.
    x = scipy_csr_to_torch(matrix, device)
    target = torch.as_tensor(y, dtype=torch.float32, device=device)
    weight = torch.zeros(matrix.shape[1], dtype=torch.float32, device=device, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = _make_optimizer(optimizer_name, [weight, bias], learning_rate)
    trace: list[dict[str, float | int]] = []

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(1, requested[-1] + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _objective(x, target, weight, bias, alpha)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite {optimizer_name} objective at step {step}")
        loss.backward()
        optimizer.step()
        if step in requested:
            with torch.no_grad():
                post = _objective(x, target, weight, bias, alpha)
            trace.append({"step": step, "objective": float(post.detach().cpu())})

    optimizer.zero_grad(set_to_none=True)
    terminal = _objective(x, target, weight, bias, alpha)
    terminal.backward()
    gradient_sq = torch.dot(weight.grad, weight.grad) + bias.grad * bias.grad
    gradient_norm = float(torch.sqrt(gradient_sq).detach().cpu())
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    coefficient = weight.detach().cpu().numpy().astype(np.float64, copy=False)
    intercept = float(bias.detach().cpu())
    objective = float(terminal.detach().cpu())
    peak = int(torch.cuda.max_memory_allocated())
    if not np.isfinite(coefficient).all() or not np.isfinite(intercept):
        raise RuntimeError("GPU optimizer produced nonfinite parameters")

    del x, target, weight, bias, optimizer, terminal, gradient_sq
    torch.cuda.empty_cache()
    return GpuFit(
        optimizer=optimizer_name,
        learning_rate=float(learning_rate),
        steps=requested[-1],
        coefficient=coefficient,
        intercept=intercept,
        objective=objective,
        terminal_gradient_norm=gradient_norm,
        elapsed_seconds=float(elapsed),
        peak_cuda_memory_bytes=peak,
        objective_trace=trace,
    )


def score(
    matrix: sparse.csr_matrix,
    coefficient: np.ndarray,
    intercept: float,
    *,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    x = scipy_csr_to_torch(matrix, device)
    weight = torch.as_tensor(coefficient, dtype=torch.float32, device=device)
    bias = torch.tensor(float(intercept), dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        raw = torch.sparse.mm(x, weight[:, None]).squeeze(1) + bias
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = raw.cpu().numpy().astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError("GPU scoring produced nonfinite raw scores")
    del x, weight, bias, raw
    torch.cuda.empty_cache()
    return result, float(elapsed)


def adequacy(
    gpu_objective: float,
    reference_objective: float,
    *,
    reference_converged: bool,
) -> dict[str, Any]:
    regret = float(gpu_objective) - float(reference_objective)
    allowed = max(1e-4, 0.005 * abs(float(reference_objective)))
    reference_above = regret < -1e-10
    return {
        "GPU_regularized_training_objective": float(gpu_objective),
        "reference_regularized_training_objective": float(reference_objective),
        "objective_regret": regret,
        "maximum_allowed_regret": allowed,
        "reference_converged": bool(reference_converged),
        "reference_above_GPU": bool(reference_above),
        "adequacy_passed": bool(reference_converged and not reference_above and regret <= allowed),
    }
