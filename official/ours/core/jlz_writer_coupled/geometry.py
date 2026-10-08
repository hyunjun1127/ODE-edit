"""Fixed-entry, full-context FP64 ridge geometry for writer-coupled v5.

There is no pooled-key outer-product approximation, jitter, inverse, or
per-request solve here.  Call once per layer at batch entry and retain only P
and its scalar metadata during fitting.  C0/H are loaded one layer at a time.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite {name}")


def _symmetric(matrix: torch.Tensor, block_size: int) -> None:
    # Avoid a second dense d_in-by-d_in tensor solely for validation.
    for start in range(0, matrix.shape[0], block_size):
        rows = matrix[start:start + block_size]
        if not torch.allclose(rows, matrix[:, start:start + block_size].T,
                              atol=0.0, rtol=1e-12):
            raise ValueError("A must be symmetric; no symmetrization is applied")


def _cholesky(matrix: torch.Tensor, name: str) -> torch.Tensor:
    factor, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
    if int(info) != 0:
        raise ValueError(f"{name} is not SPD; no jitter is applied")
    _finite(factor, f"{name} factor")
    return factor


@torch.no_grad()
def build_prior_from_npz(
    stat_path: str | Path,
    history: torch.Tensor,
    *,
    lambda_c: float,
    device: torch.device | str = "cpu",
    block_size: int = 256,
) -> torch.Tensor:
    """Build A=lambda_C*C0+H, preserving native FP32 division semantics.

The npz must contain the FP32 accumulated ``mom2.mom2`` and positive scalar
``mom2.count``.  C0 is divided by count in FP32 *before* conversion to FP64.
History is the native FP32 state and is converted to FP64 in bounded row
blocks.  Neither the archive nor the caller's history is mutated.
"""
    _positive_integer(block_size, "block_size")
    if not math.isfinite(lambda_c) or lambda_c <= 0:
        raise ValueError("lambda_c must be finite and positive")
    if not isinstance(history, torch.Tensor) or history.dtype != torch.float32:
        raise ValueError("history must be native FP32")
    if history.ndim != 2 or history.shape[0] != history.shape[1]:
        raise ValueError("history must be square")
    _finite(history, "history")
    with np.load(stat_path, allow_pickle=False) as archive:
        raw_array = archive["mom2.mom2"]
        count_array = archive["mom2.count"]
        if count_array.size != 1:
            raise ValueError("mom2.count must be scalar")
        count_value = float(count_array.item())
        if not math.isfinite(count_value) or count_value <= 0 or not count_value.is_integer():
            raise ValueError("mom2.count must be a positive integer")
        if raw_array.dtype != np.float32 or raw_array.shape != tuple(history.shape):
            raise ValueError("mom2.mom2 must be FP32 and match history shape")
        raw = torch.from_numpy(raw_array)
    _finite(raw, "mom2.mom2")
    raw.div_(int(count_value))
    A = raw.to(device=device, dtype=torch.float64)
    del raw, raw_array
    A.mul_(lambda_c)
    for start in range(0, A.shape[0], block_size):
        A[start:start + block_size].add_(
            history[start:start + block_size].to(device=device, dtype=torch.float64)
        )
    _finite(A, "A")
    return A


@torch.no_grad()
def solve_basis(
    keys: torch.Tensor,
    alpha: torch.Tensor,
    request_index: torch.Tensor,
    B: int,
    A: torch.Tensor,
    *,
    backend: str = "auto",
    dual_context_limit: int = 2048,
    block_size: int = 256,
    residual_tolerance: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return ``P[d_in,B]`` solving the complete context ridge system.

``keys[d_in,ncontext]`` are entry keys; alpha is nonnegative and sums to one
for each request.  Request contexts need not be contiguous or equal in count.
The dual uses C=sqrt(alpha)*keys and R[c,r]=sqrt(alpha[c]).  The primal adds
the same C C.T in blocks.  Both use the *whole* actual logical batch.

``residual_tolerance`` is a fixed caller/qualification setting, never relaxed
after observing a failed solve.  Metadata is scalar-only and holds no factor,
context tensor, or alternate weight state.  Inputs remain unchanged.
"""
    _positive_integer(B, "B")
    _positive_integer(block_size, "block_size")
    _positive_integer(dual_context_limit, "dual_context_limit")
    if not math.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("residual_tolerance must be finite and positive")
    aliases = {"dual": "full_context_dual", "primal": "equivalent_primal_blocked"}
    backend = aliases.get(backend, backend)
    if backend not in {"auto", "full_context_dual", "equivalent_primal_blocked"}:
        raise ValueError("unsupported geometry backend")
    if not isinstance(keys, torch.Tensor) or keys.ndim != 2 or min(keys.shape) <= 0:
        raise ValueError("keys must have nonempty shape [d_in,ncontext]")
    if not keys.is_floating_point():
        raise ValueError("keys must be floating point")
    d_in, ncontext = keys.shape
    if not isinstance(A, torch.Tensor) or A.shape != (d_in, d_in) or not A.is_floating_point():
        raise ValueError("A must have shape [d_in,d_in]")
    weights = torch.as_tensor(alpha, device=keys.device).detach().to(torch.float64)
    indices = torch.as_tensor(request_index, device=keys.device).detach()
    if weights.shape != (ncontext,) or indices.shape != (ncontext,):
        raise ValueError("alpha and request_index must have shape [ncontext]")
    if indices.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise ValueError("request_index must be integer")
    indices = indices.long()
    if bool(((indices < 0) | (indices >= B)).any()):
        raise ValueError("request_index is outside the actual batch")
    _finite(weights, "alpha")
    if bool((weights < 0).any()):
        raise ValueError("alpha must be nonnegative")
    sums = torch.zeros(B, device=keys.device, dtype=torch.float64)
    sums.index_add_(0, indices, weights)
    # Accommodates native FP32 .5/.1 weights without renormalizing them.
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-7, rtol=1e-7):
        raise ValueError("alpha must sum to one per actual request")
    key64 = keys.detach().to(dtype=torch.float64)
    A64 = A.detach().to(device=keys.device, dtype=torch.float64)
    _finite(key64, "keys")
    _finite(A64, "A")
    _symmetric(A64, block_size)
    kbar = torch.zeros((d_in, B), dtype=torch.float64, device=keys.device)
    for start in range(0, ncontext, block_size):
        stop = start + block_size
        kbar.index_add_(1, indices[start:stop], key64[:, start:stop] * weights[start:stop])
    if backend == "auto":
        backend = ("full_context_dual" if ncontext <= min(d_in, dual_context_limit)
                   else "equivalent_primal_blocked")
    # Check A itself on both paths.  Positive definiteness supplied only by
    # the current keys is not a substitute for the specified SPD prior.
    factor = _cholesky(A64, "A")
    spd_descriptor = dict(A_diagonal_min=float(A64.diagonal().min()),
                          A_diagonal_max=float(A64.diagonal().max()),
                          A_cholesky_diagonal_min=float(factor.diagonal().min()),
                          A_cholesky_diagonal_max=float(factor.diagonal().max()),
                          condition_number="NOT_ESTIMATED_DIAGONALS_ARE_NOT_A_CONDITION_NUMBER")
    if backend == "full_context_dual":
        root_alpha = weights.sqrt()
        C = key64 * root_alpha
        T = torch.cholesky_solve(C, factor)
        del factor
        S = C.T @ T
        S.diagonal().add_(1.0)
        R = torch.zeros((ncontext, B), dtype=torch.float64, device=keys.device)
        R[torch.arange(ncontext, device=keys.device), indices] = root_alpha
        dual_factor = _cholesky(S, "context dual")
        P = T @ torch.cholesky_solve(R, dual_factor)
        del C, T, S, R, dual_factor
        system_dimension = ncontext
    else:
        del factor
        M = A64.clone()
        for start in range(0, ncontext, block_size):
            Cblock = key64[:, start:start + block_size] * weights[start:start + block_size].sqrt()
            M.addmm_(Cblock, Cblock.T)
        del Cblock
        factor = _cholesky(M, "full-context primal")
        del M
        P = torch.empty_like(kbar)
        for start in range(0, B, block_size):
            P[:, start:start + block_size] = torch.cholesky_solve(kbar[:, start:start + block_size], factor)
        del factor
        system_dimension = d_in
    _finite(P, "P")
    def measured_residual(candidate: torch.Tensor) -> float:
        residual = A64 @ candidate - kbar
        for start in range(0, ncontext, block_size):
            block = key64[:, start:start + block_size]
            residual.add_(block @ (weights[start:start + block_size, None] * (block.T @ candidate)))
        return float(residual.norm() / kbar.norm().clamp_min(torch.finfo(torch.float64).tiny))

    scaled_residual = measured_residual(P)
    initial_residual = scaled_residual
    reference_residual = reference_relative_difference = None
    reference_used = False
    if not math.isfinite(scaled_residual) or scaled_residual > residual_tolerance:
        # One independent, bounded same-M reference check.  This is not a
        # jitter retry or permission to widen a qualification threshold.
        M = A64.clone()
        for start in range(0, ncontext, block_size):
            Cblock = key64[:, start:start + block_size] * weights[start:start + block_size].sqrt()
            M.addmm_(Cblock, Cblock.T)
        reference = torch.linalg.solve(M, kbar)
        del M, Cblock
        _finite(reference, "same-M reference P")
        reference_residual = measured_residual(reference)
        reference_relative_difference = float((reference - P).norm() /
                                             reference.norm().clamp_min(torch.finfo(torch.float64).tiny))
        if not math.isfinite(reference_residual) or reference_residual > residual_tolerance:
            raise FloatingPointError(
                f"geometry scaled residual {scaled_residual}; same-M reference residual "
                f"{reference_residual} exceeds fixed {residual_tolerance}"
            )
        P, scaled_residual, reference_used = reference, reference_residual, True
    return P, {
        "backend": backend, "d_in": d_in, "actual_B": B, "context_columns": ncontext,
        "system_dimension": system_dimension, "geometry_dtype": "float64",
        "scaled_residual": scaled_residual, "residual_tolerance": residual_tolerance,
        "initial_scaled_residual": initial_residual, "same_M_reference_used": reference_used,
        "same_M_reference_residual": reference_residual,
        "same_M_reference_relative_difference": reference_relative_difference,
        "alpha_sum_min": float(sums.min()), "alpha_sum_max": float(sums.max()),
        "full_context_second_moment": True, "fixed_entry_basis": True, "jitter": 0.0,
        "block_size": block_size,
        **spd_descriptor,
    }


def solve_layer_from_npz(
    stat_path: str | Path,
    history: torch.Tensor,
    keys: torch.Tensor,
    alpha: torch.Tensor,
    request_index: torch.Tensor,
    B: int,
    *,
    lambda_c: float,
    **solve_options: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """One-layer loading/solve helper; no persistent covariance/factor cache."""
    A = build_prior_from_npz(stat_path, history, lambda_c=lambda_c, device=keys.device,
                             block_size=solve_options.get("block_size", 256))
    P, metadata = solve_basis(keys, alpha, request_index, B, A, **solve_options)
    metadata.update(lambda_c=float(lambda_c), C0_normalization="FP32_mom2_div_count_then_FP64",
                    history_conversion="FP32_then_FP64", prior_storage="one_layer_transient")
    return P, metadata
