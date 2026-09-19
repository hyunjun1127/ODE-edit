"""Pinned engineering gates; CPU/GPU tensors, FP64 checks, no model calls.

These receipts report numerical comparisons, NOT provenance or empirical model
validation.  ``passed`` may be true for NUMERIC_BOUNDARY, but changed success bits
remain explicit.  All thresholds are the unchanged 2026-09-20 contract values.
No averaging can hide a failing element, RHS column, request, or evaluation row.
"""
from __future__ import annotations

import time
from typing import Sequence

import torch


RTOL = 1e-5
ATOL = 1e-6
SOLVE_REL = 1e-5
DELTA_REL = 1e-3
RESPONSE_REL = 1e-3
NLL_ATOL = 1e-4
ZERO_MAX = 1e-12
ZERO_ABS = 1e-7
REPEAT_FRACTION = 0.1


def _check(*values: torch.Tensor) -> None:
    if not values or any(not isinstance(t, torch.Tensor) for t in values):
        raise TypeError("gates require torch tensors")
    if any(t.numel() == 0 for t in values):
        raise ValueError("an empty comparison cannot establish a gate")
    if any(t.device != values[0].device for t in values):
        raise ValueError("all inputs must share a device")
    if any(t.is_complex() for t in values):
        raise ValueError("these real-valued contract gates do not accept complex inputs")


def _same(*values: torch.Tensor) -> None:
    _check(*values)
    if any(t.shape != values[0].shape for t in values):
        raise ValueError("shape mismatch; broadcasting is not a valid comparison")


def _new(name: str, *values: torch.Tensor) -> tuple[dict, float]:
    start = time.perf_counter()
    _check(*values)
    finite = all(bool(torch.isfinite(t).all().item()) for t in values)
    return {
        "gate": name,
        "status": "PASS" if finite else "FAILED",
        "passed": finite,
        "finite": finite,
        "input_dtypes": [str(t.dtype) for t in values],
        "input_shapes": [list(t.shape) for t in values],
        "device": str(values[0].device),
        "compute_dtype": "torch.float64",
        "reduction_dtype": "torch.float64",
        "reason": None if finite else "NONFINITE_INPUT",
    }, start


def _end(receipt: dict, start: float, passed: bool | None = None) -> dict:
    if passed is not None:
        receipt["passed"] = bool(passed)
        receipt["status"] = "PASS" if passed else "FAILED"
    receipt["verification_seconds"] = time.perf_counter() - start
    return receipt


def _index(flat_index: int, shape: Sequence[int]) -> list[int]:
    result = []
    for size in reversed(shape):
        result.append(flat_index % size)
        flat_index //= size
    return list(reversed(result))


def _double(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(dtype=torch.float64)


def _computed_finite(receipt: dict, *values: torch.Tensor) -> bool:
    if not all(bool(torch.isfinite(t).all().item()) for t in values):
        receipt.update(passed=False, status="FAILED", finite=False,
                       reason="NONFINITE_COMPUTATION")
        return False
    return True


def elementwise_gate(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    """Every element: abs(actual-ref) <= 1e-6 + 1e-5 * abs(ref)."""
    _same(actual, reference)
    receipt, start = _new("keys_blocks_elementwise", actual, reference)
    receipt.update(rtol=RTOL, atol=ATOL, aggregation="all_elements")
    if not receipt["finite"]:
        return _end(receipt, start)
    ref = _double(reference)
    error = (_double(actual) - ref).abs()
    allowance = ATOL + RTOL * ref.abs()
    ratio = error / allowance
    if not _computed_finite(receipt, error, allowance, ratio):
        return _end(receipt, start)
    failing = error > allowance
    worst = int(ratio.reshape(-1).argmax().item())
    receipt.update(
        elements=actual.numel(),
        failed_elements=int(failing.sum().item()),
        max_absolute_error=float(error.max().item()),
        max_tolerance_ratio=float(ratio.max().item()),
        worst_index=_index(worst, actual.shape),
        worst_absolute_error=float(error.reshape(-1)[worst].item()),
        worst_allowance=float(allowance.reshape(-1)[worst].item()),
    )
    return _end(receipt, start, not bool(failing.any().item()))


def _norm_check(error_norms: torch.Tensor, reference_norms: torch.Tensor,
                tolerance: float) -> dict:
    """Apply near-zero absolute rule per vector, without a denominator floor."""
    if not all(bool(torch.isfinite(t).all().item()) for t in (error_norms, reference_norms)):
        return {"passed": False, "finite": False, "reason": "NONFINITE_COMPUTATION"}
    zero = reference_norms <= ZERO_MAX
    relative = torch.zeros_like(error_norms)
    relative[~zero] = error_norms[~zero] / reference_norms[~zero]
    allowed = torch.where(zero, torch.full_like(error_norms, ZERO_ABS),
                          tolerance * reference_norms)
    failed = error_norms > allowed
    ratios = error_norms / allowed
    if not all(bool(torch.isfinite(t).all().item()) for t in (relative, allowed, ratios)):
        return {"passed": False, "finite": False, "reason": "NONFINITE_COMPUTATION"}
    worst = int(ratios.argmax().item())
    return {
        "passed": not bool(failed.any().item()),
        "relative_tolerance": tolerance,
        "zero_reference_max": ZERO_MAX,
        "zero_reference_absolute_atol": ZERO_ABS,
        "reference_norms": reference_norms.cpu().tolist(),
        "error_norms": error_norms.cpu().tolist(),
        "relative_errors": [None if bool(z) else float(r)
                            for z, r in zip(zero.cpu().tolist(), relative.cpu().tolist())],
        "zero_reference_flags": zero.cpu().tolist(),
        "zero_reference_count": int(zero.sum().item()),
        "failed_indices": failed.nonzero(as_tuple=False).flatten().cpu().tolist(),
        "max_relative_error": float(relative[~zero].max().item()) if bool((~zero).any().item()) else None,
        "max_zero_reference_absolute_error": float(error_norms[zero].max().item()) if bool(zero.any().item()) else None,
        "max_error_norm": float(error_norms.max().item()),
        "max_tolerance_ratio": float(ratios.max().item()),
        "worst_index": worst,
    }


def solve_residual_gate(matrix: torch.Tensor, solution: torch.Tensor,
                        rhs: torch.Tensor, *, row_chunk: int = 256) -> dict:
    """max_j ||A X_j - rhs_j||_2 / ||rhs_j||_2, never backward error.

    A and X are the original solve operands/results.  Residual products and
    reductions are FP64 and row-chunked; the gate does not rerun the solve.
    ``row_chunk`` changes verification storage only, not the checked system.
    """
    _check(matrix, solution, rhs)
    if matrix.ndim != 2 or solution.ndim != 2 or rhs.ndim != 2:
        raise ValueError("solve gate expects 2-D matrix, solution, and RHS")
    if matrix.shape[1] != solution.shape[0] or rhs.shape != (matrix.shape[0], solution.shape[1]):
        raise ValueError("incompatible linear-system dimensions")
    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")
    receipt, start = _new("solve_residual", matrix, solution, rhs)
    receipt.update(aggregation="max_rhs_column", denominator="norm2(rhs_column)", row_chunk=row_chunk)
    if not receipt["finite"]:
        return _end(receipt, start)
    x = _double(solution)
    error2 = torch.zeros(rhs.shape[1], dtype=torch.float64, device=rhs.device)
    ref2 = torch.zeros_like(error2)
    for first in range(0, rhs.shape[0], row_chunk):
        end = min(first + row_chunk, rhs.shape[0])
        b = _double(rhs[first:end])
        residual = _double(matrix[first:end]) @ x - b
        error2 += residual.square().sum(dim=0)
        ref2 += b.square().sum(dim=0)
    receipt.update(_norm_check(error2.sqrt(), ref2.sqrt(), SOLVE_REL))
    return _end(receipt, start, receipt["passed"])


def update_gate(reconstructed_delta: torch.Tensor, actual_delta: torch.Tensor) -> dict:
    """Frobenius discrepancy, denominator strictly the ACTUAL weight delta.

    Callers must form actual_delta as W_new.double() - W_old.double(), not
    cast an already rounded FP32 subtraction to double.
    """
    _same(reconstructed_delta, actual_delta)
    receipt, start = _new("actual_update", reconstructed_delta, actual_delta)
    receipt.update(aggregation="frobenius", denominator="normF(actual_delta)")
    if not receipt["finite"]:
        return _end(receipt, start)
    ref = _double(actual_delta)
    error = _double(reconstructed_delta) - ref
    receipt.update(_norm_check(torch.linalg.vector_norm(error).reshape(1),
                               torch.linalg.vector_norm(ref).reshape(1), DELTA_REL))
    return _end(receipt, start, receipt["passed"])


def response_gate(reconstructed_delta: torch.Tensor, actual_delta: torch.Tensor,
                  keys: torch.Tensor) -> dict:
    """max_request ||(delta_reconstructed-delta_actual) k|| / ||delta_actual k||."""
    _same(reconstructed_delta, actual_delta)
    _check(actual_delta, keys)
    if actual_delta.ndim != 2 or keys.ndim != 2 or actual_delta.shape[1] != keys.shape[0]:
        raise ValueError("expected deltas[out,in] and keys[in,request]")
    receipt, start = _new("actual_response", reconstructed_delta, actual_delta, keys)
    receipt.update(aggregation="max_request_column", denominator="norm2(actual_delta @ key)")
    if not receipt["finite"]:
        return _end(receipt, start)
    ref, k = _double(actual_delta), _double(keys)
    response = ref @ k
    error = (_double(reconstructed_delta) - ref) @ k
    receipt.update(_norm_check(torch.linalg.vector_norm(error, dim=0),
                               torch.linalg.vector_norm(response, dim=0), RESPONSE_REL))
    return _end(receipt, start, receipt["passed"])


def evaluation_rows_gate(actual_new_nll: torch.Tensor, actual_true_nll: torch.Tensor,
                         reference_new_nll: torch.Tensor, reference_true_nll: torch.Tensor,
                         metric_tags: Sequence[str]) -> dict:
    """Rowwise NLL/margin tolerance and strict preference bits; both-band rule.

    RS/PS safety=true-new; NS safety=new-true; exact ties are failures.  Only a
    changed bit for which BOTH margins lie within [-1e-4,+1e-4] is a numerical
    boundary.  Identity/order equality must be checked by the caller first.
    """
    values = (actual_new_nll, actual_true_nll, reference_new_nll, reference_true_nll)
    _same(*values)
    if actual_new_nll.ndim != 1 or len(metric_tags) != actual_new_nll.numel():
        raise ValueError("NLL arrays must be one-dimensional with one metric tag per row")
    if any(tag not in {"RS", "PS", "NS"} for tag in metric_tags):
        raise ValueError("metric tags must be RS, PS, or NS")
    receipt, start = _new("evaluation_rows", *values)
    receipt.update(aggregation="max_row_absolute_discrepancy", atol_nats=NLL_ATOL,
                   robust_success_bit_mismatch_allowed=0, identity_order_checked_by="caller")
    if not receipt["finite"]:
        return _end(receipt, start)
    an, at, rn, rt = map(_double, values)
    signs = torch.tensor([-1.0 if tag == "NS" else 1.0 for tag in metric_tags],
                         dtype=torch.float64, device=an.device)
    margin, ref_margin = signs * (at - an), signs * (rt - rn)
    nll_error = torch.maximum((an - rn).abs(), (at - rt).abs())
    margin_error = (margin - ref_margin).abs()
    if not _computed_finite(receipt, margin, ref_margin, nll_error, margin_error):
        return _end(receipt, start)
    mismatched = (margin > 0) != (ref_margin > 0)
    both_near = (margin.abs() <= NLL_ATOL) & (ref_margin.abs() <= NLL_ATOL)
    boundary = mismatched & both_near
    robust = mismatched & ~both_near
    failed_rows = (nll_error > NLL_ATOL) | (margin_error > NLL_ATOL) | robust
    receipt.update(
        rows=len(metric_tags),
        max_nll_absolute_error=float(nll_error.max().item()),
        max_margin_absolute_error=float(margin_error.max().item()),
        worst_nll_row=int(nll_error.argmax().item()),
        worst_margin_row=int(margin_error.argmax().item()),
        changed_success_bits=int(mismatched.sum().item()),
        numeric_boundary_count=int(boundary.sum().item()),
        numeric_boundary_rows=boundary.nonzero(as_tuple=False).flatten().cpu().tolist(),
        robust_success_bit_mismatches=int(robust.sum().item()),
        robust_mismatch_rows=robust.nonzero(as_tuple=False).flatten().cpu().tolist(),
        failed_rows=failed_rows.nonzero(as_tuple=False).flatten().cpu().tolist(),
        actual_successes=int((margin > 0).sum().item()),
        reference_successes=int((ref_margin > 0).sum().item()),
        actual_ties=int((margin == 0).sum().item()),
        reference_ties=int((ref_margin == 0).sum().item()),
    )
    _end(receipt, start, not bool(failed_rows.any().item()))
    if receipt["passed"] and receipt["numeric_boundary_count"]:
        receipt["status"] = "NUMERIC_BOUNDARY"
    return receipt


def repeat_spread_gate(observations: torch.Tensor, tolerance: float | torch.Tensor) -> dict:
    """Across >=2 repeats, elementwise max-min <= 10% of declared tolerance.

    First dimension indexes repeats.  ``tolerance`` is the preregistered
    absolute threshold in the observation's units, either scalar or exactly
    observations.shape[1:].  For key/block tensors use ATOL+RTOL*abs(reference);
    for already-normalized solve/update/response errors use their fixed relative
    threshold.  This function never estimates a tolerance from observed spread.
    """
    _check(observations)
    if observations.ndim < 1 or observations.shape[0] < 2:
        raise ValueError("repeat gate needs at least two observations")
    tol = torch.as_tensor(tolerance, dtype=torch.float64, device=observations.device).detach()
    if tol.ndim != 0 and tol.shape != observations.shape[1:]:
        raise ValueError("tolerance must be scalar or exactly match a single observation")
    if not bool(torch.isfinite(tol).all().item()) or not bool((tol > 0).all().item()):
        raise ValueError("tolerance must be finite and positive")
    receipt, start = _new("repeat_spread", observations)
    receipt.update(repeats=observations.shape[0], fraction=REPEAT_FRACTION,
                   aggregation="max_elementwise_range", tolerance_is_preregistered=True)
    if not receipt["finite"]:
        return _end(receipt, start)
    values = _double(observations)
    spread = values.amax(dim=0) - values.amin(dim=0)
    allowance = REPEAT_FRACTION * tol
    if not _computed_finite(receipt, spread, spread / allowance):
        return _end(receipt, start)
    failed = spread > allowance
    receipt.update(max_absolute_spread=float(spread.max().item()),
                   max_tolerance_ratio=float((spread / allowance).max().item()),
                   failed_elements=int(failed.sum().item()))
    return _end(receipt, start, not bool(failed.any().item()))
