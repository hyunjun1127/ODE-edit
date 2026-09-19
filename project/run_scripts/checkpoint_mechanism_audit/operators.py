"""Raw nonsymmetric history operators and native-demand algebra.

No model calls, edited checkpoint writes, inverse, CG, or Cholesky occur here.
Model/source FP32 solves and independent FP64 checks are deliberately separate.
All tensors use writer orientation K:[d,n], R:[o,n], delta:[o,d].
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch


ZERO_REFERENCE = 1e-12
ZERO_ABSOLUTE_ATOL = 1e-7


def _finite(*xs: torch.Tensor) -> None:
    if not all(bool(torch.isfinite(x).all()) for x in xs):
        raise ValueError("NONFINITE_OPERATOR_INPUT_OR_OUTPUT")


def norm_discrepancy(error: torch.Tensor, reference: torch.Tensor,
                     relative_tolerance: float, *, columnwise: bool = False) -> dict:
    """Contract norm denominator and near-zero absolute rule; never a floor."""
    _finite(error, reference)
    e, r = error.double(), reference.double()
    if columnwise:
        if e.ndim != 2 or r.shape != e.shape:
            raise ValueError("columnwise norm requires matching matrices")
        en = torch.linalg.vector_norm(e, dim=0)
        rn = torch.linalg.vector_norm(r, dim=0)
    else:
        en, rn = torch.linalg.vector_norm(e).reshape(1), torch.linalg.vector_norm(r).reshape(1)
    zero = rn <= ZERO_REFERENCE
    relative = torch.full_like(rn, float("nan"))
    relative[~zero] = en[~zero] / rn[~zero]
    passing = torch.where(zero, en <= ZERO_ABSOLUTE_ATOL,
                          relative <= relative_tolerance)
    return dict(passed=bool(passing.all()), compute_dtype="float64",
                denominator="reference_column_l2" if columnwise else "reference_frobenius",
                relative_tolerance=relative_tolerance,
                max_relative=float(relative[~zero].max()) if bool((~zero).any()) else None,
                max_absolute=float(en.max()) if en.numel() else 0.0,
                zero_reference_count=int(zero.sum()), zero_reference_limit=ZERO_REFERENCE,
                zero_reference_absolute_atol=ZERO_ABSOLUTE_ATOL,
                reference_norms=rn.cpu().tolist(), absolute_errors=en.cpu().tolist(),
                relative_errors=[None if bool(z) else float(v) for z, v in zip(zero, relative)],
                zero_reference_flags=zero.cpu().tolist(),
                failed_indices=torch.where(~passing)[0].cpu().tolist())


def elementwise_parity(new: torch.Tensor, reference: torch.Tensor,
                       rtol: float = 1e-5, atol: float = 1e-6) -> dict:
    if new.shape != reference.shape:
        raise ValueError("elementwise parity shape mismatch")
    _finite(new, reference)
    error = (new.double() - reference.double()).abs()
    tolerance = atol + rtol * reference.double().abs()
    return dict(passed=bool((error <= tolerance).all()), rtol=rtol, atol=atol,
                max_absolute=float(error.max()) if error.numel() else 0.0,
                failed_elements=int((error > tolerance).sum()), elements=error.numel(),
                check_precision="float64")


def solve_residual(matrix: torch.Tensor, solution: torch.Tensor, rhs: torch.Tensor,
                   tolerance: float = 1e-5, column_chunk: int = 128) -> dict:
    """max_j ||A x_j-b_j||/||b_j||, with FP64 contraction/reductions."""
    start = time.perf_counter()
    if matrix.shape[0] != matrix.shape[1] or solution.shape != rhs.shape:
        raise ValueError("invalid solve shapes")
    # Column chunks bound verification workspace without altering RHS semantics.
    errors = []
    matrix64 = matrix.double()
    for start_col in range(0, rhs.shape[1], column_chunk):
        sl = slice(start_col, start_col + column_chunk)
        errors.append(matrix64 @ solution[:, sl].double() - rhs[:, sl].double())
    error = torch.cat(errors, 1) if errors else rhs.double().clone()
    result = norm_discrepancy(error, rhs, tolerance, columnwise=True)
    result.update(verification_seconds=time.perf_counter()-start,
                  denominator="per_rhs_column_norm2_not_backward_error",
                  matrix_source_dtype=str(matrix.dtype))
    return result


@dataclass
class HistoryOperator:
    """One immutable raw H=lambda I+P M LU reused for any RHS bank.

    Bank columns share factorization only; each native block subsequently builds
    its own S/B. Callers own scheduling and factor/output single-writer rules.
    """
    projector: torch.Tensor
    history: torch.Tensor
    lambda_write: float = 1.0

    def __post_init__(self) -> None:
        p, m = self.projector, self.history
        if p.ndim != 2 or p.shape[0] != p.shape[1] or m.shape != p.shape:
            raise ValueError("P/M must be same square shape")
        if p.dtype != m.dtype or p.device != m.device:
            raise ValueError("P/M dtype/device must already be explicitly bound")
        if self.lambda_write <= 0 or not math.isfinite(self.lambda_write):
            raise ValueError("lambda_write must be positive finite")
        _finite(p, m)
        start = time.perf_counter()
        self.matrix = p @ m + self.lambda_write * torch.eye(p.shape[0], dtype=p.dtype, device=p.device)
        self.lu, self.pivots, info = torch.linalg.lu_factor_ex(self.matrix)
        if int(info) != 0:
            raise RuntimeError(f"LU_FACTORIZATION_FAILURE:{int(info)}")
        _finite(self.lu)
        self.factor_seconds = time.perf_counter()-start
        self.factorization_count = 1

    def solve_bank(self, keys: torch.Tensor, *, verify: bool = True,
                   column_chunk: int = 128) -> tuple[torch.Tensor, dict]:
        if keys.ndim != 2 or keys.shape[0] != self.matrix.shape[0]:
            raise ValueError("key bank shape mismatch")
        if keys.dtype != self.matrix.dtype or keys.device != self.matrix.device:
            raise ValueError("key bank dtype/device must match raw operator")
        _finite(keys)
        start = time.perf_counter()
        rhs = self.projector @ keys
        y = torch.linalg.lu_solve(self.lu, self.pivots, rhs)
        _finite(y)
        receipt = dict(solve_seconds=time.perf_counter()-start,
                       factor_seconds=self.factor_seconds, factorization_count=1,
                       solve_dtype=str(y.dtype), history_symmetric_assumed=False,
                       rhs_columns=keys.shape[1], rhs_bank_is_native_batch=False,
                       minimum_absolute_pivot=float(self.lu.diagonal().abs().min()))
        if verify:
            receipt["residual"] = solve_residual(self.matrix, y, rhs, column_chunk=column_chunk)
        return y, receipt


def factor_from_response(keys: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-native-batch S=K.T Y, B=Y(I+S)^-1; raw S need not be symmetric."""
    if keys.shape != y.shape:
        raise ValueError("K/Y shape mismatch")
    s = keys.T @ y
    small = torch.eye(s.shape[0], dtype=s.dtype, device=s.device) + s
    b = torch.linalg.solve(small.T, y.T).T
    _finite(s, b)
    return s, b


def native_dense(projector: torch.Tensor, history: torch.Tensor, keys: torch.Tensor,
                 residual: torch.Tensor, lambda_write: float = 1.0,
                 *, verify: bool = True) -> tuple[torch.Tensor, dict]:
    """Original dense RHS operation order, one orientation adaptation here."""
    start = time.perf_counter()
    matrix = projector @ (keys @ keys.T + history)
    matrix = matrix + lambda_write * torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    rhs = (projector @ keys) @ residual.T
    solution = torch.linalg.solve(matrix, rhs)
    _finite(solution)
    receipt = dict(solve_seconds=time.perf_counter()-start, compute_dtype=str(matrix.dtype),
                   equation="solve(P@(K@K.T+M)+lambda*I,(P@K)@R.T).T")
    if verify:
        receipt["residual"] = solve_residual(matrix, solution, rhs)
    return solution.T, receipt


def reconstruction_metrics(actual_delta: torch.Tensor, reconstructed_delta: torch.Tensor,
                           keys: torch.Tensor) -> dict:
    actual, reconstructed, k = actual_delta.double(), reconstructed_delta.double(), keys.double()
    error = reconstructed - actual
    return dict(update=norm_discrepancy(error, actual, 1e-3),
                response=norm_discrepancy(error @ k, actual @ k, 1e-3, columnwise=True),
                actual_norm=float(torch.linalg.vector_norm(actual)),
                reconstructed_norm=float(torch.linalg.vector_norm(reconstructed)),
                subtraction_precision="float64", contraction_precision="float64")


def fixed_probe_scores(projector: torch.Tensor, keys: torch.Tensor, y: torch.Tensor,
                       lambda_write: float = 1.0) -> list[dict]:
    """Individual-key scores; does not treat probe512 as a native batch."""
    p, k, yy = projector.double(), keys.double(), y.double()
    pk = p @ k
    ppk = p @ pk
    scores = (k * yy).sum(0)
    norm_pk2 = pk.square().sum(0)
    p_action_error = torch.linalg.vector_norm(ppk-pk, dim=0)
    rows = []
    for j in range(k.shape[1]):
        exact_zero = bool(norm_pk2[j] == 0)
        near_zero = bool(torch.sqrt(norm_pk2[j]) <= ZERO_REFERENCE)
        nu = None if exact_zero else float(lambda_write*scores[j]/norm_pk2[j])
        rows.append(dict(probe_index=j, raw_score=float(scores[j]), nu=nu,
                         projected_key_squared_norm=float(norm_pk2[j]),
                         pk_zero=exact_zero, pk_near_zero=near_zero,
                         negative_score=bool(scores[j] < 0),
                         nu_outside_ideal_range=None if nu is None else not (0 < nu <= 1),
                         projector_action_absolute_error=float(p_action_error[j]),
                         raw_score_clipped=False, precision="float64"))
    return rows


def native_mode_decomposition(b: torch.Tensor, residual: torch.Tensor,
                              actual_delta: torch.Tensor | None = None,
                              near_degenerate_relative_gap: float = 1e-3) -> dict[str, Any]:
    """Direct thin SVD(B), not Gram eigendecomposition; no mode truncation.

    The reported near-degenerate gap is descriptive, never an acceptance gate.
    Residual versus an actual/dense reference is not presumed rounding noise.
    """
    if near_degenerate_relative_gap < 0:
        raise ValueError("negative diagnostic gap")
    start = time.perf_counter()
    bb, r = b.double(), residual.double()
    left, sigma, vh = torch.linalg.svd(bb, full_matrices=False)
    loading = r @ vh.T
    loading2 = loading.square().sum(0)
    energy = sigma.square()*loading2
    modes, groups = [], []
    group = 0
    for j in range(sigma.numel()):
        if j and float(sigma[j-1]-sigma[j]) > near_degenerate_relative_gap*float(sigma[0]):
            group += 1
        modes.append(dict(mode=j, gain=float(sigma[j]), target_loading_squared=float(loading2[j]),
                          write_energy=float(energy[j]), near_degenerate_group=group))
    for gid in range(group+1):
        indices = [m["mode"] for m in modes if m["near_degenerate_group"] == gid]
        groups.append(dict(group=gid, first_mode=min(indices), last_mode=max(indices),
                           modes=len(indices), loading_squared=float(loading2[indices].sum()),
                           write_energy=float(energy[indices].sum())))
    delta = r @ bb.T
    factor_energy = float(delta.square().sum())
    result: dict[str, Any] = dict(modes=modes, groups=groups,
        mode_energy_sum=float(energy.sum()), factor_energy=factor_energy,
        energy_identity_absolute_error=abs(factor_energy-float(energy.sum())),
        svd_reconstruction_relative=float(torch.linalg.vector_norm((left*sigma)@vh-bb)/torch.linalg.vector_norm(bb)) if bool(torch.linalg.vector_norm(bb)>0) else None,
        zero_factor=bool(torch.linalg.vector_norm(bb)==0),
        left_orthogonality_error=float(torch.linalg.vector_norm(left.T@left-torch.eye(sigma.numel(),dtype=bb.dtype,device=bb.device))),
        near_degenerate_relative_gap=near_degenerate_relative_gap,
        near_degenerate_rule="adjacent gap <= diagnostic_relative_gap * largest_sigma; transitive groups",
        svd_input="direct_B", precision="float64", seconds=time.perf_counter()-start)
    if actual_delta is not None:
        error = actual_delta.double()-delta
        cross = float(2*(delta*error).sum())
        error_energy = float(error.square().sum())
        actual_energy = float(actual_delta.double().square().sum())
        result["reference_residual"] = dict(kind="FACTOR_NATIVE_REPRODUCTION_RESIDUAL_NOT_ASSUMED_ROUNDING",
            cross_term=cross, residual_energy=error_energy, actual_energy=actual_energy,
            energy_closure_error=actual_energy-factor_energy-cross-error_energy)
    return result


def fitting_transfer(residual: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """R(I+S)^(-T)S.T, preserving nonsymmetric orientation."""
    small = torch.eye(s.shape[0], dtype=s.dtype, device=s.device)+s
    return torch.linalg.solve(small, residual.T).T @ s.T


def counterfactual_summary(keys: torch.Tensor, residual: torch.Tensor, b: torch.Tensor,
                            w0_norm: float | None = None, permutations: int = 20,
                            seed: int = 20260920) -> list[dict]:
    """Fixed operator, original + exactly 20 semantic-invalid R permutations."""
    if permutations != 20:
        raise ValueError("contract requires exactly20 residual permutations")
    k, r, bb = keys.double(), residual.double(), b.double()
    _, sigma, vh = torch.linalg.svd(bb, full_matrices=False)
    transfer = bb.T @ k
    rng = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for index in range(permutations+1):
        permutation = torch.arange(r.shape[1]) if index == 0 else torch.randperm(r.shape[1],generator=rng)
        rr = r[:, permutation.to(r.device)]
        energy = float(((rr @ vh.T).square().sum(0)*sigma.square()).sum())
        error = rr @ transfer-rr
        demand = norm_discrepancy(error, rr, 1e-3)
        rows.append(dict(permutation_index=index-1, kind="ORIGINAL_ALIGNMENT" if index == 0 else "R_COLUMN_PERMUTATION",
            permutation=permutation.tolist(), seed=seed, write_norm=math.sqrt(energy), write_energy=energy,
            relative_w0_norm=math.sqrt(energy)/w0_norm if w0_norm and w0_norm>0 else None,
            target_error_relative=demand["max_relative"], target_error_absolute=demand["max_absolute"],
            zero_demand=demand["zero_reference_count"]>0, residual_frobenius=float(torch.linalg.vector_norm(rr)),
            factual_edit_arm=False, precision="float64"))
    return rows
