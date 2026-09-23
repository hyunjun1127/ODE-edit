"""Bounded FP64 diagnostics for actual native writer factors, never a writer.

Large quadratic traces use one physical layer of GPU-resident FP64 P/M and
row-block products.  The native FP32 solve, stored P, M and deltas are untouched.
CPU tests exercise the same route on tiny matrices; no GPU PASS is implied.
"""
from __future__ import annotations

import math
import time
from typing import Any, MutableMapping

import numpy as np
import torch

from .common import tensor_sha
from .geometry import GeometryError, writer_modes


BLOCK_ROWS = 128
ROUTE = "torch-FP64-rowblocks128-matmul-inner-sum-then-ordered-block-scalar-add-v1"


class WriterDiagnosticError(RuntimeError):
    pass


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _matrix(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.dtype not in (torch.float32, torch.float64):
        raise WriterDiagnosticError(f"{name}: expected FP32/FP64 matrix")
    for start in range(0, value.shape[0], BLOCK_ROWS):
        if not bool(torch.isfinite(value[start:start+BLOCK_ROWS]).all()):
            raise WriterDiagnosticError(f"{name}: nonfinite input")


def _ratio(a, b):
    return None if b == 0 else float(a/b)


def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise WriterDiagnosticError("nonfinite diagnostic reduction")
    return number


def quadratic_trace(left: torch.Tensor, middle: torch.Tensor, *, block_rows=BLOCK_ROWS) -> float:
    """tr(left middle left.T) on explicitly resident FP64 tensors.

    Each block is ``rows @ middle`` followed by the FP64 elementwise dot with
    rows. Block scalar results accumulate in increasing row order. No dense
    left-middle-left.T matrix and no CPU d-cubed operation is constructed.
    """
    if (left.dtype != torch.float64 or middle.dtype != torch.float64
            or left.device != middle.device or block_rows < 1
            or middle.shape != (left.shape[1], left.shape[1])):
        raise WriterDiagnosticError("quadratic_trace route/shape mismatch")
    total = torch.zeros((), dtype=torch.float64, device=left.device)
    for start in range(0, left.shape[0], block_rows):
        rows = left[start:start+block_rows]
        product = rows @ middle
        total = total + (rows * product).sum()
    return _number(total)


def _action_energy(left, keys):
    # Keys [d,n]; no output_dim x n tensor is retained across row blocks.
    answer = torch.zeros(keys.shape[1], dtype=torch.float64, device=left.device)
    for start in range(0, left.shape[0], BLOCK_ROWS):
        action = left[start:start+BLOCK_ROWS] @ keys
        answer = answer + action.square().sum(dim=0)
    if not bool(torch.isfinite(answer).all()):
        raise WriterDiagnosticError("nonfinite diagnostic action")
    return answer


def projector_diagnostics(P: torch.Tensor) -> dict[str, Any]:
    """Stored P symmetry/idempotence errors; no orthogonalization or PASS gate."""
    if P.dtype != torch.float64 or P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise WriterDiagnosticError("projector diagnostic requires square resident FP64 P")
    norm2 = torch.zeros((), dtype=torch.float64, device=P.device)
    sym2, idem2 = norm2.clone(), norm2.clone()
    symmax, idemmax = norm2.clone(), norm2.clone()
    for start in range(0, len(P), BLOCK_ROWS):
        rows = P[start:start+BLOCK_ROWS]
        symmetry = rows - P.T[start:start+BLOCK_ROWS]
        idempotence = rows @ P - rows
        norm2 = norm2 + rows.square().sum()
        sym2 = sym2 + symmetry.square().sum()
        idem2 = idem2 + idempotence.square().sum()
        symmax = torch.maximum(symmax, symmetry.abs().max())
        idemmax = torch.maximum(idemmax, idempotence.abs().max())
    norm = math.sqrt(_number(norm2))
    symmetry_norm, idempotence_norm = math.sqrt(_number(sym2)), math.sqrt(_number(idem2))
    return dict(trace=_number(torch.trace(P)), frobenius=norm,
                symmetry_frobenius=symmetry_norm,
                symmetry_relative_frobenius=_ratio(symmetry_norm, norm),
                symmetry_max_abs=_number(symmax),
                idempotence_frobenius=idempotence_norm,
                idempotence_relative_frobenius=_ratio(idempotence_norm, norm),
                idempotence_max_abs=_number(idemmax),
                status="MEASURED_NOT_AN_IDEAL_PROJECTOR_CERTIFICATE",
                ideal_exact_projector="NOT_ESTABLISHED", orthogonalization=False)


def _small_modes(factor):
    """Preserve raw nonsymmetric H; sym(H) is an explicitly labeled approximation."""
    K, R, delta = (factor[key].detach().cpu().numpy() for key in ("K", "R", "delta"))
    raw_H = factor.get("H")
    raw_receipt = dict(
        C_sha256=tensor_sha(factor["C"]) if factor.get("C") is not None else None,
        G_sha256=tensor_sha(factor["G"]) if factor.get("G") is not None else None,
        H_sha256=tensor_sha(raw_H) if isinstance(raw_H, torch.Tensor) else None,
        raw_factor_artifact=factor.get("artifact"), raw_C_H_preserved=True,
        gain_orientation="raw G=K.T@C; actual response gain=C.T@K",
        native_solver_validity="SEPARATE_FROM_DIAGNOSTIC_H_PSD",
        ideal_orthogonal_P_identity="NOT_ESTABLISHED")
    H_for_modes = None
    if raw_H is None:
        raw_receipt.update(status="NOT_APPLICABLE", reason="RAW_H_NOT_AVAILABLE")
    else:
        _matrix(raw_H, "raw H")
        H = raw_H.detach().cpu().double().numpy()
        if H.shape != (K.shape[1], K.shape[1]):
            raise WriterDiagnosticError("raw H request dimensions differ")
        sym = (H + H.T)*.5
        eigenvalues = np.linalg.eigvalsh(sym)
        norm = float(np.linalg.norm(H))
        tolerance = 64*np.finfo(np.float64).eps*len(H)*float(np.linalg.norm(sym, ord=np.inf))
        negative = eigenvalues[eigenvalues < 0]
        condition = float(np.linalg.cond(H))
        raw_receipt.update(
            raw_asymmetry_frobenius=float(np.linalg.norm(H-H.T)),
            raw_asymmetry_relative=_ratio(float(np.linalg.norm(H-H.T)), norm),
            raw_asymmetry_max_abs=float(np.max(np.abs(H-H.T))),
            raw_condition_2=condition if math.isfinite(condition) else None,
            raw_condition_status="FINITE" if math.isfinite(condition) else "SINGULAR_OR_UNRESOLVED",
            raw_trace=float(np.trace(H)),
            symmetric_eigenvalues_ascending=eigenvalues.tolist(),
            symmetric_minimum_eigenvalue=float(eigenvalues[0]),
            symmetric_negative_count=int(len(negative)),
            symmetric_negative_mass=float(-negative.sum()),
            symmetric_roundoff_tolerance=tolerance,
            symmetrization="DIAGNOSTIC_APPROXIMATION_ONLY_NATIVE_H_UNMODIFIED")
        if float(eigenvalues[0]) < -tolerance:
            raw_receipt.update(status="NOT_APPLICABLE", reason="SYMMETRIC_PART_NOT_PSD")
        else:
            H_for_modes = sym
            raw_receipt.update(status="APPROXIMATE_MODES_DEFINED",
                               reason="PSD_SYMMETRIC_PART_WITH_RECORDED_ROUNDOFF_POLICY")
    # No M/P/bank inputs: this path cannot execute geometry's CPU d^3 traces.
    result = writer_modes(K, R, delta, history_whitened_gram=H_for_modes)
    if H_for_modes is not None:
        result["history_whitened_modes_status"] = "SYMMETRIZED_NATIVE_H_APPROXIMATION_NOT_IDEAL_P"
        result["ideal_formula_interpretation"] = "DIAGNOSTIC_APPROXIMATION_ONLY"
    else:
        result["history_whitened_modes_status"] = "NOT_APPLICABLE"
    result["raw_native_H"] = raw_receipt
    result["native_delta_vs_R_C_transpose"] = {
        "max_abs": factor.get("reconstruction_max_abs"),
        "relative_frobenius": factor.get("reconstruction_relative_frobenius"),
        "source": "SEPARATE_NATIVE_WRITER_DIAGNOSTIC_RECONSTRUCTION",
    }
    result["physical_FP32_weight_addition"] = {
        "actual_delta_norm": factor.get("actual_delta_norm"),
        "actual_minus_solve_delta_norm": factor.get("actual_minus_solve_delta_norm"),
        "diagnostic_delta_source": "NATIVE_SOLVE_DELTA_BEFORE_ROUNDED_WEIGHT_ADDITION",
    }
    return result


@torch.no_grad()
def summarize(rt, factors, stamp, current_banks, entry_M,
              cache: MutableMapping[str, Any]) -> dict[str, Any]:
    """JSON-ready diagnostics, cached only by exact input SHA and route identity.

    Cache contains scalars/receipts, never edited weights or device tensors.
    P metrics are once/layer identity; tr(PMP.T) is once/entry/layer identity.
    Delta-history and current/stamp action penalties are recomputed per branch.
    Full banks remain all 512, weight1 including superseded requests.
    """
    result = {}
    for layer, factor in sorted(factors.items()):
        start_total = time.monotonic()
        P = rt.P[layer-4]
        M = entry_M[layer-4]
        delta = factor["delta"]
        ks, kc = stamp[layer], current_banks[layer]
        for name, value in (("P", P), ("entry M", M), ("delta", delta),
                            ("stamp", ks), ("current", kc)):
            _matrix(value, name)
        d = P.shape[0]
        if (P.shape != (d, d) or M.shape != P.shape or delta.shape[1] != d
                or ks.shape != (d, 512) or kc.shape != ks.shape):
            raise WriterDiagnosticError("writer history diagnostics require fixed full H512 operands")
        device = rt.weights[layer].device
        P_sha, M_sha, stamp_sha = tensor_sha(P), tensor_sha(M), tensor_sha(ks)
        key_prefix = (ROUTE, str(device), int(layer))
        key_P = repr((*key_prefix, "P", P_sha))
        key_trace = repr((*key_prefix, "PMP", P_sha, M_sha))
        key_stamp = repr((*key_prefix, "Pstamp", P_sha, stamp_sha))
        hit_P, hit_trace, hit_stamp = (key in cache for key in (key_P, key_trace, key_stamp))
        _sync(device)
        gpu_start = time.monotonic()
        # Peak FP64 resident payload <= 2*d*d + o*d + 2*d*512, plus row scratch.
        P64 = P.to(device=device, dtype=torch.float64)
        M64 = M.to(device=device, dtype=torch.float64)
        delta64 = delta.to(device=device, dtype=torch.float64)
        stamp64 = ks.to(device=device, dtype=torch.float64)
        current64 = kc.to(device=device, dtype=torch.float64)
        if not hit_P:
            cache[key_P] = dict(value=projector_diagnostics(P64), P_sha256=P_sha, route=ROUTE)
        if not hit_trace:
            cache[key_trace] = dict(value=quadratic_trace(P64, M64),
                                    P_sha256=P_sha, M_sha256=M_sha, route=ROUTE)
        if not hit_stamp:
            cache[key_stamp] = dict(value=_number(_action_energy(P64, stamp64).sum()),
                                    P_sha256=P_sha, stamp_sha256=stamp_sha, route=ROUTE)
        projected_current = _number(_action_energy(P64, current64).sum())
        native_penalty = quadratic_trace(delta64, M64)
        stamped = _action_energy(delta64, stamp64)
        current = _action_energy(delta64, current64)
        projected_trace = cache[key_trace]["value"]
        projected_stamp = cache[key_stamp]["value"]
        stamp_total, current_total = _number(stamped.sum()), _number(current.sum())
        bank = dict(
            schema="alpha-history-coverage-v1", bank_n=512, bank_weight=1,
            superseded_excluded_from_statistics=False,
            stamp_penalty=stamp_total, current_penalty=current_total,
            penalty_signed_current_minus_stamp_per_request=(current-stamped).cpu().tolist(),
            stamp_penalty_per_request=stamped.cpu().tolist(),
            current_penalty_per_request=current.cpu().tolist(),
            projected_stamp_trace=projected_stamp, projected_current_trace=projected_current,
            projected_history_trace=projected_trace,
            native_write_history_penalty=native_penalty,
            projected_trace_coverage=_ratio(projected_stamp, projected_trace),
            native_write_penalty_coverage=_ratio(stamp_total, native_penalty),
            total_identity="EXACT_ENTRY_M_P_SHA_ROUTE_CACHE",
            projector_policy="stored P; no orthogonalization",
            native_operands_modified=False)
        _sync(device)
        gpu_seconds = time.monotonic()-gpu_start
        del P64, M64, delta64, stamp64, current64, stamped, current
        small_start = time.monotonic()
        modes = _small_modes(factor)
        modes["history_coverage"] = bank
        modes["stored_P_numerical"] = cache[key_P]["value"]
        modes["diagnostic_execution"] = dict(
            device=str(device), route=ROUTE, block_rows=BLOCK_ROWS,
            P_sha256=P_sha, entry_M_sha256=M_sha, timestamp_sha256=stamp_sha,
            current_bank_sha256=tensor_sha(kc), delta_sha256=tensor_sha(delta),
            cache_hits={"P_numerical": hit_P, "projected_history_trace": hit_trace,
                        "projected_stamp_trace": hit_stamp},
            large_matrix_seconds=gpu_seconds,
            cpu_small_mode_and_response_seconds=time.monotonic()-small_start,
            total_seconds=time.monotonic()-start_total,
            timer_note="total includes nested large/small clocks; do not sum nested timers",
            fp64_resident_payload_upper_bytes=8*(2*d*d + delta.numel()+ks.numel()+kc.numel()),
            row_scratch_upper_bytes=8*(2*BLOCK_ROWS*d+BLOCK_ROWS*512),
            CPU_d_cubed_matmul=False, GPU_actual=device.type == "cuda")
        result[str(layer)] = modes
    return result
