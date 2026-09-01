"""FP64 residual-debt and physical weight-action reductions."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import torch

from project.run_scripts.ode_bf.functional import tensor_sha256

from .contracts import IDEAL_Q, LAYERS, ObservationBoundary, ObservationLock


def _case_hash(request_sha256: str) -> str:
    return hashlib.sha256(f"case|{request_sha256}".encode()).hexdigest()


def _finite_scalar(value: torch.Tensor, label: str) -> float:
    scalar = float(value.item())
    if not math.isfinite(scalar):
        raise ObservationBoundary(f"nonfinite scalar: {label}")
    return scalar


def residual_debt_metrics(
    *,
    z_rows: Sequence[torch.Tensor],
    pre_layer_rows: Sequence[torch.Tensor],
    terminal_rows: torch.Tensor,
    request_sha256: Sequence[str],
) -> dict[str, Any]:
    """Compute request-level q/rho/tau/debt and exact recurrence receipts."""

    lock = ObservationLock()
    if len(pre_layer_rows) != len(LAYERS) or len(z_rows) != len(request_sha256):
        raise ObservationBoundary("observer row inventory differs")
    z = torch.stack([row.detach().to("cpu", torch.float64).reshape(-1) for row in z_rows])
    pre = [row.detach().to("cpu", torch.float64) for row in pre_layer_rows]
    terminal = terminal_rows.detach().to("cpu", torch.float64)
    if any(row.ndim != 2 or row.shape != z.shape for row in pre) or terminal.shape != z.shape:
        raise ObservationBoundary("observer activation shape differs")
    if not all(torch.isfinite(row).all() for row in [z, terminal, *pre]):
        raise ObservationBoundary("observer activation contains nonfinite values")

    records: list[dict[str, Any]] = []
    maximum_closure = 0.0
    for index, request_hash in enumerate(request_sha256):
        residuals = [z[index] - row[index] for row in pre]
        post_residuals = [*residuals[1:], z[index] - terminal[index]]
        r1_norm = torch.linalg.vector_norm(residuals[0])
        if _finite_scalar(r1_norm, "R1 norm") == 0.0:
            raise ObservationBoundary("ZERO_INITIAL_RESIDUAL")
        q = [_finite_scalar(torch.linalg.vector_norm(value) / r1_norm, "q") for value in residuals]
        q_final = _finite_scalar(torch.linalg.vector_norm(post_residuals[-1]) / r1_norm, "q_final")
        layer_rows = []
        errors = []
        for layer_index, (layer, residual, post) in enumerate(
            zip(LAYERS, residuals, post_residuals, strict=True)
        ):
            divisor = len(LAYERS) - layer_index
            allocation = residual / divisor
            allocation_norm = torch.linalg.vector_norm(allocation)
            allocation_norm_value = _finite_scalar(allocation_norm, "allocation norm")
            if allocation_norm_value == 0.0:
                raise ObservationBoundary("ZERO_LAYER_ALLOCATION")
            realized = residual - post
            rho = _finite_scalar(
                torch.dot(realized, allocation) / torch.dot(allocation, allocation),
                "rho",
            )
            orthogonal = realized - rho * allocation
            tau = _finite_scalar(
                torch.linalg.vector_norm(orthogonal) / allocation_norm,
                "tau",
            )
            gap = allocation - realized
            errors.append(gap)
            layer_rows.append({
                "layer": layer,
                "q_pre": q[layer_index],
                "q_post": _finite_scalar(torch.linalg.vector_norm(post) / r1_norm, "q_post"),
                "rho": rho,
                "under_realization_coefficient": 1.0 - rho,
                "tau": tau,
                "allocation_norm_over_R1": _finite_scalar(allocation_norm / r1_norm, "allocation/R1"),
                "realized_reduction_norm_over_R1": _finite_scalar(
                    torch.linalg.vector_norm(realized) / r1_norm, "realized/R1"
                ),
                "gap_norm_over_R1": _finite_scalar(torch.linalg.vector_norm(gap) / r1_norm, "gap/R1"),
                "pre_activation_sha256": tensor_sha256(pre[layer_index][index].to(torch.float32)),
                "post_activation_sha256": tensor_sha256(
                    (pre[layer_index + 1][index] if layer_index < 4 else terminal[index]).to(torch.float32)
                ),
                "next_pre_link_sha256": (
                    tensor_sha256(pre[layer_index + 1][index].to(torch.float32))
                    if layer_index < 4 else None
                ),
                "next_pre_link_relative_difference": 0.0 if layer_index < 4 else None,
            })
        debt = residuals[4] - residuals[0] / 5.0
        d_parallel = _finite_scalar(
            torch.dot(debt, residuals[0]) / torch.dot(residuals[0], residuals[0]),
            "d_parallel",
        )
        d_perp = _finite_scalar(
            torch.linalg.vector_norm(debt - d_parallel * residuals[0]) / r1_norm,
            "d_perp",
        )
        recurrence = residuals[0] / 5.0 + errors[0] / 4.0 + errors[1] / 3.0 + errors[2] / 2.0 + errors[3]
        recurrence_error = _finite_scalar(
            torch.linalg.vector_norm(residuals[4] - recurrence) / r1_norm,
            "recurrence closure",
        )
        maximum_closure = max(maximum_closure, recurrence_error)
        records.append({
            "request_sha256": request_hash,
            "case_identity_sha256": _case_hash(request_hash),
            "q": [*q, q_final],
            "ideal_q": list(IDEAL_Q),
            "pre_L8_q": q[4],
            "final_q": q_final,
            "q_L8_gt_0_2": q[4] > 0.2,
            "d_parallel": d_parallel,
            "d_perp": d_perp,
            "recurrence_closure_relative_error": recurrence_error,
            "layers": layer_rows,
        })
    if maximum_closure >= lock.recurrence_relative_tolerance:
        raise ObservationBoundary(
            f"RECURRENCE_INSTRUMENTATION_FAILURE:{maximum_closure}"
        )
    return {
        "schema": "odeedit.s06.official-layer-realization-debt.request-metrics.v1",
        "request_count": len(records),
        "scalar_reduction_dtype": "float64",
        "recurrence_relative_tolerance": lock.recurrence_relative_tolerance,
        "maximum_recurrence_closure_relative_error": maximum_closure,
        "zero_initial_residual_count": 0,
        "zero_allocation_count": 0,
        "nonfinite_count": 0,
        "records": records,
    }


def weight_action_energy(
    touched: Mapping[str, torch.nn.Parameter],
    originals: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if set(touched) != set(originals) or len(touched) != len(LAYERS):
        raise ObservationBoundary("weight action inventory differs")
    rows = []
    total = 0.0
    for layer, (name, parameter) in zip(LAYERS, touched.items(), strict=True):
        original = originals[name].to(parameter.device, parameter.dtype)
        flat = (parameter.detach() - original).reshape(-1)
        energy = 0.0
        for begin in range(0, flat.numel(), 4 * 1024 * 1024):
            chunk = flat[begin : begin + 4 * 1024 * 1024].to(torch.float64)
            energy += _finite_scalar(torch.dot(chunk, chunk), "weight energy")
        rows.append({
            "layer": layer,
            "weight_name": name,
            "frobenius_energy": energy,
            "edited_weight_sha256": tensor_sha256(parameter),
            "entry_weight_sha256": tensor_sha256(originals[name]),
        })
        total += energy
    if not math.isfinite(total) or total <= 0.0:
        raise ObservationBoundary("nonpositive total physical weight action")
    for row in rows:
        row["energy_share"] = row["frobenius_energy"] / total
    return {
        "definition": "frobenius_energy=||Delta_W_l||_F^2",
        "scalar_reduction_dtype": "float64",
        "total_frobenius_energy": total,
        "layers": rows,
        "share_sum": sum(row["energy_share"] for row in rows),
    }
