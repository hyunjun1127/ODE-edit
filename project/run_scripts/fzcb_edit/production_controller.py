"""Uncertainty-aware, scale-neutral FzCB sketch controller for atomic B10."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from typing import Any

import torch

from .contracts import NumericalLock, ScientificBoundary
from .controller_validity import ArmExecutionFailure
from .linear import equality_null_projection, solve_minimum_action, suffix_value
from .production_contracts import ProductionNumericalLock
from .production_sensitivity import (
    NumericalInconclusive,
    robust_scalar_rectification,
    uncertainty_aware_sweep,
)
from .tolerances import ResolvedTolerances, UnitTolerancePolicy
from .transaction import WeightSnapshot
from .verifier import verify_barrier_state


def _seed(namespace: str, attempt: int) -> int:
    body = f"{ProductionNumericalLock().authority}|{namespace}|axis={attempt}"
    return int(hashlib.sha256(body.encode()).hexdigest()[:16], 16) % (2**31)


def _solve(operator: Any, rhs: torch.Tensor, tolerance: float) -> Any:
    lock = NumericalLock()
    return solve_minimum_action(
        operator,
        rhs.detach().float(),
        tolerance=tolerance,
        maximum_iterations=lock.cg_max_iterations,
        refinement_count=lock.cg_internal_refinement_count,
        preconditioner=operator.output_preconditioner,
    )


def _weight_perturbation_receipt(model: Any, entry: WeightSnapshot, requested: float) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    norm2 = torch.zeros((), dtype=torch.float64, device=next(model.parameters()).device)
    changed = 0
    total = 0
    layer_norms = []
    for name in entry.names:
        delta = parameters[name].detach().to(torch.float64) - entry.tensors[name].to(parameters[name].device, torch.float64)
        local2 = torch.dot(delta.reshape(-1), delta.reshape(-1))
        norm2 += local2
        changed += int(torch.count_nonzero(delta).item())
        total += delta.numel()
        layer_norms.append({"name": name, "norm": math.sqrt(float(local2.item()))})
    return {
        "requested_coefficient_perturbation": requested,
        "realized_weight_perturbation_norm": math.sqrt(float(norm2.item())),
        "changed_element_numerator": changed,
        "changed_element_denominator": total,
        "changed_element_fraction": changed / total if total else 0.0,
        "layer_weight_perturbation_norms": layer_norms,
        "scalar_reduction_dtype": "float64",
    }


def _normalized_suffix_at(
    *,
    model: Any,
    factory: Any,
    entry: WeightSnapshot,
    entry_operator: Any,
    direction: torch.Tensor,
    eta: float,
    solver_tolerance_scale: float,
    target: torch.Tensor,
    remaining_time: float,
    initial_budget: float,
    tolerances: ResolvedTolerances,
) -> tuple[float, dict[str, Any]]:
    requested = float(eta * math.sqrt(initial_budget))
    try:
        entry.restore(model)
        entry.set_displacement(
            model,
            entry_operator.coefficient_to_tangents(direction),
            requested,
        )
        perturbation = _weight_perturbation_receipt(model, entry, requested)
        perturbed, geometry = factory.build()
        residual = target - perturbed.phi()
        tolerance = tolerances.tau_range * float(solver_tolerance_scale)
        solution = _solve(perturbed, residual, tolerance)
        value = suffix_value(solution, remaining_time)
        return value, {
            "eta": eta,
            "solver_tolerance_scale": solver_tolerance_scale,
            "solver_tolerance": tolerance,
            "raw_suffix_value": value,
            "normalized_suffix_value": value / initial_budget,
            "geometry_identity": geometry["identity"],
            "range_residual": solution.range_residual,
            "kkt_relative_residual": solution.receipt.relative_residual,
            "cg_iterations": solution.receipt.iterations,
            "cg_converged": solution.receipt.converged,
            "rank_spectrum": [
                {
                    "weight_name": row["weight_name"],
                    "reduced_rank": row["reduced_rank"],
                    "gram_eigen_min": row["gram_eigen_min"],
                    "gram_eigen_max": row["gram_eigen_max"],
                }
                for row in geometry["layers"]
            ],
            **perturbation,
        }
    finally:
        entry.restore(model)


def _incremental_null_axes(
    *,
    operator: Any,
    axes: list[torch.Tensor],
    receipts: list[dict[str, Any]],
    seeds: list[int],
    attempts: int,
    target_count: int,
    namespace: str,
    tolerances: ResolvedTolerances,
) -> int:
    lock = NumericalLock()
    maximum = max(target_count * 8, attempts + (target_count - len(axes)) * 8)
    while len(axes) < target_count and attempts < maximum:
        seed = _seed(namespace, attempts)
        generator = torch.Generator().manual_seed(seed)
        candidate = torch.randn(operator.coefficient_dimension, generator=generator, dtype=torch.float32)
        candidate = candidate.to(operator.primals[0].device)
        projected, receipt = equality_null_projection(
            operator,
            candidate,
            tolerance=tolerances.tau_null,
            maximum_iterations=lock.cg_max_iterations,
            refinement_count=lock.cg_internal_refinement_count,
            preconditioner=operator.output_preconditioner,
        )
        for previous in axes:
            projected = projected - torch.dot(previous, projected) * previous
        norm = float(torch.linalg.vector_norm(projected).item())
        if norm > torch.finfo(torch.float32).eps:
            projected = (projected / norm).detach()
            absolute = float(torch.linalg.vector_norm(operator.apply(projected)).item())
            scale = float(receipt["operator_scale"]) + torch.finfo(torch.float32).eps
            relative = absolute / scale
            if relative <= tolerances.tau_null:
                axes.append(projected)
                seeds.append(seed)
                receipts.append({
                    "axis": len(axes) - 1,
                    "seed": seed,
                    "post_orthogonal_norm": norm,
                    "null_residual_absolute": absolute,
                    "null_residual_relative": relative,
                    **receipt,
                })
        attempts += 1
    if len(axes) != target_count:
        raise NumericalInconclusive(
            f"lazy null sketch produced {len(axes)}/{target_count} axes",
            {"attempts": attempts, "requested_width": target_count},
        )
    return attempts


def barrier_control_production(
    *,
    model: Any,
    factory: Any,
    operator: Any,
    entry: WeightSnapshot,
    equality: Any,
    target: torch.Tensor,
    s: float,
    proposed_step: float,
    spent_action: float,
    initial_budget: float,
    namespace: str,
    policy: UnitTolerancePolicy,
    tolerances: ResolvedTolerances,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return an equality-preserving sketch correction or typed uncertainty."""

    remaining = 1.0 - s
    if remaining <= 0.0:
        raise ScientificBoundary("production barrier requested at terminal state")
    residual = target - operator.phi()
    suffix_solution = _solve(operator, residual, tolerances.tau_range)
    raw_value = suffix_value(suffix_solution, remaining)
    strict_state = verify_barrier_state(
        initial_budget=initial_budget,
        spent_action=spent_action,
        predicted_suffix=raw_value,
        tolerances=tolerances,
        stage=f"production-current-s={s:.8f}",
    )
    a0 = float(initial_budget)
    sqrt_a0 = math.sqrt(a0)
    tau_normalized = tolerances.tau_cbf / a0
    equality_norm = float(torch.linalg.vector_norm(equality.coefficient.to(torch.float64)).item())
    if equality_norm <= 0.0:
        raise ScientificBoundary("equality coefficient has zero norm")
    equality_unit = (equality.coefficient / equality_norm).detach()

    def sweep(direction: torch.Tensor) -> Any:
        return uncertainty_aware_sweep(
            lambda eta, solver_scale: _normalized_suffix_at(
                model=model,
                factory=factory,
                entry=entry,
                entry_operator=operator,
                direction=direction,
                eta=eta,
                solver_tolerance_scale=solver_scale,
                target=target,
                remaining_time=remaining,
                initial_budget=a0,
                tolerances=tolerances,
            ),
            initial_budget=a0,
            base_epsilon=policy.fd_base_epsilon,
            multipliers=policy.fd_multipliers,
            repeats=policy.fd_repeats,
        )

    equality_fd = sweep(equality_unit)
    equality_scale = equality_norm / sqrt_a0
    g_eq_normalized = equality_scale * equality_fd.estimate
    u_eq_normalized = equality_scale * equality_fd.uncertainty
    equality_action_normalized = float(equality.action_squared) / a0
    suffix_rate_normalized = raw_value / (remaining * a0)
    psi_estimate = 0.5 * equality_action_normalized + g_eq_normalized + suffix_rate_normalized
    base: dict[str, Any] = {
        "stage": "production-barrier-current-state",
        "authority": "FZCB_SKETCH_K_PROTOTYPE",
        "full_suffix_gradient_available": False,
        "full_suffix_gradient_unavailable_reason": (
            "production envelope-HVP sensitivity is not implemented; lazy FD sketch is explicit"
        ),
        "full_suffix_gradient_production_call_count": 0,
        "full_projected_sensitivity_production_call_count": 0,
        "s": s,
        "proposed_step": proposed_step,
        "remaining_time": remaining,
        "A0": a0,
        "E": float(spent_action),
        "predicted_suffix": raw_value,
        "h_cc": strict_state["h_cc"],
        "normalized": {
            "E_over_A0": float(spent_action) / a0,
            "V_over_A0": raw_value / a0,
            "h_cc_over_A0": strict_state["h_cc"] / a0,
            "psi_over_A0": psi_estimate,
            "tau_cbf_over_A0": tau_normalized,
        },
        "equality": {
            "coefficient_norm": equality_norm,
            "unit_direction_norm": float(torch.linalg.vector_norm(equality_unit).item()),
            "action_squared_raw": float(equality.action_squared),
            "action_squared_over_A0": equality_action_normalized,
            "directional_derivative_normalized": equality_fd.payload(),
            "rescaling_identity": "g_eq=(||c_eq||/sqrt(A0))*D_eta(V/A0)[u_eq]",
            "g_eq_normalized": g_eq_normalized,
            "g_eq_uncertainty_normalized": u_eq_normalized,
        },
        "psi_estimate_normalized": psi_estimate,
        "psi_uncertainty_normalized": u_eq_normalized,
        "scalar_reduction_dtype": "float64",
        "coefficient_dimension": operator.coefficient_dimension,
        "output_dimension": operator.output_dimension,
        "null_dimension": max(0, operator.coefficient_dimension - operator.output_dimension),
        "kappa": 0.0,
        "tolerances": tolerances.payload(),
    }
    # Activity must be invariant throughout the equality-derivative interval.
    if psi_estimate - u_eq_normalized <= tau_normalized < psi_estimate + u_eq_normalized:
        raise ArmExecutionFailure("NUMERICAL_INCONCLUSIVE barrier activity", {
            **base,
            "status": "NUMERICAL_INCONCLUSIVE",
            "uncertainty_boundary": "BARRIER_ACTIVITY_INTERVAL_STRADDLE",
        })
    if psi_estimate + u_eq_normalized <= tau_normalized:
        base.update({
            "sensitivity_method": "NOT_REQUIRED_BARRIER_INACTIVE_UNCERTAINTY_STRICT",
            "selected_k": 0,
            "raw_fd_axis_count": 1,
            "rectification": {
                "barrier_active": False,
                "psi_interval": [psi_estimate - u_eq_normalized, psi_estimate + u_eq_normalized],
                "tau_normalized": tau_normalized,
            },
            "correction_h_norm": 0.0,
            "equality_null_residual": 0.0,
        })
        return equality.coefficient.detach(), base

    axes: list[torch.Tensor] = []
    axis_receipts: list[dict[str, Any]] = []
    seeds: list[int] = []
    derivatives = []
    attempts = 0
    ladder_rows = []
    correction_coordinates: torch.Tensor | None = None
    rectification: dict[str, Any] | None = None
    for width in ProductionNumericalLock().sketch_ladder:
        attempts = _incremental_null_axes(
            operator=operator,
            axes=axes,
            receipts=axis_receipts,
            seeds=seeds,
            attempts=attempts,
            target_count=width,
            namespace=namespace,
            tolerances=tolerances,
        )
        while len(derivatives) < width:
            derivative = sweep(axes[len(derivatives)])
            derivatives.append(derivative)
        gradient = torch.tensor(
            [row.estimate for row in derivatives],
            dtype=torch.float32,
            device=equality.coefficient.device,
        )
        uncertainties = torch.tensor(
            [row.uncertainty for row in derivatives],
            dtype=torch.float32,
            device=equality.coefficient.device,
        )
        try:
            correction_coordinates, rectification = robust_scalar_rectification(
                psi_estimate=psi_estimate,
                psi_uncertainty=u_eq_normalized,
                gradient=gradient,
                gradient_uncertainty=uncertainties,
                tau_normalized=tau_normalized,
            )
        except NumericalInconclusive as exc:
            ladder_rows.append({
                "k": width,
                "status": "NO_UNCERTAINTY_STRICT_CANDIDATE_AT_THIS_WIDTH",
                "reason": str(exc),
                "receipt": exc.receipt,
                "seed_root": hashlib.sha256(str(seeds[:width]).encode()).hexdigest(),
            })
            continue
        ladder_rows.append({
            "k": width,
            "status": "UNCERTAINTY_STRICT_CANDIDATE_FOUND",
            "seed_root": hashlib.sha256(str(seeds[:width]).encode()).hexdigest(),
        })
        break
    if correction_coordinates is None or rectification is None:
        raise ArmExecutionFailure("NUMERICAL_INCONCLUSIVE after lazy K=128 ladder", {
            **base,
            "status": "NUMERICAL_INCONCLUSIVE",
            "sketch_ladder": ladder_rows,
            "axis_receipts": axis_receipts,
            "axis_fd_sweeps": [row.payload() for row in derivatives],
        })

    selected_k = len(correction_coordinates)
    correction = torch.zeros_like(equality.coefficient)
    for scalar, axis in zip(correction_coordinates, axes[:selected_k], strict=True):
        correction = correction + sqrt_a0 * scalar * axis
    corrected = equality.coefficient.detach() + correction
    null_absolute = float(torch.linalg.vector_norm(operator.apply(correction)).item())
    equality_image = float(torch.linalg.vector_norm(operator.apply(equality.coefficient)).item())
    null_relative = null_absolute / max(equality_image, torch.finfo(torch.float32).eps)
    if null_relative > tolerances.tau_null:
        raise ScientificBoundary("production sketch correction left equality-null space")
    base.update({
        "sensitivity_method": "LAZY_UNCERTAINTY_AWARE_EQUALITY_NULL_FD_SKETCH",
        "selected_k": selected_k,
        "lazy_sketch_ladder": ladder_rows,
        "lazy_early_stop": selected_k < ProductionNumericalLock().sketch_ladder[-1],
        "axis_receipts": axis_receipts[:selected_k],
        "axis_fd_sweeps": [row.payload() for row in derivatives[:selected_k]],
        "raw_fd_axis_count": 1 + selected_k,
        "rectification": rectification,
        "equality_null_absolute": null_absolute,
        "equality_null_relative": null_relative,
        "correction_h_norm": float(torch.linalg.vector_norm(correction.to(torch.float64)).item()),
        "full_projection_status": "SKETCH_FEASIBLE_NOT_FULL_CERTIFIED",
        # Private tensors are consumed and removed by the production runtime
        # before any receipt is serialized.
        "_diagnostic_coefficients": {
            "EQUALITY_ONLY": equality.coefficient.detach().clone(),
            "FZCB_SELECTED": corrected.detach().clone(),
            "MATCHED_ACTION_NULL_OPPOSITE": (equality.coefficient.detach() - correction).clone(),
        },
    })
    return corrected.detach(), base
