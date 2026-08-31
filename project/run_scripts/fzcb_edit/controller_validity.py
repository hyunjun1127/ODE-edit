"""TECH-R1 strict controller-validity execution for the five-arm B1 audit."""

from __future__ import annotations

import hashlib
import math
import traceback
from dataclasses import asdict
from typing import Any, Callable

import torch

from .comparators import StaticCandidate, select_strong_static, true_frozen_schedule
from .contracts import (
    Arm,
    NumericalLock,
    NumericalSensitivityFailure,
    ScientificBoundary,
    SubspaceInconclusive,
)
from .geometry import FullModelControlOperator, MEMITGeometryFactory
from .hashing import canonical_hash, tensor_sha256
from .linear import equality_null_projection, scalar_rectification, solve_minimum_action, suffix_value
from .rollout import SuffixCandidate, concordance, run_actual_remaining_rollout
from .sensitivity import finite_difference_sweep, sketch_ladder_receipt
from .tolerances import ResolvedTolerances, UnitTolerancePolicy
from .transaction import AtomicEditTransaction, WeightSnapshot
from .verifier import verify_barrier_state, verify_terminal


class ArmExecutionFailure(ScientificBoundary):
    """A typed arm failure with a complete journal-ready evidence payload."""

    def __init__(self, message: str, receipt: dict[str, Any]) -> None:
        super().__init__(message)
        self.receipt = receipt


class CandidateFailure(ScientificBoundary):
    """A finite candidate may be retried from the accepted entry state."""


def _seed(namespace: str, axis: int) -> int:
    body = f"{NumericalLock().seed_namespace}|{namespace}|{axis}"
    return int(hashlib.sha256(body.encode()).hexdigest()[:16], 16) % (2**31)


def _solve(operator: FullModelControlOperator, rhs: torch.Tensor, tau_range: float) -> Any:
    lock = NumericalLock()
    return solve_minimum_action(
        operator,
        rhs.detach().float(),
        tolerance=tau_range,
        maximum_iterations=lock.cg_max_iterations,
        refinement_count=lock.cg_internal_refinement_count,
        preconditioner=operator.output_preconditioner,
    )


def _same_current_basis(model: Any, source: FullModelControlOperator) -> FullModelControlOperator:
    return FullModelControlOperator(
        model,
        source.batch,
        source.z_layer,
        source.weight_names,
        source.right_factors,
        source.direct_gram_floor,
        linearization_state="CURRENT_MODEL_WITH_ENTRY_T_H",
    )


def _orthonormal_null_axes(
    operator: FullModelControlOperator,
    *,
    count: int,
    namespace: str,
    tolerances: ResolvedTolerances,
) -> tuple[list[torch.Tensor], list[dict[str, Any]], list[int]]:
    lock = NumericalLock()
    axes: list[torch.Tensor] = []
    receipts: list[dict[str, Any]] = []
    seeds: list[int] = []
    attempts = 0
    while len(axes) < count and attempts < count * 6:
        seed_value = _seed(namespace, attempts)
        generator = torch.Generator().manual_seed(seed_value)
        candidate = torch.randn(
            operator.coefficient_dimension, generator=generator, dtype=torch.float32,
        ).to(operator.primals[0].device)
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
        if norm > tolerances.tau_grad:
            projected = (projected / norm).detach()
            null_residual_absolute = float(torch.linalg.vector_norm(operator.apply(projected)).item())
            null_residual = null_residual_absolute / (
                float(receipt["operator_scale"]) + torch.finfo(torch.float32).eps
            )
            if null_residual <= tolerances.tau_null:
                axes.append(projected)
                seeds.append(seed_value)
                receipts.append({
                    "axis": len(axes) - 1,
                    "seed": seed_value,
                    "post_orthogonal_norm": norm,
                    "post_orthogonal_null_residual_absolute": null_residual_absolute,
                    "post_orthogonal_null_residual": null_residual,
                    **receipt,
                })
        attempts += 1
    if len(axes) != count:
        raise ScientificBoundary(f"null sketch authority width={len(axes)}/{count}")
    return axes, receipts, seeds


def _suffix_at_perturbation(
    *,
    model: Any,
    factory: MEMITGeometryFactory,
    entry: WeightSnapshot,
    entry_operator: FullModelControlOperator,
    direction: torch.Tensor,
    scale: float,
    target: torch.Tensor,
    remaining_time: float,
    tolerances: ResolvedTolerances,
) -> tuple[float, dict[str, Any]]:
    try:
        entry.restore(model)
        entry.set_displacement(model, entry_operator.coefficient_to_tangents(direction), scale)
        perturbed, geometry = factory.build()
        residual = target - perturbed.phi()
        solution = _solve(perturbed, residual, tolerances.tau_range)
        value = suffix_value(solution, remaining_time)
        return value, {
            "scale": scale,
            "geometry_identity": geometry["identity"],
            "range_residual": solution.range_residual,
            "suffix_value": value,
            "operator_counts": perturbed.receipt(),
        }
    finally:
        entry.restore(model)


def _sweep_direction(
    *,
    model: Any,
    factory: MEMITGeometryFactory,
    entry: WeightSnapshot,
    operator: FullModelControlOperator,
    direction: torch.Tensor,
    target: torch.Tensor,
    remaining_time: float,
    policy: UnitTolerancePolicy,
    tolerances: ResolvedTolerances,
) -> Any:
    return finite_difference_sweep(
        lambda scale: _suffix_at_perturbation(
            model=model, factory=factory, entry=entry, entry_operator=operator,
            direction=direction, scale=scale, target=target,
            remaining_time=remaining_time, tolerances=tolerances,
        ),
        base_epsilon=policy.fd_base_epsilon,
        multipliers=policy.fd_multipliers,
        repeats=policy.fd_repeats,
        tau_grad=tolerances.tau_grad,
    )


def barrier_control_strict(
    *,
    model: Any,
    factory: MEMITGeometryFactory,
    operator: FullModelControlOperator,
    entry: WeightSnapshot,
    equality: Any,
    target: torch.Tensor,
    s: float,
    spent_action: float,
    initial_budget: float,
    namespace: str,
    policy: UnitTolerancePolicy,
    tolerances: ResolvedTolerances,
) -> tuple[torch.Tensor, dict[str, Any]]:
    remaining = 1.0 - s
    if remaining <= 0.0:
        raise ScientificBoundary("barrier requested at terminal state")
    residual = target - operator.phi()
    suffix_solution = _solve(operator, residual, tolerances.tau_range)
    value = suffix_value(suffix_solution, remaining)
    strict_state = verify_barrier_state(
        initial_budget=initial_budget,
        spent_action=spent_action,
        predicted_suffix=value,
        tolerances=tolerances,
        stage=f"current-s={s:.8f}",
    )
    equality_sweep = _sweep_direction(
        model=model, factory=factory, entry=entry, operator=operator,
        direction=equality.coefficient, target=target, remaining_time=remaining,
        policy=policy, tolerances=tolerances,
    )
    g_eq = equality_sweep.selected_derivative
    partial_s = value / remaining
    psi0 = 0.5 * equality.action_squared + g_eq + partial_s
    base_receipt: dict[str, Any] = {
        "remaining_time": remaining,
        "predicted_suffix": value,
        "spent_action": spent_action,
        "A0": initial_budget,
        "h_cc": strict_state["h_cc"],
        "partial_s_suffix": partial_s,
        "g_eq": g_eq,
        "equality_direction_fd": equality_sweep.payload(),
        "tolerances": tolerances.payload(),
        "kappa": 0.0,
        "coefficient_dimension": operator.coefficient_dimension,
        "output_dimension": operator.output_dimension,
        "null_dimension": max(0, operator.coefficient_dimension - operator.output_dimension),
    }
    if psi0 <= tolerances.tau_cbf:
        zero = torch.zeros(1, device=equality.coefficient.device, dtype=torch.float32)
        _, rectification = scalar_rectification(
            psi0, zero, tau_cbf=tolerances.tau_cbf,
            authority="BARRIER_INACTIVE_NO_PROJECTION_REQUIRED",
        )
        base_receipt.update({
            "sensitivity_method": "NOT_REQUIRED_BARRIER_INACTIVE",
            "rectification": rectification,
            "full_projection_status": "NOT_REQUIRED_BARRIER_INACTIVE",
            "correction_h_norm": 0.0,
            "equality_null_residual": 0.0,
        })
        return equality.coefficient.detach(), base_receipt

    width = NumericalLock().sketch_ladder[-1]
    axes, axis_receipts, seeds = _orthonormal_null_axes(
        operator, count=width, namespace=namespace, tolerances=tolerances,
    )
    derivatives: list[float] = []
    sweeps: list[dict[str, Any]] = []
    for ordinal, axis in enumerate(axes):
        sweep = _sweep_direction(
            model=model, factory=factory, entry=entry, operator=operator,
            direction=axis, target=target, remaining_time=remaining,
            policy=policy, tolerances=tolerances,
        )
        derivatives.append(sweep.selected_derivative)
        sweeps.append({"axis": ordinal, "seed": seeds[ordinal], **sweep.payload()})
    sketch = sketch_ladder_receipt(
        derivatives,
        coefficient_dimension=operator.coefficient_dimension,
        output_dimension=operator.output_dimension,
        ladder=NumericalLock().sketch_ladder,
        seeds=seeds,
    )
    gradient = torch.tensor(derivatives, dtype=torch.float32, device=equality.coefficient.device)
    base_receipt.update({
        "sensitivity_method": NumericalLock().sensitivity_method,
        "axis_receipts": axis_receipts,
        "axis_fd_sweeps": sweeps,
        "sketch": sketch,
    })
    try:
        correction_coordinates, rectification = scalar_rectification(
            psi0,
            gradient,
            tau_cbf=tolerances.tau_cbf,
            authority="SKETCH_LADDER_ONLY",
        )
    except (SubspaceInconclusive, ScientificBoundary) as exc:
        psi_min = psi0 - 0.5 * float(torch.dot(gradient, gradient).item())
        base_receipt.update({
            "rectification": {
                "psi0": psi0,
                "psi_min": psi_min,
                "raw_sketch_g_free_norm_squared": float(torch.dot(gradient, gradient).item()),
                "tau_cbf": tolerances.tau_cbf,
                "authority": "SKETCH_LADDER_ONLY",
            },
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        })
        raise ArmExecutionFailure(str(exc), base_receipt) from exc
    corrected = equality.coefficient.detach().clone()
    for scalar, axis in zip(correction_coordinates, axes, strict=True):
        corrected = corrected + scalar * axis
    null_absolute = float(torch.linalg.vector_norm(operator.apply(corrected - equality.coefficient)).item())
    denominator = float(torch.linalg.vector_norm(operator.apply(equality.coefficient)).item()) + torch.finfo(torch.float32).eps
    null_relative = null_absolute / denominator
    if null_relative > tolerances.tau_null:
        raise ArmExecutionFailure("barrier correction left equality-null space", {
            **base_receipt, "equality_null_absolute": null_absolute,
            "equality_null_relative": null_relative,
        })
    base_receipt.update({
        "rectification": rectification,
        "full_projection_status": "SKETCH_FEASIBLE_NOT_FULL_CERTIFIED",
        "equality_null_absolute": null_absolute,
        "equality_null_relative": null_relative,
        "correction_h_norm": float(torch.linalg.vector_norm(corrected - equality.coefficient).item()),
    })
    return corrected.detach(), base_receipt


def _corrected_candidate(
    *,
    model: Any,
    operator: FullModelControlOperator,
    entry: WeightSnapshot,
    coefficient: torch.Tensor,
    target_waypoint: torch.Tensor,
    delta_s: float,
    tolerances: ResolvedTolerances,
) -> tuple[torch.Tensor, dict[str, Any]]:
    value = coefficient.detach().clone()
    iterations = []
    for ordinal in range(NumericalLock().corrector_iterations):
        entry.set_displacement(model, operator.coefficient_to_tangents(value), delta_s)
        candidate_operator = _same_current_basis(model, operator)
        residual = target_waypoint - candidate_operator.phi()
        absolute = float(torch.linalg.vector_norm(residual).item())
        if absolute <= tolerances.tau_z:
            iterations.append({"iteration": ordinal + 1, "residual_before": absolute, "skipped_within_tau_z": True})
            continue
        try:
            correction = _solve(candidate_operator, residual / delta_s, tolerances.tau_range)
        except ScientificBoundary as exc:
            raise CandidateFailure(f"candidate corrector range failure: {exc}") from exc
        value = (value + correction.coefficient).detach()
        iterations.append({
            "iteration": ordinal + 1,
            "residual_before": absolute,
            "correction_action_norm": math.sqrt(correction.action_squared),
            "range_residual": correction.range_residual,
            "solver": asdict(correction.receipt),
        })
    entry.set_displacement(model, operator.coefficient_to_tangents(value), delta_s)
    final_operator = _same_current_basis(model, operator)
    closure = float(torch.linalg.vector_norm(final_operator.phi() - target_waypoint).item())
    if closure > tolerances.tau_z:
        raise CandidateFailure(f"nonlinear corrector closure={closure:.9g} tau_z={tolerances.tau_z:.9g}")
    action_squared = float(torch.dot(value, value).item())
    return value, {
        "iterations": iterations,
        "final_absolute_closure": closure,
        "corrected_coefficient_sha256": tensor_sha256(value),
        "corrected_action_squared": action_squared,
        "corrector_action_included": True,
        "operator_counts": final_operator.receipt(),
    }


def _initial_context(
    factory: MEMITGeometryFactory,
    target: torch.Tensor,
    policy: UnitTolerancePolicy,
) -> tuple[FullModelControlOperator, dict[str, Any], torch.Tensor, Any, float, ResolvedTolerances, dict[str, Any]]:
    operator, geometry = factory.build()
    phi0 = operator.phi()
    displacement = target - phi0
    first = _solve(operator, displacement, policy.tau_range)
    initial_budget = suffix_value(first, 1.0)
    tolerances = policy.resolve(initial_budget)
    second = _solve(operator, displacement, tolerances.tau_range)
    repeat_action_noise = abs(0.5 * first.action_squared - 0.5 * second.action_squared)
    if repeat_action_noise > tolerances.tau_budget:
        raise NumericalSensitivityFailure(
            f"initial repeated suffix solve noise={repeat_action_noise} tau_budget={tolerances.tau_budget}"
        )
    receipt = {
        "A0": initial_budget,
        "first": {"range_residual": first.range_residual, "action_squared": first.action_squared, "solver": asdict(first.receipt)},
        "repeat": {"range_residual": second.range_residual, "action_squared": second.action_squared, "solver": asdict(second.receipt)},
        "repeat_action_noise": repeat_action_noise,
        "tolerances": tolerances.payload(),
    }
    return operator, geometry, phi0, first, initial_budget, tolerances, receipt


def _actual_rollout(
    *,
    model: Any,
    factory: MEMITGeometryFactory,
    start: WeightSnapshot,
    phi0: torch.Tensor,
    displacement: torch.Tensor,
    start_progress: float,
    tolerances: ResolvedTolerances,
) -> tuple[float, tuple[float, ...], dict[str, Any]]:
    start.restore(model)

    def advance(s_start: float, s_end: float) -> tuple[float, tuple[float, ...], dict[str, Any]]:
        entry = WeightSnapshot.capture(model, start.names)
        operator, geometry = factory.build()
        waypoint = phi0 + s_end * displacement
        delta_s = s_end - s_start
        equality = _solve(operator, (waypoint - operator.phi()) / delta_s, tolerances.tau_range)
        corrected, corrector = _corrected_candidate(
            model=model, operator=operator, entry=entry, coefficient=equality.coefficient,
            target_waypoint=waypoint, delta_s=delta_s, tolerances=tolerances,
        )
        layer_action = operator.layer_action(corrected, delta_s)
        return sum(layer_action), layer_action, {
            "geometry_identity": geometry["identity"],
            "range_residual": equality.range_residual,
            "corrector": corrector,
        }

    try:
        return run_actual_remaining_rollout(
            start_progress=start_progress,
            progress_grid=NumericalLock().progress_grid,
            advance=advance,
        )
    finally:
        start.restore(model)


def run_true_frozen(
    *, model: Any, factory: MEMITGeometryFactory, w0: WeightSnapshot,
    target: torch.Tensor, policy: UnitTolerancePolicy,
) -> dict[str, Any]:
    arm = Arm.TRUE_FROZEN_C_SPLIT
    w0.restore(model)
    transaction = AtomicEditTransaction(model, w0)
    operator, geometry, phi0, equality, initial_budget, tolerances, budget_receipt = _initial_context(factory, target, policy)
    frozen = operator.frozen_clone()
    frozen_equality = _solve(frozen, target - phi0, tolerances.tau_range)
    identities = true_frozen_schedule(
        initial_budget=initial_budget,
        progress_grid=NumericalLock().progress_grid,
        tau_budget=tolerances.tau_budget,
    )
    rows = []
    try:
        for identity in identities[1:]:
            progress = float(identity["progress"])
            w0.set_displacement(model, frozen.coefficient_to_tangents(frozen_equality.coefficient), progress)
            actual, refreshed_geometry = factory.build()
            waypoint = phi0 + progress * (target - phi0)
            closure = float(torch.linalg.vector_norm(actual.phi() - waypoint).item())
            if closure > tolerances.tau_z:
                raise ScientificBoundary(f"TRUE_FROZEN_C_SPLIT actual closure={closure} tau_z={tolerances.tau_z}")
            rows.append({
                **identity,
                "actual_closure": closure,
                "frozen_operator": frozen.receipt(),
                "refreshed_observation_geometry": refreshed_geometry["identity"],
            })
        terminal = WeightSnapshot.capture(model, w0.names)
        w0.restore(model)
        commit = transaction.commit_terminal(terminal)
        return {
            "arm": arm.value,
            "status": "TERMINAL_VALID_STRICT",
            "A0": initial_budget,
            "budget_calibration": budget_receipt,
            "initial_geometry": geometry,
            "frozen_linearization_state": frozen.linearization_state,
            "frozen_identity": rows,
            "terminal": {
                "absolute_closure": rows[-1]["actual_closure"],
                "spent_action": initial_budget,
                "budget": initial_budget,
                "weight_root": terminal.root,
            },
            "atomic_commit": commit,
            "direct_z_recompute_count": 0,
            "controller_output_metric_access_count": 0,
        }
    except Exception:
        transaction.rollback()
        raise


def _run_refreshed_or_fzcb(
    *,
    arm: Arm,
    model: Any,
    factory: MEMITGeometryFactory,
    w0: WeightSnapshot,
    target: torch.Tensor,
    namespace: str,
    policy: UnitTolerancePolicy,
) -> dict[str, Any]:
    if arm not in {Arm.REFRESHED_EQUALITY_ONLY, Arm.FZCB}:
        raise ScientificBoundary("invalid refreshed controller arm")
    w0.restore(model)
    transaction = AtomicEditTransaction(model, w0)
    initial_operator, initial_geometry, phi0, _, initial_budget, tolerances, budget_receipt = _initial_context(factory, target, policy)
    displacement = target - phi0
    accepted_s = 0.0
    spent = 0.0
    waypoints: list[dict[str, Any]] = []
    accepted_count = 0
    rejected_count = 0
    backtrack_total = 0
    try:
        schedule = list(zip(NumericalLock().progress_grid[:-1], NumericalLock().progress_grid[1:], strict=True))
        while schedule:
            scheduled_start, scheduled_end = schedule.pop(0)
            if abs(scheduled_start - accepted_s) > 1e-12:
                raise ScientificBoundary("accepted progress/schedule identity mismatch")
            proposed = scheduled_end - scheduled_start
            attempt = 0
            while True:
                delta_s = proposed * NumericalLock().backtrack_factor**attempt
                next_s = accepted_s + delta_s
                entry = WeightSnapshot.capture(model, w0.names)
                operator, geometry = factory.build()
                phi_entry = operator.phi()
                waypoint = phi0 + next_s * displacement
                equality = _solve(operator, (waypoint - phi_entry) / delta_s, tolerances.tau_range)
                barrier_receipt = None
                coefficient = equality.coefficient
                if arm is Arm.FZCB:
                    coefficient, barrier_receipt = barrier_control_strict(
                        model=model, factory=factory, operator=operator, entry=entry,
                        equality=equality, target=target, s=accepted_s,
                        spent_action=spent, initial_budget=initial_budget,
                        namespace=f"{namespace}|s={accepted_s:.8f}", policy=policy,
                        tolerances=tolerances,
                    )
                try:
                    corrected, corrector = _corrected_candidate(
                        model=model, operator=operator, entry=entry, coefficient=coefficient,
                        target_waypoint=waypoint, delta_s=delta_s, tolerances=tolerances,
                    )
                    layer_action = operator.layer_action(corrected, delta_s)
                    step_action = sum(layer_action)
                    candidate = WeightSnapshot.capture(model, w0.names)
                    refreshed, refreshed_geometry = factory.build()
                    refreshed_phi = refreshed.phi()
                    if next_s < 1.0 - 1e-12:
                        suffix_solution = _solve(refreshed, target - refreshed_phi, tolerances.tau_range)
                        predicted_suffix = suffix_value(suffix_solution, 1.0 - next_s)
                    else:
                        predicted_suffix = 0.0
                    strict_receipt = verify_barrier_state(
                        initial_budget=initial_budget,
                        spent_action=spent + step_action,
                        predicted_suffix=predicted_suffix,
                        tolerances=tolerances,
                        stage=f"candidate-s={next_s:.8f}",
                    ) if arm is Arm.FZCB else None
                    next_viability = None
                    if arm is Arm.FZCB and next_s < 1.0 - 1e-12:
                        next_equality = _solve(refreshed, displacement, tolerances.tau_range)
                        try:
                            _, next_viability = barrier_control_strict(
                                model=model, factory=factory, operator=refreshed,
                                entry=candidate, equality=next_equality, target=target,
                                s=next_s, spent_action=spent + step_action,
                                initial_budget=initial_budget,
                                namespace=f"{namespace}|next-s={next_s:.8f}",
                                policy=policy, tolerances=tolerances,
                            )
                        finally:
                            candidate.restore(model)
                    realized_action, realized_layers, rollout_receipt = _actual_rollout(
                        model=model, factory=factory, start=candidate, phi0=phi0,
                        displacement=displacement, start_progress=next_s,
                        tolerances=tolerances,
                    )
                    candidate.restore(model)
                    prediction = concordance([SuffixCandidate(
                        candidate_id="ACCEPTED_CONTROLLER_CANDIDATE",
                        predicted_suffix_after=predicted_suffix,
                        realized_suffix_action=realized_action,
                        layer_action=realized_layers,
                    )])
                    waypoints.append({
                        "ordinal": len(waypoints),
                        "scheduled_start": scheduled_start,
                        "scheduled_end": scheduled_end,
                        "s_entry": accepted_s,
                        "s_accepted": next_s,
                        "delta_s": delta_s,
                        "backtrack_count": attempt,
                        "phi_entry_sha256": tensor_sha256(phi_entry),
                        "target_waypoint_sha256": tensor_sha256(waypoint),
                        "geometry": geometry,
                        "refreshed_geometry": refreshed_geometry,
                        "equality": {"range_residual": equality.range_residual, "action_squared": equality.action_squared, "solver": asdict(equality.receipt)},
                        "barrier": barrier_receipt,
                        "corrector": corrector,
                        "step_action": step_action,
                        "layer_action": list(layer_action),
                        "spent_action_after": spent + step_action,
                        "predicted_suffix_after": predicted_suffix,
                        "realized_suffix_action": realized_action,
                        "realized_suffix_layer_action": list(realized_layers),
                        "realized_rollout": rollout_receipt,
                        "predictive_concordance": prediction,
                        "strict_barrier_verification": strict_receipt,
                        "next_state_viability": next_viability,
                        "candidate_root": candidate.root,
                    })
                    accepted_s = next_s
                    spent += step_action
                    accepted_count += 1
                    backtrack_total += attempt
                    break
                except CandidateFailure:
                    rejected_count += 1
                    entry.restore(model)
                    if attempt >= NumericalLock().maximum_backtracks:
                        raise
                    attempt += 1
            if accepted_s < scheduled_end - 1e-12:
                schedule.insert(0, (accepted_s, scheduled_end))
        terminal = WeightSnapshot.capture(model, w0.names)
        terminal_operator, terminal_geometry = factory.build()
        closure = float(torch.linalg.vector_norm(terminal_operator.phi() - target).item())
        terminal_restore = w0.restore(model)
        strict_terminal = None
        status = "TERMINAL_VALID"
        if arm is Arm.FZCB:
            strict_terminal = verify_terminal(
                closure=closure,
                spent_action=spent,
                initial_budget=initial_budget,
                current_viability=True,
                all_waypoints_strict=all(bool(row["strict_barrier_verification"]["strict_pass"]) for row in waypoints),
                corrector_action_included=all(bool(row["corrector"]["corrector_action_included"]) for row in waypoints),
                rollback_exact=bool(terminal_restore["bytes_exact"] and terminal_restore["pointer_exact"]),
                tolerances=tolerances,
            )
            status = "TERMINAL_VALID_STRICT"
        commit = transaction.commit_terminal(terminal)
        return {
            "arm": arm.value,
            "status": status,
            "A0": initial_budget,
            "budget_calibration": budget_receipt,
            "tolerances": tolerances.payload(),
            "initial_geometry": initial_geometry,
            "waypoints": waypoints,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "backtrack_count": backtrack_total,
            "terminal": {
                "s": accepted_s,
                "absolute_closure": closure,
                "spent_action": spent,
                "budget": initial_budget,
                "action_over_A0": spent / initial_budget,
                "weight_root": terminal.root,
                "geometry": terminal_geometry,
                "strict_verification": strict_terminal,
            },
            "atomic_commit": commit,
            "direct_z_recompute_count": 0,
            "controller_output_metric_access_count": 0,
        }
    except Exception as exc:
        rollback = transaction.rollback()
        if isinstance(exc, ArmExecutionFailure):
            exc.receipt["rollback"] = rollback
            exc.receipt["accepted_count"] = accepted_count
            exc.receipt["rejected_count"] = rejected_count
            exc.receipt["backtrack_count"] = backtrack_total
            exc.receipt["waypoints_before_failure"] = waypoints
            raise
        raise ArmExecutionFailure(str(exc), {
            "arm": arm.value,
            "stage": "controller_path",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "A0": initial_budget,
            "E": spent,
            "tolerances": tolerances.payload(),
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "backtrack_count": backtrack_total,
            "waypoints_before_failure": waypoints,
            "rollback": rollback,
        }) from exc


def run_strong_static(
    *, model: Any, factory: MEMITGeometryFactory, w0: WeightSnapshot,
    target: torch.Tensor, policy: UnitTolerancePolicy,
) -> dict[str, Any]:
    arm = Arm.STRONG_STATIC_SAME_OBJECTIVE
    w0.restore(model)
    transaction = AtomicEditTransaction(model, w0)
    operator, geometry, phi0, equality, initial_budget, tolerances, budget_receipt = _initial_context(factory, target, policy)
    entry = WeightSnapshot.capture(model, w0.names)
    candidates: list[StaticCandidate] = []
    snapshots: dict[str, WeightSnapshot] = {}
    try:
        corrected, corrector = _corrected_candidate(
            model=model, operator=operator, entry=entry, coefficient=equality.coefficient,
            target_waypoint=target, delta_s=1.0, tolerances=tolerances,
        )
        closure = float(corrector["final_absolute_closure"])
        action = 0.5 * float(torch.dot(corrected, corrected).item())
        snapshot = WeightSnapshot.capture(model, w0.names)
        candidate = StaticCandidate(
            candidate_id="DIRECT_TRANSCRIPTION_SQP_START_EQ",
            closure=closure,
            action=action,
            coefficient_sha256=tensor_sha256(corrected),
            weight_root=snapshot.root,
            solver_iterations=len(corrector["iterations"]),
            residual_history=tuple(float(row["residual_before"]) for row in corrector["iterations"]),
            feasible_equality=closure <= tolerances.tau_z,
            feasible_budget=action <= initial_budget + tolerances.tau_budget,
        )
        candidates.append(candidate)
        snapshots[candidate.candidate_id] = snapshot
        try:
            selected, solver = select_strong_static(
                candidates, tau_z=tolerances.tau_z,
                initial_budget=initial_budget, tau_budget=tolerances.tau_budget,
            )
        except ScientificBoundary as exc:
            raise ArmExecutionFailure(str(exc), {
                "arm": arm.value,
                "stage": "direct_transcription_endpoint",
                "A0": initial_budget,
                "tolerances": tolerances.payload(),
                "candidates": [row.payload() for row in candidates],
                "best_feasible_action": min(row.action for row in candidates),
                "rollback": entry.restore(model),
            }) from exc
        terminal = snapshots[selected.candidate_id]
        w0.restore(model)
        commit = transaction.commit_terminal(terminal)
        return {
            "arm": arm.value,
            "status": "TERMINAL_VALID_STRICT",
            "A0": initial_budget,
            "budget_calibration": budget_receipt,
            "tolerances": tolerances.payload(),
            "initial_geometry": geometry,
            "solver": solver,
            "terminal": {
                "absolute_closure": selected.closure,
                "spent_action": selected.action,
                "budget": initial_budget,
                "action_over_A0": selected.action / initial_budget,
                "weight_root": terminal.root,
            },
            "atomic_commit": commit,
            "direct_z_recompute_count": 0,
            "controller_output_metric_access_count": 0,
        }
    except ArmExecutionFailure:
        transaction.rollback()
        raise
    except Exception as exc:
        rollback = transaction.rollback()
        raise ArmExecutionFailure(str(exc), {
            "arm": arm.value,
            "stage": "direct_transcription_endpoint",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "A0": initial_budget,
            "tolerances": tolerances.payload(),
            "candidates": [row.payload() for row in candidates],
            "rollback": rollback,
        }) from exc


def run_validity_arm(
    *, arm: Arm, model: Any, factory: MEMITGeometryFactory,
    w0: WeightSnapshot, target: torch.Tensor, namespace: str,
    policy: UnitTolerancePolicy,
) -> dict[str, Any]:
    if arm is Arm.TRUE_FROZEN_C_SPLIT:
        return run_true_frozen(model=model, factory=factory, w0=w0, target=target, policy=policy)
    if arm in {Arm.REFRESHED_EQUALITY_ONLY, Arm.FZCB}:
        return _run_refreshed_or_fzcb(
            arm=arm, model=model, factory=factory, w0=w0, target=target,
            namespace=namespace, policy=policy,
        )
    if arm is Arm.STRONG_STATIC_SAME_OBJECTIVE:
        return run_strong_static(model=model, factory=factory, w0=w0, target=target, policy=policy)
    raise ScientificBoundary(f"unsupported TECH-R1 validity arm: {arm}")
