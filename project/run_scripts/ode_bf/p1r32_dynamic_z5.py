"""Dynamic-Z5 target solver and direct full-residual Atomic router.

The module is additive to the frozen P1R24 source.  It deliberately keeps
target optimization, applied-coefficient routing, and materialization
coordinates distinct so that remaining-step/debt and inverse-slope paths
cannot enter the P1R32 decision path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
import torch

from .contracts import ODEBFContractError, canonical_hash
from .fixed_e8_soft_routing import (
    FIXED_E8_SOLVER_FTOL,
    FIXED_E8_SOLVER_MAXITER,
    FixedE8Arm,
)
from .functional import tensor_sha256
from .p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24KLPlan,
    P1R24KLResult,
    evaluate_p1r24_kl,
)
from .progress_simplex_routing import (
    SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
    SIMPLEX_ENERGY_RELATIVE_TOLERANCE,
    SIMPLEX_PRIMAL_TOLERANCE,
    SIMPLEX_XI_TIE_TOLERANCE,
)
from .routing import RoutingProblem
from .scalable_batched_model import (
    ScalableObjectivePlan,
    ScalableObjectiveResult,
    evaluate_scalable_target_new_objective,
)


P1R32_INSTRUCTION_ID = "ODEEDIT-S05-P1R32-DYNAMIC-Z5-FULL-RESIDUAL-ATOMIC-V1"
P1R32_METHOD_ID = "P1R32-DYNAMIC-Z5-FULL-RESIDUAL-DIRECT-C-ATOMIC-V1"
P1R32_K = 8
P1R32_H = 1.0 / P1R32_K
P1R32_INNER_ADAM_STEPS = 5
P1R32_NUMERICAL_EPSILON = 1.0e-8


@dataclass(frozen=True, slots=True)
class DynamicZ5AliasLock:
    alias: str
    learning_rate: float
    kl_factor: float
    decay_factor: float
    clamp_factor: float

    @classmethod
    def for_alias(cls, alias: str) -> "DynamicZ5AliasLock":
        inherited = P1R24AliasTargetLock.for_alias(alias)
        learning_rate = {
            "llama3-8b-inst": 0.1,
            "qwen2.5-7b-inst": 0.5,
        }[alias]
        return cls(
            alias,
            learning_rate,
            inherited.kl_factor,
            inherited.decay_factor,
            inherited.clamp_factor,
        )

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "learning_rate": self.learning_rate,
            "kl_factor": self.kl_factor,
            "decay_factor": self.decay_factor,
            "clamp_factor": self.clamp_factor,
            "inner_adam_steps": P1R32_INNER_ADAM_STEPS,
            "moment_reset_per_outer_step": True,
            "semantic_early_stop_count": 0,
        }


def verify_dynamic_z5_alias_hparams(hparams: Any, alias: str) -> dict[str, Any]:
    lock = DynamicZ5AliasLock.for_alias(alias)
    observed = (
        float(hparams.v_lr),
        float(hparams.kl_factor),
        float(hparams.v_weight_decay),
        float(hparams.clamp_norm_factor),
    )
    expected = (
        lock.learning_rate,
        lock.kl_factor,
        lock.decay_factor,
        lock.clamp_factor,
    )
    if observed != expected:
        raise ODEBFContractError("P1R32 pinned Native target hparams differ")
    payload = {
        "schema": "ode-edit-s05-p1r32-native-target-hparams/v1",
        "alias": alias,
        "observed": list(observed),
        "expected": list(expected),
        "exact": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _distribution(values: Sequence[float]) -> dict[str, float]:
    observed = np.asarray(tuple(float(item) for item in values), dtype=np.float64)
    if observed.ndim != 1 or observed.size == 0 or not np.all(np.isfinite(observed)):
        raise ODEBFContractError("P1R32 inner target statistic differs")
    return {
        "mean": float(np.mean(observed)),
        "median": float(np.median(observed)),
        "p90": float(np.quantile(observed, 0.9)),
        "worst": float(np.max(observed)),
    }


def _clamp_delta(
    delta: torch.Tensor,
    radii: torch.Tensor,
) -> tuple[list[float], int]:
    if delta.ndim != 2 or radii.shape != (delta.shape[1],):
        raise ODEBFContractError("P1R32 per-request clamp geometry differs")
    norms = torch.linalg.vector_norm(delta.detach().float(), dim=0)
    ratios = torch.ones_like(norms)
    positive = norms > radii
    ratios[positive] = radii[positive] / norms[positive]
    with torch.no_grad():
        delta.mul_(ratios.unsqueeze(0).to(device=delta.device, dtype=delta.dtype))
    return [float(item) for item in ratios.detach().cpu()], int(positive.sum().detach().cpu())


def _decay_terms(
    delta: torch.Tensor,
    y_norm: torch.Tensor,
    decay_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta64 = delta.detach().to(device="cpu", dtype=torch.float64)
    norms = torch.linalg.vector_norm(delta64, dim=0)
    denominator = torch.square(y_norm.to(dtype=torch.float64))
    if bool(torch.any(denominator <= 0.0)):
        raise ODEBFContractError("P1R32 target activation norm is degenerate")
    values = decay_factor * norms / denominator
    gradient = torch.zeros_like(delta64)
    nonzero = norms > 0.0
    gradient[:, nonzero] = (
        decay_factor
        * delta64[:, nonzero]
        / norms[nonzero].unsqueeze(0)
        / denominator[nonzero].unsqueeze(0)
    )
    return values, gradient


@dataclass(frozen=True, slots=True)
class DynamicZ5TargetResult:
    target_next: torch.Tensor
    full_residual: torch.Tensor
    field_velocity: torch.Tensor
    final_nll: ScalableObjectiveResult
    final_kl: P1R24KLResult
    rho_write_signed: float
    rho_write: float
    receipt: Mapping[str, Any]


def solve_dynamic_z5_target(
    model: torch.nn.Module,
    objective_plan: ScalableObjectivePlan,
    kl_plan: P1R24KLPlan,
    *,
    alias: str,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    step_index: int,
    target_layer_name: str,
) -> DynamicZ5TargetResult:
    """Run five exact Adam state updates at one frozen outer W state.

    Model-objective gradients are accumulated exactly over the source-frozen
    request microbatches.  A single logical ``loss.backward`` then installs
    that exact summed gradient into the request-column Adam tensor, giving
    five optimizer steps without retaining all model graphs simultaneously.
    """

    lock = DynamicZ5AliasLock.for_alias(alias)
    if step_index < 0 or step_index >= P1R32_K:
        raise ODEBFContractError("P1R32 outer step differs")
    if (
        current_target.shape != current_terminal.shape
        or current_target.ndim != 2
        or current_target.shape[1] != objective_plan.request_count
        or kl_plan.request_count != objective_plan.request_count
    ):
        raise ODEBFContractError("P1R32 target state geometry differs")
    device = next(model.parameters()).device
    terminal64 = current_terminal.detach().to(device="cpu", dtype=torch.float64)
    y_norm = torch.linalg.vector_norm(terminal64, dim=0)
    radii_cpu = lock.clamp_factor * y_norm
    warm = (
        current_target.detach().to(device="cpu", dtype=torch.float64) - terminal64
    ).to(dtype=torch.float32)
    if step_index == 0 and bool(torch.count_nonzero(warm)):
        raise ODEBFContractError("P1R32 first target warm start is not zero")
    delta = torch.nn.Parameter(warm.to(device=device).contiguous())
    radii = radii_cpu.to(device=device, dtype=torch.float32)
    warm_ratio, warm_clamp_count = _clamp_delta(delta, radii)
    optimizer = torch.optim.Adam((delta,), lr=lock.learning_rate)

    teacher_result, teacher = evaluate_p1r24_kl(
        model, kl_plan, teacher_log_probs=None
    )
    inner: list[dict[str, Any]] = []
    model_forward_count = teacher_result.model_forward_count
    model_backward_count = 0
    processed_token_count = teacher_result.processed_token_count
    optimizer_backward_count = 0
    optimizer_step_count = 0

    def observe(inner_index: int) -> tuple[
        ScalableObjectiveResult,
        P1R24KLResult,
        torch.Tensor,
        dict[str, Any],
    ]:
        target_state = current_terminal.to(device=device, dtype=torch.float32) + delta
        nll = evaluate_scalable_target_new_objective(
            model,
            objective_plan,
            target_state=target_state,
            current_terminal=current_terminal,
            target_layer_name=target_layer_name,
        )
        kl, _ = evaluate_p1r24_kl(
            model,
            kl_plan,
            teacher_log_probs=teacher,
            target_state=target_state,
            current_terminal=current_terminal,
            target_layer_name=target_layer_name,
        )
        if nll.target_gradient is None or kl.gradient is None:
            raise ODEBFContractError("P1R32 target gradient is absent")
        decay_values, decay_gradient = _decay_terms(delta, y_norm, lock.decay_factor)
        request_count = objective_plan.request_count
        gradient = (
            request_count
            * (
                nll.target_gradient.detach().to(device="cpu", dtype=torch.float64)
                + lock.kl_factor
                * kl.gradient.detach().to(device="cpu", dtype=torch.float64)
            )
            + decay_gradient
        )
        per_request = [
            float(
                nll.per_request_values[index]
                + lock.kl_factor * kl.per_request_values[index]
                + decay_values[index]
            )
            for index in range(request_count)
        ]
        grad_norm = [
            float(item) for item in torch.linalg.vector_norm(gradient, dim=0)
        ]
        delta_norm = [
            float(item)
            for item in torch.linalg.vector_norm(
                delta.detach().to(device="cpu", dtype=torch.float64), dim=0
            )
        ]
        row = {
            "schema": "ode-edit-s05-p1r32-dynamic-z5-inner-observation/v1",
            "outer_k": step_index,
            "inner_j": inner_index,
            "target_new_nll_by_request": list(nll.per_request_values),
            "kl_by_request": list(kl.per_request_values),
            "decay_by_request": [float(item) for item in decay_values],
            "combined_objective_by_request": per_request,
            "gradient_norm_by_request": grad_norm,
            "delta_norm_by_request": delta_norm,
            "target_new_nll": _distribution(nll.per_request_values),
            "kl": _distribution(kl.per_request_values),
            "decay": _distribution([float(item) for item in decay_values]),
            "combined_objective": _distribution(per_request),
            "gradient_norm": _distribution(grad_norm),
            "delta_norm": _distribution(delta_norm),
            "delta_sha256": tensor_sha256(delta),
            "target_state_sha256": tensor_sha256(target_state),
            "teacher_sha256": [tensor_sha256(item) for item in teacher],
            "nll_receipt_sha256": nll.identity_sha256,
            "kl_receipt_sha256": kl.identity_sha256,
            "optimizer_update_norm_by_request": None,
            "clamp_ratio_by_request": None,
            "clamp_count": None,
        }
        return nll, kl, gradient, row

    final_nll: ScalableObjectiveResult | None = None
    final_kl: P1R24KLResult | None = None
    entry_target_new_gradient: torch.Tensor | None = None
    for inner_index in range(P1R32_INNER_ADAM_STEPS):
        nll, kl, gradient, row = observe(inner_index)
        if inner_index == 0:
            entry_target_new_gradient = nll.target_gradient.detach().to(
                device="cpu", dtype=torch.float64
            )
        model_forward_count += nll.model_forward_count + kl.model_forward_count
        model_backward_count += nll.backward_count + kl.backward_count
        processed_token_count += nll.processed_token_count + kl.processed_token_count
        before = delta.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.sum(delta * gradient.to(device=device, dtype=delta.dtype).detach())
        loss.backward()
        optimizer_backward_count += 1
        optimizer.step()
        optimizer_step_count += 1
        clamp_ratio, clamp_count = _clamp_delta(delta, radii)
        update_norm = [
            float(item)
            for item in torch.linalg.vector_norm(
                (delta.detach() - before).to(device="cpu", dtype=torch.float64), dim=0
            )
        ]
        row.update(
            {
                "optimizer_update_norm_by_request": update_norm,
                "optimizer_update_norm": _distribution(update_norm),
                "clamp_ratio_by_request": clamp_ratio,
                "clamp_count": clamp_count,
            }
        )
        row["identity_sha256"] = canonical_hash(row)
        inner.append(row)

    final_nll, final_kl, _, final_row = observe(P1R32_INNER_ADAM_STEPS)
    model_forward_count += final_nll.model_forward_count + final_kl.model_forward_count
    model_backward_count += final_nll.backward_count + final_kl.backward_count
    processed_token_count += final_nll.processed_token_count + final_kl.processed_token_count
    final_row.update(
        {
            "optimizer_update_norm_by_request": [0.0] * objective_plan.request_count,
            "optimizer_update_norm": _distribution([0.0] * objective_plan.request_count),
            "clamp_ratio_by_request": [1.0] * objective_plan.request_count,
            "clamp_count": 0,
            "post_update_measurement_only": True,
        }
    )
    final_row["identity_sha256"] = canonical_hash(final_row)
    inner.append(final_row)

    residual = delta.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if entry_target_new_gradient is None:
        raise ODEBFContractError("P1R32 writer demand gradient is absent")
    rho_write_signed = float(
        -torch.sum(entry_target_new_gradient * residual.to(dtype=torch.float64))
    )
    rho_write = max(rho_write_signed, 0.0)
    target_next = (
        current_terminal.detach().to(device="cpu", dtype=torch.float32) + residual
    ).contiguous()
    field_velocity = (residual / P1R32_H).contiguous()
    identity_residual = float(
        torch.max(torch.abs(P1R32_H * field_velocity - residual))
    )
    if identity_residual > P1R32_NUMERICAL_EPSILON:
        raise ODEBFContractError("P1R32 full-residual coordinate differs")
    payload = {
        "schema": "ode-edit-s05-p1r32-dynamic-z5-target/v1",
        "instruction_id": P1R32_INSTRUCTION_ID,
        "method_id": P1R32_METHOD_ID,
        "outer_k": step_index,
        "alias_lock": lock.raw_free_payload(),
        "warm_start_sha256": tensor_sha256(warm),
        "first_outer_warm_start_exact_zero": (
            bool(torch.count_nonzero(warm) == 0) if step_index == 0 else None
        ),
        "warm_start_clamp_ratio_by_request": warm_ratio,
        "warm_start_clamp_count": warm_clamp_count,
        "teacher_refresh_count": 1,
        "teacher_receipt_sha256": teacher_result.identity_sha256,
        "adam_moment_reset_count": 1,
        "adam_optimizer_step_count": optimizer_step_count,
        "loss_backward_count": optimizer_backward_count,
        "final_post_update_measurement_count": 1,
        "inner_observation_count": len(inner),
        "inner_observations": inner,
        "target_next_sha256": tensor_sha256(target_next),
        "full_residual_sha256": tensor_sha256(residual),
        "field_velocity_sha256": tensor_sha256(field_velocity),
        "full_residual_norm_by_request": [
            float(item) for item in torch.linalg.vector_norm(residual.double(), dim=0)
        ],
        "writer_demand_source": "P1R24_A0_TARGET_NEW_ENTRY_GRADIENT_DOT_FULL_RESIDUAL",
        "writer_entry_target_new_gradient_sha256": tensor_sha256(
            entry_target_new_gradient
        ),
        "rho_write_signed": rho_write_signed,
        "rho_write": rho_write,
        "full_residual_divisor": 1,
        "remaining_step_division_count": 0,
        "semantic_debt_input_count": 0,
        "physical_h_application_count": 1,
        "second_h_application_count": 0,
        "inverse_slope_operation_count": 0,
        "full_residual_identity_max_abs": identity_residual,
        "target_model_forward_count": model_forward_count,
        "target_model_backward_count": model_backward_count,
        "processed_token_count": processed_token_count,
        "inner_key_factor_slope_materialization_count": 0,
        "heldout_evaluator_access_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return DynamicZ5TargetResult(
        target_next,
        residual,
        field_velocity,
        final_nll,
        final_kl,
        rho_write_signed,
        rho_write,
        payload,
    )


class DynamicZ5RoutingStatus(str, Enum):
    JOINT_WRITE = "JOINT_WRITE"
    NO_POSITIVE_DIRECTION = "NO_POSITIVE_DIRECTION"


@dataclass(frozen=True, slots=True)
class DynamicZ5RoutingResult:
    arm: FixedE8Arm
    status: DynamicZ5RoutingStatus
    raw_applied_slopes: tuple[float, ...]
    nominal_applied_coefficient: tuple[float, ...]
    selected_applied_coefficient: tuple[float, ...]
    velocity: tuple[float, ...]
    reference_progress: float
    predicted_progress: float
    equality_residual: float
    reference_energy: float
    selected_energy: float
    reference_p: float
    selected_p: float
    selected_capacity: float
    certificates: tuple[Mapping[str, Any], ...]
    fallback_to_c0: bool
    identity_sha256: str

    @property
    def alpha_req(self) -> float:
        return self.reference_progress

    @property
    def alpha_max(self) -> float:
        return self.reference_progress

    @property
    def alpha_apply(self) -> float:
        return self.predicted_progress

    @property
    def coverage(self) -> float:
        if self.reference_progress == 0.0:
            return 1.0
        return self.predicted_progress / self.reference_progress

    @property
    def applied_coefficient(self) -> tuple[float, ...]:
        return self.selected_applied_coefficient

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "instruction_id": P1R32_INSTRUCTION_ID,
                "method_id": P1R32_METHOD_ID,
                "parameterization": "DIRECT_APPLIED_COEFFICIENT_c=h*v",
                "strength_definition": "raw_applied_slope^T c",
                "inverse_slope_operation_count": 0,
                "residual_attenuation_count": 0,
                "hard_p_budget_count": 0,
                "functional_p_veto_count": 0,
                "retry_backtracking_count": 0,
            }
        )
        return payload


def _energy(velocity: np.ndarray, metric: np.ndarray) -> float:
    return float(velocity @ metric @ velocity)


def solve_dynamic_z5_direct_routing(
    problem: RoutingProblem,
    *,
    arm: FixedE8Arm | str,
    rho_write: float,
) -> DynamicZ5RoutingResult:
    selected_arm = FixedE8Arm(arm)
    raw_slopes = np.asarray(problem.signed_progress, dtype=np.float64)
    energy_metric = np.asarray(problem.trust_metric, dtype=np.float64)
    capacity_metric = np.asarray(problem.capacity_metric, dtype=np.float64)
    if (
        not math.isfinite(rho_write)
        or rho_write < 0.0
        or raw_slopes.shape != (5,)
        or not np.all(np.isfinite(raw_slopes))
        or not np.all(np.isfinite(energy_metric))
        or not np.all(np.isfinite(capacity_metric))
    ):
        raise ODEBFContractError("P1R32 direct router problem differs")
    velocity_slopes = P1R32_H * raw_slopes
    active = velocity_slopes > 0.0
    q = float(np.sum(velocity_slopes[active]))
    v0 = np.zeros(5, dtype=np.float64)
    if q > 0.0 and rho_write > 0.0:
        # Exact P1R24 A0 Neutral map with the inherited raw-coefficient to
        # velocity-coordinate conversion exposed. The rho*pi/a expression
        # simplifies to one common rho/q coefficient on active directions.
        v0[active] = rho_write / q
    c0 = P1R32_H * v0
    reference_progress = float(raw_slopes @ c0)
    reference_energy = _energy(v0, energy_metric)
    reference_p = float(problem.pretrained.value(v0))
    certificates: list[Mapping[str, Any]] = []
    fallback = False
    if not np.any(active):
        status = DynamicZ5RoutingStatus.NO_POSITIVE_DIRECTION
        chosen_v = np.zeros(5, dtype=np.float64)
    else:
        status = DynamicZ5RoutingStatus.JOINT_WRITE
        chosen_v = v0.copy()
        if selected_arm is FixedE8Arm.SOFT:
            strength_slopes = P1R32_H * raw_slopes
            energy_limit = (
                reference_energy * (1.0 + SIMPLEX_ENERGY_RELATIVE_TOLERANCE)
                + SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
            )

            def p_value(value: np.ndarray) -> float:
                return float(problem.pretrained.value(value))

            def p_grad(value: np.ndarray) -> np.ndarray:
                return 2.0 * problem.pretrained.linear + 2.0 * problem.pretrained.gram @ value

            def e_value(value: np.ndarray) -> float:
                return _energy(value, energy_metric)

            def e_grad(value: np.ndarray) -> np.ndarray:
                return 2.0 * energy_metric @ value

            constraints = (
                {
                    "type": "ineq",
                    "fun": lambda value: float(
                        strength_slopes @ value
                        - reference_progress
                        + SIMPLEX_PRIMAL_TOLERANCE
                    ),
                    "jac": lambda value: strength_slopes,
                },
                {
                    "type": "ineq",
                    "fun": lambda value: float(energy_limit - e_value(value)),
                    "jac": lambda value: -e_grad(value),
                },
            )
            stage1 = minimize(
                p_value,
                v0,
                jac=p_grad,
                method="SLSQP",
                bounds=tuple((0.0, None) for _ in range(5)),
                constraints=constraints,
                options={
                    "disp": False,
                    "ftol": FIXED_E8_SOLVER_FTOL,
                    "maxiter": FIXED_E8_SOLVER_MAXITER,
                },
            )
            v1 = np.asarray(stage1.x, dtype=np.float64)
            first = {
                "phase": "minimum-cumulative-structural-p-direct-c",
                "success": bool(stage1.success),
                "status": int(stage1.status),
                "negative_violation": max(float(-v1.min(initial=0.0)), 0.0),
                "strength_violation": max(
                    reference_progress - float(strength_slopes @ v1), 0.0
                ),
                "energy_violation": max(e_value(v1) - energy_limit, 0.0),
            }
            first["passed"] = bool(
                first["success"]
                and first["negative_violation"] <= SIMPLEX_PRIMAL_TOLERANCE
                and first["strength_violation"] <= SIMPLEX_PRIMAL_TOLERANCE
                and first["energy_violation"] <= SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
            )
            certificates.append(first)
            if first["passed"]:
                p_star = p_value(v1)
                p_scale = max(float(np.trace(problem.pretrained.gram)), 1.0e-12)
                p_tie = SIMPLEX_XI_TIE_TOLERANCE * p_scale
                constraints2 = (
                    *constraints,
                    {
                        "type": "ineq",
                        "fun": lambda value: float(p_star + p_tie - p_value(value)),
                        "jac": lambda value: -p_grad(value),
                    },
                )

                def capacity(value: np.ndarray) -> float:
                    return float(0.5 * value @ capacity_metric @ value)

                def capacity_grad(value: np.ndarray) -> np.ndarray:
                    return capacity_metric @ value

                stage2 = minimize(
                    capacity,
                    v1,
                    jac=capacity_grad,
                    method="SLSQP",
                    bounds=tuple((0.0, None) for _ in range(5)),
                    constraints=constraints2,
                    options={
                        "disp": False,
                        "ftol": FIXED_E8_SOLVER_FTOL,
                        "maxiter": FIXED_E8_SOLVER_MAXITER,
                    },
                )
                v2 = np.asarray(stage2.x, dtype=np.float64)
                second = {
                    "phase": "minimum-capacity-inside-source-p-tie-direct-c",
                    "success": bool(stage2.success),
                    "status": int(stage2.status),
                    "negative_violation": max(float(-v2.min(initial=0.0)), 0.0),
                    "strength_violation": max(
                        reference_progress - float(strength_slopes @ v2), 0.0
                    ),
                    "energy_violation": max(e_value(v2) - energy_limit, 0.0),
                    "p_tie_violation": max(p_value(v2) - (p_star + p_tie), 0.0),
                }
                second["passed"] = bool(
                    second["success"]
                    and second["negative_violation"] <= SIMPLEX_PRIMAL_TOLERANCE
                    and second["strength_violation"] <= SIMPLEX_PRIMAL_TOLERANCE
                    and second["energy_violation"] <= SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
                    and second["p_tie_violation"] <= P1R32_NUMERICAL_EPSILON
                )
                certificates.append(second)
                if second["passed"] and p_value(v2) <= reference_p:
                    chosen_v = v2
                else:
                    fallback = True
            else:
                fallback = True

    chosen_c = P1R32_H * chosen_v
    predicted = float(raw_slopes @ chosen_c)
    shortfall = max(reference_progress - predicted, 0.0)
    if status is DynamicZ5RoutingStatus.JOINT_WRITE and shortfall > SIMPLEX_PRIMAL_TOLERANCE:
        raise ODEBFContractError("P1R32 direct router attenuated reference strength")
    selected_energy = _energy(chosen_v, energy_metric)
    selected_p = float(problem.pretrained.value(chosen_v))
    selected_capacity = float(0.5 * chosen_v @ capacity_metric @ chosen_v)
    payload = {
        "arm": selected_arm.value,
        "status": status.value,
        "raw_applied_slopes": raw_slopes.tolist(),
        "velocity_slopes": velocity_slopes.tolist(),
        "rho_write": rho_write,
        "active_mask": active.tolist(),
        "q_velocity_coordinate": q,
        "nominal_applied_coefficient": c0.tolist(),
        "selected_applied_coefficient": chosen_c.tolist(),
        "velocity": chosen_v.tolist(),
        "reference_progress": reference_progress,
        "predicted_progress": predicted,
        "shortfall": shortfall,
        "reference_energy": reference_energy,
        "selected_energy": selected_energy,
        "reference_p": reference_p,
        "selected_p": selected_p,
        "selected_capacity": selected_capacity,
        "certificates": certificates,
        "fallback_to_c0": fallback,
        "inverse_slope_operation_count": 0,
        "nominal_reference_source": "P1R24_A0_MATCHED_NEUTRAL_COORDINATE_CORRECTED",
        "raw_to_velocity_slope_h_application_count": 1,
        "h_application_count": 1,
    }
    identity = canonical_hash(payload)
    return DynamicZ5RoutingResult(
        selected_arm,
        status,
        tuple(float(item) for item in raw_slopes),
        tuple(float(item) for item in c0),
        tuple(float(item) for item in chosen_c),
        tuple(float(item) for item in chosen_v),
        reference_progress,
        predicted,
        shortfall,
        reference_energy,
        selected_energy,
        reference_p,
        selected_p,
        selected_capacity,
        tuple(certificates),
        fallback,
        identity,
    )


__all__ = [
    "DynamicZ5AliasLock",
    "DynamicZ5RoutingResult",
    "DynamicZ5RoutingStatus",
    "DynamicZ5TargetResult",
    "P1R32_H",
    "P1R32_INNER_ADAM_STEPS",
    "P1R32_INSTRUCTION_ID",
    "P1R32_K",
    "P1R32_METHOD_ID",
    "solve_dynamic_z5_direct_routing",
    "solve_dynamic_z5_target",
    "verify_dynamic_z5_alias_hparams",
]
