"""Fixed-M request-independent full-FP32 Adam solver for P4 targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256


P4_INNER_ITERATIONS = 5
P4_ADAM_BETAS = (0.9, 0.999)
P4_ADAM_EPSILON = 1.0e-8


@dataclass(frozen=True, slots=True)
class P4SolverEvaluation:
    per_request_objective: torch.Tensor
    gradient_by_request: torch.Tensor
    telemetry: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class P4TargetSolution:
    final_target: torch.Tensor
    iteration_targets: tuple[torch.Tensor, ...]
    receipt: Mapping[str, Any]


def _validate_state(value: torch.Tensor, *, label: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise ODEBFContractError(f"P4 {label} state differs")


def _origin_clamp(
    candidate: torch.Tensor,
    origin: torch.Tensor,
    *,
    clamp_factor: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    displacement = candidate - origin
    origin_norm = torch.linalg.vector_norm(origin, dim=0)
    if bool(torch.any(origin_norm <= 0.0)):
        raise ODEBFContractError("P4 target origin is degenerate")
    norm = torch.linalg.vector_norm(displacement, dim=0)
    maximum = clamp_factor * origin_norm
    ratio = torch.ones_like(norm)
    hit = norm > maximum
    ratio[hit] = maximum[hit] / norm[hit]
    clamped = origin + displacement * ratio.unsqueeze(0)
    removed = candidate - clamped
    receipt = {
        "clamp_hit_by_request": [bool(item) for item in hit],
        "clamp_ratio_by_request": [float(item) for item in ratio],
        "clamp_removed_norm_by_request": [
            float(item) for item in torch.linalg.vector_norm(removed, dim=0)
        ],
        "clamp_removed_energy_by_request": [
            float(item)
            for item in torch.square(torch.linalg.vector_norm(removed, dim=0))
        ],
    }
    return clamped.contiguous(), receipt


def run_fixed_m_target_adam(
    initial_target: torch.Tensor,
    target_origin: torch.Tensor,
    evaluate: Callable[[torch.Tensor, int], P4SolverEvaluation],
    *,
    learning_rate: float,
    clamp_factor: float,
    outer_step_index: int,
    arm: str,
    paired_input_identity: str,
    inner_iterations: int = P4_INNER_ITERATIONS,
    observe_selected_final: bool = False,
) -> P4TargetSolution:
    """Run the locked Adam budget and return only its final iterate.

    ``evaluate`` must return per-request total J gradients, already including
    the arm's semantic potential plus the pinned KL and decay terms.  Each
    request column has separate Adam moments; moments are created inside this
    call and therefore reset once per outer step.
    """

    _validate_state(initial_target, label="initial")
    _validate_state(target_origin, label="origin")
    if initial_target.shape != target_origin.shape:
        raise ODEBFContractError("P4 target solver geometry differs")
    if (
        outer_step_index < 0
        or outer_step_index >= 8
        or not isinstance(learning_rate, float)
        or learning_rate <= 0.0
        or not isinstance(clamp_factor, float)
        or clamp_factor <= 0.0
        or len(paired_input_identity) != 64
        or inner_iterations not in (1, P4_INNER_ITERATIONS)
        or (inner_iterations == 1 and not observe_selected_final)
    ):
        raise ODEBFContractError("P4 target solver configuration differs")

    current = initial_target.detach().clone().contiguous()
    origin = target_origin.detach().clone().contiguous()
    first_moment = torch.zeros_like(current)
    second_moment = torch.zeros_like(current)
    beta1, beta2 = P4_ADAM_BETAS
    iterations: list[dict[str, Any]] = []
    targets: list[torch.Tensor] = []
    objective_rows: list[tuple[float, ...]] = []
    for iteration in range(inner_iterations):
        evaluation = evaluate(current.detach().clone(), iteration)
        values = evaluation.per_request_objective
        gradient = evaluation.gradient_by_request
        if (
            not isinstance(values, torch.Tensor)
            or values.dtype != torch.float32
            or values.shape != (current.shape[1],)
            or not isinstance(gradient, torch.Tensor)
            or gradient.dtype != torch.float32
            or gradient.shape != current.shape
            or not bool(torch.isfinite(values).all())
            or not bool(torch.isfinite(gradient).all())
        ):
            raise ODEBFContractError("P4 target solver evaluation differs")
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * torch.square(gradient)
        ordinal = iteration + 1
        corrected_first = first_moment / (1.0 - beta1**ordinal)
        corrected_second = second_moment / (1.0 - beta2**ordinal)
        candidate = current - learning_rate * corrected_first / (
            torch.sqrt(corrected_second) + P4_ADAM_EPSILON
        )
        clamped, clamp = _origin_clamp(
            candidate, origin, clamp_factor=clamp_factor
        )
        if clamped.dtype != torch.float32 or not bool(torch.isfinite(clamped).all()):
            raise ODEBFContractError("P4 target iterate is nonfinite or non-FP32")
        displacement = clamped - current
        row: dict[str, Any] = {
            "inner_iteration": iteration,
            "adam_update_ordinal": ordinal,
            "per_request_objective": [float(item) for item in values],
            "objective_mean": float(torch.mean(values)),
            "gradient_norm_by_request": [
                float(item) for item in torch.linalg.vector_norm(gradient, dim=0)
            ],
            "target_before_sha256": tensor_sha256(current),
            "target_after_sha256": tensor_sha256(clamped),
            "step_displacement_norm_by_request": [
                float(item) for item in torch.linalg.vector_norm(displacement, dim=0)
            ],
            "clamp": clamp,
            "telemetry": dict(evaluation.telemetry),
            "heldout_decision_influence_count": 0,
        }
        row["identity_sha256"] = canonical_hash(row)
        iterations.append(row)
        objective_rows.append(tuple(float(item) for item in values))
        current = clamped.detach().clone().contiguous()
        targets.append(current)

    means = [sum(row) / len(row) for row in objective_rows]
    best_index = min(range(inner_iterations), key=means.__getitem__)
    selected_final_observation: dict[str, Any] | None = None
    selected_final_mean: float | None = None
    if observe_selected_final:
        final_evaluation = evaluate(current.detach().clone(), inner_iterations)
        final_values = final_evaluation.per_request_objective
        final_gradient = final_evaluation.gradient_by_request
        if (
            not isinstance(final_values, torch.Tensor)
            or final_values.dtype != torch.float32
            or final_values.shape != (current.shape[1],)
            or not isinstance(final_gradient, torch.Tensor)
            or final_gradient.dtype != torch.float32
            or final_gradient.shape != current.shape
            or not bool(torch.isfinite(final_values).all())
            or not bool(torch.isfinite(final_gradient).all())
        ):
            raise ODEBFContractError("P4 selected-final observation differs")
        selected_final_mean = float(torch.mean(final_values))
        selected_final_observation = {
            "observation_role": "SELECTED_FINAL_AFTER_LAST_UPDATE",
            "selected_iterate_ordinal": inner_iterations,
            "per_request_objective": [float(item) for item in final_values],
            "objective_mean": selected_final_mean,
            "gradient_norm_by_request": [
                float(item)
                for item in torch.linalg.vector_norm(final_gradient, dim=0)
            ],
            "target_sha256": tensor_sha256(current),
            "telemetry": dict(final_evaluation.telemetry),
            "optimizer_update_count": 0,
            "observation_only": True,
            "decision_influence_count": 0,
            "heldout_decision_influence_count": 0,
        }
        selected_final_observation["identity_sha256"] = canonical_hash(
            selected_final_observation
        )
    receipt: dict[str, Any] = {
        "schema": (
            "ode-edit-s05-p4-fixed-m-target-adam/v2"
            if observe_selected_final
            else "ode-edit-s05-p4-fixed-m-target-adam/v1"
        ),
        "arm": arm,
        "outer_step_index": outer_step_index,
        "configured_inner_iterations": inner_iterations,
        "executed_inner_iterations": len(iterations),
        "optimizer": "ADAM",
        "learning_rate": learning_rate,
        "betas": list(P4_ADAM_BETAS),
        "epsilon": P4_ADAM_EPSILON,
        "dtype": "torch.float32",
        "request_independent_optimizer_state": True,
        "moment_reset_count": 1,
        "final_iterate_ordinal": inner_iterations,
        "selected_iterate_ordinal": inner_iterations,
        "best_iterate_ordinal_observation_only": best_index + 1,
        "final_minus_best_objective_observation": means[-1] - means[best_index],
        "best_iterate_decision_influence_count": 0,
        "early_stop_count": 0,
        "retry_count": 0,
        "accept_reject_count": 0,
        "fallback_count": 0,
        "heldout_decision_influence_count": 0,
        "paired_input_identity": paired_input_identity,
        "initial_target_sha256": tensor_sha256(initial_target),
        "origin_target_sha256": tensor_sha256(target_origin),
        "final_target_sha256": tensor_sha256(current),
        "iterations": iterations,
    }
    if selected_final_observation is not None and selected_final_mean is not None:
        observed_means = [*means, selected_final_mean]
        observed_labels = [
            f"PRE_UPDATE_{index}" for index in range(inner_iterations)
        ] + [f"SELECTED_FINAL_AFTER_UPDATE_{inner_iterations}"]
        observed_best = min(range(len(observed_means)), key=observed_means.__getitem__)
        receipt.update(
            {
                "iteration_observation_semantics": "PRE_UPDATE_STATE",
                "pre_update_observation_count": inner_iterations,
                "selected_final_observation_count": 1,
                "selected_final_observation": selected_final_observation,
                "legacy_preupdate_last_minus_best_objective_observation": receipt[
                    "final_minus_best_objective_observation"
                ],
                "best_observed_state_label_observation_only": observed_labels[
                    observed_best
                ],
                "final_minus_best_objective_observation": (
                    selected_final_mean - observed_means[observed_best]
                ),
                "selected_final_observation_decision_influence_count": 0,
            }
        )
    receipt["identity_sha256"] = canonical_hash(receipt)
    return P4TargetSolution(current, tuple(targets), receipt)


__all__ = [
    "P4_ADAM_BETAS",
    "P4_ADAM_EPSILON",
    "P4_INNER_ITERATIONS",
    "P4SolverEvaluation",
    "P4TargetSolution",
    "run_fixed_m_target_adam",
]
