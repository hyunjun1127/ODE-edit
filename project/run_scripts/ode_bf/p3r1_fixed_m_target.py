"""P3R1 request-wise fixed-budget Adam target solver.

This module owns target-coordinate arithmetic only.  It has no writer,
history, factor, materializer, held-out evaluator, or model-weight authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import P1R24AliasTargetLock, P1R24KLResult
from .p1r38_perrequest_target import (
    P1R38_ADAM_EPSILON,
    P1R38_BETA1,
    P1R38_BETA2,
    P1R38_LR_BY_ALIAS,
)
from .scalable_batched_model import ScalableObjectiveResult


INSTRUCTION_ID = "ODEEDIT-S05-P3R1-TWO-TIMESCALE-FASTZ-FH-C013-FP32-V1"
METHOD_ID = "P3R1-TWO-TIMESCALE-FASTZ-FH-C013-FP32-V1"
TARGET_SELECTION_POLICY = "FIXED_M_FINAL_ITERATE"
PRODUCTION_INNER_COUNT = 5
TARGET_ONLY_REFERENCE_INNER_COUNT = 25


@dataclass(frozen=True, slots=True)
class FixedMTargetResult:
    final_target: torch.Tensor
    inner_receipts: tuple[Mapping[str, Any], ...]
    receipt: Mapping[str, Any]


def _summary(values: torch.Tensor) -> dict[str, float]:
    observed = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if observed.numel() == 0 or not torch.isfinite(observed).all():
        raise ODEBFContractError("P3R1 target summary differs")
    return {
        "mean": float(torch.mean(observed)),
        "median": float(torch.median(observed)),
        "p90": float(torch.quantile(observed, 0.9)),
        "max": float(torch.max(observed)),
    }


def run_fixed_m_final_iterate(
    *,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    target_origin: torch.Tensor,
    lock: P1R24AliasTargetLock,
    alias: str,
    inner_count: int,
    outer_step_index: int | None,
    evaluate_target: Callable[[torch.Tensor, bool], ScalableObjectiveResult],
    evaluate_kl: Callable[[torch.Tensor, bool], P1R24KLResult],
    fixed_state_identity: Callable[[], str],
) -> FixedMTargetResult:
    """Run exactly ``inner_count`` request-local Adam steps and return the last.

    Objective gradients from the shared batch mean are de-meaned by the exact
    request count before being consumed by request-wise Adam moments.
    """

    if alias not in P1R38_LR_BY_ALIAS:
        raise ODEBFContractError("P3R1 target alias differs")
    if inner_count not in (PRODUCTION_INNER_COUNT, TARGET_ONLY_REFERENCE_INNER_COUNT):
        raise ODEBFContractError("P3R1 target inner count differs")
    if outer_step_index is not None and not 0 <= outer_step_index < 8:
        raise ODEBFContractError("P3R1 target outer index differs")
    if (
        current_target.ndim != 2
        or current_terminal.shape != current_target.shape
        or target_origin.shape != current_target.shape
        or current_target.dtype is not torch.float32
        or current_terminal.dtype is not torch.float32
        or target_origin.dtype is not torch.float32
    ):
        raise ODEBFContractError("P3R1 target geometry/dtype differs")
    request_count = current_target.shape[1]
    if request_count <= 0:
        raise ODEBFContractError("P3R1 target request count differs")

    entry_identity = fixed_state_identity()
    target64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    terminal64 = current_terminal.detach().to(device="cpu", dtype=torch.float64)
    origin64 = target_origin.detach().to(device="cpu", dtype=torch.float64)
    origin_norm = torch.linalg.vector_norm(origin64, dim=0)
    if bool(torch.any(origin_norm <= 0.0)):
        raise ODEBFContractError("P3R1 target origin is degenerate")
    m = torch.zeros_like(target64)
    v = torch.zeros_like(target64)
    objective_by_iterate: list[torch.Tensor] = []
    inner_receipts: list[Mapping[str, Any]] = []

    for inner_index in range(inner_count + 1):
        gradient_required = inner_index < inner_count
        target32 = target64.to(dtype=torch.float32).contiguous()
        if gradient_required:
            target32.requires_grad_(True)
        target_result = evaluate_target(target32, gradient_required)
        kl_result = evaluate_kl(target32, gradient_required)
        if gradient_required and (
            target_result.target_gradient is None or kl_result.gradient is None
        ):
            raise ODEBFContractError("P3R1 target gradient is absent")
        if (
            len(target_result.per_request_values) != request_count
            or len(kl_result.per_request_values) != request_count
        ):
            raise ODEBFContractError("P3R1 target request telemetry differs")

        displacement = target64 - origin64
        displacement_norm = torch.linalg.vector_norm(displacement, dim=0)
        decay_values = lock.decay_factor * displacement_norm / torch.square(origin_norm)
        decay_gradient_mean = torch.zeros_like(displacement)
        nonzero = displacement_norm > 0.0
        decay_gradient_mean[:, nonzero] = (
            lock.decay_factor
            * displacement[:, nonzero]
            / displacement_norm[nonzero].unsqueeze(0)
            / torch.square(origin_norm[nonzero]).unsqueeze(0)
            / request_count
        )
        request_gradient = torch.zeros_like(target64)
        if gradient_required:
            assert target_result.target_gradient is not None and kl_result.gradient is not None
            nll_gradient_mean = target_result.target_gradient.detach().to(
                device="cpu", dtype=torch.float64
            )
            kl_gradient_mean = kl_result.gradient.detach().to(
                device="cpu", dtype=torch.float64
            )
            request_gradient = request_count * (
                nll_gradient_mean
                + lock.kl_factor * kl_gradient_mean
                + decay_gradient_mean
            )
            if request_gradient.shape != target64.shape or not torch.isfinite(request_gradient).all():
                raise ODEBFStateError("P3R1 target gradient is nonfinite or malformed")
        nll_values = torch.tensor(
            target_result.per_request_values, dtype=torch.float64
        )
        kl_values = torch.tensor(kl_result.per_request_values, dtype=torch.float64)
        objective_values = nll_values + lock.kl_factor * kl_values + decay_values
        if not torch.isfinite(objective_values).all():
            raise ODEBFStateError("P3R1 target objective is nonfinite")
        objective_by_iterate.append(objective_values.clone())

        step_norm = torch.zeros(request_count, dtype=torch.float64)
        clamp_removed_norm = torch.zeros(request_count, dtype=torch.float64)
        clamp_hit = torch.zeros(request_count, dtype=torch.bool)
        next_sha: str | None = None
        if inner_index < inner_count:
            t = inner_index + 1
            m = P1R38_BETA1 * m + (1.0 - P1R38_BETA1) * request_gradient
            v = P1R38_BETA2 * v + (1.0 - P1R38_BETA2) * torch.square(
                request_gradient
            )
            m_hat = m / (1.0 - P1R38_BETA1**t)
            v_hat = v / (1.0 - P1R38_BETA2**t)
            raw_step = (
                -P1R38_LR_BY_ALIAS[alias]
                * m_hat
                / (torch.sqrt(v_hat) + P1R38_ADAM_EPSILON)
            )
            proposed = target64 + raw_step
            clamped = proposed.clone()
            for request_index in range(request_count):
                relative = proposed[:, request_index] - origin64[:, request_index]
                relative_norm = float(torch.linalg.vector_norm(relative))
                maximum = lock.clamp_factor * float(origin_norm[request_index])
                scale = 1.0 if relative_norm <= maximum or relative_norm == 0.0 else maximum / relative_norm
                clamped[:, request_index] = origin64[:, request_index] + scale * relative
                clamp_hit[request_index] = scale < 1.0
            next32 = clamped.to(dtype=torch.float32).contiguous()
            if not torch.isfinite(next32).all():
                raise ODEBFStateError("P3R1 target iterate is nonfinite")
            applied = next32.to(dtype=torch.float64) - target64
            removed = proposed - next32.to(dtype=torch.float64)
            step_norm = torch.linalg.vector_norm(applied, dim=0)
            clamp_removed_norm = torch.linalg.vector_norm(removed, dim=0)
            target64 = next32.to(dtype=torch.float64)
            next_sha = tensor_sha256(next32)

        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-p3r1-fixed-m-target-inner/v1",
            "instruction_id": INSTRUCTION_ID,
            "method_id": METHOD_ID,
            "target_selection_policy": TARGET_SELECTION_POLICY,
            "outer_step_index": outer_step_index,
            "inner_index": inner_index,
            "configured_update_count": inner_count,
            "is_final_iterate": inner_index == inner_count,
            "gradient_required": gradient_required,
            "target_sha256": tensor_sha256(target32),
            "next_target_sha256": next_sha,
            "train_target_new_nll": _summary(nll_values),
            "kl": _summary(kl_values),
            "decay": _summary(decay_values),
            "total_objective": _summary(objective_values),
            "request_gradient_norm": _summary(
                torch.linalg.vector_norm(request_gradient, dim=0)
            ),
            "adam_step_norm": _summary(step_norm),
            "clamp_hit_count": int(torch.sum(clamp_hit)),
            "clamp_removed_norm": _summary(clamp_removed_norm),
            "batch_demean_factor": request_count,
            "adam_beta1": P1R38_BETA1,
            "adam_beta2": P1R38_BETA2,
            "adam_epsilon": P1R38_ADAM_EPSILON,
            "learning_rate": P1R38_LR_BY_ALIAS[alias],
            "model_forward_count": target_result.model_forward_count + kl_result.model_forward_count,
            "model_backward_count": target_result.backward_count + kl_result.backward_count,
            "primary_count": 0,
            "rescue_count": 0,
            "current_count": 0,
            "early_stop_count": 0,
            "first_hit_stop_count": 0,
            "retry_count": 0,
            "backtracking_count": 0,
            "best_iterate_selection_count": 0,
            "heldout_decision_influence_count": 0,
            "writer_call_count": 0,
            "history_append_count": 0,
            "factor_mutation_count": 0,
            "bf16_path_call_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        inner_receipts.append(payload)

    if fixed_state_identity() != entry_identity:
        raise ODEBFStateError("P3R1 physical state changed inside target solve")
    stacked = torch.stack(objective_by_iterate, dim=0)
    final_values = stacked[-1]
    best_values = torch.min(stacked, dim=0).values
    gap = final_values - best_values
    if bool(torch.any(gap < -torch.finfo(torch.float64).eps)):
        raise ODEBFStateError("P3R1 final-vs-best diagnostic differs")
    final_target = target64.to(dtype=torch.float32).contiguous()
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p3r1-fixed-m-target/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "alias": alias,
        "outer_step_index": outer_step_index,
        "target_selection_policy": TARGET_SELECTION_POLICY,
        "configured_update_count": inner_count,
        "executed_update_count": inner_count,
        "iterate_observation_count": inner_count + 1,
        "adam_moment_reset_count": 1,
        "requestwise_moment_count": request_count,
        "final_target_sha256": tensor_sha256(final_target),
        "final_vs_best_objective_gap": _summary(gap),
        "final_objective": _summary(final_values),
        "best_objective": _summary(best_values),
        "warm_start_target_sha256": tensor_sha256(current_target),
        "target_origin_sha256": tensor_sha256(target_origin),
        "current_terminal_sha256": tensor_sha256(current_terminal),
        "fixed_physical_state_sha256": entry_identity,
        "inner_receipt_sha256": [item["identity_sha256"] for item in inner_receipts],
        "primary_count": 0,
        "rescue_count": 0,
        "current_count": 0,
        "early_stop_count": 0,
        "first_hit_stop_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "best_iterate_selection_count": 0,
        "heldout_decision_influence_count": 0,
        "bf16_path_call_count": 0,
        "inner_writer_call_count": 0,
        "inner_history_append_count": 0,
        "inner_factor_mutation_count": 0,
        "numerical_tolerances": {
            "adam_epsilon": P1R38_ADAM_EPSILON,
            "finite_check": "torch.isfinite exact boolean",
        },
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return FixedMTargetResult(final_target, tuple(inner_receipts), receipt)


__all__ = [
    "FixedMTargetResult",
    "INSTRUCTION_ID",
    "METHOD_ID",
    "PRODUCTION_INNER_COUNT",
    "TARGET_ONLY_REFERENCE_INNER_COUNT",
    "TARGET_SELECTION_POLICY",
    "run_fixed_m_final_iterate",
]
