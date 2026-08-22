"""P1R40 requestwise semantic-deficit velocity decay for P1R38 proposals.

This module is tensor arithmetic only.  It consumes the already-formed P1R38
Adam/cap/clamp proposal and the already-computed pure target-new objective.
It never calls a model, changes Adam state, or changes writer allocation.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import (
    P1R24_H,
    P1R24_K,
    P1R24_NUMERICAL_EPSILON,
)
from .p1r38_perrequest_target import (
    P1R38AdamState,
    P1R38SelectedTarget,
    P1R38TargetProposal,
    P1R38_SEMANTIC_EPSILON,
    select_p1r38_target_proposal,
)
from .scalable_batched_model import ScalableObjectiveResult


P1R40_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R40-P1R38-SEMANTIC-DEFICIT-VELOCITY-DECAY-ATOMIC-V1"
)
P1R40_METHOD_ID = "P1R38-SEMANTIC-DEFICIT-VELOCITY-DECAY-V1"
P1R40_VELOCITY_EPSILON = P1R38_SEMANTIC_EPSILON
P1R40_NLL_BINS = (
    ("[0,0.05)", 0.0, 0.05),
    ("[0.05,0.25)", 0.05, 0.25),
    ("[0.25,1)", 0.25, 1.0),
    ("[1,inf)", 1.0, math.inf),
)


class P1R40VelocityMechanismError(RuntimeError):
    """Typed scientific mechanism failure; never repaired or retried."""


def _summary(values: Sequence[float]) -> dict[str, float]:
    tensor = torch.tensor(tuple(float(item) for item in values), dtype=torch.float64)
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ODEBFContractError("P1R40 request summary differs")
    return {
        "mean": float(torch.mean(tensor)),
        "median": float(torch.median(tensor)),
        "p10": float(torch.quantile(tensor, 0.1)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "minimum": float(torch.min(tensor)),
        "maximum": float(torch.max(tensor)),
    }


def _column_sha256(value: torch.Tensor) -> list[str]:
    return [
        tensor_sha256(value[:, index].contiguous())
        for index in range(value.shape[1])
    ]


def _nll_bin_payload(
    deficits: Sequence[float], gammas: Sequence[float]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label, lower, upper in P1R40_NLL_BINS:
        selected = [
            float(gamma)
            for deficit, gamma in zip(deficits, gammas, strict=True)
            if lower <= float(deficit) < upper
        ]
        result.append(
            {
                "label": label,
                "count": len(selected),
                "gamma_summary": None if not selected else _summary(selected),
            }
        )
    return result


def apply_p1r40_semantic_deficit_velocity_decay(
    proposal: P1R38TargetProposal,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    current_nll: ScalableObjectiveResult,
    *,
    step_index: int,
) -> P1R38TargetProposal:
    """Scale the final P1R38 nominal displacement once, request by request."""

    if step_index < 0 or step_index >= P1R24_K:
        raise ODEBFContractError("P1R40 target step index differs")
    shape = current_target.shape
    if (
        current_target.ndim != 2
        or current_terminal.shape != shape
        or proposal.trial_step.target_next.shape != shape
        or proposal.trial_step.nll_gradient.shape != shape
        or current_nll.target_gradient is None
        or current_nll.target_gradient.shape != shape
        or len(current_nll.per_request_values) != shape[1]
        or len(proposal.active_mask) != shape[1]
    ):
        raise ODEBFContractError("P1R40 target geometry differs")
    deficits = torch.tensor(
        tuple(float(item) for item in current_nll.per_request_values),
        dtype=torch.float64,
    )
    if not torch.isfinite(deficits).all() or bool(torch.any(deficits < 0.0)):
        raise ODEBFContractError("P1R40 pure semantic deficit differs")

    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    terminal64 = current_terminal.detach().to(device="cpu", dtype=torch.float64)
    nominal_next = proposal.trial_step.target_next.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    nominal64 = nominal_next.to(dtype=torch.float64)
    nominal_delta64 = nominal64 - current64
    batched_gradient64 = current_nll.target_gradient.detach().to(
        device="cpu", dtype=torch.float64
    )
    request_count = shape[1]
    request_gradient64 = batched_gradient64 * request_count
    q_signed = -torch.sum(request_gradient64 * nominal_delta64, dim=0)
    q_batched = -torch.sum(batched_gradient64 * nominal_delta64, dim=0)
    if not torch.isfinite(q_signed).all():
        raise ODEBFContractError("P1R40 signed gain differs")
    gamma = torch.ones(request_count, dtype=torch.float64)
    positive = q_signed > 0.0
    gamma[positive] = torch.minimum(
        torch.ones_like(gamma[positive]),
        deficits[positive] / (q_signed[positive] + P1R40_VELOCITY_EPSILON),
    )
    if (
        not torch.isfinite(gamma).all()
        or bool(torch.any(gamma < 0.0))
        or bool(torch.any(gamma > 1.0))
    ):
        raise ODEBFContractError("P1R40 velocity scalar differs")

    scaled64_precast = current64 + nominal_delta64 * gamma.unsqueeze(0)
    scaled_next = scaled64_precast.to(dtype=torch.float32).contiguous()
    scaled64 = scaled_next.to(dtype=torch.float64)
    scaled_delta64 = scaled64 - current64
    nominal_column_sha = _column_sha256(nominal_next)
    scaled_column_sha = _column_sha256(scaled_next)
    slowed = tuple(bool(item < 1.0) for item in gamma)
    if any(
        slowed[index] and nominal_column_sha[index] == scaled_column_sha[index]
        for index in range(request_count)
    ):
        raise P1R40VelocityMechanismError(
            "P1R40 slowed target bytes did not change"
        )

    lag64 = current64 - terminal64
    remaining = P1R24_K - step_index
    required64 = scaled_delta64 + lag64 / remaining
    required_model = required64.to(dtype=torch.float32).contiguous()
    write_velocity = (required_model / P1R24_H).contiguous()
    h_residual = float(
        torch.max(torch.abs(P1R24_H * write_velocity - required_model))
    )
    if h_residual > P1R24_NUMERICAL_EPSILON:
        raise ODEBFContractError("P1R40 inherited target coordinate differs")
    rho_signed = float(-torch.sum(batched_gradient64 * required64))
    alpha_target_signed = float(-torch.sum(batched_gradient64 * scaled_delta64))
    de_mean_residual = float(
        torch.max(torch.abs(q_signed - request_count * q_batched))
    )
    nominal_norms = tuple(
        float(item) for item in torch.linalg.vector_norm(nominal_delta64, dim=0)
    )
    scaled_norms = tuple(
        float(item) for item in torch.linalg.vector_norm(scaled_delta64, dim=0)
    )
    gamma_values = tuple(float(item) for item in gamma)
    deficit_values = tuple(float(item) for item in deficits)
    q_values = tuple(float(item) for item in q_signed)
    source_receipt = dict(proposal.receipt)
    source_receipt_sha = source_receipt.get("identity_sha256")
    payload: dict[str, Any] = {
        **source_receipt,
        "schema": "ode-edit-s05-p1r40-semantic-deficit-velocity-decay/v1",
        "instruction_id": P1R40_INSTRUCTION_ID,
        "method_id": P1R40_METHOD_ID,
        "p1r38_nominal_proposal_receipt_sha256": source_receipt_sha,
        "k": int(step_index),
        "pure_target_new_deficit_by_request": list(deficit_values),
        "pure_target_new_deficit_summary": _summary(deficit_values),
        "semantic_deficit_kl_influence_count": 0,
        "semantic_deficit_decay_influence_count": 0,
        "semantic_deficit_threshold_subtraction_count": 0,
        "batch_demean_factor": request_count,
        "batched_pure_target_gradient_sha256": tensor_sha256(batched_gradient64),
        "request_pure_target_gradient_sha256": tensor_sha256(request_gradient64),
        "q_signed_batched_gradient_by_request": [float(item) for item in q_batched],
        "q_signed_by_request": list(q_values),
        "q_positive_by_request": [max(item, 0.0) for item in q_values],
        "q_batch_demean_identity_max_abs": de_mean_residual,
        "gamma_by_request": list(gamma_values),
        "gamma_summary": _summary(gamma_values),
        "gamma_lt_one_mask": list(slowed),
        "gamma_lt_one_count": sum(slowed),
        "gamma_nll_bin_distribution": _nll_bin_payload(deficit_values, gamma_values),
        "nominal_target_next_sha256": tensor_sha256(nominal_next),
        "scaled_target_next_sha256": tensor_sha256(scaled_next),
        "nominal_target_column_sha256": nominal_column_sha,
        "scaled_target_column_sha256": scaled_column_sha,
        "nominal_scaled_hash_distinct_count": sum(
            nominal_column_sha[index] != scaled_column_sha[index]
            for index in range(request_count)
        ),
        "nominal_displacement_norm_by_request": list(nominal_norms),
        "scaled_displacement_norm_by_request": list(scaled_norms),
        "nominal_displacement_sha256": tensor_sha256(nominal_delta64),
        "scaled_displacement_sha256": tensor_sha256(scaled_delta64),
        "target_next_sha256": tensor_sha256(scaled_next),
        "target_displacement_sha256": tensor_sha256(scaled_delta64),
        "required_displacement_sha256": tensor_sha256(required_model),
        "write_velocity_sha256": tensor_sha256(write_velocity),
        "rho_write_signed": rho_signed,
        "rho_write": max(rho_signed, 0.0),
        "alpha_target_signed": alpha_target_signed,
        "h_write_velocity_identity_max_abs": h_residual,
        "nominal_adam_cap_clamp_precedes_gamma": True,
        "gamma_target_displacement_application_count": 1,
        "gamma_target_to_writer_application_count": 0,
        "writer_remaining_division_change_count": 0,
        "semantic_debt_input_count": 0,
        "added_model_forward_count": 0,
        "added_backward_count": 0,
        "added_materialization_count": 0,
        "per_request_python_model_call_count": 0,
    }
    payload.pop("identity_sha256", None)
    payload["identity_sha256"] = canonical_hash(payload)
    scaled_step = replace(
        proposal.trial_step,
        target_next=scaled_next,
        target_displacement=scaled_delta64.to(dtype=torch.float32).contiguous(),
        required_displacement=required_model,
        write_velocity=write_velocity,
        rho_write_signed=rho_signed,
        rho_write=max(rho_signed, 0.0),
        alpha_target_signed=alpha_target_signed,
        receipt=payload,
    )
    return replace(proposal, trial_step=scaled_step, receipt=payload)


def select_p1r40_target_proposal(
    proposal: P1R38TargetProposal,
    state: P1R38AdamState,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    current_nll: ScalableObjectiveResult,
    trial_endpoint: ScalableObjectiveResult,
    *,
    step_index: int,
) -> P1R38SelectedTarget:
    """Reuse P1R38 semantic selection and bind the P1R40 identity."""

    selected = select_p1r38_target_proposal(
        proposal,
        state,
        current_target,
        current_terminal,
        current_nll,
        trial_endpoint,
        step_index=step_index,
    )
    p1r38_selection_sha = selected.receipt.get("identity_sha256")
    receipt: dict[str, Any] = {
        **dict(selected.receipt),
        "schema": "ode-edit-s05-p1r40-semantic-deficit-selection/v1",
        "instruction_id": P1R40_INSTRUCTION_ID,
        "method_id": P1R40_METHOD_ID,
        "p1r38_selection_receipt_sha256": p1r38_selection_sha,
        "gamma_target_displacement_application_count": 1,
        "gamma_target_to_writer_application_count": 0,
        "selection_added_model_forward_count": 0,
        "selection_added_backward_count": 0,
    }
    receipt.pop("identity_sha256", None)
    receipt["identity_sha256"] = canonical_hash(receipt)
    return replace(
        selected,
        target_step=replace(selected.target_step, receipt=receipt),
        receipt=receipt,
    )


__all__ = [
    "P1R40_INSTRUCTION_ID",
    "P1R40_METHOD_ID",
    "P1R40_NLL_BINS",
    "P1R40_VELOCITY_EPSILON",
    "P1R40VelocityMechanismError",
    "apply_p1r40_semantic_deficit_velocity_decay",
    "select_p1r40_target_proposal",
]
