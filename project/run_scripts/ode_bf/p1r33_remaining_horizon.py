"""P1R24-compatible full-current-residual remaining-horizon coordinate policy."""

from __future__ import annotations

from typing import Any

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import (
    P1R24_H,
    P1R24_K,
    P1R24_NUMERICAL_EPSILON,
    P1R24WriteCoordinate,
)


P1R33_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R33-P1R24-FULL-RESIDUAL-REMAINING-HORIZON-V1"
)
P1R33_METHOD_ID = "P1R33-P1R24-FULL-RESIDUAL-REMAINING-HORIZON-V1"


def _max_abs(value: torch.Tensor) -> float:
    return float(torch.max(torch.abs(value))) if value.numel() else 0.0


def _norms_by_request(value: torch.Tensor) -> list[float]:
    return [float(item) for item in torch.linalg.vector_norm(value, dim=0)]


def p1r33_full_residual_remaining_horizon_coordinate(
    target_next64: torch.Tensor,
    current_target64: torch.Tensor,
    current_terminal64: torch.Tensor,
    displacement: torch.Tensor,
    step_index: int,
) -> P1R24WriteCoordinate:
    """Compute (target_next-current_terminal)/(K-k) exactly once."""

    if step_index < 0 or step_index >= P1R24_K:
        raise ODEBFContractError("P1R33 target step index differs")
    if not (
        target_next64.shape
        == current_target64.shape
        == current_terminal64.shape
        == displacement.shape
    ):
        raise ODEBFContractError("P1R33 target coordinate shape differs")
    if any(
        value.device.type != "cpu" or value.dtype != torch.float64
        for value in (
            target_next64,
            current_target64,
            current_terminal64,
            displacement,
        )
    ):
        raise ODEBFContractError("P1R33 target coordinate dtype differs")

    # The controlling order is authoritative: form the full residual first,
    # then perform exactly one remaining-horizon division.
    full_current_residual = (target_next64 - current_terminal64).contiguous()
    remaining_steps = P1R24_K - step_index
    required_analytic = (
        full_current_residual / float(remaining_steps)
    ).contiguous()
    required_model = required_analytic.to(dtype=torch.float32).contiguous()
    write_velocity = (required_model / P1R24_H).contiguous()

    lag = current_target64 - current_terminal64
    full_identity = _max_abs(
        full_current_residual - (displacement + lag)
    )
    remaining_identity = _max_abs(
        required_analytic * float(remaining_steps) - full_current_residual
    )
    h_identity = _max_abs(P1R24_H * write_velocity - required_model)
    analytic_to_model = _max_abs(
        required_model.to(torch.float64) - required_analytic
    )
    if max(full_identity, remaining_identity, h_identity) > P1R24_NUMERICAL_EPSILON:
        raise ODEBFContractError("P1R33 remaining-horizon identity differs")

    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r33-target-write-coordinate/v1",
        "coordinate_policy": "FULL_CURRENT_RESIDUAL_REMAINING_HORIZON",
        "remaining_steps": remaining_steps,
        "full_current_residual_sha256": tensor_sha256(full_current_residual),
        "full_current_residual_norm_by_request": _norms_by_request(
            full_current_residual
        ),
        "required_displacement_sha256": tensor_sha256(required_model),
        "analytic_required_displacement_sha256": tensor_sha256(
            required_analytic
        ),
        "required_displacement_norm_by_request": _norms_by_request(
            required_model.to(torch.float64)
        ),
        "write_velocity_sha256": tensor_sha256(write_velocity),
        "full_residual_identity_max_abs": full_identity,
        "remaining_horizon_identity_max_abs": remaining_identity,
        "h_write_velocity_identity_max_abs": h_identity,
        "identity_max_abs_residual": h_identity,
        "analytic_to_model_cast_max_abs": analytic_to_model,
        "field_coordinate": "ACTIVATION_VELOCITY",
        "residual_presplit_count": 0,
        "full_current_residual_count": 1,
        "remaining_horizon_division_count": 1,
        "semantic_debt_input_count": 0,
        "explicit_lag_readdition_count": 0,
        "physical_h_application_count": 1,
        "second_h_application_count": 0,
        "second_remaining_division_count": 0,
        "residual_attenuation_by_barrier_count": 0,
        "instruction_id": P1R33_INSTRUCTION_ID,
        "method_id": P1R33_METHOD_ID,
    }
    receipt["coordinate_receipt_sha256"] = canonical_hash(receipt)
    return P1R24WriteCoordinate(
        required_analytic,
        required_model,
        write_velocity,
        receipt,
    )


__all__ = [
    "P1R33_INSTRUCTION_ID",
    "P1R33_METHOD_ID",
    "p1r33_full_residual_remaining_horizon_coordinate",
]
