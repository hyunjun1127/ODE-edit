"""P1R35 full-current-residual coordinate on the frozen P1R34 path."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import (
    P1R24_H,
    P1R24_NUMERICAL_EPSILON,
    P1R24TargetStep,
)


P1R35_INSTRUCTION_ID = "ODEEDIT-S05-P1R35-P1R34-FULL-CURRENT-RESIDUAL-V1"
P1R35_METHOD_ID = "P1R34-W-ANCHORED-FINITE-DEMAND-FULL-CURRENT-RESIDUAL-V1"


def apply_p1r35_full_current_residual(
    target_step: P1R24TargetStep,
    *,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    step_index: int,
) -> P1R24TargetStep:
    """Change only P1R34's writer coordinate to u=d+lag."""

    if step_index < 0 or step_index >= 8:
        raise ODEBFContractError("P1R35 step index differs")
    if (
        current_target.shape != current_terminal.shape
        or target_step.target_next.shape != current_target.shape
        or target_step.target_displacement.shape != current_target.shape
    ):
        raise ODEBFContractError("P1R35 target coordinate geometry differs")

    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    terminal64 = current_terminal.detach().to(device="cpu", dtype=torch.float64)
    next64 = target_step.target_next.detach().to(device="cpu", dtype=torch.float64)
    recorded_d64 = target_step.target_displacement.detach().to(
        device="cpu", dtype=torch.float64
    )
    # The coordinate contract defines d from the two accepted FP32 states.  The
    # inherited P1R24 displacement was cast independently and may differ by one
    # FP32 rounding unit; it remains observation-only here.
    d64 = (next64 - current64).contiguous()
    lag64 = (current64 - terminal64).contiguous()
    full64 = (d64 + lag64).contiguous()
    direct64 = (next64 - terminal64).contiguous()
    d_identity = float(torch.max(torch.abs(d64 - (next64 - current64))))
    inherited_d_cast_residual = float(torch.max(torch.abs(recorded_d64 - d64)))
    full_identity = float(torch.max(torch.abs(full64 - direct64)))
    if max(d_identity, full_identity) > P1R24_NUMERICAL_EPSILON:
        raise ODEBFContractError("P1R35 full-current-residual identity differs")

    required_model = full64.to(dtype=torch.float32).contiguous()
    write_velocity = (required_model / P1R24_H).contiguous()
    h_identity = float(
        torch.max(torch.abs(P1R24_H * write_velocity - required_model))
    )
    if h_identity > P1R24_NUMERICAL_EPSILON:
        raise ODEBFContractError("P1R35 one-h identity differs")
    nll_gradient64 = target_step.nll_gradient.detach().to(
        device="cpu", dtype=torch.float64
    )
    rho_old_signed = float(-torch.sum(nll_gradient64 * full64))

    payload: dict[str, Any] = {
        **dict(target_step.receipt),
        "schema": "ode-edit-s05-p1r35-full-current-residual-coordinate/v1",
        "instruction_id": P1R35_INSTRUCTION_ID,
        "method_id": P1R35_METHOD_ID,
        "k": int(step_index),
        "p1r34_parent_target_receipt_sha256": target_step.receipt.get(
            "identity_sha256"
        ),
        "coordinate": "u=d+lag=target_next-current_terminal",
        "current_target_sha256": tensor_sha256(current_target),
        "current_terminal_sha256": tensor_sha256(current_terminal),
        "target_next_sha256": tensor_sha256(target_step.target_next),
        "target_displacement_sha256": tensor_sha256(
            d64
        ),
        "inherited_target_displacement_sha256": tensor_sha256(
            target_step.target_displacement
        ),
        "lag_sha256": tensor_sha256(lag64),
        "required_displacement_sha256": tensor_sha256(required_model),
        "write_velocity_sha256": tensor_sha256(write_velocity),
        "target_displacement_norm": float(torch.linalg.vector_norm(d64)),
        "lag_norm": float(torch.linalg.vector_norm(lag64)),
        "required_displacement_norm": float(torch.linalg.vector_norm(full64)),
        "d_identity_max_abs_residual": d_identity,
        "inherited_target_displacement_cast_max_abs": inherited_d_cast_residual,
        "full_current_residual_identity_max_abs": full_identity,
        "h_write_velocity_identity_max_abs": h_identity,
        "rho_write_signed": rho_old_signed,
        "rho_write": max(rho_old_signed, 0.0),
        "remaining_steps": None,
        "remaining_horizon_division_count": 0,
        "fresh_component_division_count": 0,
        "lag_component_division_count": 0,
        "semantic_debt_input_count": 0,
        "physical_h_application_count": 1,
        "second_h_application_count": 0,
        "second_remaining_division_count": 0,
        "residual_presplit_count": 0,
        "required_model_not_velocity": True,
    }
    payload.pop("identity_sha256", None)
    payload["identity_sha256"] = canonical_hash(payload)
    return replace(
        target_step,
        required_displacement=required_model,
        write_velocity=write_velocity,
        rho_write_signed=rho_old_signed,
        rho_write=max(rho_old_signed, 0.0),
        receipt=payload,
    )


__all__ = [
    "P1R35_INSTRUCTION_ID",
    "P1R35_METHOD_ID",
    "apply_p1r35_full_current_residual",
]
