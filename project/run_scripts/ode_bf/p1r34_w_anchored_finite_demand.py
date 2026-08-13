"""P1R34 finite applied-step writer demand on the frozen P1R24 target path."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import (
    P1R24_NUMERICAL_EPSILON,
    P1R24TargetStep,
)
from .scalable_batched_model import ScalableObjectiveResult


P1R34_INSTRUCTION_ID = "ODEEDIT-S05-P1R34-P1R24-W-ANCHORED-FINITE-DEMAND-V1"
P1R34_METHOD_ID = "P1R24-W-ANCHORED-FINITE-APPLIED-STEP-DEMAND-V1"


class P1R34DemandStatus(str, Enum):
    FINITE_SEMANTIC_WRITE = "FINITE_SEMANTIC_WRITE"
    NUMERICAL_ZERO_WRITE = "NUMERICAL_ZERO_WRITE"
    NON_SEMANTIC_TARGET_MOVE = "NON_SEMANTIC_TARGET_MOVE"


class P1R34NonSemanticTargetMove(ODEBFContractError):
    """Typed scientific boundary: the frozen P1R24 target move increases NLL."""


@dataclass(frozen=True, slots=True)
class P1R34FiniteDemand:
    status: P1R34DemandStatus
    rho_write: float | None
    rho_finite_signed: float
    rho_old_signed: float
    receipt: Mapping[str, Any]


def build_p1r34_finite_demand(
    *,
    step_index: int,
    l_base: float,
    endpoint: ScalableObjectiveResult,
    target_step: P1R24TargetStep,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
) -> P1R34FiniteDemand:
    """Replace only P1R24 writer demand with exact finite endpoint decrease."""

    scalars = (float(l_base), float(endpoint.loss), float(target_step.rho_write_signed))
    if not all(math.isfinite(item) for item in scalars):
        raise ODEBFContractError("P1R34 finite-demand scalar is nonfinite")
    if endpoint.target_gradient is not None or endpoint.backward_count != 0:
        raise ODEBFContractError("P1R34 endpoint forward added a backward")
    if target_step.required_displacement.shape != current_target.shape:
        raise ODEBFContractError("P1R34 required-model geometry differs")
    if current_terminal.shape != current_target.shape:
        raise ODEBFContractError("P1R34 current state geometry differs")

    signed = float(l_base - endpoint.loss)
    if signed < -P1R24_NUMERICAL_EPSILON:
        status = P1R34DemandStatus.NON_SEMANTIC_TARGET_MOVE
        decision: float | None = None
    elif signed <= 0.0:
        status = P1R34DemandStatus.NUMERICAL_ZERO_WRITE
        decision = 0.0
    else:
        status = P1R34DemandStatus.FINITE_SEMANTIC_WRITE
        decision = signed

    old = float(target_step.rho_write_signed)
    if abs(old) <= P1R24_NUMERICAL_EPSILON:
        ratio: float | None = None
        ratio_status = "OLD_DEMAND_ZERO_DENOMINATOR"
    else:
        ratio = signed / old
        ratio_status = "FINITE_RATIO"

    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    terminal64 = current_terminal.detach().to(device="cpu", dtype=torch.float64)
    d64 = target_step.target_displacement.detach().to(device="cpu", dtype=torch.float64)
    u64 = target_step.required_displacement.detach().to(device="cpu", dtype=torch.float64)
    lag64 = current64 - terminal64
    endpoint_state = terminal64 + u64
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r34-w-anchored-finite-demand/v1",
        "instruction_id": P1R34_INSTRUCTION_ID,
        "method_id": P1R34_METHOD_ID,
        "k": int(step_index),
        "status": status.value,
        "L_base": float(l_base),
        "L_endpoint": float(endpoint.loss),
        "L_base_source": "REUSED_AUTHORITATIVE_NOHOOK_W_ONLY_TARGET_NEW_NLL",
        "L_base_same_state_identity_residual": 0.0,
        "L_base_identity_tolerance": P1R24_NUMERICAL_EPSILON,
        "L_endpoint_per_request_values": list(endpoint.per_request_values),
        "rho_old_signed": old,
        "rho_finite_signed": signed,
        "rho_write": decision,
        "finite_over_old_ratio": ratio,
        "finite_over_old_ratio_status": ratio_status,
        "target_new_gradient_norm": float(torch.linalg.vector_norm(target_step.nll_gradient)),
        "z_minus_y_norm": float(torch.linalg.vector_norm(lag64)),
        "target_displacement_norm": float(torch.linalg.vector_norm(d64)),
        "lag_norm": float(torch.linalg.vector_norm(lag64)),
        "required_model_norm": float(torch.linalg.vector_norm(u64)),
        "required_model_sha256": tensor_sha256(target_step.required_displacement),
        "endpoint_state_sha256": tensor_sha256(endpoint_state),
        "endpoint_objective_sha256": endpoint.identity_sha256,
        "endpoint_forward_count": endpoint.model_forward_count,
        "endpoint_backward_count": endpoint.backward_count,
        "endpoint_processed_token_count": endpoint.processed_token_count,
        "request_order_sha256": endpoint.request_order_sha256,
        "residual_presplit_count": 0,
        "second_remaining_division_count": 0,
        "physical_h_application_count": 1,
        "semantic_debt_input_count": 0,
        "finite_demand_forward_count": endpoint.model_forward_count,
        "finite_demand_backward_count": 0,
        "required_model_not_velocity": True,
        "negative_numerical_tolerance_source": "P1R24_NUMERICAL_EPSILON",
        "negative_numerical_tolerance": P1R24_NUMERICAL_EPSILON,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return P1R34FiniteDemand(status, decision, signed, old, payload)


__all__ = [
    "P1R34DemandStatus",
    "P1R34FiniteDemand",
    "P1R34NonSemanticTargetMove",
    "P1R34_INSTRUCTION_ID",
    "P1R34_METHOD_ID",
    "build_p1r34_finite_demand",
]
