"""P3R1 K8 finite-horizon waypoint schedule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256


OUTER_COUNT = 8


@dataclass(frozen=True, slots=True)
class FiniteHorizonWaypoint:
    waypoint: torch.Tensor
    residual: torch.Tensor
    lambda_k: float
    receipt: Mapping[str, Any]


def finite_horizon_waypoint(
    current_terminal: torch.Tensor,
    oracle_target: torch.Tensor,
    *,
    outer_step_index: int,
) -> FiniteHorizonWaypoint:
    if not 0 <= outer_step_index < OUTER_COUNT:
        raise ODEBFContractError("P3R1 finite-horizon outer index differs")
    if (
        current_terminal.shape != oracle_target.shape
        or current_terminal.ndim != 2
        or current_terminal.dtype is not torch.float32
        or oracle_target.dtype is not torch.float32
    ):
        raise ODEBFContractError("P3R1 finite-horizon geometry/dtype differs")
    terminal64 = current_terminal.detach().to(device="cpu", dtype=torch.float64)
    oracle64 = oracle_target.detach().to(device="cpu", dtype=torch.float64)
    residual64 = oracle64 - terminal64
    lambda_k = 1.0 / float(OUTER_COUNT - outer_step_index)
    waypoint = (terminal64 + lambda_k * residual64).to(dtype=torch.float32).contiguous()
    if not torch.isfinite(waypoint).all():
        raise ODEBFStateError("P3R1 finite-horizon waypoint is nonfinite")
    realized = waypoint.to(dtype=torch.float64) - terminal64
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p3r1-finite-horizon-waypoint/v1",
        "outer_step_index": outer_step_index,
        "outer_count": OUTER_COUNT,
        "lambda": lambda_k,
        "lambda_source": "1/(K-k)",
        "oracle_target_sha256": tensor_sha256(oracle_target),
        "current_terminal_sha256": tensor_sha256(current_terminal),
        "oracle_residual_sha256": tensor_sha256(residual64),
        "oracle_residual_norm": float(torch.linalg.vector_norm(residual64)),
        "waypoint_sha256": tensor_sha256(waypoint),
        "waypoint_residual_norm": float(torch.linalg.vector_norm(realized)),
        "debt_input_count": 0,
        "lag_carry_count": 0,
        "remaining_division_count": 1,
        "second_remaining_division_count": 0,
        "early_stop_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return FiniteHorizonWaypoint(
        waypoint,
        residual64.to(dtype=torch.float32).contiguous(),
        lambda_k,
        payload,
    )


__all__ = ["FiniteHorizonWaypoint", "OUTER_COUNT", "finite_horizon_waypoint"]
