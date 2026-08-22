"""Thin phase and receding-horizon contracts around the P4 Euler solver."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p4_euler_cache_transaction import P4_EULER_OUTER_STEPS
from .p4_euler_integrator import P4_EULER_INSTRUCTION_ID


P4_EULER_CAUSAL_ARMS = ("A+", "A±")
P4_EULER_TARGET_ONLY_ARMS = ("Z+", "Z±")


def build_receding_horizon_write_target(
    current_terminal: torch.Tensor,
    local_target: torch.Tensor,
    *,
    outer_step_index: int,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    if (
        current_terminal.dtype is not torch.float32
        or local_target.dtype is not torch.float32
        or current_terminal.ndim != 2
        or current_terminal.shape != local_target.shape
        or outer_step_index < 0
        or outer_step_index >= P4_EULER_OUTER_STEPS
        or not bool(torch.isfinite(current_terminal).all())
        or not bool(torch.isfinite(local_target).all())
    ):
        raise ODEBFContractError("P4 Euler receding-horizon geometry differs")
    remaining = P4_EULER_OUTER_STEPS - outer_step_index
    gain = 1.0 / remaining
    write_target = (
        current_terminal + gain * (local_target - current_terminal)
    ).contiguous()
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-receding-horizon-write-gain/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "outer_step_index": outer_step_index,
        "K": P4_EULER_OUTER_STEPS,
        "lambda_k": gain,
        "formula": "y_k+lambda_k*(z_star_k-y_k)",
        "name": "RECEDING_HORIZON_WRITE_GAIN",
        "fixed_endpoint_waypoint_claim_count": 0,
        "y_K_equals_z_star_claim_count": 0,
        "current_terminal_sha256": tensor_sha256(current_terminal),
        "local_target_sha256": tensor_sha256(local_target),
        "write_target_sha256": tensor_sha256(write_target),
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return write_target, payload


def verify_euler_outer_refresh(
    rows: Sequence[Mapping[str, Any]], *, arm: str
) -> Mapping[str, Any]:
    if arm not in P4_EULER_CAUSAL_ARMS or len(rows) != P4_EULER_OUTER_STEPS:
        raise ODEBFContractError("P4 Euler outer refresh inventory differs")
    for index, row in enumerate(rows):
        required_one = (
            "current_terminal_refresh_count",
            "current_key_refresh_count",
            "current_residual_refresh_count",
            "kl_teacher_capture_count",
            "target_geometry_refresh_count",
            "alpha_solve_refresh_count",
            "official_writer_apply_count",
        )
        if (
            row.get("outer_step_index") != index
            or any(row.get(name) != 1 for name in required_one)
            or row.get("fresh_target_reset_count") != 1
            or row.get("warm_start_count") != 0
            or row.get("native_compute_z_call_count") != 0
            or row.get("heldout_access_count") != 0
        ):
            raise ODEBFContractError("P4 Euler current-W refresh contract differs")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-current-state-refresh/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "arm": arm,
        "outer_step_count": P4_EULER_OUTER_STEPS,
        "fresh_target_reset_count": P4_EULER_OUTER_STEPS,
        "warm_start_count": 0,
        "current_W_refresh_count": P4_EULER_OUTER_STEPS,
        "kl_teacher_capture_count": P4_EULER_OUTER_STEPS,
        "official_writer_apply_count": P4_EULER_OUTER_STEPS,
        "native_compute_z_call_count": 0,
        "heldout_decision_influence_count": 0,
        "writer_entrypoint": (
            "easyeditor.models.alphaedit.AlphaEdit_main.apply_AlphaEdit_to_model"
        ),
        "writer_mode": "OFFICIAL_ALPHAEDIT_SEQUENTIAL_PROVIDED_Z_HOOK",
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def build_za_phase_contract(*, arm: str, local_solve_count: int) -> Mapping[str, Any]:
    if arm not in P4_EULER_TARGET_ONLY_ARMS or local_solve_count != 1:
        raise ODEBFContractError("P4 Euler ZA phase contract differs")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-za-phase/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "arm": arm,
        "W": "W0_FIXED",
        "local_M_step_solve_count": 1,
        "outer_k8_repeat_count": 0,
        "writer_apply_count": 0,
        "native_compute_z_call_count": 0,
        "heldout_decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "P4_EULER_CAUSAL_ARMS",
    "P4_EULER_TARGET_ONLY_ARMS",
    "build_receding_horizon_write_target",
    "build_za_phase_contract",
    "verify_euler_outer_refresh",
]
