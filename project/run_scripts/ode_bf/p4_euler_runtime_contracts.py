"""Model-free runtime receipts shared by P4 Euler ZA/ZB launchers."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p4_euler_integrator import P4_EULER_INSTRUCTION_ID


FULL_FP32_ROLES = (
    "model_parameters",
    "activation",
    "target_z",
    "projector",
    "cache",
    "solve_operands",
    "final_update",
)


def tensor_inventory_content_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    if not tensors:
        raise ODEBFContractError("P4 Euler tensor inventory is empty")
    rows: list[dict[str, Any]] = []
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise ODEBFContractError("P4 Euler tensor inventory member differs")
        rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": tensor_sha256(tensor),
            }
        )
    return canonical_hash(rows)


def validate_full_fp32_inventory(
    tensors_by_role: Mapping[str, Sequence[torch.Tensor]],
    *,
    autocast_enabled: bool,
    quantization_enabled: bool,
) -> Mapping[str, Any]:
    if tuple(sorted(tensors_by_role)) != tuple(sorted(FULL_FP32_ROLES)):
        raise ODEBFContractError("P4 Euler FP32 tensor roles differ")
    if autocast_enabled or quantization_enabled:
        raise ODEBFContractError("P4 Euler autocast/quantization must be disabled")
    role_rows: dict[str, Any] = {}
    parameter_gradient_count = 0
    for role in FULL_FP32_ROLES:
        tensors = tuple(tensors_by_role[role])
        if not tensors:
            raise ODEBFContractError(f"P4 Euler FP32 {role} inventory is empty")
        identities: list[str] = []
        for tensor in tensors:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype is not torch.float32
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ODEBFContractError(f"P4 Euler FP32 {role} tensor differs")
            if role == "model_parameters" and getattr(tensor, "grad", None) is not None:
                parameter_gradient_count += 1
            identities.append(tensor_sha256(tensor))
        role_rows[role] = {
            "tensor_count": len(tensors),
            "member_sha256": identities,
            "dtype": "torch.float32",
        }
    if parameter_gradient_count != 0:
        raise ODEBFContractError("P4 Euler model parameter gradient exists")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-full-fp32-inventory/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "roles": role_rows,
        "tensor_state_solver": "FULL_FP32",
        "parameter_gradient_count": 0,
        "autocast_enabled": False,
        "quantization_enabled": False,
        "tf32_disabled_claim": False,
        "report_wording": "FP32 tensor-state/solver",
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def build_compute_telemetry(
    *,
    logical_field_evaluations: int,
    actual_model_forwards: int,
    actual_autograd_grad_calls: int,
    new_suffix_processed_tokens: int,
    old_suffix_processed_tokens: int,
    kl_teacher_forwards: int,
    kl_evaluation_forwards: int,
    writer_applies: int,
    layer_solves: int,
    key_recaptures: int,
    wall_time_seconds: float,
    gpu_time_seconds: float,
    peak_gpu_memory_bytes: int,
    cache_appends: int,
    heldout_terminal_only: bool,
) -> Mapping[str, Any]:
    integer_values = {
        "logical_field_evaluations": logical_field_evaluations,
        "actual_model_forwards": actual_model_forwards,
        "actual_autograd_grad_calls": actual_autograd_grad_calls,
        "new_suffix_processed_tokens": new_suffix_processed_tokens,
        "old_suffix_processed_tokens": old_suffix_processed_tokens,
        "kl_teacher_forwards": kl_teacher_forwards,
        "kl_evaluation_forwards": kl_evaluation_forwards,
        "writer_applies": writer_applies,
        "layer_solves": layer_solves,
        "key_recaptures": key_recaptures,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "cache_appends": cache_appends,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integer_values.values()
    ):
        raise ODEBFContractError("P4 Euler compute count differs")
    if (
        not math.isfinite(float(wall_time_seconds))
        or not math.isfinite(float(gpu_time_seconds))
        or wall_time_seconds < 0.0
        or gpu_time_seconds < 0.0
        or not heldout_terminal_only
    ):
        raise ODEBFContractError("P4 Euler compute timing/evaluator contract differs")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-compute-telemetry/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        **integer_values,
        "actual_loss_backward_calls": 0,
        "wall_time_seconds": float(wall_time_seconds),
        "gpu_time_seconds": float(gpu_time_seconds),
        "heldout_terminal_only": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "FULL_FP32_ROLES",
    "build_compute_telemetry",
    "tensor_inventory_content_sha256",
    "validate_full_fp32_inventory",
]
