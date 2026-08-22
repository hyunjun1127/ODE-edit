"""Pure raw projected-explicit-Euler target-state integrator for P4.

The module intentionally knows nothing about models, writers, evaluators,
optimizers, or experiment launchers.  A caller supplies one differentiable
per-request objective callback and an optional content-hash callback for the
frozen external state (the physical model in production).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Callable, Mapping, Protocol

import torch

from .contracts import ODEBFContractError, canonical_hash


P4_EULER_INSTRUCTION_ID = "ODEEDIT-S05-P4-EULER-PROJECTED-SEMANTIC-ODE-V1"
P4_EULER_METHOD_ID = "P4-EULER-PROJECTED-SEMANTIC-ODE-V1"


@dataclass(frozen=True, slots=True)
class EulerObjectiveEvaluation:
    """Differentiable per-request objective and detached factual telemetry."""

    per_request_objective: torch.Tensor
    telemetry: Mapping[str, Any]


class EulerObjectiveCallback(Protocol):
    def __call__(self, target_state: torch.Tensor) -> EulerObjectiveEvaluation: ...


@dataclass(frozen=True, slots=True)
class ProjectedEulerResult:
    final_state: torch.Tensor
    step_receipts: tuple[Mapping[str, Any], ...]
    receipt: Mapping[str, Any]


def _validate_state(name: str, value: torch.Tensor) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype is not torch.float32
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or not bool(torch.isfinite(value).all())
    ):
        raise ODEBFContractError(f"P4 Euler {name} state differs")


def _validate_sha(value: str, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ODEBFContractError(f"P4 Euler {name} content identity differs")
    return value


def _float_list(value: torch.Tensor) -> list[float]:
    flat = value.detach().to(device="cpu", dtype=torch.float32).flatten()
    if not bool(torch.isfinite(flat).all()):
        raise ODEBFContractError("P4 Euler telemetry is nonfinite")
    return [float(item) for item in flat]


def _tensor_sha256(value: torch.Tensor) -> str:
    logical = value.detach().to(device="cpu").contiguous().view(-1)
    return hashlib.sha256(
        logical.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def project_request_columns_to_ball(
    state: torch.Tensor,
    *,
    origin: torch.Tensor,
    radius_by_request: torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Project each request column onto its fixed outer-local L2 ball."""

    _validate_state("projection candidate", state)
    _validate_state("projection origin", origin)
    if state.shape != origin.shape:
        raise ODEBFContractError("P4 Euler projection geometry differs")
    if (
        not isinstance(radius_by_request, torch.Tensor)
        or radius_by_request.dtype is not torch.float32
        or radius_by_request.ndim != 1
        or radius_by_request.shape[0] != state.shape[1]
        or not bool(torch.isfinite(radius_by_request).all())
        or bool((radius_by_request < 0.0).any())
        or radius_by_request.device != state.device
    ):
        raise ODEBFContractError("P4 Euler projection radius differs")
    delta = state - origin
    pre_norm = torch.linalg.vector_norm(delta, dim=0)
    safe_norm = torch.clamp(pre_norm, min=torch.finfo(torch.float32).tiny)
    scale = torch.minimum(
        torch.ones_like(pre_norm), radius_by_request / safe_norm
    )
    projected = (origin + delta * scale.unsqueeze(0)).contiguous()
    post_norm = torch.linalg.vector_norm(projected - origin, dim=0)
    removed = state - projected
    removed_norm = torch.linalg.vector_norm(removed, dim=0)
    telemetry = {
        "pre_norm": pre_norm,
        "post_norm": post_norm,
        "scale": scale,
        "hit": pre_norm > radius_by_request,
        "removed_norm": removed_norm,
        "removed_energy": torch.sum(removed.square(), dim=0),
    }
    return projected, telemetry


def _boundary_components(
    field: torch.Tensor, state: torch.Tensor, origin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = state - origin
    norm = torch.linalg.vector_norm(delta, dim=0)
    direction = torch.zeros_like(delta)
    nonzero = norm > 0.0
    direction[:, nonzero] = delta[:, nonzero] / norm[nonzero].unsqueeze(0)
    radial = torch.sum(field * direction, dim=0)
    outward = torch.clamp(radial, min=0.0)
    field_norm_sq = torch.sum(field.square(), dim=0)
    tangential = torch.sqrt(torch.clamp(field_norm_sq - radial.square(), min=0.0))
    return outward, tangential


def run_raw_projected_euler(
    initial_state: torch.Tensor,
    *,
    origin: torch.Tensor,
    radius_by_request: torch.Tensor,
    target_horizon: float | None,
    microsteps: int,
    objective_callback: EulerObjectiveCallback,
    frozen_content_sha256: Callable[[], str] | None = None,
    outer_step_index: int | None = None,
) -> ProjectedEulerResult:
    """Run all fixed raw Euler microsteps and select only the final iterate."""

    _validate_state("initial", initial_state)
    _validate_state("origin", origin)
    if initial_state.shape != origin.shape or not torch.equal(initial_state, origin):
        raise ODEBFContractError("P4 Euler fresh outer reset z_(k,0)=y_k differs")
    if initial_state.device != origin.device:
        raise ODEBFContractError("P4 Euler target devices differ")
    if isinstance(microsteps, bool) or not isinstance(microsteps, int) or microsteps <= 0:
        raise ODEBFContractError("P4 Euler M differs")
    if target_horizon is None:
        raise ODEBFContractError("P4 Euler T_z is unset (fail-closed)")
    try:
        horizon = float(target_horizon)
    except (TypeError, ValueError) as error:
        raise ODEBFContractError("P4 Euler T_z differs") from error
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ODEBFContractError("P4 Euler T_z differs")
    if outer_step_index is not None and (
        isinstance(outer_step_index, bool)
        or not isinstance(outer_step_index, int)
        or outer_step_index < 0
        or outer_step_index >= 8
    ):
        raise ODEBFContractError("P4 Euler outer index differs")
    if torch.is_autocast_enabled():
        raise ODEBFContractError("P4 Euler autocast must be disabled")
    project_request_columns_to_ball(
        initial_state, origin=origin, radius_by_request=radius_by_request
    )

    step_size = horizon / microsteps
    step_scalar = torch.tensor(
        step_size, device=initial_state.device, dtype=torch.float32
    )
    frozen_before = None
    if frozen_content_sha256 is not None:
        frozen_before = _validate_sha(frozen_content_sha256(), "frozen W entry")

    current = initial_state.detach().clone()
    rows: list[Mapping[str, Any]] = []
    started = time.perf_counter()
    for index in range(microsteps):
        if frozen_content_sha256 is not None:
            current_frozen = _validate_sha(
                frozen_content_sha256(), f"frozen W microstep {index}"
            )
            if current_frozen != frozen_before:
                raise ODEBFContractError("P4 Euler physical W changed during inner")
        state = current.detach().requires_grad_(True)
        evaluation = objective_callback(state)
        per_request = evaluation.per_request_objective
        if (
            not isinstance(per_request, torch.Tensor)
            or per_request.dtype is not torch.float32
            or per_request.ndim != 1
            or per_request.shape[0] != state.shape[1]
            or not bool(torch.isfinite(per_request.detach()).all())
        ):
            raise ODEBFContractError("P4 Euler per-request objective differs")
        gradient = torch.autograd.grad(
            per_request.sum(),
            state,
            create_graph=False,
            retain_graph=False,
        )[0]
        if (
            gradient.dtype is not torch.float32
            or gradient.shape != state.shape
            or not bool(torch.isfinite(gradient).all())
        ):
            raise ODEBFContractError("P4 Euler target gradient differs")
        field = -gradient
        raw_update = step_scalar * field
        candidate = state.detach() + raw_update
        projected, projection = project_request_columns_to_ball(
            candidate, origin=origin, radius_by_request=radius_by_request
        )
        outward, tangential = _boundary_components(field, state.detach(), origin)
        field_norm = torch.linalg.vector_norm(field, dim=0)
        raw_update_norm = torch.linalg.vector_norm(raw_update, dim=0)
        zero_field = field_norm == 0.0
        row: dict[str, Any] = {
            "schema": "ode-edit-s05-p4-euler-microstep/v1",
            "microstep_index": index,
            "target_before_sha256": _tensor_sha256(state),
            "target_after_sha256": _tensor_sha256(projected),
            "objective_by_request": _float_list(per_request),
            "objective_telemetry": dict(evaluation.telemetry),
            "raw_field_norm_by_request": _float_list(field_norm),
            "raw_euler_displacement_by_request": _float_list(raw_update_norm),
            "pre_clamp_displacement_by_request": _float_list(projection["pre_norm"]),
            "post_clamp_displacement_by_request": _float_list(projection["post_norm"]),
            "clamp_hit_by_request": [bool(item) for item in projection["hit"].tolist()],
            "clamp_ratio_by_request": _float_list(projection["scale"]),
            "clamp_removed_norm_by_request": _float_list(projection["removed_norm"]),
            "clamp_removed_energy_by_request": _float_list(projection["removed_energy"]),
            "boundary_outward_component_by_request": _float_list(outward),
            "boundary_tangential_component_by_request": _float_list(tangential),
            "zero_field_by_request": [bool(item) for item in zero_field.tolist()],
            "field_normalization_count": 0,
            "autograd_grad_call_count": 1,
            "update_count": int((~zero_field).sum().item()),
        }
        row["identity_sha256"] = canonical_hash(row)
        rows.append(row)
        current = projected.detach()

    frozen_after = frozen_before
    if frozen_content_sha256 is not None:
        frozen_after = _validate_sha(frozen_content_sha256(), "frozen W exit")
        if frozen_after != frozen_before:
            raise ODEBFContractError("P4 Euler physical W changed during inner")
    elapsed = time.perf_counter() - started
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-raw-projected-euler/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "method_id": P4_EULER_METHOD_ID,
        "outer_step_index": outer_step_index,
        "formula": "z_(m+1)=Projection_B(origin,r)[z_m+h*(-grad J(z_m;W_k))]",
        "target_horizon": horizon,
        "microsteps": microsteps,
        "step_size": step_size,
        "initial_state_sha256": _tensor_sha256(initial_state),
        "origin_sha256": _tensor_sha256(origin),
        "radius_by_request": _float_list(radius_by_request),
        "final_state_sha256": _tensor_sha256(current),
        "selected_iterate_index": microsteps,
        "selected_iterate": "FINAL_ONLY",
        "fresh_outer_reset": True,
        "warm_start_count": 0,
        "logical_field_evaluation_count": microsteps,
        "autograd_grad_call_count": microsteps,
        "optimizer": "NONE",
        "optimizer_object_count": 0,
        "adam_state_count": 0,
        "sgd_state_count": 0,
        "moment_count": 0,
        "bias_correction_count": 0,
        "parameter_gradient_count": 0,
        "loss_backward_count": 0,
        "early_stop_count": 0,
        "accept_reject_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "fallback_count": 0,
        "best_iterate_decision_influence_count": 0,
        "field_normalization_count": 0,
        "clamp_origin_change_count": 0,
        "clamp_radius_change_count": 0,
        "dtype": "torch.float32",
        "autocast_enabled": False,
        "quantization_enabled": False,
        "frozen_content_entry_sha256": frozen_before,
        "frozen_content_exit_sha256": frozen_after,
        "frozen_content_change_count": 0,
        "wall_time_seconds": elapsed,
        "step_identity_sha256": canonical_hash(
            [row["identity_sha256"] for row in rows]
        ),
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return ProjectedEulerResult(current, tuple(rows), payload)


__all__ = [
    "EulerObjectiveCallback",
    "EulerObjectiveEvaluation",
    "P4_EULER_INSTRUCTION_ID",
    "P4_EULER_METHOD_ID",
    "ProjectedEulerResult",
    "project_request_columns_to_ball",
    "run_raw_projected_euler",
]
