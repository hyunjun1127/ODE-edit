"""Scalar-only mechanism telemetry for Ordered Response-Barrier ODE-Edit."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch

from .contracts import ArmId, NumericalMethodBoundary, ResponseStatistics
from .semantic import SemanticObservation


def _norm(value: torch.Tensor) -> float:
    result = float(torch.linalg.vector_norm(value.detach().double()).item())
    if not math.isfinite(result):
        raise NumericalMethodBoundary("telemetry norm is non-finite")
    return result


def _mean_norm_columns(value: torch.Tensor) -> float:
    if value.ndim != 2:
        raise NumericalMethodBoundary("telemetry activation must be a matrix")
    result = float(torch.linalg.vector_norm(value.detach().double(), dim=0).mean().item())
    if not math.isfinite(result):
        raise NumericalMethodBoundary("telemetry column norm is non-finite")
    return result


@dataclass(frozen=True, slots=True)
class StepRecord:
    arm: str
    sweep: int
    layer: int
    visit_ordinal: int
    built_state_version: int
    resulting_state_version: int
    residual_denominator: int
    command_reference_state_version: int
    build_identity: str
    residual_sha256: str
    keys_sha256: str
    solver_identity: str
    factor_rank: int
    coefficient_u: float
    euler_alpha: float
    g: float | None
    r: float | None
    per_request_g: tuple[float | None, ...]
    per_request_r: tuple[float | None, ...]
    potential_before: float
    potential_after: float
    predicted_reduction: float | None
    actual_reduction: float
    discretization_defect: float
    entry_sublevel_violation: float
    residual_norm_mean_before: float
    residual_norm_mean_after: float
    response_norm_mean: float | None
    command_norm_mean: float
    writer_command_norm_mean: float
    predicted_transition_norm_mean: float | None
    actual_transition_norm_mean: float
    realization_error_norm_mean: float
    model_error_norm_mean: float | None
    per_request_D_R0: tuple[float | None, ...]
    per_request_D_model_R0: tuple[float | None, ...]
    per_request_q_res: tuple[float | None, ...]
    per_request_writer_command_R0_squared: tuple[float | None, ...]
    per_request_response_orthogonal_R0_squared: tuple[float | None, ...]
    per_request_zero_command_cross_response_R0_squared: tuple[float | None, ...]
    per_request_actual_potential_change: tuple[float | None, ...]
    per_request_actual_potential_worsened: tuple[bool | None, ...]
    per_request_metric_status: tuple[str, ...]
    direction_frobenius: float
    direction_frobenius_squared: float
    applied_update_frobenius: float
    applied_update_frobenius_squared: float
    resolution_stable_path_increment_frobenius_squared: float
    action_geometry_status: str
    native_creg_action_status: str
    semantic_all_strict: bool
    semantic_request_count: int
    semantic_strict_event_count: int
    semantic_event_count: int
    semantic_request_strict_count: int
    semantic_tie_count: int
    virtual_state_identity_sha256: str

    def __post_init__(self) -> None:
        finite = (
            self.coefficient_u,
            self.euler_alpha,
            self.potential_before,
            self.potential_after,
            self.actual_reduction,
            self.discretization_defect,
            self.entry_sublevel_violation,
            self.residual_norm_mean_before,
            self.residual_norm_mean_after,
            self.command_norm_mean,
            self.writer_command_norm_mean,
            self.actual_transition_norm_mean,
            self.realization_error_norm_mean,
            self.direction_frobenius,
            self.direction_frobenius_squared,
            self.applied_update_frobenius,
            self.applied_update_frobenius_squared,
            self.resolution_stable_path_increment_frobenius_squared,
        )
        optional = (self.g, self.r, self.predicted_reduction, self.response_norm_mean,
                    self.predicted_transition_norm_mean, self.model_error_norm_mean)
        if (
            self.arm not in {item.value for item in ArmId}
            or not all(math.isfinite(value) for value in finite)
            or not all(value is None or math.isfinite(value) for value in optional)
            or not 0.0 <= self.coefficient_u <= 1.0
            or self.euler_alpha < 0.0
            or len(self.virtual_state_identity_sha256) != 64
            or not (
                len(self.per_request_g)
                == len(self.per_request_r)
                == len(self.per_request_D_R0)
                == len(self.per_request_D_model_R0)
                == len(self.per_request_q_res)
                == len(self.per_request_writer_command_R0_squared)
                == len(self.per_request_response_orthogonal_R0_squared)
                == len(self.per_request_zero_command_cross_response_R0_squared)
                == len(self.per_request_actual_potential_change)
                == len(self.per_request_actual_potential_worsened)
                == len(self.per_request_metric_status)
                == self.semantic_request_count
            )
            or not all(
                value is None or math.isfinite(value)
                for values in (
                    self.per_request_g,
                    self.per_request_r,
                    self.per_request_D_R0,
                    self.per_request_D_model_R0,
                    self.per_request_q_res,
                    self.per_request_writer_command_R0_squared,
                    self.per_request_response_orthogonal_R0_squared,
                    self.per_request_zero_command_cross_response_R0_squared,
                    self.per_request_actual_potential_change,
                )
                for value in values
            )
            or not all(
                value in {
                    "DEFINED",
                    "DEFINED_D_R0_ONLY_NO_RESPONSE",
                    "SKIPPED_ZERO_COMMAND",
                    "ZERO_GLOBAL_ACTION_RESPONSE_OBSERVED",
                    "ZERO_GLOBAL_ACTION_WITHOUT_RESPONSE",
                    "ZERO_COMMAND_CROSS_RESPONSE_OBSERVED",
                    "ZERO_COMMAND_NO_CROSS_RESPONSE",
                    "ZERO_ANCHOR_EXCLUDED_FROM_QRES",
                }
                for value in self.per_request_metric_status
            )
            or not all(
                value is None or isinstance(value, bool)
                for value in self.per_request_actual_potential_worsened
            )
            or not 0 <= self.semantic_request_strict_count <= self.semantic_request_count
            or self.action_geometry_status != "FROBENIUS_ONLY_CREG_NOT_BOUND"
            or self.native_creg_action_status != "TELEMETRY_WITHHELD"
            or isinstance(self.factor_rank, bool)
            or self.factor_rank <= 0
        ):
            raise NumericalMethodBoundary("step telemetry contract differs")


@dataclass(slots=True)
class ArmTelemetry:
    arm: str
    steps: list[StepRecord] = field(default_factory=list)
    first_hit: dict[str, int] | None = None
    entry_semantic: Mapping[str, Any] | None = None
    terminal_status: str | None = None
    terminal_semantic: Mapping[str, Any] | None = None
    zero_action_visit_count: int = 0
    nonzero_action_visit_count: int = 0
    factor_build_count: int = 0
    key_capture_count: int = 0
    terminal_capture_count: int = 0
    jvp_call_count: int = 0
    materialization_count: int = 0
    physical_write_count: int = 0
    history_append_count: int = 0
    terminal_net_frobenius: float = 0.0
    terminal_net_frobenius_squared: float = 0.0
    anchor_active_request_count: int = 0
    anchor_zero_request_count: int = 0
    anchor_zero_semantic_miss_count: int = 0

    def latch_hit(self, *, sweep: int, layer: int, state_version: int, prefix_length: int) -> None:
        if self.first_hit is None:
            self.first_hit = {
                "sweep": int(sweep),
                "layer": int(layer),
                "state_version": int(state_version),
                "prefix_length": int(prefix_length),
            }

    def payload(self) -> dict[str, Any]:
        if (
            not math.isfinite(self.terminal_net_frobenius)
            or not math.isfinite(self.terminal_net_frobenius_squared)
            or self.terminal_net_frobenius < 0.0
            or self.terminal_net_frobenius_squared < 0.0
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (
                    self.anchor_active_request_count,
                    self.anchor_zero_request_count,
                    self.anchor_zero_semantic_miss_count,
                )
            )
            or self.anchor_zero_semantic_miss_count > self.anchor_zero_request_count
        ):
            raise NumericalMethodBoundary("terminal net Frobenius telemetry is invalid")
        step_action_squared = sum(
            step.applied_update_frobenius_squared for step in self.steps
        )
        step_path_length = sum(step.applied_update_frobenius for step in self.steps)
        resolution_stable_path_squared = sum(
            step.resolution_stable_path_increment_frobenius_squared
            for step in self.steps
        )
        return {
            "schema": "orbode.arm-telemetry.v1",
            "arm": self.arm,
            "entry_semantic": dict(self.entry_semantic or {}),
            "first_hit": self.first_hit,
            "terminal_status": self.terminal_status,
            "terminal_semantic": dict(self.terminal_semantic or {}),
            "step_count": len(self.steps),
            "zero_action_visit_count": self.zero_action_visit_count,
            "nonzero_action_visit_count": self.nonzero_action_visit_count,
            "factor_build_count": self.factor_build_count,
            "key_capture_count": self.key_capture_count,
            "terminal_capture_count": self.terminal_capture_count,
            "jvp_call_count": self.jvp_call_count,
            "materialization_count": self.materialization_count,
            "physical_write_count": self.physical_write_count,
            "history_append_count": self.history_append_count,
            "sum_step_action_frobenius_squared": step_action_squared,
            "physical_euler_path_frobenius_length": step_path_length,
            "resolution_stable_path_frobenius": math.sqrt(
                resolution_stable_path_squared
            ),
            "resolution_stable_path_frobenius_squared": resolution_stable_path_squared,
            "terminal_net_frobenius": self.terminal_net_frobenius,
            "terminal_net_frobenius_squared": self.terminal_net_frobenius_squared,
            "anchor_active_request_count": self.anchor_active_request_count,
            "anchor_zero_request_count": self.anchor_zero_request_count,
            "anchor_zero_semantic_miss_count": self.anchor_zero_semantic_miss_count,
            "anchor_zero_semantic_miss_policy": "NONBLOCKING_SCIENTIFIC_OBSERVATION",
            "native_creg_action_status": "TELEMETRY_WITHHELD",
            "steps": [asdict(step) for step in self.steps],
        }


def semantic_payload(value: SemanticObservation) -> dict[str, Any]:
    return asdict(value)


def build_step_record(
    *,
    arm: ArmId,
    sweep: int,
    layer: int,
    visit_ordinal: int,
    built_state_version: int,
    resulting_state_version: int,
    residual_denominator: int,
    build_identity: str,
    residual_sha256: str,
    keys_sha256: str,
    solver_identity: str,
    factor_rank: int,
    u: float,
    h: float,
    entry_potential: float,
    potential_before: float,
    potential_after: float,
    target: torch.Tensor,
    command_residual: torch.Tensor,
    anchor_scales: torch.Tensor,
    anchor_active: torch.Tensor,
    terminal_before: torch.Tensor,
    terminal_after: torch.Tensor,
    direction_frobenius: float,
    response: torch.Tensor | None,
    statistics: ResponseStatistics | None,
    semantic: SemanticObservation,
    virtual_state_identity_sha256: str,
) -> StepRecord:
    residual_before = target - terminal_before
    residual_after = target - terminal_after
    actual_transition = terminal_after - terminal_before
    if command_residual.shape != residual_before.shape:
        raise NumericalMethodBoundary("command residual shape differs")
    alpha = h * u
    # The locked realization-debt command is always alpha times the *full
    # current* residual.  QCL's R/n (and JAC's sweep-entry R) remains a
    # distinct writer-command observation and must not redefine D_R0.
    command = alpha * residual_before
    writer_command = alpha * command_residual
    predicted = None if response is None else h * u * response
    realization_error = command - actual_transition
    model_error = None if predicted is None else actual_transition - predicted
    request_count = int(target.shape[1])
    scales = anchor_scales.detach().to(device="cpu", dtype=torch.float64).flatten()
    active = anchor_active.detach().to(device="cpu", dtype=torch.bool).flatten()
    if (
        len(semantic.request_strict) != request_count
        or scales.shape != (request_count,)
        or active.shape != (request_count,)
        or not bool(torch.isfinite(scales).all())
        or bool((scales < 0.0).any())
        or not torch.equal(active, scales > 0.0)
    ):
        raise NumericalMethodBoundary("request-level mechanism/semantic denominator differs")
    per_request_g = (
        tuple(None for _ in range(request_count))
        if statistics is None
        else statistics.per_request_g
    )
    per_request_r = (
        tuple(None for _ in range(request_count))
        if statistics is None
        else statistics.per_request_r
    )
    if len(per_request_g) != request_count or len(per_request_r) != request_count:
        raise NumericalMethodBoundary("request-level response denominator differs")
    per_request_D_R0: list[float | None] = []
    per_request_D_model_R0: list[float | None] = []
    per_request_q_res: list[float | None] = []
    per_request_writer_command_R0_squared: list[float | None] = []
    per_request_response_orthogonal_R0_squared: list[float | None] = []
    per_request_zero_command_cross_response_R0_squared: list[float | None] = []
    per_request_actual_potential_change: list[float | None] = []
    per_request_actual_potential_worsened: list[bool | None] = []
    per_request_metric_status: list[str] = []
    for request_ordinal in range(request_count):
        if not bool(active[request_ordinal]):
            per_request_D_R0.append(None)
            per_request_D_model_R0.append(None)
            per_request_q_res.append(None)
            per_request_writer_command_R0_squared.append(None)
            per_request_response_orthogonal_R0_squared.append(None)
            per_request_zero_command_cross_response_R0_squared.append(None)
            per_request_actual_potential_change.append(None)
            per_request_actual_potential_worsened.append(None)
            per_request_metric_status.append("ZERO_ANCHOR_EXCLUDED_FROM_QRES")
            continue
        scale = float(scales[request_ordinal].item())
        q_res = float(
            torch.linalg.vector_norm(residual_after[:, request_ordinal].double()).item()
            / scale
        )
        per_request_q_res.append(q_res)
        before_vector = residual_before[:, request_ordinal].double()
        after_vector = residual_after[:, request_ordinal].double()
        actual_change = float(
            0.5 * (torch.sum(after_vector.square()) - torch.sum(before_vector.square())).item()
            / (scale * scale)
        )
        per_request_actual_potential_change.append(actual_change)
        per_request_actual_potential_worsened.append(actual_change > 0.0)
        writer_command_sq = float(
            torch.sum(writer_command[:, request_ordinal].double().square()).item()
            / (scale * scale)
        )
        per_request_writer_command_R0_squared.append(writer_command_sq)
        if response is None:
            per_request_response_orthogonal_R0_squared.append(None)
        else:
            response_vector = response[:, request_ordinal].double()
            residual_norm_sq = float(torch.sum(before_vector.square()).item())
            if residual_norm_sq == 0.0:
                orthogonal = response_vector
            else:
                projection = (
                    torch.dot(response_vector, before_vector) / residual_norm_sq
                ) * before_vector
                orthogonal = response_vector - projection
            orthogonal_sq = float(torch.sum(orthogonal.square()).item() / (scale * scale))
            per_request_response_orthogonal_R0_squared.append(orthogonal_sq)
        request_command_is_zero = bool(
            torch.count_nonzero(command[:, request_ordinal]).item() == 0
        )
        if alpha == 0.0:
            if _norm(actual_transition[:, request_ordinal]) != 0.0:
                raise NumericalMethodBoundary("zero command produced a nonzero actual transition")
            # The preregistered accounting explicitly withholds relative debt
            # for a globally skipped action instead of imputing a numeric zero.
            per_request_D_R0.append(None)
            per_request_D_model_R0.append(None)
            per_request_zero_command_cross_response_R0_squared.append(None)
            per_request_metric_status.append("SKIPPED_ZERO_COMMAND")
            continue
        if request_command_is_zero:
            cross_response = float(
                torch.sum(actual_transition[:, request_ordinal].double().square()).item()
                / (scale * scale)
            )
            per_request_zero_command_cross_response_R0_squared.append(cross_response)
        else:
            cross_response = None
            per_request_zero_command_cross_response_R0_squared.append(None)
        d_r0 = float(
            torch.sum(realization_error[:, request_ordinal].double().square()).item()
            / (scale * scale)
        )
        per_request_D_R0.append(d_r0)
        if request_command_is_zero:
            if model_error is None:
                per_request_D_model_R0.append(None)
            else:
                d_model = float(
                    torch.sum(model_error[:, request_ordinal].double().square()).item()
                    / (scale * scale)
                )
                per_request_D_model_R0.append(d_model)
            per_request_metric_status.append(
                "ZERO_COMMAND_CROSS_RESPONSE_OBSERVED"
                if cross_response is not None and cross_response > 0.0
                else "ZERO_COMMAND_NO_CROSS_RESPONSE"
            )
            continue
        if model_error is None:
            per_request_D_model_R0.append(None)
            per_request_metric_status.append("DEFINED_D_R0_ONLY_NO_RESPONSE")
        else:
            d_model = float(
                torch.sum(model_error[:, request_ordinal].double().square()).item()
                / (scale * scale)
            )
            per_request_D_model_R0.append(d_model)
            per_request_metric_status.append("DEFINED")
    predicted_reduction = None
    if statistics is not None:
        predicted_reduction = h * u * statistics.g - 0.5 * (h * u) ** 2 * statistics.r
    direction_frobenius = float(direction_frobenius)
    applied_update_frobenius = direction_frobenius * h * u
    return StepRecord(
        arm=arm.value,
        sweep=sweep,
        layer=layer,
        visit_ordinal=visit_ordinal,
        built_state_version=built_state_version,
        resulting_state_version=resulting_state_version,
        residual_denominator=residual_denominator,
        command_reference_state_version=built_state_version,
        build_identity=build_identity,
        residual_sha256=residual_sha256,
        keys_sha256=keys_sha256,
        solver_identity=solver_identity,
        factor_rank=factor_rank,
        coefficient_u=float(u),
        euler_alpha=float(h * u),
        g=None if statistics is None else statistics.g,
        r=None if statistics is None else statistics.r,
        per_request_g=tuple(per_request_g),
        per_request_r=tuple(per_request_r),
        potential_before=potential_before,
        potential_after=potential_after,
        predicted_reduction=predicted_reduction,
        actual_reduction=potential_before - potential_after,
        discretization_defect=max(0.0, potential_after - potential_before),
        entry_sublevel_violation=max(0.0, potential_after - entry_potential),
        residual_norm_mean_before=_mean_norm_columns(residual_before),
        residual_norm_mean_after=_mean_norm_columns(residual_after),
        response_norm_mean=None if response is None else _mean_norm_columns(response),
        command_norm_mean=_mean_norm_columns(command),
        writer_command_norm_mean=_mean_norm_columns(writer_command),
        predicted_transition_norm_mean=None if predicted is None else _mean_norm_columns(predicted),
        actual_transition_norm_mean=_mean_norm_columns(actual_transition),
        realization_error_norm_mean=_mean_norm_columns(realization_error),
        model_error_norm_mean=None if model_error is None else _mean_norm_columns(model_error),
        per_request_D_R0=tuple(per_request_D_R0),
        per_request_D_model_R0=tuple(per_request_D_model_R0),
        per_request_q_res=tuple(per_request_q_res),
        per_request_writer_command_R0_squared=tuple(per_request_writer_command_R0_squared),
        per_request_response_orthogonal_R0_squared=tuple(
            per_request_response_orthogonal_R0_squared
        ),
        per_request_zero_command_cross_response_R0_squared=tuple(
            per_request_zero_command_cross_response_R0_squared
        ),
        per_request_actual_potential_change=tuple(per_request_actual_potential_change),
        per_request_actual_potential_worsened=tuple(per_request_actual_potential_worsened),
        per_request_metric_status=tuple(per_request_metric_status),
        direction_frobenius=direction_frobenius,
        direction_frobenius_squared=direction_frobenius * direction_frobenius,
        applied_update_frobenius=applied_update_frobenius,
        applied_update_frobenius_squared=applied_update_frobenius * applied_update_frobenius,
        resolution_stable_path_increment_frobenius_squared=(
            direction_frobenius * direction_frobenius * h * u * u
        ),
        action_geometry_status="FROBENIUS_ONLY_CREG_NOT_BOUND",
        native_creg_action_status="TELEMETRY_WITHHELD",
        semantic_all_strict=semantic.all_strict,
        semantic_request_count=request_count,
        semantic_strict_event_count=semantic.strict_event_count,
        semantic_event_count=semantic.event_count,
        semantic_request_strict_count=semantic.request_strict_count,
        semantic_tie_count=semantic.strict_tie_count,
        virtual_state_identity_sha256=virtual_state_identity_sha256,
    )


__all__ = ["ArmTelemetry", "StepRecord", "build_step_record", "semantic_payload"]
