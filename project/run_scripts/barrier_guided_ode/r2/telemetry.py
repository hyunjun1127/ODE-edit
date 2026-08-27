"""Immutable observation-only telemetry for the BGODE-R2 numerical core."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .errors import R2ScientificBoundary


def _finite(values: tuple[float, ...], *, name: str, nonnegative: bool = False) -> None:
    if any(not math.isfinite(value) or (nonnegative and value < 0.0) for value in values):
        raise R2ScientificBoundary(f"{name} contains invalid values")


@dataclass(frozen=True, slots=True)
class EventStateTelemetry:
    target_probability: float
    source_probability: float
    pair_mass: float
    log_odds: float
    normalization_log_residual: float
    normalization_tolerance: float

    def __post_init__(self) -> None:
        _finite(
            (
                self.target_probability,
                self.source_probability,
                self.pair_mass,
                self.log_odds,
                self.normalization_log_residual,
                self.normalization_tolerance,
            ),
            name="event state",
        )
        if not (0.0 < self.target_probability < 1.0 and 0.0 < self.source_probability < 1.0):
            raise R2ScientificBoundary("distinguished event probabilities must be interior")
        if not (0.0 < self.pair_mass < 1.0):
            raise R2ScientificBoundary("pair mass must be interior")
        if self.normalization_log_residual < 0.0 or self.normalization_tolerance <= 0.0:
            raise R2ScientificBoundary("normalization diagnostics are invalid")


@dataclass(frozen=True, slots=True)
class BarrierTelemetry:
    moving_reference_kl: float
    anchored_kl: float
    unavoidable_pair_cost: float
    pair_mass_drift_kl: float
    outside_conditional_event_kl: float

    def __post_init__(self) -> None:
        _finite(
            (
                self.moving_reference_kl,
                self.anchored_kl,
                self.unavoidable_pair_cost,
                self.pair_mass_drift_kl,
                self.outside_conditional_event_kl,
            ),
            name="barrier telemetry",
            nonnegative=True,
        )


@dataclass(frozen=True, slots=True)
class NodeTelemetry:
    node_index: int
    entry_time: float
    step_size: float
    local_progress_error: float
    cumulative_progress_error: float
    entry_event: EventStateTelemetry
    exit_event: EventStateTelemetry
    controller_reference_event_log_probabilities: tuple[float, ...]
    barrier_entry: BarrierTelemetry
    barrier_exit: BarrierTelemetry
    equality_residual: float
    stationarity_residual: float
    unit_prefix_mismatch: tuple[float, ...]
    effective_prefix_mismatch: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.node_index < 0 or self.entry_time < 0.0 or self.step_size <= 0.0:
            raise R2ScientificBoundary("node clock is invalid")
        _finite(
            (
                self.entry_time,
                self.step_size,
                self.local_progress_error,
                self.cumulative_progress_error,
                self.equality_residual,
                self.stationarity_residual,
            ),
            name="node telemetry",
        )
        _finite(
            self.controller_reference_event_log_probabilities,
            name="controller reference event",
        )
        _finite(self.unit_prefix_mismatch, name="unit prefix mismatch", nonnegative=True)
        _finite(self.effective_prefix_mismatch, name="effective prefix mismatch", nonnegative=True)


@dataclass(frozen=True, slots=True)
class WeightPathTelemetry:
    terminal_displacement: float
    path_length: float
    integrated_kinetic_energy: float
    legacy_sum_squared_update: float

    def __post_init__(self) -> None:
        _finite(
            (
                self.terminal_displacement,
                self.path_length,
                self.integrated_kinetic_energy,
                self.legacy_sum_squared_update,
            ),
            name="weight path telemetry",
            nonnegative=True,
        )


@dataclass(frozen=True, slots=True)
class R2ExecutionBoundary:
    batch_size: int
    fixed_target_compute_count: int
    fixed_target_recompute_count: int
    identity_scale: float
    node_history_append_count: int
    accepted_terminal_history_append_count: int
    model_numeric_dtype: str
    numerical_core_dtype: str
    physical_write_dtype: str
    probability_floor_count: int
    damping_count: int
    ridge_count: int
    fallback_count: int
    scientific_promotion: bool

    def __post_init__(self) -> None:
        if (
            self.batch_size != 1
            or self.fixed_target_compute_count != 1
            or self.fixed_target_recompute_count != 0
            or self.identity_scale != 1.0
            or self.node_history_append_count != 0
            or self.accepted_terminal_history_append_count not in (0, 1)
        ):
            raise R2ScientificBoundary("R2 execution boundary counters are invalid")
        if (
            self.model_numeric_dtype != "torch.float32"
            or self.numerical_core_dtype != "torch.float64"
            or self.physical_write_dtype != "torch.float32"
        ):
            raise R2ScientificBoundary("R2 dtype boundary is invalid")
        if any(
            value != 0
            for value in (
                self.probability_floor_count,
                self.damping_count,
                self.ridge_count,
                self.fallback_count,
            )
        ) or self.scientific_promotion:
            raise R2ScientificBoundary("R2 prohibited influence counter is nonzero")

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "BarrierTelemetry",
    "EventStateTelemetry",
    "NodeTelemetry",
    "R2ExecutionBoundary",
    "WeightPathTelemetry",
]
