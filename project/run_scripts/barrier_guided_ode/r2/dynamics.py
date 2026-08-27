"""Generic finite-step utilities for the BGODE-R2 coefficient flow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .errors import R2ScientificBoundary
from .telemetry import WeightPathTelemetry


@dataclass(frozen=True, slots=True)
class TimeStep:
    index: int
    entry_time: float
    size: float

    def __post_init__(self) -> None:
        if self.index < 0 or not math.isfinite(self.entry_time) or self.entry_time < 0.0:
            raise R2ScientificBoundary("time step entry is invalid")
        if not math.isfinite(self.size) or self.size <= 0.0:
            raise R2ScientificBoundary("time step size is invalid")


def fixed_count_schedule(*, horizon: float, count: int) -> tuple[TimeStep, ...]:
    if not math.isfinite(float(horizon)) or float(horizon) <= 0.0:
        raise R2ScientificBoundary("horizon must be positive and finite")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise R2ScientificBoundary("step count must be positive")
    size = float(horizon) / count
    return tuple(TimeStep(index, index * size, size) for index in range(count))


def maximum_step_schedule(*, horizon: float, maximum_step: float) -> tuple[TimeStep, ...]:
    total = float(horizon)
    limit = float(maximum_step)
    if not math.isfinite(total) or total <= 0.0 or not math.isfinite(limit) or limit <= 0.0:
        raise R2ScientificBoundary("time schedule inputs must be positive and finite")
    steps: list[TimeStep] = []
    entry = 0.0
    while entry < total:
        remaining = total - entry
        size = min(limit, remaining)
        if size <= 0.0 or not math.isfinite(size):
            raise R2ScientificBoundary("time schedule did not advance")
        steps.append(TimeStep(len(steps), entry, size))
        entry = min(total, entry + size)
    if not math.isclose(sum(step.size for step in steps), total, rel_tol=0.0, abs_tol=64.0 * math.ulp(total)):
        raise R2ScientificBoundary("time schedule does not cover the horizon")
    return tuple(steps)


@dataclass(frozen=True, slots=True)
class EulerNodeRecord:
    step: TimeStep
    entry_state: torch.Tensor
    velocity: torch.Tensor
    coefficient: torch.Tensor
    exit_state: torch.Tensor
    local_progress_error: float
    cumulative_progress_error: float


@dataclass(frozen=True, slots=True)
class EulerIntegration:
    initial_state: torch.Tensor
    final_state: torch.Tensor
    nodes: tuple[EulerNodeRecord, ...]


def integrate_explicit_euler(
    initial_state: torch.Tensor,
    schedule: Sequence[TimeStep],
    *,
    velocity: Callable[[torch.Tensor, float], torch.Tensor],
    progress: Callable[[torch.Tensor], float],
) -> EulerIntegration:
    if (
        not isinstance(initial_state, torch.Tensor)
        or initial_state.dtype != torch.float64
        or initial_state.ndim != 1
        or initial_state.requires_grad
        or not bool(torch.isfinite(initial_state).all())
    ):
        raise R2ScientificBoundary("Euler state must be detached finite FP64")
    steps = tuple(schedule)
    if not steps or tuple(step.index for step in steps) != tuple(range(len(steps))):
        raise R2ScientificBoundary("Euler schedule indices are not canonical")
    state = initial_state.detach().clone().contiguous()
    initial_progress = float(progress(state))
    if not math.isfinite(initial_progress):
        raise R2ScientificBoundary("initial progress is non-finite")
    records: list[EulerNodeRecord] = []
    for step in steps:
        entry = state.detach().clone().contiguous()
        node_velocity = velocity(entry, step.entry_time)
        if (
            not isinstance(node_velocity, torch.Tensor)
            or node_velocity.dtype != torch.float64
            or node_velocity.shape != entry.shape
            or node_velocity.requires_grad
            or not bool(torch.isfinite(node_velocity).all())
        ):
            raise R2ScientificBoundary("Euler velocity is invalid")
        coefficient = (node_velocity * step.size).detach().contiguous()
        state = (entry + coefficient).detach().contiguous()
        before = float(progress(entry))
        after = float(progress(state))
        if not math.isfinite(before) or not math.isfinite(after):
            raise R2ScientificBoundary("Euler progress is non-finite")
        records.append(
            EulerNodeRecord(
                step=step,
                entry_state=entry,
                velocity=node_velocity.detach().contiguous(),
                coefficient=coefficient,
                exit_state=state.detach().clone().contiguous(),
                local_progress_error=(after - before) - step.size,
                cumulative_progress_error=(after - initial_progress) - (step.entry_time + step.size),
            )
        )
    return EulerIntegration(
        initial_state=initial_state.detach().clone().contiguous(),
        final_state=state,
        nodes=tuple(records),
    )


def weight_path_telemetry(
    coefficients: Sequence[torch.Tensor],
    step_sizes: Sequence[float],
) -> WeightPathTelemetry:
    values = tuple(coefficients)
    sizes = tuple(float(size) for size in step_sizes)
    if not values or len(values) != len(sizes):
        raise R2ScientificBoundary("weight path inputs do not align")
    if any(size <= 0.0 or not math.isfinite(size) for size in sizes):
        raise R2ScientificBoundary("weight path step size is invalid")
    shape = values[0].shape
    if any(
        value.dtype != torch.float64
        or value.shape != shape
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
        for value in values
    ):
        raise R2ScientificBoundary("weight path coefficients are invalid")
    norms = tuple(float(torch.linalg.vector_norm(value).cpu().item()) for value in values)
    terminal = torch.stack(values).sum(dim=0)
    kinetic = sum((norm * norm) / size for norm, size in zip(norms, sizes, strict=True))
    return WeightPathTelemetry(
        terminal_displacement=float(torch.linalg.vector_norm(terminal).cpu().item()),
        path_length=sum(norms),
        integrated_kinetic_energy=kinetic,
        legacy_sum_squared_update=sum(norm * norm for norm in norms),
    )


__all__ = [
    "EulerIntegration",
    "EulerNodeRecord",
    "TimeStep",
    "fixed_count_schedule",
    "integrate_explicit_euler",
    "maximum_step_schedule",
    "weight_path_telemetry",
]
