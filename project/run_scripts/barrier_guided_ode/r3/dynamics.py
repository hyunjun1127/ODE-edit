"""Coefficient-space explicit Euler fixtures and physical path accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .errors import R3ScientificBoundary


@dataclass(frozen=True, slots=True)
class TimeStep:
    index: int
    entry_time: float
    size: float

    def __post_init__(self) -> None:
        if self.index < 0 or not math.isfinite(self.entry_time) or self.entry_time < 0.0:
            raise R3ScientificBoundary("time step entry is invalid")
        if not math.isfinite(self.size) or self.size <= 0.0:
            raise R3ScientificBoundary("time step size is invalid")


def fixed_count_schedule(*, horizon: float, count: int) -> tuple[TimeStep, ...]:
    total = float(horizon)
    if not math.isfinite(total) or total <= 0.0:
        raise R3ScientificBoundary("horizon must be positive and finite")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise R3ScientificBoundary("step count must be positive")
    size = total / count
    return tuple(TimeStep(index, index * size, size) for index in range(count))


@dataclass(frozen=True, slots=True)
class EulerNode:
    step: TimeStep
    entry: torch.Tensor
    velocity: torch.Tensor
    coefficient: torch.Tensor
    exit: torch.Tensor


@dataclass(frozen=True, slots=True)
class EulerResult:
    initial: torch.Tensor
    terminal: torch.Tensor
    nodes: tuple[EulerNode, ...]


def integrate_euler(
    initial: torch.Tensor,
    schedule: Sequence[TimeStep],
    *,
    velocity: Callable[[torch.Tensor, float], torch.Tensor],
) -> EulerResult:
    if (
        not isinstance(initial, torch.Tensor)
        or initial.dtype != torch.float64
        or initial.ndim != 1
        or initial.requires_grad
        or not bool(torch.isfinite(initial).all())
    ):
        raise R3ScientificBoundary("Euler state must be detached finite FP64")
    steps = tuple(schedule)
    if not steps or tuple(step.index for step in steps) != tuple(range(len(steps))):
        raise R3ScientificBoundary("Euler schedule indices are not canonical")
    state = initial.detach().clone().contiguous()
    records: list[EulerNode] = []
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
            raise R3ScientificBoundary("Euler velocity is invalid")
        coefficient = (node_velocity * step.size).detach().contiguous()
        state = (entry + coefficient).detach().contiguous()
        records.append(EulerNode(step, entry, node_velocity.detach().contiguous(), coefficient, state))
    return EulerResult(initial.detach().clone().contiguous(), state, tuple(records))


@dataclass(frozen=True, slots=True)
class PhysicalPath:
    terminal_displacement: float
    path_length: float
    integrated_kinetic_energy: float
    legacy_sum_squared_update: float


def dense_block_path(
    physical_updates: Sequence[Sequence[torch.Tensor]],
    step_sizes: Sequence[float],
) -> PhysicalPath:
    nodes = tuple(tuple(block.detach().to(dtype=torch.float64) for block in node) for node in physical_updates)
    sizes = tuple(float(value) for value in step_sizes)
    if not nodes or len(nodes) != len(sizes) or any(value <= 0.0 or not math.isfinite(value) for value in sizes):
        raise R3ScientificBoundary("physical path inventory is invalid")
    width = len(nodes[0])
    if width <= 0 or any(len(node) != width for node in nodes):
        raise R3ScientificBoundary("physical block width drifted")
    shapes = tuple(block.shape for block in nodes[0])
    if any(
        block.requires_grad or not bool(torch.isfinite(block).all()) or block.shape != shapes[index]
        for node in nodes
        for index, block in enumerate(node)
    ):
        raise R3ScientificBoundary("physical update block is invalid")
    node_norms = tuple(
        math.sqrt(sum(float(torch.sum(block.square()).cpu().item()) for block in node))
        for node in nodes
    )
    terminal_blocks = tuple(sum((node[index] for node in nodes), torch.zeros_like(nodes[0][index])) for index in range(width))
    terminal = math.sqrt(sum(float(torch.sum(block.square()).cpu().item()) for block in terminal_blocks))
    return PhysicalPath(
        terminal_displacement=terminal,
        path_length=sum(node_norms),
        integrated_kinetic_energy=sum(norm * norm / size for norm, size in zip(node_norms, sizes, strict=True)),
        legacy_sum_squared_update=sum(norm * norm for norm in node_norms),
    )


__all__ = [
    "EulerNode",
    "EulerResult",
    "PhysicalPath",
    "TimeStep",
    "dense_block_path",
    "fixed_count_schedule",
    "integrate_euler",
]
