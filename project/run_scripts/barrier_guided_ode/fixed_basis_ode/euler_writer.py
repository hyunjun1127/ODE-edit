"""Coefficient-space Euler integration with a fixed AlphaEdit basis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch

from project.run_scripts.barrier_guided_ode.r3.events import FineEventEvaluation

from .barrier_projection import project_nominal_velocity
from .event_distribution import TargetExcludedReference


class EulerBoundary(RuntimeError):
    """The minimal fixed-basis Euler contract was violated."""


@dataclass(frozen=True, slots=True)
class EulerResult:
    theta: torch.Tensor
    nodes: tuple[dict[str, object], ...]
    horizon: float
    steps: int


def run_barrier_euler(
    *,
    theta_ae: torch.Tensor,
    reference: TargetExcludedReference,
    observe: Callable[[torch.Tensor], FineEventEvaluation],
    steps: int,
    horizon: float = 1.0,
) -> EulerResult:
    if steps not in (2, 4) or horizon != 1.0:
        raise EulerBoundary("minimal experiment locks barrier Euler to T=1 and N in {2,4}")
    if theta_ae.dtype != torch.float64 or theta_ae.shape != (5,) or not bool(torch.isfinite(theta_ae).all()):
        raise EulerBoundary("native AlphaEdit nominal velocity is invalid")
    theta = torch.zeros_like(theta_ae)
    rows: list[dict[str, object]] = []
    step_size = horizon / steps
    for node in range(steps):
        event = observe(theta)
        barrier, gradient = reference.evaluate(event)
        velocity, projection = project_nominal_velocity(theta_ae, gradient)
        rows.append(
            {
                "node": node,
                "time": node * step_size,
                "theta": [float(value) for value in theta.tolist()],
                "barrier": barrier,
                "source_probability": float(event.source_log_probability.exp()),
                "target_probability": float(event.target_log_probability.exp()),
                "projection": asdict(projection),
            }
        )
        theta = (theta + step_size * velocity).detach().to(dtype=torch.float64).contiguous()
        if not bool(torch.isfinite(theta).all()):
            raise EulerBoundary("Euler coefficient state is non-finite")
    return EulerResult(theta=theta, nodes=tuple(rows), horizon=horizon, steps=steps)
