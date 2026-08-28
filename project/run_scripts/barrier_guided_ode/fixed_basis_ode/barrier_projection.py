"""Nominal AlphaEdit velocity projected onto the KL non-increase half-space."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class ProjectionBoundary(RuntimeError):
    """Barrier projection inputs or result are invalid."""


@dataclass(frozen=True, slots=True)
class ProjectionReceipt:
    gradient_dot_nominal: float
    gradient_dot_velocity: float
    gradient_norm: float
    nominal_norm: float
    velocity_norm: float
    projection_active: bool
    zero_gradient: bool


def project_nominal_velocity(
    nominal: torch.Tensor,
    gradient: torch.Tensor,
) -> tuple[torch.Tensor, ProjectionReceipt]:
    if nominal.dtype != torch.float64 or gradient.dtype != torch.float64 or nominal.shape != gradient.shape:
        raise ProjectionBoundary("nominal velocity and KL gradient must be aligned FP64 vectors")
    if nominal.ndim != 1 or not bool(torch.isfinite(nominal).all()) or not bool(torch.isfinite(gradient).all()):
        raise ProjectionBoundary("nominal velocity or KL gradient is invalid")
    g2 = torch.dot(gradient, gradient)
    dot = torch.dot(gradient, nominal)
    nominal2 = torch.dot(nominal, nominal)
    eps = torch.finfo(torch.float64).eps
    zero_tolerance = 64.0 * eps * max(1.0, float(nominal2))
    zero = float(g2) <= zero_tolerance
    active = (not zero) and float(dot) > 0.0
    velocity = nominal - (dot / g2) * gradient if active else nominal.clone()
    post = torch.dot(gradient, velocity)
    residual_tolerance = 512.0 * eps * max(1.0, float(torch.sqrt(g2) * torch.linalg.vector_norm(velocity)))
    if not bool(torch.isfinite(velocity).all()) or (active and float(post) > residual_tolerance):
        raise ProjectionBoundary("projected velocity violates the KL half-space")
    return velocity.contiguous(), ProjectionReceipt(
        gradient_dot_nominal=float(dot),
        gradient_dot_velocity=float(post),
        gradient_norm=float(torch.sqrt(g2)),
        nominal_norm=float(torch.linalg.vector_norm(nominal)),
        velocity_norm=float(torch.linalg.vector_norm(velocity)),
        projection_active=active,
        zero_gradient=zero,
    )
