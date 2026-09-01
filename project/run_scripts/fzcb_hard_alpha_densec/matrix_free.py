"""Small, certifying matrix-free LSQR primitives.

No Gram matrix, pseudoinverse, null basis, or Kronecker matrix is formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .contracts import EngineeringBoundary


TensorMap = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class LinearOperator:
    forward: TensorMap
    adjoint: TensorMap
    domain_shape: tuple[int, ...]
    range_shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    def check_adjoint(self, seed: int = 0) -> float:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn(self.domain_shape, generator=generator, dtype=self.dtype, device=self.device)
        y = torch.randn(self.range_shape, generator=generator, dtype=self.dtype, device=self.device)
        left = torch.dot(self.forward(x).reshape(-1), y.reshape(-1))
        right = torch.dot(x.reshape(-1), self.adjoint(y).reshape(-1))
        scale = torch.maximum(left.abs(), right.abs()).clamp_min(torch.finfo(self.dtype).tiny)
        return float(((left - right).abs() / scale).item())


@dataclass(frozen=True)
class LSQRResult:
    solution: torch.Tensor
    iterations: int
    relative_residual: float
    normal_relative_residual: float
    converged: bool


def lsqr(
    operator: LinearOperator,
    rhs: torch.Tensor,
    *,
    relative_tolerance: float,
    max_iterations: int,
) -> LSQRResult:
    if tuple(rhs.shape) != operator.range_shape or rhs.dtype != operator.dtype or rhs.device != operator.device:
        raise EngineeringBoundary("LSQR rhs/operator mismatch")
    tiny = torch.finfo(operator.dtype).tiny
    beta = torch.linalg.vector_norm(rhs)
    if float(beta.item()) == 0.0:
        zero = torch.zeros(operator.domain_shape, dtype=operator.dtype, device=operator.device)
        return LSQRResult(zero, 0, 0.0, 0.0, True)
    u = rhs / beta
    v = operator.adjoint(u)
    alpha = torch.linalg.vector_norm(v)
    if float(alpha.item()) == 0.0:
        zero = torch.zeros(operator.domain_shape, dtype=operator.dtype, device=operator.device)
        return LSQRResult(zero, 0, 1.0, 0.0, False)
    v = v / alpha
    w = v.clone()
    x = torch.zeros_like(v)
    phibar = beta
    rhobar = alpha
    rhs_norm = beta.clamp_min(tiny)
    iterations = 0
    converged = False
    relative = float("inf")
    normal_relative = float("inf")
    for iteration in range(1, max_iterations + 1):
        u_next = operator.forward(v) - alpha * u
        beta = torch.linalg.vector_norm(u_next)
        u = u_next / beta if float(beta.item()) != 0.0 else torch.zeros_like(u_next)
        v_next = operator.adjoint(u) - beta * v
        alpha = torch.linalg.vector_norm(v_next)
        v = v_next / alpha if float(alpha.item()) != 0.0 else torch.zeros_like(v_next)
        rho = torch.sqrt(rhobar * rhobar + beta * beta).clamp_min(tiny)
        cosine = rhobar / rho
        sine = beta / rho
        theta = sine * alpha
        rhobar = -cosine * alpha
        phi = cosine * phibar
        phibar = sine * phibar
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w
        residual = operator.forward(x) - rhs
        relative_tensor = torch.linalg.vector_norm(residual) / rhs_norm
        normal = operator.adjoint(residual)
        normal_scale = torch.linalg.vector_norm(operator.adjoint(rhs)).clamp_min(tiny)
        normal_tensor = torch.linalg.vector_norm(normal) / normal_scale
        relative = float(relative_tensor.item())
        normal_relative = float(normal_tensor.item())
        iterations = iteration
        if relative <= relative_tolerance:
            converged = True
            break
        if float(alpha.item()) == 0.0 and float(beta.item()) == 0.0:
            break
    return LSQRResult(x, iterations, relative, normal_relative, converged)


def normal_range_operator(operator: LinearOperator) -> LinearOperator:
    """Return y -> A A^T y without ever forming A or its Gram matrix."""

    def apply(value: torch.Tensor) -> torch.Tensor:
        return operator.forward(operator.adjoint(value))

    return LinearOperator(
        forward=apply,
        adjoint=apply,
        domain_shape=operator.range_shape,
        range_shape=operator.range_shape,
        dtype=operator.dtype,
        device=operator.device,
    )
