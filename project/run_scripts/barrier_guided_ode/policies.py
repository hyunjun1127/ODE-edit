"""Explicit policy boundaries, including the native AlphaEdit bypass."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .errors import NativeBypassBoundary, UnsupportedBatchBoundary


@dataclass(frozen=True, slots=True)
class NativeBypassReceipt:
    coefficients: torch.Tensor
    rho: float
    controller_call_count: int
    root_localization_count: int
    barrier_influence_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.coefficients, torch.Tensor)
            or self.coefficients.dtype != torch.float32
            or self.coefficients.ndim != 1
            or self.coefficients.numel() == 0
            or self.coefficients.requires_grad
            or not bool(torch.equal(self.coefficients, torch.ones_like(self.coefficients)))
        ):
            raise NativeBypassBoundary("native bypass coefficients must be exact FP32 ones")
        if self.rho != 1.0:
            raise NativeBypassBoundary("native bypass rho must equal one")
        if any(
            value != 0
            for value in (
                self.controller_call_count,
                self.root_localization_count,
                self.barrier_influence_count,
            )
        ):
            raise NativeBypassBoundary("native bypass must skip controller/root/barrier")


def explicit_native_bypass(
    *,
    batch_size: int,
    euler_steps: int,
    barrier_enabled: bool,
    factor_count: int,
) -> NativeBypassReceipt:
    """Return exact native coefficients only for ``B=1,N=1,barrier-off``."""

    if batch_size != 1:
        raise UnsupportedBatchBoundary("BGODE-R1 native bypass still requires B=1")
    if euler_steps != 1 or barrier_enabled:
        raise NativeBypassBoundary("native bypass requires N=1 and barrier disabled")
    if isinstance(factor_count, bool) or not isinstance(factor_count, int) or factor_count <= 0:
        raise NativeBypassBoundary("native bypass factor_count must be positive")
    return NativeBypassReceipt(
        coefficients=torch.ones(factor_count, dtype=torch.float32),
        rho=1.0,
        controller_call_count=0,
        root_localization_count=0,
        barrier_influence_count=0,
    )
