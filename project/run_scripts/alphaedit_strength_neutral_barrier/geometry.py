"""Low-rank projection and closed-form strength-neutral correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


class BarrierStrengthInfeasible(RuntimeError):
    """Native barrier growth has no feasible strength-neutral correction."""


class GeometryBoundary(RuntimeError):
    """The projector/key/constraint geometry is numerically invalid."""


@dataclass(frozen=True)
class LowRankFactor:
    """Matrix ``left @ right.T`` without dense materialization."""

    left: torch.Tensor
    right: torch.Tensor

    def __post_init__(self) -> None:
        if self.left.ndim != 2 or self.right.ndim != 2:
            raise GeometryBoundary("low-rank factors must be matrices")
        if self.left.shape[1] != self.right.shape[1]:
            raise GeometryBoundary("low-rank factor rank mismatch")
        if self.left.dtype != torch.float32 or self.right.dtype != torch.float32:
            raise GeometryBoundary("low-rank numerical core must be FP32")


@dataclass(frozen=True)
class FactorTerm:
    coefficient: torch.Tensor
    factor: LowRankFactor


@dataclass(frozen=True)
class RightProjectionReceipt:
    gram_rank: int
    gram_condition: float
    pinv_rtol: float
    key_residual_max_abs: float


@dataclass(frozen=True)
class CorrectionSolution:
    terms: tuple[FactorTerm, ...]
    native_barrier_rate: float
    feasible_gradient_norm: float
    correction_norm: float
    directional_rate: float
    target_strength_inner_abs: tuple[float, ...]
    max_strength_inner_abs: float
    key_residual_max_abs: float
    constraint_rank: int
    constraint_condition: float
    constraint_pinv_rtol: float
    active: bool


def factor_inner(a: LowRankFactor, b: LowRankFactor) -> torch.Tensor:
    return torch.sum((a.left.T @ b.left) * (a.right.T @ b.right))


def factor_dense_inner(factor: LowRankFactor, dense: torch.Tensor) -> torch.Tensor:
    if dense.ndim != 2:
        raise GeometryBoundary("dense update must be a matrix")
    return torch.sum(factor.left * (dense @ factor.right))


def terms_inner(
    left_terms: Sequence[FactorTerm], right_terms: Sequence[FactorTerm]
) -> torch.Tensor:
    if not left_terms or not right_terms:
        reference = (
            left_terms[0].coefficient if left_terms else right_terms[0].coefficient
        )
        return reference.new_zeros(())
    total = left_terms[0].coefficient.new_zeros(())
    for left in left_terms:
        for right in right_terms:
            total = total + (
                left.coefficient
                * right.coefficient
                * factor_inner(left.factor, right.factor)
            )
    return total


def terms_norm(terms: Sequence[FactorTerm]) -> torch.Tensor:
    if not terms:
        return torch.tensor(0.0, dtype=torch.float32)
    value = terms_inner(terms, terms)
    return torch.sqrt(torch.clamp(value, min=0.0))


def project_right(
    right: torch.Tensor, projector: torch.Tensor, keys: torch.Tensor
) -> tuple[torch.Tensor, RightProjectionReceipt]:
    """Apply ``Q=P-(PK)(K^T P K)^dagger(PK)^T`` to factor columns."""

    right = right.float()
    projector = projector.float()
    keys = keys.float()
    if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
        raise GeometryBoundary("AlphaEdit P must be square")
    if right.shape[0] != projector.shape[0] or keys.shape[0] != projector.shape[0]:
        raise GeometryBoundary("projector/right/key shape mismatch")
    projected_keys = projector @ keys
    gram = keys.T @ projected_keys
    singular = torch.linalg.svdvals(gram)
    pinv_rtol = max(gram.shape) * torch.finfo(torch.float32).eps
    threshold = (
        pinv_rtol * singular.max()
        if singular.numel()
        else torch.tensor(0.0, device=gram.device)
    )
    rank = int(torch.count_nonzero(singular > threshold).item())
    positive = singular[singular > threshold]
    condition = (
        float((positive.max() / positive.min()).item()) if positive.numel() else float("inf")
    )
    gram_pinv = torch.linalg.pinv(gram, rtol=pinv_rtol)
    projected = projector @ right
    projected = projected - projected_keys @ (
        gram_pinv @ (projected_keys.T @ right)
    )
    residual = projected.T @ keys
    max_residual = float(residual.abs().max().item()) if residual.numel() else 0.0
    if not torch.isfinite(projected).all():
        raise GeometryBoundary("right-projected factor is nonfinite")
    return projected, RightProjectionReceipt(
        gram_rank=rank,
        gram_condition=condition,
        pinv_rtol=float(pinv_rtol),
        key_residual_max_abs=max_residual,
    )


def solve_strength_neutral_correction(
    *,
    barrier_gradient: LowRankFactor,
    target_gradients: Sequence[LowRankFactor],
    native_delta: torch.Tensor,
    projected_right: torch.Tensor,
    keys: torch.Tensor,
) -> CorrectionSolution:
    """Solve the attachment's minimum-norm correction in closed form."""

    g_projected = LowRankFactor(barrier_gradient.left, projected_right)
    a_projected = [
        LowRankFactor(gradient.left, projected_right) for gradient in target_gradients
    ]
    native_rate_tensor = factor_dense_inner(barrier_gradient, native_delta.float())
    native_rate = float(native_rate_tensor.item())

    if a_projected:
        gram = torch.stack(
            [
                torch.stack([factor_inner(a, b) for b in a_projected])
                for a in a_projected
            ]
        )
        rhs = torch.stack([factor_inner(a, g_projected) for a in a_projected])
        singular = torch.linalg.svdvals(gram)
        rtol = max(gram.shape) * torch.finfo(torch.float32).eps
        threshold = rtol * singular.max()
        rank = int(torch.count_nonzero(singular > threshold).item())
        positive = singular[singular > threshold]
        condition = (
            float((positive.max() / positive.min()).item())
            if positive.numel()
            else float("inf")
        )
        coefficients = torch.linalg.pinv(gram, rtol=rtol) @ rhs
    else:
        rtol = 0.0
        rank = 0
        condition = 1.0
        coefficients = native_delta.new_zeros((0,))

    h_terms = [
        FactorTerm(native_delta.new_tensor(1.0), g_projected),
        *[
            FactorTerm(-coefficient, factor)
            for coefficient, factor in zip(coefficients, a_projected)
        ],
    ]
    h_norm = terms_norm(h_terms)
    h_norm_value = float(h_norm.item())
    if not torch.isfinite(h_norm):
        raise GeometryBoundary("feasible projected gradient norm is nonfinite")

    if native_rate <= 0.0:
        correction_terms: tuple[FactorTerm, ...] = ()
        correction_norm = 0.0
        directional = native_rate
        active = False
    else:
        h_norm_sq = terms_inner(h_terms, h_terms)
        if float(h_norm_sq.item()) <= 0.0:
            raise BarrierStrengthInfeasible(
                "BARRIER_STRENGTH_INFEASIBLE: c>0 with ||H||=0"
            )
        scale = -native_rate_tensor / h_norm_sq
        correction_terms = tuple(
            FactorTerm(scale * term.coefficient, term.factor) for term in h_terms
        )
        correction_norm = float(terms_norm(correction_terms).item())
        directional = float(
            (
                native_rate_tensor
                + terms_inner(
                    [FactorTerm(native_delta.new_tensor(1.0), g_projected)],
                    correction_terms,
                )
            ).item()
        )
        active = True

    strength_values = []
    for gradient in a_projected:
        strength_values.append(
            float(
                abs(
                    terms_inner(
                        [FactorTerm(native_delta.new_tensor(1.0), gradient)],
                        correction_terms,
                    ).item()
                )
            )
        )
    max_strength = max(strength_values, default=0.0)

    key_residual = 0.0
    for term in correction_terms:
        residual = term.factor.right.T @ keys.float()
        if residual.numel():
            key_residual = max(
                key_residual,
                abs(float(term.coefficient.item())) * float(residual.abs().max().item()),
            )

    return CorrectionSolution(
        terms=correction_terms,
        native_barrier_rate=native_rate,
        feasible_gradient_norm=h_norm_value,
        correction_norm=correction_norm,
        directional_rate=directional,
        target_strength_inner_abs=tuple(strength_values),
        max_strength_inner_abs=max_strength,
        key_residual_max_abs=key_residual,
        constraint_rank=rank,
        constraint_condition=condition,
        constraint_pinv_rtol=float(rtol),
        active=active,
    )


def apply_factor_terms_(
    weight: torch.Tensor,
    native_delta: torch.Tensor,
    correction_terms: Iterable[FactorTerm],
    *,
    step_scale: float,
) -> None:
    """Apply one Euler step without materializing the low-rank correction."""

    with torch.no_grad():
        weight.add_(native_delta.to(weight), alpha=step_scale)
        for term in correction_terms:
            alpha = step_scale * float(term.coefficient.item())
            weight.addmm_(
                term.factor.left.to(weight), term.factor.right.to(weight).T, alpha=alpha
            )


def factor_terms_times_keys(
    correction_terms: Sequence[FactorTerm], keys: torch.Tensor
) -> torch.Tensor:
    """Materialize the actual summed ``D K`` residual without materializing D."""

    if not correction_terms:
        return keys.new_zeros((0, keys.shape[1]), dtype=torch.float32)
    first = correction_terms[0]
    result = first.factor.left.new_zeros(
        (first.factor.left.shape[0], keys.shape[1]), dtype=torch.float32
    )
    keys = keys.float()
    for term in correction_terms:
        result = result + term.coefficient.float() * (
            term.factor.left @ (term.factor.right.T @ keys)
        )
    if not torch.isfinite(result).all():
        raise GeometryBoundary("actual D K residual is nonfinite")
    return result
