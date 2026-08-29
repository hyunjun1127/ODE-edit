"""Cache-aware residual-writer geometry for predictive q-KL projection.

All matrices use the canonical AlphaEdit orientation: ``R`` is ``o x m``,
``K``/``J`` are ``d x m``, and a weight velocity is ``R @ J.T`` (``o x d``).
The module is intentionally independent of models, optimizers, evaluators, and
the EasyEdit runtime so its numerical contract can be tested on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class GeometryBoundary(RuntimeError):
    """The AlphaEdit factorization or projected geometry is numerically invalid."""


class ActionableNormalBoundary(GeometryBoundary):
    """A positive q-KL rate has no actionable normal in the writer image."""


@dataclass(frozen=True)
class MetricReceipt:
    rank: int
    condition: float
    pinv_rtol: float
    minimum_eigenvalue: float
    psd_tolerance: float
    fast_energy_max_abs: float
    fast_energy_relative: float
    fast_energy_tolerance: float


@dataclass(frozen=True)
class NativeFactorization:
    residual: torch.Tensor
    keys: torch.Tensor
    writer_map: torch.Tensor
    metric: torch.Tensor
    metric_pinv: torch.Tensor
    velocity: torch.Tensor
    solve_backward_error: float
    solve_backward_tolerance: float
    stock_velocity_max_abs: float
    stock_velocity_relative: float
    metric_receipt: MetricReceipt


@dataclass(frozen=True)
class ProjectedVelocity:
    residual_velocity: torch.Tensor
    weight_velocity: torch.Tensor
    residual_correction: torch.Tensor
    native_rate: float
    eta: float
    projected_rate: float
    positive_projected_rate_violation: float
    correction_energy: float
    removed_energy_fraction: float
    native_energy: float
    current_key_energy: float
    history_cache_energy: float
    l2_energy: float
    active: bool


def _require_fp32_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2:
        raise GeometryBoundary(f"{name} must be a matrix")
    if value.dtype != torch.float32:
        raise GeometryBoundary(f"{name} must be FP32")
    if not torch.isfinite(value).all():
        raise GeometryBoundary(f"{name} is nonfinite")
    return value


def _gamma(dimension: int, dtype: torch.dtype = torch.float32) -> float:
    """Standard floating-point ``gamma_n`` bound, derived only from n and eps."""

    eps = torch.finfo(dtype).eps
    product = max(1, int(dimension)) * eps
    if product >= 1.0:
        raise GeometryBoundary("matrix dimension exceeds stable FP32 gamma_n range")
    return product / (1.0 - product)


def _relative_error(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    delta = left - right
    maximum = float(delta.abs().max().item()) if delta.numel() else 0.0
    scale = max(
        float(torch.linalg.vector_norm(left).item()),
        float(torch.linalg.vector_norm(right).item()),
        torch.finfo(left.dtype).tiny,
    )
    return maximum, float(torch.linalg.vector_norm(delta).item()) / scale


def build_writer_metric(
    *,
    keys: torch.Tensor,
    writer_map: torch.Tensor,
    history_cache: torch.Tensor,
    l2: float,
) -> tuple[torch.Tensor, torch.Tensor, MetricReceipt]:
    """Build and validate the cache-aware residual metric ``M``.

    The energy form is authoritative.  ``sym(K.T @ J)`` is checked as the
    algebraic fast form, but is never used to mask a mismatch.
    """

    keys = _require_fp32_matrix("K", keys)
    writer_map = _require_fp32_matrix("J", writer_map)
    history_cache = _require_fp32_matrix("C", history_cache)
    if keys.shape != writer_map.shape:
        raise GeometryBoundary("K/J shape mismatch")
    if history_cache.shape != (keys.shape[0], keys.shape[0]):
        raise GeometryBoundary("C shape mismatch")
    if not torch.isfinite(torch.tensor(l2)) or l2 < 0:
        raise GeometryBoundary("lambda must be finite and nonnegative")

    symmetric_cache = 0.5 * (history_cache + history_cache.T)
    state_metric = keys @ keys.T + symmetric_cache
    energy = writer_map.T @ state_metric @ writer_map
    energy = energy + float(l2) * (writer_map.T @ writer_map)
    energy = 0.5 * (energy + energy.T)
    fast = keys.T @ writer_map
    fast = 0.5 * (fast + fast.T)
    maximum, relative = _relative_error(fast, energy)
    fast_tolerance = 8.0 * _gamma(keys.shape[0])
    if relative > fast_tolerance:
        raise GeometryBoundary(
            "M_fast/M_energy mismatch: "
            f"relative={relative:.9e}, tolerance={fast_tolerance:.9e}"
        )

    eigenvalues = torch.linalg.eigvalsh(energy)
    scale = max(
        float(eigenvalues.abs().max().item()) if eigenvalues.numel() else 0.0,
        torch.finfo(torch.float32).tiny,
    )
    psd_tolerance = 8.0 * _gamma(max(energy.shape)) * scale
    minimum = float(eigenvalues.min().item()) if eigenvalues.numel() else 0.0
    if minimum < -psd_tolerance:
        raise GeometryBoundary(
            f"writer metric is not PSD: min={minimum:.9e}, tol={psd_tolerance:.9e}"
        )

    pinv_rtol = max(energy.shape) * torch.finfo(torch.float32).eps
    singular = torch.linalg.svdvals(energy)
    threshold = (
        float(pinv_rtol) * singular.max()
        if singular.numel()
        else torch.tensor(0.0, device=energy.device)
    )
    positive = singular[singular > threshold]
    rank = int(positive.numel())
    condition = (
        float((positive.max() / positive.min()).item())
        if positive.numel()
        else float("inf")
    )
    metric_pinv = torch.linalg.pinv(energy, rtol=pinv_rtol, hermitian=True)
    if not torch.isfinite(metric_pinv).all():
        raise GeometryBoundary("writer metric pseudoinverse is nonfinite")
    return energy, metric_pinv, MetricReceipt(
        rank=rank,
        condition=condition,
        pinv_rtol=float(pinv_rtol),
        minimum_eigenvalue=minimum,
        psd_tolerance=psd_tolerance,
        fast_energy_max_abs=maximum,
        fast_energy_relative=relative,
        fast_energy_tolerance=fast_tolerance,
    )


def factorize_alphaedit_velocity(
    *,
    residual: torch.Tensor,
    keys: torch.Tensor,
    projector: torch.Tensor,
    history_cache: torch.Tensor,
    l2: float,
) -> NativeFactorization:
    """Return ``R,K,J,M,F`` for the stock AlphaEdit layer solve."""

    residual = _require_fp32_matrix("R", residual)
    keys = _require_fp32_matrix("K", keys)
    projector = _require_fp32_matrix("P", projector)
    history_cache = _require_fp32_matrix("C", history_cache)
    if residual.shape[1] != keys.shape[1]:
        raise GeometryBoundary("R/K request-column mismatch")
    if projector.shape != (keys.shape[0], keys.shape[0]):
        raise GeometryBoundary("P shape mismatch")
    if history_cache.shape != projector.shape:
        raise GeometryBoundary("C/P shape mismatch")

    symmetric_cache = 0.5 * (history_cache + history_cache.T)
    state_metric = keys @ keys.T + symmetric_cache
    identity = torch.eye(keys.shape[0], dtype=torch.float32, device=keys.device)
    solve_matrix = projector @ state_metric + float(l2) * identity
    rhs = projector @ keys
    writer_map = torch.linalg.solve(solve_matrix, rhs)
    velocity = residual @ writer_map.T
    stock_velocity = torch.linalg.solve(solve_matrix, rhs @ residual.T).T
    stock_maximum, stock_relative = _relative_error(velocity, stock_velocity)

    solve_residual = solve_matrix @ writer_map - rhs
    denominator = (
        torch.linalg.matrix_norm(solve_matrix) * torch.linalg.matrix_norm(writer_map)
        + torch.linalg.matrix_norm(rhs)
    )
    backward_error = float(
        (
            torch.linalg.matrix_norm(solve_residual)
            / denominator.clamp_min(torch.finfo(torch.float32).tiny)
        ).item()
    )
    backward_tolerance = 8.0 * _gamma(keys.shape[0])
    if backward_error > backward_tolerance:
        raise GeometryBoundary(
            "AJ=PK backward error exceeds machine bound: "
            f"error={backward_error:.9e}, tolerance={backward_tolerance:.9e}"
        )
    if stock_relative > backward_tolerance:
        raise GeometryBoundary(
            "factorized/native stock solve mismatch: "
            f"relative={stock_relative:.9e}, tolerance={backward_tolerance:.9e}"
        )

    metric, metric_pinv, metric_receipt = build_writer_metric(
        keys=keys,
        writer_map=writer_map,
        history_cache=symmetric_cache,
        l2=l2,
    )
    return NativeFactorization(
        residual=residual,
        keys=keys,
        writer_map=writer_map,
        metric=metric,
        metric_pinv=metric_pinv,
        velocity=velocity,
        solve_backward_error=backward_error,
        solve_backward_tolerance=backward_tolerance,
        stock_velocity_max_abs=stock_maximum,
        stock_velocity_relative=stock_relative,
        metric_receipt=metric_receipt,
    )


def low_rank_pullback(
    barrier_left: torch.Tensor,
    common_right: torch.Tensor,
    writer_map: torch.Tensor,
) -> torch.Tensor:
    """Compute ``U=GJ=L(X.T@J)`` without materializing dense ``G``."""

    barrier_left = _require_fp32_matrix("L", barrier_left)
    common_right = _require_fp32_matrix("X", common_right)
    writer_map = _require_fp32_matrix("J", writer_map)
    if common_right.shape[0] != writer_map.shape[0]:
        raise GeometryBoundary("X/J input dimension mismatch")
    if barrier_left.shape[1] != common_right.shape[1]:
        raise GeometryBoundary("L/X event-rank mismatch")
    result = barrier_left @ (common_right.T @ writer_map)
    if not torch.isfinite(result).all():
        raise GeometryBoundary("low-rank q-KL pullback is nonfinite")
    return result


def project_one_sided_velocity(
    *,
    residual: torch.Tensor,
    writer_map: torch.Tensor,
    metric: torch.Tensor,
    metric_pinv: torch.Tensor,
    barrier_pullback: torch.Tensor,
    keys: torch.Tensor,
    history_cache: torch.Tensor,
    l2: float,
) -> ProjectedVelocity:
    """Project native residual velocity onto ``<U,X> <= 0`` in the M metric."""

    residual = _require_fp32_matrix("R", residual)
    writer_map = _require_fp32_matrix("J", writer_map)
    metric = _require_fp32_matrix("M", metric)
    metric_pinv = _require_fp32_matrix("M_pinv", metric_pinv)
    barrier_pullback = _require_fp32_matrix("U", barrier_pullback)
    keys = _require_fp32_matrix("K", keys)
    history_cache = _require_fp32_matrix("C", history_cache)
    if residual.shape != barrier_pullback.shape:
        raise GeometryBoundary("R/U shape mismatch")
    if metric.shape != (residual.shape[1], residual.shape[1]):
        raise GeometryBoundary("M/R shape mismatch")

    native_rate_tensor = torch.sum(barrier_pullback * residual)
    actionable_normal = barrier_pullback @ metric_pinv
    eta_tensor = torch.sum(actionable_normal * barrier_pullback)
    native_rate = float(native_rate_tensor.item())
    eta = float(eta_tensor.item())
    eta_tolerance = 8.0 * _gamma(max(metric.shape)) * max(
        float(torch.linalg.vector_norm(barrier_pullback).item()) ** 2,
        torch.finfo(torch.float32).tiny,
    )
    if eta < -eta_tolerance:
        raise GeometryBoundary(f"normal energy is meaningfully negative: {eta:.9e}")
    if eta < 0.0:
        eta = 0.0
        eta_tensor = eta_tensor.new_zeros(())

    if native_rate > 0.0:
        if eta == 0.0:
            raise ActionableNormalBoundary(
                "ACTIONABLE_QKL_NORMAL_ABSENT: c>0 with eta=0"
            )
        correction = -(native_rate_tensor / eta_tensor) * actionable_normal
        active = True
    else:
        correction = torch.zeros_like(residual)
        active = False
    projected_residual = residual + correction
    projected_velocity = projected_residual @ writer_map.T
    projected_rate = float(torch.sum(barrier_pullback * projected_residual).item())
    expected_rate = min(native_rate, 0.0)
    positive_violation = max(projected_rate - max(expected_rate, 0.0), 0.0)

    correction_energy_tensor = torch.sum((correction @ metric) * correction)
    native_energy_tensor = torch.sum((residual @ metric) * residual)
    correction_energy = max(float(correction_energy_tensor.item()), 0.0)
    native_energy = max(float(native_energy_tensor.item()), 0.0)
    removed_fraction = correction_energy / native_energy if native_energy > 0.0 else 0.0

    correction_weight = correction @ writer_map.T
    current_key_energy = float(torch.sum((correction_weight @ keys) ** 2).item())
    history_cache_energy = float(
        torch.sum((correction_weight @ history_cache) * correction_weight).item()
    )
    l2_energy = float(l2) * float(torch.sum(correction_weight**2).item())
    decomposition = current_key_energy + history_cache_energy + l2_energy
    energy_tolerance = 8.0 * _gamma(keys.shape[0]) * max(
        correction_energy, decomposition, torch.finfo(torch.float32).tiny
    )
    if abs(decomposition - correction_energy) > energy_tolerance:
        raise GeometryBoundary(
            "correction energy decomposition mismatch: "
            f"metric={correction_energy:.9e}, terms={decomposition:.9e}, "
            f"tol={energy_tolerance:.9e}"
        )
    if not all(
        torch.isfinite(value).all()
        for value in (projected_residual, projected_velocity, correction)
    ):
        raise GeometryBoundary("projected velocity is nonfinite")
    return ProjectedVelocity(
        residual_velocity=projected_residual,
        weight_velocity=projected_velocity,
        residual_correction=correction,
        native_rate=native_rate,
        eta=eta,
        projected_rate=projected_rate,
        positive_projected_rate_violation=positive_violation,
        correction_energy=correction_energy,
        removed_energy_fraction=removed_fraction,
        native_energy=native_energy,
        current_key_energy=current_key_energy,
        history_cache_energy=history_cache_energy,
        l2_energy=l2_energy,
        active=active,
    )


def apply_velocity_(weight: torch.Tensor, velocity: torch.Tensor, *, step_size: float) -> None:
    """Apply one Euler step; ``step_size`` appears exactly once here."""

    if velocity.shape != weight.shape:
        raise GeometryBoundary("velocity/weight shape mismatch at module boundary")
    if velocity.dtype != torch.float32 or weight.dtype != torch.float32:
        raise GeometryBoundary("Euler write must remain FULL-FP32")
    with torch.no_grad():
        weight.add_(velocity, alpha=float(step_size))
