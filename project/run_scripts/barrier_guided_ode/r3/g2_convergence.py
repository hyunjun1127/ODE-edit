"""Outcome-independent BGODE-R3 G2 Cauchy convergence gate."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

import torch

from .errors import R3ScientificBoundary


N_GRID = (4, 8, 16, 32)
FIRST_ORDER_RATIO_FLOOR = 1.25
FINAL_RELATIVE_PHYSICAL_DISTANCE_CEILING = 0.05
FINAL_NORMALIZED_TARGET_LOGIT_GAP_CEILING = 0.05
FINAL_NORMALIZED_Q_KL_GAP_CEILING = 0.05
ROUNDING_MULTIPLIER = 256.0


@dataclass(frozen=True, slots=True)
class G2Endpoint:
    step_count: int
    terminal_displacement: float
    target_logit: float
    q0_conditional_kl: float

    def __post_init__(self) -> None:
        if self.step_count not in N_GRID:
            raise R3ScientificBoundary("G2 endpoint step count is outside the predeclared grid")
        values = (self.terminal_displacement, self.target_logit, self.q0_conditional_kl)
        if any(not math.isfinite(value) for value in values):
            raise R3ScientificBoundary("G2 endpoint scalar is non-finite")
        if self.terminal_displacement < 0.0 or self.q0_conditional_kl < 0.0:
            raise R3ScientificBoundary("G2 endpoint norm/KL is negative")


@dataclass(frozen=True, slots=True)
class G2PairDistance:
    lower_n: int
    upper_n: int
    physical_distance: float
    target_logit_gap: float
    q_kl_gap: float

    def __post_init__(self) -> None:
        if self.upper_n != 2 * self.lower_n or self.lower_n not in N_GRID[:-1]:
            raise R3ScientificBoundary("G2 successive pair is not canonical")
        values = (self.physical_distance, self.target_logit_gap, self.q_kl_gap)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise R3ScientificBoundary("G2 pair distance is invalid")


@dataclass(frozen=True, slots=True)
class G2CauchyReceipt:
    pairs: tuple[G2PairDistance, ...]
    physical_ratios: tuple[float | None, float | None]
    physical_contraction: bool
    target_logit_contraction: bool
    q_kl_contraction: bool
    first_order_trend: bool
    final_relative_physical_distance: float
    final_normalized_target_logit_gap: float
    final_normalized_q_kl_gap: float
    rounding_tolerance: float
    passed: bool

    def payload(self) -> dict[str, object]:
        return asdict(self)


def block_distance(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
) -> float:
    if not left or len(left) != len(right):
        raise R3ScientificBoundary("G2 physical endpoint block inventories differ")
    energy = 0.0
    for before, after in zip(left, right, strict=True):
        if (
            before.device.type != "cpu"
            or after.device.type != "cpu"
            or before.dtype != torch.float32
            or after.dtype != torch.float32
            or before.shape != after.shape
            or before.requires_grad
            or after.requires_grad
            or not bool(torch.isfinite(before).all())
            or not bool(torch.isfinite(after).all())
        ):
            raise R3ScientificBoundary("G2 physical endpoint block is not detached finite CPU FP32")
        difference = before.to(dtype=torch.float64) - after.to(dtype=torch.float64)
        energy += float(torch.sum(difference.square()).item())
    result = math.sqrt(energy)
    if not math.isfinite(result):
        raise R3ScientificBoundary("G2 physical endpoint distance is non-finite")
    return result


def _ratio(numerator: float, denominator: float, tolerance: float) -> float | None:
    if numerator <= tolerance and denominator <= tolerance:
        return None
    if denominator <= tolerance:
        return math.inf
    return numerator / denominator


def assess_cauchy(
    endpoints: Mapping[int, G2Endpoint],
    physical_distances: Mapping[tuple[int, int], float],
) -> G2CauchyReceipt:
    if tuple(sorted(endpoints)) != N_GRID:
        raise R3ScientificBoundary("G2 endpoint grid is incomplete")
    expected_pairs = tuple((value, 2 * value) for value in N_GRID[:-1])
    if tuple(sorted(physical_distances)) != expected_pairs:
        raise R3ScientificBoundary("G2 physical pair grid is incomplete")
    rows: list[G2PairDistance] = []
    for lower, upper in expected_pairs:
        before = endpoints[lower]
        after = endpoints[upper]
        rows.append(
            G2PairDistance(
                lower_n=lower,
                upper_n=upper,
                physical_distance=float(physical_distances[(lower, upper)]),
                target_logit_gap=abs(after.target_logit - before.target_logit),
                q_kl_gap=abs(after.q0_conditional_kl - before.q0_conditional_kl),
            )
        )
    scale = 1.0 + max(
        *(row.physical_distance for row in rows),
        *(row.target_logit_gap for row in rows),
        *(row.q_kl_gap for row in rows),
        endpoints[32].terminal_displacement,
        abs(endpoints[32].target_logit),
        endpoints[32].q0_conditional_kl,
    )
    tolerance = ROUNDING_MULTIPLIER * torch.finfo(torch.float32).eps * scale
    physical = tuple(row.physical_distance for row in rows)
    target = tuple(row.target_logit_gap for row in rows)
    q_kl = tuple(row.q_kl_gap for row in rows)
    physical_ratios = (
        _ratio(physical[0], physical[1], tolerance),
        _ratio(physical[1], physical[2], tolerance),
    )
    physical_contraction = physical[1] <= physical[0] + tolerance and physical[2] <= physical[1] + tolerance
    target_contraction = target[1] <= target[0] + tolerance and target[2] <= target[1] + tolerance
    q_kl_contraction = q_kl[1] <= q_kl[0] + tolerance and q_kl[2] <= q_kl[1] + tolerance
    first_order = all(
        ratio is None or ratio >= FIRST_ORDER_RATIO_FLOOR for ratio in physical_ratios
    )
    final_relative = physical[2] / (tolerance + endpoints[32].terminal_displacement)
    final_target = target[2] / (1.0 + abs(endpoints[32].target_logit))
    final_q_kl = q_kl[2] / (1.0 + endpoints[32].q0_conditional_kl)
    passed = all(
        (
            physical_contraction,
            target_contraction,
            q_kl_contraction,
            first_order,
            final_relative <= FINAL_RELATIVE_PHYSICAL_DISTANCE_CEILING,
            final_target <= FINAL_NORMALIZED_TARGET_LOGIT_GAP_CEILING,
            final_q_kl <= FINAL_NORMALIZED_Q_KL_GAP_CEILING,
        )
    )
    return G2CauchyReceipt(
        pairs=tuple(rows),
        physical_ratios=physical_ratios,
        physical_contraction=physical_contraction,
        target_logit_contraction=target_contraction,
        q_kl_contraction=q_kl_contraction,
        first_order_trend=first_order,
        final_relative_physical_distance=final_relative,
        final_normalized_target_logit_gap=final_target,
        final_normalized_q_kl_gap=final_q_kl,
        rounding_tolerance=tolerance,
        passed=passed,
    )


__all__ = [
    "FINAL_NORMALIZED_Q_KL_GAP_CEILING",
    "FINAL_NORMALIZED_TARGET_LOGIT_GAP_CEILING",
    "FINAL_RELATIVE_PHYSICAL_DISTANCE_CEILING",
    "FIRST_ORDER_RATIO_FLOOR",
    "G2CauchyReceipt",
    "G2Endpoint",
    "G2PairDistance",
    "N_GRID",
    "assess_cauchy",
    "block_distance",
]
