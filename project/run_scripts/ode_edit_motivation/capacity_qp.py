"""Small reusable capacity-aware QP for ODE-Edit layer routing.

The controller works entirely in the editor-provided low-rank subspace.  A
candidate for layer ``l`` is C-normalized, so a non-negative coefficient
``x_l`` is its C-distance.  With cumulative displacement ``Delta_l`` and W0
normalizer ``D_l``, the exact next normalized load is

``Psi_l + linear_l*x_l + quadratic_l*x_l**2``.

The five-dimensional convex QP minimizes the summed next-load increment while
meeting a rewrite-only first-order progress request.  A Euclidean trust region
and per-layer barrier caps are enforced by nested scalar dual bisections; no
general-purpose optimizer or outcome/evaluation field is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Mapping, Sequence

import torch

from .contracts import ContractError, LowRankFactor, MemitFactorProposal
from .diagnostic_math import c_inner_product, c_squared_norm
from .mv1_calibration import ActionDirection


CAPACITY_QP_SCHEMA = "ode-edit-capacity-qp/v1"
DUAL_ITERATIONS = 96
PROGRESS_TOLERANCE = 1e-9
VELOCITY_REGULARIZER = 1e-12
BARRIER_HEADROOM_STEPS = 2.0


class CapacityQPError(ContractError):
    """A capacity state or QP solution violates the controller contract."""


def _cpu_metric_factor(factor: LowRankFactor) -> LowRankFactor:
    """Match the existing capacity metric's pinned CPU/float32 convention."""

    return LowRankFactor(
        weight_name=factor.weight_name,
        left=factor.left.detach().cpu().float(),
        right=factor.right.detach().cpu().float(),
        expected_weight_sha256=factor.expected_weight_sha256,
        native_update_transposed=factor.native_update_transposed,
    )


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise CapacityQPError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise CapacityQPError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class CapacityLayerTerms:
    """Exact scalar capacity polynomial for one layer candidate."""

    layer: int
    action_id: str
    weight_name: str
    slope: float
    psi_before: float
    linear: float
    quadratic: float
    barrier: float
    coefficient_cap: float
    overloaded: bool

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, int):
            raise CapacityQPError("capacity layer must be an integer")
        if not self.action_id or not self.weight_name:
            raise CapacityQPError("capacity layer identity is empty")
        for name in (
            "slope",
            "psi_before",
            "linear",
            "quadratic",
            "barrier",
            "coefficient_cap",
        ):
            _finite(name, getattr(self, name))
        if self.slope < 0.0 or self.psi_before < 0.0:
            raise CapacityQPError("slope and capacity state must be non-negative")
        if self.quadratic <= 0.0 or self.barrier < self.psi_before:
            raise CapacityQPError("capacity quadratic/barrier is invalid")
        if self.coefficient_cap < 0.0:
            raise CapacityQPError("capacity coefficient cap is negative")
        if type(self.overloaded) is not bool:
            raise CapacityQPError("overloaded must be boolean")

    def next_psi(self, coefficient: float) -> float:
        value = _finite("coefficient", coefficient)
        if value < -PROGRESS_TOLERANCE or value > self.coefficient_cap + 1e-8:
            raise CapacityQPError("coefficient is outside the layer cap")
        return self.psi_before + self.linear * value + self.quadratic * value * value

    def to_dict(self) -> dict[str, int | str | float | bool]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class CapacityQPSolution:
    """Outcome-free solution of one progress-constrained layer QP."""

    coefficients: tuple[float, ...]
    requested_progress: float
    constrained_progress: float
    predicted_progress: float
    progress_slack: float
    trust_distance: float
    coefficient_norm: float
    predicted_capacity_increment: float
    predicted_capacity_after: float
    progress_dual: float
    trust_dual: float
    feasible_without_slack: bool

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise CapacityQPError("QP solution has no coefficients")
        if any(not math.isfinite(value) or value < 0.0 for value in self.coefficients):
            raise CapacityQPError("QP coefficients must be finite and non-negative")
        for name in (
            "requested_progress",
            "constrained_progress",
            "predicted_progress",
            "progress_slack",
            "trust_distance",
            "coefficient_norm",
            "predicted_capacity_increment",
            "predicted_capacity_after",
            "progress_dual",
            "trust_dual",
        ):
            _finite(name, getattr(self, name))
        if min(
            self.requested_progress,
            self.constrained_progress,
            self.predicted_progress,
            self.progress_slack,
            self.trust_distance,
            self.coefficient_norm,
            self.predicted_capacity_after,
            self.progress_dual,
            self.trust_dual,
        ) < 0.0:
            raise CapacityQPError("QP solution contains a negative magnitude")
        if self.coefficient_norm > self.trust_distance + 1e-7:
            raise CapacityQPError("QP solution exceeds its trust distance")
        if type(self.feasible_without_slack) is not bool:
            raise CapacityQPError("QP feasibility marker must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            name: (list(value) if isinstance(value, tuple) else value)
            for name, value in (
                (name, getattr(self, name))
                for name in self.__dataclass_fields__
            )
        }


def _factor_energy_sum(
    factors: Sequence[LowRankFactor], covariance: torch.Tensor
) -> float:
    if not factors:
        return 0.0
    metric = covariance.detach().cpu().float()
    normalized = tuple(_cpu_metric_factor(factor) for factor in factors)
    value = torch.zeros((), dtype=torch.float64)
    for first in normalized:
        for second in normalized:
            term = c_inner_product(first, second, metric)
            value += term.to(dtype=torch.float64)
    result = float(value.detach().cpu())
    tolerance = 1e-6 * max(1.0, abs(result))
    if not math.isfinite(result) or result < -tolerance:
        raise CapacityQPError("cumulative C-energy is invalid")
    return max(0.0, result)


def _factor_cross_sum(
    factors: Sequence[LowRankFactor],
    candidate: LowRankFactor,
    covariance: torch.Tensor,
) -> float:
    metric = covariance.detach().cpu().float()
    normalized_candidate = _cpu_metric_factor(candidate)
    value = 0.0
    for factor in factors:
        if factor.weight_shape != candidate.weight_shape:
            raise CapacityQPError("cumulative and candidate factor shapes differ")
        value += float(
            c_inner_product(
                _cpu_metric_factor(factor),
                normalized_candidate,
                metric,
            ).detach()
        )
    if not math.isfinite(value):
        raise CapacityQPError("cumulative/candidate C-cross term is invalid")
    return value


def _positive_barrier_root(
    *, psi: float, linear: float, quadratic: float, barrier: float
) -> float:
    if barrier < psi or quadratic <= 0.0:
        raise CapacityQPError("barrier root inputs are invalid")
    discriminant = linear * linear + 4.0 * quadratic * (barrier - psi)
    if discriminant < -1e-12:
        raise CapacityQPError("capacity barrier has no real root")
    return max(0.0, (-linear + math.sqrt(max(0.0, discriminant))) / (2.0 * quadratic))


def build_capacity_layer_terms(
    *,
    cumulative_factors: Sequence[LowRankFactor],
    unit_actions: Sequence[ActionDirection],
    slopes: Mapping[str, float],
    covariance_by_layer: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
    w0_denominators: Mapping[int, float],
    trust_distance: float,
) -> tuple[CapacityLayerTerms, ...]:
    """Build exact layer polynomials and a common normalized-load frontier."""

    distance = _finite("trust_distance", trust_distance)
    if distance <= 0.0 or not unit_actions:
        raise CapacityQPError("capacity terms require actions and positive trust distance")
    cumulative_by_weight: dict[str, list[LowRankFactor]] = {}
    for factor in cumulative_factors:
        cumulative_by_weight.setdefault(factor.weight_name, []).append(factor)

    provisional: list[dict[str, object]] = []
    for action in unit_actions:
        if len(action.proposal.factors) != 1:
            raise CapacityQPError("capacity QP requires one single-layer factor per action")
        factor = action.proposal.factors[0]
        try:
            layer = int(layer_by_weight[factor.weight_name])
            covariance = covariance_by_layer[layer]
            denominator = float(w0_denominators[layer])
            raw_slope = float(slopes[action.action_id])
        except (KeyError, TypeError, ValueError) as exc:
            raise CapacityQPError("capacity action mapping is incomplete") from exc
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise CapacityQPError("W0 capacity denominator must be positive")
        selected = cumulative_by_weight.get(factor.weight_name, ())
        energy = _factor_energy_sum(selected, covariance)
        cross = _factor_cross_sum(selected, factor, covariance)
        unit_energy = float(
            c_squared_norm(
                _cpu_metric_factor(factor),
                covariance.detach().cpu().float(),
            ).detach()
        )
        if not math.isfinite(unit_energy) or unit_energy <= 0.0:
            raise CapacityQPError("candidate C-energy must be positive")
        if not math.isclose(unit_energy, 1.0, rel_tol=2e-5, abs_tol=2e-5):
            raise CapacityQPError("capacity candidate must be unit-C normalized")
        provisional.append(
            {
                "layer": layer,
                "action_id": action.action_id,
                "weight_name": factor.weight_name,
                "slope": max(0.0, _finite("slope", raw_slope)),
                "psi_before": energy / denominator,
                "linear": 2.0 * cross / denominator,
                "quadratic": unit_energy / denominator,
            }
        )
    if len({int(item["layer"]) for item in provisional}) != len(provisional):
        raise CapacityQPError("capacity actions must map one-to-one to layers")

    psi_median = median(float(item["psi_before"]) for item in provisional)
    equal_distance = distance / math.sqrt(len(provisional))
    increments = [
        float(item["quadratic"]) * equal_distance * equal_distance
        for item in provisional
    ]
    common_frontier = psi_median + BARRIER_HEADROOM_STEPS * median(increments)
    terms: list[CapacityLayerTerms] = []
    for item in provisional:
        psi = float(item["psi_before"])
        linear = float(item["linear"])
        quadratic = float(item["quadratic"])
        overloaded = psi > common_frontier + 1e-15
        if overloaded:
            # A genuinely overloaded layer may reduce its load, but cannot
            # finish this step with a larger load than it started with.
            barrier = psi
            cap = min(
                distance,
                _positive_barrier_root(
                    psi=psi,
                    linear=linear,
                    quadratic=quadratic,
                    barrier=barrier,
                ),
            )
        else:
            # The common frontier detects overload; it is not a global
            # magnitude throttle.  Non-overloaded layers retain the complete
            # hop envelope.  A convex quadratic reaches its maximum on
            # [0, distance] at an endpoint, so this barrier is exact.
            cap = distance
            barrier = max(
                psi,
                psi + linear * distance + quadratic * distance * distance,
            )
        terms.append(
            CapacityLayerTerms(
                layer=int(item["layer"]),
                action_id=str(item["action_id"]),
                weight_name=str(item["weight_name"]),
                slope=float(item["slope"]),
                psi_before=psi,
                linear=linear,
                quadratic=quadratic,
                barrier=barrier,
                coefficient_cap=max(0.0, cap),
                overloaded=overloaded,
            )
        )
    return tuple(terms)


def _max_progress_coefficients(
    terms: Sequence[CapacityLayerTerms], trust_distance: float
) -> tuple[float, ...]:
    slopes = tuple(term.slope for term in terms)
    caps = tuple(term.coefficient_cap for term in terms)
    if not any(slope > 0.0 and cap > 0.0 for slope, cap in zip(slopes, caps)):
        return tuple(0.0 for _ in terms)
    active_caps = tuple(
        cap if slope > 0.0 else 0.0 for slope, cap in zip(slopes, caps)
    )
    cap_norm = math.sqrt(math.fsum(cap * cap for cap in active_caps))
    if cap_norm <= trust_distance:
        return active_caps

    def coefficients(multiplier: float) -> tuple[float, ...]:
        return tuple(
            min(cap, slope / multiplier) if slope > 0.0 else 0.0
            for slope, cap in zip(slopes, caps)
        )

    low = 0.0
    high = 1.0
    while math.sqrt(math.fsum(value * value for value in coefficients(high))) > trust_distance:
        high *= 2.0
        if not math.isfinite(high):
            raise CapacityQPError("max-progress trust dual overflowed")
    for _ in range(DUAL_ITERATIONS):
        middle = (low + high) / 2.0
        if math.sqrt(math.fsum(value * value for value in coefficients(middle))) > trust_distance:
            low = middle
        else:
            high = middle
    return coefficients(high)


def _solve_progress_dual(
    terms: Sequence[CapacityLayerTerms],
    target: float,
    trust_dual: float,
) -> tuple[tuple[float, ...], float]:
    def coefficients(progress_dual: float) -> tuple[float, ...]:
        return tuple(
            min(
                term.coefficient_cap,
                max(
                    0.0,
                    (progress_dual * term.slope - term.linear)
                    / (2.0 * term.quadratic + VELOCITY_REGULARIZER + 2.0 * trust_dual),
                ),
            )
            for term in terms
        )

    def progress(values: Sequence[float]) -> float:
        return math.fsum(term.slope * value for term, value in zip(terms, values))

    low = 0.0
    high = 1.0
    while progress(coefficients(high)) < target - PROGRESS_TOLERANCE:
        high *= 2.0
        if not math.isfinite(high):
            raise CapacityQPError("progress dual overflowed")
    for _ in range(DUAL_ITERATIONS):
        middle = (low + high) / 2.0
        if progress(coefficients(middle)) < target:
            low = middle
        else:
            high = middle
    return coefficients(high), high


def solve_capacity_qp(
    terms: Sequence[CapacityLayerTerms],
    *,
    requested_progress: float,
    trust_distance: float,
) -> CapacityQPSolution:
    """Solve the diagonal convex QP, exposing any unavoidable progress slack."""

    if not terms:
        raise CapacityQPError("capacity QP requires layer terms")
    request = _finite("requested_progress", requested_progress)
    distance = _finite("trust_distance", trust_distance)
    if request < 0.0 or distance <= 0.0:
        raise CapacityQPError("capacity QP progress/distance is invalid")
    max_coefficients = _max_progress_coefficients(terms, distance)
    maximum = math.fsum(
        term.slope * value for term, value in zip(terms, max_coefficients)
    )
    target = min(request, maximum)
    feasible = request <= maximum + PROGRESS_TOLERANCE
    if target <= PROGRESS_TOLERANCE:
        values = tuple(0.0 for _ in terms)
        progress_dual = 0.0
        trust_dual = 0.0
    else:
        values, progress_dual = _solve_progress_dual(terms, target, 0.0)
        norm = math.sqrt(math.fsum(value * value for value in values))
        trust_dual = 0.0
        if norm > distance + 1e-9:
            low = 0.0
            high = 1.0
            while True:
                candidate, _ = _solve_progress_dual(terms, target, high)
                candidate_norm = math.sqrt(math.fsum(value * value for value in candidate))
                if candidate_norm <= distance + 1e-10:
                    break
                high *= 2.0
                if not math.isfinite(high):
                    raise CapacityQPError("trust-region dual overflowed")
            for _ in range(DUAL_ITERATIONS):
                middle = (low + high) / 2.0
                candidate, _ = _solve_progress_dual(terms, target, middle)
                candidate_norm = math.sqrt(math.fsum(value * value for value in candidate))
                if candidate_norm > distance:
                    low = middle
                else:
                    high = middle
            trust_dual = high
            values, progress_dual = _solve_progress_dual(terms, target, trust_dual)

    predicted_progress = math.fsum(
        term.slope * value for term, value in zip(terms, values)
    )
    coefficient_norm = math.sqrt(math.fsum(value * value for value in values))
    increment = math.fsum(
        term.linear * value + term.quadratic * value * value
        for term, value in zip(terms, values)
    )
    after = math.fsum(term.next_psi(value) for term, value in zip(terms, values))
    if predicted_progress + 5e-8 < target:
        raise CapacityQPError("QP solution missed its constrained progress")
    return CapacityQPSolution(
        coefficients=tuple(float(value) for value in values),
        requested_progress=request,
        constrained_progress=target,
        predicted_progress=predicted_progress,
        progress_slack=max(0.0, request - predicted_progress),
        trust_distance=distance,
        coefficient_norm=coefficient_norm,
        predicted_capacity_increment=increment,
        predicted_capacity_after=after,
        progress_dual=progress_dual,
        trust_dual=trust_dual,
        feasible_without_slack=feasible,
    )


def build_capacity_qp_proposal(
    *,
    synchronous: MemitFactorProposal,
    unit_actions: Sequence[ActionDirection],
    terms: Sequence[CapacityLayerTerms],
    solution: CapacityQPSolution,
    solver_suffix: str,
) -> MemitFactorProposal | None:
    """Build a low-rank simultaneous proposal from solved layer distances."""

    if len(unit_actions) != len(terms) or len(terms) != len(solution.coefficients):
        raise CapacityQPError("QP proposal inputs differ in length")
    factors: list[LowRankFactor] = []
    for action, term, coefficient in zip(
        unit_actions, terms, solution.coefficients, strict=True
    ):
        action.proposal.assert_same_entry_snapshot(synchronous)
        if action.action_id != term.action_id or len(action.proposal.factors) != 1:
            raise CapacityQPError("QP action/term identity differs")
        factor = action.proposal.factors[0]
        if factor.weight_name != term.weight_name:
            raise CapacityQPError("QP factor/term weight differs")
        if coefficient > 0.0:
            factors.append(factor.scaled(coefficient))
    if not factors:
        return None
    return MemitFactorProposal(
        snapshot=synchronous.snapshot,
        factors=tuple(factors),
        semantics=synchronous.semantics,
        solver_name=f"{synchronous.solver_name}/capacity-qp/{solver_suffix}",
        residual_denominator=synchronous.residual_denominator,
    )


def exact_distance_capacity_coefficients(
    terms: Sequence[CapacityLayerTerms],
    coefficients: Sequence[float],
    *,
    target_distance: float,
) -> tuple[float, ...]:
    """Convert absolute QP coefficients into a box-feasible relative share.

    The QP determines *which layers write*.  This helper independently fixes
    *how much the joint update writes* by radially expanding that allocation to
    an exact C-distance.  Per-layer caps are preserved.  Failure to reach the
    requested distance is exposed as a contract error instead of silently
    returning an under-sized scientific endpoint.
    """

    distance = _finite("target_distance", target_distance)
    values = tuple(_finite("coefficient", value) for value in coefficients)
    if distance <= 0.0 or len(terms) != len(values):
        raise CapacityQPError("exact-distance allocation inputs are invalid")
    if any(value < 0.0 for value in values):
        raise CapacityQPError("exact-distance allocation is negative")
    if not any(value > 0.0 for value in values):
        raise CapacityQPError("exact-distance allocation has no positive share")

    reachable = math.sqrt(
        math.fsum(
            term.coefficient_cap * term.coefficient_cap
            for term, value in zip(terms, values, strict=True)
            if value > 0.0
        )
    )
    if reachable + PROGRESS_TOLERANCE < distance:
        raise CapacityQPError("QP share support cannot fill the exact hop distance")

    def radial(multiplier: float) -> tuple[float, ...]:
        return tuple(
            min(term.coefficient_cap, value * multiplier)
            for term, value in zip(terms, values, strict=True)
        )

    low = 0.0
    high = 1.0
    while math.sqrt(math.fsum(value * value for value in radial(high))) < distance:
        high *= 2.0
        if not math.isfinite(high):
            raise CapacityQPError("exact-distance radial scale overflowed")
    for _ in range(DUAL_ITERATIONS):
        middle = (low + high) / 2.0
        norm = math.sqrt(math.fsum(value * value for value in radial(middle)))
        if norm < distance:
            low = middle
        else:
            high = middle
    result = radial(high)
    norm = math.sqrt(math.fsum(value * value for value in result))
    if not math.isclose(norm, distance, rel_tol=1e-8, abs_tol=1e-10):
        raise CapacityQPError("exact-distance allocation missed its hop distance")
    return result


def build_capacity_proposal_from_coefficients(
    *,
    synchronous: MemitFactorProposal,
    unit_actions: Sequence[ActionDirection],
    terms: Sequence[CapacityLayerTerms],
    coefficients: Sequence[float],
    solver_suffix: str,
) -> MemitFactorProposal:
    """Build a simultaneous low-rank proposal from explicit layer distances."""

    values = tuple(float(value) for value in coefficients)
    if len(unit_actions) != len(terms) or len(terms) != len(values):
        raise CapacityQPError("capacity proposal inputs differ in length")
    factors: list[LowRankFactor] = []
    for action, term, coefficient in zip(unit_actions, terms, values, strict=True):
        action.proposal.assert_same_entry_snapshot(synchronous)
        if action.action_id != term.action_id or len(action.proposal.factors) != 1:
            raise CapacityQPError("capacity action/term identity differs")
        factor = action.proposal.factors[0]
        if factor.weight_name != term.weight_name:
            raise CapacityQPError("capacity factor/term weight differs")
        term.next_psi(coefficient)
        if coefficient > 0.0:
            factors.append(factor.scaled(coefficient))
    if not factors:
        raise CapacityQPError("capacity coefficient proposal is empty")
    return MemitFactorProposal(
        snapshot=synchronous.snapshot,
        factors=tuple(factors),
        semantics=synchronous.semantics,
        solver_name=f"{synchronous.solver_name}/capacity-share/{solver_suffix}",
        residual_denominator=synchronous.residual_denominator,
    )


def capacity_state_by_layer(
    *,
    cumulative_factors: Sequence[LowRankFactor],
    covariance_by_layer: Mapping[int, torch.Tensor],
    layer_by_weight: Mapping[str, int],
    w0_denominators: Mapping[int, float],
) -> dict[int, float]:
    """Return exact normalized cumulative load for compact controller metadata."""

    by_weight: dict[str, list[LowRankFactor]] = {}
    for factor in cumulative_factors:
        by_weight.setdefault(factor.weight_name, []).append(factor)
    result: dict[int, float] = {}
    for weight_name, layer in layer_by_weight.items():
        denominator = float(w0_denominators[layer])
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise CapacityQPError("capacity denominator is invalid")
        result[layer] = _factor_energy_sum(
            by_weight.get(weight_name, ()), covariance_by_layer[layer]
        ) / denominator
    return result


__all__ = [
    "BARRIER_HEADROOM_STEPS",
    "CAPACITY_QP_SCHEMA",
    "CapacityLayerTerms",
    "CapacityQPError",
    "CapacityQPSolution",
    "build_capacity_layer_terms",
    "build_capacity_proposal_from_coefficients",
    "build_capacity_qp_proposal",
    "capacity_state_by_layer",
    "exact_distance_capacity_coefficients",
    "solve_capacity_qp",
]
