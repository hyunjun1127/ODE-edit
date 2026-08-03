"""Outcome-free QP, trust, scalar-search, and Omega primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .contracts import (
    ControllerConfig,
    EventReading,
    MethodContractError,
    QPSolution,
    TrustVerdict,
    canonical_hash,
    finite,
)


def _qp_coefficients(
    slopes: Sequence[float],
    costs: Sequence[float],
    progress: float,
    trust_dual: float,
) -> tuple[float, ...]:
    denominator = math.fsum(
        slope * slope / (cost + trust_dual)
        for slope, cost in zip(slopes, costs, strict=True)
        if slope > 0.0
    )
    if denominator <= 0.0:
        return tuple(0.0 for _ in slopes)
    return tuple(
        0.0
        if slope <= 0.0
        else progress * slope / (cost + trust_dual) / denominator
        for slope, cost in zip(slopes, costs, strict=True)
    )


def solve_progress_qp(
    *,
    layers: Sequence[int],
    slopes: Sequence[float],
    omega: Mapping[int, float],
    denominators: Mapping[int, float],
    hard_phi: float,
    trust_radius: float,
    config: ControllerConfig,
) -> QPSolution:
    """Solve the exact diagonal minimum-Omega progress-equality QP.

    The objective is ``sum_l ((1+Omega_l)/D_l) y_l^2``.  There is no signed
    cross term, hard capacity barrier, or post-solver radial rescaling.
    """

    locked_layers = tuple(int(layer) for layer in layers)
    locked_slopes = tuple(
        0.0 if finite("slope", value) <= config.slope_epsilon else float(value)
        for value in slopes
    )
    if (
        not locked_layers
        or len(locked_layers) != len(locked_slopes)
        or len(set(locked_layers)) != len(locked_layers)
    ):
        raise MethodContractError("QP layers/slopes differ")
    radius = finite("trust radius", trust_radius)
    if radius <= 0.0:
        raise MethodContractError("trust radius must be positive")
    costs: list[float] = []
    for layer in locked_layers:
        try:
            load = finite("Omega", omega[layer])
            denominator = finite("D", denominators[layer])
        except KeyError as exc:
            raise MethodContractError("QP geometry mapping is incomplete") from exc
        if load < 0.0 or denominator <= config.load_denominator_epsilon:
            raise MethodContractError("Omega/D geometry is invalid")
        costs.append((1.0 + load) / denominator)
    slope_norm = math.sqrt(math.fsum(value * value for value in locked_slopes))
    maximum = radius * slope_norm
    deficit = max(0.0, finite("hard event", hard_phi))
    requested = min(config.kappa * deficit, config.beta * maximum)
    if requested <= config.progress_epsilon or slope_norm <= config.slope_epsilon:
        coefficients = tuple(0.0 for _ in locked_layers)
        return QPSolution(
            layers=locked_layers,
            coefficients=coefficients,
            requested_progress=0.0,
            predicted_progress=0.0,
            maximum_progress=maximum,
            coefficient_norm=0.0,
            equality_residual=0.0,
            objective=0.0,
            trust_dual=0.0,
        )

    coefficients = _qp_coefficients(locked_slopes, costs, requested, 0.0)
    norm = math.sqrt(math.fsum(value * value for value in coefficients))
    trust_dual = 0.0
    if norm > radius:
        low = 0.0
        high = 1.0
        while True:
            candidate = _qp_coefficients(locked_slopes, costs, requested, high)
            if math.sqrt(math.fsum(value * value for value in candidate)) <= radius:
                break
            high *= 2.0
            if not math.isfinite(high):
                raise MethodContractError("QP trust dual overflowed")
        for _ in range(config.qp_bisection_iterations):
            middle = (low + high) / 2.0
            candidate = _qp_coefficients(locked_slopes, costs, requested, middle)
            if math.sqrt(math.fsum(value * value for value in candidate)) > radius:
                low = middle
            else:
                high = middle
        trust_dual = high
        coefficients = _qp_coefficients(locked_slopes, costs, requested, trust_dual)
        norm = math.sqrt(math.fsum(value * value for value in coefficients))

    predicted = math.fsum(
        slope * coefficient
        for slope, coefficient in zip(locked_slopes, coefficients, strict=True)
    )
    residual = abs(predicted - requested)
    if norm > radius + config.qp_trust_tolerance:
        raise MethodContractError("QP output exceeds trust radius")
    if residual > config.qp_equality_tolerance:
        raise MethodContractError("QP output violates progress equality")
    objective = math.fsum(
        cost * coefficient * coefficient
        for cost, coefficient in zip(costs, coefficients, strict=True)
    )
    return QPSolution(
        layers=locked_layers,
        coefficients=coefficients,
        requested_progress=requested,
        predicted_progress=predicted,
        maximum_progress=maximum,
        coefficient_norm=norm,
        equality_residual=residual,
        objective=objective,
        trust_dual=trust_dual,
    )


def static_coefficients(
    *,
    layers: Sequence[int],
    frozen_share: Sequence[float],
    frozen_slopes: Sequence[float],
    hard_phi: float,
    trust_radius: float,
    config: ControllerConfig,
) -> QPSolution:
    """Change only the global magnitude of a frozen entry direction/share."""

    locked_layers = tuple(int(layer) for layer in layers)
    share = tuple(finite("static share", value) for value in frozen_share)
    slopes = tuple(finite("static slope", value) for value in frozen_slopes)
    if (
        len(locked_layers) != len(share)
        or len(share) != len(slopes)
        or any(value < 0.0 for value in share)
        or any(value < 0.0 for value in slopes)
    ):
        raise MethodContractError("static direction dimensions are invalid")
    share_norm = math.sqrt(math.fsum(value * value for value in share))
    if not math.isclose(share_norm, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise MethodContractError("static share must have unit Euclidean norm")
    projected_slope = math.fsum(
        slope * value for slope, value in zip(slopes, share, strict=True)
    )
    radius = finite("trust radius", trust_radius)
    maximum = radius * projected_slope
    requested = min(
        config.kappa * max(0.0, finite("hard event", hard_phi)),
        config.beta * maximum,
    )
    magnitude = 0.0 if projected_slope <= config.slope_epsilon else requested / projected_slope
    coefficients = tuple(magnitude * value for value in share)
    predicted = math.fsum(
        slope * coefficient
        for slope, coefficient in zip(slopes, coefficients, strict=True)
    )
    residual = abs(predicted - requested)
    if residual > config.qp_equality_tolerance or magnitude > radius + config.qp_trust_tolerance:
        raise MethodContractError("static magnitude violates common progress/trust")
    return QPSolution(
        layers=locked_layers,
        coefficients=coefficients,
        requested_progress=requested,
        predicted_progress=predicted,
        maximum_progress=maximum,
        coefficient_norm=magnitude,
        equality_residual=residual,
        objective=0.0,
        trust_dual=0.0,
    )


def normalized_share(coefficients: Sequence[float], epsilon: float) -> tuple[float, ...]:
    values = tuple(finite("coefficient", value) for value in coefficients)
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm <= finite("share epsilon", epsilon):
        raise MethodContractError("cannot freeze a zero static share")
    return tuple(value / norm for value in values)


def assess_trial(
    *,
    before: EventReading,
    after: EventReading,
    predicted_progress: float,
    trust_radius: float,
    trust_radius_cap: float,
    config: ControllerConfig,
) -> TrustVerdict:
    predicted = finite("predicted progress", predicted_progress)
    radius = finite("trust radius", trust_radius)
    actual = before.smooth_phi - after.smooth_phi
    ratio = actual / (predicted + config.trust_denominator_epsilon)
    hard_worsened = (
        after.hard_phi > before.hard_phi + config.hard_worsening_tolerance
    )
    if predicted <= config.progress_epsilon:
        accepted, reason = False, "no-predicted-progress"
    elif actual <= 0.0:
        accepted, reason = False, "nonpositive-actual-progress"
    elif ratio < config.eta_reject:
        accepted, reason = False, "trust-ratio-reject"
    elif hard_worsened:
        accepted, reason = False, "hard-event-worsened"
    else:
        accepted, reason = True, "accepted"
    if not accepted:
        next_radius = radius * config.gamma_down
    elif ratio >= config.eta_expand:
        cap = finite("trust radius cap", trust_radius_cap)
        if cap <= 0.0 or radius > cap + config.qp_trust_tolerance:
            raise MethodContractError("trust radius/cap relation is invalid")
        next_radius = min(cap, radius * config.gamma_up)
    else:
        next_radius = radius
    return TrustVerdict(
        accepted=accepted,
        hit=accepted and after.is_hit(config.event_tolerance),
        reason=reason,
        actual_progress=actual,
        predicted_progress=predicted,
        ratio=ratio,
        next_radius=next_radius,
    )


@dataclass(frozen=True, slots=True)
class ScalarProbe:
    alpha: float
    hard_phi: float


@dataclass(frozen=True, slots=True)
class ScalarSearchResult:
    hit: bool
    alpha: float | None
    bracket: tuple[float, float] | None
    probes: tuple[ScalarProbe, ...]
    nonmonotonic: bool


class ScalarFirstHitSearch:
    """Grid-first bracket followed only by within-bracket bisection."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config

    def run(self, evaluate_hard_phi: Callable[[float], float]) -> ScalarSearchResult:
        observed: dict[float, float] = {}

        def evaluate(alpha: float) -> float:
            if alpha < 0.0 or alpha > 1.0:
                raise MethodContractError("scalar alpha outside [0,1]")
            if alpha not in observed:
                observed[alpha] = finite("scalar hard event", evaluate_hard_phi(alpha))
            return observed[alpha]

        bracket: tuple[float, float] | None = None
        previous_alpha = self.config.scalar_grid[0]
        previous_phi = evaluate(previous_alpha)
        if previous_phi <= self.config.event_tolerance:
            bracket = (0.0, 0.0)
        else:
            for alpha in self.config.scalar_grid[1:]:
                phi = evaluate(alpha)
                if phi <= self.config.event_tolerance:
                    bracket = (previous_alpha, alpha)
                    break
                previous_alpha, previous_phi = alpha, phi

        selected: float | None = None
        if bracket is not None:
            low, high = bracket
            selected = high
            if low != high:
                for _ in range(self.config.scalar_bisection_iterations):
                    if high - low <= self.config.scalar_bisection_tolerance:
                        break
                    middle = (low + high) / 2.0
                    if not bracket[0] <= middle <= bracket[1]:
                        raise MethodContractError("scalar bisection escaped its first bracket")
                    if evaluate(middle) <= self.config.event_tolerance:
                        high = middle
                        selected = middle
                    else:
                        low = middle

        sorted_probes = tuple(
            ScalarProbe(alpha=alpha, hard_phi=observed[alpha])
            for alpha in sorted(observed)
        )
        nonmonotonic = any(
            right.hard_phi > left.hard_phi + self.config.scalar_nonmonotonic_tolerance
            for left, right in zip(sorted_probes, sorted_probes[1:])
        )
        return ScalarSearchResult(
            hit=selected is not None,
            alpha=selected,
            bracket=bracket,
            probes=sorted_probes,
            nonmonotonic=nonmonotonic,
        )


@dataclass(frozen=True, slots=True)
class OmegaReceipt:
    edit_id: str
    terminal_net_energy: tuple[tuple[int, float], ...]
    increment: tuple[tuple[int, float], ...]
    receipt_id: str


class OmegaLedger:
    """Append terminal net-write energy exactly once per completed outer edit."""

    def __init__(self, denominators: Mapping[int, float]) -> None:
        if not denominators:
            raise MethodContractError("Omega ledger needs layer denominators")
        self.denominators = {
            int(layer): finite("Omega denominator", value)
            for layer, value in denominators.items()
        }
        if any(value <= 0.0 for value in self.denominators.values()):
            raise MethodContractError("Omega denominators must be positive")
        self._omega = {layer: 0.0 for layer in self.denominators}
        self._receipts: list[OmegaReceipt] = []
        self._edit_ids: set[str] = set()

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.denominators))

    def frozen_for_edit(self, edit_id: str) -> dict[int, float]:
        if not edit_id or edit_id in self._edit_ids:
            raise MethodContractError("Omega edit identity is empty or already completed")
        return dict(self._omega)

    def append_terminal(
        self, edit_id: str, terminal_net_energy: Mapping[int, float]
    ) -> OmegaReceipt:
        if not edit_id or edit_id in self._edit_ids:
            raise MethodContractError("Omega may append an outer edit only once")
        if set(terminal_net_energy) != set(self.denominators):
            raise MethodContractError("terminal net-write layers differ from Omega geometry")
        energy = tuple(
            (layer, finite("terminal net-write energy", terminal_net_energy[layer]))
            for layer in self.layers
        )
        if any(value < 0.0 for _layer, value in energy):
            raise MethodContractError("terminal net-write energy must be non-negative")
        increment = tuple(
            (layer, value / self.denominators[layer]) for layer, value in energy
        )
        payload = {
            "edit_id": edit_id,
            "terminal_net_energy": dict(energy),
            "increment": dict(increment),
        }
        receipt = OmegaReceipt(
            edit_id=edit_id,
            terminal_net_energy=energy,
            increment=increment,
            receipt_id=canonical_hash(payload),
        )
        for layer, value in increment:
            self._omega[layer] += value
        self._edit_ids.add(edit_id)
        self._receipts.append(receipt)
        return receipt

    def state(self) -> dict[int, float]:
        return dict(self._omega)

    @property
    def receipts(self) -> tuple[OmegaReceipt, ...]:
        return tuple(self._receipts)
