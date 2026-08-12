"""P1R24-anchored coordinate-corrected semantic/lag coupling.

This module owns only the reusable P1R28 controller algebra.  Model capture,
field construction, materialization, evaluation, and transaction state remain
in the accepted P1 runtime.  The small coordinate wrappers deliberately make
it impossible to contract an applied coefficient with an applied-step slope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1_backend import P1DynamicField, SignedProgressReceipt
from .p1r24_atomic_strength import (
    P1R24_H,
    P1R24_NUMERICAL_EPSILON,
    P1R24_Q_EPSILON,
)
from .progress_simplex_routing import SIMPLEX_PRIMAL_TOLERANCE
from .scalable_batched_model import (
    ScalableObjectivePlan,
    ScalableObjectiveResult,
    evaluate_scalable_target_new_objective,
)
from .scalable_batched_runtime import P1R23_CONTEXTS_PER_REQUEST, P1R23_LAYER_ORDER


P1R28_INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R28-P1R24-ANCHORED-CORRECTED-COUPLING-KILL-TEST-V1"
)
P1R28_METHOD_ID = "P1R28-P1R24-ANCHORED-CORRECTED-COUPLING-V1"
P1R28_K = 8
P1R28_H = 1.0 / P1R28_K


def _vector(values: Sequence[float], *, label: str) -> np.ndarray:
    value = np.asarray(tuple(float(item) for item in values), dtype=np.float64)
    if value.shape != (5,) or not np.all(np.isfinite(value)):
        raise ODEBFContractError(f"P1R28 {label} geometry differs")
    return value


@dataclass(frozen=True, slots=True)
class RawCoefficientSlope:
    """Derivative with respect to a raw field coefficient ``c=theta``."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _vector(self.values, label="raw coefficient slope")

    def to_velocity(self) -> "VelocitySlope":
        return VelocitySlope(tuple(P1R28_H * item for item in self.values))

    def contract(self, coefficient: "AppliedCoefficient") -> float:
        if not isinstance(coefficient, AppliedCoefficient):
            raise ODEBFContractError("P1R28 raw slope requires applied coefficient")
        return float(_vector(self.values, label="raw coefficient slope") @ coefficient.array)


@dataclass(frozen=True, slots=True)
class VelocitySlope:
    """Applied-step derivative with respect to velocity ``v``: ``a_v=h*a_bar``."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _vector(self.values, label="velocity slope")

    @property
    def array(self) -> np.ndarray:
        return _vector(self.values, label="velocity slope")

    def contract(self, velocity: "VelocityCoefficient") -> float:
        if not isinstance(velocity, VelocityCoefficient):
            raise ODEBFContractError("P1R28 velocity slope requires velocity coefficient")
        return float(self.array @ velocity.array)


@dataclass(frozen=True, slots=True)
class VelocityCoefficient:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        value = _vector(self.values, label="velocity coefficient")
        if np.any(value < 0.0):
            raise ODEBFContractError("P1R28 velocity is negative")

    @property
    def array(self) -> np.ndarray:
        return _vector(self.values, label="velocity coefficient")

    def to_applied(self) -> "AppliedCoefficient":
        return AppliedCoefficient(tuple(P1R28_H * item for item in self.values))


@dataclass(frozen=True, slots=True)
class AppliedCoefficient:
    """Physical coefficient ``theta=h*v`` used exactly once by materialization."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        value = _vector(self.values, label="applied coefficient")
        if np.any(value < 0.0):
            raise ODEBFContractError("P1R28 applied coefficient is negative")

    @property
    def array(self) -> np.ndarray:
        return _vector(self.values, label="applied coefficient")


@dataclass(frozen=True, slots=True)
class P1R24MatchedDemand:
    semantic_signed: float
    lag_signed: float
    identity_sha256: str

    def rho(self, lag_scale: float) -> float:
        if not math.isfinite(lag_scale) or not 0.0 <= lag_scale <= 1.0:
            raise ODEBFContractError("P1R28 lag scale differs")
        return max(self.semantic_signed + lag_scale * self.lag_signed, 0.0)

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def build_p1r24_matched_demand(
    gradient: np.ndarray,
    semantic_displacement: np.ndarray,
    lag_displacement: np.ndarray,
) -> P1R24MatchedDemand:
    g = np.asarray(gradient, dtype=np.float64)
    semantic = np.asarray(semantic_displacement, dtype=np.float64)
    lag = np.asarray(lag_displacement, dtype=np.float64)
    if (
        g.shape != semantic.shape
        or g.shape != lag.shape
        or not np.all(np.isfinite(g))
        or not np.all(np.isfinite(semantic))
        or not np.all(np.isfinite(lag))
    ):
        raise ODEBFContractError("P1R28 matched-demand geometry differs")
    semantic_signed = float(-np.sum(g * semantic))
    lag_signed = float(-np.sum(g * lag))
    payload = {
        "schema": "ode-edit-s05-p1r28-p1r24-matched-demand/v1",
        "semantic_signed": semantic_signed,
        "lag_signed": lag_signed,
        "positive_part_applied_after_affine_combination": True,
        "h_application_count": 1,
        "second_h_division_count": 0,
    }
    return P1R24MatchedDemand(
        semantic_signed, lag_signed, canonical_hash(payload)
    )


@dataclass(frozen=True, slots=True)
class AffineVelocitySlope:
    semantic: VelocitySlope
    lag: VelocitySlope

    def at(self, lag_scale: float) -> VelocitySlope:
        if not math.isfinite(lag_scale) or not 0.0 <= lag_scale <= 1.0:
            raise ODEBFContractError("P1R28 lag scale differs")
        value = self.semantic.array + lag_scale * self.lag.array
        return VelocitySlope(tuple(float(item) for item in value))


@dataclass(frozen=True, slots=True)
class PhysicalCouplingBasis:
    raw_semantic: RawCoefficientSlope
    raw_lag: RawCoefficientSlope
    objective: ScalableObjectiveResult
    context_sha256: str
    identity_sha256: str

    @property
    def velocity(self) -> AffineVelocitySlope:
        return AffineVelocitySlope(
            self.raw_semantic.to_velocity(), self.raw_lag.to_velocity()
        )

    def signed_receipt(
        self, field: P1DynamicField, *, lag_scale: float
    ) -> SignedProgressReceipt:
        values = tuple(
            semantic + lag_scale * lag
            for semantic, lag in zip(
                self.raw_semantic.values, self.raw_lag.values, strict=True
            )
        )
        gradient = torch.tensor([-item for item in values], dtype=torch.float64)
        return SignedProgressReceipt(
            field.identity_sha256,
            values,
            tuple(
                layer
                for layer, value in zip(P1R23_LAYER_ORDER, values, strict=True)
                if value <= 0.0
            ),
            tensor_sha256(gradient),
            self.objective.model_forward_count,
            self.objective.processed_token_count,
            True,
            "TARGET_NEW_NLL",
            0,
            self.objective.identity_sha256,
            self.context_sha256,
            P1R23_CONTEXTS_PER_REQUEST,
            (1, 5),
            self.objective.backward_count,
            self.objective.loss,
        )

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r28-physical-coupling-basis/v1",
            "raw_semantic_slopes": list(self.raw_semantic.values),
            "raw_lag_slopes": list(self.raw_lag.values),
            "velocity_semantic_slopes": list(
                self.velocity.semantic.values
            ),
            "velocity_lag_slopes": list(self.velocity.lag.values),
            "shared_full_six_vjp_count": 1,
            "additional_model_forward_count": 0,
            "additional_backward_count": 0,
            "objective_sha256": self.objective.identity_sha256,
            "context_sha256": self.context_sha256,
            "identity_sha256": self.identity_sha256,
        }


def physical_coupling_basis(
    model: torch.nn.Module,
    plan: ScalableObjectivePlan,
    semantic_field: P1DynamicField,
    lag_field: P1DynamicField | None,
) -> PhysicalCouplingBasis:
    """Obtain both raw coefficient-slope bases in one shared full-six VJP."""

    if lag_field is not None and (
        semantic_field.request_order_sha256 != lag_field.request_order_sha256
        or semantic_field.accepted_waypoint != lag_field.accepted_waypoint
        or tuple(item.layer for item in semantic_field.layers)
        != tuple(item.layer for item in lag_field.layers)
    ):
        raise ODEBFContractError("P1R28 physical basis state differs")
    layers = (
        tuple(semantic_field.layers)
        if lag_field is None
        else (*semantic_field.layers, *lag_field.layers)
    )
    coefficients = torch.zeros(
        len(layers),
        device=next(model.parameters()).device,
        dtype=torch.float32,
        requires_grad=True,
    )
    observed = evaluate_scalable_target_new_objective(
        model, plan, coefficient_layers=layers, coefficients=coefficients
    )
    if observed.coefficient_gradient is None:
        raise ODEBFContractError("P1R28 physical basis gradient is absent")
    raw = -observed.coefficient_gradient.detach().to(
        device="cpu", dtype=torch.float64
    )
    width = len(P1R23_LAYER_ORDER)
    semantic = RawCoefficientSlope(tuple(float(item) for item in raw[:width]))
    lag = RawCoefficientSlope(
        tuple(0.0 for _ in range(width))
        if lag_field is None
        else tuple(float(item) for item in raw[width:])
    )
    payload = {
        "semantic_field_sha256": semantic_field.identity_sha256,
        "lag_field_sha256": None if lag_field is None else lag_field.identity_sha256,
        "raw_semantic_slopes": list(semantic.values),
        "raw_lag_slopes": list(lag.values),
        "objective_sha256": observed.identity_sha256,
        "shared_full_six_vjp_count": 1,
    }
    return PhysicalCouplingBasis(
        semantic, lag, observed, plan.context_sha256, canonical_hash(payload)
    )


def affine_trust_domain_from_fields(
    semantic_field: P1DynamicField,
    lag_field: P1DynamicField | None,
    *,
    write_trust_fraction: float,
) -> AffineTrustDomain:
    if len(semantic_field.layers) != len(P1R23_LAYER_ORDER) or (
        lag_field is not None
        and (
            semantic_field.request_order_sha256 != lag_field.request_order_sha256
            or semantic_field.accepted_waypoint != lag_field.accepted_waypoint
            or len(lag_field.layers) != len(P1R23_LAYER_ORDER)
        )
    ):
        raise ODEBFContractError("P1R28 affine trust state differs")
    constant: list[float] = []
    cross: list[float] = []
    lag: list[float] = []
    lag_layers = (
        tuple(None for _ in semantic_field.layers)
        if lag_field is None
        else lag_field.layers
    )
    for semantic_layer, lag_layer in zip(
        semantic_field.layers, lag_layers, strict=True
    ):
        if (
            lag_layer is not None
            and (
                semantic_layer.layer != lag_layer.layer
                or not torch.equal(semantic_layer.q, lag_layer.q)
                or not torch.equal(semantic_layer.key, lag_layer.key)
            )
        ):
            raise ODEBFContractError("P1R28 affine trust basis differs")
        right_gram = semantic_layer.q.T.double() @ semantic_layer.q.double()
        sem = semantic_layer.residual.double()
        lag_value = (
            torch.zeros_like(sem)
            if lag_layer is None
            else lag_layer.residual.double()
        )
        constant.append(
            P1R28_H**2
            * float(torch.sum((sem.T @ sem) * right_gram))
        )
        cross.append(
            P1R28_H**2
            * float(torch.sum((sem.T @ lag_value) * right_gram))
        )
        lag.append(
            P1R28_H**2
            * float(torch.sum((lag_value.T @ lag_value) * right_gram))
        )
    return AffineTrustDomain(
        tuple(constant), tuple(cross), tuple(lag), write_trust_fraction
    )


class P1R28CouplingStatus(str, Enum):
    JOINT_WRITE = "JOINT_WRITE"
    TRACKING_CONFLICT_RECTIFIED = "TRACKING_CONFLICT_RECTIFIED"
    SEMANTIC_NO_POSITIVE_DIRECTION = "SEMANTIC_NO_POSITIVE_DIRECTION"
    SEMANTIC_STALL = "SEMANTIC_STALL"
    TECHNICAL_FAIL = "TECHNICAL_FAIL"


@dataclass(frozen=True, slots=True)
class ReachabilityCandidate:
    lag_scale: float
    rho: float
    r_max: float
    feasible: bool
    active_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class AffineTrustDomain:
    """Diagonal P1R24 velocity-coordinate trust geometry along ``lambda``.

    ``T_l(lambda)=constant_l+2*lambda*cross_l+lambda^2*lag_l`` and
    ``radius(lambda)^2=write_trust_fraction^2*sum_l T_l(lambda)``.  This is
    the exact geometry emitted by :func:`build_p1_routing_problem` after its
    P1R24 ``eta=h`` velocity conversion, without the legacy layer caps.
    """

    constant: tuple[float, ...]
    cross: tuple[float, ...]
    lag: tuple[float, ...]
    write_trust_fraction: float = 1.0

    def __post_init__(self) -> None:
        values = (
            _vector(self.constant, label="trust constant"),
            _vector(self.cross, label="trust cross"),
            _vector(self.lag, label="trust lag"),
        )
        if (
            not math.isfinite(self.write_trust_fraction)
            or self.write_trust_fraction <= 0.0
            or np.any(values[0] <= 0.0)
            or np.any(values[2] < 0.0)
        ):
            raise ODEBFContractError("P1R28 trust geometry differs")

    def diagonal(self, lag_scale: float) -> np.ndarray:
        if not math.isfinite(lag_scale) or not 0.0 <= lag_scale <= 1.0:
            raise ODEBFContractError("P1R28 lag scale differs")
        value = (
            _vector(self.constant, label="trust constant")
            + 2.0 * lag_scale * _vector(self.cross, label="trust cross")
            + lag_scale * lag_scale * _vector(self.lag, label="trust lag")
        )
        if not np.all(np.isfinite(value)) or np.any(value <= 0.0):
            raise ODEBFContractError("P1R28 trust metric is degenerate")
        return value

    def radius_squared(self, lag_scale: float) -> float:
        return float(
            self.write_trust_fraction**2 * self.diagonal(lag_scale).sum()
        )


@dataclass(frozen=True, slots=True)
class CouplingSelection:
    status: P1R28CouplingStatus
    lag_scale: float
    rho: float
    r_max: float
    slopes: tuple[float, ...]
    candidates: tuple[ReachabilityCandidate, ...]
    backend_call_count: int
    added_model_forward_count: int
    added_backward_count: int
    added_materialization_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


def _r_max(
    slopes: np.ndarray, domain: AffineTrustDomain, lag_scale: float
) -> float:
    """Exact maximum over the cap-free nonnegative P1R24 trust ellipsoid."""

    positive = slopes > 0.0
    if not np.any(positive):
        return 0.0
    diagonal = domain.diagonal(lag_scale)
    value = domain.radius_squared(lag_scale) * float(
        np.sum(np.square(slopes[positive]) / diagonal[positive])
    )
    if not math.isfinite(value) or value < 0.0:
        raise ODEBFContractError("P1R28 reachability is nonfinite")
    return math.sqrt(value)


def _candidate(
    lag_scale: float,
    slopes: AffineVelocitySlope,
    demand: P1R24MatchedDemand,
    domain: AffineTrustDomain,
) -> ReachabilityCandidate:
    current = slopes.at(lag_scale).array
    rho = demand.rho(lag_scale)
    maximum = _r_max(current, domain, lag_scale)
    return ReachabilityCandidate(
        float(lag_scale),
        rho,
        maximum,
        bool(maximum + SIMPLEX_PRIMAL_TOLERANCE >= rho),
        tuple(bool(item > 0.0) for item in current),
    )


def largest_feasible_lag_scale(
    slopes: AffineVelocitySlope,
    demand: P1R24MatchedDemand,
    domain: AffineTrustDomain,
) -> CouplingSelection:
    """Find the exact largest feasible scalar on the piecewise-affine path.

    Breakpoints are only layer-slope zero crossings and the demand positive-part
    crossing.  On every resulting interval the active set and the demand's
    positive-part branch are fixed.  Clearing the strictly positive diagonal
    trust denominators turns ``r_max**2-rho**2`` into a polynomial whose full
    root set can be evaluated analytically and then certified in the original
    expression.  The existing simplex primal tolerance is used for
    certification; no lambda grid, monotonicity assumption, or new tolerance
    is introduced.
    """

    semantic = slopes.semantic.array
    lag = slopes.lag.array
    points = {0.0, 1.0}
    for base, delta in zip(semantic, lag, strict=True):
        if delta != 0.0:
            root = -base / delta
            if 0.0 < root < 1.0:
                points.add(float(root))
    if demand.lag_signed != 0.0:
        root = -demand.semantic_signed / demand.lag_signed
        if 0.0 < root < 1.0:
            points.add(float(root))
    ordered = sorted(points)
    probes: set[float] = set(ordered)
    # On each open interval, active signs and rho's positive part are fixed.
    # Clear the positive trust denominators and solve the resulting polynomial
    # globally.  This avoids an unjustified monotone/bisection assumption.
    from numpy.polynomial import Polynomial

    trust_polynomials = tuple(
        Polynomial((constant, 2.0 * cross, lag_value))
        for constant, cross, lag_value in zip(
            domain.constant, domain.cross, domain.lag, strict=True
        )
    )
    radius_polynomial = sum(
        trust_polynomials, Polynomial((0.0,))
    ) * domain.write_trust_fraction**2
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        mid = 0.5 * (left + right)
        active = semantic + mid * lag > 0.0
        demand_active = demand.semantic_signed + mid * demand.lag_signed > 0.0
        product = Polynomial((1.0,))
        for polynomial in trust_polynomials:
            product *= polynomial
        sum_term = Polynomial((0.0,))
        for index in np.flatnonzero(active):
            without = Polynomial((1.0,))
            for other, polynomial in enumerate(trust_polynomials):
                if other != index:
                    without *= polynomial
            slope_polynomial = Polynomial((semantic[index], lag[index]))
            sum_term += slope_polynomial * slope_polynomial * without
        rho_polynomial = (
            Polynomial((demand.semantic_signed, demand.lag_signed))
            if demand_active
            else Polynomial((0.0,))
        )
        cleared = radius_polynomial * sum_term - rho_polynomial * rho_polynomial * product
        for root in cleared.roots():
            if (
                abs(float(np.imag(root))) <= SIMPLEX_PRIMAL_TOLERANCE
                and left < float(np.real(root)) < right
            ):
                probes.add(float(np.real(root)))
    candidates = tuple(
        _candidate(point, slopes, demand, domain) for point in sorted(probes)
    )
    feasible = [item for item in candidates if item.feasible]
    semantic0 = _candidate(0.0, slopes, demand, domain)
    full1 = _candidate(1.0, slopes, demand, domain)
    if semantic0.rho > P1R24_Q_EPSILON and semantic0.r_max <= P1R24_Q_EPSILON:
        status = P1R28CouplingStatus.SEMANTIC_NO_POSITIVE_DIRECTION
        selected = semantic0
    elif not semantic0.feasible:
        status = P1R28CouplingStatus.SEMANTIC_STALL
        selected = semantic0
    elif full1.feasible:
        status = P1R28CouplingStatus.JOINT_WRITE
        selected = full1
    elif feasible:
        selected = max(feasible, key=lambda item: item.lag_scale)
        status = P1R28CouplingStatus.TRACKING_CONFLICT_RECTIFIED
    else:
        status = P1R28CouplingStatus.SEMANTIC_STALL
        selected = semantic0
    selected_slopes = slopes.at(selected.lag_scale)
    payload = {
        "schema": "ode-edit-s05-p1r28-largest-feasible-lag/v1",
        "status": status.value,
        "lag_scale": selected.lag_scale,
        "rho": selected.rho,
        "r_max": selected.r_max,
        "slopes": list(selected_slopes.values),
        "candidates": [asdict(item) for item in candidates],
        "reachability_definition": "maximum applied-step progress over P1R24 cap-free nonnegative trust ellipsoid",
        "solve": "exact-piecewise-rational-all-polynomial-roots-and-original-form-certification",
        "tolerance_source": "SIMPLEX_PRIMAL_TOLERANCE",
        "backend_call_count": 1,
        "added_model_forward_count": 0,
        "added_backward_count": 0,
        "added_materialization_count": 0,
    }
    return CouplingSelection(
        status,
        selected.lag_scale,
        selected.rho,
        selected.r_max,
        selected_slopes.values,
        candidates,
        1,
        0,
        0,
        0,
        canonical_hash(payload),
    )


def semantic_only_coupling(
    slopes: AffineVelocitySlope,
    demand: P1R24MatchedDemand,
    domain: AffineTrustDomain,
) -> CouplingSelection:
    selected = _candidate(0.0, slopes, demand, domain)
    if selected.rho > P1R24_Q_EPSILON and selected.r_max <= P1R24_Q_EPSILON:
        status = P1R28CouplingStatus.SEMANTIC_NO_POSITIVE_DIRECTION
    elif not selected.feasible:
        status = P1R28CouplingStatus.SEMANTIC_STALL
    else:
        status = P1R28CouplingStatus.JOINT_WRITE
    velocity = slopes.at(0.0)
    payload = {
        "schema": "ode-edit-s05-p1r28-semantic-only-coupling/v1",
        "status": status.value,
        "rho": selected.rho,
        "r_max": selected.r_max,
        "slopes": list(velocity.values),
    }
    return CouplingSelection(
        status,
        0.0,
        selected.rho,
        selected.r_max,
        velocity.values,
        (selected,),
        0,
        0,
        0,
        0,
        canonical_hash(payload),
    )


def coordinate_identity_receipt(
    raw_slope: RawCoefficientSlope,
    velocity: VelocityCoefficient,
) -> dict[str, Any]:
    applied = velocity.to_applied()
    velocity_slope = raw_slope.to_velocity()
    left = velocity_slope.contract(velocity)
    right = raw_slope.contract(applied)
    residual = abs(left - right)
    scale = max(abs(left), abs(right), P1R24_Q_EPSILON)
    ratio = 1.0 if abs(right) <= P1R24_Q_EPSILON else left / right
    tolerance = max(P1R24_NUMERICAL_EPSILON, SIMPLEX_PRIMAL_TOLERANCE) * scale
    if residual > tolerance or not math.isfinite(ratio):
        raise ODEBFContractError("P1R28 coordinate identity differs")
    payload = {
        "schema": "ode-edit-s05-p1r28-coordinate-identity/v1",
        "velocity_slope_dot_velocity": left,
        "raw_slope_dot_applied_coefficient": right,
        "identity_residual": residual,
        "ratio": ratio,
        "theta": list(applied.values),
        "h": P1R28_H,
        "h_application_count": 1,
        "second_h_division_count": 0,
        "forbidden_h_raw_slope_dot_applied_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def stall_state_receipt(
    *,
    w_before_sha256: str,
    w_after_sha256: str,
    z_before_sha256: str,
    z_after_sha256: str,
) -> dict[str, Any]:
    held = (
        w_before_sha256 == w_after_sha256
        and z_before_sha256 == z_after_sha256
    )
    if not held:
        raise ODEBFContractError("P1R28 final stall did not hold W and z")
    payload = {
        "schema": "ode-edit-s05-p1r28-stall-state/v1",
        "w_before_sha256": w_before_sha256,
        "w_after_sha256": w_after_sha256,
        "z_before_sha256": z_before_sha256,
        "z_after_sha256": z_after_sha256,
        "w_held": True,
        "z_held": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "AffineVelocitySlope",
    "AffineTrustDomain",
    "AppliedCoefficient",
    "CouplingSelection",
    "P1R24MatchedDemand",
    "P1R28CouplingStatus",
    "P1R28_H",
    "P1R28_INSTRUCTION_ID",
    "P1R28_K",
    "P1R28_METHOD_ID",
    "RawCoefficientSlope",
    "VelocityCoefficient",
    "VelocitySlope",
    "build_p1r24_matched_demand",
    "coordinate_identity_receipt",
    "largest_feasible_lag_scale",
    "semantic_only_coupling",
    "stall_state_receipt",
]
