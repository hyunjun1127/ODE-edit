"""Common P1 routing geometry and projector-only Generic/BF split."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from .contracts import BACKTRACKING_FACTORS, FIXED_K, ODEBFContractError, canonical_hash
from .functional import WaypointFactor
from .p1_backend import P1DynamicField, P1LayerField, SignedProgressReceipt
from .routing import (
    BFProjectionResult,
    CertificateObserver,
    QuadraticBarrier,
    RawVelocity,
    RoutingProblem,
    RoutingStatus,
    project_bf_velocity,
    solve_raw_velocity,
)


@dataclass(frozen=True, slots=True)
class P1ControllerLock:
    k_resolution: int = FIXED_K
    nominal_t: float = 1.0
    progress_fraction: float = 0.25
    minimum_progress: float = 1.0e-8
    rho_accept: float = 0.1
    structural_h_increment_fraction: float = 0.50
    structural_p_increment_fraction: float = 0.50
    write_trust_fraction: float = 1.0
    layer_velocity_cap: float = 1.0
    capacity_epsilon: float = 1.0e-12
    functional_h_budget_nats: float = 1.0e-3
    functional_p_budget_nats: float = 1.0e-3
    smoothmax_temperature: float = 1.0e-2
    target_bootstrap_fraction: float = 0.125
    target_trust_fraction: float = 0.125
    residual_tolerance: float = 1.0e-5
    backtracking: tuple[float, ...] = BACKTRACKING_FACTORS

    def __post_init__(self) -> None:
        if self.k_resolution != FIXED_K or self.nominal_t != 1.0:
            raise ODEBFContractError("P1 controller does not use one common K8 cycle")
        if self.backtracking != BACKTRACKING_FACTORS:
            raise ODEBFContractError("P1 backtracking schedule differs")
        for name in (
            "progress_fraction",
            "minimum_progress",
            "rho_accept",
            "structural_h_increment_fraction",
            "structural_p_increment_fraction",
            "write_trust_fraction",
            "layer_velocity_cap",
            "capacity_epsilon",
            "functional_h_budget_nats",
            "functional_p_budget_nats",
            "smoothmax_temperature",
            "target_bootstrap_fraction",
            "target_trust_fraction",
            "residual_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ODEBFContractError(f"P1 controller {name} differs")
        if self.progress_fraction > 1.0 or self.target_bootstrap_fraction > 1.0:
            raise ODEBFContractError("P1 progress/bootstrap fraction exceeds one")
        if self.rho_accept > 1.0:
            raise ODEBFContractError("P1 trust-ratio acceptance threshold exceeds one")

    @property
    def eta(self) -> float:
        return self.nominal_t / self.k_resolution

    def identity(self) -> str:
        return canonical_hash(
            {
                "k_resolution": self.k_resolution,
                "nominal_t": self.nominal_t,
                "eta": self.eta,
                "progress_fraction": self.progress_fraction,
                "minimum_progress": self.minimum_progress,
                "rho_accept": self.rho_accept,
                "progress_acceptance": (
                    "actual>=minimum_progress-and-actual/"
                    "max(beta*max(aTy,0),minimum_progress)>=rho_accept"
                ),
                "structural_h_increment_fraction": self.structural_h_increment_fraction,
                "structural_p_increment_fraction": self.structural_p_increment_fraction,
                "write_trust_fraction": self.write_trust_fraction,
                "layer_velocity_cap": self.layer_velocity_cap,
                "capacity_epsilon": self.capacity_epsilon,
                "functional_h_budget_nats": self.functional_h_budget_nats,
                "functional_p_budget_nats": self.functional_p_budget_nats,
                "smoothmax_temperature": self.smoothmax_temperature,
                "target_bootstrap_fraction": self.target_bootstrap_fraction,
                "target_trust_fraction": self.target_trust_fraction,
                "residual_tolerance": self.residual_tolerance,
                "backtracking": list(self.backtracking),
                "edit_batch_size": 10,
                "correction_cycles": 1,
            }
        )


@dataclass(slots=True)
class AcceptedLayerContribution:
    layer: int
    factor: WaypointFactor
    q: torch.Tensor
    residual: torch.Tensor
    covariance_action: torch.Tensor
    history_action: torch.Tensor
    factor_frobenius_sq: float

    @classmethod
    def from_field(
        cls,
        field: P1LayerField,
        factor: WaypointFactor,
        *,
        history_action: torch.Tensor | None = None,
    ) -> "AcceptedLayerContribution":
        observed_history = field.history_action if history_action is None else history_action
        if observed_history.ndim != 2 or observed_history.shape[0] != field.residual.shape[0]:
            raise ODEBFContractError("P1 accepted history action geometry differs")
        return cls(
            field.layer,
            factor,
            field.q.detach().to(device="cpu", dtype=torch.float32).clone(),
            field.residual.detach().to(device="cpu", dtype=torch.float32).clone(),
            field.covariance_action.detach().to(device="cpu", dtype=torch.float32).clone(),
            observed_history.detach().to(device="cpu", dtype=torch.float64).clone(),
            field.factor_frobenius_sq,
        )


@dataclass(frozen=True, slots=True)
class P1RoutingGeometry:
    layers: tuple[int, ...]
    signed_progress: np.ndarray
    capacity_metric: np.ndarray
    trust_metric: np.ndarray
    trust_radius: float
    layer_caps: np.ndarray
    requested_progress: float
    minimum_progress: float
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class P1RoutingBuild:
    geometry: P1RoutingGeometry
    problem: RoutingProblem
    historical_self_risk: tuple[float, ...]
    pretrained_self_risk: tuple[float, ...]
    temporary_load: tuple[float, ...]
    field_sha256: str
    signed_progress_sha256: str


@dataclass(frozen=True, slots=True)
class MatchedRawVelocity:
    values: np.ndarray
    geometry_sha256: str
    velocity_sha256: str
    source_problem_sha256: str
    maximum_feasible_progress: float
    certificate: object


@dataclass(frozen=True, slots=True)
class SharedBFProjection:
    problem: RoutingProblem
    result: BFProjectionResult
    raw_velocity_sha256: str
    source_geometry_sha256: str
    barrier_state_sha256: str


def _pretrained_inner(
    left: torch.Tensor,
    right: torch.Tensor,
    other_left: torch.Tensor,
    other_covariance_action: torch.Tensor,
) -> float:
    left_cross = left.T.to(dtype=torch.float64) @ other_left.to(dtype=torch.float64)
    q_cross = right.T.to(dtype=torch.float64) @ other_covariance_action.to(dtype=torch.float64)
    return float(torch.sum(left_cross * q_cross))


def _cumulative_structural_terms(
    field: P1DynamicField,
    accepted_by_layer: Mapping[int, Sequence[AcceptedLayerContribution]],
    current_history_action_by_layer: Mapping[int, torch.Tensor] | None = None,
) -> tuple[
    tuple[float, np.ndarray, np.ndarray],
    tuple[float, np.ndarray, np.ndarray],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    h_offset = 0.0
    p_offset = 0.0
    h_linear: list[float] = []
    p_linear: list[float] = []
    h_self: list[float] = []
    p_self: list[float] = []
    temporary_load: list[float] = []
    if current_history_action_by_layer is not None and set(current_history_action_by_layer) != {
        item.layer for item in field.layers
    }:
        raise ODEBFContractError("P1 arm-local history action layer set differs")
    for current in field.layers:
        prior = tuple(accepted_by_layer.get(current.layer, ()))
        h_current = torch.zeros_like(current.history_action)
        for item in prior:
            h_current = h_current + item.factor.theta * item.history_action
        h_offset += float(torch.sum(h_current * h_current))
        current_history = (
            current.history_action
            if current_history_action_by_layer is None
            else current_history_action_by_layer[current.layer]
        ).to(dtype=torch.float64)
        if current_history.shape != current.history_action.shape or not torch.isfinite(
            current_history
        ).all():
            raise ODEBFContractError("P1 arm-local history action geometry/value differs")
        h_linear.append(float(torch.sum(h_current * current_history)))
        h_self.append(float(torch.sum(current_history * current_history)))

        layer_p_offset = 0.0
        for left in prior:
            for right in prior:
                layer_p_offset += (
                    left.factor.theta
                    * right.factor.theta
                    * _pretrained_inner(
                        left.residual,
                        left.q,
                        right.residual,
                        right.covariance_action,
                    )
                )
        p_offset += layer_p_offset
        p_linear.append(
            sum(
                item.factor.theta
                * _pretrained_inner(
                    item.residual,
                    item.q,
                    current.residual,
                    current.covariance_action,
                )
                for item in prior
            )
        )
        p_self.append(
            _pretrained_inner(
                current.residual,
                current.q,
                current.residual,
                current.covariance_action,
            )
        )
        temporary_load.append(
            sum(item.factor.theta**2 * item.factor_frobenius_sq for item in prior)
        )
    if min((*h_self, *p_self), default=0.0) < -1.0e-8:
        raise ODEBFContractError("P1 structural self-risk is negative")
    return (
        (h_offset, np.asarray(h_linear), np.diag(np.maximum(h_self, 0.0))),
        (p_offset, np.asarray(p_linear), np.diag(np.maximum(p_self, 0.0))),
        tuple(max(value, 0.0) for value in h_self),
        tuple(max(value, 0.0) for value in p_self),
        tuple(temporary_load),
    )


def build_p1_routing_problem(
    field: P1DynamicField,
    signed_progress: SignedProgressReceipt,
    *,
    accepted_by_layer: Mapping[int, Sequence[AcceptedLayerContribution]],
    committed_load_by_layer: Mapping[int, float],
    lock: P1ControllerLock,
    current_history_action_by_layer: Mapping[int, torch.Tensor] | None = None,
) -> P1RoutingBuild:
    layers = tuple(item.layer for item in field.layers)
    if signed_progress.field_sha256 != field.identity_sha256:
        raise ODEBFContractError("P1 signed progress belongs to another field")
    if len(signed_progress.signed_progress) != len(layers):
        raise ODEBFContractError("P1 signed-progress dimension differs")
    if set(committed_load_by_layer) != set(layers):
        raise ODEBFContractError("P1 committed load layer set differs")
    h_terms, p_terms, h_self, p_self, temporary = _cumulative_structural_terms(
        field,
        accepted_by_layer,
        current_history_action_by_layer=current_history_action_by_layer,
    )
    eta = lock.eta
    raw_progress = np.asarray(signed_progress.signed_progress, dtype=np.float64)
    progress = eta * raw_progress
    factor_energy = np.asarray(
        [item.factor_frobenius_sq for item in field.layers], dtype=np.float64
    )
    scaled_energy = eta**2 * factor_energy
    capacity_diagonal = np.asarray(
        [
            (1.0 + float(committed_load_by_layer[layer]) + temporary[index])
            * (scaled_energy[index] + lock.capacity_epsilon)
            for index, layer in enumerate(layers)
        ],
        dtype=np.float64,
    )
    maximum_loose_progress = float(np.maximum(progress, 0.0).sum())
    requested = lock.progress_fraction * maximum_loose_progress
    if not math.isfinite(requested) or requested < lock.minimum_progress:
        raise ODEBFContractError("PROGRESS_INFEASIBLE: locked minimum progress is unavailable")
    trust_radius = lock.write_trust_fraction * math.sqrt(float(scaled_energy.sum()))
    h_offset, h_linear, h_gram = h_terms
    p_offset, p_linear, p_gram = p_terms
    h_linear = eta * h_linear
    p_linear = eta * p_linear
    h_gram = eta**2 * h_gram
    p_gram = eta**2 * p_gram
    h_budget = h_offset + lock.structural_h_increment_fraction * float(
        np.asarray(h_self).sum() * eta**2
    )
    p_budget = p_offset + lock.structural_p_increment_fraction * float(
        np.asarray(p_self).sum() * eta**2
    )
    geometry_payload = {
        "layers": layers,
        "signed_progress": progress.tolist(),
        "capacity_diagonal": capacity_diagonal.tolist(),
        "trust_diagonal": scaled_energy.tolist(),
        "trust_radius": trust_radius,
        "layer_caps": [lock.layer_velocity_cap] * len(layers),
        "requested_progress": requested,
        "minimum_progress": lock.minimum_progress,
        "controller_lock": lock.identity(),
        "field": field.identity_sha256,
    }
    geometry = P1RoutingGeometry(
        layers,
        progress,
        np.diag(capacity_diagonal),
        np.diag(scaled_energy),
        trust_radius,
        np.full(len(layers), lock.layer_velocity_cap, dtype=np.float64),
        requested,
        lock.minimum_progress,
        canonical_hash(geometry_payload),
    )
    problem = RoutingProblem(
        geometry.signed_progress,
        geometry.capacity_metric,
        geometry.trust_metric,
        geometry.trust_radius,
        geometry.layer_caps,
        geometry.requested_progress,
        geometry.minimum_progress,
        QuadraticBarrier(
            "historical",
            h_offset,
            h_linear,
            h_gram,
            h_budget,
            "layer-local-diagonal",
        ),
        QuadraticBarrier(
            "pretrained",
            p_offset,
            p_linear,
            p_gram,
            p_budget,
            "layer-local-diagonal",
        ),
    )
    return P1RoutingBuild(
        geometry,
        problem,
        h_self,
        p_self,
        temporary,
        field.identity_sha256,
        canonical_hash(list(signed_progress.signed_progress)),
    )


def solve_matched_raw_velocity(
    build: P1RoutingBuild,
    *,
    certificate_observer: CertificateObserver | None = None,
) -> MatchedRawVelocity:
    solved = solve_raw_velocity(
        build.problem,
        certificate_observer=certificate_observer,
    )
    velocity_sha = canonical_hash(
        {
            "geometry": build.geometry.identity_sha256,
            "values": solved.values.tolist(),
        }
    )
    return MatchedRawVelocity(
        solved.values.copy(),
        build.geometry.identity_sha256,
        velocity_sha,
        solved.problem_identity,
        solved.maximum_feasible_progress,
        solved.certificate,
    )


def project_matched_bf_velocity(
    build: P1RoutingBuild,
    raw: MatchedRawVelocity,
    *,
    certificate_observer: CertificateObserver | None = None,
) -> BFProjectionResult:
    if raw.geometry_sha256 != build.geometry.identity_sha256:
        raise ODEBFContractError("P1 BF projector received a different raw geometry")
    rebound = RawVelocity(
        raw.values.copy(),
        build.problem.identity(),
        raw.velocity_sha256,
        raw.maximum_feasible_progress,
        raw.certificate,  # type: ignore[arg-type]
    )
    result = project_bf_velocity(
        build.problem,
        rebound,
        certificate_observer=certificate_observer,
    )
    if result.raw_velocity_identity != raw.velocity_sha256:
        raise ODEBFContractError("P1 BF projector changed raw velocity identity")
    return result


def project_shared_raw_velocity(
    source_build: P1RoutingBuild,
    barrier_build: P1RoutingBuild,
    raw: MatchedRawVelocity,
) -> SharedBFProjection:
    """Project one shared raw velocity through an arm-local H/P barrier state.

    Capacity, progress, trust, caps, field, and requested progress are taken
    byte-for-byte from the shared Generic source.  Only the H/P barriers are
    rebound from the BF arm state.  This makes the projector the sole
    Generic-vs-BF source-level difference even after their histories diverge.
    """

    if raw.geometry_sha256 != source_build.geometry.identity_sha256:
        raise ODEBFContractError("shared BF raw velocity belongs to another geometry")
    if (
        source_build.geometry.layers != barrier_build.geometry.layers
        or source_build.field_sha256 != barrier_build.field_sha256
        or source_build.signed_progress_sha256 != barrier_build.signed_progress_sha256
    ):
        raise ODEBFContractError("shared BF barrier state uses another field/progress")
    problem = rebind_shared_problem(source_build, barrier_build)
    source = source_build.geometry
    rebound = RawVelocity(
        raw.values.copy(),
        problem.identity(),
        raw.velocity_sha256,
        raw.maximum_feasible_progress,
        raw.certificate,  # type: ignore[arg-type]
    )
    result = project_bf_velocity(problem, rebound)
    if result.raw_velocity_identity != raw.velocity_sha256:
        raise ODEBFContractError("shared BF projection changed raw velocity identity")
    barrier_identity = canonical_hash(
        {
            "historical": {
                "offset": problem.historical.offset,
                "linear": problem.historical.linear.tolist(),
                "gram": problem.historical.gram.tolist(),
                "budget": problem.historical.budget,
            },
            "pretrained": {
                "offset": problem.pretrained.offset,
                "linear": problem.pretrained.linear.tolist(),
                "gram": problem.pretrained.gram.tolist(),
                "budget": problem.pretrained.budget,
            },
        }
    )
    return SharedBFProjection(
        problem,
        result,
        raw.velocity_sha256,
        source_build.geometry.identity_sha256,
        barrier_identity,
    )


def rebind_shared_problem(
    source_build: P1RoutingBuild,
    barrier_build: P1RoutingBuild,
) -> RoutingProblem:
    """Keep raw geometry fixed while substituting only arm-local barriers."""

    if (
        source_build.geometry.layers != barrier_build.geometry.layers
        or source_build.field_sha256 != barrier_build.field_sha256
        or source_build.signed_progress_sha256 != barrier_build.signed_progress_sha256
    ):
        raise ODEBFContractError("shared routing barrier rebind identity differs")
    source = source_build.geometry
    return RoutingProblem(
        source.signed_progress.copy(),
        source.capacity_metric.copy(),
        source.trust_metric.copy(),
        source.trust_radius,
        source.layer_caps.copy(),
        source.requested_progress,
        source.minimum_progress,
        barrier_build.problem.historical,
        barrier_build.problem.pretrained,
    )


def scaled_waypoint_factors(
    field: P1DynamicField,
    velocity: Sequence[float],
    *,
    stage: int,
    beta: float,
    lock: P1ControllerLock,
) -> dict[str, WaypointFactor]:
    values = tuple(float(item) for item in velocity)
    if len(values) != len(field.layers) or stage < 1 or stage > FIXED_K:
        raise ODEBFContractError("P1 finite waypoint factor identity differs")
    if beta not in lock.backtracking:
        raise ODEBFContractError("P1 finite waypoint beta differs")
    result: dict[str, WaypointFactor] = {}
    for ordinal, (layer_field, value) in enumerate(zip(field.layers, values)):
        theta = lock.eta * beta * value
        if theta < 0.0:
            raise ODEBFContractError("P1 finite waypoint coefficient is negative")
        result[layer_field.weight_name] = WaypointFactor(
            layer_field.weight_name,
            layer_field.layer,
            0,
            stage - 1,
            ordinal,
            theta,
            layer_field.residual.clone(),
            layer_field.q.clone(),
        )
    return result


def assert_projector_only_difference(
    generic: Mapping[str, object],
    bf: Mapping[str, object],
) -> None:
    generic_value = dict(generic)
    bf_value = dict(bf)
    if generic_value.pop("barrier_projector", None) != "identity":
        raise ODEBFContractError("F_G projector identity differs")
    if bf_value.pop("barrier_projector", None) != "cbf":
        raise ODEBFContractError("F_BF projector identity differs")
    if canonical_hash(generic_value) != canonical_hash(bf_value):
        raise ODEBFContractError("F_G/F_BF differ beyond the barrier projector")
