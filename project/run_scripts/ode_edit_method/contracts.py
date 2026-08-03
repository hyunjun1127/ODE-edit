"""Fail-closed contracts shared by all five Session 02 arms."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "ode-edit-method/v1"


class MethodContractError(ValueError):
    """An implementation input violates the locked method identity."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise MethodContractError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodContractError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise MethodContractError(f"{name} must be finite")
    return result


class Arm(str, Enum):
    NATIVE_MEMIT = "native-memit"
    SCALAR_FIRST_HIT = "scalar-first-hit"
    STATIC_SYNCHRONOUS = "static-synchronous"
    ONE_REFRESH = "one-refresh"
    ORDERED_ADAPTIVE = "ordered-adaptive"
    FULL_ODE_EDIT = "full-ode-edit"


ARM_ORDER = (
    Arm.NATIVE_MEMIT,
    Arm.SCALAR_FIRST_HIT,
    Arm.STATIC_SYNCHRONOUS,
    Arm.ONE_REFRESH,
    Arm.ORDERED_ADAPTIVE,
    Arm.FULL_ODE_EDIT,
)


class ProposalSemantics(str, Enum):
    NATIVE_ORDERED_TERMINAL = "native-ordered-terminal"
    CURRENT_SAME_SNAPSHOT = "current-same-snapshot"
    ENTRY_FROZEN_REBOUND = "entry-frozen-rebound"
    CURRENT_COORDINATE = "current-coordinate"


@dataclass(frozen=True, slots=True)
class LayerProposal:
    layer: int
    state_id: str
    direction_id: str
    payload: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, int):
            raise MethodContractError("proposal layer must be an integer")
        if not self.state_id or not self.direction_id:
            raise MethodContractError("proposal state/direction identity is empty")


@dataclass(frozen=True, slots=True)
class ProposalBatch:
    """One controller-visible proposal set with explicit state semantics."""

    snapshot_id: str
    proposals: tuple[LayerProposal, ...]
    slopes: tuple[float, ...]
    semantics: ProposalSemantics

    def __post_init__(self) -> None:
        if not self.snapshot_id or not self.proposals:
            raise MethodContractError("proposal batch identity is incomplete")
        if len(self.proposals) != len(self.slopes):
            raise MethodContractError("proposal and slope counts differ")
        if not isinstance(self.semantics, ProposalSemantics):
            try:
                object.__setattr__(self, "semantics", ProposalSemantics(self.semantics))
            except ValueError as exc:
                raise MethodContractError("unknown proposal semantics") from exc
        layers = tuple(proposal.layer for proposal in self.proposals)
        if len(layers) != len(set(layers)) or layers != tuple(sorted(layers)):
            raise MethodContractError("proposal layers must be unique and ascending")
        if any(proposal.state_id != self.snapshot_id for proposal in self.proposals):
            raise MethodContractError("proposal batch mixes model snapshots")
        normalized = tuple(finite("proposal slope", value) for value in self.slopes)
        if any(value < 0.0 for value in normalized):
            raise MethodContractError("proposal slopes must be clipped non-negative")
        object.__setattr__(self, "slopes", normalized)
        if (
            self.semantics is ProposalSemantics.CURRENT_COORDINATE
            and len(self.proposals) != 1
        ):
            raise MethodContractError("ordered coordinate batches contain one layer")

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(proposal.layer for proposal in self.proposals)

    @property
    def direction_ids(self) -> tuple[str, ...]:
        return tuple(proposal.direction_id for proposal in self.proposals)

    def rebind_frozen(self, snapshot_id: str) -> "ProposalBatch":
        if self.semantics not in {
            ProposalSemantics.CURRENT_SAME_SNAPSHOT,
            ProposalSemantics.ENTRY_FROZEN_REBOUND,
        }:
            raise MethodContractError("only synchronous entry directions may be frozen")
        if not snapshot_id:
            raise MethodContractError("rebound snapshot identity is empty")
        return ProposalBatch(
            snapshot_id=snapshot_id,
            proposals=tuple(
                LayerProposal(
                    layer=proposal.layer,
                    state_id=snapshot_id,
                    direction_id=proposal.direction_id,
                    payload=proposal.payload,
                )
                for proposal in self.proposals
            ),
            slopes=self.slopes,
            semantics=ProposalSemantics.ENTRY_FROZEN_REBOUND,
        )


@dataclass(frozen=True, slots=True)
class EventReading:
    """Controller-side event only; no paraphrase/locality outcome is representable."""

    hard_phi: float
    smooth_phi: float
    context_margins: tuple[float, ...]
    nfe: int = 0

    def __post_init__(self) -> None:
        hard = finite("hard event", self.hard_phi)
        smooth = finite("smooth event", self.smooth_phi)
        margins = tuple(finite("context margin", value) for value in self.context_margins)
        if not margins:
            raise MethodContractError("event requires at least one allowed context")
        if isinstance(self.nfe, bool) or not isinstance(self.nfe, int) or self.nfe < 0:
            raise MethodContractError("event NFE must be a non-negative integer")
        object.__setattr__(self, "hard_phi", hard)
        object.__setattr__(self, "smooth_phi", smooth)
        object.__setattr__(self, "context_margins", margins)

    def is_hit(self, tolerance: float) -> bool:
        return self.hard_phi <= finite("event tolerance", tolerance)


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """One model-independent controller policy shared by both model aliases."""

    tau: float
    h0_fraction: float
    h_max_fraction: float
    kappa: float
    beta: float
    eta_reject: float
    eta_expand: float
    gamma_down: float
    gamma_up: float
    s_max: int
    max_rejections_per_state: int
    slope_epsilon: float
    progress_epsilon: float
    qp_equality_tolerance: float
    qp_trust_tolerance: float
    qp_bisection_iterations: int
    event_tolerance: float
    hard_worsening_tolerance: float
    trust_denominator_epsilon: float
    load_denominator_epsilon: float
    scalar_grid: tuple[float, ...]
    scalar_bisection_tolerance: float
    scalar_bisection_iterations: int
    scalar_nonmonotonic_tolerance: float
    functional_commit_atol: float
    functional_commit_rtol: float

    def __post_init__(self) -> None:
        positive = (
            "tau",
            "h0_fraction",
            "h_max_fraction",
            "kappa",
            "beta",
            "gamma_down",
            "gamma_up",
            "slope_epsilon",
            "progress_epsilon",
            "qp_equality_tolerance",
            "qp_trust_tolerance",
            "event_tolerance",
            "hard_worsening_tolerance",
            "trust_denominator_epsilon",
            "load_denominator_epsilon",
            "scalar_bisection_tolerance",
            "functional_commit_atol",
            "functional_commit_rtol",
        )
        for name in positive:
            value = finite(name, getattr(self, name))
            if value <= 0.0:
                raise MethodContractError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("eta_reject", "eta_expand", "scalar_nonmonotonic_tolerance"):
            value = finite(name, getattr(self, name))
            if value < 0.0:
                raise MethodContractError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if not 0.0 < self.beta < 1.0:
            raise MethodContractError("beta must leave strict trust-region interior")
        if not 0.0 < self.gamma_down < 1.0 or self.gamma_up <= 1.0:
            raise MethodContractError("trust contraction/expansion factors are invalid")
        if self.h0_fraction > self.h_max_fraction:
            raise MethodContractError("initial trust fraction exceeds its common cap")
        if self.eta_expand < self.eta_reject:
            raise MethodContractError("eta_expand precedes eta_reject")
        for name in (
            "s_max",
            "max_rejections_per_state",
            "qp_bisection_iterations",
            "scalar_bisection_iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise MethodContractError(f"{name} must be a positive integer")
        grid = tuple(finite("scalar alpha", value) for value in self.scalar_grid)
        if (
            len(grid) < 2
            or grid[0] != 0.0
            or grid[-1] != 1.0
            or any(left >= right for left, right in zip(grid, grid[1:]))
        ):
            raise MethodContractError("scalar grid must strictly increase from 0 to 1")
        object.__setattr__(self, "scalar_grid", grid)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ControllerConfig":
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise MethodContractError(
                f"controller config keys differ: missing={sorted(expected-set(value))}, "
                f"extra={sorted(set(value)-expected)}"
            )
        normalized = dict(value)
        normalized["scalar_grid"] = tuple(normalized["scalar_grid"])
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in (
                (name, getattr(self, name)) for name in self.__dataclass_fields__
            )
        }


@dataclass(frozen=True, slots=True)
class QPSolution:
    layers: tuple[int, ...]
    coefficients: tuple[float, ...]
    requested_progress: float
    predicted_progress: float
    maximum_progress: float
    coefficient_norm: float
    equality_residual: float
    objective: float
    trust_dual: float

    def __post_init__(self) -> None:
        if not self.layers or len(self.layers) != len(self.coefficients):
            raise MethodContractError("QP solution dimensions differ")
        if len(set(self.layers)) != len(self.layers):
            raise MethodContractError("QP solution layers repeat")
        coefficients = tuple(finite("QP coefficient", value) for value in self.coefficients)
        if any(value < 0.0 for value in coefficients):
            raise MethodContractError("QP coefficients must be non-negative")
        object.__setattr__(self, "coefficients", coefficients)
        for name in (
            "requested_progress",
            "predicted_progress",
            "maximum_progress",
            "coefficient_norm",
            "equality_residual",
            "objective",
            "trust_dual",
        ):
            value = finite(name, getattr(self, name))
            if value < 0.0:
                raise MethodContractError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": list(self.layers),
            "coefficients": list(self.coefficients),
            "requested_progress": self.requested_progress,
            "predicted_progress": self.predicted_progress,
            "maximum_progress": self.maximum_progress,
            "coefficient_norm": self.coefficient_norm,
            "equality_residual": self.equality_residual,
            "objective": self.objective,
            "trust_dual": self.trust_dual,
        }


@dataclass(frozen=True, slots=True)
class TrustVerdict:
    accepted: bool
    hit: bool
    reason: str
    actual_progress: float
    predicted_progress: float
    ratio: float
    next_radius: float


@dataclass(frozen=True, slots=True)
class StepRecord:
    arm: Arm
    position: int
    layer: int | None
    snapshot_id: str
    direction_ids: tuple[str, ...]
    solver_coefficients: tuple[float, ...]
    applied_coefficients: tuple[float, ...]
    accepted: bool
    reason: str
    hard_phi_before: float
    hard_phi_after: float
    smooth_phi_before: float
    smooth_phi_after: float
    trust_ratio: float

    def __post_init__(self) -> None:
        if self.solver_coefficients != self.applied_coefficients:
            raise MethodContractError("applied coefficients differ from controller output")

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "position": self.position,
            "layer": self.layer,
            "snapshot_id": self.snapshot_id,
            "direction_ids": list(self.direction_ids),
            "solver_coefficients": list(self.solver_coefficients),
            "applied_coefficients": list(self.applied_coefficients),
            "accepted": self.accepted,
            "reason": self.reason,
            "hard_phi_before": self.hard_phi_before,
            "hard_phi_after": self.hard_phi_after,
            "smooth_phi_before": self.smooth_phi_before,
            "smooth_phi_after": self.smooth_phi_after,
            "trust_ratio": self.trust_ratio,
        }


@dataclass(frozen=True, slots=True)
class ArmRunResult:
    arm: Arm
    edit_id: str
    status: str
    direct_z_compute_count: int
    terminal_state_id: str
    omega_appended: bool
    steps: tuple[StepRecord, ...]
    failure_type: str | None = None
    scalar_alpha: float | None = None
    scalar_nonmonotonic: bool | None = None

    def __post_init__(self) -> None:
        entry_hit = (
            self.status == "event_hit"
            and not self.steps
            and self.direct_z_compute_count == 0
        )
        if self.direct_z_compute_count != 1 and not entry_hit:
            raise MethodContractError(
                "direct-z must be computed once, except for a rewrite event hit at entry"
            )
        if not self.edit_id or not self.status or not self.terminal_state_id:
            raise MethodContractError("arm result identity is incomplete")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "arm": self.arm.value,
            "edit_id": self.edit_id,
            "status": self.status,
            "direct_z_compute_count": self.direct_z_compute_count,
            "terminal_state_id": self.terminal_state_id,
            "omega_appended": self.omega_appended,
            "failure_type": self.failure_type,
            "scalar_alpha": self.scalar_alpha,
            "scalar_nonmonotonic": self.scalar_nonmonotonic,
            "steps": [step.to_dict() for step in self.steps],
        }


def assert_common_policy_by_model(config_by_model: Mapping[str, Mapping[str, Any]]) -> None:
    if set(config_by_model) != {"llama3-8b-inst", "qwen2.5-7b-inst"}:
        raise MethodContractError("the lock must contain exactly the two canonical models")
    hashes = {canonical_hash(value) for value in config_by_model.values()}
    if len(hashes) != 1:
        raise MethodContractError("model-specific controller policy is forbidden")


def coefficient_mapping(
    layers: Sequence[int], coefficients: Sequence[float]
) -> dict[int, float]:
    if len(layers) != len(coefficients):
        raise MethodContractError("coefficient mapping dimensions differ")
    return {
        int(layer): finite("applied coefficient", coefficient)
        for layer, coefficient in zip(layers, coefficients, strict=True)
    }
