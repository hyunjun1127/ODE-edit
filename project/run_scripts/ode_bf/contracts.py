"""Fail-closed common contracts for the ODE-BF technical track.

This module is intentionally model-free.  The scientific unit is one joint
editor invocation containing exactly ten distinct requests.  A singleton is
accepted only when a caller explicitly marks a low-level algebra fixture; it
can never satisfy an integration or promotion gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "ode-edit/ode-bf/v1p1"
BATCH_SIZE = 10
FIXED_K = 8
COMMON_SEED = 41
BACKTRACKING_FACTORS = (1.0, 0.5, 0.25)
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")


class ODEBFContractError(ValueError):
    """An input violates a locked ODE-BF contract."""


class ODEBFStateError(RuntimeError):
    """A state transition violates transaction or rollout ownership."""


class Arm(str, Enum):
    NATIVE_ALPHA = "native-alphaedit"
    NATIVE_ALPHA_WB = "native-alphaedit-wb"
    FIXED_EQUAL_REPLAY = "fixed-z-equal-replay"
    FIXED_GENERIC = "fixed-z-generic-dynamic"
    FIXED_BF = "fixed-z-bf-projected"
    ODE_Z_STATIC = "ode-z-static-routing"
    FULL_ODE_BF = "full-ode-bf"


P0_ARMS = (Arm.NATIVE_ALPHA, Arm.NATIVE_ALPHA_WB)
LATER_ATTRIBUTION_ARMS = tuple(Arm)


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
        raise ODEBFContractError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ODEBFContractError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ODEBFContractError(f"{name} must be finite")
    return result


def nonnegative(name: str, value: Any) -> float:
    result = finite(name, value)
    if result < 0.0:
        raise ODEBFContractError(f"{name} must be nonnegative")
    return result


def positive(name: str, value: Any) -> float:
    result = finite(name, value)
    if result <= 0.0:
        raise ODEBFContractError(f"{name} must be positive")
    return result


def positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ODEBFContractError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class RolloutBudget:
    """Resolution, horizon, and correction cycles are independent axes."""

    k_resolution: int = FIXED_K
    correction_cycles: int = 1
    nominal_t_per_cycle: float = 1.0
    backtracking: tuple[float, ...] = BACKTRACKING_FACTORS
    edit_batch_size: int = BATCH_SIZE

    def __post_init__(self) -> None:
        positive_integer("k_resolution", self.k_resolution)
        positive_integer("correction_cycles", self.correction_cycles)
        object.__setattr__(
            self,
            "nominal_t_per_cycle",
            positive("nominal_t_per_cycle", self.nominal_t_per_cycle),
        )
        if self.edit_batch_size != BATCH_SIZE:
            raise ODEBFContractError("integration edit_batch_size must equal ten")
        values = tuple(positive("backtracking factor", item) for item in self.backtracking)
        if values != BACKTRACKING_FACTORS:
            raise ODEBFContractError("backtracking schedule differs from the lock")
        object.__setattr__(self, "backtracking", values)

    @property
    def eta(self) -> float:
        return self.nominal_t_per_cycle / self.k_resolution

    @property
    def k_total(self) -> int:
        return self.k_resolution * self.correction_cycles

    @property
    def maximum_trials(self) -> int:
        return self.k_total * len(self.backtracking)

    def assert_current_p0(self) -> None:
        if (
            self.k_resolution != FIXED_K
            or self.correction_cycles != 1
            or self.nominal_t_per_cycle != 1.0
        ):
            raise ODEBFContractError(
                "technical P0 requires one nominal-T cycle at K_resolution=8"
            )

    def identity(self) -> str:
        return canonical_hash(
            {
                "k_resolution": self.k_resolution,
                "correction_cycles": self.correction_cycles,
                "k_total": self.k_total,
                "nominal_t_per_cycle": self.nominal_t_per_cycle,
                "eta": self.eta,
                "backtracking": self.backtracking,
                "edit_batch_size": self.edit_batch_size,
                "per_cycle_grid": [
                    index * self.eta for index in range(self.k_resolution + 1)
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class AcceptedStepClock:
    correction_cycle: int
    cycle_count: int
    step_in_cycle: int
    k_resolution: int
    nominal_t_per_cycle: float
    accepted_beta: float
    t_acc_before: float

    def __post_init__(self) -> None:
        positive_integer("cycle_count", self.cycle_count)
        positive_integer("k_resolution", self.k_resolution)
        if (
            isinstance(self.correction_cycle, bool)
            or not isinstance(self.correction_cycle, int)
            or self.correction_cycle < 0
            or self.correction_cycle >= self.cycle_count
        ):
            raise ODEBFContractError("correction cycle index is invalid")
        if (
            isinstance(self.step_in_cycle, bool)
            or not isinstance(self.step_in_cycle, int)
            or self.step_in_cycle < 0
            or self.step_in_cycle >= self.k_resolution
        ):
            raise ODEBFContractError("step index is outside its correction cycle")
        object.__setattr__(
            self,
            "nominal_t_per_cycle",
            positive("nominal_t_per_cycle", self.nominal_t_per_cycle),
        )
        beta = nonnegative("accepted_beta", self.accepted_beta)
        if beta > 1.0:
            raise ODEBFContractError("accepted beta cannot expand nominal time")
        object.__setattr__(self, "accepted_beta", beta)
        object.__setattr__(self, "t_acc_before", nonnegative("t_acc_before", self.t_acc_before))

    @property
    def eta(self) -> float:
        return self.nominal_t_per_cycle / self.k_resolution

    @property
    def accepted_dt(self) -> float:
        return self.accepted_beta * self.eta

    @property
    def t_acc_after(self) -> float:
        return self.t_acc_before + self.accepted_dt

    @property
    def k_total_index(self) -> int:
        return self.correction_cycle * self.k_resolution + self.step_in_cycle


@dataclass(frozen=True, slots=True)
class DirectZDiagnostics:
    oracle_residual_norm: float
    projected_residual_norm: float
    projection_cosine: float
    local_progress_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "oracle_residual_norm",
            nonnegative("oracle_residual_norm", self.oracle_residual_norm),
        )
        object.__setattr__(
            self,
            "projected_residual_norm",
            nonnegative("projected_residual_norm", self.projected_residual_norm),
        )
        cosine = finite("projection_cosine", self.projection_cosine)
        if cosine < -1.0 or cosine > 1.0:
            raise ODEBFContractError("projection cosine is outside [-1,1]")
        object.__setattr__(self, "projection_cosine", cosine)
        fraction = finite("local_progress_fraction", self.local_progress_fraction)
        object.__setattr__(self, "local_progress_fraction", fraction)


@dataclass(frozen=True, slots=True)
class SealedRequest:
    case_id: int
    request_sha256: str
    ordinal: int

    def __post_init__(self) -> None:
        if isinstance(self.case_id, bool) or not isinstance(self.case_id, int) or self.case_id < 0:
            raise ODEBFContractError("case_id must be a nonnegative integer")
        if self.ordinal < 0 or self.ordinal >= BATCH_SIZE:
            raise ODEBFContractError("request ordinal is outside the joint batch")
        if len(self.request_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.request_sha256
        ):
            raise ODEBFContractError("request identity is not a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class JointBatchSeal:
    dataset_sha256: str
    salt: str
    requests: tuple[SealedRequest, ...]
    excluded_case_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.dataset_sha256) != 64:
            raise ODEBFContractError("dataset digest is not SHA-256")
        if not self.salt or self.salt.strip() != self.salt:
            raise ODEBFContractError("selection salt is empty or noncanonical")
        if len(self.requests) != BATCH_SIZE:
            raise ODEBFContractError("P0 seal must contain exactly ten requests")
        if tuple(item.ordinal for item in self.requests) != tuple(range(BATCH_SIZE)):
            raise ODEBFContractError("P0 request order is not explicit and contiguous")
        case_ids = tuple(item.case_id for item in self.requests)
        hashes = tuple(item.request_sha256 for item in self.requests)
        if len(set(case_ids)) != BATCH_SIZE or len(set(hashes)) != BATCH_SIZE:
            raise ODEBFContractError("joint batch contains a duplicate request")
        if set(case_ids).intersection(self.excluded_case_ids):
            raise ODEBFContractError("joint batch collides with a sealed exclusion")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "dataset_sha256": self.dataset_sha256,
            "salt": self.salt,
            "edit_batch_size": BATCH_SIZE,
            "requests": [
                {
                    "case_id": item.case_id,
                    "request_sha256": item.request_sha256,
                    "ordinal": item.ordinal,
                }
                for item in self.requests
            ],
            "excluded_case_ids": list(self.excluded_case_ids),
        }

    def root_digest(self) -> str:
        return canonical_hash(self.identity_payload())


def assert_joint_request_batch(
    requests: Sequence[Mapping[str, Any]],
    *,
    allow_singleton_algebra_fixture: bool = False,
) -> None:
    count = len(requests)
    if allow_singleton_algebra_fixture and count == 1:
        return
    if count != BATCH_SIZE:
        raise ODEBFContractError("integration path requires one joint ten-request call")
    case_ids = [item.get("case_id") for item in requests]
    identities = [item.get("request_sha256") for item in requests]
    if any(value is None for value in case_ids + identities):
        raise ODEBFContractError("joint request batch identity is incomplete")
    if len(set(case_ids)) != BATCH_SIZE or len(set(identities)) != BATCH_SIZE:
        raise ODEBFContractError("joint request batch is not distinct")


def assert_common_policy(policy_by_alias: Mapping[str, Any]) -> str:
    if set(policy_by_alias) != set(MODEL_ALIASES):
        raise ODEBFContractError("both locked model aliases are required")
    identities = {canonical_hash(policy_by_alias[alias]) for alias in MODEL_ALIASES}
    if len(identities) != 1:
        raise ODEBFContractError("model aliases received different controller policy")
    return identities.pop()


def assert_matched_projector_arms(
    generic_contract: Mapping[str, Any],
    bf_contract: Mapping[str, Any],
) -> None:
    """Permit only the barrier projector field to differ in a matched pair."""

    generic = dict(generic_contract)
    bf = dict(bf_contract)
    generic_projector = generic.pop("barrier_projector", None)
    bf_projector = bf.pop("barrier_projector", None)
    if generic_projector != "identity" or bf_projector != "cbf":
        raise ODEBFContractError("matched arms do not identify Generic and BF projectors")
    if canonical_hash(generic) != canonical_hash(bf):
        raise ODEBFContractError("Generic and BF differ beyond the projector")


@dataclass(frozen=True, slots=True)
class DirectSumArm:
    """One layer arm in a parameter direct sum.

    ``theta = eta * beta`` is the finite coefficient applied to the low-rank
    arm.  ``velocity`` is never reused as a finite increment.
    """

    weight_name: str
    layer: int
    out_features: int
    in_features: int
    rank: int

    def __post_init__(self) -> None:
        if not self.weight_name.endswith(".weight"):
            raise ODEBFContractError("direct-sum arm must identify a weight")
        for name, value in (
            ("out_features", self.out_features),
            ("in_features", self.in_features),
            ("rank", self.rank),
        ):
            positive_integer(name, value)
        if self.rank > BATCH_SIZE:
            raise ODEBFContractError("joint batch arm rank exceeds ten")


def assert_direct_sum_unique(arms: Sequence[DirectSumArm]) -> None:
    if len(arms) < 2:
        raise ODEBFContractError("dynamic routing requires at least two layer arms")
    if len({arm.weight_name for arm in arms}) != len(arms):
        raise ODEBFContractError("parameter direct sum repeats a weight")
    if len({arm.layer for arm in arms}) != len(arms):
        raise ODEBFContractError("parameter direct sum repeats a layer")


@dataclass(frozen=True, slots=True)
class FieldIdentity:
    batch_id: str
    accepted_waypoint: int
    virtual_state_version: int
    target_state_version: int
    proposal_version: int
    history_version: int

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ODEBFContractError("field batch identity is empty")
        for name, value in (
            ("accepted_waypoint", self.accepted_waypoint),
            ("virtual_state_version", self.virtual_state_version),
            ("target_state_version", self.target_state_version),
            ("proposal_version", self.proposal_version),
            ("history_version", self.history_version),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ODEBFContractError(f"{name} must be a nonnegative integer")

    def rejected_trial_identity(self) -> str:
        return canonical_hash(
            {
                "batch_id": self.batch_id,
                "accepted_waypoint": self.accepted_waypoint,
                "virtual_state_version": self.virtual_state_version,
                "target_state_version": self.target_state_version,
                "proposal_version": self.proposal_version,
                "history_version": self.history_version,
            }
        )
