"""Adaptive pseudo-time and routing telemetry for the P1R4 causal panel.

The clock in this module is deliberately independent of model state.  A
rejected trial may change only its proposal size and diagnostic counters;
accepted-state ownership stays with the runtime that supplies the immutable
state/field identities to the receipts.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import WaypointFactor
from .p1_backend import P1DynamicField


ADAPTIVE_INSTRUCTION_ID = (
    "ODEEDIT-S04-ODE-BF-FULL-RESIDUAL-ADAPTIVE-TAU-P1R4-V1"
)
T_MAX = Fraction(1, 1)
H_REF = Fraction(1, 8)
DELTA_TAU_MIN = Fraction(1, 128)
GAMMA_DOWN = Fraction(1, 2)
GAMMA_UP = Fraction(3, 2)
RHO_ACCEPT = 0.1
RHO_EXPAND = 0.75
SAME_STATE_ADDITIONAL_RETRY_CAP = 4
N_TRIAL_CAP = 128


class AdaptiveVariant(str, Enum):
    PS_S8 = "PS-S8"
    PS_A8 = "PS-A8"
    FR_A8 = "FR-A8"
    FR_A16 = "FR-A16"


ADAPTIVE_VARIANTS = tuple(AdaptiveVariant)


def fraction_payload(value: Fraction) -> dict[str, int | float]:
    normalized = Fraction(value)
    return {
        "numerator": normalized.numerator,
        "denominator": normalized.denominator,
        "value": float(normalized),
    }


@dataclass(frozen=True, slots=True)
class AdaptiveTauLock:
    variant: AdaptiveVariant
    delta_tau_max: Fraction
    k_acc_cap: int
    legacy_slot_cap: int = 0
    t_max: Fraction = T_MAX
    h_ref: Fraction = H_REF
    delta_tau_min: Fraction = DELTA_TAU_MIN
    gamma_down: Fraction = GAMMA_DOWN
    gamma_up: Fraction = GAMMA_UP
    rho_accept: float = RHO_ACCEPT
    rho_expand: float = RHO_EXPAND
    same_state_additional_retry_cap: int = SAME_STATE_ADDITIONAL_RETRY_CAP
    n_trial_cap: int = N_TRIAL_CAP

    def __post_init__(self) -> None:
        expected = {
            AdaptiveVariant.PS_S8: (H_REF, 8, 8),
            AdaptiveVariant.PS_A8: (H_REF, 32, 0),
            AdaptiveVariant.FR_A8: (H_REF, 32, 0),
            AdaptiveVariant.FR_A16: (Fraction(1, 16), 64, 0),
        }[self.variant]
        if (Fraction(self.delta_tau_max), self.k_acc_cap, self.legacy_slot_cap) != expected:
            raise ODEBFContractError("adaptive variant horizon contract differs")
        if (
            Fraction(self.t_max) != T_MAX
            or Fraction(self.h_ref) != H_REF
            or Fraction(self.delta_tau_min) != DELTA_TAU_MIN
            or Fraction(self.gamma_down) != GAMMA_DOWN
            or Fraction(self.gamma_up) != GAMMA_UP
            or self.rho_accept != RHO_ACCEPT
            or self.rho_expand != RHO_EXPAND
            or self.same_state_additional_retry_cap
            != SAME_STATE_ADDITIONAL_RETRY_CAP
            or self.n_trial_cap != N_TRIAL_CAP
        ):
            raise ODEBFContractError("adaptive pseudo-time numerical lock differs")
        if self.k_acc_cap <= 0 or self.n_trial_cap <= 0:
            raise ODEBFContractError("adaptive resource cap is nonpositive")

    @property
    def adaptive(self) -> bool:
        return self.variant is not AdaptiveVariant.PS_S8

    def identity(self) -> str:
        return canonical_hash(
            {
                "variant": self.variant.value,
                "t_max": fraction_payload(self.t_max),
                "h_ref": fraction_payload(self.h_ref),
                "delta_tau_max": fraction_payload(self.delta_tau_max),
                "delta_tau_min": fraction_payload(self.delta_tau_min),
                "gamma_down": fraction_payload(self.gamma_down),
                "gamma_up": fraction_payload(self.gamma_up),
                "rho_accept": self.rho_accept,
                "rho_expand": self.rho_expand,
                "same_state_additional_retry_cap": (
                    self.same_state_additional_retry_cap
                ),
                "k_acc_cap": self.k_acc_cap,
                "n_trial_cap": self.n_trial_cap,
                "legacy_slot_cap": self.legacy_slot_cap,
                "reject_consumes_accepted_index": False,
                "reject_consumes_pseudo_time": False,
                "target_and_weight_share_delta_tau": True,
            }
        )


def adaptive_lock(variant: AdaptiveVariant) -> AdaptiveTauLock:
    if variant is AdaptiveVariant.PS_S8:
        return AdaptiveTauLock(variant, H_REF, 8, 8)
    if variant in (AdaptiveVariant.PS_A8, AdaptiveVariant.FR_A8):
        return AdaptiveTauLock(variant, H_REF, 32)
    if variant is AdaptiveVariant.FR_A16:
        return AdaptiveTauLock(variant, Fraction(1, 16), 64)
    raise ODEBFContractError("unknown adaptive variant")


@dataclass(frozen=True, slots=True)
class TauTrial:
    n_trial: int
    k_acc_before: int
    retry_index: int
    tau_before: Fraction
    delta_tau_proposed: Fraction
    delta_tau_trial: Fraction
    remainder: bool

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "n_trial": self.n_trial,
            "k_acc_before": self.k_acc_before,
            "retry_index": self.retry_index,
            "tau_before": fraction_payload(self.tau_before),
            "delta_tau_proposed": fraction_payload(self.delta_tau_proposed),
            "delta_tau_trial": fraction_payload(self.delta_tau_trial),
            "remainder": self.remainder,
        }


@dataclass(frozen=True, slots=True)
class TauTransition:
    accepted: bool
    tau_before: Fraction
    tau_after: Fraction
    k_acc_before: int
    k_acc_after: int
    n_trial: int
    n_reject: int
    retry_index_after: int
    next_delta_tau: Fraction
    rho: float | None
    status: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "tau_before": fraction_payload(self.tau_before),
            "tau_after": fraction_payload(self.tau_after),
            "k_acc_before": self.k_acc_before,
            "k_acc_after": self.k_acc_after,
            "n_trial": self.n_trial,
            "n_reject": self.n_reject,
            "retry_index_after": self.retry_index_after,
            "next_delta_tau": fraction_payload(self.next_delta_tau),
            "rho": self.rho,
            "status": self.status,
        }


class AdaptiveTauClock:
    """Exact dyadic pseudo-time with rejection-local contraction."""

    def __init__(self, lock: AdaptiveTauLock) -> None:
        if not lock.adaptive:
            raise ODEBFContractError("legacy PS-S8 cannot use adaptive clock")
        self.lock = lock
        self.tau = Fraction(0, 1)
        self.delta_tau_proposed = Fraction(lock.delta_tau_max)
        self.k_acc = 0
        self.n_trial = 0
        self.n_reject = 0
        self.retry_index = 0
        self.status = "ACTIVE"
        self._open_trial: TauTrial | None = None
        self._accepted_delta: list[Fraction] = []

    @property
    def complete(self) -> bool:
        return self.tau == self.lock.t_max

    @property
    def accepted_delta(self) -> tuple[Fraction, ...]:
        return tuple(self._accepted_delta)

    def _stop(self, status: str) -> None:
        if status not in (
            "TRIAL_CAP_EXHAUSTED",
            "ACCEPTED_STEP_OR_HORIZON_CAP_UNREACHED",
            "SAME_STATE_RETRY_EXHAUSTED",
            "MIN_DT_EXHAUSTED",
        ):
            raise ODEBFContractError("adaptive stop status differs")
        self.status = status

    def begin_trial(self) -> TauTrial:
        if self._open_trial is not None:
            raise ODEBFStateError("adaptive clock already has an open trial")
        if self.status != "ACTIVE" or self.complete:
            raise ODEBFStateError("adaptive clock cannot begin another trial")
        if self.n_trial >= self.lock.n_trial_cap:
            self._stop("TRIAL_CAP_EXHAUSTED")
            raise ODEBFStateError("adaptive aggregate trial cap reached")
        if self.k_acc >= self.lock.k_acc_cap:
            self._stop("ACCEPTED_STEP_OR_HORIZON_CAP_UNREACHED")
            raise ODEBFStateError("adaptive accepted-step cap reached")
        remaining = self.lock.t_max - self.tau
        delta = min(self.delta_tau_proposed, remaining)
        remainder = delta < self.lock.delta_tau_min
        if remainder and delta != remaining:
            raise ODEBFContractError("sub-minimum adaptive step is not a final remainder")
        if delta <= 0:
            raise ODEBFContractError("adaptive trial has nonpositive pseudo-time")
        self.n_trial += 1
        self._open_trial = TauTrial(
            self.n_trial,
            self.k_acc,
            self.retry_index,
            self.tau,
            self.delta_tau_proposed,
            delta,
            remainder,
        )
        return self._open_trial

    def accept(self, *, trial: TauTrial, rho: float) -> TauTransition:
        if trial is not self._open_trial:
            raise ODEBFStateError("adaptive accepted a non-open trial")
        observed_rho = float(rho)
        if not math.isfinite(observed_rho) or observed_rho < self.lock.rho_accept:
            raise ODEBFContractError("adaptive accepted trial has invalid rho")
        before = self.tau
        self.tau += trial.delta_tau_trial
        self.k_acc += 1
        self._accepted_delta.append(trial.delta_tau_trial)
        if self.tau > self.lock.t_max or sum(self._accepted_delta, Fraction(0, 1)) != self.tau:
            raise ODEBFStateError("adaptive accepted pseudo-time accounting differs")
        next_delta = trial.delta_tau_trial
        if observed_rho > self.lock.rho_expand:
            next_delta = min(
                self.lock.delta_tau_max,
                self.lock.gamma_up * trial.delta_tau_trial,
            )
        self.delta_tau_proposed = next_delta
        self.retry_index = 0
        self._open_trial = None
        if self.complete:
            self.status = "TAU_COMPLETE"
        return TauTransition(
            True,
            before,
            self.tau,
            trial.k_acc_before,
            self.k_acc,
            self.n_trial,
            self.n_reject,
            self.retry_index,
            self.delta_tau_proposed,
            observed_rho,
            self.status,
        )

    def reject(self, *, trial: TauTrial) -> TauTransition:
        if trial is not self._open_trial:
            raise ODEBFStateError("adaptive rejected a non-open trial")
        before = self.tau
        self.n_reject += 1
        self.retry_index += 1
        next_delta = trial.delta_tau_trial * self.lock.gamma_down
        if self.retry_index > self.lock.same_state_additional_retry_cap:
            self._stop("SAME_STATE_RETRY_EXHAUSTED")
            next_delta = trial.delta_tau_trial
        elif next_delta < self.lock.delta_tau_min:
            self._stop("MIN_DT_EXHAUSTED")
            next_delta = trial.delta_tau_trial
        else:
            self.delta_tau_proposed = next_delta
        self._open_trial = None
        if self.tau != before or self.k_acc != trial.k_acc_before:
            raise ODEBFStateError("adaptive rejection advanced accepted state")
        return TauTransition(
            False,
            before,
            self.tau,
            trial.k_acc_before,
            self.k_acc,
            self.n_trial,
            self.n_reject,
            self.retry_index,
            next_delta,
            None,
            self.status,
        )

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "lock_sha256": self.lock.identity(),
            "status": self.status,
            "tau": fraction_payload(self.tau),
            "delta_tau_proposed": fraction_payload(self.delta_tau_proposed),
            "k_acc": self.k_acc,
            "n_trial": self.n_trial,
            "n_reject": self.n_reject,
            "retry_index": self.retry_index,
            "accepted_delta_tau": [
                fraction_payload(value) for value in self._accepted_delta
            ],
            "tau_sum_exact": sum(self._accepted_delta, Fraction(0, 1))
            == self.tau,
        }


@dataclass(frozen=True, slots=True)
class StepSizeIndependentVelocity:
    field_sha256: str
    h_ref: Fraction
    nominal_applied: tuple[float, ...]
    velocity: tuple[float, ...]
    nominal_sha256: str
    velocity_sha256: str

    def __post_init__(self) -> None:
        if len(self.field_sha256) != 64 or self.h_ref != H_REF:
            raise ODEBFContractError("adaptive velocity field/h_ref differs")
        if len(self.velocity) < 2 or len(self.nominal_applied) != len(self.velocity):
            raise ODEBFContractError("adaptive velocity dimension differs")
        expected = tuple(float(self.h_ref) * value for value in self.velocity)
        if self.nominal_applied != expected:
            raise ODEBFContractError("adaptive nominal coefficient/velocity differs")
        if self.nominal_sha256 != canonical_hash(list(self.nominal_applied)):
            raise ODEBFContractError("adaptive nominal coefficient hash differs")
        if self.velocity_sha256 != canonical_hash(list(self.velocity)):
            raise ODEBFContractError("adaptive velocity hash differs")

    @classmethod
    def from_controller_values(
        cls,
        field_sha256: str,
        controller_values: Sequence[float],
    ) -> "StepSizeIndependentVelocity":
        velocity = tuple(float(value) for value in controller_values)
        if not velocity or not all(math.isfinite(value) and value >= 0.0 for value in velocity):
            raise ODEBFContractError("adaptive controller velocity is invalid")
        nominal = tuple(float(H_REF) * value for value in velocity)
        return cls(
            field_sha256,
            H_REF,
            nominal,
            velocity,
            canonical_hash(list(nominal)),
            canonical_hash(list(velocity)),
        )

    def applied(self, delta_tau: Fraction) -> tuple[float, ...]:
        delta = float(Fraction(delta_tau))
        return tuple(delta * value for value in self.velocity)


def adaptive_waypoint_factors(
    field: P1DynamicField,
    velocity: StepSizeIndependentVelocity,
    *,
    accepted_index: int,
    delta_tau: Fraction,
) -> dict[str, WaypointFactor]:
    if velocity.field_sha256 != field.identity_sha256:
        raise ODEBFContractError("adaptive factor uses another field")
    if accepted_index < 0:
        raise ODEBFContractError("adaptive accepted index is negative")
    delta = Fraction(delta_tau)
    if delta <= 0 or delta > H_REF:
        raise ODEBFContractError("adaptive factor delta_tau differs")
    values = velocity.applied(delta)
    result: dict[str, WaypointFactor] = {}
    for ordinal, (layer_field, theta) in enumerate(zip(field.layers, values)):
        result[layer_field.weight_name] = WaypointFactor(
            layer_field.weight_name,
            layer_field.layer,
            0,
            accepted_index,
            ordinal,
            theta,
            layer_field.residual.clone(),
            layer_field.q.clone(),
        )
    return result


@dataclass(frozen=True, slots=True)
class BF16CapacityReceipt:
    weight_sha256: str
    previous_effective_sha256: str
    candidate_effective_sha256: str
    step_energy: float
    cumulative_energy: float
    maximum_fp32_block_elements: int
    dense_fp32_full_delta_live: int = 0
    full_effective_weight_live: int = 0

    def __post_init__(self) -> None:
        for value in (
            self.weight_sha256,
            self.previous_effective_sha256,
            self.candidate_effective_sha256,
        ):
            if len(value) != 64:
                raise ODEBFContractError("streaming BF16 capacity digest differs")
        if (
            not math.isfinite(self.step_energy)
            or not math.isfinite(self.cumulative_energy)
            or self.step_energy < 0.0
            or self.cumulative_energy < 0.0
            or self.maximum_fp32_block_elements <= 0
            or self.dense_fp32_full_delta_live != 0
            or self.full_effective_weight_live != 0
        ):
            raise ODEBFContractError("streaming BF16 capacity contract differs")


@dataclass(frozen=True, slots=True)
class BF16EndpointDistanceReceipt:
    weight_sha256: str
    left_endpoint_sha256: str
    right_endpoint_sha256: str
    frobenius_distance: float
    maximum_absolute_distance: float
    maximum_fp32_block_elements: int
    dense_fp32_full_delta_live: int = 0
    full_effective_weight_live: int = 0

    def __post_init__(self) -> None:
        if any(
            len(value) != 64
            for value in (
                self.weight_sha256,
                self.left_endpoint_sha256,
                self.right_endpoint_sha256,
            )
        ):
            raise ODEBFContractError("BF16 endpoint-distance digest differs")
        if (
            not math.isfinite(self.frobenius_distance)
            or not math.isfinite(self.maximum_absolute_distance)
            or self.frobenius_distance < 0.0
            or self.maximum_absolute_distance < 0.0
            or self.maximum_fp32_block_elements <= 0
            or self.dense_fp32_full_delta_live != 0
            or self.full_effective_weight_live != 0
        ):
            raise ODEBFContractError("BF16 endpoint-distance contract differs")


def _effective_bf16_block(
    entry_weight: torch.Tensor,
    factors: Sequence[WaypointFactor],
    *,
    start: int,
    end: int,
) -> torch.Tensor:
    accumulator = entry_weight.detach()[start:end].to(dtype=torch.float32)
    ordered = tuple(sorted(factors, key=lambda factor: factor.order_key))
    if ordered and len({factor.order_key for factor in ordered}) != len(ordered):
        raise ODEBFContractError("adaptive capacity factor order repeats")
    for factor in ordered:
        if tuple(entry_weight.shape) != (factor.left.shape[0], factor.right.shape[0]):
            raise ODEBFContractError("adaptive capacity factor geometry differs")
        left = factor.left[start:end].to(
            device=entry_weight.device, dtype=torch.float32
        )
        right = factor.right.to(device=entry_weight.device, dtype=torch.float32)
        native_order = (right @ left.T).T
        coefficient = torch.as_tensor(
            factor.theta, dtype=torch.float32, device=entry_weight.device
        )
        accumulator = accumulator + coefficient * native_order
        del left, right, native_order, coefficient
    return accumulator.to(dtype=torch.bfloat16)


def streaming_bf16_capacity(
    entry_weight: torch.Tensor,
    previous: Sequence[WaypointFactor],
    candidate: Sequence[WaypointFactor],
    *,
    weight_name: str,
    row_block: int = 64,
) -> BF16CapacityReceipt:
    """Compare endpoints blockwise without retaining a full effective tensor."""

    if entry_weight.dtype is not torch.bfloat16 or entry_weight.ndim != 2:
        raise ODEBFContractError("adaptive capacity entry is not BF16 matrix")
    if row_block <= 0:
        raise ODEBFContractError("adaptive capacity row block differs")
    previous_digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()
    step_energy = 0.0
    cumulative_energy = 0.0
    maximum_block = 0
    with torch.no_grad():
        for start in range(0, entry_weight.shape[0], row_block):
            end = min(start + row_block, entry_weight.shape[0])
            before = _effective_bf16_block(
                entry_weight, previous, start=start, end=end
            )
            after = _effective_bf16_block(
                entry_weight, candidate, start=start, end=end
            )
            previous_digest.update(
                before.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
            )
            candidate_digest.update(
                after.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
            )
            step = after.float() - before.float()
            cumulative = after.float() - entry_weight.detach()[start:end].float()
            step_energy += float(torch.sum(step.double() * step.double()))
            cumulative_energy += float(
                torch.sum(cumulative.double() * cumulative.double())
            )
            maximum_block = max(maximum_block, before.numel(), after.numel())
            del before, after, step, cumulative
    return BF16CapacityReceipt(
        canonical_hash({"weight_name": weight_name}),
        previous_digest.hexdigest(),
        candidate_digest.hexdigest(),
        step_energy,
        cumulative_energy,
        maximum_block,
    )


def streaming_bf16_endpoint_distance(
    entry_weight: torch.Tensor,
    left_factors: Sequence[WaypointFactor],
    right_factors: Sequence[WaypointFactor],
    *,
    weight_name: str,
    row_block: int = 64,
) -> BF16EndpointDistanceReceipt:
    """Compare two quantized endpoints blockwise without a dense delta."""

    if entry_weight.dtype is not torch.bfloat16 or entry_weight.ndim != 2:
        raise ODEBFContractError("endpoint-distance entry is not BF16 matrix")
    if row_block <= 0:
        raise ODEBFContractError("endpoint-distance row block differs")
    left_digest = hashlib.sha256()
    right_digest = hashlib.sha256()
    squared = 0.0
    maximum = 0.0
    maximum_block = 0
    with torch.no_grad():
        for start in range(0, entry_weight.shape[0], row_block):
            end = min(start + row_block, entry_weight.shape[0])
            left = _effective_bf16_block(
                entry_weight, left_factors, start=start, end=end
            )
            right = _effective_bf16_block(
                entry_weight, right_factors, start=start, end=end
            )
            left_digest.update(
                left.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
            )
            right_digest.update(
                right.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
            )
            difference = left.float() - right.float()
            squared += float(torch.sum(difference.double() * difference.double()))
            maximum = max(maximum, float(torch.max(torch.abs(difference))))
            maximum_block = max(maximum_block, left.numel(), right.numel())
            del left, right, difference
    return BF16EndpointDistanceReceipt(
        canonical_hash({"weight_name": weight_name}),
        left_digest.hexdigest(),
        right_digest.hexdigest(),
        math.sqrt(squared),
        maximum,
        maximum_block,
    )


def _shares(values: Sequence[float]) -> np.ndarray:
    observed = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if observed.ndim != 1 or observed.size < 2 or not np.isfinite(observed).all():
        raise ODEBFContractError("routing concentration vector differs")
    observed = np.maximum(observed, 0.0)
    total = float(observed.sum())
    return np.zeros_like(observed) if total == 0.0 else observed / total


def concentration_summary(values: Sequence[float]) -> dict[str, Any]:
    shares = _shares(values)
    nonzero = shares[shares > 0.0]
    hhi = float(shares @ shares)
    entropy = (
        0.0
        if nonzero.size == 0
        else float(-(nonzero * np.log(nonzero)).sum() / math.log(shares.size))
    )
    ordered = np.sort(shares)
    gini = (
        0.0
        if float(ordered.sum()) == 0.0
        else float(
            (2.0 * np.arange(1, shares.size + 1) @ ordered)
            / (shares.size * ordered.sum())
            - (shares.size + 1) / shares.size
        )
    )
    depth = np.linspace(0.0, 1.0, shares.size)
    descending = np.sort(shares)[::-1]
    return {
        "numeric_vector": [float(value) for value in values],
        "share": shares.tolist(),
        "share_sha256": canonical_hash(shares.tolist()),
        "top1_share": float(descending[0]),
        "top2_share": float(descending[:2].sum()),
        "max_share": float(descending[0]),
        "hhi": hhi,
        "effective_layer_count": 0.0 if hhi == 0.0 else 1.0 / hhi,
        "normalized_entropy": entropy,
        "gini": gini,
        "depth_centroid": float(shares @ depth),
    }


def routing_change(
    left: Sequence[float],
    right: Sequence[float],
) -> dict[str, float]:
    left_share = _shares(left)
    right_share = _shares(right)
    l1 = float(np.abs(left_share - right_share).sum())
    left_norm = float(np.linalg.norm(left_share))
    right_norm = float(np.linalg.norm(right_share))
    cosine = (
        1.0
        if left_norm == 0.0 and right_norm == 0.0
        else 0.0
        if left_norm == 0.0 or right_norm == 0.0
        else float(left_share @ right_share / (left_norm * right_norm))
    )
    midpoint = 0.5 * (left_share + right_share)

    def kl(value: np.ndarray, reference: np.ndarray) -> float:
        mask = value > 0.0
        return float((value[mask] * np.log(value[mask] / reference[mask])).sum())

    js = 0.0 if float(midpoint.sum()) == 0.0 else 0.5 * (
        kl(left_share, midpoint) + kl(right_share, midpoint)
    )
    left_support = set(np.flatnonzero(left_share > 0.0).tolist())
    right_support = set(np.flatnonzero(right_share > 0.0).tolist())
    union = left_support | right_support
    turnover = 0.0 if not union else 1.0 - len(left_support & right_support) / len(union)
    return {
        "l1": l1,
        "cosine": cosine,
        "jensen_shannon": js,
        "support_turnover": turnover,
    }


@dataclass(frozen=True, slots=True)
class FirstHitRecord:
    accepted_index: int
    tau: Fraction
    snapshot_sha256: str
    success_count: int
    online_component_feasible: bool


class FirstHitTracker:
    def __init__(self) -> None:
        self._records: list[FirstHitRecord] = []
        self._first_online: FirstHitRecord | None = None

    def append(self, record: FirstHitRecord) -> None:
        if record.accepted_index != len(self._records) + 1:
            raise ODEBFStateError("adaptive accepted hit records are not contiguous")
        if len(record.snapshot_sha256) != 64 or not 0 <= record.success_count <= BATCH_SIZE:
            raise ODEBFContractError("adaptive first-hit record differs")
        if self._records and record.tau <= self._records[-1].tau:
            raise ODEBFStateError("adaptive accepted hit pseudo-time did not increase")
        self._records.append(record)
        if (
            self._first_online is None
            and record.success_count == BATCH_SIZE
            and record.online_component_feasible
        ):
            self._first_online = record

    @property
    def records(self) -> tuple[FirstHitRecord, ...]:
        return tuple(self._records)

    @property
    def first_online(self) -> FirstHitRecord | None:
        return self._first_online

    def prefix_identity(self, accepted_count: int | None = None) -> str:
        values = self._records if accepted_count is None else self._records[:accepted_count]
        return canonical_hash(
            [
                {
                    **asdict(item),
                    "tau": fraction_payload(item.tau),
                }
                for item in values
            ]
        )


def explicit_euler_terminal(step: Fraction) -> float:
    """Smooth non-boundary float64 toy used only by the CPU refinement gate."""

    h = float(Fraction(step))
    count = int(T_MAX / Fraction(step))
    state = np.float64(0.25)
    for _ in range(count):
        state = np.float64(state + h * (np.float64(1.0) - state))
    return float(state)


def refinement_ratio() -> dict[str, float | bool]:
    s8 = explicit_euler_terminal(Fraction(1, 8))
    s16 = explicit_euler_terminal(Fraction(1, 16))
    s32 = explicit_euler_terminal(Fraction(1, 32))
    numerator = abs(s8 - s16)
    denominator = abs(s16 - s32)
    if denominator == 0.0:
        raise ODEBFContractError("refinement toy denominator is zero")
    ratio = numerator / denominator
    return {
        "s_1_8": s8,
        "s_1_16": s16,
        "s_1_32": s32,
        "difference_8_16": numerator,
        "difference_16_32": denominator,
        "ratio": ratio,
        "passed": 1.6 <= ratio <= 2.4,
    }
