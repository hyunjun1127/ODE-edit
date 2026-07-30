"""Pure-CPU, outcome-firewalled analysis primitives for MV-1.

The contracts in this module contain only case IDs, action IDs, hashes, status,
and scalar measurements.  They deliberately have no prompt, answer, target, or
evaluation-text field.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .contracts import ContractError, canonical_json, sha256_bytes


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    if len(value) > 256 or any(ord(char) < 32 for char in value):
        raise ContractError(f"{name} is not a valid identifier")
    return value


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def _full_hash(name: str, value: Any) -> str:
    value = _identifier(name, value).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ContractError(f"{name} must be a full SHA-256 digest")
    return value


def _case_hash(case_ids: Iterable[str]) -> str:
    normalized = sorted(_identifier("case_id", case_id) for case_id in case_ids)
    if not normalized or len(normalized) != len(set(normalized)):
        raise ContractError("case IDs must be non-empty and unique")
    return sha256_bytes(canonical_json(normalized).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class LockedAction:
    """One member of the finite action set; lower ranks win a policy tie."""

    action_id: str
    simplicity_rank: int
    compute_rank: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _identifier("action_id", self.action_id))
        for name in ("simplicity_rank", "compute_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "simplicity_rank": self.simplicity_rank,
            "compute_rank": self.compute_rank,
        }


@dataclass(frozen=True, slots=True)
class LockedActionSet:
    """Calibration-locked finite actions and replay near-tie envelope."""

    actions: tuple[LockedAction, ...]
    near_tie_tolerance: float

    def __post_init__(self) -> None:
        if not isinstance(self.actions, tuple) or not self.actions:
            raise ContractError("locked action set must be a non-empty tuple")
        action_ids = [action.action_id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ContractError("locked action IDs must be unique")
        tolerance = _finite("near_tie_tolerance", self.near_tie_tolerance)
        if tolerance < 0:
            raise ContractError("near_tie_tolerance must be non-negative")
        object.__setattr__(
            self,
            "actions",
            tuple(sorted(self.actions, key=lambda item: item.action_id)),
        )
        object.__setattr__(self, "near_tie_tolerance", tolerance)

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(action.action_id for action in self.actions)

    @property
    def action_set_id(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict(include_id=False)).encode("utf-8"))

    def action(self, action_id: str) -> LockedAction:
        for action in self.actions:
            if action.action_id == action_id:
                return action
        raise ContractError(f"action is outside the locked finite set: {action_id}")

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "ode-edit-mv1-action-set/v1",
            "actions": [action.to_dict() for action in self.actions],
            "near_tie_tolerance": self.near_tie_tolerance,
        }
        if include_id:
            payload["action_set_id"] = self.action_set_id
        return payload


class OutcomeStatus(str, Enum):
    COMPLETED = "completed"
    PRE_SATISFIED = "pre_satisfied"
    TARGET_NOT_REACHED = "target_not_reached"
    REJECTED = "rejected"
    NON_FINITE = "non_finite"
    ROLLBACK = "rollback"
    TECHNICAL_FAILURE = "technical_failure"


@dataclass(frozen=True, slots=True)
class SanitizedOutcome:
    """An ITD event-arm row with no text-bearing research input."""

    model_id: str
    case_id: str
    action_id: str
    progress: float | None
    reached: bool
    status: OutcomeStatus

    def __post_init__(self) -> None:
        for name in ("model_id", "case_id", "action_id"):
            object.__setattr__(self, name, _identifier(name, getattr(self, name)))
        if not isinstance(self.reached, bool):
            raise ContractError("reached must be boolean")
        if not isinstance(self.status, OutcomeStatus):
            try:
                object.__setattr__(self, "status", OutcomeStatus(self.status))
            except ValueError as exc:
                raise ContractError(f"unknown outcome status: {self.status}") from exc
        if self.progress is not None:
            object.__setattr__(self, "progress", _finite("progress", self.progress))
        if self.status is OutcomeStatus.COMPLETED and self.progress is None:
            raise ContractError("completed outcome requires finite progress")
        if self.status is OutcomeStatus.NON_FINITE and self.progress is not None:
            raise ContractError("non_finite outcome must not persist a non-finite value")
        if self.status is OutcomeStatus.TARGET_NOT_REACHED and self.reached:
            raise ContractError("target_not_reached outcome cannot be marked reached")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SanitizedOutcome":
        if not isinstance(raw, Mapping):
            raise ContractError("outcome must be a mapping")
        allowed = {"model_id", "case_id", "action_id", "progress", "reached", "status"}
        if set(raw) != allowed:
            raise ContractError(
                f"sanitized outcome requires exactly {sorted(allowed)}"
            )
        return cls(**{key: raw[key] for key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "progress": self.progress,
            "reached": self.reached,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class ITDFailurePolicy:
    """Precommitted score for rows whose finite progress is unavailable."""

    fallback_progress: float
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fallback_progress", _finite("fallback_progress", self.fallback_progress)
        )
        object.__setattr__(self, "label", _identifier("ITD policy label", self.label))

    @property
    def policy_id(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict(include_id=False)).encode("utf-8"))

    def score(self, outcome: SanitizedOutcome) -> float:
        return self.fallback_progress if outcome.progress is None else outcome.progress

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "ode-edit-mv1-itd-policy/v1",
            "fallback_progress": self.fallback_progress,
            "label": self.label,
        }
        if include_id:
            payload["policy_id"] = self.policy_id
        return payload


def _outcome_index(
    outcomes: Iterable[SanitizedOutcome],
) -> dict[tuple[str, str, str], SanitizedOutcome]:
    indexed: dict[tuple[str, str, str], SanitizedOutcome] = {}
    for outcome in outcomes:
        if not isinstance(outcome, SanitizedOutcome):
            raise ContractError("analysis accepts SanitizedOutcome rows only")
        key = (outcome.model_id, outcome.case_id, outcome.action_id)
        if key in indexed:
            raise ContractError(f"duplicate outcome row: {key}")
        indexed[key] = outcome
    return indexed


def _require_panel(
    indexed: Mapping[tuple[str, str, str], SanitizedOutcome],
    *,
    model_id: str,
    case_ids: Sequence[str],
    action_ids: Sequence[str],
) -> None:
    missing = [
        (model_id, case_id, action_id)
        for case_id in case_ids
        for action_id in action_ids
        if (model_id, case_id, action_id) not in indexed
    ]
    if missing:
        raise ContractError(
            f"ITD panel is missing {len(missing)} required event-arm rows; "
            f"first={missing[0]}"
        )


@dataclass(frozen=True, slots=True)
class StaticPolicyLock:
    model_id: str
    action_id: str
    action_set_id: str
    calibration_case_hash: str
    itd_policy_id: str
    near_tied_actions: tuple[str, ...]
    tie_rule: str = "within-tolerance->simplicity_rank->compute_rank->action_id"

    @property
    def lock_id(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict(include_id=False)).encode("utf-8"))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "ode-edit-mv1-static-policy/v1",
            "model_id": self.model_id,
            "action_id": self.action_id,
            "action_set_id": self.action_set_id,
            "calibration_case_hash": self.calibration_case_hash,
            "itd_policy_id": self.itd_policy_id,
            "near_tied_actions": list(self.near_tied_actions),
            "tie_rule": self.tie_rule,
        }
        if include_id:
            payload["lock_id"] = self.lock_id
        return payload


def select_calibration_best_static(
    outcomes: Iterable[SanitizedOutcome],
    *,
    model_id: str,
    calibration_case_ids: Sequence[str],
    action_set: LockedActionSet,
    itd_policy: ITDFailurePolicy,
) -> StaticPolicyLock:
    """Select one model-wide static action, never an event-wise winner."""

    model_id = _identifier("model_id", model_id)
    case_ids = tuple(sorted(_identifier("case_id", value) for value in calibration_case_ids))
    _case_hash(case_ids)
    indexed = _outcome_index(outcomes)
    _require_panel(
        indexed,
        model_id=model_id,
        case_ids=case_ids,
        action_ids=action_set.action_ids,
    )
    means = {
        action_id: sum(
            itd_policy.score(indexed[(model_id, case_id, action_id)])
            for case_id in case_ids
        )
        / len(case_ids)
        for action_id in action_set.action_ids
    }
    best_mean = max(means.values())
    tied = tuple(
        action_id
        for action_id in action_set.action_ids
        if best_mean - means[action_id] <= action_set.near_tie_tolerance
    )
    selected = min(
        tied,
        key=lambda action_id: (
            action_set.action(action_id).simplicity_rank,
            action_set.action(action_id).compute_rank,
            action_id,
        ),
    )
    return StaticPolicyLock(
        model_id=model_id,
        action_id=selected,
        action_set_id=action_set.action_set_id,
        calibration_case_hash=_case_hash(case_ids),
        itd_policy_id=itd_policy.policy_id,
        near_tied_actions=tuple(sorted(tied)),
    )


@dataclass(frozen=True, slots=True)
class FoldAssignment:
    case_id: str
    fold: int

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "fold": self.fold}


@dataclass(frozen=True, slots=True)
class ConfirmatoryFoldManifest:
    seed: str
    fold_count: int
    assignments: tuple[FoldAssignment, ...]

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(item.case_id for item in self.assignments)

    def fold_for(self, case_id: str) -> int:
        for assignment in self.assignments:
            if assignment.case_id == case_id:
                return assignment.fold
        raise ContractError(f"case is outside confirmatory fold manifest: {case_id}")

    @property
    def manifest_id(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict(include_id=False)).encode("utf-8"))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "ode-edit-mv1-confirmatory-folds/v1",
            "seed": self.seed,
            "fold_count": self.fold_count,
            "case_hash": _case_hash(self.case_ids),
            "assignments": [item.to_dict() for item in self.assignments],
        }
        if include_id:
            payload["manifest_id"] = self.manifest_id
        return payload


def build_confirmatory_fold_manifest(
    case_ids: Sequence[str],
    *,
    seed: str,
    fold_count: int = 5,
) -> ConfirmatoryFoldManifest:
    """Make balanced folds by salted case-ID hash, without any row outcome."""

    seed = _identifier("fold seed", seed)
    normalized = tuple(_identifier("case_id", value) for value in case_ids)
    _case_hash(normalized)
    if (
        isinstance(fold_count, bool)
        or not isinstance(fold_count, int)
        or fold_count < 2
        or fold_count > len(normalized)
    ):
        raise ContractError("fold_count must be an integer in [2, number of cases]")

    def rank_key(case_id: str) -> tuple[bytes, str]:
        return (
            hashlib.sha256(seed.encode() + b"\0" + case_id.encode()).digest(),
            case_id,
        )

    ranked = sorted(normalized, key=rank_key)
    folds = {case_id: index % fold_count for index, case_id in enumerate(ranked)}
    return ConfirmatoryFoldManifest(
        seed=seed,
        fold_count=fold_count,
        assignments=tuple(
            FoldAssignment(case_id=case_id, fold=folds[case_id])
            for case_id in sorted(normalized)
        ),
    )


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    model_id: str
    case_id: str
    fold: int
    action_id: str
    pre_step_feature_hash: str
    controller_lock_hash: str

    def __post_init__(self) -> None:
        for name in ("model_id", "case_id", "action_id"):
            object.__setattr__(self, name, _identifier(name, getattr(self, name)))
        if isinstance(self.fold, bool) or not isinstance(self.fold, int) or self.fold < 0:
            raise ContractError("fold must be a non-negative integer")
        for name in ("pre_step_feature_hash", "controller_lock_hash"):
            object.__setattr__(self, name, _full_hash(name, getattr(self, name)))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ControllerDecision":
        if not isinstance(raw, Mapping):
            raise ContractError("controller decision must be a mapping")
        allowed = {
            "model_id",
            "case_id",
            "fold",
            "action_id",
            "pre_step_feature_hash",
            "controller_lock_hash",
        }
        if set(raw) != allowed:
            raise ContractError(
                f"sanitized controller decision requires exactly {sorted(allowed)}"
            )
        return cls(**{key: raw[key] for key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "case_id": self.case_id,
            "fold": self.fold,
            "action_id": self.action_id,
            "pre_step_feature_hash": self.pre_step_feature_hash,
            "controller_lock_hash": self.controller_lock_hash,
        }


@dataclass(frozen=True, slots=True)
class ControllerDecisionManifest:
    model_id: str
    action_set_id: str
    fold_manifest_id: str
    controller_lock_hash: str
    decisions: tuple[ControllerDecision, ...]

    @property
    def manifest_id(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict(include_id=False)).encode("utf-8"))

    def decision_for(self, case_id: str) -> ControllerDecision:
        for decision in self.decisions:
            if decision.case_id == case_id:
                return decision
        raise ContractError(f"case is outside controller decision manifest: {case_id}")

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "ode-edit-mv1-controller-decisions/v1",
            "claim": "outcome-free precommitted actions",
            "model_id": self.model_id,
            "action_set_id": self.action_set_id,
            "fold_manifest_id": self.fold_manifest_id,
            "controller_lock_hash": self.controller_lock_hash,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }
        if include_id:
            payload["manifest_id"] = self.manifest_id
        return payload


def build_controller_decision_manifest(
    decisions: Sequence[ControllerDecision],
    *,
    model_id: str,
    action_set: LockedActionSet,
    folds: ConfirmatoryFoldManifest,
) -> ControllerDecisionManifest:
    """Validate a decision commitment before branch outcomes are supplied."""

    model_id = _identifier("model_id", model_id)
    if not decisions:
        raise ContractError("controller decisions must not be empty")
    indexed: dict[str, ControllerDecision] = {}
    lock_hashes: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, ControllerDecision):
            raise ContractError("decision manifest accepts ControllerDecision rows only")
        if decision.model_id != model_id:
            raise ContractError("controller decision model does not match manifest model")
        if decision.action_id not in action_set.action_ids:
            raise ContractError("controller decision uses an unlocked action")
        if decision.fold != folds.fold_for(decision.case_id):
            raise ContractError("controller decision fold does not match case-hash manifest")
        if decision.case_id in indexed:
            raise ContractError(f"duplicate controller decision: {decision.case_id}")
        indexed[decision.case_id] = decision
        lock_hashes.add(decision.controller_lock_hash)
    if set(indexed) != set(folds.case_ids):
        raise ContractError("controller decisions must cover every confirmatory case exactly")
    if len(lock_hashes) != 1:
        raise ContractError("controller decisions must use one precommitted controller lock")
    return ControllerDecisionManifest(
        model_id=model_id,
        action_set_id=action_set.action_set_id,
        fold_manifest_id=folds.manifest_id,
        controller_lock_hash=next(iter(lock_hashes)),
        decisions=tuple(indexed[case_id] for case_id in sorted(indexed)),
    )


@dataclass(frozen=True, slots=True)
class EffectEstimate:
    denominator: int
    mean: float
    ci_lower: float
    ci_upper: float
    median: float
    trimmed_mean: float
    positive_fraction: float
    zero_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "denominator": self.denominator,
            "mean": self.mean,
            "ci": [self.ci_lower, self.ci_upper],
            "median": self.median,
            "trimmed_mean": self.trimmed_mean,
            "positive_fraction": self.positive_fraction,
            "zero_fraction": self.zero_fraction,
        }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _effect_estimate(
    case_effects: Mapping[str, float],
    *,
    bootstrap_seed: str,
    bootstrap_replicates: int,
    confidence: float,
) -> EffectEstimate:
    if not case_effects:
        raise ContractError("paired effects must not be empty")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 100
    ):
        raise ContractError("bootstrap_replicates must be an integer >= 100")
    confidence = _finite("confidence", confidence)
    if not 0 < confidence < 1:
        raise ContractError("confidence must be in (0, 1)")
    case_ids = sorted(case_effects)
    effects = [_finite("paired effect", case_effects[case_id]) for case_id in case_ids]
    digest = hashlib.sha256(
        _identifier("bootstrap_seed", bootstrap_seed).encode()
        + b"\0"
        + canonical_json(case_ids).encode()
    ).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    n = len(effects)
    bootstrap = [
        sum(effects[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(bootstrap_replicates)
    ]
    alpha = (1 - confidence) / 2
    trim = math.floor(0.1 * n)
    ordered = sorted(effects)
    trimmed = ordered[trim : n - trim] if trim else ordered
    return EffectEstimate(
        denominator=n,
        mean=sum(effects) / n,
        ci_lower=_percentile(bootstrap, alpha),
        ci_upper=_percentile(bootstrap, 1 - alpha),
        median=float(median(effects)),
        trimmed_mean=sum(trimmed) / len(trimmed),
        positive_fraction=sum(value > 0 for value in effects) / n,
        zero_fraction=sum(value == 0 for value in effects) / n,
    )


@dataclass(frozen=True, slots=True)
class PairedModelSummary:
    model_id: str
    contrast: str
    raw: EffectEstimate
    near_tie_sensitivity: EffectEstimate
    candidate_status_counts: tuple[tuple[str, int], ...]
    comparator_status_counts: tuple[tuple[str, int], ...]
    fallback_scored_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ode-edit-mv1-paired-summary/v1",
            "model_id": self.model_id,
            "contrast": self.contrast,
            "raw": self.raw.to_dict(),
            "near_tie_sensitivity": self.near_tie_sensitivity.to_dict(),
            "denominator_flow": {
                "ITD_total": self.raw.denominator,
                "fallback_scored_rows": self.fallback_scored_rows,
                "candidate_status_counts": dict(self.candidate_status_counts),
                "comparator_status_counts": dict(self.comparator_status_counts),
            },
        }


def _paired_summary(
    *,
    model_id: str,
    contrast: str,
    case_ids: Sequence[str],
    candidate_actions: Mapping[str, str],
    comparator_action: str,
    indexed: Mapping[tuple[str, str, str], SanitizedOutcome],
    action_set: LockedActionSet,
    itd_policy: ITDFailurePolicy,
    bootstrap_seed: str,
    bootstrap_replicates: int,
    confidence: float,
) -> PairedModelSummary:
    candidate_status: Counter[str] = Counter()
    comparator_status: Counter[str] = Counter()
    raw: dict[str, float] = {}
    sensitivity: dict[str, float] = {}
    fallback_rows = 0
    for case_id in case_ids:
        candidate_action = candidate_actions[case_id]
        action_set.action(candidate_action)
        candidate = indexed.get((model_id, case_id, candidate_action))
        comparator = indexed.get((model_id, case_id, comparator_action))
        if candidate is None or comparator is None:
            raise ContractError(
                f"ITD paired row missing for model={model_id}, case={case_id}"
            )
        candidate_status[candidate.status.value] += 1
        comparator_status[comparator.status.value] += 1
        fallback_rows += int(candidate.progress is None) + int(comparator.progress is None)
        effect = itd_policy.score(candidate) - itd_policy.score(comparator)
        raw[case_id] = effect
        sensitivity[case_id] = (
            0.0 if abs(effect) <= action_set.near_tie_tolerance else effect
        )
    return PairedModelSummary(
        model_id=model_id,
        contrast=contrast,
        raw=_effect_estimate(
            raw,
            bootstrap_seed=f"{bootstrap_seed}:raw",
            bootstrap_replicates=bootstrap_replicates,
            confidence=confidence,
        ),
        near_tie_sensitivity=_effect_estimate(
            sensitivity,
            bootstrap_seed=f"{bootstrap_seed}:near-tie",
            bootstrap_replicates=bootstrap_replicates,
            confidence=confidence,
        ),
        candidate_status_counts=tuple(sorted(candidate_status.items())),
        comparator_status_counts=tuple(sorted(comparator_status.items())),
        fallback_scored_rows=fallback_rows,
    )


@dataclass(frozen=True, slots=True)
class OracleEmpiricalCeiling:
    """Retrospective candidate-set ceiling; never an achievable gain."""

    model_id: str
    candidate_count: int
    summary: PairedModelSummary
    arm_status_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    estimand_label: str = (
        "candidate-set empirical upper bound; retrospective, not achievable method gain"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ode-edit-mv1-oracle-ceiling/v1",
            "estimand_label": self.estimand_label,
            "model_id": self.model_id,
            "candidate_count": self.candidate_count,
            "arm_status_counts": {
                action_id: dict(counts) for action_id, counts in self.arm_status_counts
            },
            "summary": self.summary.to_dict(),
        }


def oracle_empirical_ceiling(
    outcomes: Iterable[SanitizedOutcome],
    *,
    model_id: str,
    case_ids: Sequence[str],
    action_set: LockedActionSet,
    static_policy: StaticPolicyLock,
    itd_policy: ITDFailurePolicy,
    bootstrap_seed: str,
    bootstrap_replicates: int = 2000,
    confidence: float = 0.95,
) -> OracleEmpiricalCeiling:
    """Compute event-wise oracle minus static as an empirical ceiling only."""

    model_id = _identifier("model_id", model_id)
    if (
        static_policy.model_id != model_id
        or static_policy.action_set_id != action_set.action_set_id
        or static_policy.itd_policy_id != itd_policy.policy_id
    ):
        raise ContractError("static policy lock does not match this analysis")
    normalized = tuple(sorted(_identifier("case_id", value) for value in case_ids))
    _case_hash(normalized)
    indexed = _outcome_index(outcomes)
    _require_panel(
        indexed,
        model_id=model_id,
        case_ids=normalized,
        action_ids=action_set.action_ids,
    )
    oracle_actions = {
        case_id: max(
            action_set.action_ids,
            key=lambda action_id: (
                itd_policy.score(indexed[(model_id, case_id, action_id)]),
                -action_set.action(action_id).simplicity_rank,
                -action_set.action(action_id).compute_rank,
                action_id,
            ),
        )
        for case_id in normalized
    }
    summary = _paired_summary(
        model_id=model_id,
        contrast="event-wise oracle - calibration-fixed best static",
        case_ids=normalized,
        candidate_actions=oracle_actions,
        comparator_action=static_policy.action_id,
        indexed=indexed,
        action_set=action_set,
        itd_policy=itd_policy,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence=confidence,
    )
    return OracleEmpiricalCeiling(
        model_id=model_id,
        candidate_count=len(action_set.actions),
        summary=summary,
        arm_status_counts=tuple(
            (
                action_id,
                tuple(
                    sorted(
                        Counter(
                            indexed[(model_id, case_id, action_id)].status.value
                            for case_id in normalized
                        ).items()
                    )
                ),
            )
            for action_id in action_set.action_ids
        ),
    )


def controller_effect_summary(
    outcomes: Iterable[SanitizedOutcome],
    *,
    decisions: ControllerDecisionManifest,
    expected_decision_manifest_id: str,
    action_set: LockedActionSet,
    static_policy: StaticPolicyLock,
    itd_policy: ITDFailurePolicy,
    bootstrap_seed: str,
    bootstrap_replicates: int = 2000,
    confidence: float = 0.95,
) -> PairedModelSummary:
    """Score a prior outcome-free decision commitment against fixed static."""

    if decisions.manifest_id != _full_hash(
        "expected_decision_manifest_id", expected_decision_manifest_id
    ):
        raise ContractError("controller decisions differ from the precommitted manifest")
    if (
        decisions.action_set_id != action_set.action_set_id
        or static_policy.action_set_id != action_set.action_set_id
        or decisions.model_id != static_policy.model_id
        or static_policy.itd_policy_id != itd_policy.policy_id
    ):
        raise ContractError("controller/static/action/ITD locks do not match")
    indexed = _outcome_index(outcomes)
    candidate_actions = {
        decision.case_id: decision.action_id for decision in decisions.decisions
    }
    return _paired_summary(
        model_id=decisions.model_id,
        contrast="outcome-free precommitted controller - calibration-fixed best static",
        case_ids=tuple(sorted(candidate_actions)),
        candidate_actions=candidate_actions,
        comparator_action=static_policy.action_id,
        indexed=indexed,
        action_set=action_set,
        itd_policy=itd_policy,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence=confidence,
    )
