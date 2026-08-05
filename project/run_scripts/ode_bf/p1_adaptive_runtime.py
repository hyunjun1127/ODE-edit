"""Adaptive-pseudo-time B10 causal diagnostic runtime.

All four live variants use the R_BF controller.  The only causal differences
are the legacy/adaptive clock, residual pre-sharing/full-residual policy, and
the declared maximum pseudo-time increment.  The module never commits a
scientific endpoint or appends history.
"""

from __future__ import annotations

import hashlib
import math
import random
import resource as system_resource
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .evaluator import ModelEvaluationReceipt
from .first_hit import FeasibilityVerdict
from .functional import CumulativeBF16FunctionalTrial, WaypointFactor, tensor_sha256
from .functional_p_secant import (
    FUNCTIONAL_P_BUDGET,
    H_REF as FUNCTIONAL_P_PROBE_H_REF,
    FunctionalPFieldPolicy,
    PSoftHardResult,
    build_functional_replay_secant,
    solve_p_soft_hard,
)
from .layer_routing_telemetry import (
    LAYER_IDS as ROUTING_LAYER_IDS,
    build_layer_routing_telemetry,
    relabel_layer_routing_telemetry,
    summarize_layer_routing_trajectory,
)
from .p0_runtime import ModelForwardCounter
from .p1_adaptive import (
    ADAPTIVE_INSTRUCTION_ID,
    ADAPTIVE_VARIANTS,
    H_REF,
    AdaptiveTauClock,
    AdaptiveVariant,
    FirstHitRecord,
    FirstHitTracker,
    StepSizeIndependentVelocity,
    adaptive_lock,
    adaptive_waypoint_factors,
    concentration_summary,
    fraction_payload,
    routing_change,
    streaming_bf16_capacity,
    streaming_bf16_endpoint_distance,
)
from .p1_backend import (
    CandidateBF16FunctionalTrial,
    FULL_CURRENT_RESIDUAL_DEFINITION,
    LEGACY_PRE_SHARED_RESIDUAL_DEFINITION,
    P1DynamicField,
    P1NativeCapture,
    PinnedCovarianceRegistry,
    TargetNewNLLReceipt,
    build_p1_dynamic_field,
    build_p1_frozen_field_from_capture,
    capture_p1_native_entry,
    evaluate_controller_margin,
    evaluate_routing_progress,
    signed_progress_gradient,
    write_aware_target_velocity,
)
from .p1_controller import (
    AcceptedLayerContribution,
    MatchedRawVelocity,
    P1ControllerLock,
    P1RoutingBuild,
    build_p1_routing_problem,
    project_matched_bf_velocity,
    solve_matched_raw_velocity,
)
from .p1_evaluator import load_counterfact_cases_after_freeze
from .p1_replay import (
    OuterEntryPretrainedCache,
    Theta0TeacherCache,
    build_outer_entry_pretrained_cache,
)
from .p1_state import (
    ArmWeightSnapshot,
    P1Arm,
    P1HistoryLedger,
    restore_arm_snapshot,
)
from .p1_stepwise import (
    StepwiseActionFreeze,
    StepwisePrimaryReceipt,
    compare_stepwise_primary,
    evaluate_counterfact_stepwise_primary,
)
from .routing import (
    PreservationConstraintPolicy,
    RoutingProblem,
    RoutingStatus,
    SolverCertificate,
    verify_backtracked_candidate,
)
from .sampling import StatelessReplaySchedule
from .target_new_nll import RoutingObjective, select_locked_routing_objective


ADAPTIVE_RESULT_TOKEN = "p1r4-adaptive-tau-causal-r2-v1"


def expected_adaptive_result_name(alias: str) -> str:
    return f"s04-p1r4-adaptive-tau-r2-{alias}-v1"


class FunctionalPDecisionPolicy(str, Enum):
    """Whether the observed functional-P bit participates in decisions."""

    PCTRL = "PCTRL"
    OBSERVATION_ONLY = "FPOFF"


@dataclass(frozen=True, slots=True)
class FunctionalPDecisionReceipt:
    policy: FunctionalPDecisionPolicy
    observation_sha256: str
    observed_pass_at_locked_budget: bool
    observed_mean_positive_damage: float
    observed_slack_at_locked_budget: float
    decision_pass: bool
    decision_influence_count: int

    def __post_init__(self) -> None:
        if len(self.observation_sha256) != 64:
            raise ODEBFContractError("functional-P observation identity differs")
        if (
            not math.isfinite(self.observed_mean_positive_damage)
            or not math.isfinite(self.observed_slack_at_locked_budget)
            or not math.isclose(
                self.observed_slack_at_locked_budget,
                1.0e-3 - self.observed_mean_positive_damage,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise ODEBFContractError("functional-P observed damage differs")
        expected_pass = (
            self.observed_pass_at_locked_budget
            if self.policy is FunctionalPDecisionPolicy.PCTRL
            else True
        )
        expected_count = (
            1 if self.policy is FunctionalPDecisionPolicy.PCTRL else 0
        )
        if (
            self.decision_pass is not expected_pass
            or self.decision_influence_count != expected_count
        ):
            raise ODEBFContractError("functional-P decision receipt differs")

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "observation_sha256": self.observation_sha256,
            "observed_pass_at_locked_budget": self.observed_pass_at_locked_budget,
            "observed_mean_positive_damage": self.observed_mean_positive_damage,
            "observed_slack_at_locked_budget": self.observed_slack_at_locked_budget,
            "decision_pass": self.decision_pass,
            "decision_influence_count": self.decision_influence_count,
            "observation_only": (
                self.policy is FunctionalPDecisionPolicy.OBSERVATION_ONLY
            ),
        }


def functional_p_decision_receipt(
    policy: FunctionalPDecisionPolicy | str,
    observation: Mapping[str, Any],
) -> FunctionalPDecisionReceipt:
    """Apply the locked PCTRL/FPOFF decision bit without changing observation."""

    try:
        selected = FunctionalPDecisionPolicy(policy)
    except (TypeError, ValueError) as exc:
        raise ODEBFContractError("functional-P decision policy differs") from exc
    observed = observation.get("passed")
    if not isinstance(observed, bool):
        raise ODEBFContractError("functional-P observed pass bit differs")
    budget = observation.get("budget")
    if isinstance(budget, bool) or not isinstance(budget, (int, float)):
        raise ODEBFContractError("functional-P observation budget differs")
    if not math.isfinite(float(budget)) or float(budget) != 1.0e-3:
        raise ODEBFContractError("functional-P observation budget differs")
    damage = observation.get("mean_positive_damage")
    if isinstance(damage, bool) or not isinstance(damage, (int, float)):
        raise ODEBFContractError("functional-P observed damage differs")
    observed_damage = float(damage)
    if not math.isfinite(observed_damage):
        raise ODEBFContractError("functional-P observed damage differs")
    observation_sha256 = canonical_hash(dict(observation))
    decision_pass = (
        observed if selected is FunctionalPDecisionPolicy.PCTRL else True
    )
    return FunctionalPDecisionReceipt(
        selected,
        observation_sha256,
        observed,
        observed_damage,
        float(budget) - observed_damage,
        decision_pass,
        1 if selected is FunctionalPDecisionPolicy.PCTRL else 0,
    )


def _preservation_decision_payload_from_observed(
    policy: PreservationConstraintPolicy | str,
    observed: Mapping[str, bool],
    functional_p: FunctionalPDecisionReceipt,
) -> dict[str, Any]:
    try:
        selected = PreservationConstraintPolicy(policy)
    except (TypeError, ValueError) as exc:
        raise ODEBFContractError("preservation decision policy differs") from exc
    expected_names = (
        "structural_h", "structural_p", "trust", "functional_h",
    )
    if tuple(observed) != expected_names or any(
        not isinstance(observed[name], bool) for name in expected_names
    ):
        raise ODEBFContractError("preservation observation payload differs")
    observed_payload = {
        **dict(observed),
        "functional_p": bool(functional_p.observed_pass_at_locked_budget),
    }
    if selected is PreservationConstraintPolicy.OBSERVATION_ONLY:
        decision = {name: True for name in observed_payload}
        influence = {name: 0 for name in observed_payload}
    else:
        decision = {
            **{
                name: observed_payload[name]
                for name in observed_payload
                if name != "functional_p"
            },
            "functional_p": functional_p.decision_pass,
        }
        influence = {
            **{
                name: 1
                for name in observed_payload
                if name != "functional_p"
            },
            "functional_p": functional_p.decision_influence_count,
        }
    payload = {
        "policy": selected.value,
        "observed_pass": observed_payload,
        "decision_pass": decision,
        "decision_influence_count": influence,
        "all_decision_pass": all(decision.values()),
        "observation_only": (
            selected is PreservationConstraintPolicy.OBSERVATION_ONLY
        ),
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def preservation_decision_payload(
    policy: PreservationConstraintPolicy | str,
    verdict: Any,
    functional_p: FunctionalPDecisionReceipt,
) -> dict[str, Any]:
    """Separate observed preservation facts from their decision influence."""

    return _preservation_decision_payload_from_observed(
        policy,
        {
            "structural_h": bool(verdict.structural_h_pass),
            "structural_p": bool(verdict.structural_p_pass),
            "trust": bool(verdict.trust_pass),
            "functional_h": bool(verdict.functional_h_pass),
        },
        functional_p,
    )


def candidate_gate_decision(
    verdict: Any,
    preservation: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Keep technical progress/BF16 gates active around preservation ablations."""

    decision = preservation.get("decision_pass")
    if not isinstance(decision, Mapping) or tuple(decision) != (
        "structural_h", "structural_p", "trust", "functional_h", "functional_p",
    ):
        raise ODEBFContractError("preservation decision gate payload differs")
    if not verdict.progress_pass:
        return False, "progress"
    for name in decision:
        if decision[name] is not True:
            return False, name
    if not verdict.authoritative_bf16_pass:
        return False, "authoritative_bf16"
    return True, None


@dataclass(frozen=True, slots=True)
class AdaptivePanelSpec:
    label: str
    clock_variant: AdaptiveVariant
    routing_objective: RoutingObjective
    functional_p_decision: FunctionalPDecisionPolicy = (
        FunctionalPDecisionPolicy.PCTRL
    )
    preservation_constraints: PreservationConstraintPolicy = (
        PreservationConstraintPolicy.LOCKED
    )
    functional_p_field_policy: FunctionalPFieldPolicy = (
        FunctionalPFieldPolicy.NONE
    )

    def __post_init__(self) -> None:
        if not self.label or "/" in self.label or self.label in (".", ".."):
            raise ODEBFContractError("adaptive panel label differs")
        select_locked_routing_objective(self.routing_objective)
        try:
            selected_functional_p = FunctionalPDecisionPolicy(
                self.functional_p_decision
            )
        except (TypeError, ValueError) as exc:
            raise ODEBFContractError(
                "adaptive panel functional-P policy differs"
            ) from exc
        object.__setattr__(self, "functional_p_decision", selected_functional_p)
        try:
            selected_preservation = PreservationConstraintPolicy(
                self.preservation_constraints
            )
        except (TypeError, ValueError) as exc:
            raise ODEBFContractError(
                "adaptive panel preservation policy differs"
            ) from exc
        object.__setattr__(
            self, "preservation_constraints", selected_preservation
        )
        try:
            selected_field_policy = FunctionalPFieldPolicy(
                self.functional_p_field_policy
            )
        except (TypeError, ValueError) as exc:
            raise ODEBFContractError(
                "adaptive panel functional-P field policy differs"
            ) from exc
        object.__setattr__(
            self, "functional_p_field_policy", selected_field_policy
        )

    @property
    def legacy_key(self) -> AdaptiveVariant | str:
        return (
            self.clock_variant
            if self.label == self.clock_variant.value
            and self.routing_objective is RoutingObjective.MARGIN
            and self.functional_p_decision is FunctionalPDecisionPolicy.PCTRL
            and self.preservation_constraints
            is PreservationConstraintPolicy.LOCKED
            and self.functional_p_field_policy is FunctionalPFieldPolicy.NONE
            else self.label
        )


def _default_panel_specs() -> tuple[AdaptivePanelSpec, ...]:
    return tuple(
        AdaptivePanelSpec(
            item.value,
            item,
            RoutingObjective.MARGIN,
        )
        for item in ADAPTIVE_VARIANTS
    )


def _factor_map(
    values: Mapping[str, Sequence[WaypointFactor]],
) -> dict[str, tuple[WaypointFactor, ...]]:
    return {name: tuple(items) for name, items in values.items()}


def _merge_factors(
    base: Mapping[str, Sequence[WaypointFactor]],
    increment: Mapping[str, WaypointFactor],
) -> dict[str, tuple[WaypointFactor, ...]]:
    result = _factor_map(base)
    for name, factor in increment.items():
        values = result.get(name, ()) + (factor,)
        if len({item.order_key for item in values}) != len(values):
            raise ODEBFContractError("adaptive cumulative factor order repeats")
        result[name] = tuple(sorted(values, key=lambda item: item.order_key))
    return result


def _factor_state(
    entry_sha256: Mapping[str, str],
    factors: Mapping[str, Sequence[WaypointFactor]],
    target_state: torch.Tensor,
) -> str:
    payload: dict[str, Any] = {
        "entry": dict(sorted(entry_sha256.items())),
        "target_state_sha256": tensor_sha256(target_state),
        "factors": {},
    }
    for name, values in sorted(factors.items()):
        payload["factors"][name] = [
            {
                "layer": item.layer,
                "cycle": item.correction_cycle,
                "step": item.step_in_cycle,
                "ordinal": item.factor_ordinal,
                "theta": item.theta,
                "left": tensor_sha256(item.left),
                "right": tensor_sha256(item.right),
                "rank": item.left.shape[1],
            }
            for item in sorted(values, key=lambda value: value.order_key)
        ]
    return canonical_hash(payload)


def _omega_state(
    accepted_by_layer: Mapping[int, Sequence[AcceptedLayerContribution]],
) -> str:
    return canonical_hash(
        {
            str(layer): [
                {
                    "factor": {
                        "order_key": list(item.factor.order_key),
                        "theta": item.factor.theta,
                        "left_sha256": tensor_sha256(item.factor.left),
                        "right_sha256": tensor_sha256(item.factor.right),
                    },
                    "q_sha256": tensor_sha256(item.q),
                    "residual_sha256": tensor_sha256(item.residual),
                    "covariance_action_sha256": tensor_sha256(
                        item.covariance_action
                    ),
                    "history_action_sha256": tensor_sha256(item.history_action),
                    "factor_frobenius_sq": item.factor_frobenius_sq,
                }
                for item in values
            ]
            for layer, values in sorted(accepted_by_layer.items())
        }
    )


def _rng_identity() -> str:
    cuda = (
        [tensor_sha256(value) for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    numpy_state = np.random.get_state()
    return canonical_hash(
        {
            "python": hashlib.sha256(repr(random.getstate()).encode("utf-8")).hexdigest(),
            "numpy": {
                "kind": str(numpy_state[0]),
                "keys_sha256": hashlib.sha256(
                    np.asarray(numpy_state[1], dtype=np.uint32).tobytes()
                ).hexdigest(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch_cpu": tensor_sha256(torch.get_rng_state()),
            "torch_cuda": cuda,
        }
    )


def _parameter_contract(parameters: Mapping[str, torch.nn.Parameter]) -> dict[str, Any]:
    payload = {
        "weights": {
            name: {
                "bytes_sha256": tensor_sha256(parameter),
                "pointer": int(parameter.data_ptr()),
                "version": int(parameter._version),
                "requires_grad": bool(parameter.requires_grad),
                "grad_sha256": (
                    None if parameter.grad is None else tensor_sha256(parameter.grad)
                ),
            }
            for name, parameter in sorted(parameters.items())
        },
        "rng_sha256": _rng_identity(),
    }
    return payload


def _parameter_contract_sha256(parameters: Mapping[str, torch.nn.Parameter]) -> str:
    return canonical_hash(_parameter_contract(parameters))


class AdaptiveReceiptRecorder:
    def __init__(
        self,
        root: Path,
        variant: AdaptiveVariant,
        write_once: Any,
        *,
        variant_label: str | None = None,
        instruction_id: str = ADAPTIVE_INSTRUCTION_ID,
        receipt_schema: str = "ode-edit-s04-ode-bf-p1r4-adaptive-receipt/v1",
    ) -> None:
        label = variant.value if variant_label is None else variant_label
        if not label or "/" in label or label in (".", ".."):
            raise ODEBFContractError("adaptive receipt variant label differs")
        self.root = root / "adaptive" / label
        self.variant = variant
        self.variant_label = label
        self.instruction_id = instruction_id
        self.receipt_schema = receipt_schema
        self.write_once = write_once
        self.field_hashes: list[str] = []
        self.solver_hashes: list[str] = []
        self.trial_hashes: list[str] = []
        self.transition_hashes: list[str] = []
        self.accepted_hashes: list[str] = []
        self.first_hit_hashes: list[str] = []
        self.terminal_hashes: list[str] = []
        self.observation_failures: list[dict[str, str]] = []

    def _write(self, category: str, ordinal: int, payload: Mapping[str, Any]) -> str:
        path = self.root / f"{category}-{ordinal:04d}.json"
        envelope = {
            "schema": self.receipt_schema,
            "instruction_id": self.instruction_id,
            "variant": self.variant_label,
            "category": category,
            "ordinal": ordinal,
            **dict(payload),
        }
        if (
            self.variant_label != self.variant.value
            or self.instruction_id != ADAPTIVE_INSTRUCTION_ID
        ):
            envelope["clock_variant"] = self.variant.value
        return self.write_once(
            path,
            envelope,
        )

    def field(self, payload: Mapping[str, Any]) -> str:
        digest = self._write("field", len(self.field_hashes), payload)
        self.field_hashes.append(digest)
        return digest

    def solver(self, payload: Mapping[str, Any]) -> str:
        digest = self._write("solver", len(self.solver_hashes), payload)
        self.solver_hashes.append(digest)
        return digest

    def trial(self, payload: Mapping[str, Any]) -> str:
        digest = self._write("trial", len(self.trial_hashes), payload)
        self.trial_hashes.append(digest)
        return digest

    def transition(self, payload: Mapping[str, Any]) -> str:
        digest = self._write(
            "transition", len(self.transition_hashes), payload
        )
        self.transition_hashes.append(digest)
        return digest

    def accepted(self, payload: Mapping[str, Any]) -> str:
        digest = self._write("accepted", len(self.accepted_hashes), payload)
        self.accepted_hashes.append(digest)
        return digest

    def first_hit(self, payload: Mapping[str, Any]) -> str:
        digest = self._write(
            "first-hit", len(self.first_hit_hashes), payload
        )
        self.first_hit_hashes.append(digest)
        return digest

    def terminal(self, payload: Mapping[str, Any]) -> str:
        digest = self._write("terminal", len(self.terminal_hashes), payload)
        self.terminal_hashes.append(digest)
        return digest

    def links(self) -> dict[str, list[str]]:
        return {
            "field": list(self.field_hashes),
            "solver": list(self.solver_hashes),
            "trial": list(self.trial_hashes),
            "transition": list(self.transition_hashes),
            "accepted": list(self.accepted_hashes),
            "first_hit": list(self.first_hit_hashes),
            "terminal": list(self.terminal_hashes),
        }

    def note_observation_failure(self, category: str, exc: BaseException) -> None:
        self.observation_failures.append(
            {
                "category": category,
                "exception_class": type(exc).__name__,
                "exception_message_sha256": hashlib.sha256(
                    str(exc).encode("utf-8")
                ).hexdigest(),
            }
        )

    def assert_observation_complete(self) -> None:
        if self.observation_failures:
            raise ODEBFContractError(
                "adaptive observation-only receipt persistence failed"
            )


def _solver_observability_payload(
    certificate: SolverCertificate,
    *,
    problem: RoutingProblem,
    accepted_index: int,
    solve_role: str,
    routing_objective: RoutingObjective,
) -> dict[str, Any]:
    """Build an append-safe receipt before a solver certificate can raise."""

    primal_tolerance = 1.0e-8
    kkt_tolerance = 1.0e-5
    numeric = (
        certificate.maximum_primal_violation,
        certificate.stationarity_residual,
        certificate.complementarity_residual,
        certificate.signed_progress,
        certificate.trust_value,
    )
    finite_flag = all(math.isfinite(float(value)) for value in numeric)
    if not finite_flag:
        first_false = "finite"
    elif not certificate.success:
        first_false = "solver_success"
    elif certificate.maximum_primal_violation > primal_tolerance:
        first_false = "maximum_primal_violation"
    elif certificate.stationarity_residual > kkt_tolerance:
        first_false = "stationarity_residual"
    elif certificate.complementarity_residual > kkt_tolerance:
        first_false = "complementarity_residual"
    else:
        first_false = None
    capacity_eigenvalues = np.linalg.eigvalsh(problem.capacity_metric)
    trust_eigenvalues = np.linalg.eigvalsh(problem.trust_metric)
    capacity_min = float(capacity_eigenvalues.min())
    capacity_max = float(capacity_eigenvalues.max())
    trust_min = float(trust_eigenvalues.min())
    trust_max = float(trust_eigenvalues.max())

    def condition(maximum: float, minimum: float) -> float | None:
        value = maximum / minimum if minimum > 0.0 else math.inf
        return float(value) if math.isfinite(value) else None

    return {
        "status": (
            "SOLVER_CERTIFICATE_PASSED"
            if certificate.passed
            else "SOLVER_CERTIFICATE_FAILED"
        ),
        "accepted_index": accepted_index,
        "solve_role": solve_role,
        "routing_objective": routing_objective.value,
        "problem_sha256": problem.identity(),
        "certificate": asdict(certificate),
        "thresholds": {
            "primal": primal_tolerance,
            "stationarity": kkt_tolerance,
            "complementarity": kkt_tolerance,
        },
        "finite": finite_flag,
        "first_false_component": first_false,
        "conditioning": {
            "dimension": int(problem.signed_progress.size),
            "positive_direction_count": int(
                np.count_nonzero(problem.positive_direction_mask)
            ),
            "capacity_min_eigenvalue": capacity_min,
            "capacity_max_eigenvalue": capacity_max,
            "capacity_condition_number": condition(capacity_max, capacity_min),
            "trust_min_eigenvalue": trust_min,
            "trust_max_eigenvalue": trust_max,
            "trust_condition_number": condition(trust_max, trust_min),
        },
        "constraints": {
            "requested_progress": problem.requested_progress,
            "minimum_progress": problem.minimum_progress,
            "trust_radius": problem.trust_radius,
            "layer_cap_min": float(problem.layer_caps.min()),
            "layer_cap_max": float(problem.layer_caps.max()),
            "historical_entry_slack": (
                problem.historical.budget - problem.historical.offset
            ),
            "pretrained_entry_slack": (
                problem.pretrained.budget - problem.pretrained.offset
            ),
        },
    }


def _solver_certificate_observer(
    *,
    recorder: AdaptiveReceiptRecorder,
    problem: RoutingProblem,
    accepted_index: int,
    solve_role: str,
    routing_objective: RoutingObjective,
) -> Any:
    """Return a failure-isolated observer called before certificate raises."""

    def observe(certificate: SolverCertificate) -> None:
        try:
            recorder.solver(
                _solver_observability_payload(
                    certificate,
                    problem=problem,
                    accepted_index=accepted_index,
                    solve_role=solve_role,
                    routing_objective=routing_objective,
                )
            )
        except BaseException as exc:
            recorder.note_observation_failure("solver", exc)

    return observe


def _field_layer_routing_payload(
    *,
    field: P1DynamicField,
    signed_progress: Sequence[float],
    problem: RoutingProblem,
    raw_velocity: Sequence[float],
    bf_velocity: Sequence[float],
    raw_certificate: SolverCertificate,
    bf_certificate: SolverCertificate,
    state_sha256: str,
    target_z_sha256: str,
    factor_state_sha256: str,
    step_index: int,
) -> dict[str, Any]:
    layers = tuple(int(item.layer) for item in field.layers)
    if layers != ROUTING_LAYER_IDS:
        raise ODEBFContractError("adaptive field telemetry layer order differs")
    signed = np.asarray(signed_progress, dtype=np.float64)
    raw = np.asarray(raw_velocity, dtype=np.float64)
    bf = np.asarray(bf_velocity, dtype=np.float64)
    caps = np.asarray(problem.layer_caps, dtype=np.float64)
    zero = np.zeros(len(layers), dtype=np.float64)
    payload = build_layer_routing_telemetry(
        step_index=step_index,
        stage="FIELD",
        layer_ids=layers,
        signed_efficiency=signed,
        raw_velocity=raw,
        bf_velocity=bf,
        applied_coefficient=zero,
        active_direction_mask=signed > 0.0,
        raw_cap_bound_mask=np.isclose(raw, caps, rtol=0.0, atol=1.0e-12),
        bf_cap_bound_mask=np.isclose(bf, caps, rtol=0.0, atol=1.0e-12),
        raw_zero_bound_mask=np.isclose(raw, 0.0, rtol=0.0, atol=1.0e-12),
        bf_zero_bound_mask=np.isclose(bf, 0.0, rtol=0.0, atol=1.0e-12),
        predicted_progress_contribution=signed * bf,
        prequantized_update_energy=zero,
        realized_bf16_update_energy=zero,
        cumulative_bf16_capacity=zero,
        bf16_capacity_contribution=zero,
        structural_h_contribution=zero,
        structural_p_contribution=zero,
        trust_contribution=zero,
        field_sha256=field.identity_sha256,
        state_sha256=state_sha256,
        target_z_sha256=target_z_sha256,
        factor_state_sha256=factor_state_sha256,
        raw_solver_certificate_sha256=canonical_hash(asdict(raw_certificate)),
        bf_solver_certificate_sha256=canonical_hash(asdict(bf_certificate)),
        candidate_sha256=None,
    )
    payload["trial_dependent_vectors_defined"] = False
    payload["identity_sha256"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "identity_sha256"}
    )
    return payload


@dataclass(slots=True)
class ActiveField:
    field: P1DynamicField
    signed_progress: Any
    routing: P1RoutingBuild
    problem: RoutingProblem
    raw: MatchedRawVelocity
    raw_velocity: StepSizeIndependentVelocity
    bf_velocity: StepSizeIndependentVelocity
    projection: Any
    target_velocity: torch.Tensor
    target_velocity_receipt: Any
    receipt_sha256: str
    layer_routing_payload: dict[str, Any]
    functional_p_field_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrialOutcome:
    gate_accepted: bool
    candidate_factors: dict[str, tuple[WaypointFactor, ...]]
    increment: dict[str, WaypointFactor]
    target_trial: torch.Tensor
    evaluation: ModelEvaluationReceipt
    margin: Any
    feasibility: FeasibilityVerdict
    functional: Any
    structural_payload: dict[str, Any]
    routing_payload: dict[str, Any]
    progress_payload: dict[str, Any]
    capacity_payload: dict[str, Any]
    snapshot_sha256: str
    receipt_sha256: str
    trust_ratio: float | None
    functional_p_decision: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AcceptedSnapshot:
    accepted_index: int
    tau: Fraction
    delta_tau: Fraction
    factors: dict[str, tuple[WaypointFactor, ...]]
    target_state: torch.Tensor
    snapshot_sha256: str
    evaluation: ModelEvaluationReceipt
    feasibility: FeasibilityVerdict
    structural_payload: dict[str, Any]
    routing_payload: dict[str, Any]
    progress_payload: dict[str, Any]
    capacity_payload: dict[str, Any]
    field_sha256: str
    raw_velocity_sha256: str
    bf_velocity_sha256: str
    accepted_receipt_sha256: str
    functional_p_decision: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VariantRollout:
    variant: AdaptiveVariant
    status: str
    termination_label: str
    residual_policy: str
    accepted_t: Fraction
    k_acc: int
    n_trial: int
    n_reject: int
    field_build_count: int
    snapshots: list[AcceptedSnapshot]
    first_hit: FirstHitTracker
    recorder: AdaptiveReceiptRecorder
    ledger: ComputeLedger
    entry_success: dict[str, Any]
    terminal_confirmation: list[dict[str, Any]]
    rollout_sha256: str
    variant_label: str = ""
    routing_objective: str = RoutingObjective.MARGIN.value
    functional_p_decision: str = FunctionalPDecisionPolicy.PCTRL.value
    preservation_constraints: str = PreservationConstraintPolicy.LOCKED.value
    functional_p_field_policy: str = FunctionalPFieldPolicy.NONE.value


def _risk_payload(receipt: Any) -> dict[str, Any]:
    return {
        "barrier": receipt.barrier,
        "sample_count": receipt.item_count,
        "sample_order_sha256": receipt.sample_order_sha256,
        "budget": receipt.budget,
        "mean_positive_damage": receipt.mean_positive_damage,
        "smooth_max_positive_damage": receipt.smooth_max_positive_damage,
        "raw_max_positive_damage": receipt.raw_max_positive_damage,
        "signed_mean_damage": receipt.signed_mean_damage,
        "decision_rule": receipt.decision_rule,
        "passed": receipt.passed,
    }


def _structural_payload(barrier: Any, velocity: np.ndarray) -> dict[str, Any]:
    value = float(barrier.value(velocity))
    return {
        "value": value,
        "budget": float(barrier.budget),
        "slack": float(barrier.budget - value),
        "passed": bool(value <= barrier.budget + 1.0e-8),
        "approximation": barrier.approximation,
    }


def _trust_payload(problem: RoutingProblem, velocity: np.ndarray) -> dict[str, Any]:
    value = float(velocity @ problem.trust_metric @ velocity)
    radius_squared = float(problem.trust_radius**2)
    return {
        "value": value,
        "radius": float(problem.trust_radius),
        "radius_squared": radius_squared,
        "slack": radius_squared - value,
        "passed": bool(value <= radius_squared + 1.0e-8),
        "role": "redundant-diagnostic",
    }


def _history_keys(history: P1HistoryLedger, layers: Sequence[int], *, risk: bool) -> dict[int, torch.Tensor]:
    ordered_layers = tuple(int(layer) for layer in layers)
    if ordered_layers != history.layer_order:
        raise ODEBFContractError("adaptive history layer order differs")
    view = history.risk_keys if risk else history.solve_keys
    return {layer: view(layer) for layer in ordered_layers}


def _history_actions(field: P1DynamicField, history: P1HistoryLedger) -> dict[int, torch.Tensor]:
    if tuple(item.layer for item in field.layers) != history.layer_order:
        raise ODEBFContractError("adaptive history field layer order differs")
    result: dict[int, torch.Tensor] = {}
    for item in field.layers:
        keys = history.risk_keys(item.layer)
        result[item.layer] = (
            item.residual.to(dtype=torch.float64)
            @ (item.q.to(dtype=torch.float64).T @ keys.to(dtype=torch.float64))
            if keys.shape[1]
            else torch.empty((item.residual.shape[0], 0), dtype=torch.float64)
        )
    return result


def _residual_policy(variant: AdaptiveVariant) -> str:
    return (
        LEGACY_PRE_SHARED_RESIDUAL_DEFINITION
        if variant in (AdaptiveVariant.PS_S8, AdaptiveVariant.PS_A8)
        else FULL_CURRENT_RESIDUAL_DEFINITION
    )


def _build_active_field(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    variant: AdaptiveVariant,
    accepted_index: int,
    factors: Mapping[str, Sequence[WaypointFactor]],
    target_state: torch.Tensor,
    capture: P1NativeCapture,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    lock: P1ControllerLock,
    history: P1HistoryLedger,
    accepted_by_layer: Mapping[int, Sequence[AcceptedLayerContribution]],
    ledger: ComputeLedger,
    recorder: AdaptiveReceiptRecorder,
    native_target: torch.Tensor,
    target_base: torch.Tensor,
    routing_objective: RoutingObjective | str = RoutingObjective.MARGIN,
    preservation_policy: PreservationConstraintPolicy | str = (
        PreservationConstraintPolicy.LOCKED
    ),
    functional_p_field_policy: FunctionalPFieldPolicy | str = (
        FunctionalPFieldPolicy.NONE
    ),
    functional_p_replay_entry: Any | None = None,
    theta0_cache: Theta0TeacherCache | None = None,
    alias: str | None = None,
    touched: Mapping[str, torch.nn.Parameter] | None = None,
    schedule: StatelessReplaySchedule | None = None,
) -> ActiveField:
    selected_objective = select_locked_routing_objective(routing_objective)
    try:
        selected_preservation = PreservationConstraintPolicy(
            preservation_policy
        )
    except (TypeError, ValueError) as exc:
        raise ODEBFContractError("adaptive preservation policy differs") from exc
    try:
        selected_functional_p_field = FunctionalPFieldPolicy(
            functional_p_field_policy
        )
    except (TypeError, ValueError) as exc:
        raise ODEBFContractError(
            "adaptive functional-P field policy differs"
        ) from exc
    layers = tuple(int(layer) for layer in hparams.layers)
    policy = _residual_policy(variant)
    started = time.perf_counter()
    counter_before = dict(ledger.counters)
    if accepted_index == 0:
        field = build_p1_frozen_field_from_capture(
            model,
            hparams,
            projector,
            capture,
            target_state=target_state,
            accepted_waypoint=0,
            history_solve_keys_by_layer=_history_keys(history, layers, risk=False),
            history_risk_keys_by_layer=_history_keys(history, layers, risk=True),
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            residual_tolerance=lock.residual_tolerance,
            residual_policy=policy,
        )
    else:
        field = build_p1_dynamic_field(
            model,
            tokenizer,
            requests,
            hparams,
            projector,
            contexts,
            target_state=target_state,
            accepted_waypoint=accepted_index,
            cumulative_factors_by_weight=factors,
            history_solve_keys_by_layer=_history_keys(history, layers, risk=False),
            history_risk_keys_by_layer=_history_keys(history, layers, risk=True),
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            residual_tolerance=lock.residual_tolerance,
            ledger=ledger,
            residual_policy=policy,
        )
    signed = signed_progress_gradient(
        model,
        tokenizer,
        requests,
        field,
        cumulative_factors_by_weight=factors,
        ledger=ledger,
        objective=selected_objective,
        contexts=contexts if selected_objective is RoutingObjective.TARGET_NEW_NLL else None,
    )
    source = build_p1_routing_problem(
        field,
        signed,
        accepted_by_layer=accepted_by_layer,
        committed_load_by_layer=history.cumulative_load(),
        lock=lock,
    )
    raw_observer = _solver_certificate_observer(
        recorder=recorder,
        problem=source.problem,
        accepted_index=accepted_index,
        solve_role="RAW_ROUTING",
        routing_objective=selected_objective,
    )
    raw = solve_matched_raw_velocity(
        source,
        preservation_policy=selected_preservation,
        certificate_observer=raw_observer,
    )
    ledger.increment("qp_solve", 2)
    ledger.increment("qp_certificate", 2)
    barrier = build_p1_routing_problem(
        field,
        signed,
        accepted_by_layer=accepted_by_layer,
        committed_load_by_layer=history.cumulative_load(),
        lock=lock,
        current_history_action_by_layer=_history_actions(field, history),
    )
    bf_observer = _solver_certificate_observer(
        recorder=recorder,
        problem=barrier.problem,
        accepted_index=accepted_index,
        solve_role="BF_PROJECTION",
        routing_objective=selected_objective,
    )
    projection = project_matched_bf_velocity(
        barrier,
        raw,
        preservation_policy=selected_preservation,
        certificate_observer=bf_observer,
    )
    ledger.increment("qp_solve", 2)
    ledger.increment("qp_certificate", 2)
    if projection.status is not RoutingStatus.FEASIBLE or projection.values is None:
        field_state_sha256 = _factor_state(
            capture.entry_sha256,
            factors,
            target_state,
        )
        failure_payload = {
            "accepted_index": accepted_index,
            "status": "PROGRESS_INFEASIBLE",
            "residual_policy": policy,
            "field": field.raw_free_payload(),
            "field_sha256": field.identity_sha256,
            "signed_slopes": list(signed.signed_progress),
            "signed_slopes_sha256": canonical_hash(list(signed.signed_progress)),
            "raw_velocity": raw.values.tolist(),
            "raw_velocity_sha256": raw.velocity_sha256,
            "raw_certificate": asdict(raw.certificate),
            "projection_status": projection.status.value,
            "projection_certificate": asdict(projection.certificate),
            "preservation_constraint_policy": selected_preservation.value,
            "preservation_solver_decision_influence_count": {
                "structural_h": (
                    1
                    if selected_preservation is PreservationConstraintPolicy.LOCKED
                    else 0
                ),
                "structural_p": (
                    1
                    if selected_preservation is PreservationConstraintPolicy.LOCKED
                    else 0
                ),
                "trust": (
                    1
                    if selected_preservation is PreservationConstraintPolicy.LOCKED
                    else 0
                ),
            },
            "field_build_wall_seconds": time.perf_counter() - started,
            "field_build_counter_delta": {
                name: ledger.counters[name] - counter_before[name]
                for name in sorted(ledger.counters)
            },
        }
        failure_payload["layer_routing"] = _field_layer_routing_payload(
            field=field,
            signed_progress=signed.signed_progress,
            problem=barrier.problem,
            raw_velocity=raw.values,
            bf_velocity=np.zeros(len(field.layers), dtype=np.float64),
            raw_certificate=raw.certificate,
            bf_certificate=projection.certificate,
            state_sha256=field_state_sha256,
            target_z_sha256=canonical_hash(list(capture.direct_z_sha256)),
            factor_state_sha256=field_state_sha256,
            step_index=accepted_index,
        )
        failure_payload["layer_routing"]["bf_velocity_defined"] = False
        failure_payload["layer_routing"]["identity_sha256"] = canonical_hash(
            {
                key: value
                for key, value in failure_payload["layer_routing"].items()
                if key != "identity_sha256"
            }
        )
        if (
            selected_objective is not RoutingObjective.MARGIN
            or recorder.variant_label != variant.value
        ):
            failure_payload.update(
                {
                    "routing_objective": selected_objective.value,
                    "target_state_objective": "MARGIN_LOCKED",
                    "target_old_access_count": signed.target_old_access_count,
                    "routing_context_sha256": signed.routing_context_sha256,
                    "routing_context_count": signed.routing_context_count,
                    "routing_context_group_sizes": list(
                        signed.routing_context_group_sizes
                    ),
                    "routing_backward_count": signed.routing_backward_count,
                }
            )
        recorder.field(failure_payload)
        raise ODEBFContractError("PROGRESS_INFEASIBLE: adaptive BF field")
    raw_velocity = StepSizeIndependentVelocity.from_controller_values(
        field.identity_sha256, raw.values
    )
    field_state_sha256 = _factor_state(
        capture.entry_sha256,
        factors,
        target_state,
    )
    final_bf_values = np.asarray(projection.values, dtype=np.float64)
    functional_p_field_payload: dict[str, Any] = {
        "policy": selected_functional_p_field.value,
        "probe_executed": False,
        "field_decision_influence_count": 0,
    }
    if selected_functional_p_field is not FunctionalPFieldPolicy.NONE:
        if (
            functional_p_replay_entry is None
            or theta0_cache is None
            or alias is None
            or touched is None
            or schedule is None
        ):
            raise ODEBFContractError(
                "functional-P field probe dependencies are absent"
            )
        final_bf_values, functional_p_field_payload = (
            _functional_p_probe_and_transform(
                model,
                tokenizer,
                alias=alias,
                field=field,
                accepted_index=accepted_index,
                factors=factors,
                target_state=target_state,
                capture=capture,
                problem=barrier.problem,
                pre_soft_velocity=projection.values,
                replay_entry=functional_p_replay_entry,
                theta0_cache=theta0_cache,
                lock=lock,
                ledger=ledger,
                touched=touched,
                history=history,
                schedule=schedule,
                policy=selected_functional_p_field,
                factor_state_sha256=field_state_sha256,
            )
        )
    bf_velocity = StepSizeIndependentVelocity.from_controller_values(
        field.identity_sha256, final_bf_values
    )
    target_velocity, target_receipt = write_aware_target_velocity(
        model,
        tokenizer,
        requests,
        field,
        final_bf_values,
        cumulative_factors_by_weight=factors,
        target_base=target_base,
        native_target=native_target,
        trust_fraction=lock.target_trust_fraction,
        ledger=ledger,
    )
    payload = {
        "accepted_index": accepted_index,
        "residual_policy": policy,
        "field": field.raw_free_payload(),
        "field_sha256": field.identity_sha256,
        "signed_slopes": list(signed.signed_progress),
        "signed_slopes_sha256": canonical_hash(list(signed.signed_progress)),
        "excluded_nonpositive_layers": list(signed.excluded_nonpositive_layers),
        "raw_velocity": list(raw_velocity.velocity),
        "raw_velocity_sha256": raw_velocity.velocity_sha256,
        "bf_velocity": list(bf_velocity.velocity),
        "bf_velocity_sha256": bf_velocity.velocity_sha256,
        "h_ref": fraction_payload(H_REF),
        "raw_to_bf": routing_change(raw_velocity.velocity, bf_velocity.velocity),
        "raw_certificate": asdict(raw.certificate),
        "bf_certificate": asdict(projection.certificate),
        "projection_distance": projection.projection_distance,
        "functional_p_field": functional_p_field_payload,
        "routing_problem_sha256": barrier.problem.identity(),
        "preservation_constraint_policy": selected_preservation.value,
        "preservation_solver_decision_influence_count": {
            "structural_h": (
                1
                if selected_preservation is PreservationConstraintPolicy.LOCKED
                else 0
            ),
            "structural_p": (
                1
                if selected_preservation is PreservationConstraintPolicy.LOCKED
                else 0
            ),
            "trust": (
                1
                if selected_preservation is PreservationConstraintPolicy.LOCKED
                else 0
            ),
        },
        "target_velocity": asdict(target_receipt),
        "field_build_wall_seconds": time.perf_counter() - started,
        "field_build_counter_delta": {
            name: ledger.counters[name] - counter_before[name]
            for name in sorted(ledger.counters)
        },
    }
    payload["layer_routing"] = _field_layer_routing_payload(
        field=field,
        signed_progress=signed.signed_progress,
        problem=barrier.problem,
        raw_velocity=raw_velocity.velocity,
        bf_velocity=bf_velocity.velocity,
        raw_certificate=raw.certificate,
        bf_certificate=projection.certificate,
        state_sha256=field_state_sha256,
        target_z_sha256=canonical_hash(list(capture.direct_z_sha256)),
        factor_state_sha256=field_state_sha256,
        step_index=accepted_index,
    )
    if (
        selected_objective is not RoutingObjective.MARGIN
        or recorder.variant_label != variant.value
    ):
        payload.update(
            {
                "routing_objective": selected_objective.value,
                "routing_objective_receipt_sha256": (
                    signed.objective_receipt_sha256
                ),
                "routing_context_sha256": signed.routing_context_sha256,
                "routing_context_count": signed.routing_context_count,
                "routing_context_group_sizes": list(
                    signed.routing_context_group_sizes
                ),
                "routing_backward_count": signed.routing_backward_count,
                "target_old_access_count": signed.target_old_access_count,
                "target_state_objective": "MARGIN_LOCKED",
            }
        )
    receipt_sha256 = recorder.field(payload)
    return ActiveField(
        field=field,
        signed_progress=signed,
        routing=barrier,
        problem=barrier.problem,
        raw=raw,
        raw_velocity=raw_velocity,
        bf_velocity=bf_velocity,
        projection=projection,
        target_velocity=target_velocity,
        target_velocity_receipt=target_receipt,
        receipt_sha256=receipt_sha256,
        layer_routing_payload=dict(payload["layer_routing"]),
        functional_p_field_payload=dict(functional_p_field_payload),
    )


def _evaluate_rewrite(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    factors: Mapping[str, Sequence[WaypointFactor]],
    ledger: ComputeLedger,
) -> ModelEvaluationReceipt:
    from .evaluator import evaluate_counterfact_rewrite_batch

    context = (
        CumulativeBF16FunctionalTrial(model, factors, row_block=64)
        if factors
        else None
    )
    if context is None:
        receipt = evaluate_counterfact_rewrite_batch(
            model, tokenizer, requests, model_alias=alias
        )
    else:
        with context:
            receipt = evaluate_counterfact_rewrite_batch(
                model, tokenizer, requests, model_alias=alias
            )
    ledger.increment("evaluator_forward", receipt.model_forward_count)
    ledger.increment("evaluator_tokens", receipt.processed_token_count)
    return receipt


def _controller_replay_entry(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    arm_state: Any,
    sample_waypoint: int,
    factors: Mapping[str, Sequence[WaypointFactor]],
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: StatelessReplaySchedule,
    outer_entry_p_cache: OuterEntryPretrainedCache,
) -> Any:
    # Reuse the frozen P1R3 helper with the real arm-local history/ledger.
    # Adaptive retries retain ``sample_waypoint``; the legacy control advances
    # it with its consumed slot, which is the named clock difference.
    from .p1_runtime import _replay_entry

    return _replay_entry(
        model,
        tokenizer,
        alias=alias,
        arm_state=arm_state,
        sequential_batch=0,
        waypoint=sample_waypoint,
        factors=factors,
        request_by_sha256=request_by_sha256,
        population_by_sha256=population_by_sha256,
        schedule=schedule,
        outer_entry_p_cache=outer_entry_p_cache,
    )


def _functional_trial(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    entry: Any,
    theta0_cache: Theta0TeacherCache,
    factors: Mapping[str, Sequence[WaypointFactor]],
    lock: P1ControllerLock,
    ledger: ComputeLedger,
) -> Any:
    from .p1_runtime import _functional_trial as frozen_functional_trial

    return frozen_functional_trial(
        model,
        tokenizer,
        alias=alias,
        entry=entry,
        theta0_cache=theta0_cache,
        factors=factors,
        lock=lock,
        ledger=ledger,
    )


def _functional_p_probe_and_transform(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    field: P1DynamicField,
    accepted_index: int,
    factors: Mapping[str, Sequence[WaypointFactor]],
    target_state: torch.Tensor,
    capture: P1NativeCapture,
    problem: RoutingProblem,
    pre_soft_velocity: Sequence[float],
    replay_entry: Any,
    theta0_cache: Theta0TeacherCache,
    lock: P1ControllerLock,
    ledger: ComputeLedger,
    touched: Mapping[str, torch.nn.Parameter],
    history: P1HistoryLedger,
    schedule: StatelessReplaySchedule,
    policy: FunctionalPFieldPolicy,
    factor_state_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Measure the cached one-sided BF16 secant and optionally rectify ``u``.

    Baseline plus five basis probes share one sealed controller-P replay entry.
    The helper is called exactly once per newly built field; adaptive retries
    retain the resulting :class:`ActiveField` and therefore cannot re-probe.
    """

    if policy not in (
        FunctionalPFieldPolicy.PROBE_ONLY,
        FunctionalPFieldPolicy.SOFT_HARD,
    ):
        raise ODEBFContractError("functional-P probe policy differs")
    if tuple(int(item.layer) for item in field.layers) != ROUTING_LAYER_IDS:
        raise ODEBFContractError("functional-P probe layer order differs")
    if not math.isclose(
        float(lock.functional_p_budget_nats),
        FUNCTIONAL_P_BUDGET,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ODEBFContractError("functional-P probe budget differs")
    before_model = _parameter_contract_sha256(touched)
    before_history = history.snapshot().digest
    before_sampler = schedule.state_digest
    before_target = tensor_sha256(target_state)
    before_factor_state = _factor_state(
        capture.entry_sha256,
        factors,
        target_state,
    )
    if before_factor_state != factor_state_sha256:
        raise ODEBFContractError("functional-P probe factor state differs")
    counter_before = dict(ledger.counters)
    wall_started = time.perf_counter()
    gpu_start: torch.cuda.Event | None = None
    gpu_end: torch.cuda.Event | None = None
    if torch.cuda.is_available():
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()

    baseline_pair = _functional_trial(
        model,
        tokenizer,
        alias=alias,
        entry=replay_entry,
        theta0_cache=theta0_cache,
        factors=factors,
        lock=lock,
        ledger=ledger,
    )
    baseline_risk = _risk_payload(baseline_pair.pretrained)
    baseline_history_risk = _risk_payload(baseline_pair.historical)
    expected_sample_order = replay_entry.pretrained_baseline.sample_order_sha256
    if (
        baseline_pair.pretrained.sample_order_sha256 != expected_sample_order
        or baseline_pair.pretrained_entry_receipt.request_order_sha256
        != expected_sample_order
        or baseline_pair.pretrained_entry_receipt.value_sha256
        != replay_entry.pretrained_baseline.entry_kl_identity_sha256
        or baseline_pair.historical_entry_receipt.value_sha256
        != replay_entry.history_entry_sha256
    ):
        raise ODEBFContractError(
            "functional-P probe baseline/sample identity differs"
        )
    probe_pairs: list[Any] = []
    probe_risks: list[dict[str, Any]] = []
    probe_damage: list[float] = []
    history_mean_probe: list[float] = []
    history_smooth_probe: list[float] = []
    dimension = len(field.layers)
    for ordinal in range(dimension):
        basis = [0.0] * dimension
        basis[ordinal] = 1.0
        basis_velocity = StepSizeIndependentVelocity.from_controller_values(
            field.identity_sha256,
            basis,
        )
        increment = adaptive_waypoint_factors(
            field,
            basis_velocity,
            accepted_index=accepted_index,
            delta_tau=H_REF,
        )
        probe_factors = _merge_factors(factors, increment)
        observed = _functional_trial(
            model,
            tokenizer,
            alias=alias,
            entry=replay_entry,
            theta0_cache=theta0_cache,
            factors=probe_factors,
            lock=lock,
            ledger=ledger,
        )
        if (
            observed.pretrained.sample_order_sha256 != expected_sample_order
            or observed.pretrained_entry_receipt.request_order_sha256
            != expected_sample_order
            or observed.pretrained_entry_receipt.value_sha256
            != replay_entry.pretrained_baseline.entry_kl_identity_sha256
            or observed.historical.sample_order_sha256
            != baseline_pair.historical.sample_order_sha256
            or observed.historical_entry_receipt.value_sha256
            != replay_entry.history_entry_sha256
        ):
            raise ODEBFContractError(
                "functional-P probe endpoint/sample identity differs"
            )
        probe_pairs.append(observed)
        probe_risks.append(_risk_payload(observed.pretrained))
        probe_damage.append(float(observed.pretrained.mean_positive_damage))
        history_mean_probe.append(
            float(observed.historical.mean_positive_damage)
        )
        history_smooth_probe.append(
            float(observed.historical.smooth_max_positive_damage)
        )

    if gpu_start is not None and gpu_end is not None:
        gpu_end.record()
        gpu_end.synchronize()
        gpu_seconds = float(gpu_start.elapsed_time(gpu_end)) / 1000.0
    else:
        gpu_seconds = 0.0
    wall_seconds = time.perf_counter() - wall_started
    ledger.add_time(
        "functional_p_secant_probe",
        wall_seconds=wall_seconds,
        gpu_seconds=gpu_seconds,
    )
    secant = build_functional_replay_secant(
        pretrained_baseline_damage=float(
            baseline_pair.pretrained.mean_positive_damage
        ),
        pretrained_probe_damage=probe_damage,
        pretrained_sample_order_sha256=expected_sample_order,
        pretrained_baseline_identity_sha256=canonical_hash(baseline_risk),
        history_item_count=int(baseline_pair.historical.item_count),
        history_sample_order_sha256=(
            baseline_pair.historical.sample_order_sha256
        ),
        history_baseline_identity_sha256=(
            replay_entry.history_entry_sha256
        ),
        historical_mean_baseline=float(
            baseline_pair.historical.mean_positive_damage
        ),
        historical_mean_probe=history_mean_probe,
        historical_smooth_baseline=float(
            baseline_pair.historical.smooth_max_positive_damage
        ),
        historical_smooth_probe=history_smooth_probe,
        factor_state_sha256=factor_state_sha256,
    )
    pre_soft = np.asarray(tuple(pre_soft_velocity), dtype=np.float64)
    soft_result: PSoftHardResult | None = None
    final = pre_soft.copy()
    if policy is FunctionalPFieldPolicy.SOFT_HARD:
        soft_result = solve_p_soft_hard(problem, pre_soft, secant)
        final = np.asarray(soft_result.soft_velocity, dtype=np.float64)
        ledger.increment("qp_solve", 2)
        ledger.increment("qp_certificate", 2)

    after_model = _parameter_contract_sha256(touched)
    after_history = history.snapshot().digest
    after_sampler = schedule.state_digest
    after_target = tensor_sha256(target_state)
    after_factor_state = _factor_state(
        capture.entry_sha256,
        factors,
        target_state,
    )
    if (
        after_model != before_model
        or after_history != before_history
        or after_sampler != before_sampler
        or after_target != before_target
        or after_factor_state != before_factor_state
    ):
        raise ODEBFStateError(
            "functional-P probe mutated model/state/history/sampler/RNG"
        )
    counter_delta = {
        name: int(ledger.counters[name] - counter_before[name])
        for name in sorted(ledger.counters)
    }
    payload: dict[str, Any] = {
        "policy": policy.value,
        "probe_executed": True,
        "field_decision_influence_count": (
            1 if policy is FunctionalPFieldPolicy.SOFT_HARD else 0
        ),
        "teacher_contract": "fixed-outer-entry-pretrained-theta0",
        "controller_p_sample_order_sha256": expected_sample_order,
        "controller_p_baseline_identity_sha256": (
            replay_entry.pretrained_baseline.entry_kl_identity_sha256
        ),
        "controller_p_schedule_sha256": replay_entry.schedule_sha256,
        "factor_state_sha256": factor_state_sha256,
        "layer_order": list(ROUTING_LAYER_IDS),
        "baseline": baseline_risk,
        "historical_baseline": baseline_history_risk,
        "probe_endpoints": probe_risks,
        "historical_probe_endpoints": [
            _risk_payload(pair.historical) for pair in probe_pairs
        ],
        "probe_endpoint_identity_sha256": [
            pair.pretrained_trial_receipt.value_sha256 for pair in probe_pairs
        ],
        "secant": secant.raw_free_payload(),
        "generic_hp_soft_capable": True,
        "historical_soft_active": secant.historical_soft_active,
        "historical_soft_reason": (
            "NONEMPTY_HISTORY"
            if secant.historical_soft_active
            else "EMPTY_HISTORY"
        ),
        "pre_soft_velocity": pre_soft.tolist(),
        "selected_velocity": final.tolist(),
        "soft_hard_result": (
            None if soft_result is None else soft_result.raw_free_payload()
        ),
        "probe_cost": {
            "baseline_evaluation_count": 1,
            "actuator_probe_count": dimension,
            "counter_delta": counter_delta,
            "wall_seconds": wall_seconds,
            "gpu_seconds": gpu_seconds,
        },
        "same_state_retry_cache": True,
        "purity": {
            "model_before_sha256": before_model,
            "model_after_sha256": after_model,
            "history_before_sha256": before_history,
            "history_after_sha256": after_history,
            "sampler_before_sha256": before_sampler,
            "sampler_after_sha256": after_sampler,
            "target_before_sha256": before_target,
            "target_after_sha256": after_target,
            "factor_state_before_sha256": before_factor_state,
            "factor_state_after_sha256": after_factor_state,
        },
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return final, payload


def _capacity_payload(
    capture: P1NativeCapture,
    field: P1DynamicField,
    previous: Mapping[str, Sequence[WaypointFactor]],
    candidate: Mapping[str, Sequence[WaypointFactor]],
    increment: Mapping[str, WaypointFactor],
    *,
    delta_tau: Fraction,
    raw_velocity: Sequence[float],
    bf_velocity: Sequence[float],
    signed_slopes: Sequence[float],
    p_self: Sequence[float],
    previous_bf_velocity: Sequence[float] | None,
    field_problem: RoutingProblem,
    raw_certificate: SolverCertificate,
    bf_certificate: SolverCertificate,
    step_index: int,
    state_sha256: str,
    target_z_sha256: str,
    factor_state_sha256: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    layers: list[int] = []
    requested_frobenius: list[float] = []
    c_energy: list[float] = []
    actual_step: list[float] = []
    cumulative: list[float] = []
    applied = [float(delta_tau) * float(value) for value in bf_velocity]
    raw_progress = [
        float(slope) * float(value)
        for slope, value in zip(signed_slopes, raw_velocity, strict=True)
    ]
    bf_progress = [
        float(slope) * float(value)
        for slope, value in zip(signed_slopes, bf_velocity, strict=True)
    ]
    streaming_receipts: dict[str, Any] = {}
    for ordinal, layer_field in enumerate(field.layers):
        name = layer_field.weight_name
        theta = increment[name].theta
        receipt = streaming_bf16_capacity(
            capture.entry_weights[name],
            previous.get(name, ()),
            candidate[name],
            weight_name=name,
        )
        layers.append(layer_field.layer)
        requested_frobenius.append(theta**2 * layer_field.factor_frobenius_sq)
        c_energy.append(theta**2 * float(p_self[ordinal]))
        actual_step.append(receipt.step_energy)
        cumulative.append(receipt.cumulative_energy)
        streaming_receipts[str(layer_field.layer)] = asdict(receipt)
    layer_ids = tuple(layers)
    if layer_ids != ROUTING_LAYER_IDS:
        raise ODEBFContractError("adaptive telemetry layer order differs")
    applied_array = np.asarray(applied, dtype=np.float64)
    historical_contribution = (
        2.0 * field_problem.historical.linear * applied_array
        + applied_array * (field_problem.historical.gram @ applied_array)
    )
    pretrained_contribution = (
        2.0 * field_problem.pretrained.linear * applied_array
        + applied_array * (field_problem.pretrained.gram @ applied_array)
    )
    trust_contribution = applied_array * (
        field_problem.trust_metric @ applied_array
    )
    raw_array = np.asarray(raw_velocity, dtype=np.float64)
    bf_array = np.asarray(bf_velocity, dtype=np.float64)
    caps = np.asarray(field_problem.layer_caps, dtype=np.float64)
    layer_routing = build_layer_routing_telemetry(
        step_index=step_index,
        stage="TRIAL",
        layer_ids=layer_ids,
        signed_efficiency=signed_slopes,
        raw_velocity=raw_velocity,
        bf_velocity=bf_velocity,
        applied_coefficient=applied,
        active_direction_mask=np.asarray(signed_slopes, dtype=np.float64) > 0.0,
        raw_cap_bound_mask=np.isclose(raw_array, caps, rtol=0.0, atol=1.0e-12),
        bf_cap_bound_mask=np.isclose(bf_array, caps, rtol=0.0, atol=1.0e-12),
        raw_zero_bound_mask=np.isclose(raw_array, 0.0, rtol=0.0, atol=1.0e-12),
        bf_zero_bound_mask=np.isclose(bf_array, 0.0, rtol=0.0, atol=1.0e-12),
        predicted_progress_contribution=(
            np.asarray(signed_slopes, dtype=np.float64) * applied_array
        ),
        prequantized_update_energy=requested_frobenius,
        realized_bf16_update_energy=actual_step,
        cumulative_bf16_capacity=cumulative,
        bf16_capacity_contribution=actual_step,
        structural_h_contribution=historical_contribution,
        structural_p_contribution=pretrained_contribution,
        trust_contribution=trust_contribution,
        field_sha256=field.identity_sha256,
        state_sha256=state_sha256,
        target_z_sha256=target_z_sha256,
        factor_state_sha256=factor_state_sha256,
        raw_solver_certificate_sha256=canonical_hash(asdict(raw_certificate)),
        bf_solver_certificate_sha256=canonical_hash(asdict(bf_certificate)),
        candidate_sha256=candidate_sha256,
    )
    result = {
        "layers": layers,
        "raw_velocity": concentration_summary(raw_velocity),
        "bf_velocity": concentration_summary(bf_velocity),
        "applied_coefficient": concentration_summary(applied),
        "raw_progress_contribution": concentration_summary(raw_progress),
        "bf_progress_contribution": concentration_summary(bf_progress),
        "requested_frobenius_energy": concentration_summary(requested_frobenius),
        "c_energy": concentration_summary(c_energy),
        "actual_bf16_step_energy": concentration_summary(actual_step),
        "cumulative_bf16_capacity": concentration_summary(cumulative),
        "raw_to_bf": routing_change(raw_velocity, bf_velocity),
        "previous_to_current_bf": (
            None
            if previous_bf_velocity is None
            else routing_change(previous_bf_velocity, bf_velocity)
        ),
        "streaming_receipts": streaming_receipts,
        "layer_routing": layer_routing,
        "dense_fp64_full_delta_live": 0,
        "dense_fp32_full_delta_live": 0,
        "effective_bf16_weight_peak_live": 1,
    }
    result["identity_sha256"] = canonical_hash(result)
    return result


def _run_trial(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    variant: AdaptiveVariant,
    accepted_index: int,
    n_trial: int,
    retry_index: int,
    tau_before: Fraction,
    delta_tau: Fraction,
    delta_tau_proposed: Fraction,
    remainder: bool,
    active: ActiveField,
    current_factors: Mapping[str, Sequence[WaypointFactor]],
    current_target: torch.Tensor,
    current_margin: Any,
    replay_entry: Any,
    theta0_cache: Theta0TeacherCache,
    capture: P1NativeCapture,
    lock: P1ControllerLock,
    ledger: ComputeLedger,
    recorder: AdaptiveReceiptRecorder,
    touched: Mapping[str, torch.nn.Parameter],
    history: P1HistoryLedger,
    schedule: StatelessReplaySchedule,
    previous_bf_velocity: Sequence[float] | None,
    accepted_by_layer: Mapping[int, Sequence[AcceptedLayerContribution]],
    contexts: Sequence[Sequence[str]],
    routing_objective: RoutingObjective | str = RoutingObjective.MARGIN,
    functional_p_policy: FunctionalPDecisionPolicy | str = (
        FunctionalPDecisionPolicy.PCTRL
    ),
    preservation_policy: PreservationConstraintPolicy | str = (
        PreservationConstraintPolicy.LOCKED
    ),
) -> TrialOutcome:
    from .p1_runtime import _controller_progress_telemetry

    selected_objective = select_locked_routing_objective(routing_objective)
    if delta_tau <= 0 or delta_tau > H_REF:
        raise ODEBFContractError("adaptive trial delta_tau differs")
    trial_wall_started = time.perf_counter()
    trial_gpu_start: torch.cuda.Event | None = None
    trial_gpu_end: torch.cuda.Event | None = None
    if torch.cuda.is_available():
        trial_gpu_start = torch.cuda.Event(enable_timing=True)
        trial_gpu_end = torch.cuda.Event(enable_timing=True)
        trial_gpu_start.record()
    counter_before = dict(ledger.counters)
    before_model = _parameter_contract_sha256(touched)
    before_history = history.snapshot().digest
    before_sampler = schedule.state_digest
    before_target = tensor_sha256(current_target)
    before_factors = _factor_state(capture.entry_sha256, current_factors, current_target)
    before_omega = _omega_state(accepted_by_layer)
    increment = adaptive_waypoint_factors(
        active.field,
        active.bf_velocity,
        accepted_index=accepted_index,
        delta_tau=delta_tau,
    )
    candidate_factors = _merge_factors(current_factors, increment)
    candidate_margin = (
        evaluate_controller_margin(
            model,
            tokenizer,
            requests,
            cumulative_factors_by_weight=candidate_factors,
        )
        if selected_objective is RoutingObjective.MARGIN
        else evaluate_routing_progress(
            model,
            tokenizer,
            requests,
            cumulative_factors_by_weight=candidate_factors,
            objective=selected_objective,
            contexts=contexts,
        )
    )
    progress = _controller_progress_telemetry(current_margin, candidate_margin)
    functional = _functional_trial(
        model,
        tokenizer,
        alias=alias,
        entry=replay_entry,
        theta0_cache=theta0_cache,
        factors=candidate_factors,
        lock=lock,
        ledger=ledger,
    )
    evaluation = _evaluate_rewrite(
        model,
        tokenizer,
        requests,
        alias=alias,
        factors=candidate_factors,
        ledger=ledger,
    )
    if evaluation.counterfact_scores is None or len(
        evaluation.counterfact_scores
    ) != BATCH_SIZE:
        raise ODEBFContractError(
            "adaptive CounterFact trial lacks canonical numeric scores"
        )
    canonical_rewrite = [
        {
            "request_index": index,
            "nll_new": score.target_new_nll,
            "nll_old": score.target_true_nll,
            "margin": score.target_true_nll - score.target_new_nll,
            "success": score.success_bit,
        }
        for index, score in enumerate(evaluation.counterfact_scores)
    ]
    functional_p_observation = _risk_payload(functional.pretrained)
    functional_p_decision = functional_p_decision_receipt(
        functional_p_policy,
        functional_p_observation,
    )
    functional_p_probe_prediction: dict[str, Any] | None = None
    if active.functional_p_field_payload.get("probe_executed") is True:
        field_probe = active.functional_p_field_payload
        if (
            field_probe.get("controller_p_sample_order_sha256")
            != functional_p_observation["sample_order_sha256"]
        ):
            raise ODEBFContractError(
                "functional-P probe/candidate sample identity differs"
            )
        secant_payload = field_probe.get("secant")
        if not isinstance(secant_payload, Mapping):
            raise ODEBFContractError("functional-P trial secant payload differs")
        signed_q = np.asarray(
            secant_payload.get("signed_secant"), dtype=np.float64
        )
        velocity = np.asarray(active.bf_velocity.velocity, dtype=np.float64)
        if (
            signed_q.shape != velocity.shape
            or not np.isfinite(signed_q).all()
            or float(secant_payload.get("h_ref", 0.0))
            != FUNCTIONAL_P_PROBE_H_REF
        ):
            raise ODEBFContractError("functional-P trial secant geometry differs")
        baseline_damage = float(secant_payload["baseline_damage"])
        predicted = baseline_damage + float(delta_tau) * float(
            signed_q @ velocity
        )
        actual = float(functional_p_decision.observed_mean_positive_damage)
        if not math.isfinite(predicted):
            raise ODEBFContractError("functional-P candidate prediction is non-finite")
        functional_p_probe_prediction = {
            "probe_cache_identity_sha256": secant_payload[
                "cache_identity_sha256"
            ],
            "controller_p_sample_order_sha256": functional_p_observation[
                "sample_order_sha256"
            ],
            "baseline_damage": baseline_damage,
            "signed_secant": signed_q.tolist(),
            "delta_tau": float(delta_tau),
            "q_dot_velocity": float(signed_q @ velocity),
            "predicted_candidate_controller_p": predicted,
            "actual_candidate_controller_p": actual,
            "signed_prediction_residual": actual - predicted,
            "predicted_pass_at_locked_budget": bool(
                predicted <= FUNCTIONAL_P_BUDGET
            ),
            "actual_pass_at_locked_budget": bool(
                functional_p_decision.observed_pass_at_locked_budget
            ),
            "decision_influence_count": 0,
        }
        history_item_count = int(secant_payload["history_item_count"])
        signed_h_mean = np.asarray(
            secant_payload["historical_mean_signed_secant"],
            dtype=np.float64,
        )
        signed_h_smooth = np.asarray(
            secant_payload["historical_smooth_signed_secant"],
            dtype=np.float64,
        )
        if (
            signed_h_mean.shape != velocity.shape
            or signed_h_smooth.shape != velocity.shape
        ):
            raise ODEBFContractError(
                "functional-H trial secant geometry differs"
            )
        predicted_h_mean = float(
            secant_payload["historical_mean_baseline"]
        ) + float(delta_tau) * float(signed_h_mean @ velocity)
        predicted_h_smooth = float(
            secant_payload["historical_smooth_baseline"]
        ) + float(delta_tau) * float(signed_h_smooth @ velocity)
        actual_h_mean = float(functional.historical.mean_positive_damage)
        actual_h_smooth = float(
            functional.historical.smooth_max_positive_damage
        )
        functional_p_probe_prediction["historical"] = {
            "history_item_count": history_item_count,
            "history_sample_order_sha256": (
                secant_payload["history_sample_order_sha256"]
            ),
            "history_baseline_identity_sha256": (
                secant_payload["history_baseline_identity_sha256"]
            ),
            "historical_soft_active": history_item_count > 0,
            "reason": (
                "NONEMPTY_HISTORY" if history_item_count > 0 else "EMPTY_HISTORY"
            ),
            "predicted_mean": predicted_h_mean,
            "actual_mean": actual_h_mean,
            "signed_mean_prediction_residual": actual_h_mean - predicted_h_mean,
            "predicted_smoothmax": predicted_h_smooth,
            "actual_smoothmax": actual_h_smooth,
            "signed_smoothmax_prediction_residual": (
                actual_h_smooth - predicted_h_smooth
            ),
            "actual_hard_gate_pass": bool(functional.historical.passed),
            "decision_influence_count": 0,
        }
    beta_alias = float(delta_tau / H_REF)
    verdict = verify_backtracked_candidate(
        active.problem,
        np.asarray(active.bf_velocity.velocity, dtype=np.float64),
        beta=beta_alias,
        actual_signed_progress=progress.actual_signed_progress,
        functional_h_pass=functional.historical.passed,
        functional_p_pass=functional_p_decision.decision_pass,
        authoritative_bf16_pass=True,
        minimum_progress=lock.minimum_progress,
        rho_accept=lock.rho_accept,
    )
    preservation_decision = preservation_decision_payload(
        preservation_policy,
        verdict,
        functional_p_decision,
    )
    preservation_pass = preservation_decision["decision_pass"]
    feasibility = FeasibilityVerdict(
        preservation_pass["structural_h"],
        preservation_pass["structural_p"],
        preservation_pass["trust"],
        preservation_pass["functional_h"],
        preservation_pass["functional_p"],
        True,
    )
    gate_accepted, first_rejecting_component = candidate_gate_decision(
        verdict, preservation_decision
    )
    applied_velocity = beta_alias * np.asarray(
        active.bf_velocity.velocity, dtype=np.float64
    )
    structural = {
        "historical": _structural_payload(
            active.problem.historical, applied_velocity
        ),
        "pretrained": _structural_payload(
            active.problem.pretrained, applied_velocity
        ),
        "trust": _trust_payload(active.problem, applied_velocity),
        "functional_h": _risk_payload(functional.historical),
        "functional_p": functional_p_observation,
    }
    target_trial = (
        current_target
        + float(delta_tau)
        * active.target_velocity.detach().to(device="cpu", dtype=torch.float32)
    ).contiguous()
    snapshot_sha256 = _factor_state(
        capture.entry_sha256, candidate_factors, target_trial
    )
    capacity = _capacity_payload(
        capture,
        active.field,
        current_factors,
        candidate_factors,
        increment,
        delta_tau=delta_tau,
        raw_velocity=active.raw_velocity.velocity,
        bf_velocity=active.bf_velocity.velocity,
        signed_slopes=active.signed_progress.signed_progress,
        p_self=active.routing.pretrained_self_risk,
        previous_bf_velocity=previous_bf_velocity,
        field_problem=active.problem,
        raw_certificate=active.raw.certificate,
        bf_certificate=active.projection.certificate,
        step_index=n_trial,
        state_sha256=before_model,
        target_z_sha256=canonical_hash(list(capture.direct_z_sha256)),
        factor_state_sha256=before_factors,
        candidate_sha256=snapshot_sha256,
    )
    after_model = _parameter_contract_sha256(touched)
    after_history = history.snapshot().digest
    after_sampler = schedule.state_digest
    if (
        after_model != before_model
        or after_history != before_history
        or after_sampler != before_sampler
        or tensor_sha256(current_target) != before_target
        or _factor_state(capture.entry_sha256, current_factors, current_target)
        != before_factors
        or _omega_state(accepted_by_layer) != before_omega
    ):
        raise ODEBFStateError(
            "adaptive virtual trial mutated model/state/history/sampler/RNG"
        )
    ledger.increment("trial")
    trial_gpu_seconds = 0.0
    if trial_gpu_start is not None and trial_gpu_end is not None:
        trial_gpu_end.record()
        trial_gpu_end.synchronize()
        trial_gpu_seconds = float(trial_gpu_start.elapsed_time(trial_gpu_end)) / 1000.0
    trial_wall_seconds = time.perf_counter() - trial_wall_started
    ledger.add_time(
        "adaptive_trial_total",
        wall_seconds=trial_wall_seconds,
        gpu_seconds=trial_gpu_seconds,
    )
    allocated_bytes = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    reserved_bytes = (
        int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
    )
    host_maxrss_kib = int(
        system_resource.getrusage(system_resource.RUSAGE_SELF).ru_maxrss
    )
    ledger.observe_memory(
        allocated_bytes=allocated_bytes,
        reserved_bytes=reserved_bytes,
        maxrss_kib=host_maxrss_kib,
    )
    counter_delta = {
        name: int(ledger.counters[name] - counter_before[name])
        for name in sorted(ledger.counters)
    }
    progress_payload = {
        "actual": progress.actual_signed_progress,
        "predicted_beta": verdict.predicted_beta_progress,
        "raw_predicted": verdict.raw_predicted_progress,
        "rho": verdict.trust_ratio,
        "progress_pass": verdict.progress_pass,
        "minimum_progress": lock.minimum_progress,
        "rho_accept": lock.rho_accept,
        "requested_progress": active.problem.requested_progress,
        "maximum_feasible_progress": active.projection.maximum_feasible_progress,
        "per_request_signed_progress": list(progress.per_request_signed_progress),
        "per_request_signed_progress_sha256": (
            progress.per_request_signed_progress_sha256
        ),
        "per_request_improved_bits": list(progress.per_request_improved_bits),
        "per_request_harm_bits": list(progress.per_request_harm_bits),
        "functional_p_decision_influence_count": (
            functional_p_decision.decision_influence_count
        ),
        "functional_p_field_policy": (
            active.functional_p_field_payload.get(
                "policy", FunctionalPFieldPolicy.NONE.value
            )
        ),
        "functional_p_field_decision_influence_count": (
            int(
                active.functional_p_field_payload.get(
                    "field_decision_influence_count", 0
                )
            )
        ),
        "preservation_decision_influence_count": dict(
            preservation_decision["decision_influence_count"]
        ),
    }
    if (
        selected_objective is not RoutingObjective.MARGIN
        or recorder.variant_label != variant.value
    ):
        progress_payload.update(
            {
                "routing_objective": selected_objective.value,
                "target_state_objective": "MARGIN_LOCKED",
                "predicted_delta": verdict.predicted_beta_progress,
                "actual_delta": progress.actual_signed_progress,
                "rho_for_routing_objective": verdict.trust_ratio,
                "target_old_decision_influence_count": 0
                if selected_objective is RoutingObjective.TARGET_NEW_NLL
                else BATCH_SIZE,
            }
        )
        if isinstance(candidate_margin, TargetNewNLLReceipt):
            progress_payload.update(
                {
                    "routing_context_sha256": candidate_margin.context_sha256,
                    "routing_context_count": candidate_margin.context_count,
                    "routing_context_group_sizes": list(
                        candidate_margin.context_group_sizes
                    ),
                    "target_span_sha256": candidate_margin.target_span_sha256,
                    "suffix_token_counts": list(
                        candidate_margin.suffix_token_counts
                    ),
                }
            )
    routing_payload = {
        "field_sha256": active.field.identity_sha256,
        "field_receipt_sha256": active.receipt_sha256,
        "residual_policy": _residual_policy(variant),
        "residual_divisors": [item.residual_divisor for item in active.field.layers],
        "residual_norms": [
            float(torch.linalg.norm(item.residual.double()))
            for item in active.field.layers
        ],
        "raw_velocity": list(active.raw_velocity.velocity),
        "raw_velocity_sha256": active.raw_velocity.velocity_sha256,
        "bf_velocity": list(active.bf_velocity.velocity),
        "bf_velocity_sha256": active.bf_velocity.velocity_sha256,
        "coefficient": list(active.bf_velocity.applied(delta_tau)),
        "coefficient_l2_norm": float(
            np.linalg.norm(
                np.asarray(active.bf_velocity.applied(delta_tau), dtype=np.float64)
            )
        ),
        "velocity_l2_norm": float(
            np.linalg.norm(
                np.asarray(active.bf_velocity.velocity, dtype=np.float64)
            )
        ),
        "velocity_layer_caps": active.problem.layer_caps.tolist(),
        "scaled_candidate_layer_caps": (
            float(delta_tau / H_REF) * active.problem.layer_caps
        ).tolist(),
        "target_velocity_sha256": active.target_velocity_receipt.velocity_sha256,
        "target_trial_sha256": tensor_sha256(target_trial),
        "target_weight_shared_delta_tau": True,
    }
    trial_payload = {
        "accepted_index_before": accepted_index,
        "n_trial": n_trial,
        "retry_index": retry_index,
        "tau_before": fraction_payload(tau_before),
        "tau_after_candidate": fraction_payload(tau_before + delta_tau),
        "eta_h_ref": fraction_payload(H_REF),
        "delta_tau_proposed": fraction_payload(delta_tau_proposed),
        "delta_tau_trial": fraction_payload(delta_tau),
        "beta_diagnostic_alias": beta_alias,
        "remainder": remainder,
        "gate_accepted": gate_accepted,
        "first_rejecting_component": first_rejecting_component,
        "progress": progress_payload,
        "structural_functional": structural,
        "functional_p_decision": functional_p_decision.raw_free_payload(),
        "functional_p_probe_prediction": functional_p_probe_prediction,
        "preservation_decision": preservation_decision,
        "official_success": evaluation.batch_success.raw_free_payload(),
        "canonical_rewrite": canonical_rewrite,
        "canonical_rewrite_sha256": canonical_hash(canonical_rewrite),
        "snapshot_sha256": snapshot_sha256,
        "routing": routing_payload,
        "capacity": capacity,
        "compute": {
            "counter_delta": counter_delta,
            "wall_seconds": trial_wall_seconds,
            "gpu_seconds": trial_gpu_seconds,
            "peak_allocated_bytes": allocated_bytes,
            "peak_reserved_bytes": reserved_bytes,
            "host_maxrss_kib": host_maxrss_kib,
        },
        "purity": {
            "model_before_sha256": before_model,
            "model_after_sha256": after_model,
            "history_before_sha256": before_history,
            "history_after_sha256": after_history,
            "sampler_before_sha256": before_sampler,
            "sampler_after_sha256": after_sampler,
            "state_before_sha256": before_factors,
            "state_after_sha256": before_factors,
            "omega_before_sha256": before_omega,
            "omega_after_sha256": before_omega,
        },
    }
    if (
        selected_objective is not RoutingObjective.MARGIN
        or recorder.variant_label != variant.value
    ):
        trial_payload["canonical_rewrite_role"] = (
            "DIAGNOSTIC_ONLY"
            if selected_objective is RoutingObjective.TARGET_NEW_NLL
            else "LOCKED_MARGIN_DIAGNOSTIC_AND_OFFICIAL_SUCCESS"
        )
    receipt_sha256 = recorder.trial(trial_payload)
    return TrialOutcome(
        gate_accepted,
        candidate_factors,
        increment,
        target_trial,
        evaluation,
        candidate_margin,
        feasibility,
        functional,
        structural,
        routing_payload,
        progress_payload,
        capacity,
        snapshot_sha256,
        receipt_sha256,
        verdict.trust_ratio,
        {
            **functional_p_decision.raw_free_payload(),
            "preservation_decision": preservation_decision,
        },
    )


def _is_progress_infeasible(exc: BaseException) -> bool:
    return isinstance(exc, ODEBFContractError) and str(exc).startswith(
        "PROGRESS_INFEASIBLE:"
    )


def _termination_label(*, status: str, exact_hit: bool) -> str:
    if exact_hit:
        return "EXACT_HIT"
    if status in (
        "TRIAL_CAP_EXHAUSTED",
        "ACCEPTED_STEP_OR_HORIZON_CAP_UNREACHED",
        "SAME_STATE_RETRY_EXHAUSTED",
        "MIN_DT_EXHAUSTED",
    ):
        return status
    if status in (
        "PROGRESS_INFEASIBLE",
        "TAU_COMPLETE",
        "LEGACY_FIXED_S8_COMPLETE",
    ):
        return "TERMINAL_INFEASIBLE"
    raise ODEBFContractError("adaptive termination taxonomy differs")


def _append_accepted_snapshot(
    *,
    outcome: TrialOutcome,
    active: ActiveField,
    accepted_index_before: int,
    tau_after: Fraction,
    delta_tau: Fraction,
    transition_sha256: str,
    accepted_by_layer: dict[int, list[AcceptedLayerContribution]],
    ledger: ComputeLedger,
    recorder: AdaptiveReceiptRecorder,
    first_hit: FirstHitTracker,
) -> AcceptedSnapshot:
    """Advance only the transaction-local logical state after a passed trial."""

    accepted_index = accepted_index_before + 1
    for layer_field in active.field.layers:
        accepted_by_layer[layer_field.layer].append(
            AcceptedLayerContribution.from_field(
                layer_field,
                outcome.increment[layer_field.weight_name],
                history_action=layer_field.history_action,
            )
        )
    ledger.record_accepted_step(
        accepted_dt=float(delta_tau), completed_k_total=accepted_index
    )
    hit_record = FirstHitRecord(
        accepted_index,
        tau_after,
        outcome.snapshot_sha256,
        outcome.evaluation.batch_success.numerator,
        outcome.feasibility.all_pass,
    )
    first_hit.append(hit_record)
    is_first_hit = first_hit.first_online is hit_record
    accepted_layer_routing = relabel_layer_routing_telemetry(
        outcome.capacity_payload["layer_routing"],
        stage="ACCEPTED_TRANSITION",
    )
    accepted_sha256 = recorder.accepted(
        {
            "accepted_index": accepted_index,
            "tau_after": fraction_payload(tau_after),
            "delta_tau": fraction_payload(delta_tau),
            "transition_sha256": transition_sha256,
            "snapshot_sha256": outcome.snapshot_sha256,
            "field_sha256": active.field.identity_sha256,
            "raw_velocity_sha256": active.raw_velocity.velocity_sha256,
            "bf_velocity_sha256": active.bf_velocity.velocity_sha256,
            "official_success": outcome.evaluation.batch_success.raw_free_payload(),
            "online_feasibility": asdict(outcome.feasibility),
            "functional_p_decision": dict(outcome.functional_p_decision),
            "structural_functional": dict(outcome.structural_payload),
            "target_state_sha256": tensor_sha256(outcome.target_trial),
            "capacity_sha256": outcome.capacity_payload["identity_sha256"],
            "layer_routing": accepted_layer_routing,
            "first_hit_observed": is_first_hit,
            "target_weight_shared_delta_tau": True,
            "history_append_count": 0,
            "persistent_commit_count": 0,
        }
    )
    if is_first_hit:
        recorder.first_hit(
            {
                "accepted_index": accepted_index,
                "tau": fraction_payload(tau_after),
                "snapshot_sha256": outcome.snapshot_sha256,
                "accepted_receipt_sha256": accepted_sha256,
                "official_success": (
                    outcome.evaluation.batch_success.raw_free_payload()
                ),
                "online_feasibility": asdict(outcome.feasibility),
                "functional_p_decision": dict(outcome.functional_p_decision),
                "structural_functional": dict(outcome.structural_payload),
                "layer_routing": relabel_layer_routing_telemetry(
                    outcome.capacity_payload["layer_routing"],
                    stage="FIRST_HIT",
                ),
                "observation_only": True,
                "controller_dependency_count": 0,
            }
        )
    return AcceptedSnapshot(
        accepted_index,
        tau_after,
        delta_tau,
        _factor_map(outcome.candidate_factors),
        outcome.target_trial.detach().to(device="cpu", dtype=torch.float32).clone(),
        outcome.snapshot_sha256,
        outcome.evaluation,
        outcome.feasibility,
        dict(outcome.structural_payload),
        dict(outcome.routing_payload),
        dict(outcome.progress_payload),
        dict(outcome.capacity_payload),
        active.field.identity_sha256,
        active.raw_velocity.velocity_sha256,
        active.bf_velocity.velocity_sha256,
        accepted_sha256,
        dict(outcome.functional_p_decision),
    )


def _terminal_confirm_snapshots(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    arm_state: Any,
    snapshots: Sequence[AcceptedSnapshot],
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: StatelessReplaySchedule,
    outer_entry_p_cache: OuterEntryPretrainedCache,
    theta0_cache: Theta0TeacherCache,
    lock: P1ControllerLock,
    ledger: ComputeLedger,
    recorder: AdaptiveReceiptRecorder,
    trajectory_complete: bool,
    functional_p_policy: FunctionalPDecisionPolicy | str = (
        FunctionalPDecisionPolicy.PCTRL
    ),
    preservation_policy: PreservationConstraintPolicy | str = (
        PreservationConstraintPolicy.LOCKED
    ),
) -> list[dict[str, Any]]:
    if not snapshots:
        return []
    from .p1_runtime import _terminal_replay_entry

    terminal_entry = _terminal_replay_entry(
        model,
        tokenizer,
        alias=alias,
        arm_state=arm_state,
        sequential_batch=0,
        selected_stage=len(snapshots),
        request_by_sha256=request_by_sha256,
        population_by_sha256=population_by_sha256,
        schedule=schedule,
        outer_entry_p_cache=outer_entry_p_cache,
    )
    values: list[dict[str, Any]] = []
    for ordinal, snapshot in enumerate(snapshots):
        pair = _functional_trial(
            model,
            tokenizer,
            alias=alias,
            entry=terminal_entry,
            theta0_cache=theta0_cache,
            factors=snapshot.factors,
            lock=lock,
            ledger=ledger,
        )
        terminal_p_observation = _risk_payload(pair.pretrained)
        terminal_p_decision = functional_p_decision_receipt(
            functional_p_policy,
            terminal_p_observation,
        )
        terminal_preservation = _preservation_decision_payload_from_observed(
            preservation_policy,
            {
                "structural_h": bool(
                    snapshot.structural_payload["historical"]["passed"]
                ),
                "structural_p": bool(
                    snapshot.structural_payload["pretrained"]["passed"]
                ),
                "trust": bool(snapshot.structural_payload["trust"]["passed"]),
                "functional_h": bool(pair.historical.passed),
            },
            terminal_p_decision,
        )
        terminal_pass = terminal_preservation["decision_pass"]
        feasibility = FeasibilityVerdict(
            terminal_pass["structural_h"],
            terminal_pass["structural_p"],
            terminal_pass["trust"],
            terminal_pass["functional_h"],
            terminal_pass["functional_p"],
            True,
        )
        exact = bool(
            trajectory_complete
            and snapshot.evaluation.batch_success.joint_exact_success
            and feasibility.all_pass
        )
        payload = {
            "accepted_index": snapshot.accepted_index,
            "tau": fraction_payload(snapshot.tau),
            "snapshot_sha256": snapshot.snapshot_sha256,
            "official_success": snapshot.evaluation.batch_success.raw_free_payload(),
            "online_feasibility": asdict(snapshot.feasibility),
            "terminal_feasibility": asdict(feasibility),
            "functional_p_decision": terminal_p_decision.raw_free_payload(),
            "preservation_decision": terminal_preservation,
            "terminal_h": _risk_payload(pair.historical),
            "terminal_p": terminal_p_observation,
            "outer_entry_baseline": {
                "baseline_kind": "outer_entry",
                "outer_entry_snapshot_sha256": (
                    terminal_entry.pretrained_baseline.outer_entry_snapshot_sha256
                ),
                "sample_order_sha256": (
                    terminal_entry.pretrained_baseline.sample_order_sha256
                ),
                "entry_kl_identity_sha256": (
                    terminal_entry.pretrained_baseline.entry_kl_identity_sha256
                ),
            },
            "terminal_confirmed_exact_hit": exact,
            "trajectory_complete_required": True,
            "layer_routing": relabel_layer_routing_telemetry(
                snapshot.capacity_payload["layer_routing"],
                stage=(
                    "FINAL_ENDPOINT"
                    if ordinal == len(snapshots) - 1
                    else "SHADOW_ENDPOINT"
                ),
            ),
        }
        payload["receipt_sha256"] = recorder.terminal(payload)
        values.append(payload)
    return values


def _run_variant(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    variant: AdaptiveVariant,
    capture: P1NativeCapture,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    lock: P1ControllerLock,
    arm_state: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: StatelessReplaySchedule,
    outer_entry_p_cache: OuterEntryPretrainedCache,
    theta0_cache: Theta0TeacherCache,
    touched: Mapping[str, torch.nn.Parameter],
    recorder: AdaptiveReceiptRecorder,
    routing_objective: RoutingObjective | str = RoutingObjective.MARGIN,
    variant_label: str | None = None,
    functional_p_policy: FunctionalPDecisionPolicy | str = (
        FunctionalPDecisionPolicy.PCTRL
    ),
    preservation_policy: PreservationConstraintPolicy | str = (
        PreservationConstraintPolicy.LOCKED
    ),
    functional_p_field_policy: FunctionalPFieldPolicy | str = (
        FunctionalPFieldPolicy.NONE
    ),
) -> VariantRollout:
    """Run one isolated R_BF-controller variant without persistent mutation."""

    variant_lock = adaptive_lock(variant)
    selected_objective = select_locked_routing_objective(routing_objective)
    try:
        selected_functional_p_policy = FunctionalPDecisionPolicy(
            functional_p_policy
        )
    except (TypeError, ValueError) as exc:
        raise ODEBFContractError("adaptive functional-P policy differs") from exc
    try:
        selected_preservation_policy = PreservationConstraintPolicy(
            preservation_policy
        )
    except (TypeError, ValueError) as exc:
        raise ODEBFContractError("adaptive preservation policy differs") from exc
    try:
        selected_functional_p_field_policy = FunctionalPFieldPolicy(
            functional_p_field_policy
        )
    except (TypeError, ValueError) as exc:
        raise ODEBFContractError(
            "adaptive functional-P field policy differs"
        ) from exc
    if (
        selected_functional_p_field_policy
        is not FunctionalPFieldPolicy.NONE
        and variant is not AdaptiveVariant.FR_A8
    ):
        raise ODEBFContractError(
            "functional-P field probe requires the locked FR-A8 clock"
        )
    selected_label = variant.value if variant_label is None else variant_label
    if recorder.variant_label != selected_label or recorder.variant is not variant:
        raise ODEBFContractError("adaptive recorder/panel variant differs")
    layers = tuple(int(layer) for layer in hparams.layers)
    history = arm_state.history
    ledger = arm_state.ledger
    entry_model_sha256 = _parameter_contract_sha256(touched)
    entry_history_sha256 = history.snapshot().digest
    entry_sampler_sha256 = schedule.state_digest
    native_target = torch.stack(capture.direct_z, dim=1).to(dtype=torch.float32)
    target_base = capture.entry_current_z_by_layer[layers[-1]].clone()
    current_target = native_target.clone()
    current_factors: dict[str, tuple[WaypointFactor, ...]] = {}
    accepted_by_layer: dict[int, list[AcceptedLayerContribution]] = {
        layer: [] for layer in layers
    }
    snapshots: list[AcceptedSnapshot] = []
    first_hit = FirstHitTracker()
    current_margin = (
        evaluate_controller_margin(
            model,
            tokenizer,
            requests,
            cumulative_factors_by_weight=current_factors,
        )
        if selected_objective is RoutingObjective.MARGIN
        else evaluate_routing_progress(
            model,
            tokenizer,
            requests,
            cumulative_factors_by_weight=current_factors,
            objective=selected_objective,
            contexts=contexts,
        )
    )
    entry_eval = _evaluate_rewrite(
        model,
        tokenizer,
        requests,
        alias=alias,
        factors=current_factors,
        ledger=ledger,
    )
    field_build_count = 0
    n_trial = 0
    n_reject = 0
    accepted_t = Fraction(0, 1)
    previous_bf_velocity: tuple[float, ...] | None = None
    status = "ACTIVE"
    active: ActiveField | None = None
    initial_layer_routing: dict[str, Any] | None = None
    replay_entry: Any | None = None
    replay_field_sha256: str | None = None
    if (
        selected_functional_p_field_policy
        is not FunctionalPFieldPolicy.NONE
    ):
        replay_entry = _controller_replay_entry(
            model,
            tokenizer,
            alias=alias,
            arm_state=arm_state,
            sample_waypoint=1,
            factors=current_factors,
            request_by_sha256=request_by_sha256,
            population_by_sha256=population_by_sha256,
            schedule=schedule,
            outer_entry_p_cache=outer_entry_p_cache,
        )
    try:
        active = _build_active_field(
            model,
            tokenizer,
            requests,
            variant=variant,
            accepted_index=0,
            factors=current_factors,
            target_state=current_target,
            capture=capture,
            hparams=hparams,
            projector=projector,
            contexts=contexts,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            lock=lock,
            history=history,
            accepted_by_layer=accepted_by_layer,
            ledger=ledger,
            recorder=recorder,
            native_target=native_target,
            target_base=target_base,
            routing_objective=selected_objective,
            preservation_policy=selected_preservation_policy,
            functional_p_field_policy=(
                selected_functional_p_field_policy
            ),
            functional_p_replay_entry=replay_entry,
            theta0_cache=theta0_cache,
            alias=alias,
            touched=touched,
            schedule=schedule,
        )
        field_build_count = 1
        initial_layer_routing = dict(active.layer_routing_payload)
        if replay_entry is not None:
            replay_field_sha256 = active.field.identity_sha256
    except ODEBFContractError as exc:
        if not _is_progress_infeasible(exc):
            raise
        status = "PROGRESS_INFEASIBLE"
        field_build_count = len(recorder.field_hashes)

    if variant is AdaptiveVariant.PS_S8:
        for slot in range(variant_lock.legacy_slot_cap):
            if active is None:
                break
            replay_entry = _controller_replay_entry(
                model,
                tokenizer,
                alias=alias,
                arm_state=arm_state,
                sample_waypoint=slot + 1,
                factors=current_factors,
                request_by_sha256=request_by_sha256,
                population_by_sha256=population_by_sha256,
                schedule=schedule,
                outer_entry_p_cache=outer_entry_p_cache,
            )
            trial_receipts: list[str] = []
            trial_outcomes: list[TrialOutcome] = []
            chosen: TrialOutcome | None = None
            chosen_delta: Fraction | None = None
            for trial_ordinal, beta in enumerate(lock.backtracking):
                n_trial += 1
                delta = H_REF * Fraction(str(beta))
                outcome = _run_trial(
                    model,
                    tokenizer,
                    requests,
                    alias=alias,
                    variant=variant,
                    accepted_index=len(snapshots),
                    n_trial=n_trial,
                    retry_index=trial_ordinal,
                    tau_before=accepted_t,
                    delta_tau=delta,
                    delta_tau_proposed=H_REF,
                    remainder=False,
                    active=active,
                    current_factors=current_factors,
                    current_target=current_target,
                    current_margin=current_margin,
                    replay_entry=replay_entry,
                    theta0_cache=theta0_cache,
                    capture=capture,
                    lock=lock,
                    ledger=ledger,
                    recorder=recorder,
                    touched=touched,
                    history=history,
                    schedule=schedule,
                    previous_bf_velocity=previous_bf_velocity,
                    accepted_by_layer=accepted_by_layer,
                    contexts=contexts,
                    routing_objective=selected_objective,
                    functional_p_policy=selected_functional_p_policy,
                    preservation_policy=selected_preservation_policy,
                )
                trial_receipts.append(outcome.receipt_sha256)
                trial_outcomes.append(outcome)
                if not outcome.gate_accepted:
                    n_reject += 1
                elif chosen is None:
                    chosen = outcome
                    chosen_delta = delta
            transition_sha256 = recorder.transition(
                {
                    "mode": "legacy-fixed-slot",
                    "slot": slot,
                    "slot_consumed": True,
                    "trial_receipt_sha256": trial_receipts,
                    "accepted": chosen is not None,
                    "accepted_index_before": len(snapshots),
                    "tau_before": fraction_payload(accepted_t),
                    "rejected_slot_state_unchanged": chosen is None,
                    "field_reused_after_reject": chosen is None,
                    "layer_routing": relabel_layer_routing_telemetry(
                        (chosen or trial_outcomes[-1]).capacity_payload[
                            "layer_routing"
                        ],
                        stage=(
                            "ACCEPTED_TRANSITION"
                            if chosen is not None
                            else "REJECTED_TRANSITION"
                        ),
                    ),
                }
            )
            if chosen is None:
                ledger.increment("reject", len(lock.backtracking))
                continue
            assert chosen_delta is not None
            accepted_t += chosen_delta
            snapshot = _append_accepted_snapshot(
                outcome=chosen,
                active=active,
                accepted_index_before=len(snapshots),
                tau_after=accepted_t,
                delta_tau=chosen_delta,
                transition_sha256=transition_sha256,
                accepted_by_layer=accepted_by_layer,
                ledger=ledger,
                recorder=recorder,
                first_hit=first_hit,
            )
            snapshots.append(snapshot)
            current_factors = _factor_map(chosen.candidate_factors)
            current_target = chosen.target_trial.clone()
            current_margin = chosen.margin
            previous_bf_velocity = active.bf_velocity.velocity
            if slot + 1 < variant_lock.legacy_slot_cap:
                try:
                    active = _build_active_field(
                        model,
                        tokenizer,
                        requests,
                        variant=variant,
                        accepted_index=len(snapshots),
                        factors=current_factors,
                        target_state=current_target,
                        capture=capture,
                        hparams=hparams,
                        projector=projector,
                        contexts=contexts,
                        covariance_registry=covariance_registry,
                        projector_sha256=projector_sha256,
                        lock=lock,
                        history=history,
                        accepted_by_layer=accepted_by_layer,
                        ledger=ledger,
                        recorder=recorder,
                        native_target=native_target,
                        target_base=target_base,
                        routing_objective=selected_objective,
                        preservation_policy=selected_preservation_policy,
                    )
                    field_build_count += 1
                except ODEBFContractError as exc:
                    if not _is_progress_infeasible(exc):
                        raise
                    status = "PROGRESS_INFEASIBLE"
                    field_build_count = len(recorder.field_hashes)
                    active = None
                    break
        if status == "ACTIVE":
            status = "LEGACY_FIXED_S8_COMPLETE"
    else:
        clock = AdaptiveTauClock(variant_lock)
        while active is not None and clock.status == "ACTIVE" and not clock.complete:
            if replay_entry is None:
                replay_entry = _controller_replay_entry(
                    model,
                    tokenizer,
                    alias=alias,
                    arm_state=arm_state,
                    sample_waypoint=len(snapshots) + 1,
                    factors=current_factors,
                    request_by_sha256=request_by_sha256,
                    population_by_sha256=population_by_sha256,
                    schedule=schedule,
                    outer_entry_p_cache=outer_entry_p_cache,
                )
                replay_field_sha256 = active.field.identity_sha256
            try:
                trial = clock.begin_trial()
            except ODEBFStateError:
                if clock.status == "ACTIVE":
                    raise
                break
            outcome = _run_trial(
                model,
                tokenizer,
                requests,
                alias=alias,
                variant=variant,
                accepted_index=len(snapshots),
                n_trial=trial.n_trial,
                retry_index=trial.retry_index,
                tau_before=trial.tau_before,
                delta_tau=trial.delta_tau_trial,
                delta_tau_proposed=trial.delta_tau_proposed,
                remainder=trial.remainder,
                active=active,
                current_factors=current_factors,
                current_target=current_target,
                current_margin=current_margin,
                replay_entry=replay_entry,
                theta0_cache=theta0_cache,
                capture=capture,
                lock=lock,
                ledger=ledger,
                recorder=recorder,
                touched=touched,
                history=history,
                schedule=schedule,
                previous_bf_velocity=previous_bf_velocity,
                accepted_by_layer=accepted_by_layer,
                contexts=contexts,
                routing_objective=selected_objective,
                functional_p_policy=selected_functional_p_policy,
                preservation_policy=selected_preservation_policy,
            )
            n_trial = clock.n_trial
            if outcome.gate_accepted:
                if outcome.trust_ratio is None:
                    raise ODEBFContractError("accepted adaptive trial lacks rho")
                transition = clock.accept(trial=trial, rho=outcome.trust_ratio)
            else:
                transition = clock.reject(trial=trial)
                n_reject = clock.n_reject
                ledger.increment("reject")
            transition_layer_routing = relabel_layer_routing_telemetry(
                outcome.capacity_payload["layer_routing"],
                stage=(
                    "ACCEPTED_TRANSITION"
                    if outcome.gate_accepted
                    else "REJECTED_TRANSITION"
                ),
            )
            transition_sha256 = recorder.transition(
                {
                    "mode": "adaptive-pseudo-time",
                    "trial_receipt_sha256": outcome.receipt_sha256,
                    "clock_trial": trial.raw_free_payload(),
                    "clock_transition": transition.raw_free_payload(),
                    "field_sha256": active.field.identity_sha256,
                    "replay_field_sha256": replay_field_sha256,
                    "same_state_retry": not outcome.gate_accepted,
                    "trust_radius_fixed_on_reject": True,
                    "layer_routing": transition_layer_routing,
                }
            )
            if not outcome.gate_accepted:
                if clock.status != "ACTIVE":
                    break
                continue
            accepted_t = clock.tau
            snapshot = _append_accepted_snapshot(
                outcome=outcome,
                active=active,
                accepted_index_before=len(snapshots),
                tau_after=clock.tau,
                delta_tau=trial.delta_tau_trial,
                transition_sha256=transition_sha256,
                accepted_by_layer=accepted_by_layer,
                ledger=ledger,
                recorder=recorder,
                first_hit=first_hit,
            )
            snapshots.append(snapshot)
            current_factors = _factor_map(outcome.candidate_factors)
            current_target = outcome.target_trial.clone()
            current_margin = outcome.margin
            previous_bf_velocity = active.bf_velocity.velocity
            replay_entry = None
            replay_field_sha256 = None
            if not clock.complete:
                if (
                    selected_functional_p_field_policy
                    is not FunctionalPFieldPolicy.NONE
                ):
                    replay_entry = _controller_replay_entry(
                        model,
                        tokenizer,
                        alias=alias,
                        arm_state=arm_state,
                        sample_waypoint=len(snapshots) + 1,
                        factors=current_factors,
                        request_by_sha256=request_by_sha256,
                        population_by_sha256=population_by_sha256,
                        schedule=schedule,
                        outer_entry_p_cache=outer_entry_p_cache,
                    )
                try:
                    active = _build_active_field(
                        model,
                        tokenizer,
                        requests,
                        variant=variant,
                        accepted_index=len(snapshots),
                        factors=current_factors,
                        target_state=current_target,
                        capture=capture,
                        hparams=hparams,
                        projector=projector,
                        contexts=contexts,
                        covariance_registry=covariance_registry,
                        projector_sha256=projector_sha256,
                        lock=lock,
                        history=history,
                        accepted_by_layer=accepted_by_layer,
                        ledger=ledger,
                        recorder=recorder,
                        native_target=native_target,
                        target_base=target_base,
                        routing_objective=selected_objective,
                        preservation_policy=selected_preservation_policy,
                        functional_p_field_policy=(
                            selected_functional_p_field_policy
                        ),
                        functional_p_replay_entry=replay_entry,
                        theta0_cache=theta0_cache,
                        alias=alias,
                        touched=touched,
                        schedule=schedule,
                    )
                    field_build_count += 1
                    if replay_entry is not None:
                        replay_field_sha256 = active.field.identity_sha256
                except ODEBFContractError as exc:
                    if not _is_progress_infeasible(exc):
                        raise
                    status = "PROGRESS_INFEASIBLE"
                    field_build_count = len(recorder.field_hashes)
                    active = None
                    break
        if status == "ACTIVE":
            status = clock.status
        n_trial = clock.n_trial
        n_reject = clock.n_reject
        accepted_t = clock.tau

    if ledger.completed_correction_cycles == 0:
        ledger.finish_cycle(0)
    trajectory_complete = (
        status == "LEGACY_FIXED_S8_COMPLETE"
        if variant is AdaptiveVariant.PS_S8
        else status == "TAU_COMPLETE" and accepted_t == Fraction(1, 1)
    )
    confirmations = _terminal_confirm_snapshots(
        model,
        tokenizer,
        alias=alias,
        arm_state=arm_state,
        snapshots=snapshots,
        request_by_sha256=request_by_sha256,
        population_by_sha256=population_by_sha256,
        schedule=schedule,
        outer_entry_p_cache=outer_entry_p_cache,
        theta0_cache=theta0_cache,
        lock=lock,
        ledger=ledger,
        recorder=recorder,
        trajectory_complete=trajectory_complete,
        functional_p_policy=selected_functional_p_policy,
        preservation_policy=selected_preservation_policy,
    )
    if (
        _parameter_contract_sha256(touched) != entry_model_sha256
        or history.snapshot().digest != entry_history_sha256
        or schedule.state_digest != entry_sampler_sha256
    ):
        raise ODEBFStateError(
            "adaptive rollout mutated model/history/sampler/RNG"
        )
    if field_build_count != len(recorder.field_hashes):
        raise ODEBFStateError("adaptive field receipt/build count differs")
    terminal_hits = [
        item for item in confirmations if item["terminal_confirmed_exact_hit"]
    ]
    termination_label = _termination_label(
        status=status,
        exact_hit=bool(terminal_hits),
    )
    if initial_layer_routing is None:
        layer_routing_trajectory: dict[str, Any] = {
            "status": "NOT_AVAILABLE_NO_FEASIBLE_FIELD",
            "observation_only": True,
            "controller_dependency_count": 0,
        }
    else:
        layer_routing_rows = [initial_layer_routing]
        layer_routing_rows.extend(
            relabel_layer_routing_telemetry(
                snapshot.capacity_payload["layer_routing"],
                stage="ACCEPTED_TRANSITION",
            )
            for snapshot in snapshots
        )
        first_hit_step_index = (
            None
            if first_hit.first_online is None
            else int(
                snapshots[
                    first_hit.first_online.accepted_index - 1
                ].capacity_payload["layer_routing"]["step_index"]
            )
        )
        layer_routing_trajectory = summarize_layer_routing_trajectory(
            layer_routing_rows,
            first_hit_step_index=first_hit_step_index,
        )
    rollout_payload = {
        "variant": selected_label,
        "status": status,
        "termination_label": termination_label,
        "residual_policy": _residual_policy(variant),
        "adaptive_lock_sha256": variant_lock.identity(),
        "accepted_t": fraction_payload(accepted_t),
        "k_acc": len(snapshots),
        "n_trial": n_trial,
        "n_reject": n_reject,
        "field_build_count": field_build_count,
        "entry_success": entry_eval.batch_success.raw_free_payload(),
        "online_first_hit": (
            None
            if first_hit.first_online is None
            else {
                **asdict(first_hit.first_online),
                "tau": fraction_payload(first_hit.first_online.tau),
            }
        ),
        "terminal_first_hit": (
            None
            if not terminal_hits
            else {
                "accepted_index": terminal_hits[0]["accepted_index"],
                "tau": terminal_hits[0]["tau"],
                "snapshot_sha256": terminal_hits[0]["snapshot_sha256"],
            }
        ),
        "accepted_snapshot_sha256": [
            item.snapshot_sha256 for item in snapshots
        ],
        "receipt_links": recorder.links(),
        "layer_routing_trajectory": layer_routing_trajectory,
        "compute": ledger.raw_free_payload(),
        "persistent_commit_count": 0,
        "history_append_count": 0,
        "heldout_access_count": 0,
        "functional_p_decision_policy": selected_functional_p_policy.value,
        "functional_p_decision_influence_count": (
            n_trial
            if selected_functional_p_policy is FunctionalPDecisionPolicy.PCTRL
            else 0
        ),
        "functional_p_field_policy": (
            selected_functional_p_field_policy.value
        ),
        "functional_p_field_decision_influence_count": (
            field_build_count
            if selected_functional_p_field_policy
            is FunctionalPFieldPolicy.SOFT_HARD
            else 0
        ),
        "functional_p_probe_field_count": (
            field_build_count
            if selected_functional_p_field_policy
            is not FunctionalPFieldPolicy.NONE
            else 0
        ),
        "preservation_constraint_policy": selected_preservation_policy.value,
        "preservation_decision_influence_count": {
            name: (
                n_trial
                if selected_preservation_policy
                is PreservationConstraintPolicy.LOCKED
                else 0
            )
            for name in (
                "structural_h", "structural_p", "trust", "functional_h"
            )
        },
    }
    if (
        selected_objective is not RoutingObjective.MARGIN
        or selected_label != variant.value
    ):
        rollout_payload.update(
            {
                "clock_variant": variant.value,
                "routing_objective": selected_objective.value,
                "target_state_objective": "MARGIN_LOCKED",
                "target_old_decision_influence_count": 0
                if selected_objective is RoutingObjective.TARGET_NEW_NLL
                else BATCH_SIZE,
            }
        )
    rollout_sha256 = canonical_hash(rollout_payload)
    recorder.terminal(
        {
            "rollout_summary": rollout_payload,
            "rollout_sha256": rollout_sha256,
            "trajectory_complete": trajectory_complete,
            "terminal_confirmation_count": len(confirmations),
        }
    )
    return VariantRollout(
        variant,
        status,
        termination_label,
        _residual_policy(variant),
        accepted_t,
        len(snapshots),
        n_trial,
        n_reject,
        field_build_count,
        snapshots,
        first_hit,
        recorder,
        ledger,
        entry_eval.batch_success.raw_free_payload(),
        confirmations,
        rollout_sha256,
        selected_label,
        selected_objective.value,
        selected_functional_p_policy.value,
        selected_preservation_policy.value,
        selected_functional_p_field_policy.value,
    )


def _evaluate_stepwise_state(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[Any],
    *,
    alias: str,
    freeze: StepwiseActionFreeze,
    ledger: ComputeLedger,
    touched: Mapping[str, torch.nn.Parameter],
    factors: Mapping[str, Sequence[WaypointFactor]] | None = None,
    candidates: Mapping[str, torch.Tensor] | None = None,
) -> tuple[StepwisePrimaryReceipt, dict[str, Any]]:
    if factors and candidates:
        raise ODEBFContractError("stepwise state has two virtual endpoint sources")
    before = _parameter_contract_sha256(touched)
    started = time.perf_counter()
    liveness: dict[str, Any]
    if candidates:
        context = CandidateBF16FunctionalTrial(model, candidates)
        with context:
            receipt = evaluate_counterfact_stepwise_primary(
                model, tokenizer, cases, model_alias=alias, freeze=freeze
            )
        liveness = {
            "candidate_bf16_weight_peak_live": context.max_live_candidate_weights,
            "effective_bf16_weight_peak_live": 0,
            "maximum_fp32_block_elements": 0,
        }
    elif factors:
        context = CumulativeBF16FunctionalTrial(model, factors, row_block=64)
        with context:
            receipt = evaluate_counterfact_stepwise_primary(
                model, tokenizer, cases, model_alias=alias, freeze=freeze
            )
        liveness = {
            "candidate_bf16_weight_peak_live": 0,
            "effective_bf16_weight_peak_live": context.max_live_effective_weights,
            "maximum_fp32_block_elements": context.max_fp32_block_elements,
        }
    else:
        receipt = evaluate_counterfact_stepwise_primary(
            model, tokenizer, cases, model_alias=alias, freeze=freeze
        )
        liveness = {
            "candidate_bf16_weight_peak_live": 0,
            "effective_bf16_weight_peak_live": 0,
            "maximum_fp32_block_elements": 0,
        }
    wall = time.perf_counter() - started
    after = _parameter_contract_sha256(touched)
    if after != before:
        raise ODEBFStateError(
            "post-freeze evaluator mutated parameter/RNG contract"
        )
    if (
        liveness["candidate_bf16_weight_peak_live"] > 1
        or liveness["effective_bf16_weight_peak_live"] > 1
    ):
        raise ODEBFStateError("stepwise evaluator retained multiple target weights")
    ledger.increment("evaluator_forward", receipt.primary.model_forward_count)
    ledger.increment("evaluator_tokens", receipt.primary.processed_token_count)
    ledger.add_time("postfreeze_primary", wall_seconds=wall)
    return receipt, {
        "parameter_rng_before_sha256": before,
        "parameter_rng_after_sha256": after,
        "wall_seconds": wall,
        "model_forward_count": receipt.primary.model_forward_count,
        "processed_token_count": receipt.primary.processed_token_count,
        "generation_call_count": receipt.primary.generation_call_count,
        **liveness,
    }


def _postfreeze_stepwise_panel(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    requests: Sequence[Mapping[str, Any]],
    dataset_path: Path,
    capture: P1NativeCapture,
    rollouts: Mapping[Any, VariantRollout],
    raw_root: Path,
    write_once: Any,
    touched: Mapping[str, torch.nn.Parameter],
    instruction_id: str = ADAPTIVE_INSTRUCTION_ID,
    schema_namespace: str = "ode-edit-s04-ode-bf-p1r4-adaptive",
) -> tuple[dict[str, Any], dict[tuple[str, int], StepwisePrimaryReceipt]]:
    """Open held-out CounterFact surfaces only after every action is frozen."""

    freeze_payload = {
        "schema": f"{schema_namespace}-action-freeze/v1",
        "instruction_id": instruction_id,
        "action_frozen_before_open": True,
        "request_order_sha256": capture.request_order_sha256,
        "rollouts": {
            rollout.variant_label or rollout.variant.value: {
                "rollout_sha256": rollout.rollout_sha256,
                "status": rollout.status,
                "termination_label": rollout.termination_label,
                "accepted_snapshot_sha256": [
                    item.snapshot_sha256 for item in rollout.snapshots
                ],
                "n_reject": rollout.n_reject,
            }
            for rollout in rollouts.values()
        },
        "heldout_controller_access_count": 0,
        "model_generate_call_count": 0,
    }
    freeze_root_sha256 = write_once(
        raw_root / "stepwise" / "action-freeze.json", freeze_payload
    )
    w0_snapshot_sha256 = canonical_hash(
        {"entry_weights": dict(sorted(capture.entry_sha256.items()))}
    )
    native_snapshot_sha256 = canonical_hash(
        {
            "native_candidates": {
                name: tensor_sha256(value)
                for name, value in sorted(capture.native_candidates.items())
            }
        }
    )
    w0_freeze = StepwiseActionFreeze(
        variant="W0_NO_EDIT",
        request_order_sha256=capture.request_order_sha256,
        rollout_sha256=freeze_root_sha256,
        snapshot_sha256=w0_snapshot_sha256,
        snapshot_index=0,
        accepted_snapshot_count=0,
        rejected_retry_count=0,
        trajectory_status="COMMON_W0_REFERENCE",
    )
    # The loader checks the action-frozen/request-order contract and opens only
    # the ten sealed cases.  No held-out object exists before this line.
    cases = load_counterfact_cases_after_freeze(
        dataset_path, requests, w0_freeze
    )
    ledger = ComputeLedger()
    counter = ModelForwardCounter(model, ledger)
    receipts: dict[tuple[str, int], StepwisePrimaryReceipt] = {}
    receipt_hashes: dict[str, str] = {}
    try:
        w0, w0_compute = _evaluate_stepwise_state(
            model,
            tokenizer,
            cases,
            alias=alias,
            freeze=w0_freeze,
            ledger=ledger,
            touched=touched,
        )
        receipts[("W0_NO_EDIT", 0)] = w0
        receipt_hashes["W0_NO_EDIT"] = write_once(
            raw_root / "stepwise" / "W0_NO_EDIT.json",
            {
                "schema": f"{schema_namespace}-step-receipt/v1",
                "variant": "W0_NO_EDIT",
                "snapshot_index": 0,
                "snapshot_sha256": w0_snapshot_sha256,
                "freeze_sha256": w0_freeze.identity(),
                "primary": w0.raw_free_payload(),
                "compute": w0_compute,
            },
        )
        native_freeze = StepwiseActionFreeze(
            variant="N32_NATIVE",
            request_order_sha256=capture.request_order_sha256,
            rollout_sha256=freeze_root_sha256,
            snapshot_sha256=native_snapshot_sha256,
            snapshot_index=1,
            accepted_snapshot_count=1,
            rejected_retry_count=0,
            trajectory_status="COMMON_N32_REFERENCE",
        )
        native, native_compute = _evaluate_stepwise_state(
            model,
            tokenizer,
            cases,
            alias=alias,
            freeze=native_freeze,
            ledger=ledger,
            touched=touched,
            candidates=capture.native_candidates,
        )
        receipts[("N32_NATIVE", 1)] = native
        receipt_hashes["N32_NATIVE"] = write_once(
            raw_root / "stepwise" / "N32_NATIVE.json",
            {
                "schema": f"{schema_namespace}-step-receipt/v1",
                "variant": "N32_NATIVE",
                "snapshot_index": 1,
                "snapshot_sha256": native_snapshot_sha256,
                "freeze_sha256": native_freeze.identity(),
                "primary": native.raw_free_payload(),
                "compute": native_compute,
            },
        )
        for rollout in rollouts.values():
            variant_label = rollout.variant_label or rollout.variant.value
            confirmations = {
                item["snapshot_sha256"]: item
                for item in rollout.terminal_confirmation
            }
            for snapshot in rollout.snapshots:
                freeze = StepwiseActionFreeze(
                    variant=variant_label,
                    request_order_sha256=capture.request_order_sha256,
                    rollout_sha256=rollout.rollout_sha256,
                    snapshot_sha256=snapshot.snapshot_sha256,
                    snapshot_index=snapshot.accepted_index,
                    accepted_snapshot_count=len(rollout.snapshots),
                    rejected_retry_count=rollout.n_reject,
                    trajectory_status=rollout.status,
                )
                observed, compute = _evaluate_stepwise_state(
                    model,
                    tokenizer,
                    cases,
                    alias=alias,
                    freeze=freeze,
                    ledger=ledger,
                    touched=touched,
                    factors=snapshot.factors,
                )
                comparison = compare_stepwise_primary(w0, native, observed)
                key = (variant_label, snapshot.accepted_index)
                receipts[key] = observed
                relative = (
                    Path("stepwise")
                    / variant_label
                    / f"accepted-{snapshot.accepted_index:04d}.json"
                )
                receipt_hashes[
                    f"{variant_label}:{snapshot.accepted_index}"
                ] = write_once(
                    raw_root / relative,
                    {
                        "schema": f"{schema_namespace}-step-receipt/v1",
                        "variant": variant_label,
                        "accepted_index": snapshot.accepted_index,
                        "tau": fraction_payload(snapshot.tau),
                        "delta_tau": fraction_payload(snapshot.delta_tau),
                        "snapshot_sha256": snapshot.snapshot_sha256,
                        "freeze_sha256": freeze.identity(),
                        "primary": observed.raw_free_payload(),
                        "comparison_to_w0_native": comparison,
                        "terminal_components": confirmations.get(
                            snapshot.snapshot_sha256
                        ),
                        "functional_p_decision": dict(
                            snapshot.functional_p_decision
                        ),
                        "routing": snapshot.routing_payload,
                        "capacity": snapshot.capacity_payload,
                        "compute": compute,
                        "action_frozen_before_open": True,
                        "controller_heldout_access_count": 0,
                    },
                )
    finally:
        counter.close()
    from .p1_runtime import _observed_memory

    _observed_memory(ledger)
    panel = {
        "schema": f"{schema_namespace}-stepwise-panel/v1",
        "action_freeze_sha256": freeze_root_sha256,
        "request_order_sha256": capture.request_order_sha256,
        "receipt_sha256": dict(sorted(receipt_hashes.items())),
        "unique_accepted_snapshot_count": sum(
            len(item.snapshots) for item in rollouts.values()
        ),
        "rejected_retry_alias_count": sum(
            item.n_reject for item in rollouts.values()
        ),
        "rejected_retry_duplicate_evaluation_count": 0,
        "heldout_controller_access_count": 0,
        "action_frozen_before_open": True,
        "generation_call_count": 0,
        "evaluation_compute": ledger.raw_free_payload(),
    }
    panel["panel_sha256"] = canonical_hash(panel)
    return panel, receipts


def _refinement_payload(
    capture: P1NativeCapture,
    rollouts: Mapping[AdaptiveVariant, VariantRollout],
    receipts: Mapping[tuple[str, int], StepwisePrimaryReceipt],
) -> dict[str, Any]:
    left = rollouts[AdaptiveVariant.FR_A8]
    right = rollouts[AdaptiveVariant.FR_A16]
    if (
        not left.snapshots
        or not right.snapshots
        or left.accepted_t != Fraction(1, 1)
        or right.accepted_t != Fraction(1, 1)
    ):
        return {
            "status": "REFINEMENT_ENDPOINT_UNAVAILABLE",
            "fr_a8_status": left.status,
            "fr_a16_status": right.status,
            "fr_a8_tau": fraction_payload(left.accepted_t),
            "fr_a16_tau": fraction_payload(right.accepted_t),
            "ode_claim": "HOLD_PILOT",
        }
    left_snapshot = left.snapshots[-1]
    right_snapshot = right.snapshots[-1]
    distance_by_weight: dict[str, Any] = {}
    squared = 0.0
    maximum = 0.0
    for name, entry in sorted(capture.entry_weights.items()):
        receipt = streaming_bf16_endpoint_distance(
            entry,
            left_snapshot.factors.get(name, ()),
            right_snapshot.factors.get(name, ()),
            weight_name=name,
        )
        distance_by_weight[name] = asdict(receipt)
        squared += receipt.frobenius_distance**2
        maximum = max(maximum, receipt.maximum_absolute_distance)
    left_primary = receipts[
        (AdaptiveVariant.FR_A8.value, left_snapshot.accepted_index)
    ]
    right_primary = receipts[
        (AdaptiveVariant.FR_A16.value, right_snapshot.accepted_index)
    ]
    # Treat FR-A8 as both W0/reference only for direct FR-A16-minus-FR-A8
    # paired/NLL diagnostics; the production W0/Native comparisons remain in
    # each step receipt.
    pairwise = compare_stepwise_primary(
        left_primary, left_primary, right_primary
    )
    left_hit = next(
        (
            item
            for item in left.terminal_confirmation
            if item["terminal_confirmed_exact_hit"]
        ),
        None,
    )
    right_hit = next(
        (
            item
            for item in right.terminal_confirmation
            if item["terminal_confirmed_exact_hit"]
        ),
        None,
    )
    return {
        "status": "REFINEMENT_REPORTED_NO_PROMOTION",
        "fr_a8_snapshot_sha256": left_snapshot.snapshot_sha256,
        "fr_a16_snapshot_sha256": right_snapshot.snapshot_sha256,
        "bf16_endpoint_frobenius_distance": math.sqrt(squared),
        "bf16_endpoint_max_abs_distance": maximum,
        "distance_by_weight": distance_by_weight,
        "fr_a16_minus_fr_a8": pairwise,
        "first_terminal_hit_tau": {
            "FR-A8": None if left_hit is None else left_hit["tau"],
            "FR-A16": None if right_hit is None else right_hit["tau"],
        },
        "terminal_h_p_capacity": {
            "FR-A8": {
                "components": left.terminal_confirmation[-1],
                "capacity": left_snapshot.capacity_payload,
            },
            "FR-A16": {
                "components": right.terminal_confirmation[-1],
                "capacity": right_snapshot.capacity_payload,
            },
        },
        "material_stability_decision": "GH_REVIEW_REQUIRED",
        "ode_claim": "HOLD_PILOT",
    }


def _target_new_panel_contrast(
    capture: P1NativeCapture,
    rollouts: Mapping[Any, VariantRollout],
    receipts: Mapping[tuple[str, int], StepwisePrimaryReceipt],
    *,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left = rollouts[left_label]
    right = rollouts[right_label]
    if not left.snapshots or not right.snapshots:
        return {
            "status": "ENDPOINT_UNAVAILABLE",
            "left": left_label,
            "right": right_label,
            "left_status": left.status,
            "right_status": right.status,
        }
    left_snapshot = left.snapshots[-1]
    right_snapshot = right.snapshots[-1]
    distance_by_weight: dict[str, Any] = {}
    squared = 0.0
    maximum = 0.0
    for name, entry in sorted(capture.entry_weights.items()):
        receipt = streaming_bf16_endpoint_distance(
            entry,
            left_snapshot.factors.get(name, ()),
            right_snapshot.factors.get(name, ()),
            weight_name=name,
        )
        distance_by_weight[name] = asdict(receipt)
        squared += receipt.frobenius_distance**2
        maximum = max(maximum, receipt.maximum_absolute_distance)
    left_primary = receipts[(left_label, left_snapshot.accepted_index)]
    right_primary = receipts[(right_label, right_snapshot.accepted_index)]
    return {
        "status": "REPORTED_NO_PROMOTION",
        "left": left_label,
        "right": right_label,
        "left_snapshot_sha256": left_snapshot.snapshot_sha256,
        "right_snapshot_sha256": right_snapshot.snapshot_sha256,
        "bf16_endpoint_frobenius_distance": math.sqrt(squared),
        "bf16_endpoint_max_abs_distance": maximum,
        "distance_by_weight": distance_by_weight,
        "right_minus_left": compare_stepwise_primary(
            left_primary, left_primary, right_primary
        ),
        "left_routing_objective": left.routing_objective,
        "right_routing_objective": right.routing_objective,
        "left_terminal_components": left.terminal_confirmation[-1],
        "right_terminal_components": right.terminal_confirmation[-1],
        "left_terminal_capacity": left_snapshot.capacity_payload,
        "right_terminal_capacity": right_snapshot.capacity_payload,
    }


def _target_new_panel_payload(
    capture: P1NativeCapture,
    rollouts: Mapping[Any, VariantRollout],
    receipts: Mapping[tuple[str, int], StepwisePrimaryReceipt],
) -> dict[str, Any]:
    required = (
        "FR-A8-MARGIN",
        "FR-A8-NEWNLL",
    )
    if tuple(rollouts) != required:
        raise ODEBFContractError("target-new routing panel order differs")
    return {
        "schema": "ode-edit-s05-target-new-nll-routing-contrasts/v1",
        "routing_loss_change_only": True,
        "target_z_velocity_objective": "MARGIN_LOCKED",
        "fr_a8_newnll_minus_margin": _target_new_panel_contrast(
            capture,
            rollouts,
            receipts,
            left_label="FR-A8-MARGIN",
            right_label="FR-A8-NEWNLL",
        ),
        "fr_a16_newnll": {
            "executed": False,
            "forecast_seconds": 161920,
            "allocation_seconds": 86400,
            "exclusion_reason": "SIX_CONTEXT_A16_EXCEEDS_LOCKED_24H_ENVELOPE",
            "outcome_metric_used": False,
        },
        "scientific_promotion_authorized": False,
    }


def run_adaptive_diagnostic(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    destination: Path,
    raw_root: Path,
    stages: Any,
    source_head: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    stream: Mapping[str, Any],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    controller_lock: P1ControllerLock,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: StatelessReplaySchedule,
    theta0_cache: Theta0TeacherCache,
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    artifact_guard: Any,
    artifact_receipt: Any,
    numerical_sha256: str,
    context_sha256: str,
    cuda_runtime_receipt: Mapping[str, Any],
    job_ledger: ComputeLedger,
    write_once: Any,
    panel_specs: Sequence[AdaptivePanelSpec] | None = None,
    panel_instruction_id: str = ADAPTIVE_INSTRUCTION_ID,
    panel_schema_namespace: str = "ode-edit-s04-ode-bf-p1r4-adaptive",
    panel_terminal_status: str = "CAUSAL_DIAGNOSTIC_COMPLETE_NO_PROMOTION",
    panel_refinement_builder: Callable[
        [
            P1NativeCapture,
            Mapping[Any, VariantRollout],
            Mapping[tuple[str, int], StepwisePrimaryReceipt],
        ],
        dict[str, Any],
    ]
    | None = None,
    rollout_validator: Callable[
        [str, AdaptivePanelSpec, VariantRollout], None
    ]
    | None = None,
    panel_terminal_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the single authorized reused-seal B10 causal diagnostic."""

    from .p1_runtime import (
        ArmRuntimeState,
        _entry_parameter_snapshot_sha256,
        _evaluate_native_rewrite,
        _observed_memory,
    )

    specs = _default_panel_specs() if panel_specs is None else tuple(panel_specs)
    custom_panel = panel_specs is not None
    if (
        not specs
        or len({item.label for item in specs}) != len(specs)
        or not panel_instruction_id
        or not panel_schema_namespace
        or not panel_terminal_status
    ):
        raise ODEBFContractError("adaptive panel specification differs")
    if len(stream_batches) != 4 or len(stream_batches[0]) != BATCH_SIZE:
        raise ODEBFContractError("adaptive diagnostic reused stream differs")
    requests = tuple(stream_batches[0])
    if capture_order := stream.get("batch_ordered_request_digest_v1"):
        if len(capture_order) != 4:
            raise ODEBFContractError("adaptive reused stream digest count differs")
    else:
        raise ODEBFContractError("adaptive reused stream lacks ordered digests")
    base_bytes = {
        name: tensor_sha256(parameter) for name, parameter in sorted(touched.items())
    }
    if tuple(sorted(base_bytes.items())) != tuple(base_receipt.parameter_sha256):
        raise ODEBFStateError("adaptive runtime did not start at W0 bytes")
    native_history = P1HistoryLedger(
        layer_order=tuple(int(layer) for layer in hparams.layers),
        maximum_records=40,
    )
    native_ledger = ComputeLedger()
    native_counter = ModelForwardCounter(model, native_ledger)
    try:
        capture = capture_p1_native_entry(
            model,
            tokenizer,
            requests,
            hparams,
            projector,
            contexts,
            history_keys_by_layer=_history_keys(
                native_history,
                tuple(int(layer) for layer in hparams.layers),
                risk=False,
            ),
            mutation_lock=mutation_lock,
            ledger=native_ledger,
            residual_tolerance=controller_lock.residual_tolerance,
        )
        if capture.request_order_sha256 != capture_order[0]:
            raise ODEBFContractError(
                "adaptive capture/reused seal request digest differs"
            )
        native_online = _evaluate_native_rewrite(
            model,
            tokenizer,
            requests,
            alias=alias,
            candidates=capture.native_candidates,
            ledger=native_ledger,
        )
    finally:
        native_counter.close()
    after_capture_contract = _parameter_contract_sha256(touched)
    if {
        name: tensor_sha256(parameter)
        for name, parameter in sorted(touched.items())
    } != base_bytes:
        raise ODEBFStateError("adaptive N32 capture did not restore W0 bytes")
    _observed_memory(native_ledger)
    n32_sha256 = write_once(
        raw_root / "adaptive" / "N32_NATIVE-online.json",
        {
            "schema": f"{panel_schema_namespace}-native/v1",
            "instruction_id": panel_instruction_id,
            "status": "VIRTUAL_ONLY_NO_COMMIT",
            "alias": alias,
            "request_order_sha256": capture.request_order_sha256,
            "joint_rank": BATCH_SIZE,
            "capture": capture.raw_free_payload(),
            "official_success": native_online.batch_success.raw_free_payload(),
            "compute": native_ledger.raw_free_payload(),
            "persistent_commit_count": 0,
            "history_append_count": 0,
            "heldout_access_count": 0,
        },
    )
    outer_population = tuple(
        population_by_sha256[item] for item in theta0_cache.request_order
    )
    outer_entry_snapshot_sha256 = _entry_parameter_snapshot_sha256(
        model, capture.entry_sha256
    )
    outer_counter = ModelForwardCounter(model, job_ledger)
    outer_started = time.perf_counter()
    try:
        outer_entry_p_cache = build_outer_entry_pretrained_cache(
            model,
            tokenizer,
            outer_population,
            theta0_cache,
            outer_entry_snapshot_sha256=outer_entry_snapshot_sha256,
        )
    finally:
        outer_counter.close()
    job_ledger.add_time(
        "adaptive_functional_p_outer_entry_cache",
        wall_seconds=time.perf_counter() - outer_started,
    )
    if (
        outer_entry_p_cache.population_sha256 != theta0_cache.population_sha256
        or _entry_parameter_snapshot_sha256(model, capture.entry_sha256)
        != outer_entry_snapshot_sha256
    ):
        raise ODEBFStateError("adaptive fixed outer-entry P cache differs")
    stages.record(
        "post_adaptive_common_capture",
        {
            "instruction_id": panel_instruction_id,
            "request_order_sha256": capture.request_order_sha256,
            "n32_receipt_sha256": n32_sha256,
            "outer_entry_snapshot_sha256": outer_entry_snapshot_sha256,
            "outer_entry_p_cache_sha256": outer_entry_p_cache.receipt_sha256,
            "variants": [item.label for item in specs],
            "functional_p_decision_by_label": {
                item.label: item.functional_p_decision.value for item in specs
            },
            "preservation_constraints_by_label": {
                item.label: item.preservation_constraints.value for item in specs
            },
            "functional_p_field_policy_by_label": {
                item.label: item.functional_p_field_policy.value for item in specs
            },
            "scientific_promotion_authorized": False,
        },
    )

    rollouts: dict[Any, VariantRollout] = {}
    recorders: list[AdaptiveReceiptRecorder] = []
    for spec in specs:
        variant = spec.clock_variant
        if _parameter_contract_sha256(touched) != after_capture_contract:
            raise ODEBFStateError("adaptive variant did not start from common W0")
        arm_receipt = ArmWeightSnapshot(
            P1Arm.R_BF,
            0,
            base_receipt.parameter_sha256,
            canonical_hash(
                {
                    "variant": spec.label,
                    "batch": 0,
                    "weights": base_receipt.parameter_sha256,
                }
            ),
        )
        arm_state = ArmRuntimeState(
            P1Arm.R_BF,
            P1HistoryLedger(
                layer_order=tuple(int(layer) for layer in hparams.layers),
                maximum_records=40,
            ),
            ComputeLedger(),
            arm_receipt,
            dict(base_values),
        )
        recorder = AdaptiveReceiptRecorder(
            raw_root,
            variant,
            write_once,
            variant_label=spec.label,
            instruction_id=panel_instruction_id,
            receipt_schema=f"{panel_schema_namespace}-receipt/v1",
        )
        recorders.append(recorder)
        counter = ModelForwardCounter(model, arm_state.ledger)
        try:
            rollout = _run_variant(
                model,
                tokenizer,
                requests,
                alias=alias,
                variant=variant,
                capture=capture,
                hparams=hparams,
                projector=projector,
                contexts=contexts,
                covariance_registry=covariance_registry,
                projector_sha256=projector_sha256,
                lock=controller_lock,
                arm_state=arm_state,
                request_by_sha256=request_by_sha256,
                population_by_sha256=population_by_sha256,
                schedule=schedule,
                outer_entry_p_cache=outer_entry_p_cache,
                theta0_cache=theta0_cache,
                touched=touched,
                recorder=recorder,
                routing_objective=spec.routing_objective,
                variant_label=spec.label,
                functional_p_policy=spec.functional_p_decision,
                preservation_policy=spec.preservation_constraints,
                functional_p_field_policy=spec.functional_p_field_policy,
            )
        finally:
            counter.close()
        _observed_memory(arm_state.ledger)
        if arm_state.history.version != 0:
            raise ODEBFStateError("adaptive causal variant appended history")
        if rollout_validator is not None:
            rollout_validator(alias, spec, rollout)
        rollouts[spec.legacy_key] = rollout
        stages.record(
            f"post_adaptive_{spec.label.lower().replace('-', '_')}",
            {
                "variant": spec.label,
                "status": rollout.status,
                "termination_label": rollout.termination_label,
                "rollout_sha256": rollout.rollout_sha256,
                "accepted_t": fraction_payload(rollout.accepted_t),
                "k_acc": rollout.k_acc,
                "n_trial": rollout.n_trial,
                "n_reject": rollout.n_reject,
                "field_build_count": rollout.field_build_count,
                "online_first_hit": (
                    None
                    if rollout.first_hit.first_online is None
                    else rollout.first_hit.first_online.accepted_index
                ),
            },
        )

    if _parameter_contract_sha256(touched) != after_capture_contract:
        raise ODEBFStateError("adaptive rollout panel did not preserve W0/RNG")
    panel, step_receipts = _postfreeze_stepwise_panel(
        model,
        tokenizer,
        alias=alias,
        requests=requests,
        dataset_path=dataset_path,
        capture=capture,
        rollouts=rollouts,
        raw_root=raw_root,
        write_once=write_once,
        touched=touched,
        instruction_id=panel_instruction_id,
        schema_namespace=panel_schema_namespace,
    )
    if _parameter_contract_sha256(touched) != after_capture_contract:
        raise ODEBFStateError("adaptive post-freeze panel did not restore W0/RNG")
    if panel_refinement_builder is not None:
        refinement = panel_refinement_builder(capture, rollouts, step_receipts)
    elif custom_panel:
        refinement = _target_new_panel_payload(capture, rollouts, step_receipts)
    else:
        refinement = _refinement_payload(capture, rollouts, step_receipts)
    stepwise_sha256 = write_once(
        raw_root / "stepwise" / "panel.json",
        {**panel, "refinement": refinement},
    )
    stages.record(
        "post_adaptive_stepwise_evaluation",
        {
            "action_frozen_before_open": True,
            "stepwise_panel_sha256": stepwise_sha256,
            "unique_accepted_snapshot_count": panel[
                "unique_accepted_snapshot_count"
            ],
            "refinement_status": refinement.get(
                "status", "TARGET_NEW_ROUTING_CONTRASTS_RECORDED"
            ),
            "generation_call_count": 0,
        },
    )
    artifact_guard.assert_unchanged()
    final_bytes = {
        name: tensor_sha256(parameter) for name, parameter in sorted(touched.items())
    }
    if final_bytes != base_bytes:
        raise ODEBFStateError("adaptive diagnostic final W0 restore differs")
    for recorder in recorders:
        recorder.assert_observation_complete()
    _observed_memory(job_ledger)
    terminal = {
        "schema": f"{panel_schema_namespace}-terminal/v1",
        "instruction_id": panel_instruction_id,
        "status": panel_terminal_status,
        "alias": alias,
        "source_head": source_head,
        "edit_batch_size": BATCH_SIZE,
        "sequential_batch_count": 1,
        "variants": [item.label for item in specs],
        "variant_rollout_sha256": {
            item.label: rollouts[item.legacy_key].rollout_sha256 for item in specs
        },
        "variant_status": {
            item.label: rollouts[item.legacy_key].status for item in specs
        },
        "variant_termination_label": {
            item.label: rollouts[item.legacy_key].termination_label
            for item in specs
        },
        "functional_p_decision_by_label": {
            item.label: item.functional_p_decision.value for item in specs
        },
        "functional_p_decision_influence_count_by_label": {
            item.label: rollouts[item.legacy_key].n_trial
            if item.functional_p_decision is FunctionalPDecisionPolicy.PCTRL
            else 0
            for item in specs
        },
        "preservation_constraints_by_label": {
            item.label: item.preservation_constraints.value for item in specs
        },
        "functional_p_field_policy_by_label": {
            item.label: item.functional_p_field_policy.value for item in specs
        },
        "functional_p_field_decision_influence_count_by_label": {
            item.label: (
                rollouts[item.legacy_key].field_build_count
                if item.functional_p_field_policy
                is FunctionalPFieldPolicy.SOFT_HARD
                else 0
            )
            for item in specs
        },
        "preservation_decision_influence_count_by_label": {
            item.label: {
                name: (
                    rollouts[item.legacy_key].n_trial
                    if item.preservation_constraints
                    is PreservationConstraintPolicy.LOCKED
                    else 0
                )
                for name in (
                    "structural_h", "structural_p", "trust", "functional_h"
                )
            }
            for item in specs
        },
        "n32_receipt_sha256": n32_sha256,
        "stepwise_panel_sha256": stepwise_sha256,
        "refinement": refinement,
        "scientific_sample_reused_for_causal_diagnostic": True,
        "scientific_promotion_authorized": False,
        "controller_identity_sha256": controller_lock.identity(),
        "numerical_lock_sha256": numerical_sha256,
        "stream_root_digest": stream["root_digest"],
        "artifact_receipt": asdict(artifact_receipt),
        "context_sha256": context_sha256,
        "cuda_preflight": dict(cuda_runtime_receipt),
        "job_compute": job_ledger.raw_free_payload(),
        "final_w0_restored": True,
        "persistent_endpoint_commit_count": 0,
        "history_append_count": 0,
        "heldout_controller_access_count": 0,
        "generation_call_count": 0,
        "retry_submission_count": 0,
    }
    if custom_panel:
        terminal.update(
            dict(panel_terminal_metadata)
            if panel_terminal_metadata is not None
            else {
                "routing_loss_change_only": True,
                "routing_context_group_sizes": [1, 5],
                "routing_context_count": 6,
                "routing_context_weighting": (
                    "uniform-over-all-six-locked-rendered-contexts"
                ),
                "target_z_velocity_objective": "MARGIN_LOCKED",
                "old_nll_decision_influence_count_newnll": 0,
                "target_new_routing_contrasts": refinement,
                "stepwise_layer_routing_telemetry": {
                    "schema": (
                        "ode-edit-stepwise-layer-routing-telemetry/v1"
                    ),
                    "trajectory_schema": (
                        "ode-edit-stepwise-layer-routing-trajectory/v1"
                    ),
                    "candidate_layer_ids": list(ROUTING_LAYER_IDS),
                    "variant_labels": [item.label for item in specs],
                    "optional_a16_definition_uses_same_schema": True,
                    "observation_only": True,
                    "controller_dependency_count": 0,
                    "diagnostic_severe_concentration_only": True,
                },
            }
        )
    else:
        terminal["legacy_p1r3_and_full_residual_baseline_comparison"] = (
            "REQUIRED_IN_TERMINAL_ANALYSIS_NO_RUNTIME_TUNING"
        )
    terminal_sha256 = write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": f"{panel_schema_namespace}-manifest/v1",
        "instruction_id": panel_instruction_id,
        "status": terminal["status"],
        "alias": alias,
        "source_head": source_head,
        "terminal_sha256": terminal_sha256,
        "n32_receipt_sha256": n32_sha256,
        "stepwise_panel_sha256": stepwise_sha256,
        "variant_rollout_sha256": terminal["variant_rollout_sha256"],
        "stepwise_layer_routing_telemetry_schema": (
            "ode-edit-stepwise-layer-routing-telemetry/v1"
            if custom_panel
            else None
        ),
        "stepwise_layer_routing_candidate_layer_ids": (
            list(ROUTING_LAYER_IDS) if custom_panel else None
        ),
        "retry_submission_count": 0,
    }
    manifest_sha256 = write_once(destination / "manifest.json", manifest)
    return {
        "status": terminal["status"],
        "alias": alias,
        "terminal_sha256": terminal_sha256,
        "manifest_sha256": manifest_sha256,
        "variant_status": terminal["variant_status"],
        "variant_termination_label": terminal["variant_termination_label"],
        "final_w0_restored": True,
    }
