"""Runtime for the fixed-grid E8 preservation-aware diagnostic.

The implementation is a sibling of the adaptive P1R6 runtime.  It reuses the
locked full-residual field, W64/BF16 functional assembler, layer-local cold
target overlay, replay evaluator, and transaction contracts, while owning its
clock and routing decisions.  No adaptive/retry or scientific candidate gate
is reachable from this module.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field as dataclass_field
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .first_hit import FeasibilityVerdict
from .fixed_e8_soft_routing import (
    FIXED_E8_GRID_COUNT,
    FIXED_E8_H,
    FIXED_E8_KAPPA,
    FIXED_E8_LAYER_ORDER,
    FIXED_E8_METHOD_ID,
    FixedE8Arm,
    FixedE8Clock,
    FixedE8GridPoint,
    FixedE8OperationCeiling,
    FixedE8RoutingResult,
    FixedE8SoftInventory,
    FixedE8StepMode,
    FunctionalBasisMetric,
    fixed_e8_semantic_receipt,
    solve_fixed_e8_routing,
    structural_contribution_vector,
)
from .functional import WaypointFactor, assemble_effective_bf16, tensor_sha256
from .layer_routing_telemetry import build_layer_routing_telemetry
from .p0_runtime import ModelForwardCounter
from .p1_adaptive import (
    fraction_payload,
    streaming_bf16_capacity,
)
from .p1_adaptive_runtime import (
    FunctionalPDecisionPolicy,
    _controller_replay_entry,
    _evaluate_rewrite,
    _factor_map,
    _factor_state,
    _functional_trial,
    _history_actions,
    _history_keys,
    _merge_factors,
    _omega_state,
    _parameter_contract_sha256,
    _postfreeze_stepwise_panel,
    _risk_payload,
    _rng_identity,
    _target_new_panel_contrast,
    _terminal_confirm_snapshots,
)
from .p1_backend import (
    FULL_CURRENT_RESIDUAL_DEFINITION,
    P1DynamicField,
    PinnedCovarianceRegistry,
    SignedProgressReceipt,
    _CoefficientOverlay,
    _virtual_context,
    build_p1_dynamic_field,
    capture_p1_native_entry,
    evaluate_routing_progress,
)
from .p1_controller import (
    AcceptedLayerContribution,
    P1ControllerLock,
    _cumulative_structural_terms,
)
from .p1_replay import (
    OuterEntryPretrainedCache,
    Theta0TeacherCache,
    build_outer_entry_pretrained_cache,
)
from .p1_state import ArmWeightSnapshot, P1Arm, P1HistoryLedger
from .request_digest import ordered_request_digest_v1
from .routing import PreservationConstraintPolicy, QuadraticBarrier, RoutingProblem
from .sampling import StatelessReplaySchedule
from .target_new_nll import RoutingObjective, evaluate_routing_objective
from .cold_start_target import (
    ColdTargetMetric,
    cold_field_semantic_receipt,
    cold_lookup_positions,
    cold_target_step_validator,
    capture_cold_z_base,
    evaluate_cold_target_objective,
    write_aware_cold_target_velocity,
)


FIXED_E8_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-COLD-FIXED-E8-STRUCTFUNC-SOFT-P1R7-V1"
)
FIXED_E8_SCHEMA_NAMESPACE = "ode-edit-s05-cold-fixed-e8-structfunc-soft-p1r7"
FIXED_E8_PANEL_LABELS = tuple(item.value for item in FixedE8Arm)


@dataclass(slots=True)
class FixedE8EntryCapture:
    entry_weights: dict[str, torch.Tensor]
    entry_sha256: dict[str, str]


@dataclass(frozen=True, slots=True)
class FixedE8ProblemReceipt:
    problem: RoutingProblem
    historical_self_risk: tuple[float, ...]
    pretrained_self_risk: tuple[float, ...]
    temporary_load: tuple[float, ...]
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class FixedE8FieldReceipt:
    field_sha256: str
    field_semantic_sha256: str
    signed_progress_sha256: str
    problem_sha256: str
    functional_inventory_sha256: str
    routing_sha256: str
    target_velocity_sha256: str
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class FixedE8HitRecord:
    accepted_index: int
    tau: Fraction
    snapshot_sha256: str
    success_count: int
    online_component_feasible: bool = True


@dataclass(slots=True)
class FixedE8HitTracker:
    records: list[FixedE8HitRecord] = dataclass_field(default_factory=list)

    def append(self, value: FixedE8HitRecord) -> None:
        if value.accepted_index != len(self.records) + 1:
            raise ODEBFStateError("fixed E8 first-hit record order differs")
        self.records.append(value)

    @property
    def first_online(self) -> FixedE8HitRecord | None:
        return next(
            (item for item in self.records if item.success_count == BATCH_SIZE),
            None,
        )


@dataclass(slots=True)
class FixedE8Snapshot:
    accepted_index: int
    tau: Fraction
    delta_tau: Fraction
    factors: dict[str, tuple[WaypointFactor, ...]]
    target_state: torch.Tensor
    snapshot_sha256: str
    evaluation: Any
    feasibility: FeasibilityVerdict
    structural_payload: dict[str, Any]
    routing_payload: dict[str, Any]
    progress_payload: dict[str, Any]
    capacity_payload: dict[str, Any]
    field_sha256: str
    raw_velocity_sha256: str
    bf_velocity_sha256: str
    accepted_receipt_sha256: str
    functional_p_decision: dict[str, Any]


@dataclass(slots=True)
class FixedE8Rollout:
    variant: FixedE8Arm
    status: str
    termination_label: str
    residual_policy: str
    accepted_t: Fraction
    k_acc: int
    n_trial: int
    n_reject: int
    field_build_count: int
    snapshots: list[FixedE8Snapshot]
    first_hit: FixedE8HitTracker
    recorder: "FixedE8ReceiptRecorder"
    ledger: ComputeLedger
    entry_success: dict[str, Any]
    terminal_confirmation: list[dict[str, Any]]
    rollout_sha256: str
    variant_label: str
    routing_objective: str
    functional_p_decision: str
    preservation_constraints: str
    functional_p_field_policy: str


def _fixed_e8_solver_accounting(
    routing: FixedE8RoutingResult,
) -> dict[str, Any]:
    logical = len(routing.certificates)
    backend = sum(item.optimizer_pass_count for item in routing.certificates)
    continuation = sum(
        max(item.optimizer_pass_count - 1, 0)
        for item in routing.certificates
    )
    if backend != logical + continuation:
        raise ODEBFContractError("fixed E8 optimizer accounting differs")
    maximum_schedule = bool(
        routing.mode is FixedE8StepMode.JOINT_WRITE
        and logical == 4
        and backend == 5
        and continuation == 1
    )
    return {
        "actual_logical_qp_certificate_count": logical,
        "actual_optimizer_backend_invocation_count": backend,
        "actual_numerical_backend_continuation_count": continuation,
        "numerical_backend_continuation_role": (
            "FIXED_NUMERICAL_BACKEND_CONTINUATION_SAME_QP"
            if continuation
            else "NONE"
        ),
        "solver_schedule_matches_static_maximum": maximum_schedule,
        "static_operation_counts_are_maximum_ceiling": True,
        "scientific_retry_count": 0,
    }


def _fixed_e8_factual_online_feasibility(
    structural_functional: Mapping[str, Any],
    *,
    history_item_count: int,
) -> tuple[FeasibilityVerdict, dict[str, Any]]:
    def observed(component: str, key: str = "passed") -> bool:
        value = structural_functional.get(component)
        if not isinstance(value, Mapping) or type(value.get(key)) is not bool:
            raise ODEBFContractError(
                f"fixed E8 factual {component} observation differs"
            )
        return bool(value[key])

    verdict = FeasibilityVerdict(
        observed("historical"),
        observed("pretrained"),
        observed("trust"),
        observed("functional_h"),
        observed("functional_p"),
        True,
    )
    h_status = (
        "ACTIVE_OBSERVATION_ONLY"
        if history_item_count > 0
        else "INACTIVE_EMPTY_HISTORY"
    )
    payload = {
        "role": "OBSERVATION_ONLY",
        "clock_decision_influence_count": 0,
        "first_hit_decision_influence_count": 0,
        "endpoint_decision_influence_count": 0,
        "components": {
            "structural_h": {
                "passed": verdict.structural_h,
                "status": h_status,
                "decision_influence_count": 0,
            },
            "structural_p": {
                "passed": verdict.structural_p,
                "status": "ACTIVE_OBSERVATION_ONLY",
                "decision_influence_count": 0,
            },
            "functional_h": {
                "passed": verdict.functional_h,
                "status": h_status,
                "decision_influence_count": 0,
            },
            "functional_p": {
                "passed": verdict.functional_p,
                "status": "ACTIVE_OBSERVATION_ONLY",
                "decision_influence_count": 0,
            },
            "write_trust": {
                "passed": verdict.trust,
                "status": "TECHNICAL_INTEGRATION_BOUND",
                "decision_influence_count": 1,
            },
            "authoritative_bf16": {
                "passed": verdict.authoritative_bf16,
                "status": "TECHNICAL_FAIL_CLOSE_VERIFIED",
                "decision_influence_count": 1,
            },
        },
        "all_observed_components_pass": verdict.all_pass,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return verdict, payload


def _fixed_e8_advance_grid_transition(
    ledger: ComputeLedger,
    clock: FixedE8Clock,
    point: FixedE8GridPoint,
    *,
    scientific_observation: Mapping[str, Any],
) -> dict[str, Any]:
    before = ledger.counters["trial"]
    ledger.increment("trial")
    transition = clock.advance(
        point, scientific_observation=scientific_observation
    )
    if ledger.counters["trial"] != before + 1:
        raise ODEBFStateError("fixed E8 trajectory trial accounting differs")
    return transition


class FixedE8ReceiptRecorder:
    """Create-once raw-free receipt writer with no adaptive clock surface."""

    def __init__(self, root: Path, arm: FixedE8Arm, write_once: Any) -> None:
        self.root = root / "fixed-e8" / arm.value
        self.arm = arm
        self.variant_label = arm.value
        self.write_once = write_once
        self.field_hashes: list[str] = []
        self.solver_hashes: list[str] = []
        self.trial_hashes: list[str] = []
        self.transition_hashes: list[str] = []
        self.accepted_hashes: list[str] = []
        self.first_hit_hashes: list[str] = []
        self.terminal_hashes: list[str] = []

    def _write(self, category: str, ordinal: int, payload: Mapping[str, Any]) -> str:
        return self.write_once(
            self.root / f"{category}-{ordinal:04d}.json",
            {
                "schema": f"{FIXED_E8_SCHEMA_NAMESPACE}-receipt/v1",
                "instruction_id": FIXED_E8_INSTRUCTION_ID,
                "arm": self.arm.value,
                "category": category,
                "ordinal": ordinal,
                **dict(payload),
            },
        )

    def _append(self, category: str, hashes: list[str], payload: Mapping[str, Any]) -> str:
        digest = self._write(category, len(hashes), payload)
        hashes.append(digest)
        return digest

    def field(self, payload: Mapping[str, Any]) -> str:
        return self._append("field", self.field_hashes, payload)

    def solver(self, payload: Mapping[str, Any]) -> str:
        return self._append("solver", self.solver_hashes, payload)

    def trial(self, payload: Mapping[str, Any]) -> str:
        return self._append("trial", self.trial_hashes, payload)

    def transition(self, payload: Mapping[str, Any]) -> str:
        return self._append("transition", self.transition_hashes, payload)

    def accepted(self, payload: Mapping[str, Any]) -> str:
        return self._append("accepted", self.accepted_hashes, payload)

    def first_hit(self, payload: Mapping[str, Any]) -> str:
        return self._append("first-hit", self.first_hit_hashes, payload)

    def terminal(self, payload: Mapping[str, Any]) -> str:
        return self._append("terminal", self.terminal_hashes, payload)

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

def fixed_e8_waypoint_factors(
    field: P1DynamicField,
    velocity: Sequence[float],
    *,
    step_index: int,
) -> dict[str, WaypointFactor]:
    values = tuple(float(item) for item in velocity)
    if (
        tuple(int(item.layer) for item in field.layers) != FIXED_E8_LAYER_ORDER
        or len(values) != len(FIXED_E8_LAYER_ORDER)
        or step_index < 0
        or step_index >= FIXED_E8_GRID_COUNT
        or any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in values)
    ):
        raise ODEBFContractError("fixed E8 waypoint geometry differs")
    return {
        layer.weight_name: WaypointFactor(
            layer.weight_name,
            layer.layer,
            0,
            step_index,
            ordinal,
            float(FIXED_E8_H) * values[ordinal],
            layer.residual.clone(),
            layer.q.clone(),
        )
        for ordinal, layer in enumerate(field.layers)
    }


def _fixed_e8_signed_progress_gradient(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    field: P1DynamicField,
    *,
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    contexts: Sequence[Sequence[str]],
    ledger: ComputeLedger,
) -> SignedProgressReceipt:
    """Target-new signed layer efficiency without all-nonpositive rejection."""

    device = next(model.parameters()).device
    coefficients = torch.zeros(
        len(field.layers), dtype=torch.float32, device=device, requires_grad=True
    )
    before = tuple(
        (parameter.data_ptr(), parameter._version, parameter.grad)
        for parameter in model.parameters()
    )
    with _virtual_context(model, cumulative_factors_by_weight):
        with _CoefficientOverlay(model, field.layers, coefficients):
            observed = evaluate_routing_objective(
                model,
                tokenizer,
                requests,
                objective=RoutingObjective.TARGET_NEW_NLL,
                contexts=contexts,
                gradient_input=coefficients,
            )
    if (
        observed.input_gradient is None
        or observed.backward_count != BATCH_SIZE
        or observed.target_true_suffix_token_counts != (None,) * BATCH_SIZE
    ):
        raise ODEBFContractError("fixed E8 target-new signed gradient differs")
    gradient = observed.input_gradient.detach()
    raw = -gradient.to(device="cpu", dtype=torch.float64)
    progress = tuple(float(item) for item in raw)
    if (
        len(progress) != len(FIXED_E8_LAYER_ORDER)
        or not all(math.isfinite(item) for item in progress)
        or tuple(int(item.layer) for item in field.layers)
        != FIXED_E8_LAYER_ORDER
    ):
        raise ODEBFContractError("fixed E8 signed-progress geometry differs")
    after = tuple(
        (parameter.data_ptr(), parameter._version, parameter.grad)
        for parameter in model.parameters()
    )
    if after != before:
        raise ODEBFStateError("fixed E8 signed gradient mutated model state")
    ledger.increment("backward", observed.backward_count)
    payload = {
        "objective": RoutingObjective.TARGET_NEW_NLL.value,
        "request_order_sha256": observed.request_order_sha256,
        "suffix_token_counts": list(observed.suffix_token_counts),
        "target_span_sha256": observed.target_span_sha256,
        "context_group_sizes": list(observed.context_group_sizes),
        "context_count": observed.context_count,
        "context_sha256": observed.context_sha256,
        "target_old_access_count": 0,
        "routing_backward_count": observed.backward_count,
        "all_nonpositive_is_scientific_zero_write": True,
    }
    return SignedProgressReceipt(
        field.identity_sha256,
        progress,
        tuple(
            field.layers[index].layer
            for index, value in enumerate(progress)
            if value <= 0.0
        ),
        tensor_sha256(gradient),
        observed.model_forward_count,
        observed.processed_token_count,
        True,
        RoutingObjective.TARGET_NEW_NLL.value,
        0,
        canonical_hash(payload),
        observed.context_sha256,
        observed.context_count,
        observed.context_group_sizes,
        observed.backward_count,
    )


def _build_fixed_e8_problem(
    field: P1DynamicField,
    signed: SignedProgressReceipt,
    *,
    accepted_by_layer: Mapping[int, Sequence[AcceptedLayerContribution]],
    committed_load_by_layer: Mapping[int, float],
    lock: P1ControllerLock,
    current_history_action_by_layer: Mapping[int, torch.Tensor],
) -> FixedE8ProblemReceipt:
    """Build the P1 geometry while omitting legacy H/P decision budgets."""

    layers = tuple(int(item.layer) for item in field.layers)
    if layers != FIXED_E8_LAYER_ORDER or signed.field_sha256 != field.identity_sha256:
        raise ODEBFContractError("fixed E8 field/problem identity differs")
    if set(committed_load_by_layer) != set(layers):
        raise ODEBFContractError("fixed E8 cumulative-load inventory differs")
    h_terms, p_terms, h_self, p_self, temporary = _cumulative_structural_terms(
        field,
        accepted_by_layer,
        current_history_action_by_layer=current_history_action_by_layer,
    )
    h = float(FIXED_E8_H)
    if not math.isclose(lock.eta, h, rel_tol=0.0, abs_tol=0.0):
        raise ODEBFContractError("fixed E8 controller step differs")
    raw_progress = np.asarray(signed.signed_progress, dtype=np.float64)
    applied_progress = h * raw_progress
    factor_energy = np.asarray(
        [item.factor_frobenius_sq for item in field.layers], dtype=np.float64
    )
    applied_energy = h**2 * factor_energy
    capacity = np.asarray(
        [
            (1.0 + float(committed_load_by_layer[layer]) + temporary[index])
            * (applied_energy[index] + lock.capacity_epsilon)
            for index, layer in enumerate(layers)
        ],
        dtype=np.float64,
    )
    trust_radius = lock.write_trust_fraction * math.sqrt(float(applied_energy.sum()))
    h_offset, h_linear, h_gram = h_terms
    p_offset, p_linear, p_gram = p_terms
    h_linear_scaled = h * np.asarray(h_linear, dtype=np.float64)
    p_linear_scaled = h * np.asarray(p_linear, dtype=np.float64)
    h_gram_scaled = h**2 * np.asarray(h_gram, dtype=np.float64)
    p_gram_scaled = h**2 * np.asarray(p_gram, dtype=np.float64)
    # Legacy 0.5 thresholds remain attached only as comparison telemetry.  The
    # new solver never reads QuadraticBarrier.budget.
    h_budget = float(h_offset) + lock.structural_h_increment_fraction * float(
        np.asarray(h_self, dtype=np.float64).sum() * h**2
    )
    p_budget = float(p_offset) + lock.structural_p_increment_fraction * float(
        np.asarray(p_self, dtype=np.float64).sum() * h**2
    )
    dummy = min(lock.minimum_progress, 1.0e-12)
    problem = RoutingProblem(
        applied_progress,
        np.diag(capacity),
        np.diag(applied_energy),
        trust_radius,
        np.minimum(
            np.full(len(layers), lock.layer_velocity_cap, dtype=np.float64), 1.0
        ),
        dummy,
        dummy,
        QuadraticBarrier(
            "historical",
            float(h_offset),
            h_linear_scaled,
            h_gram_scaled,
            h_budget,
            "layer-local-diagonal",
        ),
        QuadraticBarrier(
            "pretrained",
            float(p_offset),
            p_linear_scaled,
            p_gram_scaled,
            p_budget,
            "layer-local-diagonal",
        ),
    )
    payload = {
        "schema": "ode-edit-fixed-e8-problem/v1",
        "field_sha256": field.identity_sha256,
        "signed_progress_sha256": canonical_hash(list(signed.signed_progress)),
        "problem_sha256": problem.identity(),
        "step_coordinates": "problem-linear=h*g;problem-gram=h^2*M",
        "h_applied_exactly_once": True,
        "legacy_structural_budget_influence_count": 0,
        "historical_self_risk": list(h_self),
        "pretrained_self_risk": list(p_self),
        "temporary_load": list(temporary),
    }
    return FixedE8ProblemReceipt(
        problem,
        tuple(float(item) for item in h_self),
        tuple(float(item) for item in p_self),
        tuple(float(item) for item in temporary),
        canonical_hash(payload),
    )


def _fixed_e8_functional_basis_probe(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    field: P1DynamicField,
    step_index: int,
    factors: Mapping[str, Sequence[WaypointFactor]],
    target_state: torch.Tensor,
    capture: FixedE8EntryCapture,
    replay_entry: Any,
    theta0_cache: Theta0TeacherCache,
    lock: P1ControllerLock,
    ledger: ComputeLedger,
    touched: Mapping[str, torch.nn.Parameter],
    history: P1HistoryLedger,
    schedule: StatelessReplaySchedule,
    factor_state_sha256: str,
    field_semantic_sha256: str,
) -> tuple[FixedE8SoftInventory, dict[str, Any]]:
    """Measure one baseline plus exactly five fixed-h basis endpoints.

    The fixed-E8 controller consumes only positive raw endpoint increments.
    No legacy 1e-3 hinge, hard pass bit, or adaptive retry surface enters this
    inventory. Historical values share the same endpoints and become inactive
    when history is empty, so functional H never creates extra probes.
    """

    if tuple(int(item.layer) for item in field.layers) != FIXED_E8_LAYER_ORDER:
        raise ODEBFContractError("fixed E8 functional basis layer order differs")
    if not isinstance(field_semantic_sha256, str) or len(field_semantic_sha256) != 64:
        raise ODEBFContractError("fixed E8 functional basis semantic field differs")
    before_model = _parameter_contract_sha256(touched)
    before_history = history.snapshot().digest
    before_sampler = schedule.state_digest
    before_rng = _rng_identity()
    before_target = tensor_sha256(target_state)
    before_factor_state = _factor_state(
        capture.entry_sha256, factors, target_state
    )
    if before_factor_state != factor_state_sha256:
        raise ODEBFContractError("fixed E8 functional basis state differs")
    counter_before = dict(ledger.counters)
    wall_started = time.perf_counter()
    gpu_start: torch.cuda.Event | None = None
    gpu_end: torch.cuda.Event | None = None
    if torch.cuda.is_available():
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()

    baseline = _functional_trial(
        model,
        tokenizer,
        alias=alias,
        entry=replay_entry,
        theta0_cache=theta0_cache,
        factors=factors,
        lock=lock,
        ledger=ledger,
    )
    sample_order = replay_entry.pretrained_baseline.sample_order_sha256
    if (
        baseline.pretrained.sample_order_sha256 != sample_order
        or baseline.pretrained_entry_receipt.request_order_sha256
        != sample_order
        or baseline.pretrained_entry_receipt.value_sha256
        != replay_entry.pretrained_baseline.entry_kl_identity_sha256
        or baseline.historical_entry_receipt.value_sha256
        != replay_entry.history_entry_sha256
    ):
        raise ODEBFContractError(
            "fixed E8 functional basis baseline identity differs"
        )
    probes: list[Any] = []
    for ordinal in range(len(FIXED_E8_LAYER_ORDER)):
        basis = [0.0] * len(FIXED_E8_LAYER_ORDER)
        basis[ordinal] = 1.0
        increment = fixed_e8_waypoint_factors(
            field, basis, step_index=step_index
        )
        observed = _functional_trial(
            model,
            tokenizer,
            alias=alias,
            entry=replay_entry,
            theta0_cache=theta0_cache,
            factors=_merge_factors(factors, increment),
            lock=lock,
            ledger=ledger,
        )
        if (
            observed.pretrained.sample_order_sha256 != sample_order
            or observed.pretrained_entry_receipt.request_order_sha256
            != sample_order
            or observed.pretrained_entry_receipt.value_sha256
            != replay_entry.pretrained_baseline.entry_kl_identity_sha256
            or observed.historical.sample_order_sha256
            != baseline.historical.sample_order_sha256
            or observed.historical_entry_receipt.value_sha256
            != replay_entry.history_entry_sha256
        ):
            raise ODEBFContractError(
                "fixed E8 functional basis endpoint identity differs"
            )
        probes.append(observed)

    if gpu_start is not None and gpu_end is not None:
        gpu_end.record()
        gpu_end.synchronize()
        gpu_seconds = float(gpu_start.elapsed_time(gpu_end)) / 1000.0
    else:
        gpu_seconds = 0.0
    wall_seconds = time.perf_counter() - wall_started
    ledger.add_time(
        "fixed_e8_functional_basis",
        wall_seconds=wall_seconds,
        gpu_seconds=gpu_seconds,
    )
    history_count = int(baseline.historical.item_count)
    active_h = history_count > 0
    reason = None if active_h else "INACTIVE_EMPTY_HISTORY"
    inventory = FixedE8SoftInventory(
        FunctionalBasisMetric(
            "functional_p",
            float(baseline.pretrained.mean_positive_damage),
            tuple(
                float(item.pretrained.mean_positive_damage) for item in probes
            ),
            True,
        ),
        FunctionalBasisMetric(
            "functional_h_mean",
            float(baseline.historical.mean_positive_damage),
            tuple(
                float(item.historical.mean_positive_damage) for item in probes
            ),
            active_h,
            reason,
        ),
        FunctionalBasisMetric(
            "functional_h_smoothmax",
            float(baseline.historical.smooth_max_positive_damage),
            tuple(
                float(item.historical.smooth_max_positive_damage)
                for item in probes
            ),
            active_h,
            reason,
        ),
        history_count,
        sample_order,
        field_semantic_sha256,
    )
    after_model = _parameter_contract_sha256(touched)
    after_history = history.snapshot().digest
    after_sampler = schedule.state_digest
    after_rng = _rng_identity()
    after_target = tensor_sha256(target_state)
    after_factor_state = _factor_state(
        capture.entry_sha256, factors, target_state
    )
    if (
        after_model != before_model
        or after_history != before_history
        or after_sampler != before_sampler
        or after_rng != before_rng
        or after_target != before_target
        or after_factor_state != before_factor_state
    ):
        raise ODEBFStateError(
            "fixed E8 functional basis mutated model/state/history/sampler/RNG"
        )
    counter_delta = {
        name: int(ledger.counters[name] - counter_before[name])
        for name in sorted(ledger.counters)
    }
    payload: dict[str, Any] = {
        "schema": "ode-edit-fixed-e8-functional-basis/v1",
        "field_sha256": field.identity_sha256,
        "field_semantic_sha256": field_semantic_sha256,
        "factor_state_sha256": factor_state_sha256,
        "controller_p_sample_order_sha256": sample_order,
        "controller_p_baseline_identity_sha256": (
            replay_entry.pretrained_baseline.entry_kl_identity_sha256
        ),
        "controller_p_schedule_sha256": replay_entry.schedule_sha256,
        "inventory": inventory.raw_free_payload(),
        "baseline": {
            "functional_p": _risk_payload(baseline.pretrained),
            "functional_h": _risk_payload(baseline.historical),
        },
        "probe_endpoints": [
            {
                "layer": FIXED_E8_LAYER_ORDER[index],
                "functional_p": _risk_payload(item.pretrained),
                "functional_h": _risk_payload(item.historical),
                "pretrained_endpoint_identity_sha256": (
                    item.pretrained_trial_receipt.value_sha256
                ),
            }
            for index, item in enumerate(probes)
        ],
        "exact_basis_endpoint_count": 6,
        "baseline_endpoint_count": 1,
        "layer_probe_endpoint_count": 5,
        "functional_h_additional_probe_endpoint_count": 0,
        "historical_soft_status": (
            "ACTIVE" if active_h else "INACTIVE_EMPTY_HISTORY"
        ),
        "legacy_one_e_minus_three_hinge_access_count": 0,
        "hard_functional_decision_influence_count": 0,
        "counter_delta": counter_delta,
        "wall_seconds": wall_seconds,
        "gpu_seconds": gpu_seconds,
    }
    payload["identity_sha256"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "identity_sha256"}
    )
    return inventory, payload


def _build_fixed_field_with_metric(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    metric: ColdTargetMetric,
    alias: str,
    arm: FixedE8Arm,
    step_index: int,
    current_factors: Mapping[str, Sequence[WaypointFactor]],
    current_target: torch.Tensor,
    capture: FixedE8EntryCapture,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    lock: P1ControllerLock,
    history: P1HistoryLedger,
    accepted_by_layer: Mapping[int, Sequence[AcceptedLayerContribution]],
    ledger: ComputeLedger,
    replay_entry: Any,
    theta0_cache: Theta0TeacherCache,
    touched: Mapping[str, torch.nn.Parameter],
    schedule: StatelessReplaySchedule,
    recorder: FixedE8ReceiptRecorder,
    lookup_positions: Sequence[int],
    target_layer_name: str,
) -> tuple[
    P1DynamicField,
    SignedProgressReceipt,
    FixedE8ProblemReceipt,
    FixedE8SoftInventory,
    FixedE8RoutingResult,
    torch.Tensor,
    Any,
    FixedE8FieldReceipt,
    Mapping[str, Any],
]:
    """Build one field using the immutable outer-entry target metric."""
    before = _parameter_contract_sha256(touched)
    before_history = history.snapshot().digest
    before_sampler = schedule.state_digest
    before_rng = _rng_identity()
    solve_history = _history_keys(history, FIXED_E8_LAYER_ORDER, risk=False)
    risk_history = _history_keys(history, FIXED_E8_LAYER_ORDER, risk=True)
    field = build_p1_dynamic_field(
        model,
        tokenizer,
        requests,
        hparams,
        projector,
        contexts,
        target_state=current_target,
        accepted_waypoint=step_index,
        cumulative_factors_by_weight=current_factors,
        history_solve_keys_by_layer=solve_history,
        history_risk_keys_by_layer=risk_history,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256,
        residual_tolerance=lock.residual_tolerance,
        ledger=ledger,
        residual_policy=FULL_CURRENT_RESIDUAL_DEFINITION,
        allow_zero_capacity=True,
    )
    signed = _fixed_e8_signed_progress_gradient(
        model,
        tokenizer,
        requests,
        field,
        cumulative_factors_by_weight=current_factors,
        contexts=contexts,
        ledger=ledger,
    )
    semantic_field = cold_field_semantic_receipt(
        field,
        history_solve_keys_by_layer=solve_history,
        history_risk_keys_by_layer=risk_history,
    )
    built = _build_fixed_e8_problem(
        field,
        signed,
        accepted_by_layer=accepted_by_layer,
        committed_load_by_layer=history.cumulative_load(),
        lock=lock,
        current_history_action_by_layer=_history_actions(field, history),
    )
    factor_state = _factor_state(capture.entry_sha256, current_factors, current_target)
    inventory, probe = _fixed_e8_functional_basis_probe(
        model,
        tokenizer,
        alias=alias,
        field=field,
        step_index=step_index,
        factors=current_factors,
        target_state=current_target,
        capture=capture,
        replay_entry=replay_entry,
        theta0_cache=theta0_cache,
        lock=lock,
        ledger=ledger,
        touched=touched,
        history=history,
        schedule=schedule,
        factor_state_sha256=factor_state,
        field_semantic_sha256=semantic_field["semantic_identity_sha256"],
    )
    routing = solve_fixed_e8_routing(built.problem, inventory, arm=arm)
    ledger.increment("qp_solve", len(routing.certificates))
    ledger.increment("qp_certificate", len(routing.certificates))
    if routing.mode is FixedE8StepMode.ZERO_WRITE_TARGET_RECOVERY:
        observed = evaluate_cold_target_objective(
            model,
            tokenizer,
            requests,
            contexts,
            layer_name=target_layer_name,
            lookup_positions=lookup_positions,
            target_state=current_target,
            require_gradient=True,
        )
        if observed.gradient is None or observed.backward_count != BATCH_SIZE:
            raise ODEBFContractError(
                "fixed E8 zero-write target recovery gradient differs"
            )
        target_velocity, gradient_norms = metric.unit_descent(observed.gradient)
        ledger.increment("backward", observed.backward_count)
        ledger.increment("target_backward", observed.backward_count)
        target_receipt: Mapping[str, Any] = {
            "mode": FixedE8StepMode.ZERO_WRITE_TARGET_RECOVERY.value,
            "field_sha256": field.identity_sha256,
            "velocity_sha256": tensor_sha256(target_velocity),
            "metric_sha256": metric.identity_sha256,
            "gradient_norms": list(gradient_norms),
            "unit_g_norms": list(
                metric.distance(target_velocity, torch.zeros_like(target_velocity))
            ),
            "target_backward_count": observed.backward_count,
            "processed_token_count": observed.processed_token_count,
            "target_old_access_count": 0,
            "native_or_direct_z_access_count": 0,
            "weight_velocity": [0.0] * len(FIXED_E8_LAYER_ORDER),
            "target_only_recovery": True,
        }
    else:
        target_velocity, observed_target_receipt = write_aware_cold_target_velocity(
            model,
            tokenizer,
            requests,
            contexts,
            field=field,
            coefficients=routing.velocity,
            cumulative_factors_by_weight=current_factors,
            metric=metric,
            ledger=ledger,
        )
        target_receipt = asdict(observed_target_receipt)
    if (
        _parameter_contract_sha256(touched) != before
        or history.snapshot().digest != before_history
        or schedule.state_digest != before_sampler
        or _rng_identity() != before_rng
    ):
        raise ODEBFStateError("fixed E8 field build mutated state/history/sampler/RNG")
    routing_payload = routing.raw_free_payload()
    field_payload = {
        "method_id": FIXED_E8_METHOD_ID,
        "arm": arm.value,
        "step_index": step_index,
        "field": field.raw_free_payload(),
        "field_sha256": field.identity_sha256,
        "field_semantic": semantic_field,
        "signed_progress": asdict(signed),
        "problem_receipt_sha256": built.identity_sha256,
        "functional_basis": inventory.raw_free_payload(),
        "probe_receipt": probe,
        "routing": routing_payload,
        "target_velocity": dict(target_receipt),
        "layer_local_full_residual_target_overlay": True,
        "target_overlay_definition": "R_l(z_k)+(z-z_k)",
        "target_velocity_step_semantics": (
            "STEP_INDEPENDENT_FINITE_UNIT_FIELD_LOOKAHEAD"
        ),
        "candidate_coupled": False,
        "retry_recomputes_target_velocity": False,
        "hard_h_p_budget_influence_count": 0,
        "functional_candidate_veto_influence_count": 0,
    }
    field_sha = recorder.field(field_payload)
    for certificate in routing.certificates:
        recorder.solver(
            {
                "step_index": step_index,
                "arm": arm.value,
                "routing_sha256": routing.identity_sha256,
                "certificate": certificate.raw_free_payload(),
                "hard_h_p_budget_influence_count": 0,
            }
        )
    receipt_payload = {
        "field_sha256": field.identity_sha256,
        "field_semantic_sha256": semantic_field["semantic_identity_sha256"],
        "signed_progress_sha256": canonical_hash(list(signed.signed_progress)),
        "problem_sha256": built.problem.identity(),
        "functional_inventory_sha256": inventory.raw_free_payload()["identity_sha256"],
        "routing_sha256": routing.identity_sha256,
        "target_velocity_sha256": str(target_receipt["velocity_sha256"]),
        "persisted_field_receipt_sha256": field_sha,
    }
    receipt = FixedE8FieldReceipt(
        receipt_payload["field_sha256"],
        receipt_payload["field_semantic_sha256"],
        receipt_payload["signed_progress_sha256"],
        receipt_payload["problem_sha256"],
        receipt_payload["functional_inventory_sha256"],
        receipt_payload["routing_sha256"],
        receipt_payload["target_velocity_sha256"],
        canonical_hash(receipt_payload),
    )
    return (
        field,
        signed,
        built,
        inventory,
        routing,
        target_velocity,
        target_receipt,
        receipt,
        probe,
    )


def _fixed_capacity_payload(
    capture: FixedE8EntryCapture,
    field: P1DynamicField,
    previous: Mapping[str, Sequence[WaypointFactor]],
    candidate: Mapping[str, Sequence[WaypointFactor]],
    increment: Mapping[str, WaypointFactor],
    *,
    problem: RoutingProblem,
    routing: FixedE8RoutingResult,
    signed: SignedProgressReceipt,
    step_index: int,
    state_sha256: str,
    target_sha256: str,
    factor_state_sha256: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    requested: list[float] = []
    realized: list[float] = []
    cumulative: list[float] = []
    effective_sha256: dict[str, str] = {}
    streaming: dict[str, Any] = {}
    for ordinal, layer in enumerate(field.layers):
        name = layer.weight_name
        theta = float(increment[name].theta)
        if not math.isclose(
            theta,
            float(FIXED_E8_H) * routing.velocity[ordinal],
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ):
            raise ODEBFContractError("fixed E8 factor theta differs from h*v")
        requested.append(theta**2 * layer.factor_frobenius_sq)
        receipt = streaming_bf16_capacity(
            capture.entry_weights[name],
            previous.get(name, ()),
            candidate.get(name, ()),
            weight_name=name,
        )
        candidate_values = candidate.get(name, ())
        if candidate_values:
            effective, stats = assemble_effective_bf16(
                capture.entry_weights[name], candidate_values, row_block=64
            )
            if stats.effective_bf16_sha256 != tensor_sha256(effective):
                raise ODEBFContractError(
                    "fixed E8 authoritative BF16 identity differs"
                )
            effective_sha256[name] = stats.effective_bf16_sha256
            del effective
        else:
            effective_sha256[name] = tensor_sha256(capture.entry_weights[name])
        realized.append(receipt.step_energy)
        cumulative.append(receipt.cumulative_energy)
        streaming[str(layer.layer)] = asdict(receipt)
    velocity = np.asarray(routing.velocity, dtype=np.float64)
    applied = np.asarray(routing.applied_coefficient, dtype=np.float64)
    raw_slopes = np.asarray(signed.signed_progress, dtype=np.float64)
    structural_h = structural_contribution_vector(
        problem.historical, velocity
    )
    structural_p = structural_contribution_vector(
        problem.pretrained, velocity
    )
    trust = velocity * (problem.trust_metric @ velocity)
    layer_routing = build_layer_routing_telemetry(
        step_index=step_index,
        stage="TRIAL",
        layer_ids=FIXED_E8_LAYER_ORDER,
        signed_efficiency=raw_slopes,
        raw_velocity=velocity,
        bf_velocity=velocity,
        applied_coefficient=applied,
        active_direction_mask=raw_slopes > 0.0,
        raw_cap_bound_mask=np.isclose(velocity, problem.layer_caps, rtol=0.0, atol=1.0e-12),
        bf_cap_bound_mask=np.isclose(velocity, problem.layer_caps, rtol=0.0, atol=1.0e-12),
        raw_zero_bound_mask=np.isclose(velocity, 0.0, rtol=0.0, atol=1.0e-12),
        bf_zero_bound_mask=np.isclose(velocity, 0.0, rtol=0.0, atol=1.0e-12),
        predicted_progress_contribution=raw_slopes * applied,
        prequantized_update_energy=requested,
        realized_bf16_update_energy=realized,
        cumulative_bf16_capacity=cumulative,
        bf16_capacity_contribution=realized,
        structural_h_contribution=structural_h,
        structural_p_contribution=structural_p,
        trust_contribution=trust,
        field_sha256=field.identity_sha256,
        state_sha256=state_sha256,
        target_z_sha256=target_sha256,
        factor_state_sha256=factor_state_sha256,
        raw_solver_certificate_sha256=canonical_hash(
            [item.raw_free_payload() for item in routing.certificates]
        ),
        bf_solver_certificate_sha256=routing.identity_sha256,
        candidate_sha256=candidate_sha256,
    )
    payload = {
        "layer_order": list(FIXED_E8_LAYER_ORDER),
        "velocity": list(routing.velocity),
        "applied_coefficient": list(routing.applied_coefficient),
        "requested_frobenius_energy": requested,
        "realized_bf16_step_energy": realized,
        "cumulative_bf16_capacity": cumulative,
        "structural_h_contribution": structural_h.tolist(),
        "structural_p_contribution": structural_p.tolist(),
        "trust_contribution": trust.tolist(),
        "effective_bf16_sha256": dict(sorted(effective_sha256.items())),
        "streaming_receipts": streaming,
        "layer_routing": layer_routing,
        "h_applied_exactly_once": True,
        "dense_fp64_full_delta_live": 0,
        "dense_fp32_full_delta_live": 0,
        "effective_bf16_weight_peak_live": 1,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _run_fixed_variant(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    arm: FixedE8Arm,
    capture: FixedE8EntryCapture,
    z_base: torch.Tensor,
    metric: ColdTargetMetric,
    lookup_positions: Sequence[int],
    target_layer_name: str,
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
    recorder: FixedE8ReceiptRecorder,
) -> FixedE8Rollout:
    clock = FixedE8Clock()
    history = arm_state.history
    ledger = arm_state.ledger
    before_model = _parameter_contract_sha256(touched)
    before_history = history.snapshot().digest
    before_sampler = schedule.state_digest
    before_rng = _rng_identity()
    current_target = z_base.clone()
    current_factors: dict[str, tuple[WaypointFactor, ...]] = {}
    accepted_by_layer: dict[int, list[AcceptedLayerContribution]] = {
        layer: [] for layer in FIXED_E8_LAYER_ORDER
    }
    entry_omega = _omega_state(accepted_by_layer)
    current_objective = evaluate_routing_progress(
        model,
        tokenizer,
        requests,
        cumulative_factors_by_weight=current_factors,
        objective=RoutingObjective.TARGET_NEW_NLL,
        contexts=contexts,
    )
    entry_eval = _evaluate_rewrite(
        model,
        tokenizer,
        requests,
        alias=alias,
        factors=current_factors,
        ledger=ledger,
    )
    snapshots: list[FixedE8Snapshot] = []
    first_hit = FixedE8HitTracker()
    functional_basis_endpoint_count = 0
    controller_candidate_functional_endpoint_count = 0
    field_backward_batch_count = 0
    target_backward_batch_count = 0
    actual_logical_qp_certificate_count = 0
    actual_optimizer_backend_invocation_count = 0
    actual_numerical_backend_continuation_count = 0
    zero_write_field_count = 0
    for step_index in range(FIXED_E8_GRID_COUNT):
        point = clock.begin_field()
        replay_entry = _controller_replay_entry(
            model,
            tokenizer,
            alias=alias,
            arm_state=arm_state,
            sample_waypoint=step_index + 1,
            factors=current_factors,
            request_by_sha256=request_by_sha256,
            population_by_sha256=population_by_sha256,
            schedule=schedule,
            outer_entry_p_cache=outer_entry_p_cache,
        )
        (
            field,
            signed,
            built,
            inventory,
            routing,
            target_velocity,
            target_velocity_receipt,
            field_receipt,
            probe_payload,
        ) = _build_fixed_field_with_metric(
            model,
            tokenizer,
            requests,
            metric=metric,
            alias=alias,
            arm=arm,
            step_index=step_index,
            current_factors=current_factors,
            current_target=current_target,
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
            replay_entry=replay_entry,
            theta0_cache=theta0_cache,
            touched=touched,
            schedule=schedule,
            recorder=recorder,
            lookup_positions=lookup_positions,
            target_layer_name=target_layer_name,
        )
        solver_accounting = _fixed_e8_solver_accounting(routing)
        actual_logical_qp_certificate_count += int(
            solver_accounting["actual_logical_qp_certificate_count"]
        )
        actual_optimizer_backend_invocation_count += int(
            solver_accounting["actual_optimizer_backend_invocation_count"]
        )
        actual_numerical_backend_continuation_count += int(
            solver_accounting["actual_numerical_backend_continuation_count"]
        )
        if routing.mode is FixedE8StepMode.ZERO_WRITE_TARGET_RECOVERY:
            zero_write_field_count += 1
        if probe_payload["exact_basis_endpoint_count"] != 6:
            raise ODEBFContractError(
                "fixed E8 functional basis operation count differs"
            )
        functional_basis_endpoint_count += 6
        field_backward_batch_count += 1
        target_backward_batch_count += 1
        increment = fixed_e8_waypoint_factors(
            field, routing.velocity, step_index=step_index
        )
        candidate_factors = (
            _factor_map(current_factors)
            if routing.mode is FixedE8StepMode.ZERO_WRITE_TARGET_RECOVERY
            else _merge_factors(current_factors, increment)
        )
        target_trial = (
            current_target
            + float(FIXED_E8_H)
            * target_velocity.detach().to(device="cpu", dtype=torch.float32)
        ).contiguous()
        target_certificate = cold_target_step_validator(metric, z_base)(
            current_target, target_trial, point.tau_before, FIXED_E8_H
        )
        candidate_objective = evaluate_routing_progress(
            model,
            tokenizer,
            requests,
            cumulative_factors_by_weight=candidate_factors,
            objective=RoutingObjective.TARGET_NEW_NLL,
            contexts=contexts,
        )
        actual_progress = float(current_objective.value - candidate_objective.value)
        predicted_progress = float(
            built.problem.signed_progress @ np.asarray(routing.velocity)
        )
        rho = actual_progress / max(
            predicted_progress, lock.minimum_progress
        )
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
        controller_candidate_functional_endpoint_count += 1
        evaluation = _evaluate_rewrite(
            model,
            tokenizer,
            requests,
            alias=alias,
            factors=candidate_factors,
            ledger=ledger,
        )
        snapshot_sha = _factor_state(
            capture.entry_sha256, candidate_factors, target_trial
        )
        factor_state = _factor_state(
            capture.entry_sha256, current_factors, current_target
        )
        capacity = _fixed_capacity_payload(
            capture,
            field,
            current_factors,
            candidate_factors,
            increment,
            problem=built.problem,
            routing=routing,
            signed=signed,
            step_index=step_index,
            state_sha256=_parameter_contract_sha256(touched),
            target_sha256=tensor_sha256(current_target),
            factor_state_sha256=factor_state,
            candidate_sha256=snapshot_sha,
        )
        structural_h_value = built.problem.historical.value(
            np.asarray(routing.velocity)
        )
        structural_p_value = built.problem.pretrained.value(
            np.asarray(routing.velocity)
        )
        trust_value = float(
            np.asarray(routing.velocity)
            @ built.problem.trust_metric
            @ np.asarray(routing.velocity)
        )
        functional_h_observation = {
            **_risk_payload(functional.historical),
            "status": (
                "ACTIVE_OBSERVATION_ONLY"
                if inventory.history_item_count > 0
                else "INACTIVE_EMPTY_HISTORY"
            ),
            "role": "OBSERVATION_ONLY",
            "decision_influence_count": 0,
        }
        functional_p_observation = {
            **_risk_payload(functional.pretrained),
            "status": "ACTIVE_OBSERVATION_ONLY",
            "role": "OBSERVATION_ONLY",
            "decision_influence_count": 0,
        }
        structural_payload = {
            "historical": {
                "value": structural_h_value,
                "legacy_budget": built.problem.historical.budget,
                "legacy_pass": structural_h_value
                <= built.problem.historical.budget + 1.0e-8,
                "passed": structural_h_value
                <= built.problem.historical.budget + 1.0e-8,
                "decision_influence_count": 0,
                "role": "OBSERVATION_ONLY",
                "status": (
                    "INACTIVE_EMPTY_HISTORY"
                    if inventory.history_item_count == 0
                    else "ACTIVE_SOFT_SCORE"
                ),
            },
            "pretrained": {
                "value": structural_p_value,
                "legacy_budget": built.problem.pretrained.budget,
                "legacy_pass": structural_p_value
                <= built.problem.pretrained.budget + 1.0e-8,
                "passed": structural_p_value
                <= built.problem.pretrained.budget + 1.0e-8,
                "decision_influence_count": 0,
                "role": "OBSERVATION_ONLY",
                "status": "ACTIVE_SOFT_SCORE",
            },
            "trust": {
                "value": trust_value,
                "radius_squared": built.problem.trust_radius**2,
                "passed": trust_value
                <= built.problem.trust_radius**2 + 1.0e-8,
                "role": "TECHNICAL_INTEGRATION_BOUND",
                "decision_influence_count": 1,
            },
            "functional_h": functional_h_observation,
            "functional_p": functional_p_observation,
            "functional_candidate_veto_influence_count": 0,
        }
        factual_feasibility, factual_feasibility_payload = (
            _fixed_e8_factual_online_feasibility(
                structural_payload,
                history_item_count=inventory.history_item_count,
            )
        )
        structural_payload["online_feasibility_observation"] = (
            factual_feasibility_payload
        )
        progress_payload = {
            "predicted": predicted_progress,
            "actual": actual_progress,
            "rho": rho,
            "p_max": routing.p_max,
            "kappa": FIXED_E8_KAPPA,
            "requested_progress": routing.requested_progress,
            "negative_actual_progress_observation_only": actual_progress <= 0.0,
            "rho_below_point_one_observation_only": rho < 0.1,
            "scientific_rejection_count": 0,
            "routing_objective": RoutingObjective.TARGET_NEW_NLL.value,
            "target_old_decision_influence_count": 0,
        }
        routing_payload = {
            "method_id": FIXED_E8_METHOD_ID,
            "field_receipt_sha256": field_receipt.identity_sha256,
            "field_sha256": field.identity_sha256,
            "field_semantic_sha256": field_receipt.field_semantic_sha256,
            "routing": routing.raw_free_payload(),
            "functional_basis": inventory.raw_free_payload(),
            "actual_solver_accounting": solver_accounting,
            "probe_receipt_sha256": probe_payload["identity_sha256"],
            "target_velocity": dict(target_velocity_receipt),
            "target_step_certificate": target_certificate,
            "residual_policy": FULL_CURRENT_RESIDUAL_DEFINITION,
            "h": float(FIXED_E8_H),
            "theta_equals_h_times_v": True,
        }
        trial_payload = {
            "step_index": step_index,
            "tau_before": fraction_payload(point.tau_before),
            "tau_after": fraction_payload(point.tau_after),
            "field_receipt_sha256": field_receipt.identity_sha256,
            "routing": routing_payload,
            "progress": progress_payload,
            "structural_functional": structural_payload,
            "official_success": evaluation.batch_success.raw_free_payload(),
            "snapshot_sha256": snapshot_sha,
            "capacity_sha256": capacity["identity_sha256"],
            "accepted": True,
            "scientific_candidate_veto_count": 0,
            "mode": routing.mode.value,
        }
        trial_sha = recorder.trial(trial_payload)
        transition = _fixed_e8_advance_grid_transition(
            ledger,
            clock,
            point,
            scientific_observation={
                "actual_progress": actual_progress,
                "rho": rho,
                "structural_h": structural_h_value,
                "structural_p": structural_p_value,
                "functional_h": functional.historical.mean_positive_damage,
                "functional_p": functional.pretrained.mean_positive_damage,
                "success_count": evaluation.batch_success.numerator,
            },
        )
        transition_sha = recorder.transition(
            {
                **transition,
                "trial_receipt_sha256": trial_sha,
                "snapshot_sha256": snapshot_sha,
                "zero_write_target_recovery": routing.mode
                is FixedE8StepMode.ZERO_WRITE_TARGET_RECOVERY,
                "scientific_observations_do_not_gate": True,
            }
        )
        if routing.mode is FixedE8StepMode.JOINT_WRITE:
            for layer in field.layers:
                accepted_by_layer[layer.layer].append(
                    AcceptedLayerContribution.from_field(
                        layer,
                        increment[layer.weight_name],
                        history_action=layer.history_action,
                    )
                )
        ledger.record_accepted_step(
            accepted_dt=float(FIXED_E8_H), completed_k_total=step_index + 1
        )
        hit = FixedE8HitRecord(
            step_index + 1,
            point.tau_after,
            snapshot_sha,
            evaluation.batch_success.numerator,
            factual_feasibility.all_pass,
        )
        first_hit.append(hit)
        first_observed = first_hit.first_online is hit
        accepted_sha = recorder.accepted(
            {
                "accepted_index": step_index + 1,
                "tau_after": fraction_payload(point.tau_after),
                "delta_tau": fraction_payload(FIXED_E8_H),
                "transition_sha256": transition_sha,
                "snapshot_sha256": snapshot_sha,
                "field_sha256": field.identity_sha256,
                "official_success": evaluation.batch_success.raw_free_payload(),
                "structural_functional": structural_payload,
                "progress": progress_payload,
                "routing": routing_payload,
                "capacity": capacity,
                "first_hit_observed": first_observed,
                "first_hit_decision_influence_count": 0,
                "history_append_count": 0,
                "persistent_commit_count": 0,
            }
        )
        if first_observed:
            recorder.first_hit(
                {
                    "accepted_index": step_index + 1,
                    "tau": fraction_payload(point.tau_after),
                    "snapshot_sha256": snapshot_sha,
                    "accepted_receipt_sha256": accepted_sha,
                    "official_success": evaluation.batch_success.raw_free_payload(),
                    "observation_only": True,
                    "controller_dependency_count": 0,
                }
            )
        snapshots.append(
            FixedE8Snapshot(
                step_index + 1,
                point.tau_after,
                FIXED_E8_H,
                _factor_map(candidate_factors),
                target_trial.clone(),
                snapshot_sha,
                evaluation,
                factual_feasibility,
                structural_payload,
                routing_payload,
                progress_payload,
                capacity,
                field.identity_sha256,
                canonical_hash(list(routing.velocity)),
                routing.identity_sha256,
                accepted_sha,
                {
                    "policy": "OBSERVATION_ONLY",
                    "observation": _risk_payload(functional.pretrained),
                    "decision_influence_count": 0,
                },
            )
        )
        current_factors = _factor_map(candidate_factors)
        current_target = target_trial.clone()
        current_objective = candidate_objective
    clock_receipt = clock.terminal_receipt()
    if (
        len(snapshots) != FIXED_E8_GRID_COUNT
        or clock.field_count != FIXED_E8_GRID_COUNT
        or clock.tau != Fraction(1, 1)
        or functional_basis_endpoint_count != 48
        or controller_candidate_functional_endpoint_count != 8
        or field_backward_batch_count != 8
        or target_backward_batch_count != 8
        or ledger.counters["trial"] != FIXED_E8_GRID_COUNT
        or ledger.counters["qp_solve"]
        != actual_logical_qp_certificate_count
        or ledger.counters["qp_certificate"]
        != actual_logical_qp_certificate_count
    ):
        raise ODEBFStateError("fixed E8 terminal grid invariant differs")
    if ledger.completed_correction_cycles == 0:
        ledger.finish_cycle(0)
    terminal_confirmations = _terminal_confirm_snapshots(
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
        trajectory_complete=True,
        functional_p_policy=FunctionalPDecisionPolicy.OBSERVATION_ONLY,
        preservation_policy=PreservationConstraintPolicy.OBSERVATION_ONLY,
    )
    if len(terminal_confirmations) != FIXED_E8_GRID_COUNT:
        raise ODEBFContractError(
            "fixed E8 terminal functional audit count differs"
        )
    if (
        _parameter_contract_sha256(touched) != before_model
        or history.snapshot().digest != before_history
        or schedule.state_digest != before_sampler
        or _rng_identity() != before_rng
        or history.version != 0
        or (not snapshots and _omega_state(accepted_by_layer) != entry_omega)
    ):
        raise ODEBFStateError("fixed E8 rollout mutated model/history/sampler/RNG")
    rollout_payload = {
        "method_id": FIXED_E8_METHOD_ID,
        "variant": arm.value,
        "status": "FIXED_E8_TAU_COMPLETE",
        "termination_label": "FIXED_E8_TAU_COMPLETE",
        "accepted_t": fraction_payload(clock.tau),
        "k_acc": len(snapshots),
        "n_trial": len(snapshots),
        "n_reject": 0,
        "field_build_count": len(recorder.field_hashes),
        "operation_accounting": {
            "functional_basis_endpoint_count": (
                functional_basis_endpoint_count
            ),
            "functional_h_extra_endpoint_count": 0,
            "controller_candidate_functional_endpoint_count": (
                controller_candidate_functional_endpoint_count
            ),
            "field_backward_batch_count": field_backward_batch_count,
            "target_backward_batch_count": target_backward_batch_count,
            "terminal_audit_functional_endpoint_count": len(
                terminal_confirmations
            ),
            "solver_receipt_count": len(recorder.solver_hashes),
            "actual_logical_qp_certificate_count": (
                actual_logical_qp_certificate_count
            ),
            "actual_optimizer_backend_invocation_count": (
                actual_optimizer_backend_invocation_count
            ),
            "actual_numerical_backend_continuation_count": (
                actual_numerical_backend_continuation_count
            ),
            "actual_trial_count": ledger.counters["trial"],
            "actual_zero_write_field_count": zero_write_field_count,
            "solver_schedule_matches_static_maximum": (
                zero_write_field_count == 0
                and actual_logical_qp_certificate_count
                == FIXED_E8_GRID_COUNT * 4
                and actual_optimizer_backend_invocation_count
                == FIXED_E8_GRID_COUNT * 5
                and actual_numerical_backend_continuation_count
                == FIXED_E8_GRID_COUNT
            ),
            "static_operation_counts_semantics": "MAXIMUM_CEILING",
            "static_operation_ceiling": FixedE8OperationCeiling()
            .raw_free_payload()["per_arm"],
            "candidate_count": len(snapshots),
            "field_count": len(recorder.field_hashes),
        },
        "clock": clock_receipt,
        "accepted_snapshot_sha256": [item.snapshot_sha256 for item in snapshots],
        "entry_success": entry_eval.batch_success.raw_free_payload(),
        "online_first_hit": (
            None
            if first_hit.first_online is None
            else {
                "accepted_index": first_hit.first_online.accepted_index,
                "tau": fraction_payload(first_hit.first_online.tau),
                "snapshot_sha256": first_hit.first_online.snapshot_sha256,
            }
        ),
        "first_hit_observation_only": True,
        "full_grid_continues_after_hit": True,
        "routing_objective": RoutingObjective.TARGET_NEW_NLL.value,
        "native_or_direct_z_cold_access_count": 0,
        "scientific_retry_count": 0,
        "scientific_rejection_count": 0,
        "hard_structural_h_budget_influence_count": 0,
        "hard_structural_p_budget_influence_count": 0,
        "functional_candidate_veto_influence_count": 0,
        "persistent_commit_count": 0,
        "history_append_count": 0,
        "heldout_access_count": 0,
        "terminal_audit_count": len(terminal_confirmations),
        "receipt_links": recorder.links(),
        "compute": ledger.raw_free_payload(),
    }
    rollout_sha = canonical_hash(rollout_payload)
    recorder.terminal(
        {
            "rollout_summary": rollout_payload,
            "rollout_sha256": rollout_sha,
            "trajectory_complete": True,
        }
    )
    return FixedE8Rollout(
        arm,
        "FIXED_E8_TAU_COMPLETE",
        "FIXED_E8_TAU_COMPLETE",
        FULL_CURRENT_RESIDUAL_DEFINITION,
        Fraction(1, 1),
        8,
        8,
        0,
        8,
        snapshots,
        first_hit,
        recorder,
        ledger,
        entry_eval.batch_success.raw_free_payload(),
        terminal_confirmations,
        rollout_sha,
        arm.value,
        RoutingObjective.TARGET_NEW_NLL.value,
        "OBSERVATION_ONLY",
        "STRUCTFUNC_SOFT_ONLY",
        "BOUNDED_BASIS_SOFT",
    )


def run_fixed_e8_diagnostic(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    destination: Path,
    raw_root: Path,
    stages: Any,
    source_head: str,
    requests: Sequence[Mapping[str, Any]],
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
) -> dict[str, Any]:
    """Run both fixed E8 actions, freeze, then open N32 and held-out metrics."""

    from .p1_runtime import (
        ArmRuntimeState,
        _entry_parameter_snapshot_sha256,
        _evaluate_native_rewrite,
        _observed_memory,
    )

    if len(requests) != BATCH_SIZE:
        raise ODEBFContractError("fixed E8 diagnostic requires one sealed B10")
    request_order = ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in requests]
    )
    if request_order != stream["batch_ordered_request_digest_v1"][0]:
        raise ODEBFContractError("fixed E8 request/seal order differs")
    base_bytes = {
        name: tensor_sha256(value) for name, value in sorted(touched.items())
    }
    base_contract = _parameter_contract_sha256(touched)
    if tuple(sorted(base_bytes.items())) != tuple(base_receipt.parameter_sha256):
        raise ODEBFStateError("fixed E8 diagnostic did not start from W0")
    z_base = capture_cold_z_base(model, tokenizer, requests, hparams)
    metric = ColdTargetMetric.from_z_base(z_base, request_order)
    lookup_positions = cold_lookup_positions(
        tokenizer,
        requests,
        contexts,
        fact_token_strategy=hparams.fact_token,
    )
    target_layer_name = hparams.layer_module_tmp.format(int(hparams.layers[-1]))
    z_base_payload = {
        "schema": f"{FIXED_E8_SCHEMA_NAMESPACE}-z-base/v1",
        "request_order_sha256": request_order,
        "z_base_sha256": tensor_sha256(z_base),
        "metric": metric.raw_free_payload(),
        "native_or_direct_z_access_count": 0,
        "separate_bootstrap_clock_count": 0,
    }
    z_base_sha = write_once(raw_root / "fixed-e8" / "z-base.json", z_base_payload)
    outer_population = tuple(
        population_by_sha256[item] for item in theta0_cache.request_order
    )
    outer_entry_snapshot = _entry_parameter_snapshot_sha256(
        model, dict(base_receipt.parameter_sha256)
    )
    outer_counter = ModelForwardCounter(model, job_ledger)
    try:
        outer_entry_p_cache = build_outer_entry_pretrained_cache(
            model,
            tokenizer,
            outer_population,
            theta0_cache,
            outer_entry_snapshot_sha256=outer_entry_snapshot,
        )
    finally:
        outer_counter.close()
    capture = FixedE8EntryCapture(
        {
            name: value.detach().to(device="cpu").clone()
            for name, value in base_values.items()
        },
        dict(base_receipt.parameter_sha256),
    )
    rollouts: dict[str, FixedE8Rollout] = {}
    initial_pre_soft: dict[str, Any] = {}
    for arm in (FixedE8Arm.NEUTRAL, FixedE8Arm.SOFT):
        if _parameter_contract_sha256(touched) != base_contract:
            raise ODEBFStateError("fixed E8 arm W0 entry differs")
        receipt = ArmWeightSnapshot(
            P1Arm.R_BF,
            0,
            base_receipt.parameter_sha256,
            canonical_hash({"arm": arm.value, "weights": base_receipt.parameter_sha256}),
        )
        arm_state = ArmRuntimeState(
            P1Arm.R_BF,
            P1HistoryLedger(layer_order=FIXED_E8_LAYER_ORDER, maximum_records=40),
            ComputeLedger(),
            receipt,
            dict(base_values),
        )
        recorder = FixedE8ReceiptRecorder(raw_root, arm, write_once)
        counter = ModelForwardCounter(model, arm_state.ledger)
        try:
            rollout = _run_fixed_variant(
                model,
                tokenizer,
                requests,
                alias=alias,
                arm=arm,
                capture=capture,
                z_base=z_base,
                metric=metric,
                lookup_positions=lookup_positions,
                target_layer_name=target_layer_name,
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
            )
        finally:
            counter.close()
        _observed_memory(arm_state.ledger)
        if arm_state.history.version != 0:
            raise ODEBFStateError("fixed E8 arm appended history")
        rollouts[arm.value] = rollout
        first_routing = rollout.snapshots[0].routing_payload["routing"]
        initial_pre_soft[arm.value] = {
            "field_semantic_sha256": rollout.snapshots[0].routing_payload[
                "field_semantic_sha256"
            ],
            "signed_slopes": first_routing["signed_slopes"],
            "p_max": first_routing["p_max"],
            "requested_progress": first_routing["requested_progress"],
            "pre_soft_velocity": first_routing["pre_soft_velocity"],
            "soft_shadow_velocity": first_routing["soft_velocity"],
            "functional_inventory_sha256": rollout.snapshots[0].routing_payload[
                "functional_basis"
            ]["identity_sha256"],
            "matched_solver_schedule": first_routing[
                "matched_solver_schedule"
            ],
        }
        stages.record(
            f"post_{arm.value.lower().replace('-', '_')}",
            {
                "status": rollout.status,
                "accepted_t": fraction_payload(rollout.accepted_t),
                "k_acc": rollout.k_acc,
                "field_build_count": rollout.field_build_count,
                "rollout_sha256": rollout.rollout_sha256,
            },
        )
    if initial_pre_soft[FixedE8Arm.NEUTRAL.value] != initial_pre_soft[FixedE8Arm.SOFT.value]:
        raise ODEBFContractError("fixed E8 common initial pre-soft field differs")
    if {
        name: tensor_sha256(value) for name, value in sorted(touched.items())
    } != base_bytes:
        raise ODEBFStateError("fixed E8 actions did not preserve W0")
    action_freeze = {
        "schema": f"{FIXED_E8_SCHEMA_NAMESPACE}-action-freeze/v1",
        "instruction_id": FIXED_E8_INSTRUCTION_ID,
        "request_order_sha256": request_order,
        "z_base_sha256": z_base_sha,
        "rollout_sha256": {
            label: rollout.rollout_sha256 for label, rollout in rollouts.items()
        },
        "actions_frozen_before_native_and_heldout": True,
        "cold_native_or_direct_z_access_count": 0,
        "heldout_controller_access_count": 0,
        "common_initial_pre_soft_identity_sha256": canonical_hash(initial_pre_soft),
    }
    action_freeze_sha = write_once(
        raw_root / "fixed-e8" / "action-freeze.json", action_freeze
    )
    native_history = P1HistoryLedger(
        layer_order=FIXED_E8_LAYER_ORDER, maximum_records=40
    )
    native_ledger = ComputeLedger()
    native_counter = ModelForwardCounter(model, native_ledger)
    try:
        native_capture = capture_p1_native_entry(
            model,
            tokenizer,
            requests,
            hparams,
            projector,
            contexts,
            history_keys_by_layer=_history_keys(
                native_history, FIXED_E8_LAYER_ORDER, risk=False
            ),
            mutation_lock=mutation_lock,
            ledger=native_ledger,
            residual_tolerance=controller_lock.residual_tolerance,
        )
        native_online = _evaluate_native_rewrite(
            model,
            tokenizer,
            requests,
            alias=alias,
            candidates=native_capture.native_candidates,
            ledger=native_ledger,
        )
    finally:
        native_counter.close()
    n32_sha = write_once(
        raw_root / "fixed-e8" / "N32_NATIVE-postfreeze.json",
        {
            "schema": f"{FIXED_E8_SCHEMA_NAMESPACE}-native-reference/v1",
            "action_freeze_sha256": action_freeze_sha,
            "opened_after_fixed_e8_action_freeze": True,
            "capture": native_capture.raw_free_payload(),
            "official_success": native_online.batch_success.raw_free_payload(),
            "compute": native_ledger.raw_free_payload(),
            "controller_dependency_count": 0,
            "cold_receipt_backflow_count": 0,
        },
    )
    panel, step_receipts = _postfreeze_stepwise_panel(
        model,
        tokenizer,
        alias=alias,
        requests=requests,
        dataset_path=dataset_path,
        capture=native_capture,
        rollouts=rollouts,
        raw_root=raw_root,
        write_once=write_once,
        touched=touched,
        instruction_id=FIXED_E8_INSTRUCTION_ID,
        schema_namespace=FIXED_E8_SCHEMA_NAMESPACE,
    )
    refinement = {
        "schema": f"{FIXED_E8_SCHEMA_NAMESPACE}-contrasts/v1",
        "soft_minus_neutral": _target_new_panel_contrast(
            native_capture,
            rollouts,
            step_receipts,
            left_label=FixedE8Arm.NEUTRAL.value,
            right_label=FixedE8Arm.SOFT.value,
        ),
        "native_opened_only_after_action_freeze": True,
        "scientific_promotion_authorized": False,
    }
    stepwise_sha = write_once(
        raw_root / "stepwise" / "panel.json", {**panel, "refinement": refinement}
    )
    artifact_guard.assert_unchanged()
    final_bytes = {
        name: tensor_sha256(value) for name, value in sorted(touched.items())
    }
    if final_bytes != base_bytes:
        raise ODEBFStateError("fixed E8 diagnostic final W0 restore differs")
    terminal = {
        "schema": f"{FIXED_E8_SCHEMA_NAMESPACE}-terminal/v1",
        "instruction_id": FIXED_E8_INSTRUCTION_ID,
        "status": "FIXED_E8_DIAGNOSTIC_COMPLETE_NO_PROMOTION",
        "method": fixed_e8_semantic_receipt(),
        "alias": alias,
        "source_head": source_head,
        "request_order_sha256": request_order,
        "stream_root_digest": stream["root_digest"],
        "z_base_sha256": z_base_sha,
        "action_freeze_sha256": action_freeze_sha,
        "n32_postfreeze_sha256": n32_sha,
        "stepwise_panel_sha256": stepwise_sha,
        "variant_status": {label: value.status for label, value in rollouts.items()},
        "variant_rollout_sha256": {
            label: value.rollout_sha256 for label, value in rollouts.items()
        },
        "cold_native_or_direct_z_access_count": 0,
        "n32_opened_only_after_action_freeze": True,
        "scientific_retry_count": 0,
        "scientific_rejection_count": 0,
        "hard_h_p_budget_influence_count": 0,
        "functional_candidate_veto_influence_count": 0,
        "first_hit_observation_only": True,
        "full_grid_tau": 1.0,
        "numerical_lock_sha256": numerical_sha256,
        "artifact_receipt": asdict(artifact_receipt),
        "context_sha256": context_sha256,
        "cuda_preflight": dict(cuda_runtime_receipt),
        "job_compute": job_ledger.raw_free_payload(),
        "final_w0_restored": True,
        "persistent_endpoint_commit_count": 0,
        "history_append_count": 0,
        "heldout_controller_access_count": 0,
        "generation_call_count": 0,
        "scientific_promotion_authorized": False,
    }
    terminal_sha = write_once(destination / "terminal.json", terminal)
    manifest_sha = write_once(
        destination / "manifest.json",
        {
            "schema": f"{FIXED_E8_SCHEMA_NAMESPACE}-manifest/v1",
            "instruction_id": FIXED_E8_INSTRUCTION_ID,
            "status": terminal["status"],
            "alias": alias,
            "source_head": source_head,
            "terminal_sha256": terminal_sha,
            "action_freeze_sha256": action_freeze_sha,
            "n32_postfreeze_sha256": n32_sha,
            "stepwise_panel_sha256": stepwise_sha,
            "retry_submission_count": 0,
        },
    )
    return {
        "status": terminal["status"],
        "alias": alias,
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "final_w0_restored": True,
    }
