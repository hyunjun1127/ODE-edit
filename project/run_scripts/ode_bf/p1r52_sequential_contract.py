"""P1R52-specific sequential/Historical adapter contracts.

The frozen atomic P1R52 implementation is called unchanged.  This module owns
only the scoped sequential adaptations: active raw/projected history keys in
the Alpha/Woodbury field, cumulative Structural-H layer allocation, the
100-record ledger/anchor transaction, and outer-round identity receipts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import math
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from scipy.optimize import minimize

from .accounting import ComputeLedger
from .common_cold_coordinate import SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1
from .contracts import BATCH_SIZE, FIXED_K, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import assemble_effective_bf16, tensor_sha256
from .p1_backend import PinnedCovarianceRegistry, build_p1_dynamic_field
from .p1_controller import AcceptedLayerContribution, P1ControllerLock
from .p1_scalable_batched_experiment import _model_w0_contract
from .p1_evaluator import _evaluate_prefixes_pinned_with_accuracy
from .p1_state import P1HistoryRecord
from .p1r24_atomic_strength import P1R24RoutingResult, P1R24RoutingStatus
from .p1r29_sequential_preparation import SequentialArmState, SequentialRoundStaging
from .p1r43_full_strength_routing import P1R43RoutingResult
from .progress_simplex_routing import (
    SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
    SIMPLEX_ENERGY_RELATIVE_TOLERANCE,
    SIMPLEX_PRIMAL_TOLERANCE,
    SIMPLEX_XI_TIE_TOLERANCE,
)
from .routing import RoutingProblem
from .scalable_batched_field import scalable_terminal_residual
from .scalable_batched_runtime import P1R23_H, P1R23_LAYER_ORDER
from .target_new_nll import _surface
from .fixed_e8_soft_routing import FixedE8Arm
from .transaction import AtomicBatchTransaction


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-LLAMA-SOFT-SEQUENTIAL-HISTORICAL-10XB10-V1"
METHOD_ID = "P1R52-REPAIR-R1-LLAMA-SOFT-SEQUENTIAL-STRUCTURAL-H-V1"
ROUND_COUNT = 10
HISTORY_COUNTS = tuple(index * BATCH_SIZE for index in range(ROUND_COUNT))
COLLISION_POLICY = "KEEP_IMMUTABLE_LIFETIME_ANCHOR_RETIRE_SOLVE_RISK_ON_TARGET_COLLISION"
SOLVER_FTOL = 1.0e-12
SOLVER_MAXITER = 1_000


def _weight_hashes(parameters: Mapping[str, torch.nn.Parameter]) -> dict[str, str]:
    return {name: tensor_sha256(value) for name, value in sorted(parameters.items())}


def assemble_terminal_candidates(
    entry_values: Mapping[str, torch.Tensor],
    factors_by_weight: Mapping[str, Sequence[Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if set(entry_values) != set(factors_by_weight):
        raise ODEBFContractError("sequential terminal factor inventory differs")
    candidates: dict[str, torch.Tensor] = {}
    receipts: dict[str, Any] = {}
    for name in sorted(entry_values):
        value, stats = assemble_effective_bf16(
            entry_values[name],
            factors_by_weight[name],
            row_block=64,
        )
        candidates[name] = value.detach().to(device="cpu", dtype=torch.bfloat16).clone()
        receipts[name] = {
            "candidate_sha256": tensor_sha256(candidates[name]),
            "factor_count": stats.factor_count,
            "effective_bf16_sha256": stats.effective_bf16_sha256,
        }
    payload = {"weights": receipts, "materialization_count": 0, "candidate_only": True}
    payload["identity_sha256"] = canonical_hash(payload)
    return candidates, payload


@dataclass(frozen=True, slots=True)
class LifetimeAnchor:
    request_sha256: str
    case_id: int
    collision_sha256: str
    round_index: int
    target_new_nll_by_context: tuple[float, ...]
    margin_by_context: tuple[float, ...]
    context_provenance_sha256: str
    endpoint_weight_sha256: str
    identity_sha256: str

    def __post_init__(self) -> None:
        if len(self.request_sha256) != 64 or len(self.collision_sha256) != 64:
            raise ODEBFContractError("lifetime anchor request identity differs")
        if self.round_index < 1 or self.round_index > ROUND_COUNT or self.case_id < 0:
            raise ODEBFContractError("lifetime anchor round/case differs")
        if len(self.target_new_nll_by_context) != 6 or len(self.margin_by_context) != 6:
            raise ODEBFContractError("lifetime anchor does not contain full-six contexts")
        if not all(math.isfinite(value) for value in (*self.target_new_nll_by_context, *self.margin_by_context)):
            raise ODEBFContractError("lifetime anchor is non-finite")
        if any(len(value) != 64 for value in (self.context_provenance_sha256, self.endpoint_weight_sha256, self.identity_sha256)):
            raise ODEBFContractError("lifetime anchor digest differs")

    @classmethod
    def create(
        cls,
        *,
        request_sha256: str,
        case_id: int,
        collision_sha256: str,
        round_index: int,
        target_new_nll_by_context: Sequence[float],
        margin_by_context: Sequence[float],
        context_provenance_sha256: str,
        endpoint_weight_sha256: str,
    ) -> "LifetimeAnchor":
        nll = tuple(float(value) for value in target_new_nll_by_context)
        margin = tuple(float(value) for value in margin_by_context)
        payload = {
            "request_sha256": request_sha256,
            "case_id": int(case_id),
            "collision_sha256": collision_sha256,
            "round_index": round_index,
            "target_new_nll_by_context": list(nll),
            "margin_by_context": list(margin),
            "context_provenance_sha256": context_provenance_sha256,
            "endpoint_weight_sha256": endpoint_weight_sha256,
            "collision_policy": COLLISION_POLICY,
        }
        return cls(
            request_sha256,
            int(case_id),
            collision_sha256,
            round_index,
            nll,
            margin,
            context_provenance_sha256,
            endpoint_weight_sha256,
            canonical_hash(payload),
        )


@dataclass(slots=True)
class LifetimeAnchorLedger:
    anchors: list[LifetimeAnchor]
    finalized: MutableMapping[str, str]
    batch_size: int = BATCH_SIZE

    @classmethod
    def empty(cls, *, batch_size: int = BATCH_SIZE) -> "LifetimeAnchorLedger":
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ODEBFContractError("lifetime anchor batch size differs")
        return cls([], {}, batch_size)

    def identity(self) -> str:
        return canonical_hash(
            {
                "anchors": [item.identity_sha256 for item in self.anchors],
                "finalized": sorted(self.finalized.items()),
                "collision_policy": COLLISION_POLICY,
            }
        )

    def append_once(self, transaction_id: str, anchors: Sequence[LifetimeAnchor]) -> int:
        batch = tuple(anchors)
        if len(batch) != self.batch_size or len({item.request_sha256 for item in batch}) != self.batch_size:
            raise ODEBFContractError("lifetime anchor append is not distinct B10")
        payload = canonical_hash([item.identity_sha256 for item in batch])
        if transaction_id in self.finalized:
            if self.finalized[transaction_id] != payload:
                raise ODEBFStateError("lifetime anchor transaction identity was reused")
            return 0
        self.anchors.extend(batch)
        self.finalized[transaction_id] = payload
        return self.batch_size

    def restore(self, anchors: Sequence[LifetimeAnchor], finalized: Mapping[str, str]) -> None:
        self.anchors = list(anchors)
        self.finalized = dict(finalized)


@dataclass(frozen=True, slots=True)
class HRouteTelemetry:
    step_index: int
    history_width: int
    status: str
    h_selected: float
    h_neutral: float
    h_delta: float
    selected_minus_neutral_norm: float
    allocation_entropy: float
    maximum_layer_velocity: float
    h_disabled_velocity: tuple[float, ...]
    h_aware_velocity: tuple[float, ...]
    strength_residual: float
    energy_violation: float
    p_violation: float
    selected_action_sha256: str
    selected_action_summary: Mapping[str, float]
    selected_strength: float
    strength_limit: float
    selected_energy: float
    energy_limit: float
    selected_p: float
    p_limit: float
    external_tolerances: Mapping[str, float]
    solver_stages: Mapping[str, Mapping[str, Any]]
    polish_status: str
    post_energy_status: str
    post_energy_warning_magnitude: float
    post_energy_relative_excess: float
    post_energy_warning_count_cumulative_in_batch: int
    post_energy_warning_magnitude_cumulative_in_batch: float
    certificate_first_false_gate: str | None
    certificate_hard_gate_status: str
    identity_sha256: str


class SequentialHRouteHardFailure(ODEBFStateError):
    """Fail-close exception carrying the pre-conjunction raw-free receipt."""

    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.raw_free_receipt = dict(receipt)


class SequentialHRouter:
    """Five-dimensional H-first router with frozen R52 strength semantics."""

    def __init__(
        self,
        history_width: int,
        atomic_solver: Any,
        *,
        postsolve_energy_warn_enabled: bool = False,
        telemetry_sink: Callable[[HRouteTelemetry], None] | None = None,
    ) -> None:
        self.history_width = int(history_width)
        self.atomic_solver = atomic_solver
        self.postsolve_energy_warn_enabled = bool(postsolve_energy_warn_enabled)
        self.telemetry_sink = telemetry_sink
        self.receipts: list[HRouteTelemetry] = []
        self.post_energy_warning_count = 0
        self.post_energy_warning_magnitude = 0.0

    def _append_receipt(self, payload: Mapping[str, Any]) -> None:
        receipt = HRouteTelemetry(
            **payload,
            identity_sha256=canonical_hash(payload),
        )
        self.receipts.append(receipt)
        if self.telemetry_sink is not None:
            self.telemetry_sink(receipt)

    @staticmethod
    def _stage_receipt(stage: Any) -> dict[str, Any]:
        message = str(stage.message)
        value = np.asarray(stage.x, dtype=np.float64)
        return {
            "success": bool(stage.success),
            "status": int(stage.status),
            "iterations": int(stage.nit),
            "message_identity_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
            "candidate_sha256": canonical_hash(value.tolist()) if np.isfinite(value).all() else canonical_hash({"shape": list(value.shape), "finite": False}),
        }

    def _solver_failure(
        self,
        *,
        stage_name: str,
        solver_stages: Mapping[str, Mapping[str, Any]],
    ) -> SequentialHRouteHardFailure:
        payload = {
            "schema": "ode-edit-s05-p1r52-sequential-h-hard-failure/v1",
            "status": "HARD_FAIL",
            "first_false_gate": "SOLVER_STAGE_SUCCESS",
            "failing_solver_stage": stage_name,
            "solver_stages": dict(solver_stages),
            "selected_action_status": "NOT_AVAILABLE_SOLVER_STAGE_FAILURE",
            "selected_minimum": None,
            "strength_residual": None,
            "energy_residual": None,
            "p_residual": None,
            "selected_strength": None,
            "selected_energy": None,
            "selected_p": None,
            "strength_limit": None,
            "energy_limit": None,
            "p_limit": None,
            "external_tolerances": {
                "strength_residual": SIMPLEX_PRIMAL_TOLERANCE,
                "postsolve_energy_residual": SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
                "pretrained_p_residual": SIMPLEX_PRIMAL_TOLERANCE,
                "selected_nonnegative": SIMPLEX_PRIMAL_TOLERANCE,
            },
            "polish_status": "NOT_APPLICABLE",
            "post_energy_status": "NOT_EVALUATED",
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return SequentialHRouteHardFailure(
            f"sequential {stage_name} solver failed",
            payload,
        )

    @staticmethod
    def _entropy(value: np.ndarray) -> float:
        total = float(value.sum())
        if total <= 0.0:
            return 0.0
        probability = value[value > 0.0] / total
        return float(-np.sum(probability * np.log(probability)))

    def solve(self, problem: RoutingProblem, *, arm: FixedE8Arm | str, alpha_req: float) -> P1R43RoutingResult:
        from .p1r52_piru_postenergy_warn import classify_h_postsolve_certificate

        requested = FixedE8Arm(arm)
        atomic_soft = self.atomic_solver(problem, arm=requested, alpha_req=alpha_req)
        atomic_neutral = self.atomic_solver(problem, arm=FixedE8Arm.NEUTRAL, alpha_req=alpha_req)
        neutral = np.asarray(atomic_neutral.velocity, dtype=np.float64)
        disabled = np.asarray(atomic_soft.velocity, dtype=np.float64)
        slopes = np.asarray(problem.signed_progress, dtype=np.float64)
        active = np.flatnonzero(slopes > 0.0)
        solver_stages: dict[str, Mapping[str, Any]] = {
            name: {
                "success": True,
                "status": 0,
                "iterations": 0,
                "message_identity_sha256": canonical_hash("NOT_APPLICABLE"),
                "candidate_sha256": canonical_hash("NOT_APPLICABLE"),
            }
            for name in ("H", "P", "CAPACITY")
        }
        certificate_decision: dict[str, Any] | None = None
        if self.history_width == 0 or alpha_req == 0.0 or active.size <= 1:
            selected = disabled
            status = "H_EMPTY_EXACT_ATOMIC_EQUIVALENCE" if self.history_width == 0 else "NO_ROUTING_DOF"
            result = atomic_soft
            certificates: tuple[Mapping[str, Any], ...] = ()
        else:
            neutral_energy = float(neutral @ problem.trust_metric @ neutral)
            energy_limit = neutral_energy * (1.0 + SIMPLEX_ENERGY_RELATIVE_TOLERANCE) + SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
            p_limit = problem.pretrained.value(neutral) + SIMPLEX_XI_TIE_TOLERANCE * max(
                float(np.trace(problem.pretrained.gram)), 1.0
            )
            transform = alpha_req / slopes[active]

            def expand(pi: np.ndarray) -> np.ndarray:
                value = np.zeros_like(slopes)
                value[active] = transform * pi
                return value

            def h_value(pi: np.ndarray) -> float:
                return problem.historical.value(expand(pi))

            def p_value(pi: np.ndarray) -> float:
                return problem.pretrained.value(expand(pi))

            def energy(pi: np.ndarray) -> float:
                value = expand(pi)
                return float(value @ problem.trust_metric @ value)

            def capacity(pi: np.ndarray) -> float:
                value = expand(pi)
                return float(0.5 * value @ problem.capacity_metric @ value)

            seed = np.asarray(atomic_soft.pi, dtype=np.float64)[active]
            if not np.isfinite(seed).all() or abs(float(seed.sum()) - 1.0) > SIMPLEX_PRIMAL_TOLERANCE:
                seed = np.asarray(atomic_neutral.pi, dtype=np.float64)[active]
            base_constraints = (
                {"type": "eq", "fun": lambda pi: float(pi.sum() - 1.0)},
                {"type": "ineq", "fun": lambda pi: float(energy_limit - energy(pi))},
                {"type": "ineq", "fun": lambda pi: float(p_limit - p_value(pi))},
            )
            options = {"disp": False, "ftol": SOLVER_FTOL, "maxiter": SOLVER_MAXITER}
            stage_h = minimize(h_value, seed, method="SLSQP", bounds=tuple((0.0, None) for _ in active), constraints=base_constraints, options=options)
            solver_stages["H"] = self._stage_receipt(stage_h)
            if not stage_h.success:
                raise self._solver_failure(
                    stage_name="H",
                    solver_stages=solver_stages,
                )
            pi_h = np.asarray(stage_h.x, dtype=np.float64)
            h_star = h_value(pi_h)
            h_tie = SIMPLEX_XI_TIE_TOLERANCE * max(abs(h_star), float(np.trace(problem.historical.gram)), 1.0)
            stage_p = minimize(
                p_value,
                pi_h,
                method="SLSQP",
                bounds=tuple((0.0, None) for _ in active),
                constraints=(*base_constraints, {"type": "ineq", "fun": lambda pi: float(h_star + h_tie - h_value(pi))}),
                options=options,
            )
            solver_stages["P"] = self._stage_receipt(stage_p)
            if not stage_p.success:
                raise self._solver_failure(
                    stage_name="P",
                    solver_stages=solver_stages,
                )
            pi_p = np.asarray(stage_p.x, dtype=np.float64)
            p_star = p_value(pi_p)
            p_tie = SIMPLEX_XI_TIE_TOLERANCE * max(abs(p_star), float(np.trace(problem.pretrained.gram)), 1.0)
            stage_c = minimize(
                capacity,
                pi_p,
                method="SLSQP",
                bounds=tuple((0.0, None) for _ in active),
                constraints=(
                    *base_constraints,
                    {"type": "ineq", "fun": lambda pi: float(h_star + h_tie - h_value(pi))},
                    {"type": "ineq", "fun": lambda pi: float(p_star + p_tie - p_value(pi))},
                ),
                options=options,
            )
            solver_stages["CAPACITY"] = self._stage_receipt(stage_c)
            if not stage_c.success:
                raise self._solver_failure(
                    stage_name="CAPACITY",
                    solver_stages=solver_stages,
                )
            selected = expand(np.asarray(stage_c.x, dtype=np.float64))
            strength_residual = abs(float(slopes @ selected) - alpha_req)
            energy_violation = max(float(selected @ problem.trust_metric @ selected) - energy_limit, 0.0)
            p_violation = max(problem.pretrained.value(selected) - p_limit, 0.0)
            certificate_decision = classify_h_postsolve_certificate(
                solver_stage_success={
                    name: bool(value["success"])
                    for name, value in solver_stages.items()
                },
                strength_residual=strength_residual,
                energy_residual=energy_violation,
                p_residual=p_violation,
                selected_minimum=float(np.min(selected)),
                primal_tolerance=SIMPLEX_PRIMAL_TOLERANCE,
                energy_tolerance=SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
                postsolve_energy_warn_enabled=self.postsolve_energy_warn_enabled,
            )
            if certificate_decision["status"] == "HARD_FAIL":
                hard_receipt = {
                    "schema": "ode-edit-s05-p1r52-sequential-h-hard-failure/v1",
                    **certificate_decision,
                    "selected_action_sha256": canonical_hash(selected.tolist()),
                    "selected_action_summary": {
                        "minimum": float(np.min(selected)),
                        "maximum": float(np.max(selected)),
                        "l1_norm": float(np.linalg.norm(selected, ord=1)),
                        "l2_norm": float(np.linalg.norm(selected)),
                    },
                    "selected_strength": float(slopes @ selected),
                    "strength_limit": float(alpha_req),
                    "selected_energy": float(selected @ problem.trust_metric @ selected),
                    "energy_limit": float(energy_limit),
                    "selected_p": float(problem.pretrained.value(selected)),
                    "p_limit": float(p_limit),
                    "solver_stages": solver_stages,
                    "polish_status": "NOT_APPLICABLE",
                }
                hard_receipt.pop("identity_sha256", None)
                hard_receipt["identity_sha256"] = canonical_hash(hard_receipt)
                raise SequentialHRouteHardFailure(
                    "sequential H routing certificate failed",
                    hard_receipt,
                )
            if certificate_decision["status"] == "WARN_POSTSOLVE_ENERGY_RESIDUAL":
                self.post_energy_warning_count += 1
                self.post_energy_warning_magnitude += energy_violation
            selected_pi = np.zeros_like(slopes)
            selected_pi[active] = np.asarray(stage_c.x, dtype=np.float64)
            base = atomic_soft.selected
            certificate = {
                "phase": "H_THEN_P_THEN_CAPACITY",
                "history_width": self.history_width,
                "h_success": bool(stage_h.success),
                "p_success": bool(stage_p.success),
                "capacity_success": bool(stage_c.success),
                "strength_residual": strength_residual,
                "energy_violation": energy_violation,
                "post_energy_status": certificate_decision["status"],
                "post_energy_warning_decision_influence_count": 0,
                "p_violation": p_violation,
                "legacy_h_budget_influence_count": 0,
                "legacy_p_budget_influence_count": 0,
            }
            selected_result: P1R24RoutingResult = replace(
                base,
                arm=requested,
                status=P1R24RoutingStatus.JOINT_WRITE,
                soft_pi=tuple(float(value) for value in selected_pi),
                pi=tuple(float(value) for value in selected_pi),
                soft_velocity=tuple(float(value) for value in selected),
                velocity=tuple(float(value) for value in selected),
                predicted_progress=float(slopes @ selected),
                equality_residual=strength_residual,
                selected_energy=float(selected @ problem.trust_metric @ selected),
                soft_p=problem.pretrained.value(selected),
                selected_p=problem.pretrained.value(selected),
                selected_capacity=float(0.5 * selected @ problem.capacity_metric @ selected),
                certificates=(*base.certificates, certificate),
                identity_sha256=canonical_hash({"base": base.identity_sha256, "certificate": certificate, "velocity": selected.tolist()}),
            )
            result = P1R43RoutingResult(
                requested,
                selected_result,
                False,
                None,
                canonical_hash({"selected": selected_result.identity_sha256, "history_width": self.history_width}),
            )
            status = "H_ACTIVE_CERTIFIED" if np.linalg.norm(selected - disabled) > 1.0e-10 else "H_NO_DECISION_INFLUENCE"
            certificates = (certificate,)
        del certificates
        h_selected = problem.historical.value(selected)
        h_neutral = problem.historical.value(neutral)
        strength_residual = abs(float(slopes @ selected) - alpha_req) if alpha_req > 0.0 else abs(float(slopes @ selected))
        energy_limit = float(neutral @ problem.trust_metric @ neutral) * (1.0 + SIMPLEX_ENERGY_RELATIVE_TOLERANCE) + SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
        p_limit = problem.pretrained.value(neutral) + SIMPLEX_XI_TIE_TOLERANCE * max(float(np.trace(problem.pretrained.gram)), 1.0)
        selected_energy = float(selected @ problem.trust_metric @ selected)
        selected_p = float(problem.pretrained.value(selected))
        energy_violation = max(selected_energy - energy_limit, 0.0)
        p_violation = max(selected_p - p_limit, 0.0)
        if certificate_decision is None:
            certificate_decision = {
                "status": "NOT_APPLICABLE",
                "first_false_gate": None,
            }
        telemetry_payload = {
            "step_index": len(self.receipts),
            "history_width": self.history_width,
            "status": status,
            "h_selected": h_selected,
            "h_neutral": h_neutral,
            "h_delta": h_selected - h_neutral,
            "selected_minus_neutral_norm": float(np.linalg.norm(selected - neutral)),
            "allocation_entropy": self._entropy(selected),
            "maximum_layer_velocity": float(np.max(selected, initial=0.0)),
            "h_disabled_velocity": disabled.tolist(),
            "h_aware_velocity": selected.tolist(),
            "strength_residual": strength_residual,
            "energy_violation": energy_violation,
            "p_violation": p_violation,
            "selected_action_sha256": canonical_hash(selected.tolist()),
            "selected_action_summary": {
                "minimum": float(np.min(selected)),
                "maximum": float(np.max(selected)),
                "l1_norm": float(np.linalg.norm(selected, ord=1)),
                "l2_norm": float(np.linalg.norm(selected)),
            },
            "selected_strength": float(slopes @ selected),
            "strength_limit": float(alpha_req),
            "selected_energy": selected_energy,
            "energy_limit": energy_limit,
            "selected_p": selected_p,
            "p_limit": p_limit,
            "external_tolerances": {
                "strength_residual": SIMPLEX_PRIMAL_TOLERANCE,
                "postsolve_energy_residual": SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
                "pretrained_p_residual": SIMPLEX_PRIMAL_TOLERANCE,
                "selected_nonnegative": SIMPLEX_PRIMAL_TOLERANCE,
            },
            "solver_stages": solver_stages,
            "polish_status": "NOT_APPLICABLE",
            "post_energy_status": str(certificate_decision["status"]),
            "post_energy_warning_magnitude": (
                energy_violation
                if certificate_decision["status"] == "WARN_POSTSOLVE_ENERGY_RESIDUAL"
                else 0.0
            ),
            "post_energy_relative_excess": (
                energy_violation / max(abs(energy_limit), np.finfo(np.float64).tiny)
                if certificate_decision["status"] == "WARN_POSTSOLVE_ENERGY_RESIDUAL"
                else 0.0
            ),
            "post_energy_warning_count_cumulative_in_batch": self.post_energy_warning_count,
            "post_energy_warning_magnitude_cumulative_in_batch": self.post_energy_warning_magnitude,
            "certificate_first_false_gate": certificate_decision.get("first_false_gate"),
            "certificate_hard_gate_status": (
                "PASS"
                if certificate_decision["status"] != "HARD_FAIL"
                else "FAIL"
            ),
        }
        self._append_receipt(telemetry_payload)
        return result


def build_history_aware_scalable_field(
    history_state: SequentialArmState,
    model: Any,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    *,
    target_state: torch.Tensor,
    current_terminal: torch.Tensor,
    captured_keys_by_layer: Mapping[int, torch.Tensor],
    accepted_waypoint: int,
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    residual_tolerance: float,
    ledger: ComputeLedger,
    allow_zero_capacity: bool = False,
    maximum_history_columns: int = HISTORY_COUNTS[-1],
) -> Any:
    layers = tuple(int(layer) for layer in hparams.layers)
    if layers != P1R23_LAYER_ORDER:
        raise ODEBFContractError("sequential field layer inventory differs")
    # scalable_terminal_residual requires the canonical ordered digest.
    from .scalable_batched_runtime import scalable_ordered_request_digest

    residual = scalable_terminal_residual(
        target_state,
        current_terminal,
        scalable_ordered_request_digest([str(item["request_sha256"]) for item in requests]),
    )
    solve = {layer: history_state.ledger.solve_keys(layer) for layer in layers}
    risk = {layer: history_state.ledger.risk_keys(layer) for layer in layers}
    field = build_p1_dynamic_field(
        model,
        tokenizer,
        requests,
        hparams,
        projector,
        contexts,
        target_state=target_state,
        accepted_waypoint=accepted_waypoint,
        cumulative_factors_by_weight={},
        history_solve_keys_by_layer=solve,
        history_risk_keys_by_layer=risk,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256,
        residual_tolerance=residual_tolerance,
        ledger=ledger,
        residual_policy=SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
        allow_zero_capacity=allow_zero_capacity,
        shared_terminal_residual=residual,
        captured_keys_by_layer=captured_keys_by_layer,
        allow_inner_empty_cache=False,
        expected_batch_size=len(requests),
        maximum_history_columns=maximum_history_columns,
    )
    expected = history_state.history_entry_count()
    if any(item.history_action.shape[1] != expected for item in field.layers):
        raise ODEBFStateError("sequential history risk width did not enter field")
    return field


@contextmanager
def scoped_atomic_sequential_adapter(
    experiment_module: Any,
    history_state: SequentialArmState,
    router: SequentialHRouter,
    *,
    structural_h_decision_enabled: bool = True,
    pir_history_rebind_enabled: bool = False,
    maximum_history_columns: int = HISTORY_COUNTS[-1],
) -> Iterator[None]:
    """Patch one B10 while keeping Alpha solve history separate from H routing."""

    original_field = experiment_module.build_scalable_dynamic_field
    original_disable = experiment_module.p1r24_disable_historical
    original_solver = experiment_module.solve_p1r43_full_strength_routing
    original_pir_planner = (
        experiment_module.plan_pir_writer
        if pir_history_rebind_enabled
        else None
    )
    original_from_field = AcceptedLayerContribution.__dict__["from_field"]

    def field_adapter(*args: Any, **kwargs: Any) -> Any:
        return build_history_aware_scalable_field(
            history_state,
            *args,
            maximum_history_columns=maximum_history_columns,
            **kwargs,
        )

    def from_field_adapter(cls: type[AcceptedLayerContribution], field: Any, factor: Any, *, history_action: torch.Tensor | None = None) -> AcceptedLayerContribution:
        del history_action
        return original_from_field.__func__(cls, field, factor, history_action=field.history_action)

    def pir_planner_adapter(*args: Any, **kwargs: Any) -> Any:
        from .p1r52_piru_sequential_adapter import bind_piru_sequential_history

        if original_pir_planner is None:
            raise ODEBFContractError("P1R52 PIR-U sequential planner is absent")
        result = original_pir_planner(*args, **kwargs)
        entry_field = kwargs.get("entry_field")
        if entry_field is None:
            raise ODEBFContractError("P1R52 PIR-U sequential entry field is absent")
        return bind_piru_sequential_history(result, entry_field)

    experiment_module.build_scalable_dynamic_field = field_adapter
    experiment_module.p1r24_disable_historical = (
        (lambda problem: problem)
        if structural_h_decision_enabled
        else original_disable
    )
    experiment_module.solve_p1r43_full_strength_routing = router.solve
    if pir_history_rebind_enabled:
        experiment_module.plan_pir_writer = pir_planner_adapter
    AcceptedLayerContribution.from_field = classmethod(from_field_adapter)
    try:
        yield
    finally:
        experiment_module.build_scalable_dynamic_field = original_field
        experiment_module.p1r24_disable_historical = original_disable
        experiment_module.solve_p1r43_full_strength_routing = original_solver
        if pir_history_rebind_enabled:
            experiment_module.plan_pir_writer = original_pir_planner
        AcceptedLayerContribution.from_field = original_from_field


def make_history_records(
    requests: Sequence[Mapping[str, Any]],
    collision_by_request: Mapping[str, str],
    *,
    version: int,
    terminal_event_sha256: str,
) -> tuple[P1HistoryRecord, ...]:
    result: list[P1HistoryRecord] = []
    for request in requests:
        request_sha = str(request["request_sha256"])
        target_sha = hashlib.sha256(str(request["target_new"]).encode("utf-8")).hexdigest()
        result.append(
            P1HistoryRecord(
                request_sha,
                int(request["case_id"]),
                collision_by_request[request_sha],
                target_sha,
                terminal_event_sha256,
                version,
            )
        )
    return tuple(result)


def capture_controller_legal_lifetime_anchors(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    *,
    alias: str,
    round_index: int,
    collision_by_request: Mapping[str, str],
    endpoint_weight_sha256: str,
) -> tuple[tuple[LifetimeAnchor, ...], dict[str, Any]]:
    """Capture full-six new/old NLL anchors without held-out surfaces."""

    templates = tuple(template for group in contexts for template in group)
    if tuple(len(group) for group in contexts) != (1, 5) or len(templates) != 6:
        raise ODEBFContractError("lifetime anchor controller contexts differ")
    llama = alias == "llama3-8b-inst"
    device = next(model.parameters()).device
    anchors: list[LifetimeAnchor] = []
    processed = 0
    with torch.no_grad():
        for request in requests:
            prefixes = tuple(
                _surface(request, "target_new", context_template=template)[0]
                for template in templates
            )
            values, tokens, _ = _evaluate_prefixes_pinned_with_accuracy(
                model,
                tokenizer,
                prefixes,
                str(request["target_new"]),
                str(request["target_true"]),
                llama=llama,
                device=device,
            )
            request_sha = str(request["request_sha256"])
            anchors.append(
                LifetimeAnchor.create(
                    request_sha256=request_sha,
                    case_id=int(request["case_id"]),
                    collision_sha256=collision_by_request[request_sha],
                    round_index=round_index,
                    target_new_nll_by_context=[item.nll.target_new_nll for item in values],
                    margin_by_context=[item.nll.target_true_nll - item.nll.target_new_nll for item in values],
                    context_provenance_sha256=canonical_hash(
                        {
                            "request_sha256": request_sha,
                            "controller_context_sha256": [
                                hashlib.sha256(prefix.encode("utf-8")).hexdigest()
                                for prefix in prefixes
                            ],
                            "heldout_context_count": 0,
                        }
                    ),
                    endpoint_weight_sha256=endpoint_weight_sha256,
                )
            )
            processed += tokens
    payload = {
        "round": round_index,
        "anchor_count": len(anchors),
        "full_six_context_count": 6,
        "controller_legal_context_only": True,
        "heldout_context_count": 0,
        "model_forward_count": len(requests),
        "backward_count": 0,
        "generation_call_count": 0,
        "processed_token_count": processed,
        "anchor_sha256": [item.identity_sha256 for item in anchors],
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return tuple(anchors), payload


def commit_sequential_batch(
    parameters: Mapping[str, torch.nn.Parameter],
    candidates: Mapping[str, torch.Tensor],
    *,
    mutation_lock: Any,
    transaction_id: str,
    history_state: SequentialArmState,
    prospective: Any,
    load_increment_by_layer: Mapping[int, float],
    anchor_ledger: LifetimeAnchorLedger,
    anchor_factory: Any,
    history_factory: Any | None = None,
    fault_phase: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """Commit W/history/anchors exactly once with complete enclosing rollback."""

    if (
        fault_phase not in (None, "after_weights", "after_anchor", "after_history")
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
        or anchor_ledger.batch_size != batch_size
        or history_state.ledger.batch_size != batch_size
    ):
        raise ODEBFContractError("sequential transaction fault phase differs")
    if set(parameters) != set(candidates):
        raise ODEBFContractError("sequential transaction weight inventory differs")
    if (prospective is None) == (history_factory is None):
        raise ODEBFContractError("sequential history preparation source differs")
    entry_values = {
        name: parameter.detach().to(device="cpu").clone()
        for name, parameter in sorted(parameters.items())
    }
    entry_pointers = {name: int(parameter.data_ptr()) for name, parameter in parameters.items()}
    history_checkpoint = history_state.ledger.transaction_checkpoint()
    anchor_checkpoint = tuple(anchor_ledger.anchors)
    finalized_checkpoint = dict(anchor_ledger.finalized)
    expected = {name: tensor_sha256(value) for name, value in candidates.items()}
    try:
        with mutation_lock:
            transaction = AtomicBatchTransaction(
                parameters,
                transaction_id=transaction_id,
                mutation_lock=mutation_lock,
            )
            for name in sorted(candidates):
                transaction.stage(name, candidates[name])
            weight_receipt = transaction.commit(
                post_commit_verify=lambda: _weight_hashes(parameters) == expected
            )
            if fault_phase == "after_weights":
                raise RuntimeError("injected sequential fault after weights")
            if history_factory is not None:
                prospective, history_capture_receipt = history_factory()
            else:
                history_capture_receipt = {
                    "status": "PREPARED_BEFORE_COMMIT_TEST_ONLY",
                    "post_commit_capture": False,
                }
            anchors, anchor_receipt = anchor_factory()
            anchor_append = anchor_ledger.append_once(transaction_id, anchors)
            if fault_phase == "after_anchor":
                raise RuntimeError("injected sequential fault after anchors")
            history_receipt = history_state.ledger.finalize(
                prospective,
                post_commit_verified=True,
                load_increment_by_layer=load_increment_by_layer,
            )
            if fault_phase == "after_history":
                raise RuntimeError("injected sequential fault after history")
            history_replay = history_state.ledger.finalize(
                prospective,
                post_commit_verified=True,
                load_increment_by_layer=load_increment_by_layer,
            )
            anchor_replay = anchor_ledger.append_once(transaction_id, anchors)
            if (
                anchor_append != batch_size
                or history_replay.appended_count != 0
                or not history_replay.idempotent_replay
                or anchor_replay != 0
            ):
                raise ODEBFStateError("sequential transaction idempotence differs")
    except BaseException:
        history_state.ledger.restore_transaction_checkpoint(history_checkpoint)
        anchor_ledger.restore(anchor_checkpoint, finalized_checkpoint)
        with mutation_lock, torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(entry_values[name].to(device=parameter.device))
        if (
            any(int(parameters[name].data_ptr()) != entry_pointers[name] for name in parameters)
            or any(
                not torch.equal(parameters[name].detach().to(device="cpu"), entry_values[name])
                for name in parameters
            )
            or history_state.ledger.snapshot().digest != history_checkpoint.state_sha256
            or tuple(anchor_ledger.anchors) != anchor_checkpoint
            or anchor_ledger.finalized != finalized_checkpoint
        ):
            raise ODEBFStateError("sequential enclosing rollback differs")
        raise
    payload = {
        "transaction_id": transaction_id,
        "weight": asdict(weight_receipt),
        "history": asdict(history_receipt),
        "history_capture": history_capture_receipt,
        "anchor_capture": anchor_receipt,
        "anchor_append_count": anchor_append,
        "idempotent_history_replay_append_count": history_replay.appended_count,
        "idempotent_anchor_replay_append_count": anchor_replay,
        "post_commit_weight_sha256": _weight_hashes(parameters),
        "post_commit_history_sha256": history_state.ledger.snapshot().digest,
        "post_commit_anchor_sha256": anchor_ledger.identity(),
        "fault_phase": fault_phase,
        "rollback_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def dry_plan() -> dict[str, Any]:
    payload = {
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "model": "llama3-8b-inst",
        "arms": ["P1R52_SOFT_SEQUENTIAL_H", "NATIVE_ALPHAEDIT_SEQUENTIAL"],
        "rounds": ROUND_COUNT,
        "batch_size": BATCH_SIZE,
        "requests": ROUND_COUNT * BATCH_SIZE,
        "k": FIXED_K,
        "h": P1R23_H,
        "history_width_at_entry": list(HISTORY_COUNTS),
        "controller_reset_each_round": True,
        "interbatch_w0_restore_count": 0,
        "terminal_w0_restore_count": 1,
        "inner_heldout_evaluation_count": 0,
        "batch_entry_pre_evaluations": ROUND_COUNT,
        "batch_entry_pre_evaluator_added_forward_count": ROUND_COUNT * BATCH_SIZE,
        "batch_entry_pre_evaluator_added_backward_generation": [0, 0],
        "batch_entry_pre_decision_influence_count": 0,
        "postcommit_checkpoint_evaluations": sum(range(1, ROUND_COUNT + 1)),
        "full_b100_terminal_cohort_evaluations": ROUND_COUNT,
        "pre_post_final_machine_tables": [
            "b1-b10-pre-post-checkpoints.json",
            "entry-pre-immediate-post-final-w10-requests.json",
            "batch-age-pre-post-final-cohorts.json",
            "true-entry-post-final-aggregate.json",
        ],
        "sequential_accuracy_added_forward_backward_generation": [0, 0, 0],
        "retry_backtracking_first_hit": [0, 0, 0],
        "server1_gpu_max": 2,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "COLLISION_POLICY",
    "HISTORY_COUNTS",
    "INSTRUCTION_ID",
    "LifetimeAnchor",
    "LifetimeAnchorLedger",
    "METHOD_ID",
    "ROUND_COUNT",
    "SequentialHRouter",
    "assemble_terminal_candidates",
    "build_history_aware_scalable_field",
    "capture_controller_legal_lifetime_anchors",
    "commit_sequential_batch",
    "dry_plan",
    "make_history_records",
    "scoped_atomic_sequential_adapter",
]
