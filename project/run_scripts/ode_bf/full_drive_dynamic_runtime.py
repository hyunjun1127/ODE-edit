"""P1R17 full-drive Dynamic-target field construction.

The module deliberately owns only the refreshed field and one-step routing
contract.  Panel setup, post-freeze evaluation, and job plumbing remain
separate so that the physical slope/routing block can be reused and tested
without model execution or an outcome-dependent lambda.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .common_cold_coordinate import (
    CommonColdScale,
    CommonColdScaleMetric,
    common_terminal_residual_input,
    write_aware_common_target_velocity,
)
from .common_coldcoord_fixed_e8_runtime import (
    _common_field_semantic_receipt,
)
from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .fixed_e8_soft_routing import FIXED_E8_H, FIXED_E8_LAYER_ORDER
from .functional import WaypointFactor, tensor_sha256
from .full_drive_physical_field import (
    PhysicalEditSlopeReceipt,
    measure_physical_edit_slopes,
    physical_slope_problem_adapter,
)
from .full_drive_soft_routing import (
    FULL_DRIVE_METHOD_ID,
    FullDriveArm,
    FullDriveRoutingResult,
    FullDriveStepMode,
    solve_full_drive_routing,
)
from .p1_adaptive_runtime import (
    _factor_state,
    _history_actions,
    _history_keys,
    _parameter_contract_sha256,
)
from .p1_backend import (
    PinnedCovarianceRegistry,
    SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
    build_p1_dynamic_field,
)
from .p1_controller import AcceptedLayerContribution, P1ControllerLock
from .p1_replay import Theta0TeacherCache
from .p1_state import P1HistoryLedger
from .request_digest import ordered_request_digest_v1
from .sampling import StatelessReplaySchedule
from . import fixed_e8_runtime as legacy


class FullDriveAllocation(str, Enum):
    ROBUST_SHARED = "RS"
    BATCH_GLOBAL = "BG"

    @property
    def scale(self) -> CommonColdScale:
        return (
            CommonColdScale.ROBUST_SHARED
            if self is FullDriveAllocation.ROBUST_SHARED
            else CommonColdScale.BATCH_GLOBAL
        )


class FullDriveCell(str, Enum):
    RS_FULL_NOSOFT = "RS-FULL-NOSOFT"
    RS_FULL_SOFT = "RS-FULL-SOFT"
    BG_FULL_NOSOFT = "BG-FULL-NOSOFT"
    BG_FULL_SOFT = "BG-FULL-SOFT"

    @property
    def allocation(self) -> FullDriveAllocation:
        return (
            FullDriveAllocation.ROBUST_SHARED
            if self.value.startswith("RS-")
            else FullDriveAllocation.BATCH_GLOBAL
        )

    @property
    def routing_arm(self) -> FullDriveArm:
        return (
            FullDriveArm.SOFT
            if self.value.endswith("-SOFT")
            else FullDriveArm.NO_SOFT
        )


FULL_DRIVE_CELL_ORDER = (
    FullDriveCell.RS_FULL_NOSOFT,
    FullDriveCell.RS_FULL_SOFT,
    FullDriveCell.BG_FULL_NOSOFT,
    FullDriveCell.BG_FULL_SOFT,
)


@dataclass(frozen=True, slots=True)
class FullDriveFieldBundle:
    cell: FullDriveCell
    step_index: int
    field: Any
    physical_slope: PhysicalEditSlopeReceipt
    problem_receipt: Any
    inventory: Any
    functional_probe: Mapping[str, Any]
    routing: FullDriveRoutingResult
    target_velocity: torch.Tensor
    target_velocity_receipt: Mapping[str, Any]
    residual_receipt: Mapping[str, Any]
    field_semantic_receipt: Mapping[str, Any]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "ode-edit-s05-p1r17-full-drive-field/v1",
            "method_id": FULL_DRIVE_METHOD_ID,
            "cell": self.cell.value,
            "allocation": self.cell.allocation.value,
            "target_dynamics": "DYNAMIC_TARGET",
            "step_index": self.step_index,
            "field": self.field.raw_free_payload(),
            "field_semantic": dict(self.field_semantic_receipt),
            "residual": dict(self.residual_receipt),
            "physical_w_only_slope": self.physical_slope.raw_free_payload(),
            "problem_receipt_sha256": self.problem_receipt.identity_sha256,
            "problem_signed_progress": [
                float(item) for item in self.problem_receipt.problem.signed_progress
            ],
            "problem_slope_equals_physical_endpoint_slope": True,
            "functional_inventory": self.inventory.raw_free_payload(),
            "functional_probe": dict(self.functional_probe),
            "routing": self.routing.raw_free_payload(),
            "target_velocity": dict(self.target_velocity_receipt),
            "target_velocity_sha256": tensor_sha256(self.target_velocity),
            "proposal_target_state_is_current_state": True,
            "stale_or_unrelated_desired_target_access_count": 0,
            "progress_floor": 0.0,
            "kappa": 0.0,
            "requested_progress": None,
            "hard_h_p_budget_influence_count": 0,
            "functional_veto_count": 0,
            "transport_decision_influence_count": 0,
            "scientific_retry_count": 0,
            "target_hold_access_count": 0,
            "native_or_direct_z_controller_access_count": 0,
        }
        if canonical_hash(payload) != self.identity_sha256:
            raise ODEBFContractError("full-drive field identity differs")
        return payload


@dataclass(frozen=True, slots=True)
class FullDriveTransitionProposal:
    step_index: int
    mode: FullDriveStepMode
    target_trial: torch.Tensor
    increment: Mapping[str, WaypointFactor]
    target_advanced: bool
    weight_advanced: bool
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "ode-edit-s05-p1r17-transition-proposal/v1",
            "step_index": self.step_index,
            "mode": self.mode.value,
            "target_trial_sha256": tensor_sha256(self.target_trial),
            "increment": {
                name: {
                    "layer": int(item.layer),
                    "theta": float(item.theta),
                    "left_sha256": tensor_sha256(item.left),
                    "right_sha256": tensor_sha256(item.right),
                }
                for name, item in self.increment.items()
            },
            "target_advanced": self.target_advanced,
            "weight_advanced": self.weight_advanced,
            "target_clock_advance": 1,
            "weight_clock_advance": 1,
            "tau_advance": float(FIXED_E8_H),
            "scientific_retry_count": 0,
            "target_and_weight_h_equal": True,
            "pmax_zero_weight_unchanged": (
                self.mode is not FullDriveStepMode.TARGET_ONLY_PMAX_ZERO
                or not self.weight_advanced
            ),
        }
        if canonical_hash(payload) != self.identity_sha256:
            raise ODEBFContractError("full-drive transition identity differs")
        return payload


def _bundle_payload(
    *,
    cell: FullDriveCell,
    step_index: int,
    field: Any,
    physical_slope: PhysicalEditSlopeReceipt,
    problem_receipt: Any,
    inventory: Any,
    functional_probe: Mapping[str, Any],
    routing: FullDriveRoutingResult,
    target_velocity: torch.Tensor,
    target_velocity_receipt: Mapping[str, Any],
    residual_receipt: Mapping[str, Any],
    field_semantic_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": "ode-edit-s05-p1r17-full-drive-field/v1",
        "method_id": FULL_DRIVE_METHOD_ID,
        "cell": cell.value,
        "allocation": cell.allocation.value,
        "target_dynamics": "DYNAMIC_TARGET",
        "step_index": step_index,
        "field": field.raw_free_payload(),
        "field_semantic": dict(field_semantic_receipt),
        "residual": dict(residual_receipt),
        "physical_w_only_slope": physical_slope.raw_free_payload(),
        "problem_receipt_sha256": problem_receipt.identity_sha256,
        "problem_signed_progress": [
            float(item) for item in problem_receipt.problem.signed_progress
        ],
        "problem_slope_equals_physical_endpoint_slope": True,
        "functional_inventory": inventory.raw_free_payload(),
        "functional_probe": dict(functional_probe),
        "routing": routing.raw_free_payload(),
        "target_velocity": dict(target_velocity_receipt),
        "target_velocity_sha256": tensor_sha256(target_velocity),
        "proposal_target_state_is_current_state": True,
        "stale_or_unrelated_desired_target_access_count": 0,
        "progress_floor": 0.0,
        "kappa": 0.0,
        "requested_progress": None,
        "hard_h_p_budget_influence_count": 0,
        "functional_veto_count": 0,
        "transport_decision_influence_count": 0,
        "scientific_retry_count": 0,
        "target_hold_access_count": 0,
        "native_or_direct_z_controller_access_count": 0,
    }
    return payload


def build_full_drive_field(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    cell: FullDriveCell | str,
    lambda_p: float,
    step_index: int,
    current_factors: Mapping[str, Sequence[WaypointFactor]],
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    capture: legacy.FixedE8EntryCapture,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    lock: P1ControllerLock,
    history: P1HistoryLedger,
    accepted_by_layer: Mapping[int, Sequence[AcceptedLayerContribution]],
    ledger: Any,
    replay_entry: Any,
    theta0_cache: Theta0TeacherCache,
    touched: Mapping[str, torch.nn.Parameter],
    schedule: StatelessReplaySchedule,
    metric: CommonColdScaleMetric,
    lookup_positions: Sequence[int],
    target_layer_name: str,
) -> FullDriveFieldBundle:
    """Build one refreshed, state-pure P1R17 field.

    The function performs no accepted-state transition.  It measures the
    physical endpoints, selects ``v``, and computes the matching Dynamic-z
    velocity at the same ``h*v`` probe.  The caller atomically applies the
    returned target and weight increments.
    """

    selected = FullDriveCell(cell)
    if metric.scale is not selected.allocation.scale:
        raise ODEBFContractError("full-drive allocation metric differs")
    if step_index not in range(8) or not math.isfinite(lambda_p) or lambda_p < 0:
        raise ODEBFContractError("full-drive step/lambda differs")
    before = (
        _parameter_contract_sha256(touched),
        history.snapshot().digest,
        schedule.state_digest,
        legacy._rng_identity(),
    )
    solve_history = _history_keys(history, FIXED_E8_LAYER_ORDER, risk=False)
    risk_history = _history_keys(history, FIXED_E8_LAYER_ORDER, risk=True)
    request_order = ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in requests]
    )
    residual_input = common_terminal_residual_input(
        current_target, current_terminal, request_order
    )
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
        residual_policy=SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
        allow_zero_capacity=False,
        shared_terminal_residual=residual_input,
    )
    physical_slope = measure_physical_edit_slopes(
        model,
        tokenizer,
        requests,
        field,
        step_index=step_index,
        cumulative_factors_by_weight=current_factors,
        contexts=contexts,
        ledger=ledger,
        touched=touched,
        history=history,
        schedule=schedule,
        rng_identity=legacy._rng_identity,
    )
    signed_adapter = physical_slope_problem_adapter(physical_slope)
    semantic = _common_field_semantic_receipt(
        field, solve_history=solve_history, risk_history=risk_history
    )
    built = legacy._build_fixed_e8_problem(
        field,
        signed_adapter,
        accepted_by_layer=accepted_by_layer,
        committed_load_by_layer=history.cumulative_load(),
        lock=lock,
        current_history_action_by_layer=_history_actions(field, history),
    )
    if not np.array_equal(
        built.problem.signed_progress,
        np.asarray(physical_slope.edit_slopes, dtype=np.float64),
    ):
        raise ODEBFContractError("full-drive endpoint/problem slope differs")
    factor_state = _factor_state(
        capture.entry_sha256, current_factors, current_target
    )
    inventory, probe = legacy._fixed_e8_functional_basis_probe(
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
        field_semantic_sha256=semantic["semantic_identity_sha256"],
    )
    routing = solve_full_drive_routing(
        built.problem,
        inventory,
        edit_slopes=physical_slope.edit_slopes,
        arm=selected.routing_arm,
        lambda_p=(lambda_p if selected.routing_arm is FullDriveArm.SOFT else 0.0),
    )
    target_velocity, target_receipt = write_aware_common_target_velocity(
        model,
        tokenizer,
        requests,
        contexts,
        target_layer_name=target_layer_name,
        lookup_positions=lookup_positions,
        field=field,
        velocity_coefficients=routing.velocity,
        cumulative_factors_by_weight=current_factors,
        metric=metric,
        ledger=ledger,
    )
    after = (
        _parameter_contract_sha256(touched),
        history.snapshot().digest,
        schedule.state_digest,
        legacy._rng_identity(),
    )
    if after != before:
        raise ODEBFStateError("full-drive field build mutated controller state")
    residual_payload = residual_input.raw_free_payload()
    payload = _bundle_payload(
        cell=selected,
        step_index=step_index,
        field=field,
        physical_slope=physical_slope,
        problem_receipt=built,
        inventory=inventory,
        functional_probe=probe,
        routing=routing,
        target_velocity=target_velocity,
        target_velocity_receipt=target_receipt,
        residual_receipt=residual_payload,
        field_semantic_receipt=semantic,
    )
    return FullDriveFieldBundle(
        selected,
        step_index,
        field,
        physical_slope,
        built,
        inventory,
        probe,
        routing,
        target_velocity,
        target_receipt,
        residual_payload,
        semantic,
        canonical_hash(payload),
    )


def propose_full_drive_transition(
    field: Any,
    routing: FullDriveRoutingResult,
    current_target: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    step_index: int,
) -> FullDriveTransitionProposal:
    """Propose the atomic target/weight Euler transition for one field."""

    if (
        step_index not in range(8)
        or current_target.shape != target_velocity.shape
        or not torch.isfinite(current_target).all()
        or not torch.isfinite(target_velocity).all()
    ):
        raise ODEBFContractError("full-drive transition input differs")
    target_trial = (
        current_target.detach().to(device="cpu", dtype=torch.float32)
        + float(FIXED_E8_H)
        * target_velocity.detach().to(device="cpu", dtype=torch.float32)
    ).contiguous()
    if not torch.isfinite(target_trial).all():
        raise ODEBFContractError("full-drive target trial is non-finite")
    if routing.mode is FullDriveStepMode.TARGET_ONLY_PMAX_ZERO:
        if routing.velocity != (0.0,) * len(FIXED_E8_LAYER_ORDER):
            raise ODEBFContractError("full-drive target-only weight velocity differs")
        if torch.equal(target_trial, current_target.cpu()):
            raise ODEBFContractError("TARGET_NO_DESCENT")
        increment: Mapping[str, WaypointFactor] = {}
        weight_advanced = False
    else:
        increment = legacy.fixed_e8_waypoint_factors(
            field, routing.velocity, step_index=step_index
        )
        physical = tuple(
            float(increment[item.weight_name].theta) for item in field.layers
        )
        if physical != routing.applied_coefficient:
            raise ODEBFContractError("full-drive target/write coefficient differs")
        weight_advanced = True
    payload = {
        "schema": "ode-edit-s05-p1r17-transition-proposal/v1",
        "step_index": step_index,
        "mode": routing.mode.value,
        "target_trial_sha256": tensor_sha256(target_trial),
        "increment": {
            name: {
                "layer": int(item.layer),
                "theta": float(item.theta),
                "left_sha256": tensor_sha256(item.left),
                "right_sha256": tensor_sha256(item.right),
            }
            for name, item in increment.items()
        },
        "target_advanced": not torch.equal(target_trial, current_target.cpu()),
        "weight_advanced": weight_advanced,
        "target_clock_advance": 1,
        "weight_clock_advance": 1,
        "tau_advance": float(FIXED_E8_H),
        "scientific_retry_count": 0,
        "target_and_weight_h_equal": True,
        "pmax_zero_weight_unchanged": (
            routing.mode is not FullDriveStepMode.TARGET_ONLY_PMAX_ZERO
            or not weight_advanced
        ),
    }
    return FullDriveTransitionProposal(
        step_index,
        routing.mode,
        target_trial,
        increment,
        bool(payload["target_advanced"]),
        weight_advanced,
        canonical_hash(payload),
    )


def full_drive_runtime_contract() -> dict[str, Any]:
    payload = {
        "method_id": FULL_DRIVE_METHOD_ID,
        "cell_order": [item.value for item in FULL_DRIVE_CELL_ORDER],
        "factorial_shape": [2, 2, 1],
        "target_dynamics": "DYNAMIC_TARGET",
        "layer_order": list(FIXED_E8_LAYER_ORDER),
        "k": 8,
        "h": float(FIXED_E8_H),
        "target_only_pmax_zero_transition": True,
        "progress_floor": 0.0,
        "kappa": 0.0,
        "requested_progress": None,
        "retry_count": 0,
        "target_hold_access_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "FULL_DRIVE_CELL_ORDER",
    "FullDriveAllocation",
    "FullDriveCell",
    "FullDriveFieldBundle",
    "FullDriveTransitionProposal",
    "build_full_drive_field",
    "full_drive_runtime_contract",
    "propose_full_drive_transition",
]
