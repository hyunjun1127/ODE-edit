"""Universal, detached observability contracts for fixed-grid ODE-BF arms.

This module contains no model calls and no controller decisions.  It defines
the exact factorial arm registry and validates that every new live arm opts in
to the same D1--D3 observation schema before runtime dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .contracts import ODEBFContractError, canonical_hash


ODE_BF_OBSERVABILITY_CONTRACT_V1 = "ODE_BF_OBSERVABILITY_CONTRACT_V1"
UNIVERSAL_OBSERVABILITY_SCHEMA = (
    "ode-edit-s05-ode-bf-universal-observability-contract/v1"
)


class TargetAllocation(str, Enum):
    ROBUST_SHARED = "RS"
    BATCH_GLOBAL = "BG"


class LayerRouting(str, Enum):
    NEUTRAL = "NEUTRAL"
    SOFT = "SOFT"


class TargetDynamics(str, Enum):
    DYNAMIC_TARGET = "DYNAMIC_TARGET"
    TARGET_HOLD = "TARGET_HOLD"


@dataclass(frozen=True)
class ODEBFObservabilityArm:
    """One first-class trajectory in the locked 2x2x2 factorial."""

    arm_id: str
    target_allocation: TargetAllocation
    layer_routing: LayerRouting
    target_dynamics: TargetDynamics

    @property
    def target_hold(self) -> bool:
        return self.target_dynamics is TargetDynamics.TARGET_HOLD

    @property
    def dynamic_peer_id(self) -> str:
        return (
            self.arm_id.removesuffix("-TARGET-HOLD")
            if self.target_hold
            else self.arm_id
        )

    @property
    def hold_peer_id(self) -> str:
        return (
            self.arm_id
            if self.target_hold
            else f"{self.arm_id}-TARGET-HOLD"
        )

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "schema": UNIVERSAL_OBSERVABILITY_SCHEMA,
            "contract_identity": ODE_BF_OBSERVABILITY_CONTRACT_V1,
            "arm_id": self.arm_id,
            "target_allocation": self.target_allocation.value,
            "layer_routing": self.layer_routing.value,
            "target_dynamics": self.target_dynamics.value,
            "target_hold": self.target_hold,
            "d1": "EVERY_FIELD_OVERLAY_VS_NOHOOK_BEFORE_GUARD",
            "d2": "POSTFREEZE_W_ONLY_VS_ADDITIVE_Z_ORACLE",
            "d3": "ENTRY_AND_BOUNDARY_SELECTED_V_SINGLE_ALL_WRITE_AUDIT",
            "field_decision_influence_count": 0,
            "solver_decision_influence_count": 0,
            "selected_velocity_decision_influence_count": 0,
            "candidate_decision_influence_count": 0,
            "clock_decision_influence_count": 0,
            "first_hit_decision_influence_count": 0,
            "endpoint_decision_influence_count": 0,
            "sampler_history_rng_mutation_count": 0,
            "controller_heldout_access_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


def _arm(
    allocation: TargetAllocation,
    routing: LayerRouting,
    dynamics: TargetDynamics,
) -> ODEBFObservabilityArm:
    base = f"{allocation.value}-{routing.value}"
    arm_id = (
        base
        if dynamics is TargetDynamics.DYNAMIC_TARGET
        else f"{base}-TARGET-HOLD"
    )
    return ODEBFObservabilityArm(arm_id, allocation, routing, dynamics)


ODE_BF_FACTORIAL_ARMS = tuple(
    _arm(allocation, routing, dynamics)
    for allocation in TargetAllocation
    for routing in LayerRouting
    for dynamics in TargetDynamics
)
ODE_BF_ARM_REGISTRY = {arm.arm_id: arm for arm in ODE_BF_FACTORIAL_ARMS}

if len(ODE_BF_ARM_REGISTRY) != 8:
    raise RuntimeError("ODE-BF universal arm registry is not 2x2x2")


R13_LIVE_CELL_IDS = ("RS-NEUTRAL", "RS-SOFT", "BG-NEUTRAL")
R12_REUSABLE_CELL_ID = "BG-SOFT"


def require_observability_arm(arm_id: str) -> ODEBFObservabilityArm:
    """Fail closed when a new dispatch omits or misspells the contract arm."""

    if type(arm_id) is not str or arm_id not in ODE_BF_ARM_REGISTRY:
        raise ODEBFContractError("ODE-BF observability arm registry differs")
    return ODE_BF_ARM_REGISTRY[arm_id]


def paired_arm_ids(cell_id: str) -> tuple[str, str]:
    if cell_id not in R13_LIVE_CELL_IDS + (R12_REUSABLE_CELL_ID,):
        raise ODEBFContractError("ODE-BF observability factorial cell differs")
    dynamic = require_observability_arm(cell_id)
    if dynamic.target_hold:
        raise ODEBFContractError("ODE-BF observability cell is not dynamic")
    hold = require_observability_arm(dynamic.hold_peer_id)
    if (
        hold.target_allocation is not dynamic.target_allocation
        or hold.layer_routing is not dynamic.layer_routing
        or not hold.target_hold
    ):
        raise ODEBFContractError("ODE-BF observability target-dynamics pair differs")
    return dynamic.arm_id, hold.arm_id


def universal_observability_contract_receipt(
    *,
    instruction_id: str,
    amendment_id: str,
    cell_id: str,
) -> dict[str, Any]:
    dynamic_id, hold_id = paired_arm_ids(cell_id)
    payload = {
        "schema": UNIVERSAL_OBSERVABILITY_SCHEMA,
        "contract_identity": ODE_BF_OBSERVABILITY_CONTRACT_V1,
        "instruction_id": instruction_id,
        "amendment_id": amendment_id,
        "factorial_shape": [2, 2, 2],
        "factorial_axes": [
            "TARGET_ALLOCATION_RS_VS_BG",
            "LAYER_ROUTING_NEUTRAL_VS_SOFT",
            "TARGET_DYNAMICS_DYNAMIC_VS_HOLD",
        ],
        "registered_arm_ids": sorted(ODE_BF_ARM_REGISTRY),
        "cell_id": cell_id,
        "live_arms": [dynamic_id, hold_id],
        "d1_d2_d3_observation_only": True,
        "observability_total_decision_influence_count": 0,
        "observability_state_mutation_count": 0,
        "observability_rng_advance_count": 0,
        "heldout_open_before_action_freeze_count": 0,
        "scientific_promotion_authorized": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def validate_observability_contract_receipt(
    value: Mapping[str, Any],
    *,
    instruction_id: str,
    amendment_id: str,
    cell_id: str,
) -> dict[str, Any]:
    expected = universal_observability_contract_receipt(
        instruction_id=instruction_id,
        amendment_id=amendment_id,
        cell_id=cell_id,
    )
    if dict(value) != expected:
        raise ODEBFContractError("ODE-BF observability contract receipt differs")
    return dict(expected)


__all__ = [
    "LayerRouting",
    "ODE_BF_ARM_REGISTRY",
    "ODE_BF_FACTORIAL_ARMS",
    "ODE_BF_OBSERVABILITY_CONTRACT_V1",
    "ODEBFObservabilityArm",
    "R12_REUSABLE_CELL_ID",
    "R13_LIVE_CELL_IDS",
    "TargetAllocation",
    "TargetDynamics",
    "UNIVERSAL_OBSERVABILITY_SCHEMA",
    "paired_arm_ids",
    "require_observability_arm",
    "universal_observability_contract_receipt",
    "validate_observability_contract_receipt",
]
