"""S05 preservation-all-off full-tau pilot panel.

All preservation quantities are still computed and serialized.  The ablation
removes only their solver/verifier decision influence; finite values, signed
progress/rho, non-negativity, layer caps, solver certificates, BF16 virtual /
commit identity, transactions, and rollback remain fail-closed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import BATCH_SIZE, MODEL_ALIASES, ODEBFContractError, canonical_hash
from .p1_adaptive import AdaptiveVariant, adaptive_lock
from .p1_adaptive_runtime import (
    AdaptivePanelSpec,
    FunctionalPDecisionPolicy,
    VariantRollout,
    _target_new_panel_contrast,
)
from .p1_controller import P1ControllerLock
from .p1_stepwise import StepwisePrimaryReceipt
from .resource import forecast_p1_adaptive_b10_memory
from .routing import PreservationConstraintPolicy
from .target_new_nll import RoutingObjective


PRESERVATION_ALL_OFF_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-PRESERVATION-ALLOFF-FULLTAU-P1R4-V1"
)
PRESERVATION_ALL_OFF_SCHEMA_NAMESPACE = (
    "ode-edit-s05-ode-bf-preservation-alloff-fulltau-p1r4"
)
PRESERVATION_ALL_OFF_RESULT_TOKEN = "preservation-alloff-fulltau-p1r4-v1"
PRESERVATION_ALL_OFF_TERMINAL_STATUS = (
    "PRESERVATION_ALLOFF_FULLTAU_PILOT_COMPLETE_NO_PROMOTION"
)
PRESERVATION_ALL_OFF_PANEL_LABELS = (
    "FR-A8-MARGIN-ALLOFF",
    "FR-A8-NEWNLL-ALLOFF",
)
PARENT_SOURCE_HEAD = "b29f5ac293b18c8b2513eac214ba539d6ea74517"
PARENT_NUMERICAL_LOCK_SHA256 = (
    "bacf045cbe249c5abd99698749d9824fec243b58272e0f61893ec90997ac95c7"
)


def preservation_all_off_panel_specs() -> tuple[AdaptivePanelSpec, ...]:
    return (
        AdaptivePanelSpec(
            PRESERVATION_ALL_OFF_PANEL_LABELS[0],
            AdaptiveVariant.FR_A8,
            RoutingObjective.MARGIN,
            FunctionalPDecisionPolicy.OBSERVATION_ONLY,
            PreservationConstraintPolicy.OBSERVATION_ONLY,
        ),
        AdaptivePanelSpec(
            PRESERVATION_ALL_OFF_PANEL_LABELS[1],
            AdaptiveVariant.FR_A8,
            RoutingObjective.TARGET_NEW_NLL,
            FunctionalPDecisionPolicy.OBSERVATION_ONLY,
            PreservationConstraintPolicy.OBSERVATION_ONLY,
        ),
    )


def expected_preservation_all_off_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("preservation-all-off result alias differs")
    return f"s05-preservation-alloff-fulltau-p1r4-{alias}-v1"


@dataclass(frozen=True, slots=True)
class PreservationAllOffForecast:
    alias: str
    variant_count: int
    maximum_trial_count_per_variant: int
    maximum_field_count_per_variant: int
    maximum_unique_accepted_snapshots: int
    conservative_parent_memory_identity: str
    pre_reserve_seconds: int
    reserve_fraction: float
    conservative_gpu_peak_mib: int
    conservative_host_peak_mib: int
    conservative_time_seconds: int
    allocation_memory_mib: int
    allocation_time_seconds: int
    fits_same_envelope: bool

    def __post_init__(self) -> None:
        if (
            self.alias not in MODEL_ALIASES
            or self.variant_count != 2
            or self.maximum_trial_count_per_variant != 128
            or self.maximum_field_count_per_variant != 32
            or self.maximum_unique_accepted_snapshots != 66
            or self.reserve_fraction != 0.10
            or self.conservative_time_seconds
            != math.ceil(self.pre_reserve_seconds * 1.10)
            or self.conservative_gpu_peak_mib > self.allocation_memory_mib
            or self.conservative_host_peak_mib > self.allocation_memory_mib
            or self.conservative_time_seconds > self.allocation_time_seconds
            or not self.fits_same_envelope
        ):
            raise ODEBFContractError("preservation-all-off forecast differs")

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> str:
        return canonical_hash(self.raw_free_payload())


def forecast_preservation_all_off_panel(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> PreservationAllOffForecast:
    parent = forecast_p1_adaptive_b10_memory(
        artifact_lock_path, base_model_lock_path, alias
    )
    pre_reserve = 128 * 60 + 128 * 6 * 60
    pre_reserve += 32 * 60 + 32 * 6 * 60
    pre_reserve += 66 * 10 + 68 * 60 + 1_800
    return PreservationAllOffForecast(
        alias=alias,
        variant_count=2,
        maximum_trial_count_per_variant=128,
        maximum_field_count_per_variant=32,
        maximum_unique_accepted_snapshots=66,
        conservative_parent_memory_identity=parent.identity(),
        pre_reserve_seconds=pre_reserve,
        reserve_fraction=0.10,
        conservative_gpu_peak_mib=parent.forecast_gpu_peak_mib,
        conservative_host_peak_mib=parent.forecast_host_peak_mib,
        conservative_time_seconds=math.ceil(pre_reserve * 1.10),
        allocation_memory_mib=65_000,
        allocation_time_seconds=86_400,
        fits_same_envelope=True,
    )


def preservation_all_off_refinement(
    capture: Any,
    rollouts: Mapping[Any, VariantRollout],
    receipts: Mapping[tuple[str, int], StepwisePrimaryReceipt],
) -> dict[str, Any]:
    if tuple(rollouts) != PRESERVATION_ALL_OFF_PANEL_LABELS:
        raise ODEBFContractError("preservation-all-off panel order differs")
    return {
        "schema": "ode-edit-s05-preservation-alloff-contrasts/v1",
        "preservation_observation_retained": True,
        "preservation_decision_influence_count": 0,
        "technical_progress_and_bf16_gates_active": True,
        "newnll_minus_margin": _target_new_panel_contrast(
            capture,
            rollouts,
            receipts,
            left_label=PRESERVATION_ALL_OFF_PANEL_LABELS[0],
            right_label=PRESERVATION_ALL_OFF_PANEL_LABELS[1],
        ),
        "accepted_tau_by_label": {
            label: {
                "numerator": rollout.accepted_t.numerator,
                "denominator": rollout.accepted_t.denominator,
            }
            for label, rollout in rollouts.items()
        },
        "scientific_promotion_authorized": False,
    }


def preservation_all_off_terminal_metadata() -> dict[str, Any]:
    return {
        "preservation_constraints_observation_only": [
            "structural_h", "structural_p", "trust", "functional_h",
            "functional_p",
        ],
        "preservation_decision_influence_count": 0,
        "positive_progress_gate_active": True,
        "solver_certificate_gate_active": True,
        "layer_caps_active": True,
        "authoritative_bf16_gate_active": True,
        "transaction_and_rollback_active": True,
        "first_hit_observation_only": True,
        "target_z_velocity_objective": "MARGIN_LOCKED",
        "routing_context_group_sizes": [1, 5],
        "routing_context_count": 6,
        "old_nll_decision_influence_count_newnll": 0,
    }


def validate_preservation_all_off_lock(
    value: Mapping[str, Any],
    *,
    controller_identity_sha256: str,
    stream_root_digest: str,
    population_root_digest: str,
) -> None:
    specs = preservation_all_off_panel_specs()
    expected = {
        "schema_version": (
            "ode-edit-s05-preservation-alloff-fulltau-numerical-lock/v1"
        ),
        "instruction_id": PRESERVATION_ALL_OFF_INSTRUCTION_ID,
        "parent_source_head": PARENT_SOURCE_HEAD,
        "parent_numerical_lock_sha256": PARENT_NUMERICAL_LOCK_SHA256,
        "status": "SEALED_REUSED_SAMPLE_CAUSAL_PILOT",
        "benchmark": "counterfact",
        "common_seed": 41,
        "edit_batch_size": BATCH_SIZE,
        "sequential_batch_count": 1,
        "scientific_sample_reused_for_causal_diagnostic": True,
        "scientific_promotion_authorized": False,
        "stream_root_digest": stream_root_digest,
        "p_population_root_digest": population_root_digest,
        "panel_labels": list(PRESERVATION_ALL_OFF_PANEL_LABELS),
        "routing_objective_by_label": {
            item.label: item.routing_objective.value for item in specs
        },
        "clock_variant_by_label": {
            item.label: item.clock_variant.value for item in specs
        },
        "functional_p_decision_by_label": {
            item.label: item.functional_p_decision.value for item in specs
        },
        "preservation_constraints_by_label": {
            item.label: item.preservation_constraints.value for item in specs
        },
        "variant_lock_sha256": {
            item.label: adaptive_lock(item.clock_variant).identity()
            for item in specs
        },
        "controller_identity_sha256": controller_identity_sha256,
        "all_off_ablation": {
            "observation_computed_every_trial": True,
            "decision_influence_count": {
                "structural_h": 0,
                "structural_p": 0,
                "trust": 0,
                "functional_h": 0,
                "functional_p": 0,
            },
            "huge_finite_budget_emulation": False,
            "positive_progress_active": True,
            "solver_certificate_active": True,
            "layer_caps_active": True,
            "authoritative_bf16_active": True,
        },
        "unchanged_contract": {
            "target_state_objective": "MARGIN_LOCKED",
            "adaptive_t_max": 1.0,
            "h_ref": 0.125,
            "delta_tau_min": 0.0078125,
            "rho_accept": 0.1,
            "same_state_additional_retry_cap": 4,
            "k_acc_cap": 32,
            "n_trial_cap": 128,
            "first_hit_observation_only": True,
            "persistent_endpoint_commit_count": 0,
            "history_append_count": 0,
        },
        "resource": {
            "server1_gpu_cap": 3,
            "gpu_per_job": 1,
            "cpu_per_job": 8,
            "memory_mib_per_job": 65000,
            "time_limit_seconds": 86400,
            "maximum_unique_accepted_snapshots": 66,
            "conservative_time_seconds": 81114,
        },
        "model_alias_specific_controller_branches": 0,
        "retry_submission_count": 0,
    }
    observed = dict(value)
    observed.pop("root_digest", None)
    if observed != expected:
        raise ODEBFContractError("preservation-all-off numerical lock differs")


def controller_identity_for_lock() -> str:
    return P1ControllerLock().identity()
