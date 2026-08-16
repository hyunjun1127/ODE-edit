"""Locked S05 functional-P decision ablation and outcome-blind forecast."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
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
from .p1_stepwise import StepwisePrimaryReceipt, compare_stepwise_primary
from .resource import forecast_p1_adaptive_b10_memory
from .target_new_nll import RoutingObjective


FUNCTIONAL_P_OFF_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-FUNCTIONAL-P-OFF-FULLTAU-P1R3-V1"
)
FUNCTIONAL_P_OFF_SCHEMA_NAMESPACE = (
    "ode-edit-s05-ode-bf-functional-p-off-fulltau-p1r3"
)
FUNCTIONAL_P_OFF_RESULT_TOKEN = "functional-p-off-fulltau-p1r3-v1"
FUNCTIONAL_P_OFF_TERMINAL_STATUS = (
    "FUNCTIONAL_P_OFF_FULLTAU_DIAGNOSTIC_COMPLETE_NO_PROMOTION"
)
FUNCTIONAL_P_OFF_PANEL_LABELS = (
    "FR-A8-MARGIN-PCTRL",
    "FR-A8-NEWNLL-PCTRL",
    "FR-A8-MARGIN-FPOFF",
    "FR-A8-NEWNLL-FPOFF",
)
PARENT_SOURCE_HEAD = "46c788df3031ae40fdb7a72f6d1ce70eb4dc5370"
PARENT_TARGET_LOCK_SHA256 = (
    "792cbb148f58aaae8fca3bc6bf68afd80e41b50462dddb13be65ac81b4f4d6a9"
)
R2_TERMINAL_SHA256 = {
    "llama3-8b-inst": (
        "cef80834a0708d3fe0f6545eab3786ae197f2555633cfbafab08f254f22a38e9"
    ),
    "qwen2.5-7b-inst": (
        "ff0481a5d31ecc5f06bb3666e120b813354e8be7da9217aad25cdfd0581e9da5"
    ),
}
_R2_CONTROL = {
    ("llama3-8b-inst", "FR-A8-MARGIN-PCTRL"): {
        "status": "SAME_STATE_RETRY_EXHAUSTED",
        "tau": Fraction(1, 8),
        "snapshot": "75db974d71c37cf012c3ebf328edc19c0a424880a9731d01f6e500f2208e6a2a",
    },
    ("llama3-8b-inst", "FR-A8-NEWNLL-PCTRL"): {
        "status": "SAME_STATE_RETRY_EXHAUSTED",
        "tau": Fraction(1, 8),
        "snapshot": "d3d601af7dcb2f8835766f72f7a11e8112ee8a26b3205c99a2b0fae382ca1ce4",
    },
    ("qwen2.5-7b-inst", "FR-A8-MARGIN-PCTRL"): {
        "status": "MIN_DT_EXHAUSTED",
        "tau": Fraction(1, 32),
        "snapshot": "8e3ab67bb17b0d2a59d2d97daea818d978ac22c90d2a18a7004989d084642623",
    },
    ("qwen2.5-7b-inst", "FR-A8-NEWNLL-PCTRL"): {
        "status": "MIN_DT_EXHAUSTED",
        "tau": Fraction(1, 16),
        "snapshot": "35e3a18211c6f1eca6d2da38adfabb388e6c9f7618e98b7b3c7890569b276017",
    },
}


def functional_p_off_panel_specs() -> tuple[AdaptivePanelSpec, ...]:
    return (
        AdaptivePanelSpec(
            FUNCTIONAL_P_OFF_PANEL_LABELS[0], AdaptiveVariant.FR_A8,
            RoutingObjective.MARGIN, FunctionalPDecisionPolicy.PCTRL,
        ),
        AdaptivePanelSpec(
            FUNCTIONAL_P_OFF_PANEL_LABELS[1], AdaptiveVariant.FR_A8,
            RoutingObjective.TARGET_NEW_NLL, FunctionalPDecisionPolicy.PCTRL,
        ),
        AdaptivePanelSpec(
            FUNCTIONAL_P_OFF_PANEL_LABELS[2], AdaptiveVariant.FR_A8,
            RoutingObjective.MARGIN, FunctionalPDecisionPolicy.OBSERVATION_ONLY,
        ),
        AdaptivePanelSpec(
            FUNCTIONAL_P_OFF_PANEL_LABELS[3], AdaptiveVariant.FR_A8,
            RoutingObjective.TARGET_NEW_NLL,
            FunctionalPDecisionPolicy.OBSERVATION_ONLY,
        ),
    )


def expected_functional_p_off_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("functional-P result alias differs")
    return f"s05-functional-p-off-fulltau-p1r3-{alias}-v1"


@dataclass(frozen=True, slots=True)
class FunctionalPOffPanelForecast:
    alias: str
    variant_count: int
    maximum_unique_accepted_snapshots: int
    maximum_factors_per_endpoint_layer: int
    maximum_retained_factors_per_layer: int
    conservative_parent_memory_identity: str
    locked_pctrl_seconds: int
    maximum_margin_fpoff_trial_count: int
    maximum_newnll_fpoff_trial_count: int
    maximum_margin_fpoff_field_count: int
    maximum_newnll_fpoff_field_count: int
    routing_context_count: int
    maximum_terminal_replay_count: int
    maximum_postfreeze_state_count: int
    pre_reserve_seconds: int
    reserve_fraction: float
    conservative_gpu_peak_mib: int
    conservative_host_peak_mib: int
    conservative_time_seconds: int
    allocation_memory_mib: int
    allocation_time_seconds: int
    fits_same_envelope: bool

    def __post_init__(self) -> None:
        if self.alias not in MODEL_ALIASES:
            raise ODEBFContractError("functional-P forecast alias differs")
        if (
            self.variant_count != 4
            or self.maximum_unique_accepted_snapshots != 66
            or self.maximum_factors_per_endpoint_layer != 32
            or self.maximum_retained_factors_per_layer != 66
            or self.locked_pctrl_seconds != 1_600
            or self.maximum_margin_fpoff_trial_count != 128
            or self.maximum_newnll_fpoff_trial_count != 128
            or self.maximum_margin_fpoff_field_count != 32
            or self.maximum_newnll_fpoff_field_count != 32
            or self.routing_context_count != 6
            or self.maximum_terminal_replay_count != 66
            or self.maximum_postfreeze_state_count != 68
            or self.reserve_fraction != 0.10
            or not self.fits_same_envelope
        ):
            raise ODEBFContractError("functional-P forecast geometry differs")
        expected_pre_reserve = (
            self.locked_pctrl_seconds
            + self.maximum_margin_fpoff_trial_count * 60
            + self.maximum_newnll_fpoff_trial_count * self.routing_context_count * 60
            + self.maximum_margin_fpoff_field_count * 60
            + self.maximum_newnll_fpoff_field_count * self.routing_context_count * 60
            + self.maximum_terminal_replay_count * 10
            + self.maximum_postfreeze_state_count * 60
            + 1_800
        )
        if (
            self.pre_reserve_seconds != expected_pre_reserve
            or self.conservative_time_seconds
            != math.ceil(expected_pre_reserve * 1.10)
            or self.conservative_gpu_peak_mib > self.allocation_memory_mib
            or self.conservative_host_peak_mib > self.allocation_memory_mib
            or self.conservative_time_seconds > self.allocation_time_seconds
        ):
            raise ODEBFContractError("functional-P forecast envelope differs")

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> str:
        return canonical_hash(self.raw_free_payload())


def forecast_functional_p_off_panel(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> FunctionalPOffPanelForecast:
    """Bound controls by immutable R2 reproduction and FPOFF outcome-blind caps."""

    parent = forecast_p1_adaptive_b10_memory(
        artifact_lock_path, base_model_lock_path, alias
    )
    pre_reserve = 1_600 + 128 * 60 + 128 * 6 * 60 + 32 * 60 + 32 * 6 * 60
    pre_reserve += 66 * 10 + 68 * 60 + 1_800
    return FunctionalPOffPanelForecast(
        alias=alias,
        variant_count=4,
        maximum_unique_accepted_snapshots=66,
        maximum_factors_per_endpoint_layer=32,
        maximum_retained_factors_per_layer=66,
        conservative_parent_memory_identity=parent.identity(),
        locked_pctrl_seconds=1_600,
        maximum_margin_fpoff_trial_count=128,
        maximum_newnll_fpoff_trial_count=128,
        maximum_margin_fpoff_field_count=32,
        maximum_newnll_fpoff_field_count=32,
        routing_context_count=6,
        maximum_terminal_replay_count=66,
        maximum_postfreeze_state_count=68,
        pre_reserve_seconds=pre_reserve,
        reserve_fraction=0.10,
        conservative_gpu_peak_mib=parent.forecast_gpu_peak_mib,
        conservative_host_peak_mib=parent.forecast_host_peak_mib,
        conservative_time_seconds=math.ceil(pre_reserve * 1.10),
        allocation_memory_mib=65_000,
        allocation_time_seconds=86_400,
        fits_same_envelope=True,
    )


def validate_r2_control_rollout(
    alias: str,
    spec: AdaptivePanelSpec,
    rollout: VariantRollout,
) -> None:
    """Fail before FPOFF if either paired control no longer reproduces R2."""

    if spec.functional_p_decision is FunctionalPDecisionPolicy.OBSERVATION_ONLY:
        return
    expected = _R2_CONTROL.get((alias, spec.label))
    if expected is None:
        raise ODEBFContractError("functional-P control identity differs")
    snapshots = [item.snapshot_sha256 for item in rollout.snapshots]
    if (
        rollout.functional_p_decision != FunctionalPDecisionPolicy.PCTRL.value
        or rollout.status != expected["status"]
        or rollout.termination_label != expected["status"]
        or rollout.accepted_t != expected["tau"]
        or rollout.k_acc != 1
        or rollout.n_trial != 6
        or rollout.n_reject != 5
        or rollout.field_build_count != 2
        or snapshots != [expected["snapshot"]]
    ):
        raise ODEBFContractError("functional-P PCTRL/R2 reproduction differs")


def _matched_prefix(
    rollouts: Mapping[Any, VariantRollout],
    receipts: Mapping[tuple[str, int], StepwisePrimaryReceipt],
    pctrl_label: str,
    fpoff_label: str,
) -> dict[str, Any]:
    pctrl = rollouts[pctrl_label]
    fpoff = rollouts[fpoff_label]
    if not pctrl.snapshots or not fpoff.snapshots:
        raise ODEBFContractError("functional-P matched prefix is unavailable")
    left = pctrl.snapshots[0]
    right = fpoff.snapshots[0]
    if (
        left.snapshot_sha256 != right.snapshot_sha256
        or left.tau != right.tau
        or left.functional_p_decision["observation_sha256"]
        != right.functional_p_decision["observation_sha256"]
        or left.functional_p_decision["decision_influence_count"] != 1
        or right.functional_p_decision["decision_influence_count"] != 0
    ):
        raise ODEBFContractError("functional-P matched prefix differs")
    left_primary = receipts[(pctrl_label, left.accepted_index)]
    right_primary = receipts[(fpoff_label, right.accepted_index)]
    if left_primary.numeric_vectors_sha256 != right_primary.numeric_vectors_sha256:
        raise ODEBFContractError("functional-P matched-prefix evaluation differs")
    return {
        "snapshot_sha256": left.snapshot_sha256,
        "tau": {"numerator": left.tau.numerator, "denominator": left.tau.denominator},
        "functional_p_observation_sha256": left.functional_p_decision[
            "observation_sha256"
        ],
        "postfreeze_primary_exact": True,
        "paired_primary": compare_stepwise_primary(
            left_primary, left_primary, right_primary
        ),
    }


def functional_p_off_refinement(
    capture: Any,
    rollouts: Mapping[Any, VariantRollout],
    receipts: Mapping[tuple[str, int], StepwisePrimaryReceipt],
) -> dict[str, Any]:
    if tuple(rollouts) != FUNCTIONAL_P_OFF_PANEL_LABELS:
        raise ODEBFContractError("functional-P panel order differs")
    margin_prefix = _matched_prefix(
        rollouts, receipts, FUNCTIONAL_P_OFF_PANEL_LABELS[0],
        FUNCTIONAL_P_OFF_PANEL_LABELS[2],
    )
    newnll_prefix = _matched_prefix(
        rollouts, receipts, FUNCTIONAL_P_OFF_PANEL_LABELS[1],
        FUNCTIONAL_P_OFF_PANEL_LABELS[3],
    )
    return {
        "schema": "ode-edit-s05-functional-p-off-fulltau-contrasts/v1",
        "functional_p_observation_unchanged": True,
        "functional_p_decision_change_only": True,
        "structural_p_active": True,
        "margin_matched_prefix": margin_prefix,
        "newnll_matched_prefix": newnll_prefix,
        "margin_fpoff_minus_pctrl": _target_new_panel_contrast(
            capture, rollouts, receipts,
            left_label=FUNCTIONAL_P_OFF_PANEL_LABELS[0],
            right_label=FUNCTIONAL_P_OFF_PANEL_LABELS[2],
        ),
        "newnll_fpoff_minus_pctrl": _target_new_panel_contrast(
            capture, rollouts, receipts,
            left_label=FUNCTIONAL_P_OFF_PANEL_LABELS[1],
            right_label=FUNCTIONAL_P_OFF_PANEL_LABELS[3],
        ),
        "pctrl_newnll_minus_margin": _target_new_panel_contrast(
            capture, rollouts, receipts,
            left_label=FUNCTIONAL_P_OFF_PANEL_LABELS[0],
            right_label=FUNCTIONAL_P_OFF_PANEL_LABELS[1],
        ),
        "fpoff_newnll_minus_margin": _target_new_panel_contrast(
            capture, rollouts, receipts,
            left_label=FUNCTIONAL_P_OFF_PANEL_LABELS[2],
            right_label=FUNCTIONAL_P_OFF_PANEL_LABELS[3],
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


def functional_p_off_terminal_metadata() -> dict[str, Any]:
    return {
        "functional_p_observation_budget": 1.0e-3,
        "functional_p_decision_ablation_only": True,
        "structural_p_active": True,
        "first_hit_observation_only": True,
        "full_tau_required_absent_non_p_boundary": True,
        "target_z_velocity_objective": "MARGIN_LOCKED",
        "routing_context_group_sizes": [1, 5],
        "routing_context_count": 6,
        "routing_context_weighting": (
            "uniform-over-all-six-locked-rendered-contexts"
        ),
        "old_nll_decision_influence_count_newnll": 0,
        "stepwise_layer_routing_telemetry": {
            "candidate_layer_ids": [4, 5, 6, 7, 8],
            "observation_only": True,
            "controller_dependency_count": 0,
        },
    }


def validate_functional_p_off_lock(
    value: Mapping[str, Any],
    *,
    controller_identity_sha256: str,
    stream_root_digest: str,
    population_root_digest: str,
) -> None:
    specs = functional_p_off_panel_specs()
    expected = {
        "schema_version": "ode-edit-s05-functional-p-off-fulltau-numerical-lock/v1",
        "instruction_id": FUNCTIONAL_P_OFF_INSTRUCTION_ID,
        "parent_source_head": PARENT_SOURCE_HEAD,
        "parent_target_lock_sha256": PARENT_TARGET_LOCK_SHA256,
        "status": "SEALED_REUSED_SAMPLE_CAUSAL_DIAGNOSTIC",
        "benchmark": "counterfact",
        "common_seed": 41,
        "edit_batch_size": BATCH_SIZE,
        "sequential_batch_count": 1,
        "scientific_sample_reused_for_causal_diagnostic": True,
        "scientific_promotion_authorized": False,
        "stream_root_digest": stream_root_digest,
        "p_population_root_digest": population_root_digest,
        "panel_labels": list(FUNCTIONAL_P_OFF_PANEL_LABELS),
        "routing_objective_by_label": {
            item.label: item.routing_objective.value for item in specs
        },
        "clock_variant_by_label": {
            item.label: item.clock_variant.value for item in specs
        },
        "functional_p_decision_by_label": {
            item.label: item.functional_p_decision.value for item in specs
        },
        "variant_lock_sha256": {
            item.label: adaptive_lock(item.clock_variant).identity() for item in specs
        },
        "controller_identity_sha256": controller_identity_sha256,
        "functional_p_ablation": {
            "budget_nats": 0.001,
            "observation_computed_every_trial": True,
            "observation_numerically_identical_for_matched_candidates": True,
            "pctrl_decision_influence_count_per_trial": 1,
            "fpoff_decision_influence_count_per_trial": 0,
            "structural_p_active": True,
            "huge_finite_budget_emulation": False,
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
        "r2_control": {
            "terminal_sha256_by_alias": dict(R2_TERMINAL_SHA256),
            "exact_reproduction_required_before_fpoff": True,
            "accepted_snapshot_count": 1,
            "trial_count": 6,
            "reject_count": 5,
            "field_build_count": 2,
        },
        "resource": {
            "server1_gpu_cap": 3,
            "gpu_per_job": 1,
            "cpu_per_job": 8,
            "memory_mib_per_job": 65000,
            "time_limit_seconds": 86400,
            "maximum_unique_accepted_snapshots": 66,
            "maximum_retained_factors_per_layer": 66,
            "conservative_time_seconds": 82874,
        },
        "model_alias_specific_controller_branches": 0,
        "retry_submission_count": 0,
    }
    observed = dict(value)
    observed.pop("root_digest", None)
    if observed != expected:
        raise ODEBFContractError("functional-P numerical lock differs")
