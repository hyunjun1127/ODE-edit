"""Locked S05 target-new-NLL routing panel and outcome-blind forecast."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import BATCH_SIZE, MODEL_ALIASES, ODEBFContractError, canonical_hash
from .p1_adaptive import AdaptiveVariant, adaptive_lock
from .p1_adaptive_runtime import AdaptivePanelSpec
from .resource import (
    forecast_p1_adaptive_b10_memory,
)
from .target_new_nll import RoutingObjective


TARGET_NEW_INSTRUCTION_ID = "ODEEDIT-S05-ODE-BF-TARGET-NEW-NLL-ROUTING-P1-V1"
TARGET_NEW_SCHEMA_NAMESPACE = "ode-edit-s05-ode-bf-target-new-nll-routing"
TARGET_NEW_RESULT_TOKEN = "target-new-nll-routing-causal-p1-r1-v1"
TARGET_NEW_TERMINAL_STATUS = "TARGET_NEW_NLL_CAUSAL_DIAGNOSTIC_COMPLETE_NO_PROMOTION"
SOURCE_HANDOFF_COMMIT = "18d13fbea2d5f58ef665f50fcc9fa255d01097e5"
SOURCE_BUNDLE_SHA256 = "1393bdca2a30605152ae545291d5620a48e56e3b1b16cea2a04aaab5702e45c1"
SOURCE_PATH_MANIFEST_SHA256 = (
    "8aaf32cb5e697e28913f4053e23bced58d7ef0ea5bcd7c4b678a515b900a9dd3"
)
SOURCE_RECEIPT_SHA256 = "223094727bb29e8faac0d3118b0e452f7c039e21a472aafc7660151730ba91a7"
TARGET_NEW_PANEL_LABELS = (
    "FR-A8-MARGIN",
    "FR-A8-NEWNLL",
)
TARGET_NEW_OPTIONAL_A16_LABEL = "FR-A16-NEWNLL"
TARGET_NEW_CONTEXT_COUNT = 6


def target_new_panel_specs() -> tuple[AdaptivePanelSpec, ...]:
    return (
        AdaptivePanelSpec(
            TARGET_NEW_PANEL_LABELS[0],
            AdaptiveVariant.FR_A8,
            RoutingObjective.MARGIN,
        ),
        AdaptivePanelSpec(
            TARGET_NEW_PANEL_LABELS[1],
            AdaptiveVariant.FR_A8,
            RoutingObjective.TARGET_NEW_NLL,
        ),
    )


def expected_target_new_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("target-new routing result alias differs")
    return f"s05-target-new-nll-routing-r1-{alias}-v1"


@dataclass(frozen=True, slots=True)
class TargetNewPanelForecast:
    alias: str
    variant_count: int
    maximum_unique_accepted_snapshots: int
    maximum_factors_per_endpoint_layer: int
    maximum_retained_factors_per_layer: int
    conservative_parent_memory_identity: str
    routing_context_count: int
    maximum_margin_trial_count: int
    maximum_target_new_trial_count: int
    maximum_margin_field_build_count: int
    maximum_target_new_field_build_count: int
    maximum_terminal_replay_count: int
    maximum_postfreeze_state_count: int
    seconds_per_trial_or_field_bound: int
    seconds_per_terminal_replay_bound: int
    seconds_per_postfreeze_state_bound: int
    fixed_initialization_and_cleanup_seconds: int
    pre_reserve_seconds: int
    reserve_fraction: float
    conservative_gpu_peak_mib: int
    conservative_host_peak_mib: int
    conservative_time_seconds: int
    allocation_memory_mib: int
    allocation_time_seconds: int
    fr_a16_included: bool
    fr_a16_forecast_seconds: int
    fr_a16_exclusion_reason: str
    fits_same_envelope: bool

    def __post_init__(self) -> None:
        if self.alias not in MODEL_ALIASES:
            raise ODEBFContractError("target-new forecast alias differs")
        if (
            self.variant_count != 2
            or self.maximum_unique_accepted_snapshots != 64
            or self.maximum_factors_per_endpoint_layer != 32
            or self.maximum_retained_factors_per_layer != 64
            or self.routing_context_count != TARGET_NEW_CONTEXT_COUNT
            or self.maximum_margin_trial_count != 128
            or self.maximum_target_new_trial_count != 128
            or self.maximum_margin_field_build_count != 32
            or self.maximum_target_new_field_build_count != 32
            or self.maximum_terminal_replay_count != 64
            or self.maximum_postfreeze_state_count != 66
            or self.seconds_per_trial_or_field_bound != 60
            or self.seconds_per_terminal_replay_bound != 10
            or self.seconds_per_postfreeze_state_bound != 60
            or self.fixed_initialization_and_cleanup_seconds != 1800
            or self.reserve_fraction != 0.10
            or self.fr_a16_included
            or self.fr_a16_forecast_seconds <= self.allocation_time_seconds
            or self.fr_a16_exclusion_reason
            != "SIX_CONTEXT_A16_EXCEEDS_LOCKED_24H_ENVELOPE"
            or not self.fits_same_envelope
        ):
            raise ODEBFContractError("target-new forecast panel geometry differs")
        expected_pre_reserve = (
            self.maximum_margin_trial_count
            * self.seconds_per_trial_or_field_bound
            + self.maximum_target_new_trial_count
            * self.routing_context_count
            * self.seconds_per_trial_or_field_bound
            + self.maximum_margin_field_build_count
            * self.seconds_per_trial_or_field_bound
            + self.maximum_target_new_field_build_count
            * self.routing_context_count
            * self.seconds_per_trial_or_field_bound
            + self.maximum_terminal_replay_count
            * self.seconds_per_terminal_replay_bound
            + self.maximum_postfreeze_state_count
            * self.seconds_per_postfreeze_state_bound
            + self.fixed_initialization_and_cleanup_seconds
        )
        if (
            self.pre_reserve_seconds != expected_pre_reserve
            or self.conservative_time_seconds
            != math.ceil(expected_pre_reserve * (1.0 + self.reserve_fraction))
        ):
            raise ODEBFContractError("target-new forecast time arithmetic differs")
        if (
            self.conservative_gpu_peak_mib > self.allocation_memory_mib
            or self.conservative_host_peak_mib > self.allocation_memory_mib
            or self.conservative_time_seconds > self.allocation_time_seconds
        ):
            raise ODEBFContractError("target-new panel exceeds its envelope")

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> str:
        return canonical_hash(self.raw_free_payload())


def forecast_target_new_panel(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> TargetNewPanelForecast:
    """Bound the two executed A8 arms with explicit six-context accounting.

    The inherited 60-second field/trial bound covers one ten-forward B10
    objective evaluation.  TARGET_NEW_NLL evaluates six locked contexts and is
    therefore charged six such bounds.  Under that outcome-blind arithmetic an
    additional A16 shadow cannot fit the locked 24-hour envelope and is omitted
    for both aliases before any model action.
    """

    parent_memory = forecast_p1_adaptive_b10_memory(
        artifact_lock_path, base_model_lock_path, alias
    )
    maximum_snapshots = 32 + 32
    seconds_per_trial_or_field = 60
    seconds_per_terminal = 10
    seconds_per_postfreeze = 60
    fixed_seconds = 1800
    pre_reserve = (
        128 * seconds_per_trial_or_field
        + 128 * TARGET_NEW_CONTEXT_COUNT * seconds_per_trial_or_field
        + 32 * seconds_per_trial_or_field
        + 32 * TARGET_NEW_CONTEXT_COUNT * seconds_per_trial_or_field
        + maximum_snapshots * seconds_per_terminal
        + (2 + maximum_snapshots) * seconds_per_postfreeze
        + fixed_seconds
    )
    forecast_seconds = math.ceil(pre_reserve * 1.10)
    a16_pre_reserve = (
        pre_reserve
        + 128 * TARGET_NEW_CONTEXT_COUNT * seconds_per_trial_or_field
        + 64 * TARGET_NEW_CONTEXT_COUNT * seconds_per_trial_or_field
        + 64 * seconds_per_terminal
        + 64 * seconds_per_postfreeze
    )
    a16_forecast_seconds = math.ceil(a16_pre_reserve * 1.10)
    return TargetNewPanelForecast(
        alias=alias,
        variant_count=2,
        maximum_unique_accepted_snapshots=maximum_snapshots,
        maximum_factors_per_endpoint_layer=32,
        maximum_retained_factors_per_layer=maximum_snapshots,
        conservative_parent_memory_identity=parent_memory.identity(),
        routing_context_count=TARGET_NEW_CONTEXT_COUNT,
        maximum_margin_trial_count=128,
        maximum_target_new_trial_count=128,
        maximum_margin_field_build_count=32,
        maximum_target_new_field_build_count=32,
        maximum_terminal_replay_count=maximum_snapshots,
        maximum_postfreeze_state_count=2 + maximum_snapshots,
        seconds_per_trial_or_field_bound=seconds_per_trial_or_field,
        seconds_per_terminal_replay_bound=seconds_per_terminal,
        seconds_per_postfreeze_state_bound=seconds_per_postfreeze,
        fixed_initialization_and_cleanup_seconds=fixed_seconds,
        pre_reserve_seconds=pre_reserve,
        reserve_fraction=0.10,
        conservative_gpu_peak_mib=parent_memory.forecast_gpu_peak_mib,
        conservative_host_peak_mib=parent_memory.forecast_host_peak_mib,
        conservative_time_seconds=forecast_seconds,
        allocation_memory_mib=65_000,
        allocation_time_seconds=24 * 60 * 60,
        fr_a16_included=False,
        fr_a16_forecast_seconds=a16_forecast_seconds,
        fr_a16_exclusion_reason="SIX_CONTEXT_A16_EXCEEDS_LOCKED_24H_ENVELOPE",
        fits_same_envelope=True,
    )


def validate_target_new_lock(
    value: Mapping[str, Any],
    *,
    controller_identity_sha256: str,
    stream_root_digest: str,
    population_root_digest: str,
) -> None:
    specs = target_new_panel_specs()
    expected_variant_locks = {
        item.label: adaptive_lock(item.clock_variant).identity() for item in specs
    }
    expected_routing_loss = {
        "target_new_definition": (
            "uniform-request-and-authorized-context-mean-of-per-suffix-token-mean-"
            "teacher-forced-nll"
        ),
        "context_group_sizes": [1, 5],
        "context_count": 6,
        "context_weighting": "uniform-over-all-six-locked-rendered-contexts",
        "signed_layer_efficiency": (
            "negative-directional-derivative-of-target-new-nll"
        ),
        "full_signed_vector_serialized": True,
        "nonpositive_directions_excluded": True,
        "predicted_delta": "delta_tau_times_a_transpose_v",
        "actual_delta": "phi_new_state_minus_phi_new_trial",
        "nll_old_routing_influence": 0,
        "margin_live_control_unchanged": True,
        "gradient_reduction": {
            "streaming_unit": "one-request-six-context-mean",
            "autograd_calls_per_field": 10,
            "maximum_live_context_graphs": 6,
            "fp32_gradient_accumulation": True,
            "mathematical_objective_unchanged": True,
        },
    }
    expected_adaptive = {
        "t_max": 1.0,
        "h_ref": 0.125,
        "delta_tau_max_a8": 0.125,
        "delta_tau_max_a16": 0.0625,
        "delta_tau_min": 0.0078125,
        "gamma_down": 0.5,
        "gamma_up": 1.5,
        "rho_accept": 0.1,
        "rho_expand": 0.75,
        "same_state_additional_retry_cap": 4,
        "k_acc_cap_a8": 32,
        "k_acc_cap_a16": 64,
        "n_trial_cap": 128,
        "reject_advances_tau": False,
        "reject_refreshes_field": False,
        "target_and_weight_share_delta_tau": True,
    }
    expected_parent = {
        "path": "project/run_scripts/ode_bf/locks/numerical_lock_p1r2.json",
        "sha256": "0cdb4ff528f0eddea7b40b9d36a103fa372dca8433da8a0a2cf36f77ad2fa903",
        "functional_p_budget_nats": 0.001,
        "minimum_progress": 1.0e-8,
        "w64_residual_tolerance": 1.0e-5,
    }
    expected_postfreeze = {
        "states": "W0-N32-and-every-unique-accepted-snapshot",
        "teacher_forced_only": True,
        "generation_call_count": 0,
        "metrics": ["efficacy", "generalization", "locality-preservation"],
        "numeric_nll_new_old": True,
        "paired_native_bits": True,
        "old_degradation_only_success": True,
        "controller_heldout_access_count": 0,
    }
    expected_resource = {
        "server1_gpu_cap": 3,
        "gpu_per_job": 1,
        "cpu_per_job": 8,
        "memory_mib_per_job": 65000,
        "time_limit_seconds": 86400,
        "maximum_unique_accepted_snapshots": 64,
        "maximum_factors_per_endpoint_layer": 32,
        "maximum_retained_factors_per_layer": 64,
        "routing_context_count": 6,
        "routing_context_time_multiplier": 6,
        "conservative_time_seconds": 80960,
        "fr_a16_included": False,
        "fr_a16_forecast_seconds": 161920,
        "fr_a16_exclusion_reason": "SIX_CONTEXT_A16_EXCEEDS_LOCKED_24H_ENVELOPE",
        "llama_forecast_identity": (
            "25e12b466e6d26f3e2688c3de43de1400ad3019c07b25907b78590e78b6976c6"
        ),
        "qwen_forecast_identity": (
            "e33d412c103571fe057a58049ed1b204afca0f580412921719bda49a456cc62c"
        ),
        "dense_fp64_full_delta_live": 0,
        "dense_fp32_full_delta_live": 0,
        "effective_bf16_target_weight_peak_live": 1,
    }
    if (
        value.get("instruction_id") != TARGET_NEW_INSTRUCTION_ID
        or value.get("source_handoff_commit") != SOURCE_HANDOFF_COMMIT
        or value.get("source_bundle_sha256") != SOURCE_BUNDLE_SHA256
        or value.get("source_manifest_sha256") != SOURCE_PATH_MANIFEST_SHA256
        or value.get("source_receipt_sha256") != SOURCE_RECEIPT_SHA256
        or value.get("status") != "SEALED_REUSED_SAMPLE_CAUSAL_DIAGNOSTIC"
        or value.get("benchmark") != "counterfact"
        or value.get("common_seed") != 41
        or value.get("edit_batch_size") != BATCH_SIZE
        or value.get("sequential_batch_count") != 1
        or value.get("scientific_sample_reused_for_causal_diagnostic") is not True
        or value.get("panel_labels") != list(TARGET_NEW_PANEL_LABELS)
        or value.get("routing_objective_by_label")
        != {item.label: item.routing_objective.value for item in specs}
        or value.get("clock_variant_by_label")
        != {item.label: item.clock_variant.value for item in specs}
        or value.get("variant_lock_sha256") != expected_variant_locks
        or value.get("controller_identity_sha256") != controller_identity_sha256
        or value.get("stream_root_digest") != stream_root_digest
        or value.get("p_population_root_digest") != population_root_digest
        or value.get("target_state_objective") != "MARGIN_LOCKED"
        or value.get("old_nll_decision_influence_count_newnll") != 0
        or value.get("routing_loss") != expected_routing_loss
        or value.get("adaptive_pseudo_time") != expected_adaptive
        or value.get("frozen_parent_lock") != expected_parent
        or value.get("postfreeze_evaluation") != expected_postfreeze
        or value.get("resource") != expected_resource
        or value.get("scientific_promotion_authorized") is not False
        or value.get("fr_a16_same_envelope_forecast_pass") is not False
        or value.get("model_alias_specific_controller_branches") != 0
        or value.get("persistent_endpoint_commit_count") != 0
        or value.get("history_append_count") != 0
        or value.get("retry_submission_count") != 0
    ):
        raise ODEBFContractError("target-new routing numerical lock differs")
