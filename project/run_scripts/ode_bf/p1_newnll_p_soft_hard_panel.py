"""Locked S05 TARGET_NEW_NLL functional H/P soft-hard causal pilot."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from .contracts import BATCH_SIZE, MODEL_ALIASES, ODEBFContractError, canonical_hash
from .functional_p_secant import FunctionalPFieldPolicy
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


NEWNLL_P_SOFT_HARD_INSTRUCTION_ID = (
    "ODEEDIT-S05-ODE-BF-NEWNLL-P-SOFT-HARD-P1R5-V1"
)
NEWNLL_P_SOFT_HARD_SCHEMA_NAMESPACE = (
    "ode-edit-s05-ode-bf-newnll-p-soft-hard-p1r5"
)
NEWNLL_P_SOFT_HARD_RESULT_TOKEN = "newnll-p-soft-hard-p1r5-v1"
NEWNLL_P_SOFT_HARD_TERMINAL_STATUS = (
    "DIAGNOSTIC_COMPLETE_NO_PROMOTION"
)
NEWNLL_P_SOFT_HARD_PANEL_LABELS = (
    "FR-A8-NEWNLL-PCTRL-PROBE",
    "FR-A8-NEWNLL-PSOFT-HARD",
    "FR-A8-NEWNLL-FPOFF-PROBE",
)
PARENT_SOURCE_HEAD = "c3d45eba301d3e449a03a21f7ce65b5aa70d07c2"
PARENT_ALLOFF_LOCK_SHA256 = (
    "fc05f9924a368834594b5f4c298975aedb6cdce86027593f557ea488ebeedde0"
)
_PCTRL_CONTROL = {
    "llama3-8b-inst": {
        "status": "SAME_STATE_RETRY_EXHAUSTED",
        "tau": Fraction(1, 8),
        "snapshot": (
            "d3d601af7dcb2f8835766f72f7a11e8112ee8a26b3205c99a2b0fae382ca1ce4"
        ),
    },
    "qwen2.5-7b-inst": {
        "status": "MIN_DT_EXHAUSTED",
        "tau": Fraction(1, 16),
        "snapshot": (
            "35e3a18211c6f1eca6d2da38adfabb388e6c9f7618e98b7b3c7890569b276017"
        ),
    },
}


def newnll_p_soft_hard_panel_specs() -> tuple[AdaptivePanelSpec, ...]:
    common = (
        AdaptiveVariant.FR_A8,
        RoutingObjective.TARGET_NEW_NLL,
    )
    return (
        AdaptivePanelSpec(
            NEWNLL_P_SOFT_HARD_PANEL_LABELS[0],
            *common,
            FunctionalPDecisionPolicy.PCTRL,
            PreservationConstraintPolicy.LOCKED,
            FunctionalPFieldPolicy.PROBE_ONLY,
        ),
        AdaptivePanelSpec(
            NEWNLL_P_SOFT_HARD_PANEL_LABELS[1],
            *common,
            FunctionalPDecisionPolicy.PCTRL,
            PreservationConstraintPolicy.LOCKED,
            FunctionalPFieldPolicy.SOFT_HARD,
        ),
        AdaptivePanelSpec(
            NEWNLL_P_SOFT_HARD_PANEL_LABELS[2],
            *common,
            FunctionalPDecisionPolicy.OBSERVATION_ONLY,
            PreservationConstraintPolicy.LOCKED,
            FunctionalPFieldPolicy.PROBE_ONLY,
        ),
    )


def expected_newnll_p_soft_hard_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P-soft result alias differs")
    return f"s05-newnll-p-soft-hard-p1r5-{alias}-v1"


@dataclass(frozen=True, slots=True)
class NewNLLPSoftHardForecast:
    alias: str
    variant_count: int
    maximum_soft_arm_trial_count: int
    maximum_soft_arm_field_count: int
    maximum_probe_fields: int
    probes_per_field: int
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
            or self.variant_count != 3
            or self.maximum_soft_arm_trial_count != 128
            or self.maximum_soft_arm_field_count != 32
            or self.maximum_probe_fields != 66
            or self.probes_per_field != 6
            or self.reserve_fraction != 0.05
            or self.conservative_time_seconds
            != math.ceil(self.pre_reserve_seconds * 1.05)
            or self.conservative_gpu_peak_mib > self.allocation_memory_mib
            or self.conservative_host_peak_mib > self.allocation_memory_mib
            or self.conservative_time_seconds > self.allocation_time_seconds
            or not self.fits_same_envelope
        ):
            raise ODEBFContractError("P-soft resource forecast differs")

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> str:
        return canonical_hash(self.raw_free_payload())


def forecast_newnll_p_soft_hard_panel(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> NewNLLPSoftHardForecast:
    """Observed-control plus outcome-blind soft-arm/probe upper bound."""

    parent = forecast_p1_adaptive_b10_memory(
        artifact_lock_path, base_model_lock_path, alias
    )
    # PCTRL/FPOFF are reused-seal controls with calibrated 3,600 s combined.
    # The new arm retains the full 128-trial/32-field cap. Target-new routing
    # uses six locked contexts; each field adds baseline+five P/H-shared
    # teacher probes. The probe overlay is streaming, so peak memory is parent.
    pre_reserve = 3_600
    pre_reserve += 128 * 6 * 60
    pre_reserve += 32 * 6 * 60
    pre_reserve += 66 * 6 * 30
    pre_reserve += 66 * 10 + 68 * 60 + 1_800
    conservative = math.ceil(pre_reserve * 1.05)
    return NewNLLPSoftHardForecast(
        alias,
        3,
        128,
        32,
        66,
        6,
        parent.identity(),
        pre_reserve,
        0.05,
        parent.forecast_gpu_peak_mib,
        parent.forecast_host_peak_mib,
        conservative,
        65_000,
        86_400,
        conservative <= 86_400,
    )


def validate_pctrl_probe_control(
    alias: str,
    spec: AdaptivePanelSpec,
    rollout: VariantRollout,
) -> None:
    if spec.label != NEWNLL_P_SOFT_HARD_PANEL_LABELS[0]:
        return
    expected = _PCTRL_CONTROL.get(alias)
    if expected is None:
        raise ODEBFContractError("P-soft control alias differs")
    snapshots = [item.snapshot_sha256 for item in rollout.snapshots]
    if (
        rollout.routing_objective != RoutingObjective.TARGET_NEW_NLL.value
        or rollout.functional_p_decision
        != FunctionalPDecisionPolicy.PCTRL.value
        or rollout.functional_p_field_policy
        != FunctionalPFieldPolicy.PROBE_ONLY.value
        or rollout.status != expected["status"]
        or rollout.termination_label != expected["status"]
        or rollout.accepted_t != expected["tau"]
        or rollout.k_acc != 1
        or rollout.n_trial != 6
        or rollout.n_reject != 5
        or rollout.field_build_count != 2
        or snapshots != [expected["snapshot"]]
    ):
        raise ODEBFContractError("HOLD_INVALID_CONTROL: PCTRL probe differs")


def _read_receipts(rollout: VariantRollout, category: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in sorted(rollout.recorder.root.glob(f"{category}-*.json")):
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError("P-soft receipt path differs")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("category") != category:
            raise ODEBFContractError("P-soft receipt category differs")
        values.append(value)
    return values


def _field_common_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    probe = value["functional_p_field"]
    return {
        "field_sha256": value["field_sha256"],
        "signed_slopes": value["signed_slopes"],
        "raw_velocity": value["raw_velocity"],
        "pre_soft_velocity": probe["pre_soft_velocity"],
        "controller_p_sample_order_sha256": probe[
            "controller_p_sample_order_sha256"
        ],
        "controller_p_baseline_identity_sha256": probe[
            "controller_p_baseline_identity_sha256"
        ],
        "factor_state_sha256": probe["factor_state_sha256"],
        "secant_cache_identity_sha256": probe["secant"][
            "cache_identity_sha256"
        ],
        "generic_hp_soft_capable": probe["generic_hp_soft_capable"],
        "historical_soft_active": probe["historical_soft_active"],
        "historical_soft_reason": probe["historical_soft_reason"],
    }


def _matched_policy_prefix(
    left: VariantRollout,
    right: VariantRollout,
) -> dict[str, Any]:
    left_fields = _read_receipts(left, "field")
    right_fields = _read_receipts(right, "field")
    if not left_fields or not right_fields:
        raise ODEBFContractError("P-soft matched field prefix is absent")
    if _field_common_projection(left_fields[0]) != _field_common_projection(
        right_fields[0]
    ):
        raise ODEBFContractError(
            "PCTRL/FPOFF initial field or probe differs"
        )
    left_trials = _read_receipts(left, "trial")
    right_trials = _read_receipts(right, "trial")
    common = min(len(left_trials), len(right_trials))
    divergence: int | None = None
    for index in range(common):
        ltrial, rtrial = left_trials[index], right_trials[index]
        lcommon = {
            "snapshot_sha256": ltrial["snapshot_sha256"],
            "field_sha256": ltrial["routing"]["field_sha256"],
            "coefficient": ltrial["routing"]["coefficient"],
            "controller_p_observation": ltrial["functional_p_decision"][
                "observation_sha256"
            ],
        }
        rcommon = {
            "snapshot_sha256": rtrial["snapshot_sha256"],
            "field_sha256": rtrial["routing"]["field_sha256"],
            "coefficient": rtrial["routing"]["coefficient"],
            "controller_p_observation": rtrial["functional_p_decision"][
                "observation_sha256"
            ],
        }
        if lcommon != rcommon:
            raise ODEBFContractError(
                "PCTRL/FPOFF common candidate differs before policy divergence"
            )
        if ltrial["gate_accepted"] != rtrial["gate_accepted"]:
            divergence = index
            break
    return {
        "initial_field_common_projection_sha256": canonical_hash(
            _field_common_projection(left_fields[0])
        ),
        "common_candidate_count_through_policy_divergence": (
            common if divergence is None else divergence + 1
        ),
        "first_policy_caused_divergence_trial": divergence,
        "later_state_equality_required": False,
    }


def _local_p_model_diagnostic(rollout: VariantRollout) -> dict[str, Any]:
    """Check the predeclared secant prediction boundary on accepted trials."""

    rows: list[dict[str, Any]] = []
    violating_trials: list[int] = []
    for trial in _read_receipts(rollout, "trial"):
        if trial.get("gate_accepted") is not True:
            continue
        prediction = trial.get("functional_p_probe_prediction")
        if not isinstance(prediction, Mapping):
            raise ODEBFContractError(
                "accepted P-soft trial lacks P prediction receipt"
            )
        baseline = float(prediction["baseline_damage"])
        predicted = float(prediction["predicted_candidate_controller_p"])
        actual = float(prediction["actual_candidate_controller_p"])
        residual = actual - predicted
        predicted_delta = predicted - baseline
        actual_delta = actual - baseline
        if not all(
            math.isfinite(value)
            for value in (
                baseline,
                predicted,
                actual,
                residual,
                predicted_delta,
                actual_delta,
            )
        ):
            raise ODEBFContractError(
                "accepted P-soft P prediction is non-finite"
            )

        def signed(value: float) -> int:
            return int(value > 0.0) - int(value < 0.0)

        sign_disagreement = signed(predicted_delta) != signed(actual_delta)
        residual_exceeds_lock = abs(residual) > 1.0e-3
        trial_ordinal = int(trial["ordinal"])
        if sign_disagreement or residual_exceeds_lock:
            violating_trials.append(trial_ordinal)
        rows.append(
            {
                "trial_ordinal": trial_ordinal,
                "baseline_controller_p": baseline,
                "predicted_candidate_controller_p": predicted,
                "actual_candidate_controller_p": actual,
                "predicted_damage_delta": predicted_delta,
                "actual_damage_delta": actual_delta,
                "signed_prediction_residual": residual,
                "sign_disagreement": sign_disagreement,
                "absolute_residual_exceeds_1e_3": residual_exceeds_lock,
            }
        )
    return {
        "status": (
            "HOLD_LOCAL_P_MODEL"
            if violating_trials
            else "PASS_LOCAL_P_MODEL"
        ),
        "accepted_trial_count": len(rows),
        "violating_trial_ordinals": violating_trials,
        "sign_rule": "sign(predicted_minus_baseline)==sign(actual_minus_baseline)",
        "absolute_residual_limit": 1.0e-3,
        "accepted_trial_receipts": rows,
    }


def newnll_p_soft_hard_refinement(
    capture: Any,
    rollouts: Mapping[Any, VariantRollout],
    receipts: Mapping[tuple[str, int], StepwisePrimaryReceipt],
) -> dict[str, Any]:
    if tuple(rollouts) != NEWNLL_P_SOFT_HARD_PANEL_LABELS:
        raise ODEBFContractError("P-soft panel order differs")
    control = rollouts[NEWNLL_P_SOFT_HARD_PANEL_LABELS[0]]
    soft = rollouts[NEWNLL_P_SOFT_HARD_PANEL_LABELS[1]]
    off = rollouts[NEWNLL_P_SOFT_HARD_PANEL_LABELS[2]]
    control_field = _field_common_projection(_read_receipts(control, "field")[0])
    soft_field = _field_common_projection(_read_receipts(soft, "field")[0])
    if control_field != soft_field:
        raise ODEBFContractError("PCTRL/PSOFT raw field or pre-soft u differs")
    return {
        "schema": "ode-edit-s05-newnll-p-soft-hard-contrasts/v1",
        "pctrl_control_reproduced": True,
        "pctrl_psoft_initial_raw_field_and_u_exact": True,
        "pctrl_fpoff_matched_prefix": _matched_policy_prefix(control, off),
        "treatment_divergence_allows_later_state_difference": True,
        "psoft_minus_pctrl": _target_new_panel_contrast(
            capture,
            rollouts,
            receipts,
            left_label=NEWNLL_P_SOFT_HARD_PANEL_LABELS[0],
            right_label=NEWNLL_P_SOFT_HARD_PANEL_LABELS[1],
        ),
        "psoft_minus_fpoff": _target_new_panel_contrast(
            capture,
            rollouts,
            receipts,
            left_label=NEWNLL_P_SOFT_HARD_PANEL_LABELS[2],
            right_label=NEWNLL_P_SOFT_HARD_PANEL_LABELS[1],
        ),
        "local_p_model_diagnostic": _local_p_model_diagnostic(soft),
        "generic_hp_soft_capable": True,
        "historical_soft_active": False,
        "historical_soft_reason": "EMPTY_HISTORY",
        "future_nonempty_history_gate_required": True,
        "scientific_promotion_authorized": False,
    }


def newnll_p_soft_hard_terminal_metadata() -> dict[str, Any]:
    return {
        "online_rewrite_batch_efficacy_first_hit_name": (
            "online_rewrite_batch_efficacy_first_hit"
        ),
        "postfreeze_primary_names": ["efficacy", "generalization", "locality"],
        "controller_p_name": "controller_p_fixed_outer_entry_schedule",
        "terminal_audit_p_name": "terminal_outer_entry_audit_schedule",
        "controller_and_terminal_p_pooling": False,
        "generic_hp_soft_capable": True,
        "historical_soft_active": False,
        "historical_soft_reason": "EMPTY_HISTORY",
        "actual_functional_h_hard_gate_active": True,
        "actual_functional_p_hard_gate_pctrl_and_psoft": True,
        "structural_h_p_trust_active": True,
        "first_hit_observation_only": True,
        "shadow_to_tau_one": True,
        "persistent_endpoint_commit_count": 0,
        "history_append_count": 0,
        "scientific_promotion_authorized": False,
    }


def validate_newnll_p_soft_hard_lock(
    value: Mapping[str, Any],
    *,
    controller_identity_sha256: str,
    stream_root_digest: str,
    population_root_digest: str,
) -> None:
    specs = newnll_p_soft_hard_panel_specs()
    expected = {
        "schema_version": (
            "ode-edit-s05-newnll-p-soft-hard-p1r5-numerical-lock/v1"
        ),
        "instruction_id": NEWNLL_P_SOFT_HARD_INSTRUCTION_ID,
        "parent_source_head": PARENT_SOURCE_HEAD,
        "parent_alloff_numerical_lock_sha256": PARENT_ALLOFF_LOCK_SHA256,
        "status": "SEALED_REUSED_SAMPLE_CAUSAL_DIAGNOSTIC",
        "benchmark": "counterfact",
        "common_seed": 41,
        "edit_batch_size": BATCH_SIZE,
        "sequential_batch_count": 1,
        "scientific_sample_reused_for_causal_diagnostic": True,
        "scientific_promotion_authorized": False,
        "stream_root_digest": stream_root_digest,
        "p_population_root_digest": population_root_digest,
        "panel_labels": list(NEWNLL_P_SOFT_HARD_PANEL_LABELS),
        "routing_objective_by_label": {
            item.label: item.routing_objective.value for item in specs
        },
        "clock_variant_by_label": {
            item.label: item.clock_variant.value for item in specs
        },
        "functional_p_decision_by_label": {
            item.label: item.functional_p_decision.value for item in specs
        },
        "functional_p_field_policy_by_label": {
            item.label: item.functional_p_field_policy.value for item in specs
        },
        "preservation_constraints_by_label": {
            item.label: item.preservation_constraints.value for item in specs
        },
        "variant_lock_sha256": {
            item.label: adaptive_lock(item.clock_variant).identity()
            for item in specs
        },
        "controller_identity_sha256": controller_identity_sha256,
        "functional_replay_soft_hard": {
            "h_ref": 0.125,
            "functional_p_budget_nats": 0.001,
            "functional_h_budget_nats": 0.001,
            "probe_basis": "ordered-unit-actuator-bf16-overlay",
            "probe_layer_order": [4, 5, 6, 7, 8],
            "probe_baseline_count_per_field": 1,
            "probe_endpoint_count_per_field": 5,
            "same_state_retry_reuses_probe": True,
            "signed_secants_unclipped": True,
            "stage1": "minimum-single-worst-normalized-slack",
            "stage2": "capacity-metric-closest-to-structural-bf-u",
            "sigma_tie_tolerance": 1.0e-8,
            "actual_candidate_p_hard_gate_unchanged": True,
            "actual_candidate_h_hard_gate_unchanged": True,
            "generic_hp_soft_capable": True,
            "historical_soft_active": False,
            "historical_soft_reason": "EMPTY_HISTORY",
        },
        "unchanged_contract": {
            "routing_objective": "TARGET_NEW_NLL",
            "target_state_objective": "MARGIN_LOCKED",
            "adaptive_t_max": 1.0,
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
            "conservative_time_seconds": 83601,
        },
        "model_alias_specific_controller_branches": 0,
        "retry_submission_count": 0,
    }
    observed = dict(value)
    observed.pop("root_digest", None)
    if observed != expected:
        raise ODEBFContractError("P-soft numerical lock differs")


def controller_identity_for_lock() -> str:
    return P1ControllerLock().identity()
