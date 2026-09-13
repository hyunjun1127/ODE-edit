"""P1R55 warm-state RMS/PDZ-floored-rate experiment bindings.

The warm entries are reconstructed by the sealed canonical A0 sequential
prefix.  Prefix replay is setup-only; exactly one B100 probe contributes to
each Phase-1 scientific cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import ODEBFContractError, canonical_hash
from .p1r52_c_writer_kstep_cache_sequential import (
    Phase3SequentialExperimentBinding,
    run_phase3,
)
from .p1r52_target_timescale import (
    TargetTimescaleCell,
    reconstruct_target_subcycle_final_selected,
    schedule_for_cell,
)
from .p1r54_energyfree_localz import EnergyFreeLocalZArm, EnergyFreeLocalZPolicy
from .p1r54_realization_policy import RealizationPolicy, RealizationTransitionController
from .p1r55_objective_risk import ObjectiveRiskPolicy, RequestLocalRiskEvaluator
from .p1r55_pdz_floored_rate import AmplitudePolicy, RequestLocalPDZPolicy
from .scalable_batched_model import evaluate_scalable_target_new_objective


INSTRUCTION_ID = "ODEEDIT-S05-P1R55-REQUEST-LOCAL-RMS-PDZ-FLOORED-RATE-V1"
METHOD_ID = "P1R55-REQUEST-LOCAL-RMS-PDZ-FLOORED-RATE-WARM-T1"
PHASE1_ARRAY = "0-11%3"
PHASE1_SNAPSHOTS = (1, 6, 10)
HELDOUT_K_INDICES = (0, 3, 7)
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"


@dataclass(frozen=True, slots=True)
class P1R55Arm:
    index: int
    label: str
    risk_policy: ObjectiveRiskPolicy
    amplitude_policy: AmplitudePolicy

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "label": self.label,
            "risk_policy": self.risk_policy.value,
            "amplitude_policy": self.amplitude_policy.value,
            "selector_objective_policy": self.risk_policy.value,
            "writer": "C3_OFFICIAL_ALPHAEDIT_KSTEP",
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


ARMS = (
    P1R55Arm(0, "A0", ObjectiveRiskPolicy.MEAN, AmplitudePolicy.PDZ),
    P1R55Arm(1, "A1", ObjectiveRiskPolicy.MEAN, AmplitudePolicy.PDZ_FLOORED_RATE),
    P1R55Arm(2, "A2", ObjectiveRiskPolicy.RMS, AmplitudePolicy.PDZ),
    P1R55Arm(3, "A3", ObjectiveRiskPolicy.RMS, AmplitudePolicy.PDZ_FLOORED_RATE),
)


@dataclass(frozen=True, slots=True)
class Phase1WarmCell:
    cell: int
    snapshot_batch: int
    arm: P1R55Arm
    role: str
    result_name: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cell": self.cell,
            "snapshot_batch": self.snapshot_batch,
            "snapshot_label": f"B{self.snapshot_batch}_ENTRY",
            "arm": self.arm.raw_free_payload(),
            "role": self.role,
            "result_name": self.result_name,
            "scientific_probe_batch_count": 1,
            "scientific_probe_request_count": 100,
            "canonical_prefix_replay_batch_count": self.snapshot_batch - 1,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


PHASE1_CELLS = tuple(
    Phase1WarmCell(
        snapshot_position * len(ARMS) + arm.index,
        snapshot_batch,
        arm,
        (
            f"r55-rms-pdz-rate-phase1-warm-b{snapshot_batch:02d}-"
            f"{arm.label.lower()}"
        ),
        (
            "s05-p1r55-rms-pdz-rate-phase1-warm-"
            f"b{snapshot_batch:02d}-{arm.label.lower()}-t1-tech-r2-v1"
        ),
    )
    for snapshot_position, snapshot_batch in enumerate(PHASE1_SNAPSHOTS)
    for arm in ARMS
)
PHASE1_ROLES = tuple(item.role for item in PHASE1_CELLS)
PHASE1_RESULT_NAMES = {item.role: item.result_name for item in PHASE1_CELLS}


def phase1_cell(cell: int) -> Phase1WarmCell:
    if isinstance(cell, bool) or not 0 <= cell < len(PHASE1_CELLS):
        raise ODEBFContractError("P1R55 Phase1 cell differs")
    return PHASE1_CELLS[cell]


def phase1_cell_for_role(role: str) -> Phase1WarmCell:
    for item in PHASE1_CELLS:
        if item.role == role:
            return item
    raise ODEBFContractError("P1R55 Phase1 role differs")


def is_p1r55_phase1_role(role: str | None) -> bool:
    return role in PHASE1_ROLES


def expected_phase1_result_name(role: str) -> str:
    try:
        return PHASE1_RESULT_NAMES[role]
    except KeyError as exc:
        raise ODEBFContractError("P1R55 Phase1 result role differs") from exc


def _probe_amplitude_factory(config: Phase1WarmCell):
    def factory(batch_index: int, request_count: int) -> RequestLocalPDZPolicy:
        if batch_index != config.snapshot_batch or request_count != 100:
            raise ODEBFContractError("P1R55 probe amplitude batch differs")
        return RequestLocalPDZPolicy(
            risk_policy=config.arm.risk_policy,
            amplitude_policy=config.arm.amplitude_policy,
            request_count=request_count,
            outer_count=8,
            microsteps_per_outer=1,
        )

    return factory


def _probe_objective_factory(config: Phase1WarmCell):
    def factory(batch_index: int, request_count: int) -> RequestLocalRiskEvaluator:
        if batch_index != config.snapshot_batch or request_count != 100:
            raise ODEBFContractError("P1R55 probe objective batch differs")
        return RequestLocalRiskEvaluator(config.arm.risk_policy)

    return factory


def _prefix_amplitude_factory(
    batch_index: int, request_count: int
) -> EnergyFreeLocalZPolicy:
    if not 1 <= batch_index <= 9 or request_count != 100:
        raise ODEBFContractError("P1R55 prefix amplitude batch differs")
    return EnergyFreeLocalZPolicy(
        EnergyFreeLocalZArm.PDZ, request_count=request_count, outer_count=8
    )


def _prefix_objective_factory(batch_index: int, request_count: int) -> Any:
    if not 1 <= batch_index <= 9 or request_count != 100:
        raise ODEBFContractError("P1R55 prefix objective batch differs")
    return evaluate_scalable_target_new_objective


def _realization_factory(
    batch_index: int, request_count: int
) -> RealizationTransitionController:
    if not 1 <= batch_index <= 10 or request_count != 100:
        raise ODEBFContractError("P1R55 realization batch differs")
    return RealizationTransitionController(
        RealizationPolicy.POST_WRITE_W_REALIZATION_RESET,
        request_count=request_count,
        outer_count=8,
    )


def phase1_metadata(config: Phase1WarmCell) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r55-rms-pdz-rate-phase1-cell/v1",
        "instruction_id": INSTRUCTION_ID,
        "cell": config.raw_free_payload(),
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "model_alias": "llama3-8b-inst",
        "model_precision": "FULL_FP32",
        "K": 8,
        "target_h": 0.125,
        "target_microsteps_per_outer": 1,
        "target_horizon": 1.0,
        "writer_calls_scientific_probe": 8,
        "writer_layer_applies_scientific_probe": 40,
        "accepted_z_heldout_outer_indices": [1, 4, 8],
        "warm_entry_construction": "CANONICAL_A0_PREFIX_REPLAY",
        "prefix_replay_scientific_denominator_count": 0,
        "setup_vs_scientific_probe_separated": True,
        "current_batch_history_inclusion_count": 0,
        "shared_or_cross_request_strength_decision_count": 0,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return {"p1r55_rms_pdz_rate": payload}


def build_phase1_binding(
    config: Phase1WarmCell,
    *,
    easyedit_root: Path = Path("/mnt/raid5/janghj/EasyEdit"),
) -> Phase3SequentialExperimentBinding:
    arm_tag = config.arm.label.lower()
    return Phase3SequentialExperimentBinding(
        role=config.role,
        writer_arm="C3-KSTEP-CACHE",
        instruction_id=INSTRUCTION_ID,
        method_id=f"{METHOD_ID}-{config.arm.label}-B{config.snapshot_batch:02d}",
        batch_schema="ode-edit-s05-p1r55-rms-pdz-rate-phase1-batch/v1",
        final_schema="ode-edit-s05-p1r55-rms-pdz-rate-phase1-final/v1",
        terminal_schema="ode-edit-s05-p1r55-rms-pdz-rate-phase1-terminal/v1",
        manifest_schema="ode-edit-s05-p1r55-rms-pdz-rate-phase1-manifest/v1",
        terminal_status="TERMINAL_VALID",
        return_status="P1R55_RMS_PDZ_RATE_PHASE1_TERMINAL",
        stage_prefix=(
            f"p1r55_rms_pdz_rate_phase1_b{config.snapshot_batch:02d}_{arm_tag}"
        ),
        selected_target_resolver=reconstruct_target_subcycle_final_selected,
        heldout_step_indices=HELDOUT_K_INDICES,
        target_subcycle_schedule=schedule_for_cell(TargetTimescaleCell.Z0_COARSE),
        amplitude_policy_factory=_probe_amplitude_factory(config),
        objective_evaluator_factory=_probe_objective_factory(config),
        realization_controller_factory=_realization_factory,
        easyedit_root=easyedit_root,
        metadata=phase1_metadata(config),
        probe_batch_index=config.snapshot_batch,
        canonical_prefix_amplitude_policy_factory=_prefix_amplitude_factory,
        canonical_prefix_objective_evaluator_factory=_prefix_objective_factory,
    )


def run_p1r55_rms_pdz_rate(
    model: Any,
    tokenizer: Any,
    *,
    role: str,
    easyedit_root: Path = Path("/mnt/raid5/janghj/EasyEdit"),
    **kwargs: Any,
) -> dict[str, Any]:
    config = phase1_cell_for_role(role)
    return run_phase3(
        model,
        tokenizer,
        role=role,
        experiment_binding=build_phase1_binding(
            config, easyedit_root=easyedit_root
        ),
        **kwargs,
    )


__all__ = [
    "ARMS",
    "HELDOUT_K_INDICES",
    "INSTRUCTION_ID",
    "METHOD_ID",
    "PHASE1_ARRAY",
    "PHASE1_CELLS",
    "PHASE1_RESULT_NAMES",
    "PHASE1_ROLES",
    "PHASE1_SNAPSHOTS",
    "Phase1WarmCell",
    "P1R55Arm",
    "build_phase1_binding",
    "expected_phase1_result_name",
    "is_p1r55_phase1_role",
    "phase1_cell",
    "phase1_cell_for_role",
    "phase1_metadata",
    "run_p1r55_rms_pdz_rate",
]
