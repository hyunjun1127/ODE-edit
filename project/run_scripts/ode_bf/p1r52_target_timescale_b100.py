"""One sealed B100 target-timescale cell with C3 K-step writer authority."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .contracts import COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1_runtime import _atomic_write_once
from .p1_scalable_batched_experiment import _model_w0_contract
from .p1_state import ArmWeightSnapshot, P1Arm, snapshot_touched_weights
from .p1r36_independent_b10x10_runtime import _hashes
from .p1r52_b100x10_stream import BATCH_SIZE, ROUND_COUNT
from .p1r52_c_writer_kstep import CKStepWriterRuntime
from .p1r52_c_writer_kstep_cache import (
    KStepBatchEntryCachePolicy,
    restore_alpha_module_cache,
    snapshot_alpha_module_cache,
)
from .p1r52_joint_pc_fp32_runtime import _fp32_target_and_j0, _restore
from .p1r52_joint_pc_independent_fp32_runtime import (
    _assert_no_low_precision_activity,
    _gpu_observation,
    _raw_free_json_tree,
    full_fp32_parameter_inventory,
)
from .p1r52_joint_pc_runtime import STREAM_ORDER, STREAM_ROOT
from .p1r52_target_official_alphaedit_writer import isolated_alphaedit_module_state
from .p1r52_target_timescale import (
    INSTRUCTION_ID,
    METHOD_ID,
    SCHEDULES,
    TargetSubcycleSchedule,
    TargetTimescaleCell,
    reconstruct_target_subcycle_final_selected,
    schedule_for_cell,
)
from .scalable_batched_runtime import P1R23_GRID_COUNT, scalable_ordered_request_digest


ROLE_PREFIX = "r52-target-timescale-b100-c3-kstep-"
ROLES = tuple(f"{ROLE_PREFIX}{item.cell.value.lower()}" for item in SCHEDULES)
TECHNICAL_ATTEMPT_SUFFIX = "tech-r4"
RESULT_NAMES = {
    role: (
        "s05-p1r52-target-timescale-b100-"
        f"{item.cell.value.lower()}-{TECHNICAL_ATTEMPT_SUFFIX}-v1"
    )
    for role, item in zip(ROLES, SCHEDULES, strict=True)
}
C3_ARM = "C3-KSTEP-CACHE"
HELDOUT_K_INDICES = (0, 3, 7)


def role_for_cell(cell: int) -> str:
    if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < len(ROLES):
        raise ODEBFContractError("target-timescale array cell differs")
    return ROLES[cell]


def schedule_for_role(role: str) -> TargetSubcycleSchedule:
    if role not in ROLES:
        raise ODEBFContractError("target-timescale role differs")
    return SCHEDULES[ROLES.index(role)]


def expected_result_name(role: str) -> str:
    try:
        return RESULT_NAMES[role]
    except KeyError as exc:
        raise ODEBFContractError("target-timescale result role differs") from exc


def is_target_timescale_role(role: str | None) -> bool:
    return role in ROLES


def run_target_timescale_b100(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    role: str,
    destination: Path,
    raw_root: Path,
    stages: Any,
    source_head: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    stream: Mapping[str, Any],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: Any,
    theta0_cache: Any,
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    job_ledger: ComputeLedger,
    request_microbatch_size: int,
    fp32_runtime: Any | None = None,
    easyedit_root: Path = Path("/data/janghj/EasyEdit"),
    target_schedule_override: TargetSubcycleSchedule | None = None,
    amplitude_policy: Any | None = None,
    objective_evaluator: Any | None = None,
    realization_controller: Any | None = None,
    heldout_k_indices: tuple[int, ...] | None = None,
    experiment_instruction_id: str = INSTRUCTION_ID,
    experiment_method_id: str = METHOD_ID,
    terminal_schema: str = "ode-edit-s05-p1r52-target-timescale-cell-terminal/v1",
    manifest_schema: str = "ode-edit-s05-p1r52-target-timescale-cell-manifest/v1",
    terminal_status: str = "P1R52_TARGET_TIMESCALE_TERMINAL",
    stage_prefix: str = "target_timescale",
    amplitude_writer_layer_apply_count_key: str = "p1r53_writer_layer_apply_count",
    experiment_metadata: Mapping[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    target_schedule = (
        schedule_for_role(role)
        if target_schedule_override is None
        else target_schedule_override
    )
    heldout_indices = HELDOUT_K_INDICES if heldout_k_indices is None else heldout_k_indices
    if (
        alias != "llama3-8b-inst"
        or len(stream_batches) != ROUND_COUNT
        or any(len(batch) != BATCH_SIZE for batch in stream_batches)
        or stream.get("root_digest") != STREAM_ROOT
        or stream.get("all_request_order_sha256") != STREAM_ORDER
        or fp32_runtime is None
        or torch.is_autocast_enabled()
        or torch.is_autocast_enabled("cpu")
    ):
        raise ODEBFContractError("target-timescale stream/FP32 boundary differs")
    requests = tuple(stream_batches[0])
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    parameter_inventory = full_fp32_parameter_inventory(model)
    w0_contract = _model_w0_contract(touched)
    w0_hashes = _hashes(touched)
    w0_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    if w0_hashes != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("target-timescale W0 differs")
    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    seed_all(COMMON_SEED)
    entry_values = {name: value.detach().cpu().clone() for name, value in touched.items()}
    entry_receipt, _ = snapshot_touched_weights(P1Arm.R_BF, 0, touched)
    cache_checkpoint = snapshot_alpha_module_cache(alpha_main)
    policy = KStepBatchEntryCachePolicy(
        arm=C3_ARM,
        batch_index=1,
        history_keys_by_layer=None,
        history_version=0,
        alpha_main=alpha_main,
        alpha_entry_snapshot=cache_checkpoint,
        expected_official_entry_sha256=None,
    )
    created: list[CKStepWriterRuntime] = []

    def factory(objective_plan: Any, capture_plan: Any) -> CKStepWriterRuntime:
        runtime = CKStepWriterRuntime(
            arm=C3_ARM,
            model=model,
            tokenizer=tokenizer,
            requests=requests,
            hparams=hparams,
            projector=projector,
            contexts=contexts,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            controller_lock=controller_lock,
            objective_plan=objective_plan,
            capture_plan=capture_plan,
            dataset_path=dataset_path,
            private_root=raw_root / "private",
            job_ledger=job_ledger,
            alpha_cache_status="ALPHA_CACHE_BATCH_ENTRY_SNAPSHOT",
            cache_policy=policy,
            selected_target_resolver=reconstruct_target_subcycle_final_selected,
            heldout_step_indices=heldout_indices,
        )
        created.append(runtime)
        return runtime

    started = time.perf_counter()
    cache_commit: Any = None
    public: Mapping[str, Any] | None = None
    target: torch.Tensor | None = None
    try:
        with isolated_alphaedit_module_state() as alpha_state:
            try:
                target, public, _, _ = _fp32_target_and_j0(
                    model,
                    tokenizer,
                    requests,
                    case_root=raw_root,
                    hparams=hparams,
                    projector=projector,
                    contexts=contexts,
                    covariance_registry=covariance_registry,
                    projector_sha256=projector_sha256,
                    controller_lock=controller_lock,
                    request_by_sha256=request_by_sha256,
                    population_by_sha256=population_by_sha256,
                    schedule=schedule,
                    theta0_cache=theta0_cache,
                    touched=touched,
                    base_receipt=entry_receipt,
                    base_values=entry_values,
                    request_microbatch_size=request_microbatch_size,
                    job_ledger=job_ledger,
                    c_kstep_writer_factory=factory,
                    target_subcycle_schedule=target_schedule,
                    amplitude_policy=amplitude_policy,
                    objective_evaluator=objective_evaluator,
                    realization_controller=realization_controller,
                    easyedit_root=easyedit_root,
                )
                if len(created) != 1:
                    raise ODEBFStateError("target-timescale writer runtime count differs")
                runtime = created[0]
                runtime.assert_complete()
                writer_layer_apply_count = sum(
                    int(item.compute["native_apply_count"])
                    for item in runtime.executions
                )
                if (
                    len(runtime.executions) != P1R23_GRID_COUNT
                    or sum(
                        int(item.compute["heldout_schedule_step_count"])
                        for item in runtime.executions
                    )
                    != len(heldout_indices)
                    or any(
                        int(item.compute["heldout_evaluator_count"])
                        != (3 if item.step_index in heldout_indices else 0)
                        for item in runtime.executions
                    )
                ):
                    raise ODEBFStateError("target-timescale heldout clock differs")
                if amplitude_policy is not None and writer_layer_apply_count != 40:
                    raise ODEBFStateError("external amplitude writer layer clock differs")
                cache_commit = policy.commit_after_k8()
                commit_hashes = _hashes(touched)
                if commit_hashes == w0_hashes:
                    raise ODEBFStateError("target-timescale C3 writer made no transition")
                if not alpha_state["restored"]:
                    # The context marks restoration only on exit; this branch is
                    # intentionally evaluated after the context below.
                    pass
            except BaseException:
                _restore(
                    touched,
                    entry_values,
                    mutation_lock=mutation_lock,
                    entry_contract=w0_contract,
                )
                restore_alpha_module_cache(alpha_main, cache_checkpoint)
                raise
        if not alpha_state["restored"]:
            raise ODEBFStateError("target-timescale Alpha module state restore differs")
    except BaseException:
        _restore(
            touched,
            base_values,
            mutation_lock=mutation_lock,
            entry_contract=w0_contract,
        )
        raise

    assert target is not None and public is not None and cache_commit is not None
    runtime = created[0]
    accepted_root = raw_root / "raw" / "target" / "ode" / C3_ARM.lower()
    accepted_transitions = [
        json.loads((accepted_root / f"accepted-k{index}.json").read_text(encoding="utf-8"))
        for index in range(1, P1R23_GRID_COUNT + 1)
    ]
    if [item["identity_sha256"] for item in accepted_transitions] != list(
        public["accepted_receipt_sha256"]
    ):
        raise ODEBFStateError("target-timescale accepted receipt chain differs")
    micro_receipts = [
        micro
        for accepted in accepted_transitions
        for micro in accepted["target_update"]["microstep_receipts"]
    ]
    expected_field_evaluations = target_schedule.total_field_evaluations
    if (
        len(micro_receipts) != expected_field_evaluations
        or [int(item["global_field_evaluation_ordinal"]) for item in micro_receipts]
        != list(range(expected_field_evaluations))
        or any(float(item["target_dt"]) != float(target_schedule.target_dt) for item in micro_receipts)
        or any(int(item["early_break_count"]) != 0 for item in micro_receipts)
    ):
        raise ODEBFStateError("target-timescale field clock ledger differs")
    restore = _restore(
        touched,
        base_values,
        mutation_lock=mutation_lock,
        entry_contract=w0_contract,
    )
    if _hashes(touched) != w0_hashes or any(
        int(touched[name].data_ptr()) != w0_pointers[name] for name in touched
    ):
        raise ODEBFStateError("target-timescale terminal W0 restore differs")
    transition_payload: dict[str, Any] = {
        "schema": terminal_schema,
        "instruction_id": experiment_instruction_id,
        "method_id": experiment_method_id,
        "status": "TERMINAL_VALID",
        "source_head": source_head,
        "role": role,
        "cell": target_schedule.cell.value,
        "schedule": target_schedule.raw_free_payload(),
        "request_count": len(requests),
        "request_order_sha256": request_order,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "sample_duplication_count": 0,
        "completed_outer_count": P1R23_GRID_COUNT,
        "configured_target_microstep_count": expected_field_evaluations,
        "executed_target_microstep_count": len(micro_receipts),
        "target_field_evaluation_count": len(micro_receipts),
        "writer_call_count": len(runtime.executions),
        "writer_call_indices": [item.step_index + 1 for item in runtime.executions],
        "writer_authority": "FINAL_SELECTED_TARGET_ONLY",
        "intermediate_writer_authority_count": 0,
        "heldout_outer_indices": [index + 1 for index in heldout_indices],
        "heldout_evaluator_count": sum(
            int(item.compute["heldout_evaluator_count"])
            for item in runtime.executions
        ),
        "heldout_decision_influence_count": 0,
        "last_micro_entry_objective_primary_count": 0,
        "final_selected_endpoint_objective_primary_count": P1R23_GRID_COUNT,
        "target_terminal_sha256": tensor_sha256(target),
        "target_public_identity": public["identity_sha256"],
        "microstep_trajectory": micro_receipts,
        "outer_transitions": accepted_transitions,
        "kstep_executions": [item.raw_free_payload() for item in runtime.executions],
        "alpha_cache": cache_commit.receipt,
        "cache_entry_reuse_count": P1R23_GRID_COUNT,
        "cache_append_count": 1,
        "same_batch_current_key_history_inclusion_count": 0,
        "rollback_count": 0,
        "retry_count": 0,
        "native_execution_count": 0,
        "native_reference_status": "CANONICAL_EXTERNAL_REFERENCE_REUSE_ONLY",
        "dtype_contract": {
            "status": "FULL_FP32_PASS",
            "parameter_inventory": parameter_inventory,
            "bf16_fp16_path_count": 0,
            "autocast_count": 0,
        },
        "job_compute": job_ledger.raw_free_payload(),
        "runtime_after_model_preflight_seconds": time.perf_counter() - started,
        "gpu_host_observation": _gpu_observation(),
        "terminal_W0_restore": restore,
        "W0_restored": True,
        "scientific_promotion": False,
    }
    if amplitude_policy is not None:
        if not amplitude_writer_layer_apply_count_key:
            raise ODEBFContractError("amplitude writer layer counter key differs")
        transition_payload["amplitude_policy_terminal"] = dict(
            amplitude_policy.terminal_receipt()
        )
        transition_payload[amplitude_writer_layer_apply_count_key] = sum(
            int(item.compute["native_apply_count"])
            for item in runtime.executions
        )
    if experiment_metadata:
        overlap = set(transition_payload).intersection(experiment_metadata)
        if overlap:
            raise ODEBFContractError(
                f"target-timescale experiment metadata collides: {sorted(overlap)}"
            )
        transition_payload.update(experiment_metadata)
    _assert_no_low_precision_activity(transition_payload)
    transition_payload, tensor_paths = _raw_free_json_tree(transition_payload)
    transition_payload["raw_free_tensor_paths"] = list(tensor_paths)
    transition_payload["identity_sha256"] = canonical_hash(transition_payload)
    terminal_sha = _atomic_write_once(destination / "terminal.json", transition_payload)
    manifest: dict[str, Any] = {
        "schema": manifest_schema,
        "instruction_id": experiment_instruction_id,
        "source_head": source_head,
        "role": role,
        "cell": target_schedule.cell.value,
        "terminal_sha256": terminal_sha,
        "schedule_identity": target_schedule.raw_free_payload()["identity_sha256"],
        "request_order_sha256": request_order,
        "W0_restored": True,
        "terminal_W0_restore": restore,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    stages.record(
        f"{stage_prefix}_{target_schedule.cell.value.lower()}_terminal",
        {
            "target_field_evaluation_count": expected_field_evaluations,
            "writer_call_count": P1R23_GRID_COUNT,
            "cache_append_count": 1,
            "W0_restored": True,
        },
    )
    return {
        "status": terminal_status,
        "cell": target_schedule.cell.value,
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "W0_restored": True,
    }


__all__ = [
    "RESULT_NAMES",
    "ROLES",
    "TECHNICAL_ATTEMPT_SUFFIX",
    "expected_result_name",
    "is_target_timescale_role",
    "role_for_cell",
    "run_target_timescale_b100",
    "schedule_for_role",
]
