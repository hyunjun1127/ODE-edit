"""Phase-3 sequential B100 runner for K-step C writers with Alpha cache."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .contracts import COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1_evaluator import EndpointActionFreeze, load_counterfact_cases_after_freeze
from .p1_runtime import _atomic_write_once
from .p1_scalable_batched_experiment import _model_w0_contract
from .p1_state import ArmWeightSnapshot, P1Arm, snapshot_touched_weights
from .p1r36_independent_b10x10_runtime import _hashes
from .p1r52_b100x10_stream import BATCH_SIZE, ROUND_COUNT
from .p1r52_c_writer_kstep import CACHE_ARMS, CKStepWriterRuntime
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
from .p1r52_target_official_alphaedit_writer import _endpoint_summary, _evaluate_w, isolated_alphaedit_module_state
from .scalable_batched_runtime import P1R23_GRID_COUNT, scalable_ordered_request_digest
from .writer_cadence import WriterCadence


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-C-WRITER-THREE-PHASE-V1"
METHOD_ID = "P1R52-C-WRITER-PHASE3-KSTEP-ALPHA-CACHE-SEQUENTIAL-FULL-FP32"
ROLE_PREFIX = "r52-c-writer-phase3-kstep-cache-sequential-full-fp32-"
ROLES = tuple(f"{ROLE_PREFIX}{arm.lower()}" for arm in CACHE_ARMS)
RESULT_NAMES = {
    role: f"s05-p1r52-c-writer-phase3-kstep-cache-sequential-full-fp32-{arm.lower()}-10xb100-v1"
    for role, arm in zip(ROLES, CACHE_ARMS, strict=True)
}


@dataclass(frozen=True, slots=True)
class Phase3SequentialExperimentBinding:
    """Optional scientific binding over the unchanged Phase-3 transaction.

    ``None`` keeps the legacy Phase-3 role map, target field, evaluator clock,
    schemas, and result identities.  A binding may only provide a target
    schedule/amplitude policy and namespaced receipts while reusing the exact
    C3 writer and cache-sequential transaction below.
    """

    role: str
    writer_arm: str
    instruction_id: str
    method_id: str
    batch_schema: str
    final_schema: str
    terminal_schema: str
    manifest_schema: str
    terminal_status: str
    return_status: str
    stage_prefix: str
    selected_target_resolver: Callable[..., tuple[Any, Any]]
    heldout_step_indices: tuple[int, ...]
    target_subcycle_schedule: Any
    amplitude_policy_factory: Callable[[int, int], Any]
    easyedit_root: Path
    metadata: Mapping[str, Any]
    realization_controller_factory: Callable[[int, int], Any] | None = None
    writer_cadence: WriterCadence = WriterCadence.KSTEP_EACH_OUTER

    def __post_init__(self) -> None:
        if (
            not self.role
            or self.writer_arm != "C3-KSTEP-CACHE"
            or not all(
                isinstance(value, str) and value
                for value in (
                    self.instruction_id,
                    self.method_id,
                    self.batch_schema,
                    self.final_schema,
                    self.terminal_schema,
                    self.manifest_schema,
                    self.terminal_status,
                    self.return_status,
                    self.stage_prefix,
                )
            )
            or not callable(self.selected_target_resolver)
            or self.heldout_step_indices != (7,)
            or not callable(self.amplitude_policy_factory)
            or (
                self.realization_controller_factory is not None
                and not callable(self.realization_controller_factory)
            )
            or not self.easyedit_root.is_absolute()
            or not isinstance(self.writer_cadence, WriterCadence)
        ):
            raise ODEBFContractError("Phase3 external experiment binding differs")


def validate_phase3_batch_chain(
    rows: Sequence[Mapping[str, Any]], *, expected_count: int = ROUND_COUNT
) -> Mapping[str, Any]:
    """Validate sequential W/cache continuity without touching model state."""

    if (
        isinstance(expected_count, bool)
        or expected_count != ROUND_COUNT
        or len(rows) != expected_count
        or any(
            row.get("batch_index") != index
            or row.get("cache_entry_width") != (index - 1) * BATCH_SIZE
            or row.get("cache_exit_width") != index * BATCH_SIZE
            for index, row in enumerate(rows, start=1)
        )
        or any(
            rows[index - 1].get("commit_weight_sha256")
            != rows[index].get("entry_weight_sha256")
            for index in range(1, len(rows))
        )
    ):
        raise ODEBFStateError("Phase3 B commit/entry/cache chain differs")
    payload: dict[str, Any] = {
        "schema": "ode-edit-phase3-batch-chain/v1",
        "batch_count": len(rows),
        "entry_widths": [row["cache_entry_width"] for row in rows],
        "exit_widths": [row["cache_exit_width"] for row in rows],
        "commit_to_next_entry_match_count": len(rows) - 1,
        "sample_duplication_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def role_for_cell(cell: int) -> str:
    if isinstance(cell, bool) or not 0 <= cell < len(ROLES):
        raise ODEBFContractError("Phase3 cell differs")
    return ROLES[cell]


def arm_for_role(role: str) -> str:
    if role not in ROLES:
        raise ODEBFContractError("Phase3 role differs")
    return CACHE_ARMS[ROLES.index(role)]


def expected_result_name(role: str) -> str:
    try:
        return RESULT_NAMES[role]
    except KeyError as exc:
        raise ODEBFContractError("Phase3 result role differs") from exc


def is_phase3_role(role: str | None) -> bool:
    return role in ROLES


def run_phase3(
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
    experiment_binding: Phase3SequentialExperimentBinding | None = None,
    **_: Any,
) -> dict[str, Any]:
    if experiment_binding is None:
        arm = arm_for_role(role)
    else:
        if role != experiment_binding.role:
            raise ODEBFContractError("Phase3 external experiment role differs")
        arm = experiment_binding.writer_arm
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
        raise ODEBFContractError("Phase3 matrix/stream/FP32 boundary differs")
    parameter_inventory = full_fp32_parameter_inventory(model)
    w0_contract = _model_w0_contract(touched)
    w0_hashes = _hashes(touched)
    w0_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    if w0_hashes != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("Phase3 W0 differs")
    history: Mapping[int, torch.Tensor] | None = None
    history_version = 0
    prior_official_exit_sha256: str | None = None
    batch_rows: list[dict[str, Any]] = []
    batch_shas: list[str] = []
    cohort_cases: list[tuple[Any, ...]] = []
    started = time.perf_counter()
    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    try:
        with isolated_alphaedit_module_state() as alpha_state:
            for batch_index, request_batch in enumerate(stream_batches, start=1):
                requests = tuple(request_batch)
                seed_all(COMMON_SEED)
                batch_root = raw_root / "batches" / f"b{batch_index:02d}"
                batch_root.mkdir(mode=0o700, parents=True, exist_ok=False)
                entry_contract = _model_w0_contract(touched)
                entry_hashes = _hashes(touched)
                entry_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
                entry_values = {name: value.detach().cpu().clone() for name, value in touched.items()}
                entry_receipt, _ = snapshot_touched_weights(P1Arm.R_BF, batch_index - 1, touched)
                cache_checkpoint = snapshot_alpha_module_cache(alpha_main)
                history_checkpoint = (
                    None if history is None else {layer: value.clone() for layer, value in history.items()}
                )
                policy = KStepBatchEntryCachePolicy(
                    arm=arm,
                    batch_index=batch_index,
                    history_keys_by_layer=(history if arm.startswith(("C0", "C1")) else None),
                    history_version=history_version,
                    alpha_main=(alpha_main if arm.startswith("C3") else None),
                    alpha_entry_snapshot=(cache_checkpoint if arm.startswith("C3") else None),
                    expected_official_entry_sha256=prior_official_exit_sha256,
                    writer_cadence=(
                        WriterCadence.KSTEP_EACH_OUTER
                        if experiment_binding is None
                        else experiment_binding.writer_cadence
                    ),
                )
                amplitude_policy = (
                    None
                    if experiment_binding is None
                    else experiment_binding.amplitude_policy_factory(
                        batch_index, len(requests)
                    )
                )
                realization_controller = (
                    None
                    if experiment_binding is None
                    or experiment_binding.realization_controller_factory is None
                    else experiment_binding.realization_controller_factory(
                        batch_index, len(requests)
                    )
                )
                created: list[CKStepWriterRuntime] = []

                def factory(objective_plan: Any, capture_plan: Any) -> CKStepWriterRuntime:
                    runtime_kwargs: dict[str, Any] = {}
                    if experiment_binding is not None:
                        runtime_kwargs.update(
                            selected_target_resolver=(
                                experiment_binding.selected_target_resolver
                            ),
                            heldout_step_indices=(
                                experiment_binding.heldout_step_indices
                            ),
                            writer_cadence=experiment_binding.writer_cadence,
                        )
                    runtime = CKStepWriterRuntime(
                        arm=arm,
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
                        private_root=batch_root / "private",
                        job_ledger=job_ledger,
                        solve_history_keys_by_layer=(history if arm.startswith(("C0", "C1")) else None),
                        alpha_cache_status="ALPHA_CACHE_BATCH_ENTRY_SNAPSHOT",
                        cache_policy=policy,
                        **runtime_kwargs,
                    )
                    created.append(runtime)
                    return runtime

                batch_started = time.perf_counter()
                try:
                    target_kwargs: dict[str, Any] = {}
                    if experiment_binding is not None:
                        target_kwargs.update(
                            target_subcycle_schedule=(
                                experiment_binding.target_subcycle_schedule
                            ),
                            amplitude_policy=amplitude_policy,
                            realization_controller=realization_controller,
                            easyedit_root=experiment_binding.easyedit_root,
                        )
                    target, public, _, _ = _fp32_target_and_j0(
                        model,
                        tokenizer,
                        requests,
                        case_root=batch_root,
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
                        **target_kwargs,
                    )
                    if len(created) != 1:
                        raise ODEBFStateError("Phase3 writer runtime count differs")
                    runtime = created[0]
                    runtime.assert_complete()
                    commit_hashes = _hashes(touched)
                    if commit_hashes == entry_hashes:
                        raise ODEBFStateError("Phase3 batch endpoint equals entry W")
                    cache_commit = policy.commit_after_k8()
                    history = cache_commit.history_keys_by_layer
                    history_version += 1
                    if arm.startswith("C3"):
                        prior_official_exit_sha256 = str(cache_commit.receipt["official_exit_sha256"])
                    last_metrics = runtime.executions[-1].metrics
                    amplitude_terminal: Mapping[str, Any] | None = None
                    outer_transitions: list[Mapping[str, Any]] | None = None
                    microstep_trajectory: list[Mapping[str, Any]] | None = None
                    if experiment_binding is not None:
                        if amplitude_policy is None:
                            raise ODEBFStateError("Phase3 external amplitude policy missing")
                        amplitude_terminal = amplitude_policy.terminal_receipt()
                        accepted_root = (
                            batch_root / "raw" / "target" / "ode" / arm.lower()
                        )
                        outer_transitions = [
                            json.loads(
                                (accepted_root / f"accepted-k{index}.json").read_text(
                                    encoding="utf-8"
                                )
                            )
                            for index in range(1, P1R23_GRID_COUNT + 1)
                        ]
                        if [
                            item["identity_sha256"] for item in outer_transitions
                        ] != list(public["accepted_receipt_sha256"]):
                            raise ODEBFStateError(
                                "Phase3 external accepted receipt chain differs"
                            )
                        microstep_trajectory = [
                            micro
                            for accepted in outer_transitions
                            for micro in accepted["target_update"]["microstep_receipts"]
                        ]
                        if (
                            len(microstep_trajectory) != P1R23_GRID_COUNT
                            or amplitude_terminal.get("amplitude_call_count")
                            != P1R23_GRID_COUNT
                            or amplitude_terminal.get("rho_refresh_count") != 0
                            or amplitude_terminal.get("forbidden_decision_access_count")
                            != 0
                            or amplitude_terminal.get("additional_model_forward_count")
                            != 0
                            or amplitude_terminal.get("additional_backward_count") != 0
                        ):
                            raise ODEBFStateError(
                                "Phase3 external amplitude/field clock differs"
                            )
                    loaded_cases = tuple(
                        # Each K used the exact same frozen case inventory; retain
                        # the final score rows now and reload only for final W10.
                        # The dataset loader remains outside writer decisions.
                        load_counterfact_cases_after_freeze(
                            dataset_path,
                            requests,
                            EndpointActionFreeze(
                                arm=f"{arm}-b{batch_index:02d}-cohort",
                                sequential_batch=batch_index - 1,
                                request_order_sha256=scalable_ordered_request_digest(
                                    [str(item["request_sha256"]) for item in requests]
                                ),
                                selected_snapshot_sha256=public["identity_sha256"],
                                fixed_budget_slots_completed=P1R23_GRID_COUNT,
                            ),
                            expected_batch_size=BATCH_SIZE,
                        )
                    )
                    cohort_cases.append(loaded_cases)
                    payload: dict[str, Any] = {
                        "schema": (
                            "ode-edit-s05-p1r52-c-writer-phase3-batch/v1"
                            if experiment_binding is None
                            else experiment_binding.batch_schema
                        ),
                        "instruction_id": (
                            INSTRUCTION_ID
                            if experiment_binding is None
                            else experiment_binding.instruction_id
                        ),
                        "method_id": (
                            METHOD_ID
                            if experiment_binding is None
                            else experiment_binding.method_id
                        ),
                        "arm": arm,
                        "batch_index": batch_index,
                        "request_count": BATCH_SIZE,
                        "request_order_sha256": scalable_ordered_request_digest(
                            [str(item["request_sha256"]) for item in requests]
                        ),
                        "entry_weight_sha256": entry_hashes,
                        "commit_weight_sha256": commit_hashes,
                        "terminal_target_sha256": tensor_sha256(target),
                        "target_public_identity": public["identity_sha256"],
                        "kstep_executions": [item.raw_free_payload() for item in runtime.executions],
                        "immediate_post_W": last_metrics["post_writer_W"],
                        "alpha_cache": cache_commit.receipt,
                        "K_writer_call_count": sum(
                            int(item.compute["logical_commit_count"])
                            for item in runtime.executions
                        ),
                        "cache_append_count": 1,
                        "commit_count": 1,
                        "rollback_count": 0,
                        "retry_count": 0,
                        "imputation_count": 0,
                        "batch_total_seconds": time.perf_counter() - batch_started,
                        "dtype_contract": {
                            "status": "FULL_FP32_PASS",
                            "numeric_storage_cast_count": 0,
                            "bf16_fp16_path_count": 0,
                        },
                    }
                    if experiment_binding is not None:
                        if set(payload).intersection(experiment_binding.metadata):
                            raise ODEBFContractError(
                                "Phase3 external batch metadata collides"
                            )
                        payload.update(
                            {
                                "target_field_evaluation_count": len(
                                    microstep_trajectory or ()
                                ),
                                "outer_transitions": outer_transitions,
                                "microstep_trajectory": microstep_trajectory,
                                "amplitude_policy_terminal": amplitude_terminal,
                                "heldout_outer_indices": [8],
                                "heldout_evaluator_decision_influence_count": 0,
                                "writer_layer_apply_count": sum(
                                    int(item.compute["native_apply_count"])
                                    for item in runtime.executions
                                ),
                                "cache_entry_reuse_count": sum(
                                    int(item.compute["logical_commit_count"])
                                    for item in runtime.executions
                                ),
                                "same_batch_current_key_history_inclusion_count": 0,
                                **dict(experiment_binding.metadata),
                            }
                        )
                    _assert_no_low_precision_activity(payload)
                    payload, tensor_paths = _raw_free_json_tree(payload)
                    payload["raw_free_tensor_paths"] = list(tensor_paths)
                    payload["identity_sha256"] = canonical_hash(payload)
                    terminal_sha = _atomic_write_once(batch_root / "terminal.json", payload)
                except BaseException:
                    _restore(touched, entry_values, mutation_lock=mutation_lock, entry_contract=entry_contract)
                    restore_alpha_module_cache(alpha_main, cache_checkpoint)
                    history = history_checkpoint
                    if _hashes(touched) != entry_hashes or any(
                        int(touched[name].data_ptr()) != entry_pointers[name] for name in touched
                    ):
                        raise ODEBFStateError("Phase3 batch rollback W differs")
                    raise
                batch_shas.append(terminal_sha)
                batch_rows.append(
                    {
                        "batch_index": batch_index,
                        "request_order_sha256": payload["request_order_sha256"],
                        "entry_weight_sha256": entry_hashes,
                        "commit_weight_sha256": commit_hashes,
                        "terminal_sha256": terminal_sha,
                        "cache_entry_width": (batch_index - 1) * BATCH_SIZE,
                        "cache_exit_width": batch_index * BATCH_SIZE,
                        "writer_call_count": sum(
                            int(item.compute["logical_commit_count"])
                            for item in runtime.executions
                        ),
                        **(
                            {}
                            if amplitude_terminal is None
                            else {
                                "rho_sha256": amplitude_terminal["rho_sha256"],
                                "field_evaluation_count": len(
                                    microstep_trajectory or ()
                                ),
                            }
                        ),
                    }
                )
                stages.record(
                    f"phase3_{arm.lower()}_b{batch_index:02d}",
                    {
                        "completed_batch_count": len(batch_rows),
                        "K_count": P1R23_GRID_COUNT,
                        "cache_exit_width": batch_index * BATCH_SIZE,
                        "full_fp32": True,
                    },
                )

            final_hashes = _hashes(touched)
            final_rows = []
            for batch_index, loaded_cases in enumerate(cohort_cases, start=1):
                freeze = EndpointActionFreeze(
                    arm=f"{arm}-final-b{batch_index:02d}",
                    sequential_batch=ROUND_COUNT - 1,
                    request_order_sha256=batch_rows[batch_index - 1]["request_order_sha256"],
                    selected_snapshot_sha256=canonical_hash(final_hashes),
                    fixed_budget_slots_completed=P1R23_GRID_COUNT,
                )
                scores = _evaluate_w(model, tokenizer, loaded_cases, freeze=freeze, ledger=job_ledger)
                final_rows.append(
                    {"batch_index": batch_index, "summary": _endpoint_summary(scores), "scores": scores}
                )
            sequential_chain = (
                None
                if experiment_binding is None
                else validate_phase3_batch_chain(batch_rows)
            )
            final_payload = {
                "schema": (
                    "ode-edit-s05-p1r52-c-writer-phase3-final-w10/v1"
                    if experiment_binding is None
                    else experiment_binding.final_schema
                ),
                "arm": arm,
                "cohort_count": len(final_rows),
                "request_count": len(final_rows) * BATCH_SIZE,
                "final_weight_sha256": final_hashes,
                "cohorts": final_rows,
            }
            final_payload["identity_sha256"] = canonical_hash(final_payload)
            final_sha = _atomic_write_once(raw_root / "final-w10.json", final_payload)
        if not alpha_state["restored"]:
            raise ODEBFStateError("Phase3 Alpha module state restore differs")
    except BaseException:
        _restore(touched, base_values, mutation_lock=mutation_lock, entry_contract=w0_contract)
        raise

    restore = _restore(touched, base_values, mutation_lock=mutation_lock, entry_contract=w0_contract)
    if _hashes(touched) != w0_hashes or any(int(touched[name].data_ptr()) != w0_pointers[name] for name in touched):
        raise ODEBFStateError("Phase3 terminal W0 restore differs")
    terminal = {
        "schema": (
            "ode-edit-s05-p1r52-c-writer-phase3-terminal/v1"
            if experiment_binding is None
            else experiment_binding.terminal_schema
        ),
        "instruction_id": (
            INSTRUCTION_ID
            if experiment_binding is None
            else experiment_binding.instruction_id
        ),
        "method_id": (
            METHOD_ID
            if experiment_binding is None
            else experiment_binding.method_id
        ),
        "status": (
            "TERMINAL_VALID"
            if experiment_binding is None
            else experiment_binding.terminal_status
        ),
        "source_head": source_head,
        "role": role,
        "arm": arm,
        "completed_batch_count": len(batch_rows),
        "valid_request_count": len(batch_rows) * BATCH_SIZE,
        "K_writer_call_count": sum(
            int(row.get("writer_call_count", P1R23_GRID_COUNT))
            for row in batch_rows
        ),
        "batch_terminal_sha256": batch_shas,
        "final_w10_sha256": final_sha,
        "alpha_cache_status": "ALPHA_CACHE_CONTINUITY_ON_BATCH_ENTRY_SNAPSHOT",
        "alpha_cache_entry_widths": [index * BATCH_SIZE for index in range(ROUND_COUNT)],
        "alpha_cache_append_count": ROUND_COUNT,
        "same_batch_current_key_history_inclusion_count": 0,
        "dtype_contract": {
            "status": "FULL_FP32_PASS",
            "parameter_inventory": parameter_inventory,
            "numeric_storage_cast_count": 0,
            "bf16_fp16_path_count": 0,
        },
        "runtime_after_model_preflight_seconds": time.perf_counter() - started,
        "job_compute": job_ledger.raw_free_payload(),
        "final_gpu_host_observation": _gpu_observation(),
        "terminal_W0_restore": restore,
        "W0_restored": True,
        "technical_failure_count": 0,
        "scientific_failure_count": 0,
        "imputation_count": 0,
        "scientific_promotion": False,
    }
    if experiment_binding is not None:
        if set(terminal).intersection(experiment_binding.metadata):
            raise ODEBFContractError("Phase3 external terminal metadata collides")
        terminal.update(
            {
                "target_field_evaluation_count": len(batch_rows)
                * P1R23_GRID_COUNT,
                "writer_layer_apply_count": len(batch_rows)
                * experiment_binding.writer_cadence.writer_calls_per_block
                * 5,
                "batch_commit_to_next_entry_chain": True,
                "sequential_chain_receipt": sequential_chain,
                "heldout_outer_indices_per_batch": [8],
                "heldout_evaluator_decision_influence_count": 0,
                "forbidden_decision_access_count": 0,
                "additional_model_forward_count": 0,
                "additional_backward_count": 0,
                **dict(experiment_binding.metadata),
            }
        )
    _assert_no_low_precision_activity(terminal)
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": (
            "ode-edit-s05-p1r52-c-writer-phase3-manifest/v1"
            if experiment_binding is None
            else experiment_binding.manifest_schema
        ),
        "source_head": source_head,
        "role": role,
        "arm": arm,
        "terminal_sha256": terminal_sha,
        "batch_terminal_sha256": batch_shas,
        "final_w10_sha256": final_sha,
        "W0_restored": True,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": (
            "P1R52_C_WRITER_PHASE3_TERMINAL"
            if experiment_binding is None
            else experiment_binding.return_status
        ),
        "arm": arm,
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "W0_restored": True,
    }


__all__ = [
    "CACHE_ARMS",
    "Phase3SequentialExperimentBinding",
    "RESULT_NAMES",
    "ROLES",
    "arm_for_role",
    "expected_result_name",
    "is_phase3_role",
    "role_for_cell",
    "run_phase3",
    "validate_phase3_batch_chain",
]
