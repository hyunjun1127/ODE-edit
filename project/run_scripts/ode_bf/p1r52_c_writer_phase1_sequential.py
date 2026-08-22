"""Phase-1 full-FP32 sequential C-writer matrix.

Baseline cells delegate to the pinned EasyEdit entrypoints through the existing
sequential runtime.  C cells reuse the released P1R52 target and writer
implementations; this module owns only B1--B10 orchestration and cache policy.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .contracts import COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_evaluator import EndpointActionFreeze, load_counterfact_cases_after_freeze
from .p1_runtime import _atomic_write_once
from .p1_scalable_batched_experiment import _model_w0_contract
from .p1_state import ArmWeightSnapshot, P1Arm, snapshot_touched_weights
from .p1r36_independent_b10x10_runtime import _hashes
from .p1r52_accepted_z_observation import evaluate_accepted_z_batch, r52_binding
from .p1r52_b100x10_stream import BATCH_SIZE, ROUND_COUNT
from .p1r52_joint_pc_execution import JointPCWriterArm
from .p1r52_joint_pc_fp32_runtime import _fp32_target_and_j0, _restore, _run_pc_arm_fp32
from .p1r52_joint_pc_independent_fp32_runtime import (
    _assert_no_low_precision_activity,
    _gpu_observation,
    _ledger_delta,
    _raw_free_json_tree,
    _sync,
    _timed,
    _timing_policy,
    full_fp32_parameter_inventory,
)
from .p1r52_joint_pc_runtime import STREAM_ORDER, STREAM_ROOT, _writer_entry
from .p1r52_residual_reserve_phase_a_execution import fp32_weight_energy
from .p1r52_target_official_alphaedit_writer import (
    _endpoint_summary,
    _evaluate_w,
    _writer_gap,
    accepted_z_cache_template,
    isolated_alphaedit_module_state,
)
from .scalable_batched_model import capture_scalable_physical_state
from .scalable_batched_native import run_official_native_apply
from .scalable_batched_runtime import P1R23_GRID_COUNT, P1R23_LAYER_ORDER, scalable_ordered_request_digest


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-C-WRITER-THREE-PHASE-V1"
METHOD_ID = "P1R52-C-WRITER-PHASE1-FULL-FP32-ALPHA-CACHE-SEQUENTIAL"
CELL_LABELS = ("OFFICIAL-ALPHAEDIT", "OFFICIAL-MEMIT", "C0", "C1", "C3")
ROLE_PREFIX = "r52-c-writer-phase1-full-fp32-sequential-"
ROLES = tuple(f"{ROLE_PREFIX}{label.lower()}" for label in CELL_LABELS)
RESULT_NAMES = {
    role: f"s05-p1r52-c-writer-phase1-full-fp32-sequential-{label.lower()}-10xb100-tech-r1-v1"
    for role, label in zip(ROLES, CELL_LABELS, strict=True)
}
ALPHA_APPLICABLE = frozenset(("OFFICIAL-ALPHAEDIT", "C0", "C1", "C3"))


def role_for_cell(cell_index: int) -> str:
    if isinstance(cell_index, bool) or not 0 <= cell_index < len(ROLES):
        raise ODEBFContractError("Phase1 cell index differs")
    return ROLES[cell_index]


def label_for_role(role: str) -> str:
    if role not in ROLES:
        raise ODEBFContractError("Phase1 role differs")
    return CELL_LABELS[ROLES.index(role)]


def expected_result_name(role: str) -> str:
    try:
        return RESULT_NAMES[role]
    except KeyError as exc:
        raise ODEBFContractError("Phase1 result role differs") from exc


def is_phase1_role(role: str | None) -> bool:
    return role in ROLES


def _entry_values(touched: Mapping[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in touched.items()}


def _cache_snapshot(alpha_main: Any) -> dict[str, Any]:
    return {
        "cache_c_present": hasattr(alpha_main, "cache_c"),
        "cache_c": (
            getattr(alpha_main, "cache_c").detach().clone()
            if isinstance(getattr(alpha_main, "cache_c", None), torch.Tensor)
            else copy.deepcopy(getattr(alpha_main, "cache_c", None))
        ),
        "cache_c_new_present": hasattr(alpha_main, "cache_c_new"),
        "cache_c_new": copy.deepcopy(getattr(alpha_main, "cache_c_new", None)),
    }


def _restore_cache(alpha_main: Any, snapshot: Mapping[str, Any]) -> None:
    for name in ("cache_c", "cache_c_new"):
        present = bool(snapshot[f"{name}_present"])
        if present:
            value = snapshot[name]
            setattr(alpha_main, name, value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value))
        elif hasattr(alpha_main, name):
            delattr(alpha_main, name)


def _freeze(role: str, round_index: int, order: str, identity: str) -> EndpointActionFreeze:
    return EndpointActionFreeze(
        arm=f"{role}-b{round_index:02d}",
        sequential_batch=round_index - 1,
        request_order_sha256=order,
        selected_snapshot_sha256=identity,
        fixed_budget_slots_completed=P1R23_GRID_COUNT,
    )


def _run_c_sequential(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    role: str,
    cell: str,
    destination: Path,
    raw_root: Path,
    stages: Any,
    source_head: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    collision_by_request: Mapping[str, str],
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
) -> dict[str, Any]:
    del collision_by_request
    if cell not in ("C0", "C1", "C3"):
        raise ODEBFContractError("Phase1 C arm differs")
    if alias != "llama3-8b-inst":
        raise ODEBFContractError("Phase1 C arm model differs")
    w0_contract = _model_w0_contract(touched)
    w0_hashes = _hashes(touched)
    w0_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    if w0_hashes != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("Phase1 C arm W0 differs")
    history: dict[int, torch.Tensor] | None = None
    history_version = 0
    prior_cache_exit: str | None = None
    cases_for_final: list[tuple[Any, ...]] = []
    batch_rows: list[dict[str, Any]] = []
    batch_shas: list[str] = []
    runtime_started = time.perf_counter()
    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    alpha_scope = isolated_alphaedit_module_state()
    try:
        with alpha_scope as alpha_state:
            for round_index, request_batch in enumerate(stream_batches, start=1):
                requests = tuple(request_batch)
                seed_all(COMMON_SEED)
                batch_root = raw_root / "batches" / f"b{round_index:02d}"
                batch_root.mkdir(mode=0o700, parents=True, exist_ok=False)
                entry_contract = _model_w0_contract(touched)
                entry_hashes = _hashes(touched)
                entry_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
                entry_values = _entry_values(touched)
                entry_receipt, _ = snapshot_touched_weights(P1Arm.R_BF, round_index - 1, touched)
                cache_checkpoint = _cache_snapshot(alpha_main)
                history_checkpoint = None if history is None else {layer: value.clone() for layer, value in history.items()}
                before = job_ledger.raw_free_payload()
                batch_started = time.perf_counter()
                try:
                    (target_result, target_wall) = _timed(lambda: _fp32_target_and_j0(
                        model, tokenizer, requests, case_root=batch_root,
                        hparams=hparams, projector=projector, contexts=contexts,
                        covariance_registry=covariance_registry,
                        projector_sha256=projector_sha256,
                        controller_lock=controller_lock,
                        request_by_sha256=request_by_sha256,
                        population_by_sha256=population_by_sha256,
                        schedule=schedule, theta0_cache=theta0_cache,
                        touched=touched, base_receipt=entry_receipt,
                        base_values=entry_values,
                        request_microbatch_size=request_microbatch_size,
                        job_ledger=job_ledger,
                    ))
                    target, target_public, objective_plan, capture_plan = target_result
                    _assert_no_low_precision_activity(target_public)
                    target_hash = tensor_sha256(target)
                    order = scalable_ordered_request_digest([str(item["request_sha256"]) for item in requests])
                    freeze = _freeze(role, round_index, order, target_public["identity_sha256"])
                    loaded_cases = tuple(load_counterfact_cases_after_freeze(
                        dataset_path, requests, freeze, expected_batch_size=BATCH_SIZE
                    ))
                    cases_for_final.append(loaded_cases)
                    (target_restore, target_restore_wall) = _timed(lambda: _restore(
                        touched, entry_values, mutation_lock=mutation_lock,
                        entry_contract=entry_contract,
                    ))
                    if _hashes(touched) != entry_hashes or any(
                        int(touched[name].data_ptr()) != entry_pointers[name] for name in touched
                    ):
                        raise ODEBFStateError("Phase1 target did not restore batch-entry W")
                    binding = r52_binding(role, requests, hparams, target)
                    counter = ModelForwardCounter(model, job_ledger)
                    try:
                        (z_result, z_wall) = _timed(lambda: evaluate_accepted_z_batch(
                            model, tokenizer, requests, loaded_cases, role=role,
                            round_index=round_index, binding=binding,
                            committed_weight_sha256=entry_hashes,
                        ))
                    finally:
                        counter.close()
                    z_observation, _ = z_result
                    z_summary = _endpoint_summary(z_observation["scores"])
                    if cell in ("C0", "C1"):
                        entry = _writer_entry(
                            model, tokenizer, requests, target=target, hparams=hparams,
                            projector=projector, contexts=contexts,
                            covariance_registry=covariance_registry,
                            projector_sha256=projector_sha256,
                            controller_lock=controller_lock,
                            objective_plan=objective_plan, capture_plan=capture_plan,
                        )
                        current_keys = {
                            layer: entry["physical"].keys_by_layer[layer].detach().cpu().float().contiguous()
                            for layer in P1R23_LAYER_ORDER
                        }
                        if history is None:
                            history = {
                                layer: torch.empty((current_keys[layer].shape[0], 0), dtype=torch.float32)
                                for layer in P1R23_LAYER_ORDER
                            }
                        expected_width = (round_index - 1) * BATCH_SIZE
                        if any(value.shape[1] != expected_width for value in history.values()):
                            raise ODEBFStateError("Phase1 committed Alpha cache width differs")
                        arm = JointPCWriterArm.C0 if cell == "C0" else JointPCWriterArm.C1
                        (writer, writer_wall) = _timed(lambda: _run_pc_arm_fp32(
                            model, arm, target=target, entry=entry, hparams=hparams,
                            projector=projector, covariance_registry=covariance_registry,
                            projector_sha256=projector_sha256,
                            controller_lock=controller_lock, capture_plan=capture_plan,
                            solve_history_keys_by_layer=history,
                        ))
                        cache_receipt = {
                            "alpha_cache_status": "ALPHA_CACHE_CONTINUITY_ON",
                            "cache_kind": "COMMITTED_PAST_ALPHA_SOLVE_KEYS",
                            "entry_width": expected_width,
                            "consume_width": expected_width,
                            "append_width": BATCH_SIZE,
                            "exit_width": round_index * BATCH_SIZE,
                            "ledger_version_at_entry": history_version,
                            "current_batch_key_inclusion_count": 0,
                            "current_uncommitted_prefix_key_inclusion_count": 0,
                            "entry_sha256": {str(layer): tensor_sha256(history[layer]) for layer in P1R23_LAYER_ORDER},
                            "append_sha256": {str(layer): tensor_sha256(current_keys[layer]) for layer in P1R23_LAYER_ORDER},
                        }
                        writer["entry_control_pi"] = list(entry["control"].pi)
                        writer["entry_joint_pi"] = list(entry["joint"].pi)
                        writer["entry_joint_route"] = entry["joint"].receipt.raw_free_payload()
                    else:
                        with accepted_z_cache_template(
                            requests, target, hparams, parent=batch_root / "private"
                        ) as (cache_template, bridge):
                            counter = ModelForwardCounter(model, job_ledger)
                            try:
                                (native_result, writer_wall) = _timed(lambda: run_official_native_apply(
                                    model, tokenizer, requests, hparams, touched=touched,
                                    reset_cache=round_index == 1,
                                    cache_history_width=(round_index - 1) * BATCH_SIZE,
                                    cache_template=cache_template,
                                    expected_native_compute_z_call_count=0,
                                    accepted_z_source="P1R52_K8_TERMINAL_TARGET",
                                ))
                            finally:
                                counter.close()
                        apply_payload, originals = native_result
                        if any(tensor_sha256(originals[name]) != entry_hashes[name] for name in touched):
                            raise ODEBFStateError("Phase1 C3 entry W copy differs")
                        dynamic = apply_payload["alphaedit_dynamic_cache_contract"]
                        if (
                            dynamic["logical_history_width_at_entry"] != (round_index - 1) * BATCH_SIZE
                            or dynamic["logical_history_width_after_append"] != round_index * BATCH_SIZE
                            or (round_index > 1 and dynamic["entry"]["sha256"] != prior_cache_exit)
                            or (round_index > 1 and not dynamic["solver_consumed_entry_cache"])
                        ):
                            raise ODEBFStateError("Phase1 C3 Alpha cache continuity differs")
                        prior_cache_exit = dynamic["exit"]["sha256"]
                        cache_receipt = {
                            "alpha_cache_status": "ALPHA_CACHE_CONTINUITY_ON",
                            "cache_kind": "OFFICIAL_ALPHAEDIT_DYNAMIC_CACHE_C",
                            "entry_width": (round_index - 1) * BATCH_SIZE,
                            "consume_width": (round_index - 1) * BATCH_SIZE,
                            "append_width": BATCH_SIZE,
                            "exit_width": round_index * BATCH_SIZE,
                            "current_batch_key_inclusion_count": 0,
                            "current_uncommitted_prefix_key_inclusion_count": 0,
                            "official_contract": dynamic,
                        }
                        writer = {
                            "method_label": "P1R52_K8_TARGET_PLUS_DIRECT_OFFICIAL_ALPHAEDIT_ACCEPTED_Z_ADAPTER",
                            "official_call_path": apply_payload["official_entrypoint"],
                            "apply": apply_payload,
                            "accepted_z_bridge": bridge,
                            "native_alphaedit_compute_z_call_count": 0,
                        }
                    _assert_no_low_precision_activity(writer)
                    (w_scores, evaluator_wall) = _timed(lambda: _evaluate_w(
                        model, tokenizer, loaded_cases, freeze=freeze, ledger=job_ledger
                    ))
                    w_summary = _endpoint_summary(w_scores)
                    update_energy = fp32_weight_energy(touched, entry_values)
                    committed_hashes = _hashes(touched)
                    if committed_hashes == entry_hashes:
                        raise ODEBFStateError("Phase1 writer produced no W transition")
                    if cell in ("C0", "C1"):
                        assert history is not None
                        history = {
                            layer: torch.cat((history[layer], current_keys[layer]), dim=1).contiguous()
                            for layer in P1R23_LAYER_ORDER
                        }
                        history_version += 1
                        cache_receipt["ledger_version_after_commit"] = history_version
                        cache_receipt["append_count"] = 1
                    else:
                        cache_receipt["append_count"] = 1
                    cache_receipt["identity_sha256"] = canonical_hash(cache_receipt)
                    payload: dict[str, Any] = {
                        "schema": "ode-edit-s05-p1r52-c-writer-phase1-batch/v1",
                        "instruction_id": INSTRUCTION_ID,
                        "method_id": METHOD_ID,
                        "role": role,
                        "cell": cell,
                        "round": round_index,
                        "request_count": BATCH_SIZE,
                        "request_order_sha256": order,
                        "entry_weight_sha256": entry_hashes,
                        "commit_weight_sha256": committed_hashes,
                        "accepted_z_sha256": target_hash,
                        "selected_target_receipt_identity": target_public["identity_sha256"],
                        "accepted_z": binding.raw_free_payload(),
                        "z": {"summary": z_summary, "scores": z_observation["scores"]},
                        "W": {"summary": w_summary, "scores": w_scores},
                        "gap": _writer_gap(w_summary, z_summary),
                        "writer": writer,
                        "update_energy": update_energy,
                        "alpha_cache": cache_receipt,
                        "timing": {
                            "target_accepted_z_generation_seconds": target_wall,
                            "writer_edit_core_seconds": writer_wall,
                            "target_plus_writer_seconds": target_wall + writer_wall,
                            "accepted_z_evaluator_seconds": z_wall,
                            "immediate_post_evaluator_seconds": evaluator_wall,
                            "target_J0_restore_seconds": target_restore_wall,
                            "case_total_seconds": time.perf_counter() - batch_started,
                            "policy": _timing_policy(),
                        },
                        "compute_delta": _ledger_delta(before, job_ledger.raw_free_payload()),
                        "dtype_contract": {
                            "status": "FULL_FP32_PASS",
                            "model_storage": "torch.float32",
                            "algorithm_tensors": "torch.float32",
                            "autocast_count": 0,
                            "bf16_fp16_conversion_count": 0,
                            "numeric_storage_cast_count": 0,
                            "bf16_candidate_materializer_call_count": 0,
                        },
                        "commit_count": 1,
                        "rollback_count": 0,
                        "retry_count": 0,
                        "imputation_count": 0,
                        "physical_W_persists_to_next_batch": round_index < ROUND_COUNT,
                    }
                    payload, tensor_paths = _raw_free_json_tree(payload)
                    payload["raw_free_tensor_paths"] = list(tensor_paths)
                    payload["identity_sha256"] = canonical_hash(payload)
                    terminal_sha = _atomic_write_once(batch_root / "terminal.json", payload)
                    batch_shas.append(terminal_sha)
                    batch_rows.append({
                        "round": round_index,
                        "request_order_sha256": order,
                        "accepted_z_sha256": target_hash,
                        "entry_weight_sha256": entry_hashes,
                        "commit_weight_sha256": committed_hashes,
                        "terminal_sha256": terminal_sha,
                    })
                    stages.record(f"phase1_{cell.lower()}_b{round_index:02d}", {
                        "round": round_index,
                        "completed_batch_count": len(batch_rows),
                        "alpha_cache_entry_width": (round_index - 1) * BATCH_SIZE,
                        "alpha_cache_exit_width": round_index * BATCH_SIZE,
                        "full_fp32": True,
                    })
                except BaseException:
                    _restore(touched, entry_values, mutation_lock=mutation_lock, entry_contract=entry_contract)
                    _restore_cache(alpha_main, cache_checkpoint)
                    history = history_checkpoint
                    if _hashes(touched) != entry_hashes:
                        raise ODEBFStateError("Phase1 batch rollback W differs")
                    raise

            final_hashes = _hashes(touched)
            final_rows: list[dict[str, Any]] = []
            for round_index, loaded_cases in enumerate(cases_for_final, start=1):
                freeze = EndpointActionFreeze(
                    arm=f"{role}-final-b{round_index:02d}",
                    sequential_batch=ROUND_COUNT - 1,
                    request_order_sha256=batch_rows[round_index - 1]["request_order_sha256"],
                    selected_snapshot_sha256=canonical_hash(final_hashes),
                    fixed_budget_slots_completed=P1R23_GRID_COUNT,
                )
                scores = _evaluate_w(model, tokenizer, loaded_cases, freeze=freeze, ledger=job_ledger)
                final_rows.append({"round": round_index, "summary": _endpoint_summary(scores), "scores": scores})
            final_payload = {
                "schema": "ode-edit-s05-p1r52-c-writer-phase1-final-w10/v1",
                "cell": cell,
                "cohort_count": len(final_rows),
                "request_count": len(final_rows) * BATCH_SIZE,
                "final_weight_sha256": final_hashes,
                "cohorts": final_rows,
            }
            final_payload["identity_sha256"] = canonical_hash(final_payload)
            final_sha = _atomic_write_once(raw_root / "final-w10.json", final_payload)
        if not alpha_state["restored"]:
            raise ODEBFStateError("Phase1 Alpha module state restore differs")
    except BaseException:
        _restore(touched, base_values, mutation_lock=mutation_lock, entry_contract=w0_contract)
        raise

    restore = _restore(touched, base_values, mutation_lock=mutation_lock, entry_contract=w0_contract)
    if _hashes(touched) != w0_hashes or any(int(touched[name].data_ptr()) != w0_pointers[name] for name in touched):
        raise ODEBFStateError("Phase1 terminal W0 restore differs")
    terminal = {
        "schema": "ode-edit-s05-p1r52-c-writer-phase1-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "status": "TERMINAL_VALID",
        "source_head": source_head,
        "role": role,
        "cell": cell,
        "completed_batch_count": len(batch_rows),
        "valid_request_count": len(batch_rows) * BATCH_SIZE,
        "batch_terminal_sha256": batch_shas,
        "final_w10_sha256": final_sha,
        "alpha_cache_status": "ALPHA_CACHE_CONTINUITY_ON",
        "alpha_cache_applicable": True,
        "alpha_cache_entry_widths": [index * BATCH_SIZE for index in range(ROUND_COUNT)],
        "alpha_cache_append_count": ROUND_COUNT,
        "current_batch_history_inclusion_count": 0,
        "dtype_contract": {
            "status": "FULL_FP32_PASS",
            "requested_dtype": "torch.float32",
            "loaded_model_dtype": "torch.float32",
            "parameter_inventory": full_fp32_parameter_inventory(model),
            "algorithm_tensor_dtypes": "torch.float32",
            "autocast_count": 0,
            "bf16_fp16_conversion_count": 0,
            "numeric_storage_cast_count": 0,
        },
        "runtime_after_model_preflight_seconds": time.perf_counter() - runtime_started,
        "job_compute": job_ledger.raw_free_payload(),
        "final_gpu_host_observation": _gpu_observation(),
        "terminal_W0_restore": restore,
        "W0_restored": True,
        "technical_failure_count": 0,
        "scientific_failure_count": 0,
        "imputation_count": 0,
        "scientific_promotion": False,
    }
    _assert_no_low_precision_activity(terminal)
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r52-c-writer-phase1-manifest/v1",
        "source_head": source_head,
        "role": role,
        "cell": cell,
        "terminal_sha256": terminal_sha,
        "batch_terminal_sha256": batch_shas,
        "final_w10_sha256": final_sha,
        "W0_restored": True,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P1R52_C_WRITER_PHASE1_TERMINAL",
        "cell": cell,
        "completed_batch_count": ROUND_COUNT,
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "W0_restored": True,
    }


def run_phase1(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    role: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    stream: Mapping[str, Any],
    fp32_runtime: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    cell = label_for_role(role)
    if (
        len(stream_batches) != ROUND_COUNT
        or any(len(batch) != BATCH_SIZE for batch in stream_batches)
        or stream.get("root_digest") != STREAM_ROOT
        or stream.get("all_request_order_sha256") != STREAM_ORDER
        or fp32_runtime is None
        or torch.is_autocast_enabled()
        or torch.is_autocast_enabled("cpu")
    ):
        raise ODEBFContractError("Phase1 matrix/stream/FP32 boundary differs")
    if cell in ("OFFICIAL-ALPHAEDIT", "OFFICIAL-MEMIT"):
        from .p1r52_sequential_runtime import (
            MEMIT_ROLE,
            NATIVE_CORRECTED_ROLE,
            P1R52_B100X10_SCALE,
            run_p1r52_sequential,
        )
        delegated_role = NATIVE_CORRECTED_ROLE if cell == "OFFICIAL-ALPHAEDIT" else MEMIT_ROLE
        result = run_p1r52_sequential(
            model,
            tokenizer,
            role=delegated_role,
            stream_batches=stream_batches,
            stream=stream,
            scale=P1R52_B100X10_SCALE,
            batch_entry_evaluation_enabled=False,
            accepted_z_observation_enabled=True,
            accepted_z_reference_root=None,
            accepted_z_sealed_w_reuse=False,
            fp32_runtime=fp32_runtime,
            **kwargs,
        )
        envelope = {
            "schema": "ode-edit-s05-p1r52-c-writer-phase1-baseline-boundary/v1",
            "instruction_id": INSTRUCTION_ID,
            "role": role,
            "cell": cell,
            "delegated_role": delegated_role,
            "scientific_call_path": (
                "easyeditor.models.alphaedit.AlphaEdit_main.apply_AlphaEdit_to_model"
                if cell == "OFFICIAL-ALPHAEDIT"
                else "easyeditor.models.memit.memit_main.apply_memit_to_model"
            ),
            "native_compute_z_writer_semantics": True,
            "p1r52_accepted_z_injection_count": 0,
            "c_writer_routing_influence_count": 0,
            "full_fp32": True,
            "alpha_cache_status": (
                "ALPHA_CACHE_CONTINUITY_ON"
                if cell == "OFFICIAL-ALPHAEDIT"
                else "ALPHA_CACHE_NOT_APPLICABLE_NATIVE_MEMIT"
            ),
            "alpha_cache_entry_width": (
                list(range(0, ROUND_COUNT * BATCH_SIZE, BATCH_SIZE))
                if cell == "OFFICIAL-ALPHAEDIT" else "NOT_APPLICABLE"
            ),
            "alpha_cache_consume_width": (
                list(range(0, ROUND_COUNT * BATCH_SIZE, BATCH_SIZE))
                if cell == "OFFICIAL-ALPHAEDIT" else "NOT_APPLICABLE"
            ),
            "alpha_cache_append_width": (
                [BATCH_SIZE] * ROUND_COUNT if cell == "OFFICIAL-ALPHAEDIT" else "NOT_APPLICABLE"
            ),
            "alpha_cache_influence_count": 1 if cell == "OFFICIAL-ALPHAEDIT" else 0,
            "request_history_width": ROUND_COUNT * BATCH_SIZE if cell == "OFFICIAL-ALPHAEDIT" else 0,
            "memit_static_cov_cache_status": (
                "NATIVE_STATIC_COV_CACHE_SEPARATE_FROM_ALPHA_CACHE"
                if cell == "OFFICIAL-MEMIT" else "NOT_APPLICABLE"
            ),
            "cache_applicable_denominator": 4,
            "cache_valid_denominator_claim": "4/4_APPLICABLE_ARMS",
        }
        envelope["identity_sha256"] = canonical_hash(envelope)
        _atomic_write_once(kwargs["destination"] / "phase1-boundary.json", envelope)
        return {**result, "phase1_boundary_sha256": envelope["identity_sha256"], "cell": cell}
    return _run_c_sequential(
        model,
        tokenizer,
        role=role,
        cell=cell,
        stream_batches=stream_batches,
        **kwargs,
    )


__all__ = [
    "ALPHA_APPLICABLE", "CELL_LABELS", "INSTRUCTION_ID", "METHOD_ID",
    "RESULT_NAMES", "ROLES", "expected_result_name", "is_phase1_role",
    "label_for_role", "role_for_cell", "run_phase1",
]
