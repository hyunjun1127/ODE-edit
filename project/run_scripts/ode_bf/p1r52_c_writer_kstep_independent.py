"""Phase-2 independent B100 P1R52 K-step C-writer runtime."""

from __future__ import annotations

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
from .p1r52_c_writer_kstep import ARMS, CKStepWriterRuntime
from .p1r52_joint_pc_fp32_runtime import _fp32_target_and_j0, _restore
from .p1r52_joint_pc_independent_fp32_runtime import (
    _assert_no_low_precision_activity,
    _gpu_observation,
    _raw_free_json_tree,
    full_fp32_parameter_inventory,
)
from .p1r52_joint_pc_runtime import STREAM_ORDER, STREAM_ROOT
from .p1r52_target_official_alphaedit_writer import isolated_alphaedit_module_state
from .scalable_batched_runtime import P1R23_GRID_COUNT, scalable_ordered_request_digest


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-C-WRITER-THREE-PHASE-V1"
METHOD_ID = "P1R52-C-WRITER-PHASE2-INDEPENDENT-KSTEP-FULL-FP32"
ROLE_PREFIX = "r52-c-writer-phase2-independent-full-fp32-"
ROLES = tuple(f"{ROLE_PREFIX}{arm.lower()}" for arm in ARMS)
RESULT_NAMES = {
    role: f"s05-p1r52-c-writer-phase2-independent-full-fp32-{arm.lower()}-10xb100-v1"
    for role, arm in zip(ROLES, ARMS, strict=True)
}


def role_for_cell(cell: int) -> str:
    if isinstance(cell, bool) or not 0 <= cell < len(ROLES):
        raise ODEBFContractError("Phase2 cell differs")
    return ROLES[cell]


def arm_for_role(role: str) -> str:
    if role not in ROLES:
        raise ODEBFContractError("Phase2 role differs")
    return ARMS[ROLES.index(role)]


def expected_result_name(role: str) -> str:
    try:
        return RESULT_NAMES[role]
    except KeyError as exc:
        raise ODEBFContractError("Phase2 result role differs") from exc


def is_phase2_role(role: str | None) -> bool:
    return role in ROLES


def run_phase2(
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
    **_: Any,
) -> dict[str, Any]:
    arm = arm_for_role(role)
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
        raise ODEBFContractError("Phase2 matrix/stream/FP32 boundary differs")
    parameter_inventory = full_fp32_parameter_inventory(model)
    w0_contract = _model_w0_contract(touched)
    w0_hashes = _hashes(touched)
    w0_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    if w0_hashes != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("Phase2 W0 differs")
    rows: list[dict[str, Any]] = []
    shas: list[str] = []
    started = time.perf_counter()
    for case_index, request_batch in enumerate(stream_batches, start=1):
        requests = tuple(request_batch)
        seed_all(COMMON_SEED)
        if _hashes(touched) != w0_hashes or any(int(touched[name].data_ptr()) != w0_pointers[name] for name in touched):
            raise ODEBFStateError("Phase2 case did not enter exact W0")
        case_root = raw_root / "cases" / f"case-{case_index:02d}"
        case_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        entry_receipt, entry_values = snapshot_touched_weights(P1Arm.R_BF, 0, touched)
        created: list[CKStepWriterRuntime] = []

        def factory(objective_plan: Any, capture_plan: Any) -> CKStepWriterRuntime:
            runtime = CKStepWriterRuntime(
                arm=arm, model=model, tokenizer=tokenizer, requests=requests,
                hparams=hparams, projector=projector, contexts=contexts,
                covariance_registry=covariance_registry,
                projector_sha256=projector_sha256,
                controller_lock=controller_lock, objective_plan=objective_plan,
                capture_plan=capture_plan, dataset_path=dataset_path,
                private_root=case_root / "private", job_ledger=job_ledger,
                solve_history_keys_by_layer=None,
                alpha_cache_status="ALPHA_CACHE_OFF_CONTROL",
            )
            created.append(runtime)
            return runtime

        case_started = time.perf_counter()
        try:
            with isolated_alphaedit_module_state() as alpha_state:
                target, public, _, _ = _fp32_target_and_j0(
                    model, tokenizer, requests, case_root=case_root,
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
                    c_kstep_writer_factory=factory,
                )
            if not alpha_state["restored"] or len(created) != 1:
                raise ODEBFStateError("Phase2 writer runtime/module scope differs")
            runtime = created[0]
            runtime.assert_complete()
            if len(runtime.executions) != P1R23_GRID_COUNT:
                raise ODEBFStateError("Phase2 K8 execution count differs")
            commit_hashes = _hashes(touched)
            if commit_hashes == w0_hashes:
                raise ODEBFStateError("Phase2 K-step endpoint equals W0")
            case = {
                "schema": "ode-edit-s05-p1r52-c-writer-phase2-case/v1",
                "case_index": case_index,
                "arm": arm,
                "request_count": BATCH_SIZE,
                "request_order_sha256": scalable_ordered_request_digest([str(item["request_sha256"]) for item in requests]),
                "entry_W0_parameter_sha256": w0_hashes,
                "terminal_target_sha256": public["terminal_target_sha256"],
                "terminal_target_tensor_sha256": tensor_sha256(target),
                "commit_weight_sha256": commit_hashes,
                "kstep_writer": runtime.raw_free_payload(),
                "kstep_executions": [item.raw_free_payload() for item in runtime.executions],
                "target_public": public,
                "alpha_cache_status": "ALPHA_CACHE_OFF_CONTROL",
                "cross_case_W_cache_history_carry_count": 0,
                "dtype_contract": {"status": "FULL_FP32_PASS", "numeric_storage_cast_count": 0, "bf16_fp16_path_count": 0},
                "case_total_seconds": time.perf_counter() - case_started,
                "retry_count": 0,
                "imputation_count": 0,
            }
            _assert_no_low_precision_activity(case)
            case, tensor_paths = _raw_free_json_tree(case)
            case["raw_free_tensor_paths"] = list(tensor_paths)
            case["identity_sha256"] = canonical_hash(case)
            restore = _restore(touched, base_values, mutation_lock=mutation_lock, entry_contract=w0_contract)
            if _hashes(touched) != w0_hashes or any(int(touched[name].data_ptr()) != w0_pointers[name] for name in touched):
                raise ODEBFStateError("Phase2 case W0 restore differs")
            # Publish a scientific endpoint only after its exact W0 restore gate.
            # The payload is already fully constructed and hashed, so publication
            # cannot expose a terminal receipt for an un-restored case.
            sha = _atomic_write_once(case_root / "terminal.json", case)
        except BaseException:
            _restore(touched, base_values, mutation_lock=mutation_lock, entry_contract=w0_contract)
            raise
        shas.append(sha)
        rows.append({
            "case_index": case_index,
            "request_order_sha256": case["request_order_sha256"],
            "terminal_sha256": sha,
            "W0_restore": restore,
            "K_count": P1R23_GRID_COUNT,
        })
        stages.record(f"phase2_{arm.lower()}_case_{case_index:02d}", {"completed_case_count": len(rows), "W0_restored": True, "K_count": 8})
    terminal = {
        "schema": "ode-edit-s05-p1r52-c-writer-phase2-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "status": "TERMINAL_VALID",
        "source_head": source_head,
        "role": role,
        "arm": arm,
        "case_count": ROUND_COUNT,
        "valid_request_count": ROUND_COUNT * BATCH_SIZE,
        "K_writer_call_count": ROUND_COUNT * P1R23_GRID_COUNT,
        "cases": rows,
        "case_terminal_sha256": shas,
        "alpha_cache_status": "ALPHA_CACHE_OFF_CONTROL",
        "cross_case_state_count": 0,
        "dtype_contract": {"status": "FULL_FP32_PASS", "parameter_inventory": parameter_inventory, "numeric_storage_cast_count": 0, "bf16_fp16_path_count": 0},
        "runtime_after_model_preflight_seconds": time.perf_counter() - started,
        "job_compute": job_ledger.raw_free_payload(),
        "final_gpu_host_observation": _gpu_observation(),
        "W0_restored": True,
        "technical_failure_count": 0,
        "scientific_failure_count": 0,
        "imputation_count": 0,
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {"schema": "ode-edit-s05-p1r52-c-writer-phase2-manifest/v1", "source_head": source_head, "role": role, "terminal_sha256": terminal_sha, "case_terminal_sha256": shas, "W0_restored": True}
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {"status": "P1R52_C_WRITER_PHASE2_TERMINAL", "arm": arm, "terminal_sha256": terminal_sha, "manifest_sha256": manifest_sha, "W0_restored": True}


__all__ = ["ARMS", "RESULT_NAMES", "ROLES", "arm_for_role", "expected_result_name", "is_phase2_role", "role_for_cell", "run_phase2"]
