"""User-authorized one-B100 canonical Native references for target-timescale.

This module deliberately reuses the already sealed Official AlphaEdit/MEMIT
apply, accepted-z capture, evaluator, and W0 restore implementation.  It owns
only the B1-only execution boundary and create-once terminal receipts.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .contracts import COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .p1_runtime import _atomic_write_once
from .p1_scalable_batched_experiment import _model_w0_contract
from .p1_state import ArmWeightSnapshot
from .p1r36_independent_b10x10_runtime import _hashes
from .p1r52_b100x10_stream import BATCH_SIZE, ROUND_COUNT
from .p1r52_joint_pc_independent_fp32_runtime import (
    _assert_no_low_precision_activity,
    _assert_w0,
    _gpu_observation,
    _raw_free_json_tree,
    _run_native_method,
    full_fp32_parameter_inventory,
)
from .p1r52_joint_pc_runtime import STREAM_ORDER, STREAM_ROOT
from .scalable_batched_runtime import scalable_ordered_request_digest


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-TARGET-TIMESCALE-NATIVE-B100-USER-OVERRIDE-V1"
PARENT_INSTRUCTION_ID = "ODEEDIT-S05-P1R52-TARGET-TIMESCALE-ABLATION-B100-V1"
POLICY_PROVENANCE = "USER_DIRECTED_NATIVE_REFERENCE_RUN_OVERRIDE"
METHODS = ("OFFICIAL-ALPHAEDIT", "OFFICIAL-MEMIT")
ROLE_PREFIX = "r52-target-timescale-native-b100-user-override-"
ROLES = tuple(f"{ROLE_PREFIX}{method.lower()}" for method in METHODS)
RESULT_NAMES = {
    role: (
        "s05-p1r52-target-timescale-native-b100-"
        f"{method.lower()}-user-override-v1"
    )
    for role, method in zip(ROLES, METHODS, strict=True)
}
RESULT_NAMES[ROLES[1]] = (
    "s05-p1r52-target-timescale-native-b100-"
    "official-memit-user-override-tech-r2-v1"
)


def role_for_cell(cell: int) -> str:
    if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < len(ROLES):
        raise ODEBFContractError("target-timescale Native array cell differs")
    return ROLES[cell]


def method_for_role(role: str) -> str:
    if role not in ROLES:
        raise ODEBFContractError("target-timescale Native role differs")
    return METHODS[ROLES.index(role)]


def expected_result_name(role: str) -> str:
    try:
        return RESULT_NAMES[role]
    except KeyError as exc:
        raise ODEBFContractError("target-timescale Native result role differs") from exc


def is_target_timescale_native_reference_role(role: str | None) -> bool:
    return role in ROLES


def run_target_timescale_native_reference_b100(
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
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    job_ledger: ComputeLedger,
    fp32_runtime: Any | None = None,
    **_: Any,
) -> dict[str, Any]:
    method = method_for_role(role)
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
        raise ODEBFContractError("target-timescale Native stream/FP32 boundary differs")

    requests = tuple(stream_batches[0])
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    parameter_inventory = full_fp32_parameter_inventory(model)
    entry_contract = _model_w0_contract(touched)
    entry_hashes = _hashes(touched)
    entry_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    if entry_hashes != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("target-timescale Native W0 differs")

    seed_all(COMMON_SEED)
    started = time.perf_counter()
    native = _run_native_method(
        model,
        tokenizer,
        requests,
        case_index=1,
        kind=method,
        dataset_path=dataset_path,
        hparams=hparams,
        touched=touched,
        base_values=base_values,
        mutation_lock=mutation_lock,
        entry_contract=entry_contract,
        job_ledger=job_ledger,
        memit_easyedit_root=Path("/data/janghj/EasyEdit"),
    )
    _assert_w0(
        touched,
        expected_contract=entry_contract,
        expected_hashes=entry_hashes,
        expected_pointers=entry_pointers,
    )
    if (
        native.get("method") != method
        or native.get("request_order_sha256") != request_order
        or native.get("module_state_restored") is not True
        or native.get("model_storage_dtype") != "torch.float32"
        or native.get("numeric_storage_cast_count") != 0
        or native.get("autocast_count") != 0
        or native.get("bf16_path_call_count") != 0
        or native.get("fp16_conversion_count") != 0
    ):
        raise ODEBFStateError("target-timescale Native terminal invariant differs")

    terminal: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-target-timescale-native-b100-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "parent_instruction_id": PARENT_INSTRUCTION_ID,
        "policy_provenance": POLICY_PROVENANCE,
        "status": "TERMINAL_VALID",
        "source_head": source_head,
        "role": role,
        "method": method,
        "model_alias": alias,
        "selected_batch": "B1",
        "request_count": len(requests),
        "request_order_sha256": request_order,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "sample_duplication_count": 0,
        "native": native,
        "native_execution_count": 1,
        "official_apply_count": 1,
        "target_timescale_field_execution_count": 0,
        "C3_K8_writer_execution_count": 0,
        "reference_only": True,
        "selection_influence_count": 0,
        "promotion_influence_count": 0,
        "dtype_contract": {
            "status": "FULL_FP32_PASS",
            "parameter_inventory": parameter_inventory,
            "bf16_fp16_path_count": 0,
            "autocast_count": 0,
        },
        "job_compute": job_ledger.raw_free_payload(),
        "runtime_after_model_preflight_seconds": time.perf_counter() - started,
        "gpu_host_observation": _gpu_observation(),
        "W0_restored": True,
        "scientific_promotion": False,
        "technical_attempt": "TECH-R2" if method == "OFFICIAL-MEMIT" else "R1",
    }
    _assert_no_low_precision_activity(terminal)
    terminal, tensor_paths = _raw_free_json_tree(terminal)
    terminal["raw_free_tensor_paths"] = list(tensor_paths)
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r52-target-timescale-native-b100-manifest/v1",
        "instruction_id": INSTRUCTION_ID,
        "parent_instruction_id": PARENT_INSTRUCTION_ID,
        "policy_provenance": POLICY_PROVENANCE,
        "source_head": source_head,
        "role": role,
        "method": method,
        "terminal_sha256": terminal_sha,
        "request_order_sha256": request_order,
        "stream_root": STREAM_ROOT,
        "W0_restored": True,
        "reference_only": True,
        "selection_influence_count": 0,
        "scientific_promotion": False,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    stages.record(
        f"target_timescale_native_{method.lower()}_terminal",
        {
            "native_execution_count": 1,
            "official_apply_count": 1,
            "W0_restored": True,
            "reference_only": True,
        },
    )
    return {
        "status": "P1R52_TARGET_TIMESCALE_NATIVE_REFERENCE_TERMINAL",
        "method": method,
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "W0_restored": True,
    }


__all__ = [
    "INSTRUCTION_ID",
    "METHODS",
    "POLICY_PROVENANCE",
    "RESULT_NAMES",
    "ROLES",
    "expected_result_name",
    "is_target_timescale_native_reference_role",
    "method_for_role",
    "role_for_cell",
    "run_target_timescale_native_reference_b100",
]
