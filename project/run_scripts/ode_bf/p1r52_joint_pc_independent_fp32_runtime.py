"""Full-FP32 independent ten-B100 Joint-P/C production cells.

The scientific target and writers are imported from the sealed one-B100
implementation.  This module owns only case isolation, cell selection,
observation-only timing, and create-once terminal receipts.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
import resource
import time
from typing import Any, Callable, Mapping, Sequence, TypeVar

import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .contracts import COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_evaluator import EndpointActionFreeze, load_counterfact_cases_after_freeze
from .p1_runtime import _atomic_write_once
from .p1_scalable_batched_experiment import _model_w0_contract
from .p1_state import ArmWeightSnapshot
from .p1r36_independent_b10x10_runtime import _hashes
from .p1r52_accepted_z_observation import (
    OfficialNativeZCapture,
    evaluate_accepted_z_batch,
    r52_binding,
)
from .p1r52_b100x10_stream import BATCH_SIZE, ROUND_COUNT
from .p1r52_joint_pc_execution import JointPCWriterArm
from .p1r52_joint_pc_fp32_runtime import (
    _fp32_target_and_j0,
    _raw_free_json_tree,
    _restore,
    _run_pc_arm_fp32,
)
from .p1r52_joint_pc_runtime import STREAM_ORDER, STREAM_ROOT, _writer_entry
from .p1r52_official_sequential_baselines import (
    load_official_memit_hparams,
    run_official_memit_apply,
)
from .p1r52_residual_reserve_phase_a_execution import fp32_weight_energy
from .p1r52_target_official_alphaedit_writer import (
    _endpoint_summary,
    _evaluate_w,
    _writer_gap,
    accepted_z_cache_template,
    isolated_alphaedit_module_state,
)
from .scalable_batched_native import run_official_native_apply
from .scalable_batched_runtime import P1R23_GRID_COUNT, scalable_ordered_request_digest


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-JOINT-PC-FULL-FP32-INDEPENDENT-10XB100-V1"
METHOD_ID = "P1R52-JOINT-PC-FULL-FP32-INDEPENDENT-B100"
CELL_LABELS = ("BASELINE-GROUP", "C0", "C1", "C2", "C3")
ROLE_PREFIX = "r52-joint-pc-full-fp32-independent-b100x10-"
ROLES = tuple(f"{ROLE_PREFIX}{label.lower()}" for label in CELL_LABELS)
RESULT_NAMES = {
    role: f"s05-p1r52-joint-pc-full-fp32-independent-b100x10-{label.lower()}-v1"
    for role, label in zip(ROLES, CELL_LABELS, strict=True)
}
BASELINE_METHODS = ("OFFICIAL-ALPHAEDIT", "OFFICIAL-MEMIT")
T = TypeVar("T")


def role_for_cell(cell_index: int) -> str:
    if isinstance(cell_index, bool) or cell_index < 0 or cell_index >= len(ROLES):
        raise ODEBFContractError("independent FP32 cell index differs")
    return ROLES[cell_index]


def label_for_role(role: str) -> str:
    if role not in ROLES:
        raise ODEBFContractError("independent FP32 role differs")
    return CELL_LABELS[ROLES.index(role)]


def expected_result_name(role: str) -> str:
    try:
        return RESULT_NAMES[role]
    except KeyError as exc:
        raise ODEBFContractError("independent FP32 result role differs") from exc


def is_joint_pc_independent_fp32_role(role: str | None) -> bool:
    return role in ROLES


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(call: Callable[[], T]) -> tuple[T, float]:
    _sync()
    started = time.perf_counter()
    value = call()
    _sync()
    return value, time.perf_counter() - started


def _timing_policy() -> dict[str, Any]:
    payload = {
        "schema": "ode-edit-observation-only-cuda-wall-timing/v1",
        "start_completion_sync": True,
        "start_sync_excluded_from_interval": True,
        "end_completion_sync": True,
        "end_sync_wait_included_in_interval": True,
        "decision_influence_count": 0,
        "additional_model_forward_count": 0,
        "additional_model_backward_count": 0,
        "additional_evaluator_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


@contextmanager
def _isolated_memit_cache() -> Any:
    from easyeditor.models.memit import memit_main

    prior = memit_main.COV_CACHE
    memit_main.COV_CACHE = {}
    state = {"entry_count": 0, "restored": False}
    try:
        yield state
    finally:
        memit_main.COV_CACHE = prior
        state["restored"] = memit_main.COV_CACHE is prior


def _ledger_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    counters = {
        key: int(after["counters"][key]) - int(before["counters"].get(key, 0))
        for key in after["counters"]
    }
    walls = {
        key: float(after["component_wall_seconds"][key])
        - float(before["component_wall_seconds"].get(key, 0.0))
        for key in after["component_wall_seconds"]
    }
    payload = {
        "counters": counters,
        "component_wall_seconds": walls,
        "additional_timing_model_forward_count": 0,
        "additional_timing_model_backward_count": 0,
        "additional_timing_evaluator_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _gpu_observation() -> dict[str, Any]:
    maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if not torch.cuda.is_available():
        return {
            "available": False,
            "host_maxrss_kib": maxrss,
            "contention_snapshot_status": "CUDA_NOT_AVAILABLE",
        }
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "available": True,
        "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_RECORDED"),
        "logical_device_index": device,
        "device_name": properties.name,
        "device_uuid": str(getattr(properties, "uuid", "NOT_RECORDED")),
        "physical_total_bytes": int(properties.total_memory),
        "free_bytes_at_observation": int(free_bytes),
        "allocator_visible_total_bytes": int(total_bytes),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "host_maxrss_kib": maxrss,
        "contention_snapshot_status": "POINT_IN_TIME_LOCAL_DEVICE_ONLY",
    }


def full_fp32_parameter_inventory(model: torch.nn.Module) -> dict[str, Any]:
    dtype_counts: dict[str, int] = {}
    element_counts: dict[str, int] = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1
        element_counts[key] = element_counts.get(key, 0) + int(parameter.numel())
    parameter_count = sum(dtype_counts.values())
    fp32_count = dtype_counts.get("torch.float32", 0)
    requires_grad_count = sum(int(parameter.requires_grad) for parameter in model.parameters())
    config = getattr(model, "config", None)
    quantization_markers = {
        "is_loaded_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
        "is_loaded_in_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
        "is_quantized": bool(getattr(model, "is_quantized", False)),
        "hf_quantizer_present": getattr(model, "hf_quantizer", None) is not None,
        "config_quantization_present": getattr(config, "quantization_config", None) is not None,
    }
    payload = {
        "schema": "ode-edit-full-fp32-model-parameter-inventory/v1",
        "status": "FULL_FP32_PASS",
        "requested_dtype": "torch.float32",
        "loaded_model_dtype": "torch.float32",
        "parameter_tensor_count": parameter_count,
        "parameter_dtype_counts": dict(sorted(dtype_counts.items())),
        "parameter_element_counts": dict(sorted(element_counts.items())),
        "fp32_parameter_tensor_count": fp32_count,
        "fp16_parameter_tensor_count": dtype_counts.get("torch.float16", 0),
        "bf16_parameter_tensor_count": dtype_counts.get("torch.bfloat16", 0),
        "non_fp32_parameter_tensor_count": parameter_count - fp32_count,
        "requires_grad_parameter_tensor_count": requires_grad_count,
        "quantized_parameter_count": sum(quantization_markers.values()),
        "quantization_markers": quantization_markers,
        "autocast_cuda_enabled": bool(torch.is_autocast_enabled()),
        "autocast_cpu_enabled": bool(torch.is_autocast_enabled("cpu")),
        "bf16_conversion_count": 0,
        "fp16_conversion_count": 0,
        "numeric_storage_cast_count": 0,
    }
    if (
        parameter_count == 0
        or fp32_count != parameter_count
        or requires_grad_count != 0
        or payload["quantized_parameter_count"] != 0
        or payload["autocast_cuda_enabled"]
        or payload["autocast_cpu_enabled"]
    ):
        payload["status"] = "TECHNICAL_INVALID_DTYPE"
    payload["identity_sha256"] = canonical_hash(payload)
    if payload["status"] != "FULL_FP32_PASS":
        raise ODEBFContractError("TECHNICAL_INVALID_DTYPE")
    return payload


def _assert_w0(
    touched: Mapping[str, torch.nn.Parameter],
    *, expected_contract: str,
    expected_hashes: Mapping[str, str],
    expected_pointers: Mapping[str, int],
) -> None:
    if (
        _model_w0_contract(touched) != expected_contract
        or _hashes(touched) != dict(expected_hashes)
        or any(int(touched[name].data_ptr()) != expected_pointers[name] for name in touched)
    ):
        raise ODEBFStateError("independent FP32 W0 entry/restore differs")


def _validate_official_fp32_apply(kind: str, payload: Mapping[str, Any]) -> None:
    if kind in ("OFFICIAL-ALPHAEDIT", "C3"):
        adapter = payload.get("solver_key_dtype_adapter")
        cache = payload.get("alphaedit_dynamic_cache_contract")
        if (
            not isinstance(adapter, Mapping)
            or int(adapter.get("call_count", -1)) <= 0
            or set(adapter.get("input_dtypes", ())) != {"torch.float32"}
            or adapter.get("output_dtype") != "torch.float32"
            or not isinstance(cache, Mapping)
            or cache.get("static_projection", {}).get("dtype") != "torch.float32"
            or cache.get("exit", {}).get("dtype") != "torch.float32"
        ):
            raise ODEBFContractError("TECHNICAL_INVALID_DTYPE")
    elif kind == "OFFICIAL-MEMIT":
        cache = payload.get("covariance_cache")
        if not isinstance(cache, Mapping):
            raise ODEBFContractError("TECHNICAL_INVALID_DTYPE")
        rows = cache.get("exit", {}).get("entries", ())
        if not rows or any(row.get("dtype") != "torch.float32" for row in rows):
            raise ODEBFContractError("TECHNICAL_INVALID_DTYPE")
    else:
        raise ODEBFContractError("independent FP32 official apply kind differs")


def _assert_no_low_precision_activity(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if (
                any(token in lowered for token in ("bf16", "fp16", "autocast", "quantized"))
                and any(token in lowered for token in ("count", "influence", "enabled"))
                and item not in (0, 0.0, False, None)
            ):
                raise ODEBFContractError(f"TECHNICAL_INVALID_DTYPE:{path}.{key}")
            _assert_no_low_precision_activity(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_low_precision_activity(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and value in ("torch.float16", "torch.bfloat16"):
        raise ODEBFContractError(f"TECHNICAL_INVALID_DTYPE:{path}")


def _freeze(
    *, role: str, case_index: int, order: str, action_identity: str
) -> EndpointActionFreeze:
    return EndpointActionFreeze(
        arm=role,
        sequential_batch=case_index - 1,
        request_order_sha256=order,
        selected_snapshot_sha256=action_identity,
        fixed_budget_slots_completed=P1R23_GRID_COUNT,
    )


def _run_native_method(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    case_index: int,
    kind: str,
    dataset_path: Path,
    hparams: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_values: Mapping[str, torch.Tensor],
    mutation_lock: Any,
    entry_contract: str,
    job_ledger: ComputeLedger,
    memit_easyedit_root: Path | None = None,
) -> dict[str, Any]:
    if kind == "OFFICIAL-ALPHAEDIT":
        native_role = "native-alphaedit-sequential-cache-on-corrected"
        native_hparams = hparams
        apply = lambda: run_official_native_apply(
            model, tokenizer, requests, native_hparams, touched=touched,
            reset_cache=True, cache_history_width=0,
        )
        context = isolated_alphaedit_module_state()
    elif kind == "OFFICIAL-MEMIT":
        native_role = "official-memit-sequential"
        native_hparams = load_official_memit_hparams(
            easyedit_root=memit_easyedit_root
        )
        apply = lambda: run_official_memit_apply(
            model, tokenizer, requests, native_hparams, touched=touched,
        )
        context = _isolated_memit_cache()
    else:
        raise ODEBFContractError("independent FP32 baseline method differs")
    order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    entry_hashes = _hashes(touched)
    before = job_ledger.raw_free_payload()
    method_started = time.perf_counter()
    with context as module_state:
        with OfficialNativeZCapture(
            role=native_role, requests=requests, hparams=native_hparams
        ) as capture:
            counter = ModelForwardCounter(model, job_ledger)
            try:
                (apply_result, apply_wall) = _timed(apply)
            finally:
                counter.close()
        apply_payload, originals = apply_result
        _validate_official_fp32_apply(kind, apply_payload)
        _assert_no_low_precision_activity(apply_payload)
        binding = capture.finalize()
        target_timing = capture.timing_payload()
        if any(tensor_sha256(originals[name]) != entry_hashes[name] for name in touched):
            raise ODEBFStateError("independent FP32 native entry W differs")
        action_identity = canonical_hash({
            "method": kind,
            "case_index": case_index,
            "request_order_sha256": order,
            "apply": apply_payload,
        })
        freeze = _freeze(
            role=native_role, case_index=case_index, order=order,
            action_identity=action_identity,
        )
        cases = load_counterfact_cases_after_freeze(
            dataset_path, requests, freeze, expected_batch_size=BATCH_SIZE
        )
        counter = ModelForwardCounter(model, job_ledger)
        try:
            (z_result, z_wall) = _timed(lambda: evaluate_accepted_z_batch(
                model, tokenizer, requests, cases, role=native_role,
                round_index=case_index, binding=binding,
                committed_weight_sha256=_hashes(touched),
            ))
        finally:
            counter.close()
        z_observation, _ = z_result
        (scores, evaluator_wall) = _timed(
            lambda: _evaluate_w(model, tokenizer, cases, freeze=freeze, ledger=job_ledger)
        )
        energy = fp32_weight_energy(touched, base_values)
        (restore, restore_wall) = _timed(lambda: _restore(
            touched, base_values, mutation_lock=mutation_lock,
            entry_contract=entry_contract,
        ))
    if not bool(module_state.get("restored", True)):
        raise ODEBFStateError("independent FP32 native module state did not restore")
    target_wall = float(target_timing["compute_z_wall_seconds"])
    writer_wall = apply_wall - target_wall
    if writer_wall < 0.0 or not math.isfinite(writer_wall):
        raise ODEBFStateError("independent FP32 native timing nesting differs")
    z_summary = _endpoint_summary(z_observation["scores"])
    w_summary = _endpoint_summary(scores)
    total_wall = time.perf_counter() - method_started
    return {
        "method": kind,
        "request_order_sha256": order,
        "entry_W0_parameter_sha256": entry_hashes,
        "accepted_z": binding.raw_free_payload(),
        "z": {"summary": z_summary, "scores": z_observation["scores"]},
        "W": {"summary": w_summary, "scores": scores},
        "gap": _writer_gap(w_summary, z_summary),
        "apply": apply_payload,
        "update_energy": energy,
        "restore": restore,
        "timing": {
            "target_accepted_z_generation_seconds": target_wall,
            "writer_edit_core_seconds": writer_wall,
            "native_apply_scope_seconds": apply_wall,
            "target_plus_writer_seconds": apply_wall,
            "accepted_z_evaluator_seconds": z_wall,
            "immediate_post_evaluator_seconds": evaluator_wall,
            "restore_seconds": restore_wall,
            "case_method_total_seconds": total_wall,
            "policy": _timing_policy(),
        },
        "compute_delta": _ledger_delta(before, job_ledger.raw_free_payload()),
        "model_storage_dtype": "torch.float32",
        "numeric_storage_cast_count": 0,
        "autocast_count": 0,
        "bf16_path_call_count": 0,
        "fp16_conversion_count": 0,
        "module_state_restored": True,
    }


def _run_baseline_case(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    case_index: int,
    case_root: Path,
    dataset_path: Path,
    hparams: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_values: Mapping[str, torch.Tensor],
    mutation_lock: Any,
    entry_contract: str,
    entry_hashes: Mapping[str, str],
    entry_pointers: Mapping[str, int],
    job_ledger: ComputeLedger,
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    method_entry: dict[str, Mapping[str, str]] = {}
    case_started = time.perf_counter()
    for kind in BASELINE_METHODS:
        _assert_w0(
            touched, expected_contract=entry_contract,
            expected_hashes=entry_hashes, expected_pointers=entry_pointers,
        )
        seed_all(COMMON_SEED)
        method_entry[kind] = _hashes(touched)
        methods[kind] = _run_native_method(
            model, tokenizer, requests, case_index=case_index, kind=kind,
            dataset_path=dataset_path, hparams=hparams, touched=touched,
            base_values=base_values, mutation_lock=mutation_lock,
            entry_contract=entry_contract, job_ledger=job_ledger,
        )
        _assert_w0(
            touched, expected_contract=entry_contract,
            expected_hashes=entry_hashes, expected_pointers=entry_pointers,
        )
    if len({canonical_hash(value) for value in method_entry.values()}) != 1:
        raise ODEBFStateError("independent FP32 baseline entry identities differ")
    return {
        "cell": "BASELINE-GROUP",
        "case_index": case_index,
        "request_count": BATCH_SIZE,
        "request_order_sha256": scalable_ordered_request_digest(
            [str(item["request_sha256"]) for item in requests]
        ),
        "method_entry_W0_parameter_sha256": method_entry,
        "methods": methods,
        "case_total_seconds": time.perf_counter() - case_started,
        "W0_restored": True,
        "cross_method_state_count": 0,
        "sequential_carry_count": 0,
    }


def _run_c_case(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    role: str,
    cell: str,
    case_index: int,
    case_root: Path,
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
    entry_contract: str,
    entry_hashes: Mapping[str, str],
    entry_pointers: Mapping[str, int],
    request_microbatch_size: int,
    job_ledger: ComputeLedger,
) -> dict[str, Any]:
    before = job_ledger.raw_free_payload()
    case_started = time.perf_counter()
    order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    (target_result, target_wall) = _timed(lambda: _fp32_target_and_j0(
        model, tokenizer, requests, case_root=case_root, hparams=hparams,
        projector=projector, contexts=contexts,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256, controller_lock=controller_lock,
        request_by_sha256=request_by_sha256,
        population_by_sha256=population_by_sha256, schedule=schedule,
        theta0_cache=theta0_cache, touched=touched, base_receipt=base_receipt,
        base_values=base_values, request_microbatch_size=request_microbatch_size,
        job_ledger=job_ledger,
    ))
    target, target_public, objective_plan, capture_plan = target_result
    _assert_no_low_precision_activity(target_public)
    target_hash = tensor_sha256(target)
    freeze = _freeze(
        role=role, case_index=case_index, order=order,
        action_identity=target_public["identity_sha256"],
    )
    cases = load_counterfact_cases_after_freeze(
        dataset_path, requests, freeze, expected_batch_size=BATCH_SIZE
    )
    (target_restore, target_restore_wall) = _timed(lambda: _restore(
        touched, base_values, mutation_lock=mutation_lock,
        entry_contract=entry_contract,
    ))
    _assert_w0(
        touched, expected_contract=entry_contract,
        expected_hashes=entry_hashes, expected_pointers=entry_pointers,
    )
    binding = r52_binding(role, requests, hparams, target)
    counter = ModelForwardCounter(model, job_ledger)
    try:
        (z_result, z_wall) = _timed(lambda: evaluate_accepted_z_batch(
            model, tokenizer, requests, cases, role=role,
            round_index=case_index, binding=binding,
            committed_weight_sha256=_hashes(touched),
        ))
    finally:
        counter.close()
    z_observation, _ = z_result
    z_summary = _endpoint_summary(z_observation["scores"])
    writer_entry_hashes = _hashes(touched)
    if cell in ("C0", "C1", "C2"):
        arm = {
            "C0": JointPCWriterArm.C0,
            "C1": JointPCWriterArm.C1,
            "C2": JointPCWriterArm.C2,
        }[cell]

        def apply_writer() -> dict[str, Any]:
            entry = _writer_entry(
                model, tokenizer, requests, target=target, hparams=hparams,
                projector=projector, contexts=contexts,
                covariance_registry=covariance_registry,
                projector_sha256=projector_sha256,
                controller_lock=controller_lock,
                objective_plan=objective_plan, capture_plan=capture_plan,
            )
            writer = _run_pc_arm_fp32(
                model, arm, target=target, entry=entry, hparams=hparams,
                projector=projector, covariance_registry=covariance_registry,
                projector_sha256=projector_sha256,
                controller_lock=controller_lock, capture_plan=capture_plan,
            )
            writer["entry_control_pi"] = list(entry["control"].pi)
            writer["entry_joint_pi"] = list(entry["joint"].pi)
            writer["entry_joint_route"] = entry["joint"].receipt.raw_free_payload()
            writer["entry_nll"] = entry["entry_nll"]
            writer["endpoint_nll"] = entry["endpoint_nll"]
            if any(
                row["storage_dtype"] != "torch.float32"
                or row["numeric_storage_cast_count"] != 0
                or row["bf16_path_call_count"] != 0
                for row in writer["layers"]
            ):
                raise ODEBFContractError("TECHNICAL_INVALID_DTYPE")
            return writer

        (writer, writer_wall) = _timed(apply_writer)
        _assert_no_low_precision_activity(writer)
    elif cell == "C3":
        alpha_context = isolated_alphaedit_module_state()
        with alpha_context as alpha_state:
            with accepted_z_cache_template(
                requests, target, hparams, parent=case_root / "private"
            ) as (cache_template, bridge):
                counter = ModelForwardCounter(model, job_ledger)
                try:
                    (native_result, writer_wall) = _timed(
                        lambda: run_official_native_apply(
                            model, tokenizer, requests, hparams, touched=touched,
                            reset_cache=True, cache_history_width=0,
                            cache_template=cache_template,
                            expected_native_compute_z_call_count=0,
                            accepted_z_source="P1R52_K8_TERMINAL_TARGET",
                        )
                    )
                finally:
                    counter.close()
                c3_apply, originals = native_result
                _validate_official_fp32_apply("C3", c3_apply)
                _assert_no_low_precision_activity(c3_apply)
                if any(
                    tensor_sha256(originals[name]) != writer_entry_hashes[name]
                    for name in touched
                ):
                    raise ODEBFStateError("independent FP32 C3 entry differs")
                writer = {
                    "official_entrypoint": c3_apply["official_entrypoint"],
                    "official_source_file": str(__import__(
                        "easyeditor.models.alphaedit.AlphaEdit_main",
                        fromlist=["__file__"],
                    ).__file__),
                    "apply": c3_apply,
                    "accepted_z_bridge": bridge,
                    "native_compute_z_call_count": 0,
                    "p1r52_pc_router_decision_influence_count": 0,
                    "p1r52_barrier_decision_influence_count": 0,
                }
        if not alpha_state["restored"]:
            raise ODEBFStateError("independent FP32 C3 module state did not restore")
    else:
        raise ODEBFContractError("independent FP32 C cell differs")
    (scores, evaluator_wall) = _timed(
        lambda: _evaluate_w(model, tokenizer, cases, freeze=freeze, ledger=job_ledger)
    )
    w_summary = _endpoint_summary(scores)
    energy = fp32_weight_energy(touched, base_values)
    (restore, restore_wall) = _timed(lambda: _restore(
        touched, base_values, mutation_lock=mutation_lock,
        entry_contract=entry_contract,
    ))
    _assert_w0(
        touched, expected_contract=entry_contract,
        expected_hashes=entry_hashes, expected_pointers=entry_pointers,
    )
    case_total = time.perf_counter() - case_started
    return {
        "cell": cell,
        "case_index": case_index,
        "request_count": BATCH_SIZE,
        "request_order_sha256": order,
        "entry_W0_parameter_sha256": dict(entry_hashes),
        "writer_entry_W0_parameter_sha256": writer_entry_hashes,
        "accepted_z_sha256": target_hash,
        "accepted_z": binding.raw_free_payload(),
        "selected_target_receipt_identity": target_public["identity_sha256"],
        "target_public": target_public,
        "z": {"summary": z_summary, "scores": z_observation["scores"]},
        "W": {"summary": w_summary, "scores": scores},
        "gap": _writer_gap(w_summary, z_summary),
        "writer": writer,
        "update_energy": energy,
        "target_J0_restore": target_restore,
        "restore": restore,
        "timing": {
            "target_accepted_z_generation_seconds": target_wall,
            "writer_edit_core_seconds": writer_wall,
            "target_plus_writer_seconds": target_wall + writer_wall,
            "accepted_z_evaluator_seconds": z_wall,
            "immediate_post_evaluator_seconds": evaluator_wall,
            "target_J0_restore_seconds": target_restore_wall,
            "endpoint_restore_seconds": restore_wall,
            "case_total_seconds": case_total,
            "policy": _timing_policy(),
        },
        "compute_delta": _ledger_delta(before, job_ledger.raw_free_payload()),
        "dtype_contract": {
            "model_storage": "torch.float32",
            "projector_covariance_key_residual_q_update": "torch.float32",
            "numeric_storage_cast_count": 0,
            "autocast_count": 0,
            "bf16_conversion_count": 0,
            "fp16_conversion_count": 0,
            "bf16_candidate_materializer_call_count": 0,
        },
        "history_cache": {
            "cross_case_physical_W_carry_count": 0,
            "cross_case_controller_state_carry_count": 0,
            "cross_case_alpha_cache_history_carry_count": 0,
            "cross_case_pc_history_carry_count": 0,
        },
        "heldout_inner_evaluator_count": 0,
        "retry_count": 0,
        "imputation_count": 0,
        "W0_restored": True,
    }


def run_joint_pc_independent_full_fp32(
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
    cell = label_for_role(role)
    if (
        alias != "llama3-8b-inst"
        or len(stream_batches) != ROUND_COUNT
        or any(len(batch) != BATCH_SIZE for batch in stream_batches)
        or stream.get("root_digest") != STREAM_ROOT
        or stream.get("all_request_order_sha256") != STREAM_ORDER
    ):
        raise ODEBFContractError("independent FP32 matrix/stream differs")
    if (
        fp32_runtime is None
        or projector.dtype is not torch.float32
        or torch.is_autocast_enabled()
        or torch.is_autocast_enabled("cpu")
    ):
        raise ODEBFContractError("independent FP32 dtype boundary differs")
    parameter_inventory = full_fp32_parameter_inventory(model)
    entry_contract = _model_w0_contract(touched)
    entry_hashes = _hashes(touched)
    entry_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    if entry_hashes != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("independent FP32 base receipt differs")
    runtime_started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    case_shas: list[str] = []
    for case_index, requests in enumerate(stream_batches, start=1):
        case_root = raw_root / "cases" / f"case-{case_index:02d}"
        case_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        seed_all(COMMON_SEED)
        _assert_w0(
            touched, expected_contract=entry_contract,
            expected_hashes=entry_hashes, expected_pointers=entry_pointers,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            if cell == "BASELINE-GROUP":
                case = _run_baseline_case(
                    model, tokenizer, requests, case_index=case_index,
                    case_root=case_root, dataset_path=dataset_path,
                    hparams=hparams, touched=touched, base_values=base_values,
                    mutation_lock=mutation_lock, entry_contract=entry_contract,
                    entry_hashes=entry_hashes, entry_pointers=entry_pointers,
                    job_ledger=job_ledger,
                )
            else:
                case = _run_c_case(
                    model, tokenizer, requests, role=role, cell=cell,
                    case_index=case_index, case_root=case_root, hparams=hparams,
                    projector=projector, contexts=contexts,
                    covariance_registry=covariance_registry,
                    projector_sha256=projector_sha256,
                    controller_lock=controller_lock,
                    request_by_sha256=request_by_sha256,
                    population_by_sha256=population_by_sha256,
                    schedule=schedule, theta0_cache=theta0_cache,
                    dataset_path=dataset_path, mutation_lock=mutation_lock,
                    touched=touched, base_receipt=base_receipt,
                    base_values=base_values, entry_contract=entry_contract,
                    entry_hashes=entry_hashes, entry_pointers=entry_pointers,
                    request_microbatch_size=request_microbatch_size,
                    job_ledger=job_ledger,
                )
        except Exception as exc:
            restore = _restore(
                touched, base_values, mutation_lock=mutation_lock,
                entry_contract=entry_contract,
            )
            failure = {
                "schema": "ode-edit-s05-p1r52-joint-pc-independent-fp32-case-failure/v1",
                "instruction_id": INSTRUCTION_ID,
                "cell": cell,
                "case_index": case_index,
                "classification": "TECHNICAL_FAIL",
                "exception_class": type(exc).__name__,
                "exception_message_sha256": canonical_hash(str(exc)),
                "W0_restore": restore,
                "retry_count": 0,
                "imputation_count": 0,
            }
            failure["identity_sha256"] = canonical_hash(failure)
            _atomic_write_once(case_root / "failure.json", failure)
            raise
        _sync()
        case["gpu_host_observation"] = _gpu_observation()
        case["slice_identity"] = {
            "stream_root": STREAM_ROOT,
            "stream_order": STREAM_ORDER,
            "case_index": case_index,
            "request_order_sha256": case["request_order_sha256"],
        }
        _assert_no_low_precision_activity(case)
        case, tensor_paths = _raw_free_json_tree(case)
        case["terminal_raw_free_tensor_adapter"] = {
            "replacement_count": len(tensor_paths),
            "paths": list(tensor_paths),
            "serialized_tensor_value_count": 0,
            "decision_influence_count": 0,
        }
        case["identity_sha256"] = canonical_hash(case)
        case_sha = _atomic_write_once(case_root / "terminal.json", case)
        case_shas.append(case_sha)
        cases.append({
            "case_index": case_index,
            "request_order_sha256": case["request_order_sha256"],
            "accepted_z_sha256": case.get("accepted_z_sha256"),
            "terminal_sha256": case_sha,
            "W0_restored": True,
        })
        stages.record(
            f"joint_pc_independent_fp32_case_{case_index}",
            {
                "cell": cell,
                "case_index": case_index,
                "completed_case_count": len(cases),
                "W0_restored": True,
                "sequential_carry_count": 0,
            },
        )
    _assert_w0(
        touched, expected_contract=entry_contract,
        expected_hashes=entry_hashes, expected_pointers=entry_pointers,
    )
    terminal = {
        "schema": "ode-edit-s05-p1r52-joint-pc-independent-full-fp32-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "status": "TERMINAL_VALID",
        "source_head": source_head,
        "alias": alias,
        "role": role,
        "cell": cell,
        "case_count": ROUND_COUNT,
        "completed_case_count": len(cases),
        "valid_request_count": ROUND_COUNT * BATCH_SIZE,
        "technical_failure_count": 0,
        "scientific_failure_count": 0,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "cases": cases,
        "case_terminal_sha256": case_shas,
        "independent_W0_entry_count": ROUND_COUNT,
        "independent_W0_restore_count": ROUND_COUNT,
        "cross_case_state_count": 0,
        "sequential_continuity_count": 0,
        "retry_count": 0,
        "imputation_count": 0,
        "dtype_contract": {
            "status": "FULL_FP32_PASS",
            "requested_dtype": "torch.float32",
            "loaded_model_dtype": "torch.float32",
            "parameter_dtype_counts": parameter_inventory["parameter_dtype_counts"],
            "parameter_inventory_identity_sha256": parameter_inventory["identity_sha256"],
            "model_and_floating_parameters": "torch.float32",
            "algorithm_tensors": "torch.float32",
            "autocast_count": 0,
            "bf16_conversion_count": 0,
            "fp16_conversion_count": 0,
            "numeric_storage_cast_count": 0,
            "bf16_candidate_materializer_call_count": 0,
        },
        "timing_policy": _timing_policy(),
        "runtime_after_model_preflight_seconds": time.perf_counter() - runtime_started,
        "model_load_preflight": {
            key: value for key, value in job_ledger.component_wall_seconds.items()
            if key in ("model_load", "context_generation_twice", "theta0_teacher_population")
        },
        "job_compute": job_ledger.raw_free_payload(),
        "final_gpu_host_observation": _gpu_observation(),
        "W0_restored": True,
        "scientific_promotion": False,
    }
    _assert_no_low_precision_activity(terminal)
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r52-joint-pc-independent-full-fp32-manifest/v1",
        "source_head": source_head,
        "role": role,
        "cell": cell,
        "terminal_sha256": terminal_sha,
        "case_terminal_sha256": case_shas,
        "member_count": 2 + ROUND_COUNT,
        "members": ["terminal.json", "manifest.json"] + [
            f"raw/cases/case-{index:02d}/terminal.json"
            for index in range(1, ROUND_COUNT + 1)
        ],
        "W0_restored": True,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    return {
        "status": "P1R52_JOINT_PC_INDEPENDENT_FULL_FP32_TERMINAL",
        "cell": cell,
        "completed_case_count": ROUND_COUNT,
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "W0_restored": True,
    }


__all__ = [
    "BASELINE_METHODS",
    "CELL_LABELS",
    "INSTRUCTION_ID",
    "METHOD_ID",
    "RESULT_NAMES",
    "ROLES",
    "expected_result_name",
    "full_fp32_parameter_inventory",
    "is_joint_pc_independent_fp32_role",
    "label_for_role",
    "role_for_cell",
    "run_joint_pc_independent_full_fp32",
]
