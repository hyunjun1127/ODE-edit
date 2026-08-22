#!/usr/bin/env python3
"""P4 Euler R1 Stage1: one 10-step trajectory per model/arm/h."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EASYEDIT_ROOT = Path("/data/janghj/EasyEdit")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EASYEDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(EASYEDIT_ROOT))

import torch

from project.run_scripts.ode_bf.alpha_backend import fresh_contexts_twice, seed_all
from project.run_scripts.ode_bf.contracts import (
    BATCH_SIZE,
    COMMON_SEED,
    MODEL_ALIASES,
    ODEBFContractError,
    ODEBFStateError,
    canonical_hash,
)
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p1_runtime import P1StageRecorder, _atomic_write_once
from project.run_scripts.ode_bf.p1_scalable_batched_experiment import _model_w0_contract
from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    build_p1r24_kl_plan,
    evaluate_p1r24_kl,
    verify_p1r24_alphaedit_geometry,
)
from project.run_scripts.ode_bf.p1r24_independent_b10x10_selection import (
    load_historical_h0_batches,
)
from project.run_scripts.ode_bf.p4_euler_binding import (
    build_euler_nonsemantic_identity,
    build_p4_euler_objective_callback,
)
from project.run_scripts.ode_bf.p4_euler_integrator import (
    P4_EULER_INSTRUCTION_ID,
    run_raw_projected_euler,
)
from project.run_scripts.ode_bf.p4_euler_model_binding import (
    evaluate_checkpointed_model_terms,
    observe_model_terms_no_grad,
)
from project.run_scripts.ode_bf.p4_hf_consumed_closure import (
    load_p4_full_fp32_from_sealed_snapshot,
    load_p4_hf_consumed_closure_seal,
)
from project.run_scripts.ode_bf.p4_semantic_barrier import P4TargetArm
from project.run_scripts.ode_bf.p4_target_solver_binding import (
    build_p4_paired_objective_plans,
)
from project.run_scripts.ode_bf.scalable_batched_model import (
    _parameter_inventory_sha256,
    build_scalable_capture_plan,
    capture_scalable_physical_state,
)
from project.run_scripts.ode_bf.scalable_batched_runtime import (
    initial_target_from_capture,
    scalable_ordered_request_digest,
)


H_GRID = (0.0625, 0.25, 1.0, 4.0)
PREFIXES = (1, 3, 5, 10)
STAGE1_LOCK_ROOT = "649d4a271b29cceec3f20080db7c763e47c7f143a9a43fafaa7a4db3806bbe0b"
STAGE2_LOCK_ROOT = "b60e1b442a96700e5bccbe5d72ab4be0be08128f740fbbc707c0165c653036af"
SELECTED_H = 0.25
SELECTED_TARGET_HORIZON = 1.25
HF_SEAL = REPO_ROOT / "agents/server4/p4-hf-consumed-closure-seal.json"
STREAM_SEAL_RELATIVE = Path("canonical/p1r24_independent_b10x10_stream_seal.json")
DATASET = EASYEDIT_ROOT / "data/counterfact/counterfact.json"
HPARAMS = {
    "llama3-8b-inst": EASYEDIT_ROOT / "hparams/AlphaEdit/llama3-8b.yaml",
    "qwen2.5-7b-inst": EASYEDIT_ROOT / "hparams/AlphaEdit/qwen2.5-7b.yaml",
}
EXECUTION_MICROBATCH = 1


def _load_regular_json(path: Path) -> dict[str, Any]:
    observed = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise ODEBFContractError("P4 Euler calibration JSON differs")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ODEBFContractError("P4 Euler calibration JSON object differs")
    return value


def _load_final_preflight(
    path: Path, *, alias: str, source_head: str, stage: str
) -> dict[str, Any]:
    value = _load_regular_json(path)
    payload = dict(value)
    identity = payload.pop("identity_sha256", None)
    binding = value.get("model_input_binding")
    execution = value.get("execution_binding")
    expected_schema = (
        "ode-edit-s05-p4-euler-calibration-final-pre-gpu/v1"
        if stage == "stage1"
        else "ode-edit-s05-p4-euler-calibration-stage2-final-pre-gpu/v1"
    )
    expected_root = STAGE1_LOCK_ROOT if stage == "stage1" else STAGE2_LOCK_ROOT
    if (
        value.get("schema") != expected_schema
        or value.get("status") != "FINAL_PRE_GPU_PASS"
        or value.get("model_load_authorized") is not True
        or value.get("source_head") != source_head
        or identity != canonical_hash(payload)
        or not isinstance(binding, Mapping)
        or alias not in binding
        or not isinstance(execution, Mapping)
        or execution.get("calibration_lock_root") != expected_root
        or (
            stage == "stage1"
            and execution.get("single_trajectory_prefix_reuse") is not True
        )
        or (
            stage == "stage2"
            and (
                execution.get("selected_h") != SELECTED_H
                or execution.get("selected_target_horizon")
                != SELECTED_TARGET_HORIZON
                or execution.get("microsteps") != [5, 10]
                or execution.get("step_sizes") != [0.25, 0.125]
            )
        )
        or value.get("project_gpu_cap") != 2
    ):
        raise ODEBFContractError("P4 Euler calibration final preflight differs")
    return value


def _assert_full_fp32(model: torch.nn.Module) -> Mapping[str, Any]:
    floating = [value for value in model.parameters() if value.is_floating_point()]
    if (
        not floating
        or any(value.dtype is not torch.float32 for value in floating)
        or any(value.requires_grad or value.grad is not None for value in floating)
        or torch.is_autocast_enabled()
        or torch.is_autocast_enabled("cpu")
        or next(model.parameters()).device != torch.device("cuda:0")
    ):
        raise ODEBFContractError("P4 Euler calibration FULL-FP32 differs")
    return {
        "dtype": "torch.float32",
        "floating_parameter_count": len(floating),
        "requires_grad_parameter_count": 0,
        "parameter_gradient_count": 0,
        "autocast_enabled": False,
        "cpu_autocast_enabled": False,
        "quantization": False,
        "device": "cuda:0",
    }


def _touched_weights(model: torch.nn.Module, hparams: Any) -> dict[str, torch.nn.Parameter]:
    result = {
        f"{hparams.rewrite_module_tmp.format(int(layer))}.weight": model.get_parameter(
            f"{hparams.rewrite_module_tmp.format(int(layer))}.weight"
        )
        for layer in hparams.layers
    }
    if not result or any(value.dtype is not torch.float32 for value in result.values()):
        raise ODEBFContractError("P4 Euler calibration W0 inventory differs")
    return result


def _summary(values: Sequence[float]) -> Mapping[str, float]:
    tensor = torch.tensor(tuple(values), dtype=torch.float32)
    return {
        "median": float(torch.quantile(tensor, 0.5)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "max": float(torch.max(tensor)),
        "mean": float(torch.mean(tensor)),
    }


def _source_telemetry(row: Mapping[str, Any]) -> Mapping[str, Any]:
    objective = row.get("objective_telemetry")
    source = objective.get("source_telemetry") if isinstance(objective, Mapping) else None
    if not isinstance(source, Mapping):
        raise ODEBFContractError("P4 Euler calibration source telemetry differs")
    return source


def _snapshot_tensor(path: Path, value: torch.Tensor) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("P4 Euler calibration snapshot exists")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value.detach().to(device="cpu", dtype=torch.float32), temporary)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    return tensor_sha256(value)


def _trajectory(
    model: torch.nn.Module,
    paired: Any,
    kl_plan: Any,
    teacher: Sequence[torch.Tensor],
    *,
    arm: P4TargetArm,
    h: float,
    initial_target: torch.Tensor,
    current_terminal: torch.Tensor,
    target_layer_name: str,
    lock: P1R24AliasTargetLock,
    request_order_sha256: str,
    context_sha256: str,
    teacher_sha256: str,
    touched: Mapping[str, torch.nn.Parameter],
    expected_w0: str,
    snapshot_root: Path,
) -> Mapping[str, Any]:
    origin = initial_target.detach().clone().to(dtype=torch.float32)
    radius = lock.clamp_factor * torch.linalg.vector_norm(origin, dim=0)
    radius_sha = tensor_sha256(radius)
    nonsemantic = build_euler_nonsemantic_identity(
        request_order_sha256=request_order_sha256,
        context_sha256=context_sha256,
        teacher_sha256=teacher_sha256,
        origin_sha256=tensor_sha256(origin),
        radius_sha256=radius_sha,
        target_layer_name=target_layer_name,
        kl_factor=lock.kl_factor,
        decay_factor=lock.decay_factor,
        clamp_factor=lock.clamp_factor,
        target_horizon=10.0 * h,
        microsteps=10,
    )

    def terms(candidate: torch.Tensor):
        return evaluate_checkpointed_model_terms(
            model,
            paired,
            kl_plan,
            teacher,
            target_state=candidate,
            current_terminal=current_terminal,
            target_origin=origin,
            target_layer_name=target_layer_name,
        )

    objective = build_p4_euler_objective_callback(
        terms, arm=arm, kl_factor=lock.kl_factor, decay_factor=lock.decay_factor
    )
    pre_states: list[torch.Tensor] = []

    def observed_objective(candidate: torch.Tensor):
        pre_states.append(candidate.detach().clone())
        return objective(candidate)

    inventory_before = _parameter_inventory_sha256(model)
    started = time.perf_counter()
    result = run_raw_projected_euler(
        origin,
        origin=origin,
        radius_by_request=radius,
        target_horizon=10.0 * h,
        microsteps=10,
        objective_callback=observed_objective,
        frozen_content_sha256=lambda: _parameter_inventory_sha256(model),
        outer_step_index=0,
    )
    wall = time.perf_counter() - started
    if len(pre_states) != 10:
        raise ODEBFContractError("P4 Euler calibration trajectory count differs")
    endpoints = {
        1: pre_states[1],
        3: pre_states[3],
        5: pre_states[5],
        10: result.final_state.detach().clone(),
    }
    final_observation = observe_model_terms_no_grad(
        model,
        paired,
        kl_plan,
        teacher,
        target_state=endpoints[10],
        current_terminal=current_terminal,
        target_origin=origin,
        target_layer_name=target_layer_name,
    )
    prefix_rows: list[dict[str, Any]] = []
    for prefix in PREFIXES:
        executed = result.step_receipts[:prefix]
        hits = sum(
            int(hit)
            for row in executed
            for hit in row["clamp_hit_by_request"]
        )
        denominator = BATCH_SIZE * prefix
        endpoint_source = (
            final_observation
            if prefix == 10
            else _source_telemetry(result.step_receipts[prefix])
        )
        displacement = result.step_receipts[prefix - 1][
            "post_clamp_displacement_by_request"
        ]
        field_row = result.step_receipts[prefix if prefix < 10 else 9]
        relative = Path(arm.value.replace("±", "pm").replace("+", "plus")) / (
            f"h-{h:g}-M{prefix}.pt"
        )
        snapshot_sha = _snapshot_tensor(snapshot_root / relative, endpoints[prefix])
        row = {
            "M": prefix,
            "h": h,
            "target_horizon": prefix * h,
            "target_sha256": snapshot_sha,
            "target_snapshot_relative_path": str(relative),
            "new_nll": _summary(endpoint_source["new_nll_by_request"]),
            "old_nll": _summary(endpoint_source["old_nll_by_request"]),
            "new_minus_old_margin": _summary(
                endpoint_source["new_minus_old_margin_by_request"]
            ),
            "displacement": _summary(displacement),
            "raw_field_norm": _summary(field_row["raw_field_norm_by_request"]),
            "raw_field_state_semantics": (
                f"z_M{prefix}"
                if prefix < 10
                else "z_M9_PRE_FINAL_UPDATE_NO_DUPLICATE_AUTOGRAD"
            ),
            "clamp_hit_count": hits,
            "clamp_denominator": denominator,
            "clamp_fraction": hits / denominator,
            "finite": True,
            "prefix_observation_decision_influence_count": 0,
        }
        row["identity_sha256"] = canonical_hash(row)
        prefix_rows.append(row)
    inventory_after = _parameter_inventory_sha256(model)
    if inventory_after != inventory_before or _model_w0_contract(touched) != expected_w0:
        raise ODEBFStateError("P4 Euler calibration W0 pointer/bytes changed")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ODEBFStateError("P4 Euler calibration parameter gradient exists")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-calibration-stage1-trajectory/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "arm": arm.value,
        "h": h,
        "executed_microsteps": 10,
        "trajectory_count": 1,
        "separate_prefix_trajectory_count": 0,
        "logical_field_evaluation_count": 10,
        "actual_autograd_grad_call_count": 10,
        "duplicate_autograd_evaluation_count": 0,
        "checkpoint_recompute_enabled": True,
        "nonsemantic_identity": nonsemantic,
        "prefix_rows": prefix_rows,
        "integrator_receipt": result.receipt,
        "final_value_only_observation": final_observation,
        "W0_pointer_version_inventory_entry": inventory_before,
        "W0_pointer_version_inventory_exit": inventory_after,
        "W0_pointer_version_change_count": 0,
        "W0_bytes_change_count": 0,
        "optimizer": "NONE",
        "adam_state_count": 0,
        "parameter_gradient_count": 0,
        "loss_backward_count": 0,
        "writer_materialization_count": 0,
        "cache_append_count": 0,
        "heldout_access_count": 0,
        "wall_time_seconds": wall,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _fixed_endpoint_trajectory(
    model: torch.nn.Module,
    paired: Any,
    kl_plan: Any,
    teacher: Sequence[torch.Tensor],
    *,
    arm: P4TargetArm,
    microsteps: int,
    initial_target: torch.Tensor,
    current_terminal: torch.Tensor,
    target_layer_name: str,
    lock: P1R24AliasTargetLock,
    request_order_sha256: str,
    context_sha256: str,
    teacher_sha256: str,
    touched: Mapping[str, torch.nn.Parameter],
    expected_w0: str,
    snapshot_root: Path,
) -> tuple[Mapping[str, Any], torch.Tensor]:
    """Run one fresh-origin Stage2 endpoint at the selected common horizon."""

    origin = initial_target.detach().clone().to(dtype=torch.float32)
    radius = lock.clamp_factor * torch.linalg.vector_norm(origin, dim=0)
    step_size = SELECTED_TARGET_HORIZON / microsteps
    nonsemantic = build_euler_nonsemantic_identity(
        request_order_sha256=request_order_sha256,
        context_sha256=context_sha256,
        teacher_sha256=teacher_sha256,
        origin_sha256=tensor_sha256(origin),
        radius_sha256=tensor_sha256(radius),
        target_layer_name=target_layer_name,
        kl_factor=lock.kl_factor,
        decay_factor=lock.decay_factor,
        clamp_factor=lock.clamp_factor,
        target_horizon=SELECTED_TARGET_HORIZON,
        microsteps=microsteps,
    )

    def terms(candidate: torch.Tensor):
        return evaluate_checkpointed_model_terms(
            model,
            paired,
            kl_plan,
            teacher,
            target_state=candidate,
            current_terminal=current_terminal,
            target_origin=origin,
            target_layer_name=target_layer_name,
        )

    objective = build_p4_euler_objective_callback(
        terms, arm=arm, kl_factor=lock.kl_factor, decay_factor=lock.decay_factor
    )
    inventory_before = _parameter_inventory_sha256(model)
    started = time.perf_counter()
    result = run_raw_projected_euler(
        origin,
        origin=origin,
        radius_by_request=radius,
        target_horizon=SELECTED_TARGET_HORIZON,
        microsteps=microsteps,
        objective_callback=objective,
        frozen_content_sha256=lambda: _parameter_inventory_sha256(model),
        outer_step_index=0,
    )
    wall = time.perf_counter() - started
    endpoint = result.final_state.detach().clone()
    observation = observe_model_terms_no_grad(
        model,
        paired,
        kl_plan,
        teacher,
        target_state=endpoint,
        current_terminal=current_terminal,
        target_origin=origin,
        target_layer_name=target_layer_name,
    )
    hits = sum(
        int(hit)
        for row in result.step_receipts
        for hit in row["clamp_hit_by_request"]
    )
    denominator = BATCH_SIZE * microsteps
    displacement = result.step_receipts[-1]["post_clamp_displacement_by_request"]
    relative = Path(arm.value.replace("±", "pm").replace("+", "plus")) / (
        f"M{microsteps}.pt"
    )
    snapshot_sha = _snapshot_tensor(snapshot_root / relative, endpoint)
    inventory_after = _parameter_inventory_sha256(model)
    if inventory_after != inventory_before or _model_w0_contract(touched) != expected_w0:
        raise ODEBFStateError("P4 Euler Stage2 W0 pointer/bytes changed")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ODEBFStateError("P4 Euler Stage2 parameter gradient exists")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-calibration-stage2-trajectory/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "arm": arm.value,
        "M": microsteps,
        "h": step_size,
        "target_horizon": SELECTED_TARGET_HORIZON,
        "fresh_origin": True,
        "target_sha256": snapshot_sha,
        "target_snapshot_relative_path": str(relative),
        "new_nll": _summary(observation["new_nll_by_request"]),
        "old_nll": _summary(observation["old_nll_by_request"]),
        "new_minus_old_margin": _summary(
            observation["new_minus_old_margin_by_request"]
        ),
        "displacement": _summary(displacement),
        "clamp_hit_count": hits,
        "clamp_denominator": denominator,
        "clamp_fraction": hits / denominator,
        "finite": True,
        "executed_microsteps": microsteps,
        "logical_field_evaluation_count": microsteps,
        "actual_autograd_grad_call_count": microsteps,
        "duplicate_autograd_evaluation_count": 0,
        "nonsemantic_identity": nonsemantic,
        "integrator_receipt": result.receipt,
        "final_value_only_observation": observation,
        "W0_pointer_version_inventory_entry": inventory_before,
        "W0_pointer_version_inventory_exit": inventory_after,
        "W0_pointer_version_change_count": 0,
        "W0_bytes_change_count": 0,
        "optimizer": "NONE",
        "adam_state_count": 0,
        "parameter_gradient_count": 0,
        "loss_backward_count": 0,
        "writer_materialization_count": 0,
        "cache_append_count": 0,
        "heldout_access_count": 0,
        "native_access_count": 0,
        "wall_time_seconds": wall,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload, endpoint


def _run_stage2(
    args: argparse.Namespace,
    *,
    model: torch.nn.Module,
    paired: Any,
    kl_plan: Any,
    teacher: Sequence[torch.Tensor],
    initial: Any,
    hparams: Any,
    lock: P1R24AliasTargetLock,
    request_order: str,
    context_sha: str,
    teacher_sha: str,
    touched: Mapping[str, torch.nn.Parameter],
    expected_w0: str,
    inventory_w0: str,
    fp32: Mapping[str, Any],
) -> Mapping[str, Any]:
    comparisons: list[Mapping[str, Any]] = []
    trajectory_rows: list[Mapping[str, Any]] = []
    target_layer_name = hparams.layer_module_tmp.format(int(hparams.layers[-1]))
    for arm in (P4TargetArm.POSITIVE, P4TargetArm.POSITIVE_NEGATIVE):
        endpoints: dict[int, torch.Tensor] = {}
        arm_rows: list[Mapping[str, Any]] = []
        for microsteps in (5, 10):
            row, endpoint = _fixed_endpoint_trajectory(
                model,
                paired,
                kl_plan,
                teacher,
                arm=arm,
                microsteps=microsteps,
                initial_target=initial.target_z,
                current_terminal=initial.current_terminal_z,
                target_layer_name=target_layer_name,
                lock=lock,
                request_order_sha256=request_order,
                context_sha256=context_sha,
                teacher_sha256=teacher_sha,
                touched=touched,
                expected_w0=expected_w0,
                snapshot_root=args.output_root / "snapshots",
            )
            endpoints[microsteps] = endpoint
            arm_rows.append(row)
            trajectory_rows.append(row)
            path = (
                args.output_root
                / "trajectories"
                / arm.value.replace("±", "pm").replace("+", "plus")
                / f"M{microsteps}.json"
            )
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _atomic_write_once(path, row)
        origin = initial.target_z.to(dtype=torch.float32)
        numerator = torch.linalg.vector_norm(endpoints[5] - endpoints[10], dim=0)
        denominator = torch.clamp(
            torch.linalg.vector_norm(endpoints[10] - origin, dim=0), min=1e-12
        )
        discrepancy = numerator / denominator
        summary = _summary([float(value) for value in discrepancy])
        clamp_pass = all(float(row["clamp_fraction"]) < 0.5 for row in arm_rows)
        threshold_pass = (
            summary["median"] <= 0.10
            and summary["p90"] <= 0.25
            and summary["max"] <= 0.50
        )
        comparison: dict[str, Any] = {
            "schema": "ode-edit-s05-p4-euler-calibration-stage2-comparison/v1",
            "arm": arm.value,
            "target_horizon": SELECTED_TARGET_HORIZON,
            "M5_h": 0.25,
            "M10_h": 0.125,
            "endpoint_discrepancy": summary,
            "thresholds": {"median": 0.10, "p90": 0.25, "max": 0.50},
            "threshold_pass": threshold_pass,
            "clamp_fraction_M5": arm_rows[0]["clamp_fraction"],
            "clamp_fraction_M10": arm_rows[1]["clamp_fraction"],
            "clamp_pass": clamp_pass,
            "finite": bool(torch.isfinite(discrepancy).all()),
            "M5_target_sha256": arm_rows[0]["target_sha256"],
            "M10_target_sha256": arm_rows[1]["target_sha256"],
            "decision_influence_source": "TRAIN_ONLY_NUMERICAL_ENDPOINT",
            "heldout_decision_influence_count": 0,
        }
        comparison["identity_sha256"] = canonical_hash(comparison)
        comparisons.append(comparison)
        path = args.output_root / "comparisons" / (
            arm.value.replace("±", "pm").replace("+", "plus") + ".json"
        )
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _atomic_write_once(path, comparison)
    if (
        _model_w0_contract(touched) != expected_w0
        or _parameter_inventory_sha256(model) != inventory_w0
        or any(parameter.grad is not None for parameter in model.parameters())
    ):
        raise ODEBFStateError("P4 Euler Stage2 terminal W0 freeze differs")
    passed = all(
        row["threshold_pass"] and row["clamp_pass"] and row["finite"]
        for row in comparisons
    )
    terminal: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-calibration-stage2-terminal/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "status": (
            "STAGE2_MODEL_CELL_PASS"
            if passed
            else "STAGE2_MODEL_CELL_SCIENTIFIC_HOLD"
        ),
        "source_head": args.source_head,
        "alias": args.model,
        "case_index": 1,
        "calibration_label": "PERMANENT_CALIBRATION_ONLY",
        "confirmatory_eligibility": False,
        "selected_h": SELECTED_H,
        "selected_target_horizon": SELECTED_TARGET_HORIZON,
        "M": [5, 10],
        "h": [0.25, 0.125],
        "trajectory_count": 4,
        "executed_microstep_count": 30,
        "actual_autograd_grad_call_count": 30,
        "duplicate_autograd_evaluation_count": 0,
        "comparison_identity_sha256": [
            row["identity_sha256"] for row in comparisons
        ],
        "trajectory_identity_sha256": [
            row["identity_sha256"] for row in trajectory_rows
        ],
        "W0_pointer_bytes_sha256": expected_w0,
        "W0_restored": True,
        "optimizer": "NONE",
        "adam_state_count": 0,
        "parameter_gradient_count": 0,
        "loss_backward_count": 0,
        "writer_materialization_count": 0,
        "cache_append_count": 0,
        "heldout_access_count": 0,
        "native_access_count": 0,
        "full_fp32": fp32,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(args.output_root / "terminal.json", terminal)
    return {
        "status": terminal["status"],
        "alias": args.model,
        "terminal_sha256": terminal_sha,
        "identity_sha256": terminal["identity_sha256"],
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if os.environ.get("PROJECT_GPU_CAP") != "2" or any(
        os.environ.get(name) != "1"
        for name in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    ):
        raise ODEBFContractError("P4 Euler calibration cap/offline differs")
    observed_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True,
        capture_output=True,
    ).stdout.strip()
    if observed_head != args.source_head:
        raise ODEBFContractError("P4 Euler calibration source HEAD differs")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError("P4 Euler calibration output root is create-once")
    os.umask(0o077)
    args.output_root.mkdir(mode=0o700, parents=True)
    stages = P1StageRecorder(args.output_root / "stages")
    final = _load_final_preflight(
        args.final_pre_gpu_receipt,
        alias=args.model,
        source_head=args.source_head,
        stage=args.stage,
    )
    binding = final["model_input_binding"][args.model]
    stages.record("pre_model_final_gate", {
        "final_pre_gpu_identity": final["identity_sha256"],
        "calibration_lock_root": (
            STAGE1_LOCK_ROOT if args.stage == "stage1" else STAGE2_LOCK_ROOT
        ),
        "model_input_binding": binding,
    })
    seed_all(COMMON_SEED)
    hf_seal = load_p4_hf_consumed_closure_seal(HF_SEAL, repo_root=REPO_ROOT)
    model, tokenizer, hf_receipt = load_p4_full_fp32_from_sealed_snapshot(
        hf_seal,
        args.model,
        stream_receipt=final["transfer_receipt"],
        final_pre_gpu_receipt=final,
        reuse_final_verified_closure=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams

    hparams = AlphaEditHyperParams.from_hparams(str(HPARAMS[args.model]))
    hparams.device = 0
    hparams.stats_dir = str(EASYEDIT_ROOT / "examples/data/stats")
    model.config._name_or_path = hparams.model_name
    fp32 = _assert_full_fp32(model)
    stages.record("post_model_full_fp32_offline", {
        "hf": hf_receipt.to_dict(), "full_fp32": fp32, "offline": True,
    })
    stream = _load_regular_json(args.stream_root / STREAM_SEAL_RELATIVE)
    requests = load_historical_h0_batches(DATASET, stream)[0]
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    if (
        len(requests) != BATCH_SIZE
        or request_order != binding["request_order_sha256"]
        or binding["case_index"] != 1
        or binding["calibration_label"] != "PERMANENT_CALIBRATION_ONLY"
    ):
        raise ODEBFContractError("P4 Euler B1 calibration binding differs")
    contexts, context_sha = fresh_contexts_twice(model, tokenizer, seed=COMMON_SEED)
    paired = build_p4_paired_objective_plans(
        model,
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=EXECUTION_MICROBATCH,
        fact_token_strategy=hparams.fact_token,
    )
    capture_plan = build_scalable_capture_plan(
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=EXECUTION_MICROBATCH,
        fact_token_strategy=hparams.fact_token,
    )
    physical = capture_scalable_physical_state(model, capture_plan, hparams)
    initial = initial_target_from_capture(physical)
    lock = P1R24AliasTargetLock.for_alias(args.model)
    alpha_geometry = verify_p1r24_alphaedit_geometry(
        hparams, lock, easyedit_root=EASYEDIT_ROOT
    )
    kl_plan = build_p1r24_kl_plan(
        tokenizer,
        requests,
        request_order_sha256=request_order,
        request_microbatch_size=EXECUTION_MICROBATCH,
        fact_token_strategy=hparams.fact_token,
    )
    teacher_result, teacher = evaluate_p1r24_kl(
        model, kl_plan, teacher_log_probs=None
    )
    teacher_sha = canonical_hash({
        "teacher_tensor_sha256": [tensor_sha256(value) for value in teacher],
        "request_order_sha256": request_order,
        "capture_count": 1,
    })
    touched = _touched_weights(model, hparams)
    expected_w0 = _model_w0_contract(touched)
    inventory_w0 = _parameter_inventory_sha256(model)
    stages.record("post_shared_B1_W0_gate", {
        "alias": args.model,
        "request_order_sha256": request_order,
        "context_sha256": context_sha,
        "W0_pointer_bytes_sha256": expected_w0,
        "W0_pointer_version_inventory": inventory_w0,
        "alpha_geometry": alpha_geometry,
        "teacher": teacher_result.raw_free_payload(),
        "teacher_sha256": teacher_sha,
        "writer_materialization_count": 0,
        "heldout_access_count": 0,
    })
    torch.cuda.reset_peak_memory_stats()
    if args.stage == "stage2":
        return _run_stage2(
            args,
            model=model,
            paired=paired,
            kl_plan=kl_plan,
            teacher=teacher,
            initial=initial,
            hparams=hparams,
            lock=lock,
            request_order=request_order,
            context_sha=context_sha,
            teacher_sha=teacher_sha,
            touched=touched,
            expected_w0=expected_w0,
            inventory_w0=inventory_w0,
            fp32=fp32,
        )
    trajectory_rows: list[Mapping[str, Any]] = []
    for arm in (P4TargetArm.POSITIVE, P4TargetArm.POSITIVE_NEGATIVE):
        for h in H_GRID:
            row = _trajectory(
                model,
                paired,
                kl_plan,
                teacher,
                arm=arm,
                h=h,
                initial_target=initial.target_z,
                current_terminal=initial.current_terminal_z,
                target_layer_name=hparams.layer_module_tmp.format(
                    int(hparams.layers[-1])
                ),
                lock=lock,
                request_order_sha256=request_order,
                context_sha256=context_sha,
                teacher_sha256=teacher_sha,
                touched=touched,
                expected_w0=expected_w0,
                snapshot_root=args.output_root / "snapshots",
            )
            trajectory_rows.append(row)
            trajectory_path = (
                args.output_root
                / "trajectories"
                / arm.value.replace("±", "pm").replace("+", "plus")
                / f"h-{h:g}.json"
            )
            trajectory_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _atomic_write_once(trajectory_path, row)
            if arm is P4TargetArm.POSITIVE and h == H_GRID[0]:
                first = row["prefix_rows"][0]
                stages.record("first_valid_prefix_gate", {
                    "alias": args.model,
                    "arm": arm.value,
                    "h": h,
                    "M": 1,
                    "finite": first["finite"],
                    "clamp_fraction": first["clamp_fraction"],
                    "target_sha256": first["target_sha256"],
                    "W0_pointer_bytes_unchanged": True,
                    "optimizer": "NONE",
                    "adam_state_count": 0,
                    "parameter_gradient_count": 0,
                    "loss_backward_count": 0,
                    "prefix_observation_decision_influence_count": 0,
                })
    if (
        _model_w0_contract(touched) != expected_w0
        or _parameter_inventory_sha256(model) != inventory_w0
        or any(parameter.grad is not None for parameter in model.parameters())
    ):
        raise ODEBFStateError("P4 Euler calibration terminal W0 freeze differs")
    terminal: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-calibration-stage1-terminal/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "status": "STAGE1_MODEL_CELL_TERMINAL",
        "source_head": args.source_head,
        "alias": args.model,
        "case_index": 1,
        "calibration_label": "PERMANENT_CALIBRATION_ONLY",
        "confirmatory_eligibility": False,
        "request_count": BATCH_SIZE,
        "request_order_sha256": request_order,
        "context_sha256": context_sha,
        "h_grid": list(H_GRID),
        "prefix_M": list(PREFIXES),
        "trajectory_count": len(trajectory_rows),
        "executed_microstep_count": 80,
        "actual_autograd_grad_call_count": 80,
        "duplicate_autograd_evaluation_count": 0,
        "trajectory_identity_sha256": [row["identity_sha256"] for row in trajectory_rows],
        "W0_pointer_bytes_sha256": expected_w0,
        "W0_restored": True,
        "optimizer": "NONE",
        "adam_state_count": 0,
        "parameter_gradient_count": 0,
        "loss_backward_count": 0,
        "writer_materialization_count": 0,
        "cache_append_count": 0,
        "heldout_access_count": 0,
        "native_access_count": 0,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "full_fp32": fp32,
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(args.output_root / "terminal.json", terminal)
    return {
        "status": terminal["status"],
        "alias": args.model,
        "terminal_sha256": terminal_sha,
        "identity_sha256": terminal["identity_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", choices=MODEL_ALIASES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stream-root", type=Path, required=True)
    parser.add_argument("--final-pre-gpu-receipt", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage1")
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        traceback.print_exc()
        failure = {
            "schema": f"ode-edit-s05-p4-euler-calibration-{args.stage}-failure/v1",
            "instruction_id": P4_EULER_INSTRUCTION_ID,
            "alias": args.model,
            "exception_class": type(error).__name__,
            "exception_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "technical_retry_count": 0,
            "scientific_delta_count": 0,
        }
        failure["identity_sha256"] = canonical_hash(failure)
        try:
            _atomic_write_once(args.output_root / "failure.json", failure)
        except (FileExistsError, OSError):
            pass
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
