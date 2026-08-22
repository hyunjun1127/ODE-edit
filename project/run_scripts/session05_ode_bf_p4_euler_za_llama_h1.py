#!/usr/bin/env python3
"""Llama-only P4-Euler ZA h=1,M=5,Tz=5 over one sealed B2-B10 slice."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
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

from project.run_scripts.ode_bf.alpha_backend import (
    _normalize_requests,
    fresh_contexts_twice,
    seed_all,
)
from project.run_scripts.ode_bf.contracts import (
    BATCH_SIZE,
    COMMON_SEED,
    ODEBFContractError,
    ODEBFStateError,
    canonical_hash,
)
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p1_runtime import P1StageRecorder, _atomic_write_once
from project.run_scripts.ode_bf.p1_scalable_batched_experiment import (
    _action_frozen_cases,
    _model_w0_contract,
)
from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    build_p1r24_kl_plan,
    evaluate_p1r24_kl,
    verify_p1r24_alphaedit_geometry,
)
from project.run_scripts.ode_bf.p1r24_independent_b10x10_selection import (
    load_historical_h0_batches,
)
from project.run_scripts.ode_bf.p2r1_target_only_runtime import (
    _evaluate_terminal_z_panel,
)
from project.run_scripts.ode_bf.p4_euler_binding import (
    build_euler_nonsemantic_identity,
    build_p4_euler_objective_callback,
)
from project.run_scripts.ode_bf.p4_euler_integrator import (
    P4_EULER_INSTRUCTION_ID,
    P4_EULER_METHOD_ID,
    run_raw_projected_euler,
)
from project.run_scripts.ode_bf.p4_euler_model_binding import (
    evaluate_checkpointed_model_terms,
    observe_model_terms_no_grad,
)
from project.run_scripts.ode_bf.p4_euler_native_reference import (
    build_native_source_contract,
    verify_native_execution_boundary,
)
from project.run_scripts.ode_bf.p4_euler_orchestration import (
    build_za_phase_contract,
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


MODEL_ALIAS = "llama3-8b-inst"
CASE_INDICES = tuple(range(2, 11))
SELECTED_H = 1.0
SELECTED_M = 5
SELECTED_TARGET_HORIZON = 5.0
POLICY = "USER_DIRECTED_POST_CALIBRATION_HYPERPARAMETER_REVISION"
INTERPRETATION = "EXPLORATORY_CONFIRMATION_AFTER_CALIBRATION"
STREAM_SEAL_RELATIVE = Path("canonical/p1r24_independent_b10x10_stream_seal.json")
STREAM_IDENTITY_RELATIVE = Path("canonical/stream-order-context-identity.json")
HF_SEAL = REPO_ROOT / "agents/server4/p4-hf-consumed-closure-seal.json"
DATASET = EASYEDIT_ROOT / "data/counterfact/counterfact.json"
HPARAMS = EASYEDIT_ROOT / "hparams/AlphaEdit/llama3-8b.yaml"
PROJECTOR = EASYEDIT_ROOT / (
    "examples/null_space_project_Meta-Llama-3-8B-Instruct.pt"
)
EXECUTION_MICROBATCH = 1


def _load_regular_json(path: Path) -> dict[str, Any]:
    observed = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise ODEBFContractError("P4 Euler ZA JSON input differs")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ODEBFContractError("P4 Euler ZA JSON object differs")
    return value


def _load_final_preflight(
    path: Path, *, source_head: str, case_index: int
) -> dict[str, Any]:
    value = _load_regular_json(path)
    payload = dict(value)
    identity = payload.pop("identity_sha256", None)
    binding = value.get("model_input_binding")
    execution = value.get("execution_binding")
    case_binding = (
        binding.get("case_bindings", {}).get(f"B{case_index}")
        if isinstance(binding, Mapping)
        else None
    )
    if (
        value.get("schema")
        != "ode-edit-s05-p4-euler-za-llama-h1-b2b10-final-pre-gpu/v1"
        or value.get("instruction_id") != P4_EULER_INSTRUCTION_ID
        or value.get("status") != "FINAL_PRE_GPU_PASS"
        or value.get("model_load_authorized") is not True
        or value.get("source_head") != source_head
        or identity != canonical_hash(payload)
        or not isinstance(binding, Mapping)
        or binding.get("alias") != MODEL_ALIAS
        or binding.get("B1_excluded") is not True
        or not isinstance(case_binding, Mapping)
        or case_binding.get("case_index") != case_index
        or not isinstance(execution, Mapping)
        or execution.get("policy_provenance") != POLICY
        or execution.get("interpretation") != INTERPRETATION
        or execution.get("h") != SELECTED_H
        or execution.get("M") != SELECTED_M
        or execution.get("T_z") != SELECTED_TARGET_HORIZON
        or execution.get("case_indices") != list(CASE_INDICES)
        or execution.get("writer_materialization_count") != 0
        or execution.get("cache_append_count") != 0
        or execution.get("outer_k8_repeat_count") != 0
        or value.get("project_gpu_cap") != 2
    ):
        raise ODEBFContractError("P4 Euler ZA final preflight differs")
    return value


def _evaluation_row(
    identity: Mapping[str, Any], *, case_index: int
) -> Mapping[str, Any]:
    rows = identity.get("evaluation_context_identities")
    rows = rows.get("rows") if isinstance(rows, Mapping) else None
    matches = [
        row
        for row in rows or []
        if isinstance(row, Mapping)
        and row.get("model_alias") == MODEL_ALIAS
        and row.get("case_index") == case_index
    ]
    if len(matches) != 1:
        raise ODEBFContractError("P4 Euler ZA evaluator row differs")
    return matches[0]


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
        raise ODEBFContractError("P4 Euler ZA FULL-FP32 differs")
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


def _touched_weights(
    model: torch.nn.Module, hparams: Any
) -> dict[str, torch.nn.Parameter]:
    result = {
        f"{hparams.rewrite_module_tmp.format(int(layer))}.weight": model.get_parameter(
            f"{hparams.rewrite_module_tmp.format(int(layer))}.weight"
        )
        for layer in hparams.layers
    }
    if not result or any(value.dtype is not torch.float32 for value in result.values()):
        raise ODEBFContractError("P4 Euler ZA W0 inventory differs")
    return result


def _summary(values: Sequence[float]) -> Mapping[str, float]:
    tensor = torch.tensor(tuple(values), dtype=torch.float32)
    return {
        "mean": float(torch.mean(tensor)),
        "median": float(torch.quantile(tensor, 0.5)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "max": float(torch.max(tensor)),
    }


def _target_arm(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    case_index: int,
    arm: P4TargetArm,
    root: Path,
    hparams: Any,
    paired: Any,
    physical: Any,
    initial: Any,
    kl_plan: Any,
    teacher: Sequence[torch.Tensor],
    lock: P1R24AliasTargetLock,
    nonsemantic: Mapping[str, Any],
    touched: Mapping[str, torch.nn.Parameter],
    expected_w0: str,
    inventory_w0: str,
) -> Mapping[str, Any]:
    started = time.perf_counter()
    origin = initial.target_z.detach().clone().to(dtype=torch.float32)
    current_terminal = initial.current_terminal_z.detach().clone().to(dtype=torch.float32)
    radius = lock.clamp_factor * torch.linalg.vector_norm(origin, dim=0)
    target_layer_name = hparams.layer_module_tmp.format(int(hparams.layers[-1]))

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
        terms,
        arm=arm,
        kl_factor=lock.kl_factor,
        decay_factor=lock.decay_factor,
    )
    inventory_before = _parameter_inventory_sha256(model)
    result = run_raw_projected_euler(
        origin,
        origin=origin,
        radius_by_request=radius,
        target_horizon=SELECTED_TARGET_HORIZON,
        microsteps=SELECTED_M,
        objective_callback=objective,
        frozen_content_sha256=lambda: _parameter_inventory_sha256(model),
        outer_step_index=None,
    )
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
    step_files: list[str] = []
    for row in result.step_receipts:
        step_files.append(
            _atomic_write_once(
                root
                / "raw"
                / "target"
                / f"microstep-{int(row['microstep_index']):02d}.json",
                row,
            )
        )
    hits_by_request = [
        sum(int(step["clamp_hit_by_request"][ordinal]) for step in result.step_receipts)
        for ordinal in range(BATCH_SIZE)
    ]
    hit_count = sum(hits_by_request)
    denominator = BATCH_SIZE * SELECTED_M
    phase = build_za_phase_contract(arm=arm.value, local_solve_count=1)
    inventory_after = _parameter_inventory_sha256(model)
    if (
        inventory_after != inventory_before
        or inventory_after != inventory_w0
        or _model_w0_contract(touched) != expected_w0
        or any(parameter.grad is not None for parameter in model.parameters())
        or not bool(torch.isfinite(endpoint).all())
    ):
        raise ODEBFStateError("P4 Euler ZA target/W0/finite gate differs")
    action: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-za-h1-action-freeze/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "method_id": P4_EULER_METHOD_ID,
        "policy_provenance": POLICY,
        "interpretation": INTERPRETATION,
        "alias": MODEL_ALIAS,
        "case_index": case_index,
        "arm": arm.value,
        "request_order_sha256": paired.new.request_order_sha256,
        "h": SELECTED_H,
        "M": SELECTED_M,
        "T_z": SELECTED_TARGET_HORIZON,
        "phase_contract": phase,
        "nonsemantic_identity_sha256": nonsemantic["identity_sha256"],
        "microstep_receipt_file_sha256": step_files,
        "microstep_identity_sha256": [
            row["identity_sha256"] for row in result.step_receipts
        ],
        "integrator_receipt": result.receipt,
        "terminal_target_sha256": tensor_sha256(endpoint),
        "endpoint_observation": observation,
        "clamp_hit_by_request_count": hits_by_request,
        "clamp_hit_count": hit_count,
        "clamp_denominator": denominator,
        "clamp_fraction": hit_count / denominator,
        "W0_pointer_version_inventory_entry": inventory_before,
        "W0_pointer_version_inventory_exit": inventory_after,
        "W0_pointer_version_change_count": 0,
        "W0_bytes_change_count": 0,
        "optimizer": "NONE",
        "optimizer_object_count": 0,
        "adam_state_count": 0,
        "sgd_state_count": 0,
        "loss_backward_count": 0,
        "parameter_gradient_count": 0,
        "writer_materialization_count": 0,
        "cache_append_count": 0,
        "outer_k8_repeat_count": 0,
        "heldout_access_before_action_freeze_count": 0,
        "high_clamp_fraction_stop_rule_count": 0,
        "finite": True,
    }
    action["identity_sha256"] = canonical_hash(action)
    action_file_sha = _atomic_write_once(root / "action-freeze.json", action)
    cases, freeze_receipt = _action_frozen_cases(
        DATASET,
        requests,
        arm=f"P4-EULER-ZA-H1-{arm.value}",
        selected_snapshot_sha256=action_file_sha,
        fixed_budget_slots_completed=0,
    )
    panel, evaluator_wall = _evaluate_terminal_z_panel(
        model,
        tokenizer,
        requests,
        cases,
        alias=MODEL_ALIAS,
        request_order_sha256=paired.new.request_order_sha256,
        action_sha256=action["identity_sha256"],
        hparams=hparams,
        terminal_target=endpoint,
        terminal_physical=physical,
        method=f"P4-EULER-ZA-H1-{arm.value}",
        instruction_id=P4_EULER_INSTRUCTION_ID,
        method_id=P4_EULER_METHOD_ID,
        schema="ode-edit-s05-p4-euler-za-h1-terminal-z-panel/v1",
        trajectory_status=f"ACTION_FROZEN_P4_EULER_ZA_H1_{arm.value}",
        freeze_snapshot_index=0,
        freeze_accepted_snapshot_count=1,
    )
    if (
        _model_w0_contract(touched) != expected_w0
        or _parameter_inventory_sha256(model) != inventory_w0
        or any(parameter.grad is not None for parameter in model.parameters())
    ):
        raise ODEBFStateError("P4 Euler ZA evaluator mutated W0")
    terminal: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-za-h1-target-arm-terminal/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "method_id": P4_EULER_METHOD_ID,
        "policy_provenance": POLICY,
        "interpretation": INTERPRETATION,
        "alias": MODEL_ALIAS,
        "case_index": case_index,
        "arm": arm.value,
        "request_count": BATCH_SIZE,
        "request_order_sha256": paired.new.request_order_sha256,
        "h": SELECTED_H,
        "M": SELECTED_M,
        "T_z": SELECTED_TARGET_HORIZON,
        "action_freeze_file_sha256": action_file_sha,
        "terminal_target_sha256": tensor_sha256(endpoint),
        "endpoint_new_nll": _summary(observation["new_nll_by_request"]),
        "endpoint_true_nll": _summary(observation["old_nll_by_request"]),
        "endpoint_margin": _summary(
            observation["new_minus_old_margin_by_request"]
        ),
        "clamp_hit_by_request_count": hits_by_request,
        "clamp_hit_count": hit_count,
        "clamp_denominator": denominator,
        "clamp_fraction": hit_count / denominator,
        "terminal_z_panel": panel,
        "evaluator_action_freeze": freeze_receipt,
        "target_wall_seconds": result.receipt["wall_time_seconds"],
        "evaluator_wall_seconds": evaluator_wall,
        "total_wall_seconds": time.perf_counter() - started,
        "logical_field_evaluation_count": SELECTED_M,
        "actual_autograd_grad_call_count": SELECTED_M,
        "duplicate_evaluation_count": 0,
        "W0_restored": True,
        "finite": True,
        "writer_materialization_count": 0,
        "cache_append_count": 0,
        "outer_k8_repeat_count": 0,
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    _atomic_write_once(root / "terminal.json", terminal)
    return terminal


def _native_arm(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    case_index: int,
    root: Path,
    hparams: Any,
    contexts: Sequence[Sequence[str]],
    physical: Any,
    request_order: str,
    touched: Mapping[str, torch.nn.Parameter],
    expected_w0: str,
    inventory_w0: str,
) -> Mapping[str, Any]:
    from easyeditor.models.alphaedit.compute_z import compute_z

    started = time.perf_counter()
    source_contract = verify_native_execution_boundary(
        build_native_source_contract(EASYEDIT_ROOT, method="native-alphaedit")
    )
    normalized = _normalize_requests(requests)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        values = [
            compute_z(
                model,
                tokenizer,
                request,
                hparams,
                int(hparams.layers[-1]),
                list(contexts),
            )
            for request in normalized
        ]
    target = torch.stack(
        [value.detach().to(dtype=torch.float32) for value in values], dim=1
    )
    if (
        len(values) != BATCH_SIZE
        or target.dtype is not torch.float32
        or not bool(torch.isfinite(target).all())
        or _model_w0_contract(touched) != expected_w0
        or _parameter_inventory_sha256(model) != inventory_w0
    ):
        raise ODEBFStateError("P4 Euler ZA Native-Z target/W0 differs")
    action: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-za-native-z-action-freeze/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "method_id": P4_EULER_METHOD_ID,
        "policy_provenance": POLICY,
        "interpretation": INTERPRETATION,
        "alias": MODEL_ALIAS,
        "case_index": case_index,
        "arm": "Native-Z",
        "request_order_sha256": request_order,
        "terminal_target_sha256": tensor_sha256(target),
        "native_source_contract": source_contract,
        "native_compute_z_call_count": BATCH_SIZE,
        "native_selection_influence_count": 0,
        "writer_materialization_count": 0,
        "cache_append_count": 0,
        "W0_sha256": expected_w0,
        "heldout_access_before_action_freeze_count": 0,
    }
    action["identity_sha256"] = canonical_hash(action)
    action_file_sha = _atomic_write_once(root / "action-freeze.json", action)
    cases, freeze_receipt = _action_frozen_cases(
        DATASET,
        requests,
        arm="P4-EULER-ZA-NATIVE-Z",
        selected_snapshot_sha256=action_file_sha,
        fixed_budget_slots_completed=0,
    )
    panel, evaluator_wall = _evaluate_terminal_z_panel(
        model,
        tokenizer,
        requests,
        cases,
        alias=MODEL_ALIAS,
        request_order_sha256=request_order,
        action_sha256=action["identity_sha256"],
        hparams=hparams,
        terminal_target=target,
        terminal_physical=physical,
        method="P4-EULER-ZA-NATIVE-Z",
        instruction_id=P4_EULER_INSTRUCTION_ID,
        method_id=P4_EULER_METHOD_ID,
        schema="ode-edit-s05-p4-euler-za-native-z-terminal-panel/v1",
        trajectory_status="ACTION_FROZEN_P4_EULER_ZA_NATIVE_Z",
        freeze_snapshot_index=0,
        freeze_accepted_snapshot_count=1,
        native_endpoint_runtime_access_count=1,
    )
    if (
        _model_w0_contract(touched) != expected_w0
        or _parameter_inventory_sha256(model) != inventory_w0
    ):
        raise ODEBFStateError("P4 Euler ZA Native-Z evaluator mutated W0")
    terminal: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-za-native-z-terminal/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "method_id": P4_EULER_METHOD_ID,
        "policy_provenance": POLICY,
        "interpretation": INTERPRETATION,
        "alias": MODEL_ALIAS,
        "case_index": case_index,
        "arm": "Native-Z",
        "request_count": BATCH_SIZE,
        "request_order_sha256": request_order,
        "terminal_target_sha256": tensor_sha256(target),
        "action_freeze_file_sha256": action_file_sha,
        "terminal_z_panel": panel,
        "evaluator_action_freeze": freeze_receipt,
        "native_source_contract": source_contract,
        "native_compute_z_call_count": BATCH_SIZE,
        "native_selection_influence_count": 0,
        "evaluator_wall_seconds": evaluator_wall,
        "total_wall_seconds": time.perf_counter() - started,
        "W0_restored": True,
        "finite": True,
        "writer_materialization_count": 0,
        "cache_append_count": 0,
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    _atomic_write_once(root / "terminal.json", terminal)
    return terminal


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if (
        args.model != MODEL_ALIAS
        or args.case_index not in CASE_INDICES
        or os.environ.get("PROJECT_GPU_CAP") != "2"
        or any(
            os.environ.get(name) != "1"
            for name in (
                "HF_DATASETS_OFFLINE",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            )
        )
    ):
        raise ODEBFContractError("P4 Euler ZA model/case/cap/offline differs")
    observed_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if observed_head != args.source_head:
        raise ODEBFContractError("P4 Euler ZA source HEAD differs")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError("P4 Euler ZA output root is create-once")
    os.umask(0o077)
    args.output_root.mkdir(mode=0o700, parents=True)
    stages = P1StageRecorder(args.output_root / "stages")
    final = _load_final_preflight(
        args.final_pre_gpu_receipt,
        source_head=args.source_head,
        case_index=args.case_index,
    )
    binding = final["model_input_binding"]
    case_binding = binding["case_bindings"][f"B{args.case_index}"]
    stages.record(
        "pre_model_final_gate",
        {
            "final_pre_gpu_identity": final["identity_sha256"],
            "policy_provenance": POLICY,
            "interpretation": INTERPRETATION,
            "alias": MODEL_ALIAS,
            "case_binding": case_binding,
            "h": SELECTED_H,
            "M": SELECTED_M,
            "T_z": SELECTED_TARGET_HORIZON,
            "B1_excluded": True,
        },
    )

    seed_all(COMMON_SEED)
    hf_seal = load_p4_hf_consumed_closure_seal(HF_SEAL, repo_root=REPO_ROOT)
    model, tokenizer, hf_receipt = load_p4_full_fp32_from_sealed_snapshot(
        hf_seal,
        MODEL_ALIAS,
        stream_receipt=final["transfer_receipt"],
        final_pre_gpu_receipt=final,
        reuse_final_verified_closure=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams

    hparams = AlphaEditHyperParams.from_hparams(str(HPARAMS))
    hparams.device = 0
    hparams.P_loc = str(PROJECTOR)
    hparams.stats_dir = str(EASYEDIT_ROOT / "examples/data/stats")
    model.config._name_or_path = hparams.model_name
    fp32 = _assert_full_fp32(model)
    stages.record(
        "post_model_full_fp32_offline",
        {
            "hf": hf_receipt.to_dict(),
            "full_fp32": fp32,
            "offline": True,
            "local_files_only": True,
            "qwen_load_count": 0,
        },
    )

    stream_root = args.stream_root.resolve(strict=True)
    stream = _load_regular_json(stream_root / STREAM_SEAL_RELATIVE)
    identity = _load_regular_json(stream_root / STREAM_IDENTITY_RELATIVE)
    batches = load_historical_h0_batches(DATASET, stream)
    requests = batches[args.case_index - 1]
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    if (
        len(batches) != 10
        or len(requests) != BATCH_SIZE
        or request_order != case_binding["request_order_sha256"]
    ):
        raise ODEBFContractError("P4 Euler ZA B2-B10 membership/order differs")
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
    transferred = _evaluation_row(identity, case_index=args.case_index)
    if (
        paired.new.request_order_sha256 != request_order
        or capture_plan.request_order_sha256 != request_order
        or transferred["request_order_sha256"] != request_order
        or transferred["evaluation_case_identity_sha256"]
        != case_binding["evaluation_case_identity_sha256"]
    ):
        raise ODEBFContractError("P4 Euler ZA runtime/evaluator binding differs")
    stages.record(
        "post_sealed_input_plan",
        {
            "alias": MODEL_ALIAS,
            "case_index": args.case_index,
            "request_order_sha256": request_order,
            "context_sha256": context_sha,
            "execution_objective_plan_sha256": paired.new.identity_sha256,
            "execution_capture_plan_sha256": capture_plan.identity_sha256,
            "transferred_objective_plan_sha256_observation_only": transferred[
                "objective_plan_sha256"
            ],
            "transferred_capture_plan_sha256_observation_only": transferred[
                "capture_plan_sha256"
            ],
            "transferred_plan_decision_influence_count": 0,
            "execution_microbatch_size": EXECUTION_MICROBATCH,
        },
    )

    touched = _touched_weights(model, hparams)
    expected_w0 = _model_w0_contract(touched)
    inventory_w0 = _parameter_inventory_sha256(model)
    physical = capture_scalable_physical_state(model, capture_plan, hparams)
    initial = initial_target_from_capture(physical)
    lock = P1R24AliasTargetLock.for_alias(MODEL_ALIAS)
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
    teacher_sha = canonical_hash(
        {
            "teacher_tensor_sha256": [tensor_sha256(value) for value in teacher],
            "request_order_sha256": request_order,
            "capture_count": 1,
        }
    )
    origin = initial.target_z.detach().to(dtype=torch.float32)
    radius = lock.clamp_factor * torch.linalg.vector_norm(origin, dim=0)
    nonsemantic = build_euler_nonsemantic_identity(
        request_order_sha256=request_order,
        context_sha256=context_sha,
        teacher_sha256=teacher_sha,
        origin_sha256=tensor_sha256(origin),
        radius_sha256=tensor_sha256(radius),
        target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
        kl_factor=lock.kl_factor,
        decay_factor=lock.decay_factor,
        clamp_factor=lock.clamp_factor,
        target_horizon=SELECTED_TARGET_HORIZON,
        microsteps=SELECTED_M,
    )
    stages.record(
        "post_shared_W0_nonsemantic_gate",
        {
            "W0_sha256": expected_w0,
            "model_inventory_sha256": inventory_w0,
            "nonsemantic_identity": nonsemantic,
            "alpha_geometry": alpha_geometry,
            "teacher": teacher_result.raw_free_payload(),
            "teacher_sha256": teacher_sha,
            "writer_materialization_count": 0,
            "cache_append_count": 0,
            "outer_k8_repeat_count": 0,
        },
    )
    torch.cuda.reset_peak_memory_stats()

    target_results: list[Mapping[str, Any]] = []
    for arm in (P4TargetArm.POSITIVE, P4TargetArm.POSITIVE_NEGATIVE):
        if (
            _model_w0_contract(touched) != expected_w0
            or _parameter_inventory_sha256(model) != inventory_w0
        ):
            raise ODEBFStateError("P4 Euler ZA same-W0 entry differs")
        target_results.append(
            _target_arm(
                model,
                tokenizer,
                requests,
                case_index=args.case_index,
                arm=arm,
                root=args.output_root
                / ("z-plus" if arm is P4TargetArm.POSITIVE else "z-plus-minus"),
                hparams=hparams,
                paired=paired,
                physical=physical,
                initial=initial,
                kl_plan=kl_plan,
                teacher=teacher,
                lock=lock,
                nonsemantic=nonsemantic,
                touched=touched,
                expected_w0=expected_w0,
                inventory_w0=inventory_w0,
            )
        )
    stages.record(
        "first_valid_slice_gate",
        {
            "alias": MODEL_ALIAS,
            "case_index": args.case_index,
            "h": SELECTED_H,
            "M": SELECTED_M,
            "T_z": SELECTED_TARGET_HORIZON,
            "arms": [row["arm"] for row in target_results],
            "actual_autograd_grad_call_count": [
                row["actual_autograd_grad_call_count"] for row in target_results
            ],
            "duplicate_evaluation_count": [
                row["duplicate_evaluation_count"] for row in target_results
            ],
            "finite": [row["finite"] for row in target_results],
            "clamp": [
                {
                    "arm": row["arm"],
                    "numerator": row["clamp_hit_count"],
                    "denominator": row["clamp_denominator"],
                    "fraction": row["clamp_fraction"],
                    "by_request_count": row["clamp_hit_by_request_count"],
                }
                for row in target_results
            ],
            "endpoint_new_nll": [
                {"arm": row["arm"], **row["endpoint_new_nll"]}
                for row in target_results
            ],
            "endpoint_true_nll": [
                {"arm": row["arm"], **row["endpoint_true_nll"]}
                for row in target_results
            ],
            "endpoint_margin": [
                {"arm": row["arm"], **row["endpoint_margin"]}
                for row in target_results
            ],
            "W0_pointer_version_change_count": 0,
            "W0_bytes_change_count": 0,
            "W0_restored": True,
            "optimizer": "NONE",
            "adam_state_count": 0,
            "loss_backward_count": 0,
            "parameter_gradient_count": 0,
            "high_clamp_fraction_stop_rule_count": 0,
        },
    )

    native = _native_arm(
        model,
        tokenizer,
        requests,
        case_index=args.case_index,
        root=args.output_root / "native-z",
        hparams=hparams,
        contexts=contexts,
        physical=physical,
        request_order=request_order,
        touched=touched,
        expected_w0=expected_w0,
        inventory_w0=inventory_w0,
    )
    if (
        _model_w0_contract(touched) != expected_w0
        or _parameter_inventory_sha256(model) != inventory_w0
        or any(parameter.grad is not None for parameter in model.parameters())
    ):
        raise ODEBFStateError("P4 Euler ZA terminal W0 differs")
    arms = [*target_results, native]
    terminal: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-za-llama-h1-case-terminal/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "method_id": P4_EULER_METHOD_ID,
        "policy_provenance": POLICY,
        "interpretation": INTERPRETATION,
        "status": "P4_EULER_ZA_LLAMA_H1_CASE_TERMINAL",
        "source_head": args.source_head,
        "alias": MODEL_ALIAS,
        "case_index": args.case_index,
        "request_count": BATCH_SIZE,
        "request_order_sha256": request_order,
        "h": SELECTED_H,
        "M": SELECTED_M,
        "T_z": SELECTED_TARGET_HORIZON,
        "arms": [row["arm"] for row in arms],
        "arm_terminal_identity_sha256": [row["identity_sha256"] for row in arms],
        "same_W0": True,
        "W0_pointer_bytes_sha256": expected_w0,
        "W0_inventory_sha256": inventory_w0,
        "W0_restored": True,
        "finite": True,
        "full_fp32": fp32,
        "target_arm_autograd_grad_call_count": 2 * SELECTED_M,
        "duplicate_evaluation_count": 0,
        "optimizer": "NONE",
        "adam_state_count": 0,
        "loss_backward_count": 0,
        "parameter_gradient_count": 0,
        "writer_materialization_count": 0,
        "cache_append_count": 0,
        "outer_k8_repeat_count": 0,
        "native_compute_z_call_count": BATCH_SIZE,
        "native_selection_influence_count": 0,
        "heldout_terminal_only": True,
        "high_clamp_fraction_stop_rule_count": 0,
        "qwen_submission_count": 0,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(args.output_root / "terminal.json", terminal)
    return {
        "status": terminal["status"],
        "alias": MODEL_ALIAS,
        "case_index": args.case_index,
        "terminal_sha256": terminal_sha,
        "identity_sha256": terminal["identity_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", choices=(MODEL_ALIAS,), required=True)
    parser.add_argument("--case-index", type=int, choices=CASE_INDICES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stream-root", type=Path, required=True)
    parser.add_argument("--final-pre-gpu-receipt", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        traceback.print_exc()
        failure = {
            "schema": "ode-edit-s05-p4-euler-za-llama-h1-case-failure/v1",
            "instruction_id": P4_EULER_INSTRUCTION_ID,
            "policy_provenance": POLICY,
            "interpretation": INTERPRETATION,
            "alias": args.model,
            "case_index": args.case_index,
            "h": SELECTED_H,
            "M": SELECTED_M,
            "T_z": SELECTED_TARGET_HORIZON,
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
