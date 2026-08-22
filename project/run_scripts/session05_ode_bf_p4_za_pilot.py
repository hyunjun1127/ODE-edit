#!/usr/bin/env python3
"""P4 ZA case01 target-only pilot: Z+, Z±, and Native-Z at shared W0."""

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
    MODEL_ALIASES,
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
from project.run_scripts.ode_bf.p4_fixed_target_solver import (
    P4SolverEvaluation,
    run_fixed_m_target_adam,
)
from project.run_scripts.ode_bf.p4_hf_consumed_closure import (
    load_p4_full_fp32_from_sealed_snapshot,
    load_p4_hf_consumed_closure_seal,
)
from project.run_scripts.ode_bf.p4_semantic_barrier import (
    P4_INSTRUCTION_ID,
    P4_METHOD_ID,
    P4TargetArm,
)
from project.run_scripts.ode_bf.p4_target_solver_binding import (
    build_p4_paired_objective_plans,
    evaluate_p4_target_objective,
    non_barrier_arm_identity,
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


M5_RUN_TOKEN = "p4-za-case01-server4-v1"
M1_RUN_TOKEN = "p4-za-m1-case01-server4-v1"
RUN_TOKEN_TO_INNER_ITERATIONS = {M5_RUN_TOKEN: 5, M1_RUN_TOKEN: 1}
STREAM_SEAL_RELATIVE = Path("canonical/p1r24_independent_b10x10_stream_seal.json")
STREAM_IDENTITY_RELATIVE = Path("canonical/stream-order-context-identity.json")
HF_SEAL = REPO_ROOT / "agents/server4/p4-hf-consumed-closure-seal.json"
DATASET = EASYEDIT_ROOT / "data/counterfact/counterfact.json"
HPARAMS = {
    "llama3-8b-inst": EASYEDIT_ROOT / "hparams/AlphaEdit/llama3-8b.yaml",
    "qwen2.5-7b-inst": EASYEDIT_ROOT / "hparams/AlphaEdit/qwen2.5-7b.yaml",
}
EXECUTION_MICROBATCH = 1


def _load_json_regular(path: Path) -> dict[str, object]:
    observed = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise ODEBFContractError("P4 ZA JSON input is not regular/non-symlink")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ODEBFContractError("P4 ZA JSON input is not an object")
    return value


def _load_final_receipt(
    path: Path,
    *,
    alias: str,
    source_head: str,
    inner_iterations: int,
) -> dict[str, object]:
    value = _load_json_regular(path)
    identity = value.get("identity_sha256")
    payload = dict(value)
    payload.pop("identity_sha256", None)
    binding = value.get("model_input_binding")
    execution = value.get("execution_binding")
    if (
        value.get("schema") != "ode-edit-s05-p4-za-server4-final-pre-gpu/v1"
        or value.get("status") != "FINAL_PRE_GPU_PASS"
        or value.get("model_load_authorized") is not True
        or value.get("source_head") != source_head
        or identity != canonical_hash(payload)
        or not isinstance(binding, Mapping)
        or alias not in binding
        or not isinstance(execution, Mapping)
        or execution.get("inner_iterations") != inner_iterations
        or execution.get("selected_final_observation") != (inner_iterations == 1)
        or value.get("project_gpu_cap") != 2
    ):
        raise ODEBFContractError("P4 ZA final PRE-GPU receipt differs")
    return value


def _assert_full_fp32(model: torch.nn.Module) -> dict[str, object]:
    floating = [item for item in model.parameters() if item.is_floating_point()]
    if (
        not floating
        or any(item.dtype is not torch.float32 for item in floating)
        or any(item.requires_grad for item in floating)
        or torch.is_autocast_enabled()
        or torch.is_autocast_enabled("cpu")
        or next(model.parameters()).device != torch.device("cuda:0")
    ):
        raise ODEBFContractError("P4 ZA FULL-FP32 runtime differs")
    payload: dict[str, object] = {
        "dtype": "torch.float32",
        "floating_parameter_count": len(floating),
        "requires_grad_parameter_count": 0,
        "autocast_enabled": False,
        "cpu_autocast_enabled": False,
        "quantization": False,
        "device": "cuda:0",
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _touched_weights(model: torch.nn.Module, hparams: Any) -> dict[str, torch.nn.Parameter]:
    result = {
        f"{hparams.rewrite_module_tmp.format(int(layer))}.weight": model.get_parameter(
            f"{hparams.rewrite_module_tmp.format(int(layer))}.weight"
        )
        for layer in hparams.layers
    }
    if not result or any(value.dtype is not torch.float32 for value in result.values()):
        raise ODEBFContractError("P4 ZA touched W0 inventory differs")
    return result


def _evaluation_row(identity: Mapping[str, object], alias: str) -> Mapping[str, object]:
    rows = identity.get("evaluation_context_identities")
    rows = rows.get("rows") if isinstance(rows, Mapping) else None
    matches = [
        row for row in rows or []
        if isinstance(row, Mapping)
        and row.get("model_alias") == alias
        and row.get("case_index") == 1
    ]
    if len(matches) != 1:
        raise ODEBFContractError("P4 ZA transferred evaluator row differs")
    return matches[0]


def _target_arm(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    arm: P4TargetArm,
    root: Path,
    hparams: Any,
    paired: Any,
    physical: Any,
    initial: Any,
    kl_plan: Any,
    teacher: Sequence[torch.Tensor],
    teacher_sha: str,
    lock: P1R24AliasTargetLock,
    non_barrier: Mapping[str, Any],
    dataset_path: Path,
    touched: Mapping[str, torch.nn.Parameter],
    expected_w0: str,
    stages: P1StageRecorder,
    inner_iterations: int,
) -> dict[str, object]:
    started = time.perf_counter()
    current = initial.target_z.detach().clone().to(dtype=torch.float32)
    origin = initial.target_z.detach().clone().to(dtype=torch.float32)
    terminal = initial.current_terminal_z.detach().clone().to(dtype=torch.float32)
    outer_receipts: list[str] = []
    outer_identities: list[str] = []
    for outer_step in range(8):
        def evaluate(candidate: torch.Tensor, _inner: int) -> P4SolverEvaluation:
            variable = candidate.detach().clone().requires_grad_(True)
            return evaluate_p4_target_objective(
                model,
                paired,
                kl_plan,
                teacher,
                target_state=variable,
                current_terminal=terminal,
                target_origin=origin,
                target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
                arm=arm,
                kl_factor=lock.kl_factor,
                decay_factor=lock.decay_factor,
                record_gradient_comparison=(inner_iterations == 1),
            )

        solution = run_fixed_m_target_adam(
            current,
            origin,
            evaluate,
            learning_rate=float(hparams.v_lr),
            clamp_factor=lock.clamp_factor,
            outer_step_index=outer_step,
            arm=arm.value,
            paired_input_identity=str(non_barrier["identity_sha256"]),
            inner_iterations=inner_iterations,
            observe_selected_final=(inner_iterations == 1),
        )
        current = solution.final_target.detach().clone()
        receipt = dict(solution.receipt)
        receipt["W0_sha256"] = _model_w0_contract(touched)
        receipt["writer_call_count"] = 0
        receipt["current_terminal_refresh_count"] = 1
        receipt["terminal_state"] = "FROZEN_W0_TARGET_ONLY"
        receipt["identity_sha256"] = canonical_hash(
            {key: value for key, value in receipt.items() if key != "identity_sha256"}
        )
        outer_identities.append(str(receipt["identity_sha256"]))
        outer_receipts.append(
            _atomic_write_once(root / "target" / f"outer-{outer_step:02d}.json", receipt)
        )
        if (
            not bool(torch.isfinite(current).all())
            or _model_w0_contract(touched) != expected_w0
        ):
            raise ODEBFStateError("P4 ZA first/outer target W0 or finite gate failed")
        if arm is P4TargetArm.POSITIVE and outer_step == 0:
            stages.record(
                "first_target_completion_nonfinite_restore_gate",
                {
                    "alias": alias,
                    "arm": arm.value,
                    "outer_step_index": 0,
                    "configured_inner_iterations": inner_iterations,
                    "executed_inner_iterations": inner_iterations,
                    "selected_final_observation_count": (
                        1 if inner_iterations == 1 else 0
                    ),
                    "target_nonfinite_count": 0,
                    "gradient_nonfinite_count": 0,
                    "W0_unchanged": True,
                    "restore_required": False,
                    "W0_restored": True,
                    "writer_call_count": 0,
                },
            )

    action: dict[str, object] = {
        "schema": "ode-edit-s05-p4-za-target-action-freeze/v1",
        "instruction_id": P4_INSTRUCTION_ID,
        "method_id": P4_METHOD_ID,
        "alias": alias,
        "arm": arm.value,
        "case_index": 1,
        "request_order_sha256": paired.new.request_order_sha256,
        "outer_step_count": 8,
        "inner_iterations_per_outer": inner_iterations,
        "inner_iteration_count": 8 * inner_iterations,
        "selected_final_observation_count": (
            8 if inner_iterations == 1 else 0
        ),
        "outer_receipt_sha256": outer_receipts,
        "outer_identity_sha256": outer_identities,
        "terminal_target_sha256": tensor_sha256(current),
        "non_barrier_identity_sha256": non_barrier["identity_sha256"],
        "W0_sha256": _model_w0_contract(touched),
        "writer_call_count": 0,
        "heldout_access_before_freeze_count": 0,
        "retry_count": 0,
    }
    action["identity_sha256"] = canonical_hash(action)
    action_file_sha = _atomic_write_once(root / "action-freeze.json", action)
    cases, _ = _action_frozen_cases(
        dataset_path,
        requests,
        arm=arm.value,
        selected_snapshot_sha256=action_file_sha,
        fixed_budget_slots_completed=8,
    )
    panel, evaluator_wall = _evaluate_terminal_z_panel(
        model,
        tokenizer,
        requests,
        cases,
        alias=alias,
        request_order_sha256=paired.new.request_order_sha256,
        action_sha256=str(action["identity_sha256"]),
        hparams=hparams,
        terminal_target=current,
        terminal_physical=physical,
        method=f"P4-ZA-{arm.value}",
        instruction_id=P4_INSTRUCTION_ID,
        method_id=P4_METHOD_ID,
        schema="ode-edit-s05-p4-za-terminal-z-panel/v1",
        trajectory_status=f"ACTION_FROZEN_P4_ZA_{arm.value}_K8",
    )
    if _model_w0_contract(touched) != expected_w0:
        raise ODEBFStateError("P4 ZA target arm mutated W0")
    result: dict[str, object] = {
        "schema": "ode-edit-s05-p4-za-target-arm-terminal/v1",
        "alias": alias,
        "arm": arm.value,
        "case_index": 1,
        "request_order_sha256": paired.new.request_order_sha256,
        "terminal_target_sha256": tensor_sha256(current),
        "action_freeze_file_sha256": action_file_sha,
        "terminal_z_panel": panel,
        "evaluator_wall_seconds": evaluator_wall,
        "total_wall_seconds": time.perf_counter() - started,
        "outer_step_count": 8,
        "inner_iterations_per_outer": inner_iterations,
        "inner_iteration_count": 8 * inner_iterations,
        "selected_final_observation_count": (
            8 if inner_iterations == 1 else 0
        ),
        "writer_call_count": 0,
        "W0_restored": True,
        "scientific_promotion": False,
    }
    result["identity_sha256"] = canonical_hash(result)
    _atomic_write_once(root / "terminal.json", result)
    return result


def _native_arm(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    root: Path,
    hparams: Any,
    contexts: Sequence[Sequence[str]],
    physical: Any,
    dataset_path: Path,
    touched: Mapping[str, torch.nn.Parameter],
    expected_w0: str,
    request_order: str,
) -> dict[str, object]:
    from easyeditor.models.alphaedit.compute_z import compute_z

    started = time.perf_counter()
    normalized = _normalize_requests(requests)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
    target = torch.stack([value.detach().to(dtype=torch.float32) for value in values], dim=1)
    if (
        len(values) != BATCH_SIZE
        or target.dtype is not torch.float32
        or not bool(torch.isfinite(target).all())
        or _model_w0_contract(touched) != expected_w0
    ):
        raise ODEBFStateError("P4 ZA Native-Z target/W0 differs")
    action: dict[str, object] = {
        "schema": "ode-edit-s05-p4-za-native-action-freeze/v1",
        "instruction_id": P4_INSTRUCTION_ID,
        "method_id": P4_METHOD_ID,
        "alias": alias,
        "arm": "Native-Z",
        "case_index": 1,
        "request_order_sha256": request_order,
        "terminal_target_sha256": tensor_sha256(target),
        "native_compute_z_direct_path": "easyeditor.models.alphaedit.compute_z.compute_z",
        "native_compute_z_call_count": BATCH_SIZE,
        "writer_call_count": 0,
        "W0_sha256": expected_w0,
        "heldout_access_before_freeze_count": 0,
        "retry_count": 0,
    }
    action["identity_sha256"] = canonical_hash(action)
    action_file_sha = _atomic_write_once(root / "action-freeze.json", action)
    cases, _ = _action_frozen_cases(
        dataset_path,
        requests,
        arm="Native-Z",
        selected_snapshot_sha256=action_file_sha,
        fixed_budget_slots_completed=8,
    )
    panel, evaluator_wall = _evaluate_terminal_z_panel(
        model,
        tokenizer,
        requests,
        cases,
        alias=alias,
        request_order_sha256=request_order,
        action_sha256=str(action["identity_sha256"]),
        hparams=hparams,
        terminal_target=target,
        terminal_physical=physical,
        method="P4-ZA-Native-Z",
        instruction_id=P4_INSTRUCTION_ID,
        method_id=P4_METHOD_ID,
        schema="ode-edit-s05-p4-za-terminal-z-panel/v1",
        trajectory_status="ACTION_FROZEN_P4_ZA_NATIVE_Z",
    )
    if _model_w0_contract(touched) != expected_w0:
        raise ODEBFStateError("P4 ZA Native-Z evaluator mutated W0")
    result: dict[str, object] = {
        "schema": "ode-edit-s05-p4-za-native-arm-terminal/v1",
        "alias": alias,
        "arm": "Native-Z",
        "case_index": 1,
        "request_order_sha256": request_order,
        "terminal_target_sha256": tensor_sha256(target),
        "action_freeze_file_sha256": action_file_sha,
        "terminal_z_panel": panel,
        "evaluator_wall_seconds": evaluator_wall,
        "total_wall_seconds": time.perf_counter() - started,
        "native_compute_z_call_count": BATCH_SIZE,
        "writer_call_count": 0,
        "W0_restored": True,
        "scientific_promotion": False,
    }
    result["identity_sha256"] = canonical_hash(result)
    _atomic_write_once(root / "terminal.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    inner_iterations = RUN_TOKEN_TO_INNER_ITERATIONS[args.run_token]
    if os.environ.get("PROJECT_GPU_CAP") != "2":
        raise ODEBFContractError("P4 ZA PROJECT_GPU_CAP differs")
    if any(
        os.environ.get(name) != expected
        for name, expected in {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }.items()
    ):
        raise ODEBFContractError("P4 ZA offline environment differs")
    observed_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if observed_head != args.source_head:
        raise ODEBFContractError("P4 ZA source HEAD differs")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError("P4 ZA output root is create-once")
    os.umask(0o077)
    args.output_root.mkdir(mode=0o700, parents=True)
    stages = P1StageRecorder(args.output_root / "stages")
    final = _load_final_receipt(
        args.final_pre_gpu_receipt,
        alias=args.model,
        source_head=args.source_head,
        inner_iterations=inner_iterations,
    )
    stream_receipt = final["transfer_receipt"]
    binding = final["model_input_binding"][args.model]
    stages.record("pre_model_final_gate", {
        "final_pre_gpu_identity": final["identity_sha256"],
        "transfer_identity": stream_receipt["identity_sha256"],
        "model_input_binding": binding,
    })

    seed_all(COMMON_SEED)
    hf_seal = load_p4_hf_consumed_closure_seal(HF_SEAL, repo_root=REPO_ROOT)
    model, tokenizer, hf_receipt = load_p4_full_fp32_from_sealed_snapshot(
        hf_seal,
        args.model,
        stream_receipt=stream_receipt,
        final_pre_gpu_receipt=final,
        reuse_final_verified_closure=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams

    hparams = AlphaEditHyperParams.from_hparams(str(HPARAMS[args.model]))
    hparams.device = 0
    hparams.P_loc = str(
        EASYEDIT_ROOT
        / (
            "examples/null_space_project_Meta-Llama-3-8B-Instruct.pt"
            if args.model == "llama3-8b-inst"
            else "examples/null_space_project_Qwen2.5-7B-Instruct.pt"
        )
    )
    hparams.stats_dir = str(EASYEDIT_ROOT / "examples/data/stats")
    model.config._name_or_path = hparams.model_name
    fp32 = _assert_full_fp32(model)
    stages.record("post_model_full_fp32_offline", {
        "hf": hf_receipt.to_dict(), "full_fp32": fp32,
        "offline": True, "local_files_only": True,
    })

    stream_root = args.stream_root.resolve(strict=True)
    stream = _load_json_regular(stream_root / STREAM_SEAL_RELATIVE)
    identity = _load_json_regular(stream_root / STREAM_IDENTITY_RELATIVE)
    batches = load_historical_h0_batches(DATASET, stream)
    requests = batches[0]
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    if (
        request_order != binding["request_order_sha256"]
        or binding["alias"] != args.model
        or binding["native_name"] != hf_receipt.native_name
    ):
        raise ODEBFContractError("P4 ZA alias/sealed case01 binding differs")

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
    transferred_row = _evaluation_row(identity, args.model)
    if (
        request_order != transferred_row.get("request_order_sha256")
        or paired.new.request_order_sha256 != request_order
        or capture_plan.request_order_sha256 != request_order
    ):
        raise ODEBFContractError("P4 ZA runtime input/order binding differs")
    stages.record("post_alias_sealed_input_plan", {
        "alias": args.model,
        "request_order_sha256": request_order,
        "transferred_bf16_objective_plan_sha256_observation_only": transferred_row["objective_plan_sha256"],
        "transferred_bf16_capture_plan_sha256_observation_only": transferred_row["capture_plan_sha256"],
        "capture_plan_sha256": capture_plan.identity_sha256,
        "execution_objective_plan_sha256": paired.new.identity_sha256,
        "execution_microbatch_size": EXECUTION_MICROBATCH,
        "execution_microbatch_role": "FULL_FP32_MEMORY_DEPLOYMENT_ONLY",
        "transferred_bf16_plan_decision_influence_count": 0,
        "context_sha256": context_sha,
        "transferred_evaluation_case_identity_sha256": transferred_row["evaluation_case_identity_sha256"],
    })

    touched = _touched_weights(model, hparams)
    expected_w0 = _model_w0_contract(touched)
    inventory_w0 = _parameter_inventory_sha256(model)
    physical = capture_scalable_physical_state(model, capture_plan, hparams)
    initial = initial_target_from_capture(physical)
    if (
        initial.target_z.dtype is not torch.float32
        or initial.current_terminal_z.dtype is not torch.float32
        or not bool(torch.isfinite(initial.target_z).all())
    ):
        raise ODEBFContractError("P4 ZA initial target is not finite FULL-FP32")
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
    non_barrier = non_barrier_arm_identity(
        paired,
        kl_plan,
        teacher_sha256=teacher_sha,
        origin_target_sha256=tensor_sha256(initial.target_z),
        terminal_target_sha256=tensor_sha256(initial.current_terminal_z),
        target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
        learning_rate=float(hparams.v_lr),
        kl_factor=lock.kl_factor,
        decay_factor=lock.decay_factor,
        clamp_factor=lock.clamp_factor,
        inner_iterations=inner_iterations,
        selected_final_observation=(inner_iterations == 1),
    )
    stages.record("post_shared_W0_non_barrier_gate", {
        "W0_sha256": expected_w0,
        "model_inventory_sha256": inventory_w0,
        "non_barrier_identity": non_barrier,
        "alpha_geometry": alpha_geometry,
        "teacher": teacher_result.raw_free_payload(),
        "teacher_sha256": teacher_sha,
        "writer_count": 0,
    })

    arm_results: list[dict[str, object]] = []
    for target_arm in (P4TargetArm.POSITIVE, P4TargetArm.POSITIVE_NEGATIVE):
        if _model_w0_contract(touched) != expected_w0:
            raise ODEBFStateError("P4 ZA same-W0 entry differs")
        arm_results.append(_target_arm(
            model, tokenizer, requests,
            alias=args.model, arm=target_arm,
            root=args.output_root / ("z-plus" if target_arm is P4TargetArm.POSITIVE else "z-plus-minus"),
            hparams=hparams, paired=paired, physical=physical, initial=initial,
            kl_plan=kl_plan, teacher=teacher, teacher_sha=teacher_sha,
            lock=lock, non_barrier=non_barrier, dataset_path=DATASET,
            touched=touched, expected_w0=expected_w0, stages=stages,
            inner_iterations=inner_iterations,
        ))
    arm_results.append(_native_arm(
        model, tokenizer, requests,
        alias=args.model, root=args.output_root / "native-z", hparams=hparams,
        contexts=contexts, physical=physical, dataset_path=DATASET,
        touched=touched, expected_w0=expected_w0, request_order=request_order,
    ))
    if _model_w0_contract(touched) != expected_w0:
        raise ODEBFStateError("P4 ZA panel final W0 differs")
    terminal: dict[str, object] = {
        "schema": "ode-edit-s05-p4-za-case01-pilot-terminal/v1",
        "instruction_id": P4_INSTRUCTION_ID,
        "method_id": P4_METHOD_ID,
        "source_head": args.source_head,
        "alias": args.model,
        "case_index": 1,
        "request_count": BATCH_SIZE,
        "request_order_sha256": request_order,
        "arms": [item["arm"] for item in arm_results],
        "arm_terminal_identity_sha256": [item["identity_sha256"] for item in arm_results],
        "same_W0": True,
        "W0_sha256": expected_w0,
        "W0_restored": True,
        "writer_call_count": 0,
        "inner_iterations_per_outer": inner_iterations,
        "target_optimizer_update_count_per_arm": 8 * inner_iterations,
        "selected_final_observation_count_per_target_arm": (
            8 if inner_iterations == 1 else 0
        ),
        "scientific_delta": (
            "M1_TARGET_UPDATE_BUDGET_ONLY"
            if inner_iterations == 1
            else "AUTHORITATIVE_M5"
        ),
        "non_barrier_identity_sha256": non_barrier["identity_sha256"],
        "full_fp32": fp32,
        "offline": True,
        "retry_count": 0,
        "scientific_promotion": False,
        "status": "P4_ZA_CASE01_PILOT_TERMINAL",
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
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--stream-root", required=True, type=Path)
    parser.add_argument("--final-pre-gpu-receipt", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument(
        "--run-token", required=True, choices=tuple(RUN_TOKEN_TO_INNER_ITERATIONS)
    )
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        failure = {
            "schema": "ode-edit-s05-p4-za-case01-pilot-failure/v1",
            "instruction_id": P4_INSTRUCTION_ID,
            "alias": args.model,
            "exception_class": type(exc).__name__,
            "exception_message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
            "retry_count": 0,
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
