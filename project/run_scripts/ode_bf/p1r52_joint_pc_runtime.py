"""Independent B100 runtime for P1R52 C0/C1/C2 joint-P/C writers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .atomic_runtime_optimization import AcceptedPhysicalStateMaterializer
from .contracts import COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .fixed_e8_soft_routing import FixedE8Arm
from .functional import tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_evaluator import EndpointActionFreeze, load_counterfact_cases_after_freeze
from .p1_replay import build_outer_entry_pretrained_cache
from .p1_runtime import ArmRuntimeState, _atomic_write_once, _entry_parameter_snapshot_sha256
from .p1_scalable_batched_experiment import _model_w0_contract, _run_ode_arm
from .p1_state import ArmWeightSnapshot, P1Arm, P1HistoryLedger
from .p1r36_independent_b10x10_runtime import _hashes
from .p1r43_full_strength_routing import solve_p1r43_full_strength_routing
from .p1r52_accepted_z_observation import evaluate_accepted_z_batch, r52_binding
from .p1r52_joint_pc_execution import (
    INSTRUCTION_ID,
    METHOD_ID,
    JointPCWriterArm,
    JointPCWriterResult,
    plan_c1_writer,
    plan_c2_writer,
    post_commit_joint_pc_identity,
)
from .p1r52_joint_pc_router import proxies_from_entry_problem, solve_joint_pc_router
from .p1r52_target_official_alphaedit_writer import _endpoint_summary, _evaluate_w, _writer_gap
from .scalable_batched_field import (
    build_scalable_dynamic_field,
    build_scalable_routing_problem,
    scalable_physical_signed_progress,
)
from .scalable_batched_model import (
    build_scalable_capture_plan,
    build_scalable_objective_plan,
    capture_scalable_physical_state,
    evaluate_scalable_target_new_objective,
)
from .scalable_batched_runtime import P1R23_GRID_COUNT, P1R23_LAYER_ORDER, scalable_ordered_request_digest


PILOT_ROLE = "r52-joint-pc-c1-c2-pilot"
PILOT_TECH_R1_ROLE = "r52-joint-pc-c1-c2-pilot-tech-r1"
PILOT_TECH_R2_ROLE = "r52-joint-pc-c1-c2-pilot-tech-r2"
PRODUCTION_ROLE_PREFIX = "r52-joint-pc-c1-c2-production-case-"
PILOT_RESULT_NAME = "s05-p1r52-joint-pc-c1-c2-pilot-b100-v1"
PILOT_TECH_R1_RESULT_NAME = "s05-p1r52-joint-pc-c1-c2-pilot-b100-tech-r1-v1"
PILOT_TECH_R2_RESULT_NAME = "s05-p1r52-joint-pc-c1-c2-pilot-b100-tech-r2-v1"
PRODUCTION_RESULT_PREFIX = "s05-p1r52-joint-pc-c1-c2-independent-b100-case-"
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"


def production_role(case_index: int) -> str:
    if case_index < 1 or case_index > 10:
        raise ODEBFContractError("joint P/C production case index differs")
    return f"{PRODUCTION_ROLE_PREFIX}{case_index:02d}"


def expected_result_name(role: str) -> str:
    if role == PILOT_ROLE:
        return PILOT_RESULT_NAME
    if role == PILOT_TECH_R1_ROLE:
        return PILOT_TECH_R1_RESULT_NAME
    if role == PILOT_TECH_R2_ROLE:
        return PILOT_TECH_R2_RESULT_NAME
    if role.startswith(PRODUCTION_ROLE_PREFIX):
        suffix = role.removeprefix(PRODUCTION_ROLE_PREFIX)
        if len(suffix) == 2 and suffix.isdigit() and 1 <= int(suffix) <= 10:
            return f"{PRODUCTION_RESULT_PREFIX}{int(suffix):02d}-v1"
    raise ODEBFContractError("joint P/C runtime role differs")


def _case_index(role: str) -> int:
    if role in (PILOT_ROLE, PILOT_TECH_R1_ROLE, PILOT_TECH_R2_ROLE):
        return 1
    expected_result_name(role)
    return int(role.removeprefix(PRODUCTION_ROLE_PREFIX))


def _entry_state(
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    *,
    case_index: int,
) -> ArmRuntimeState:
    hashes = _hashes(touched)
    if hashes != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("joint P/C target entry W differs")
    return ArmRuntimeState(
        P1Arm.R_BF,
        P1HistoryLedger(
            layer_order=P1R23_LAYER_ORDER,
            maximum_records=400,
            batch_size=100,
        ),
        ComputeLedger(),
        ArmWeightSnapshot(
            P1Arm.R_BF,
            0,
            base_receipt.parameter_sha256,
            canonical_hash({"joint_pc_case": case_index, "entry": hashes}),
        ),
        dict(base_values),
    )


def _frozen_target_once(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
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
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    request_microbatch_size: int,
    job_ledger: ComputeLedger,
) -> tuple[torch.Tensor, Mapping[str, Any], Any, Any, Any]:
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    objective_plan = build_scalable_objective_plan(
        model,
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=min(request_microbatch_size, len(requests)),
        fact_token_strategy=hparams.fact_token,
    )
    capture_plan = build_scalable_capture_plan(
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=min(request_microbatch_size, len(requests)),
        fact_token_strategy=hparams.fact_token,
    )
    if objective_plan.request_order_sha256 != request_order or capture_plan.request_order_sha256 != request_order:
        raise ODEBFStateError("joint P/C target plan order differs")
    outer_population = tuple(
        population_by_sha256[item] for item in theta0_cache.request_order
    )
    snapshot = _entry_parameter_snapshot_sha256(
        model, dict(base_receipt.parameter_sha256)
    )
    counter = ModelForwardCounter(model, job_ledger)
    try:
        outer_entry_p_cache = build_outer_entry_pretrained_cache(
            model,
            tokenizer,
            outer_population,
            theta0_cache,
            outer_entry_snapshot_sha256=snapshot,
        )
    finally:
        counter.close()
    rollout = _run_ode_arm(
        model,
        tokenizer,
        requests,
        alias="llama3-8b-inst",
        arm=FixedE8Arm.SOFT,
        allocation="RS",
        capture_plan=capture_plan,
        objective_plan=objective_plan,
        hparams=hparams,
        projector=projector,
        contexts=contexts,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256,
        controller_lock=controller_lock,
        arm_state=_entry_state(
            touched, base_receipt, base_values, case_index=case_index
        ),
        request_by_sha256=request_by_sha256,
        population_by_sha256=population_by_sha256,
        schedule=schedule,
        outer_entry_p_cache=outer_entry_p_cache,
        theta0_cache=theta0_cache,
        touched=touched,
        base_receipt=base_receipt,
        base_values=base_values,
        raw_root=case_root / "raw" / "target",
        write_once=_atomic_write_once,
        p1r24=True,
        p1r34=True,
        p1r35=True,
        p1r52=True,
    )
    public = rollout["public"]
    if public["accepted_update_count"] != P1R23_GRID_COUNT or public["tau_final"] != 1.0:
        raise ODEBFStateError("joint P/C P1R52 K8 target differs")
    if _hashes(touched) != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("joint P/C target did not restore W0")
    target = rollout["terminal_target"].detach().to(device="cpu", dtype=torch.float32).contiguous()
    return target, public, objective_plan, capture_plan, outer_entry_p_cache


def _writer_entry(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    target: torch.Tensor,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    objective_plan: Any,
    capture_plan: Any,
) -> dict[str, Any]:
    physical = capture_scalable_physical_state(model, capture_plan, hparams)
    field_ledger = ComputeLedger()
    field = build_scalable_dynamic_field(
        model,
        tokenizer,
        requests,
        hparams,
        projector,
        contexts,
        target_state=target,
        current_terminal=physical.terminal_z,
        captured_keys_by_layer=physical.keys_by_layer,
        accepted_waypoint=0,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256,
        residual_tolerance=controller_lock.residual_tolerance,
        ledger=field_ledger,
    )
    signed, slope = scalable_physical_signed_progress(model, objective_plan, field)
    problem_receipt = build_scalable_routing_problem(
        field,
        signed,
        accepted_by_layer={layer: () for layer in P1R23_LAYER_ORDER},
        committed_load_by_layer={layer: 0.0 for layer in P1R23_LAYER_ORDER},
        lock=controller_lock,
    )
    endpoint_target = target.detach().to(
        device=next(model.parameters()).device, dtype=torch.float32
    )
    endpoint = evaluate_scalable_target_new_objective(
        model,
        objective_plan,
        target_state=endpoint_target,
        current_terminal=physical.terminal_z,
        target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
        target_gradient_required=False,
    )
    entry_nll = float(slope.loss)
    endpoint_nll = float(endpoint.loss)
    alpha_star = max(entry_nll - endpoint_nll, 0.0)
    if not math.isfinite(alpha_star) or alpha_star <= 0.0:
        raise ODEBFStateError("JOINT_PC_NONPOSITIVE_FINITE_TARGET_DEMAND")
    control = solve_p1r43_full_strength_routing(
        problem_receipt.problem,
        arm=FixedE8Arm.SOFT,
        alpha_req=alpha_star,
    )
    p_proxy, c_proxy = proxies_from_entry_problem(problem_receipt.problem)
    joint = solve_joint_pc_router(p_proxy, c_proxy)
    return {
        "physical": physical,
        "field": field,
        "signed": signed,
        "slope": slope,
        "problem": problem_receipt,
        "endpoint": endpoint,
        "entry_nll": entry_nll,
        "endpoint_nll": endpoint_nll,
        "alpha_star": alpha_star,
        "control": control,
        "joint": joint,
        "field_compute": field_ledger.raw_free_payload(),
    }


def _plan_arm(
    model: torch.nn.Module,
    arm: JointPCWriterArm,
    *,
    entry: Mapping[str, Any],
    target: torch.Tensor,
    hparams: Any,
    projector: torch.Tensor,
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    objective_plan: Any,
    capture_plan: Any,
    base_values: Mapping[str, torch.Tensor],
) -> JointPCWriterResult:
    common = {
        "hparams": hparams,
        "projector": projector,
        "covariance_registry": covariance_registry,
        "projector_sha256": projector_sha256,
        "residual_tolerance": controller_lock.residual_tolerance,
        "objective_plan": objective_plan,
        "capture_plan": capture_plan,
        "base_values": base_values,
        "current_factors": {name: () for name in base_values},
        "entry_field": entry["field"],
        "target_state": target,
        "step_index": 0,
    }
    if arm in (JointPCWriterArm.C0, JointPCWriterArm.C1):
        pi = (
            entry["control"].pi
            if arm is JointPCWriterArm.C0
            else tuple(float(item) for item in entry["joint"].pi)
        )
        return plan_c1_writer(
            model,
            arm=arm,
            entry_applied_slopes=entry["problem"].problem.signed_progress,
            entry_pi=pi,
            entry_velocity=entry["control"].velocity,
            alpha_star=float(entry["alpha_star"]),
            endpoint_nll=float(entry["endpoint_nll"]),
            entry_nll=float(entry["entry_nll"]),
            **common,
        )
    return plan_c2_writer(
        model,
        entry_pi=tuple(float(item) for item in entry["joint"].pi),
        **common,
    )


def run_joint_pc_case(
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
    **_: Any,
) -> dict[str, Any]:
    del mutation_lock
    if (
        alias != "llama3-8b-inst"
        or len(stream_batches) != 10
        or any(len(batch) != 100 for batch in stream_batches)
        or stream.get("root_digest") != STREAM_ROOT
        or stream.get("all_request_order_sha256") != STREAM_ORDER
    ):
        raise ODEBFContractError("joint P/C stream/model matrix differs")
    case_index = _case_index(role)
    requests = tuple(stream_batches[case_index - 1])
    case_root = raw_root / f"case-{case_index:02d}"
    case_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    seed_all(COMMON_SEED)
    w0 = _model_w0_contract(touched)
    w0_hashes = _hashes(touched)
    w0_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    target, target_public, objective_plan, capture_plan, _ = _frozen_target_once(
        model,
        tokenizer,
        requests,
        case_index=case_index,
        case_root=case_root,
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
        base_receipt=base_receipt,
        base_values=base_values,
        request_microbatch_size=request_microbatch_size,
        job_ledger=job_ledger,
    )
    target_hash = tensor_sha256(target)
    entry = _writer_entry(
        model,
        tokenizer,
        requests,
        target=target,
        hparams=hparams,
        projector=projector,
        contexts=contexts,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256,
        controller_lock=controller_lock,
        objective_plan=objective_plan,
        capture_plan=capture_plan,
    )

    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    binding = r52_binding(role, requests, hparams, target)
    freeze = EndpointActionFreeze(
        arm=f"{role}-case-{case_index:02d}",
        sequential_batch=case_index - 1,
        request_order_sha256=request_order,
        selected_snapshot_sha256=target_public["identity_sha256"],
        fixed_budget_slots_completed=P1R23_GRID_COUNT,
    )
    cases = load_counterfact_cases_after_freeze(
        dataset_path, requests, freeze, expected_batch_size=100
    )
    counter = ModelForwardCounter(model, job_ledger)
    try:
        z_observation, _ = evaluate_accepted_z_batch(
            model,
            tokenizer,
            requests,
            cases,
            role=role,
            round_index=case_index,
            binding=binding,
            committed_weight_sha256=w0_hashes,
        )
    finally:
        counter.close()
    z_scores = z_observation["scores"]
    z_summary = _endpoint_summary(z_scores)

    arm_rows: dict[str, Any] = {}
    entry_hash_by_arm: dict[str, Mapping[str, str]] = {}
    target_hash_by_arm: dict[str, str] = {}
    for arm in (JointPCWriterArm.C0, JointPCWriterArm.C1, JointPCWriterArm.C2):
        if _model_w0_contract(touched) != w0 or _hashes(touched) != w0_hashes:
            raise ODEBFStateError("joint P/C arm entry W differs")
        entry_hash_by_arm[arm.value] = dict(_hashes(touched))
        target_hash_by_arm[arm.value] = tensor_sha256(target)
        planned = _plan_arm(
            model,
            arm,
            entry=entry,
            target=target,
            hparams=hparams,
            projector=projector,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            controller_lock=controller_lock,
            objective_plan=objective_plan,
            capture_plan=capture_plan,
            base_values=base_values,
        )
        factors = {name: () for name in base_values}
        for name, factor in planned.increment.items():
            factors[name] = (factor,)
        materializer = AcceptedPhysicalStateMaterializer(model, base_values)
        materialization = materializer.materialize(factors, transition_index=1)
        post_physical = capture_scalable_physical_state(model, capture_plan, hparams)
        post_identity = post_commit_joint_pc_identity(
            planned, materialization, post_physical.terminal_z
        )
        scores = _evaluate_w(model, tokenizer, cases, freeze=freeze, ledger=job_ledger)
        summary = _endpoint_summary(scores)
        gap = _writer_gap(summary, z_summary)
        restore = materializer.restore()
        if (
            _model_w0_contract(touched) != w0
            or _hashes(touched) != w0_hashes
            or any(int(touched[name].data_ptr()) != w0_pointers[name] for name in touched)
        ):
            raise ODEBFStateError("joint P/C arm restore/isolation differs")
        arm_rows[arm.value] = {
            "summary": summary,
            "scores": scores,
            "gap": gap,
            "writer": planned.receipt,
            "materialization": materialization,
            "post_commit_identity": post_identity,
            "post_terminal_sha256": tensor_sha256(post_physical.terminal_z),
            "post_terminal_residual_norm": float(
                torch.linalg.norm(
                    (target - post_physical.terminal_z.detach().cpu().float()).double()
                )
            ),
            "restore": restore,
        }

    if len(set(target_hash_by_arm.values())) != 1 or len({canonical_hash(item) for item in entry_hash_by_arm.values()}) != 1:
        raise ODEBFStateError("joint P/C paired target/W entry differs")
    payload = {
        "schema": "ode-edit-s05-p1r52-joint-pc-c1-c2-independent-b100-case/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "source_head": source_head,
        "role": role,
        "case_index": case_index,
        "request_count": 100,
        "request_order_sha256": request_order,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "accepted_z_sha256": target_hash,
        "accepted_z_hash_by_arm": target_hash_by_arm,
        "accepted_z_hash_exact_across_arms": True,
        "writer_entry_hash_by_arm": entry_hash_by_arm,
        "writer_entry_hash_exact_across_arms": True,
        "target_compute_count": 1,
        "target": target_public,
        "z_direct": {
            "summary": z_summary,
            "scores": z_scores,
            "observation": z_observation,
        },
        "writer_entry": {
            "physical_sha256": entry["physical"].identity_sha256,
            "field_sha256": entry["field"].identity_sha256,
            "signed_progress": list(entry["problem"].problem.signed_progress),
            "entry_nll": entry["entry_nll"],
            "endpoint_nll": entry["endpoint_nll"],
            "alpha_star": entry["alpha_star"],
            "control_route": entry["control"].raw_free_payload(),
            "joint_route": entry["joint"].receipt.raw_free_payload(),
            "field_compute": entry["field_compute"],
            "entry_field_solve_count": 5,
            "joint_router_solve_count": 1,
        },
        "arms": arm_rows,
        "c1_c2_joint_pi_exact": (
            arm_rows[JointPCWriterArm.C1.value]["writer"]["entry_pi"]
            == arm_rows[JointPCWriterArm.C2.value]["writer"]["entry_pi"]
        ),
        "C0_source_decision_delta_count": 0,
        "target_recompute_per_arm_count": 0,
        "heldout_writer_decision_access_count": 0,
        "cross_arm_weight_contamination_count": 0,
        "authoritative_materialization_count_by_arm": {
            arm.value: 1 for arm in JointPCWriterArm
        },
        "retry_count": 0,
        "backtracking_count": 0,
        "imputation_count": 0,
        "W0_restored": _model_w0_contract(touched) == w0,
        "job_compute": job_ledger.raw_free_payload(),
        "scientific_promotion": False,
    }
    if not payload["c1_c2_joint_pi_exact"] or not payload["W0_restored"]:
        raise ODEBFStateError("joint P/C terminal pairing differs")
    payload["identity_sha256"] = canonical_hash(payload)
    case_sha = _atomic_write_once(case_root / "terminal.json", payload)
    terminal = {
        "schema": "ode-edit-s05-p1r52-joint-pc-c1-c2-independent-b100-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "source_head": source_head,
        "role": role,
        "case_index": case_index,
        "request_attempt_count": 100,
        "valid_case_count": 1,
        "typed_failure_count": 0,
        "technical_failure_count": 0,
        "case_terminal_sha256": case_sha,
        "accepted_z_sha256": target_hash,
        "joint_pi": arm_rows[JointPCWriterArm.C1.value]["writer"]["entry_pi"],
        "W0_restored": payload["W0_restored"],
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r52-joint-pc-c1-c2-independent-b100-manifest/v1",
        "source_head": source_head,
        "role": role,
        "case_index": case_index,
        "terminal_sha256": terminal_sha,
        "case_terminal_sha256": case_sha,
        "W0_restored": payload["W0_restored"],
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    stages.record(
        "joint_pc_terminal",
        {
            "role": role,
            "case_index": case_index,
            "target_compute_count": 1,
            "arm_count": 3,
            "W0_restored": payload["W0_restored"],
        },
    )
    return {
        "status": "P1R52_JOINT_PC_CASE_TERMINAL",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "case_index": case_index,
        "W0_restored": payload["W0_restored"],
    }


__all__ = [
    "PILOT_RESULT_NAME",
    "PILOT_ROLE",
    "PILOT_TECH_R1_RESULT_NAME",
    "PILOT_TECH_R1_ROLE",
    "PILOT_TECH_R2_RESULT_NAME",
    "PILOT_TECH_R2_ROLE",
    "PRODUCTION_RESULT_PREFIX",
    "PRODUCTION_ROLE_PREFIX",
    "STREAM_ORDER",
    "STREAM_ROOT",
    "expected_result_name",
    "production_role",
    "run_joint_pc_case",
]
