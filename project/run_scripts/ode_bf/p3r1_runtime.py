"""P3R1 fixed-M fast-target / finite-horizon FP32 Atomic runtime.

The module is an orchestration layer over the frozen P1R24 objective,
P1R52 C0/C1 routers, FP32 transaction, and direct EasyEdit writers.  It does
not duplicate any target objective, projector, covariance, evaluator, or
native writer kernel.
"""

from __future__ import annotations

import contextlib
import copy
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .contracts import COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .fixed_e8_soft_routing import FixedE8Arm
from .functional import tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_evaluator import (
    EndpointActionFreeze,
    evaluate_counterfact_success_accuracy_batch,
    load_counterfact_cases_after_freeze,
)
from .p1_runtime import _atomic_write_once
from .p1_scalable_batched_experiment import _model_w0_contract
from .p1_state import ArmWeightSnapshot
from .p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    build_p1r24_kl_plan,
    evaluate_p1r24_kl,
    verify_p1r24_alphaedit_geometry,
)
from .p1r36_independent_b10x10_runtime import _hashes, _restore_exact_w0
from .p1r43_full_strength_routing import solve_p1r43_full_strength_routing
from .p1r52_accepted_z_observation import (
    OfficialNativeZCapture,
    evaluate_accepted_z_batch,
    r52_binding,
)
from .p1r52_frozen_pi_quota_writer import build_current_layer_field
from .p1r52_joint_pc_fp32_runtime import _raw_free_json_tree
from .p1r52_joint_pc_router import proxies_from_entry_problem, solve_joint_pc_router
from .p1r52_pir_writer import remaining_pi_beta
from .p1r52_residual_reserve_fp32_transaction import (
    FP32TransactionMode,
    OfficialStyleFP32SequentialTransaction,
)
from .p1r52_residual_reserve_phase_a_execution import fp32_weight_energy
from .p1r52_residual_reserve_update_binding import (
    apply_prepared_low_rank_update,
    build_prepared_low_rank_update,
)
from .p1r52_target_official_alphaedit_writer import (
    _endpoint_summary,
    _writer_gap,
    accepted_z_cache_template,
    isolated_alphaedit_module_state,
)
from .p1r52_official_sequential_baselines import run_official_memit_apply
from .p3r1_finite_horizon import OUTER_COUNT, finite_horizon_waypoint
from .p3r1_fixed_m_target import (
    INSTRUCTION_ID,
    METHOD_ID,
    PRODUCTION_INNER_COUNT,
    TARGET_ONLY_REFERENCE_INNER_COUNT,
    run_fixed_m_final_iterate,
)
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
from .scalable_batched_native import run_official_native_apply
from .scalable_batched_runtime import (
    P1R23_LAYER_ORDER,
    initial_target_from_capture,
    scalable_ordered_request_digest,
)


ROLE_PREFIX = "p3r1-two-timescale-fastz-fh-case-"
RESULT_PARENT_NAME = "p3r1-two-timescale-fastz-fh-c013-fp32-tech-r2"
RESULT_PREFIX = "s05-p3r1-two-timescale-fastz-fh-c013-fp32-tech-r2"
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"


def case_role(case_index: int) -> str:
    if not 1 <= case_index <= 10:
        raise ODEBFContractError("P3R1 case index differs")
    return f"{ROLE_PREFIX}{case_index:02d}"


def is_p3r1_role(role: str | None) -> bool:
    if role is None or not role.startswith(ROLE_PREFIX):
        return False
    suffix = role.removeprefix(ROLE_PREFIX)
    return len(suffix) == 2 and suffix.isdigit() and 1 <= int(suffix) <= 10


def role_case_index(role: str) -> int:
    if not is_p3r1_role(role):
        raise ODEBFContractError("P3R1 role differs")
    return int(role.removeprefix(ROLE_PREFIX))


def expected_result_name(alias: str, role: str) -> str:
    case_index = role_case_index(role)
    if alias not in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        raise ODEBFContractError("P3R1 alias differs")
    return f"{RESULT_PREFIX}-{alias}-case-{case_index:02d}-v1"


def expected_result_parent(repo_root: Path) -> Path:
    return (
        repo_root / "local" / "odebf" / "results" / RESULT_PARENT_NAME
    ).resolve(strict=False)


def writer_route_requirement(arm: str) -> str:
    """Return the only authoritative entry router for a P3R1 writer arm."""
    if arm == "C0-FH":
        return "LEGACY_SOFT"
    if arm == "C1-FH":
        return "JOINT_PC"
    if arm == "C3-FH":
        return "DIRECT_OFFICIAL"
    raise ODEBFContractError("P3R1 writer arm differs")


def expected_writer_solver_counts(arm: str, *, positive_demand: bool) -> tuple[int, int]:
    requirement = writer_route_requirement(arm)
    if not positive_demand or requirement == "DIRECT_OFFICIAL":
        return 0, 0
    if requirement == "LEGACY_SOFT":
        return 1, 0
    return 0, 1


def _bindings(model: torch.nn.Module, hparams: Any) -> dict[int, tuple[str, torch.nn.Parameter]]:
    named = dict(model.named_parameters())
    return {
        layer: (
            f"{hparams.rewrite_module_tmp.format(layer)}.weight",
            named[f"{hparams.rewrite_module_tmp.format(layer)}.weight"],
        )
        for layer in P1R23_LAYER_ORDER
    }


def _state_identity(touched: Mapping[str, torch.nn.Parameter]) -> str:
    return canonical_hash(_hashes(touched))


def _restore(
    touched: Mapping[str, torch.nn.Parameter],
    base_values: Mapping[str, torch.Tensor],
    *,
    mutation_lock: Any,
    entry_contract: str,
) -> Mapping[str, Any]:
    return _restore_exact_w0(
        touched,
        base_values,
        mutation_lock=mutation_lock,
        expected_contract=entry_contract,
    )


def _evaluate_w(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[Any],
    *,
    alias: str,
    freeze: EndpointActionFreeze,
    ledger: ComputeLedger,
) -> Mapping[str, Any]:
    counter = ModelForwardCounter(model, ledger)
    try:
        return evaluate_counterfact_success_accuracy_batch(
            model,
            tokenizer,
            cases,
            model_alias=alias,
            freeze=freeze,
            expected_batch_size=len(cases),
        ).raw_free_payload()
    finally:
        counter.close()


def _load_memit_hparams(alias: str) -> Any:
    from easyeditor.models.memit.memit_hparams import MEMITHyperParams

    filename = {
        "llama3-8b-inst": "llama3-8b.yaml",
        "qwen2.5-7b-inst": "qwen2.5-7b.yaml",
    }[alias]
    value = MEMITHyperParams.from_hparams(
        str(Path("/mnt/raid5/janghj/EasyEdit/hparams/MEMIT") / filename)
    )
    value.device = 0
    value.stats_dir = "/mnt/raid5/janghj/EasyEdit/examples/data/stats"
    if list(value.layers) != list(P1R23_LAYER_ORDER) or value.mom2_dtype != "float32":
        raise ODEBFContractError("P3R1 Official MEMIT hparams differ")
    return value


def _target_solver(
    model: torch.nn.Module,
    *,
    alias: str,
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    target_origin: torch.Tensor,
    inner_count: int,
    outer_step_index: int | None,
    objective_plan: Any,
    kl_plan: Any,
    teacher: Sequence[torch.Tensor],
    hparams: Any,
    touched: Mapping[str, torch.nn.Parameter],
) -> Any:
    target_layer_name = hparams.layer_module_tmp.format(int(hparams.layers[-1]))

    def target_eval(value: torch.Tensor, gradient_required: bool) -> Any:
        variable = value.to(next(model.parameters()).device)
        if gradient_required:
            variable.requires_grad_(True)
        return evaluate_scalable_target_new_objective(
            model,
            objective_plan,
            target_state=variable,
            current_terminal=current_terminal,
            target_layer_name=target_layer_name,
            target_gradient_required=gradient_required,
        )

    def kl_eval(value: torch.Tensor, gradient_required: bool) -> Any:
        variable = value.to(next(model.parameters()).device)
        if gradient_required:
            variable.requires_grad_(True)
        result, _ = evaluate_p1r24_kl(
            model,
            kl_plan,
            teacher_log_probs=teacher,
            target_state=variable,
            current_terminal=current_terminal,
            target_layer_name=target_layer_name,
            target_gradient_required=gradient_required,
        )
        return result

    return run_fixed_m_final_iterate(
        current_target=current_target,
        current_terminal=current_terminal,
        target_origin=target_origin,
        lock=P1R24AliasTargetLock.for_alias(alias),
        alias=alias,
        inner_count=inner_count,
        outer_step_index=outer_step_index,
        evaluate_target=target_eval,
        evaluate_kl=kl_eval,
        fixed_state_identity=lambda: _state_identity(touched),
    )


def _writer_entry(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    waypoint: torch.Tensor,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    objective_plan: Any,
    capture_plan: Any,
) -> Mapping[str, Any]:
    route_requirement = writer_route_requirement(arm)
    physical = capture_scalable_physical_state(model, capture_plan, hparams)
    field_ledger = ComputeLedger()
    field = build_scalable_dynamic_field(
        model,
        tokenizer,
        requests,
        hparams,
        projector,
        contexts,
        target_state=waypoint,
        current_terminal=physical.terminal_z,
        captured_keys_by_layer=physical.keys_by_layer,
        accepted_waypoint=0,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256,
        residual_tolerance=controller_lock.residual_tolerance,
        ledger=field_ledger,
    )
    signed, slope = scalable_physical_signed_progress(model, objective_plan, field)
    problem = None
    if route_requirement != "DIRECT_OFFICIAL":
        problem = build_scalable_routing_problem(
            field,
            signed,
            accepted_by_layer={layer: () for layer in P1R23_LAYER_ORDER},
            committed_load_by_layer={layer: 0.0 for layer in P1R23_LAYER_ORDER},
            lock=controller_lock,
        )
    endpoint_state = waypoint.to(next(model.parameters()).device)
    endpoint = evaluate_scalable_target_new_objective(
        model,
        objective_plan,
        target_state=endpoint_state,
        current_terminal=physical.terminal_z,
        target_layer_name=hparams.layer_module_tmp.format(int(hparams.layers[-1])),
        target_gradient_required=False,
    )
    alpha = max(float(slope.loss) - float(endpoint.loss), 0.0)
    if not math.isfinite(alpha):
        raise ODEBFStateError("P3R1 finite writer demand is nonfinite")
    control = None
    joint = None
    if alpha > 0.0:
        if route_requirement == "LEGACY_SOFT":
            if problem is None:
                raise ODEBFStateError("P3R1 C0 routing problem missing")
            control = solve_p1r43_full_strength_routing(
                problem.problem,
                arm=FixedE8Arm.SOFT,
                alpha_req=alpha,
            )
            if control.fallback_to_neutral:
                raise ODEBFStateError("C0_ROUTER_INVALID")
        elif route_requirement == "JOINT_PC":
            if problem is None:
                raise ODEBFStateError("P3R1 C1 routing problem missing")
            p_proxy, c_proxy = proxies_from_entry_problem(problem.problem)
            joint = solve_joint_pc_router(p_proxy, c_proxy)
            if joint.receipt.fallback_count != 0:
                raise ODEBFStateError("P3R1 C1 fallback differs")
    control_solver_call_count = int(control is not None)
    joint_solver_call_count = int(joint is not None)
    expected_solver_counts = expected_writer_solver_counts(
        arm, positive_demand=alpha > 0.0
    )
    if (control_solver_call_count, joint_solver_call_count) != expected_solver_counts:
        raise ODEBFStateError("P3R1 cross-arm solver count differs")
    return {
        "route_requirement": route_requirement,
        "control_solver_call_count": control_solver_call_count,
        "joint_solver_call_count": joint_solver_call_count,
        "cross_arm_solver_call_count": 0,
        "physical": physical,
        "field": field,
        "signed": signed,
        "slope": slope,
        "problem": problem,
        "endpoint": endpoint,
        "entry_nll": float(slope.loss),
        "endpoint_nll": float(endpoint.loss),
        "alpha": alpha,
        "control": control,
        "joint": joint,
        "field_compute": field_ledger.raw_free_payload(),
    }


def _apply_suffix_writer(
    model: torch.nn.Module,
    *,
    arm: str,
    outer_step_index: int,
    waypoint: torch.Tensor,
    entry: Mapping[str, Any],
    hparams: Any,
    projector: torch.Tensor,
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    capture_plan: Any,
) -> Mapping[str, Any]:
    if arm == "C0-FH":
        route = entry["control"]
        if route is None or route.fallback_to_neutral:
            raise ODEBFStateError("C0_ROUTER_INVALID")
        pi = tuple(float(value) for value in route.pi)
        route_payload = route.raw_free_payload()
    elif arm == "C1-FH":
        route = entry["joint"]
        if route is None or route.receipt.fallback_count != 0:
            raise ODEBFStateError("P3R1 C1 route differs")
        pi = tuple(float(value) for value in route.pi)
        route_payload = route.receipt.raw_free_payload()
    else:
        raise ODEBFContractError("P3R1 suffix writer arm differs")
    beta = remaining_pi_beta(pi)
    transaction = OfficialStyleFP32SequentialTransaction(
        _bindings(model, hparams),
        mode=FP32TransactionMode.AUTHORITATIVE,
        transaction_id=f"p3r1-{arm}-k{outer_step_index}",
    )
    rows: list[dict[str, Any]] = []
    current_physical = entry["physical"]
    with transaction:
        for ordinal, layer in enumerate(P1R23_LAYER_ORDER):
            current_terminal = current_physical.terminal_z.detach().cpu().float().contiguous()
            before_norm = float(torch.linalg.vector_norm((waypoint - current_terminal).double()))
            if ordinal == 0:
                field = entry["field"].layers[0]
                source = "ENTRY_FIELD_REUSE"
            else:
                field, _ = build_current_layer_field(
                    model,
                    hparams,
                    projector,
                    covariance_registry,
                    layer=layer,
                    key=current_physical.keys_by_layer[layer],
                    target_state=waypoint,
                    current_terminal=current_terminal,
                    step_index=outer_step_index,
                    factor_ordinal=ordinal,
                    projector_sha256=projector_sha256,
                    residual_tolerance=controller_lock.residual_tolerance,
                    q_only=True,
                )
                source = "CURRENT_PREFIX_KEY_Q"
            coefficient = beta[ordinal]
            weight_name, parameter = _bindings(model, hparams)[layer]
            construction = build_prepared_low_rank_update(
                construction_id=f"p3r1-{arm}-k{outer_step_index}-layer-{layer}",
                layer=layer,
                weight_name=weight_name,
                parameter_shape=tuple(parameter.shape),
                beta_applied_left32=(coefficient * field.residual).to(
                    parameter.device, torch.float32
                ).contiguous(),
                q_side32=field.q.detach().to(parameter.device, torch.float32).contiguous(),
            )
            application = apply_prepared_low_rank_update(transaction, construction)
            receipt = application.m3a_layer_receipt
            current_physical = capture_scalable_physical_state(model, capture_plan, hparams)
            after_terminal = current_physical.terminal_z.detach().cpu().float().contiguous()
            after_norm = float(torch.linalg.vector_norm((waypoint - after_terminal).double()))
            rows.append(
                {
                    "layer": layer,
                    "field_source": source,
                    "pi": pi[ordinal],
                    "suffix_mass": math.fsum(pi[ordinal:]),
                    "beta": coefficient,
                    "residual_norm_before": before_norm,
                    "residual_norm_after": after_norm,
                    "residual_reduction": before_norm - after_norm,
                    "key_sha256": tensor_sha256(field.key),
                    "q_sha256": tensor_sha256(field.q),
                    "key_norm": float(torch.linalg.vector_norm(field.key.double())),
                    "q_norm": float(torch.linalg.vector_norm(field.q.double())),
                    "update_norm": receipt.actual_post_storage_delta32_norm,
                    "update_energy": receipt.actual_post_storage_delta32_energy,
                    "numeric_storage_cast_count": receipt.numeric_storage_cast_count,
                    "bf16_path_call_count": receipt.bf16_path_call_count,
                }
            )
        prepared = transaction.prepare_authoritative_commit()
        transaction_receipt = transaction.commit_prepared(prepared.identity_sha256)
    total_energy = math.fsum(row["update_energy"] for row in rows)
    for row in rows:
        row["update_energy_share"] = row["update_energy"] / total_energy if total_energy else 0.0
    post = current_physical
    return {
        "arm": arm,
        "pi": list(pi),
        "beta": list(beta),
        "route": route_payload,
        "layers": rows,
        "transaction": transaction_receipt.raw_free_payload(),
        "post_terminal_sha256": tensor_sha256(post.terminal_z),
        "post_residual_norm": float(
            torch.linalg.vector_norm((waypoint - post.terminal_z.cpu().float()).double())
        ),
        "total_update_energy": total_energy,
        "fallback_count": 0,
        "residual_division_count": 0,
        "second_h_application_count": 0,
        "numeric_storage_cast_count": sum(row["numeric_storage_cast_count"] for row in rows),
        "bf16_path_call_count": sum(row["bf16_path_call_count"] for row in rows),
    }


def _run_dynamic_arm(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    arm: str,
    case_root: Path,
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    objective_plan: Any,
    capture_plan: Any,
    kl_plan: Any,
    teacher: Sequence[torch.Tensor],
    target_origin: torch.Tensor,
    touched: Mapping[str, torch.nn.Parameter],
    base_values: Mapping[str, torch.Tensor],
) -> Mapping[str, Any]:
    current_target = target_origin.clone()
    outer_rows: list[Mapping[str, Any]] = []
    alpha_append_rows: list[int] = []
    arm_started = time.perf_counter()
    module_context = isolated_alphaedit_module_state() if arm == "C3-FH" else contextlib.nullcontext({"restored": True})
    with module_context as module_state:
        for k in range(OUTER_COUNT):
            physical = capture_scalable_physical_state(model, capture_plan, hparams)
            current_terminal = physical.terminal_z.detach().cpu().float().contiguous()
            target_result = _target_solver(
                model,
                alias=alias,
                current_target=current_target,
                current_terminal=current_terminal,
                target_origin=target_origin,
                inner_count=PRODUCTION_INNER_COUNT,
                outer_step_index=k,
                objective_plan=objective_plan,
                kl_plan=kl_plan,
                teacher=teacher,
                hparams=hparams,
                touched=touched,
            )
            oracle = target_result.final_target
            schedule = finite_horizon_waypoint(current_terminal, oracle, outer_step_index=k)
            waypoint = schedule.waypoint
            entry = _writer_entry(
                model,
                tokenizer,
                requests,
                arm=arm,
                waypoint=waypoint,
                hparams=hparams,
                projector=projector,
                contexts=contexts,
                covariance_registry=covariance_registry,
                projector_sha256=projector_sha256,
                controller_lock=controller_lock,
                objective_plan=objective_plan,
                capture_plan=capture_plan,
            )
            pre_nll = entry["entry_nll"]
            if entry["alpha"] == 0.0:
                writer = {
                    "arm": arm,
                    "status": "ZERO_VECTOR_FIELD",
                    "fallback_count": 0,
                    "apply_count": 0,
                    "cache_append_count": 0,
                    "total_update_energy": 0.0,
                    "layers": [],
                }
                alpha_append_rows.append(0)
            elif arm in ("C0-FH", "C1-FH"):
                writer = _apply_suffix_writer(
                    model,
                    arm=arm,
                    outer_step_index=k,
                    waypoint=waypoint,
                    entry=entry,
                    hparams=hparams,
                    projector=projector,
                    covariance_registry=covariance_registry,
                    projector_sha256=projector_sha256,
                    controller_lock=controller_lock,
                    capture_plan=capture_plan,
                )
                alpha_append_rows.append(0)
            else:
                with accepted_z_cache_template(
                    requests,
                    waypoint,
                    hparams,
                    parent=case_root / "accepted-z-cache" / f"k{k}",
                ) as (cache_template, bridge):
                    nested = isolated_alphaedit_module_state() if k < OUTER_COUNT - 1 else contextlib.nullcontext({"restored": True})
                    with nested as step_state:
                        apply_payload, _ = run_official_native_apply(
                            model,
                            tokenizer,
                            requests,
                            hparams,
                            touched=touched,
                            reset_cache=True,
                            cache_history_width=0,
                            cache_template=cache_template,
                            expected_native_compute_z_call_count=0,
                            accepted_z_source="P3R1_DYNAMIC_FINITE_HORIZON_WAYPOINT",
                        )
                    writer = {
                        "arm": arm,
                        "status": "DIRECT_OFFICIAL_ALPHAEDIT_APPLY",
                        "apply": apply_payload,
                        "accepted_z_bridge": bridge,
                        "module_step_state_restored": bool(step_state["restored"]),
                        "fallback_count": 0,
                        "apply_count": 1,
                        "cache_append_count": 1 if k == OUTER_COUNT - 1 else 0,
                        "native_compute_z_call_count": apply_payload["native_alphaedit_compute_z_call_count"],
                        "p1r52_pc_router_decision_influence_count": 0,
                        "p1r52_barrier_decision_influence_count": 0,
                        "total_update_energy": None,
                    }
                    alpha_append_rows.append(writer["cache_append_count"])
            post_objective = evaluate_scalable_target_new_objective(model, objective_plan)
            post_physical = capture_scalable_physical_state(model, capture_plan, hparams)
            actual = pre_nll - float(post_objective.loss)
            outer = {
                "outer_step_index": k,
                "target": target_result.receipt,
                "target_inner": list(target_result.inner_receipts),
                "waypoint": schedule.receipt,
                "oracle_target_sha256": tensor_sha256(oracle),
                "waypoint_sha256": tensor_sha256(waypoint),
                "entry_terminal_sha256": tensor_sha256(current_terminal),
                "post_terminal_sha256": tensor_sha256(post_physical.terminal_z),
                "alpha": entry["alpha"],
                "W_nll_pre": pre_nll,
                "waypoint_injection_nll": entry["endpoint_nll"],
                "W_nll_post": float(post_objective.loss),
                "predicted_progress": entry["alpha"],
                "actual_progress": actual,
                "negative_progress": actual < 0.0,
                "writer": writer,
                "current_key_q_refresh_count": 4 if arm in ("C0-FH", "C1-FH") and entry["alpha"] > 0.0 else 0,
            }
            outer["identity_sha256"] = canonical_hash(outer)
            _atomic_write_once(case_root / arm / f"outer-{k:02d}.json", outer)
            outer_rows.append(outer)
            current_target = oracle
        if arm == "C3-FH" and (sum(alpha_append_rows[:-1]) != 0 or alpha_append_rows[-1] not in (0, 1)):
            raise ODEBFStateError("P3R1 C3 cache append schedule differs")
    if arm == "C3-FH" and not module_state["restored"]:
        raise ODEBFStateError("P3R1 C3 module state did not restore")
    return {
        "arm": arm,
        "outer": outer_rows,
        "terminal_target": current_target,
        "terminal_target_sha256": tensor_sha256(current_target),
        "wall_seconds": time.perf_counter() - arm_started,
        "fallback_count": sum(int(row["writer"]["fallback_count"]) for row in outer_rows),
        "retry_count": 0,
        "cache_append_by_outer": alpha_append_rows,
        "cache_append_count": sum(alpha_append_rows),
        "pre_k8_cache_append_count": sum(alpha_append_rows[:-1]),
        "module_state_restored": bool(module_state["restored"]),
        "update_energy": fp32_weight_energy(touched, base_values),
    }


def _native_baseline(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    cases: Sequence[Any],
    *,
    alias: str,
    kind: str,
    hparams: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_values: Mapping[str, torch.Tensor],
    freeze: EndpointActionFreeze,
    ledger: ComputeLedger,
) -> Mapping[str, Any]:
    if kind == "Official-AlphaEdit":
        native_hparams = hparams
        role = "native-alphaedit-sequential-cache-on-corrected"
        apply = lambda: run_official_native_apply(
            model,
            tokenizer,
            requests,
            native_hparams,
            touched=touched,
            reset_cache=True,
            cache_history_width=0,
        )
        context = isolated_alphaedit_module_state()
    elif kind == "Official-MEMIT":
        native_hparams = _load_memit_hparams(alias)
        role = "official-memit-sequential"
        apply = lambda: run_official_memit_apply(
            model, tokenizer, requests, native_hparams, touched=touched
        )
        context = contextlib.nullcontext({"restored": True})
    else:
        raise ODEBFContractError("P3R1 baseline differs")
    started = time.perf_counter()
    with context as state:
        with OfficialNativeZCapture(role=role, requests=requests, hparams=native_hparams) as capture:
            counter = ModelForwardCounter(model, ledger)
            try:
                apply_payload, _ = apply()
            finally:
                counter.close()
        binding = capture.finalize()
        z_obs, _ = evaluate_accepted_z_batch(
            model,
            tokenizer,
            requests,
            cases,
            role=role,
            round_index=1,
            binding=binding,
            committed_weight_sha256=_hashes(touched),
            model_alias=alias,
        )
        w_scores = _evaluate_w(
            model, tokenizer, cases, alias=alias, freeze=freeze, ledger=ledger
        )
        z_summary = _endpoint_summary(z_obs["scores"])
        w_summary = _endpoint_summary(w_scores)
        return {
            "kind": kind,
            "apply": apply_payload,
            "accepted_z": binding.raw_free_payload(),
            "z": {"summary": z_summary, "scores": z_obs["scores"]},
            "W": {"summary": w_summary, "scores": w_scores},
            "gap": _writer_gap(w_summary, z_summary),
            "update_energy": fp32_weight_energy(touched, base_values),
            "module_state_restored": bool(state.get("restored", True)),
            "wall_seconds": time.perf_counter() - started,
        }


def run_p3r1_case(
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
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
    job_ledger: ComputeLedger,
    request_microbatch_size: int,
    fp32_runtime: Any | None = None,
    **_: Any,
) -> Mapping[str, Any]:
    case_index = role_case_index(role)
    if (
        len(stream_batches) != 10
        or any(len(batch) != 100 for batch in stream_batches)
        or stream.get("root_digest") != STREAM_ROOT
        or stream.get("all_request_order_sha256") != STREAM_ORDER
        or fp32_runtime is None
        or any(
            parameter.dtype is not torch.float32
            for parameter in model.parameters()
            if parameter.is_floating_point()
        )
        or torch.is_autocast_enabled()
        or torch.is_autocast_enabled("cpu")
    ):
        raise ODEBFContractError("P3R1 runtime boundary differs")
    requests = tuple(stream_batches[case_index - 1])
    request_order = scalable_ordered_request_digest(
        [str(item["request_sha256"]) for item in requests]
    )
    case_root = raw_root / f"case-{case_index:02d}"
    case_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    seed_all(COMMON_SEED)
    entry_contract = _model_w0_contract(touched)
    entry_hashes = _hashes(touched)
    entry_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    if entry_hashes != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("P3R1 W0 entry differs")
    verify_p1r24_alphaedit_geometry(
        hparams,
        P1R24AliasTargetLock.for_alias(alias),
        easyedit_root=Path("/mnt/raid5/janghj/EasyEdit"),
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
    kl_plan = build_p1r24_kl_plan(
        tokenizer,
        requests,
        request_order_sha256=request_order,
        request_microbatch_size=min(request_microbatch_size, len(requests)),
        fact_token_strategy=hparams.fact_token,
    )
    if objective_plan.request_order_sha256 != request_order or capture_plan.request_order_sha256 != request_order:
        raise ODEBFStateError("P3R1 plan order differs")
    physical0 = capture_scalable_physical_state(model, capture_plan, hparams)
    initial = initial_target_from_capture(physical0)
    target_origin = initial.target_z.detach().cpu().float().contiguous()
    _, teacher = evaluate_p1r24_kl(model, kl_plan, teacher_log_probs=None)
    freeze = EndpointActionFreeze(
        arm=role,
        sequential_batch=case_index - 1,
        request_order_sha256=request_order,
        selected_snapshot_sha256=canonical_hash({"source": source_head, "case": case_index}),
        fixed_budget_slots_completed=OUTER_COUNT,
    )
    cases = load_counterfact_cases_after_freeze(
        dataset_path, requests, freeze, expected_batch_size=100
    )
    pre_edit_scores = _evaluate_w(
        model, tokenizer, cases, alias=alias, freeze=freeze, ledger=job_ledger
    )
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p3r1-two-timescale-fastz-fh-case/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "source_head": source_head,
        "alias": alias,
        "case_index": case_index,
        "request_count": 100,
        "stream_root": STREAM_ROOT,
        "stream_order": STREAM_ORDER,
        "request_order_sha256": request_order,
        "W0_sha256": entry_hashes,
        "target_origin_sha256": tensor_sha256(target_origin),
        "pre_edit": {"summary": _endpoint_summary(pre_edit_scores), "scores": pre_edit_scores},
        "target_only": None,
        "arms": {},
        "baselines": {},
    }
    if case_index == 1:
        target_only: dict[str, Any] = {}
        for count in (PRODUCTION_INNER_COUNT, TARGET_ONLY_REFERENCE_INNER_COUNT):
            result = _target_solver(
                model,
                alias=alias,
                current_target=target_origin,
                current_terminal=physical0.terminal_z.detach().cpu().float().contiguous(),
                target_origin=target_origin,
                inner_count=count,
                outer_step_index=None,
                objective_plan=objective_plan,
                kl_plan=kl_plan,
                teacher=teacher,
                hparams=hparams,
                touched=touched,
            )
            binding = r52_binding(f"p3r1-target-only-m{count}", requests, hparams, result.final_target)
            z_obs, _ = evaluate_accepted_z_batch(
                model,
                tokenizer,
                requests,
                cases,
                role=f"p3r1-target-only-m{count}",
                round_index=1,
                binding=binding,
                committed_weight_sha256=_hashes(touched),
                model_alias=alias,
            )
            target_only[f"M{count}"] = {
                "target": result.receipt,
                "inner": list(result.inner_receipts),
                "binding": binding.raw_free_payload(),
                "z": {"summary": _endpoint_summary(z_obs["scores"]), "scores": z_obs["scores"]},
                "writer_call_count": 0,
            }
        payload["target_only"] = target_only
    for arm in ("C0-FH", "C1-FH", "C3-FH"):
        if _hashes(touched) != entry_hashes:
            raise ODEBFStateError("P3R1 arm entry W differs")
        result = _run_dynamic_arm(
            model,
            tokenizer,
            requests,
            alias=alias,
            arm=arm,
            case_root=case_root,
            hparams=hparams,
            projector=projector,
            contexts=contexts,
            covariance_registry=covariance_registry,
            projector_sha256=projector_sha256,
            controller_lock=controller_lock,
            objective_plan=objective_plan,
            capture_plan=capture_plan,
            kl_plan=kl_plan,
            teacher=teacher,
            target_origin=target_origin,
            touched=touched,
            base_values=base_values,
        )
        binding = r52_binding(role, requests, hparams, result["terminal_target"])
        z_obs, _ = evaluate_accepted_z_batch(
            model,
            tokenizer,
            requests,
            cases,
            role=role,
            round_index=1,
            binding=binding,
            committed_weight_sha256=_hashes(touched),
            model_alias=alias,
        )
        w_scores = _evaluate_w(
            model, tokenizer, cases, alias=alias, freeze=freeze, ledger=job_ledger
        )
        z_summary = _endpoint_summary(z_obs["scores"])
        w_summary = _endpoint_summary(w_scores)
        arm_payload = dict(result)
        arm_payload.update(
            {
                "z": {"summary": z_summary, "scores": z_obs["scores"]},
                "W": {"summary": w_summary, "scores": w_scores},
                "gap": _writer_gap(w_summary, z_summary),
            }
        )
        arm_payload["restore"] = _restore(
            touched, base_values, mutation_lock=mutation_lock, entry_contract=entry_contract
        )
        payload["arms"][arm] = arm_payload
    if case_index == 1:
        for kind in ("Official-AlphaEdit", "Official-MEMIT"):
            if _hashes(touched) != entry_hashes:
                raise ODEBFStateError("P3R1 baseline entry W differs")
            baseline = dict(
                _native_baseline(
                    model,
                    tokenizer,
                    requests,
                    cases,
                    alias=alias,
                    kind=kind,
                    hparams=hparams,
                    touched=touched,
                    base_values=base_values,
                    freeze=freeze,
                    ledger=job_ledger,
                )
            )
            baseline["restore"] = _restore(
                touched, base_values, mutation_lock=mutation_lock, entry_contract=entry_contract
            )
            payload["baselines"][kind] = baseline
    if (
        _hashes(touched) != entry_hashes
        or any(int(touched[name].data_ptr()) != entry_pointers[name] for name in touched)
    ):
        raise ODEBFStateError("P3R1 terminal W0 restore differs")
    payload.update(
        {
            "dtype_contract": {
                "model_storage": "torch.float32",
                "target": "torch.float32",
                "planner": "torch.float32",
                "solve": "torch.float32",
                "update": "torch.float32",
                "autocast_count": 0,
                "bf16_conversion_count": 0,
                "fp16_conversion_count": 0,
                "numeric_storage_cast_count": 0,
                "bf16_materializer_call_count": 0,
            },
            "fixed_M": PRODUCTION_INNER_COUNT,
            "K": OUTER_COUNT,
            "target_selection_policy": "FIXED_M_FINAL_ITERATE",
            "primary_rescue_current_count": 0,
            "early_first_hit_retry_backtracking_count": 0,
            "heldout_decision_influence_count": 0,
            "imputation_count": 0,
            "W0_restored": True,
            "job_compute": job_ledger.raw_free_payload(),
            "scientific_promotion": False,
        }
    )
    raw_free, tensor_paths = _raw_free_json_tree(payload)
    raw_free["raw_free_tensor_adapter"] = {
        "replacement_count": len(tensor_paths),
        "paths": list(tensor_paths),
        "decision_influence_count": 0,
    }
    raw_free["identity_sha256"] = canonical_hash(raw_free)
    case_sha = _atomic_write_once(case_root / "terminal.json", raw_free)
    terminal = {
        "schema": "ode-edit-s05-p3r1-case-terminal/v1",
        "status": "TERMINAL_VALID",
        "alias": alias,
        "case_index": case_index,
        "source_head": source_head,
        "request_attempt_count": 100,
        "valid_request_count": 100,
        "technical_failure_count": 0,
        "typed_scientific_failure_count": 0,
        "case_terminal_sha256": case_sha,
        "W0_restored": True,
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p3r1-case-manifest/v1",
        "source_head": source_head,
        "alias": alias,
        "case_index": case_index,
        "terminal_sha256": terminal_sha,
        "case_terminal_sha256": case_sha,
        "members": ["terminal.json", f"raw/case-{case_index:02d}/terminal.json"],
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    stages.record(
        "p3r1_terminal",
        {"alias": alias, "case_index": case_index, "W0_restored": True},
    )
    return {
        "status": "P3R1_TERMINAL_VALID",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
    }


__all__ = [
    "RESULT_PREFIX",
    "RESULT_PARENT_NAME",
    "ROLE_PREFIX",
    "case_role",
    "expected_result_name",
    "expected_result_parent",
    "is_p3r1_role",
    "role_case_index",
    "run_p3r1_case",
]
