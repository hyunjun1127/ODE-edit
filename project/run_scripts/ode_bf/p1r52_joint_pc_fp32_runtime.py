"""Full-FP32 one-B100 P1R52 Joint-P/C writer comparison."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .alpha_backend import seed_all
from .contracts import COMMON_SEED, ODEBFContractError, ODEBFStateError, canonical_hash
from .fixed_e8_soft_routing import FixedE8Arm
from .functional import WaypointFactor, tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_evaluator import EndpointActionFreeze, load_counterfact_cases_after_freeze
from .p1_replay import build_outer_entry_pretrained_cache
from .p1_runtime import ArmRuntimeState, _atomic_write_once, _entry_parameter_snapshot_sha256
from .p1_scalable_batched_experiment import _model_w0_contract, _run_ode_arm
from .p1_state import ArmWeightSnapshot, P1Arm, P1HistoryLedger
from .p1r36_independent_b10x10_runtime import _hashes, _restore_exact_w0
from .p1r52_accepted_z_observation import (
    OfficialNativeZCapture,
    evaluate_accepted_z_batch,
    r52_binding,
)
from .p1r52_frozen_pi_quota_writer import build_current_layer_field
from .p1r52_joint_pc_execution import JointPCWriterArm
from .p1r52_joint_pc_runtime import STREAM_ORDER, STREAM_ROOT, _writer_entry
from .p1r52_official_sequential_baselines import (
    load_official_memit_hparams,
    run_official_memit_apply,
)
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
    _evaluate_w,
    _writer_gap,
    accepted_z_cache_template,
    isolated_alphaedit_module_state,
)
from .scalable_batched_model import (
    build_scalable_capture_plan,
    build_scalable_objective_plan,
    capture_scalable_physical_state,
)
from .scalable_batched_native import run_official_native_apply
from .scalable_batched_runtime import (
    P1R23_GRID_COUNT,
    P1R23_H,
    P1R23_LAYER_ORDER,
    scalable_ordered_request_digest,
)


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-JOINT-PC-C0-C1-C2-C3-FULL-FP32-B100-V1"
METHOD_ID = "P1R52-JOINT-PC-C0-C1-C2-C3-FULL-FP32"
ROLE = "r52-joint-pc-c0-c1-c2-c3-full-fp32-b100-tech-r2"
RESULT_NAME = "s05-p1r52-joint-pc-c0-c1-c2-c3-full-fp32-b100-tech-r2-v1"


def is_joint_pc_full_fp32_role(role: str | None) -> bool:
    return role == ROLE


def _raw_free_json_tree(value: Any, *, path: str = "$") -> tuple[Any, tuple[str, ...]]:
    """Replace leaked receipt tensors by raw-free identity metadata.

    This adapter is terminal-serialization plumbing only.  It never runs on a
    controller, target, route, update, evaluator, or model input.  Numeric
    endpoint summaries remain ordinary JSON scalars; an unexpected tensor in
    a nested raw-free receipt is represented solely by its hash/shape/dtype.
    """

    if isinstance(value, torch.Tensor):
        observed = value.detach().contiguous()
        return (
            {
                "schema": "ode-edit-raw-free-tensor-identity/v1",
                "sha256": tensor_sha256(observed),
                "shape": list(observed.shape),
                "dtype": str(observed.dtype),
                "device_class": observed.device.type,
                "serialized_value_count": 0,
            },
            (path,),
        )
    if isinstance(value, Mapping):
        converted: dict[Any, Any] = {}
        paths: list[str] = []
        for key, item in value.items():
            item_value, item_paths = _raw_free_json_tree(
                item, path=f"{path}.{key}"
            )
            converted[key] = item_value
            paths.extend(item_paths)
        return converted, tuple(paths)
    if isinstance(value, (list, tuple)):
        converted_items: list[Any] = []
        paths = []
        for index, item in enumerate(value):
            item_value, item_paths = _raw_free_json_tree(
                item, path=f"{path}[{index}]"
            )
            converted_items.append(item_value)
            paths.extend(item_paths)
        return converted_items, tuple(paths)
    return value, ()


def _entry_state(
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor],
) -> ArmRuntimeState:
    hashes = _hashes(touched)
    if hashes != dict(base_receipt.parameter_sha256):
        raise ODEBFStateError("FP32 Joint-P/C target entry W differs")
    return ArmRuntimeState(
        P1Arm.R_BF,
        P1HistoryLedger(layer_order=P1R23_LAYER_ORDER, maximum_records=400, batch_size=100),
        ComputeLedger(),
        ArmWeightSnapshot(
            P1Arm.R_BF,
            0,
            base_receipt.parameter_sha256,
            canonical_hash({"role": ROLE, "entry": hashes}),
        ),
        dict(base_values),
    )


def _fp32_target_and_j0(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
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
    c_kstep_writer_factory: Any | None = None,
    target_subcycle_schedule: Any | None = None,
    amplitude_policy: Any | None = None,
    target_subcycle_runner: Any | None = None,
    objective_evaluator: Any | None = None,
    realization_controller: Any | None = None,
    easyedit_root: Path = Path("/mnt/raid5/janghj/EasyEdit"),
) -> tuple[torch.Tensor, Mapping[str, Any], Any, Any]:
    order = scalable_ordered_request_digest([str(item["request_sha256"]) for item in requests])
    objective_plan = build_scalable_objective_plan(
        model, tokenizer, requests, contexts=contexts,
        request_microbatch_size=min(request_microbatch_size, len(requests)),
        fact_token_strategy=hparams.fact_token,
    )
    capture_plan = build_scalable_capture_plan(
        tokenizer, requests, contexts=contexts,
        request_microbatch_size=min(request_microbatch_size, len(requests)),
        fact_token_strategy=hparams.fact_token,
    )
    if objective_plan.request_order_sha256 != order or capture_plan.request_order_sha256 != order:
        raise ODEBFStateError("FP32 Joint-P/C target plan order differs")
    outer_population = tuple(population_by_sha256[item] for item in theta0_cache.request_order)
    snapshot = _entry_parameter_snapshot_sha256(model, dict(base_receipt.parameter_sha256))
    counter = ModelForwardCounter(model, job_ledger)
    try:
        outer_entry_p_cache = build_outer_entry_pretrained_cache(
            model, tokenizer, outer_population, theta0_cache,
            outer_entry_snapshot_sha256=snapshot,
        )
    finally:
        counter.close()
    c_kstep_writer = (
        None
        if c_kstep_writer_factory is None
        else c_kstep_writer_factory(objective_plan, capture_plan)
    )
    rollout = _run_ode_arm(
        model, tokenizer, requests,
        alias="llama3-8b-inst", arm=FixedE8Arm.SOFT, allocation="RS",
        capture_plan=capture_plan, objective_plan=objective_plan, hparams=hparams,
        projector=projector, contexts=contexts,
        covariance_registry=covariance_registry, projector_sha256=projector_sha256,
        controller_lock=controller_lock,
        arm_state=_entry_state(touched, base_receipt, base_values),
        request_by_sha256=request_by_sha256,
        population_by_sha256=population_by_sha256,
        schedule=schedule, outer_entry_p_cache=outer_entry_p_cache,
        theta0_cache=theta0_cache, touched=touched, base_receipt=base_receipt,
        base_values=base_values, raw_root=case_root / "raw" / "target",
        write_once=_atomic_write_once, p1r24=True, p1r34=True, p1r35=True,
        p1r52=True, p1r52_fp32_phase_a=True,
        p1r52_phase_a_method_label=(
            "P1R52-J0-FULL-FP32"
            if c_kstep_writer is None else c_kstep_writer.arm
        ),
        p1r52_c_kstep_writer=c_kstep_writer,
        p1r52_target_subcycle_schedule=target_subcycle_schedule,
        p1r52_amplitude_policy=amplitude_policy,
        p1r52_target_subcycle_runner=target_subcycle_runner,
        p1r55_objective_evaluator=objective_evaluator,
        p1r54_realization_controller=realization_controller,
        easyedit_root=easyedit_root,
    )
    public = rollout["public"]
    if public["accepted_update_count"] != P1R23_GRID_COUNT or public["tau_final"] != 1.0:
        raise ODEBFStateError("FP32 Joint-P/C K8 target differs")
    target = rollout["terminal_target"].detach().cpu().float().contiguous()
    return target, public, objective_plan, capture_plan


def _restore(
    touched: Mapping[str, torch.nn.Parameter],
    base_values: Mapping[str, torch.Tensor],
    *, mutation_lock: Any, entry_contract: str,
) -> Mapping[str, Any]:
    return _restore_exact_w0(
        touched, base_values, mutation_lock=mutation_lock,
        expected_contract=entry_contract,
    )


def _bindings(model: torch.nn.Module, hparams: Any) -> dict[int, tuple[str, torch.nn.Parameter]]:
    named = dict(model.named_parameters())
    return {
        layer: (
            f"{hparams.rewrite_module_tmp.format(layer)}.weight",
            named[f"{hparams.rewrite_module_tmp.format(layer)}.weight"],
        )
        for layer in P1R23_LAYER_ORDER
    }


def _run_pc_arm_fp32(
    model: torch.nn.Module,
    arm: JointPCWriterArm,
    *, target: torch.Tensor, entry: Mapping[str, Any], hparams: Any,
    projector: torch.Tensor, covariance_registry: Any, projector_sha256: str,
    controller_lock: Any, capture_plan: Any,
    solve_history_keys_by_layer: Mapping[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if arm not in (JointPCWriterArm.C0, JointPCWriterArm.C1, JointPCWriterArm.C2):
        raise ODEBFContractError("FP32 Joint-P/C arm differs")
    pi = tuple(float(item) for item in (
        entry["control"].pi if arm is JointPCWriterArm.C0 else entry["joint"].pi
    ))
    beta = remaining_pi_beta(pi)
    entry_terminal = entry["physical"].terminal_z.detach().cpu().float().contiguous()
    entry_residual = (target - entry_terminal).contiguous()
    transaction = OfficialStyleFP32SequentialTransaction(
        _bindings(model, hparams), mode=FP32TransactionMode.AUTHORITATIVE,
        transaction_id=f"{ROLE}-{arm.value}",
    )
    layer_rows: list[dict[str, Any]] = []
    capture_receipts: list[dict[str, Any]] = [
        entry["physical"].raw_free_payload()
    ]
    applied = []
    if solve_history_keys_by_layer is not None and (
        set(solve_history_keys_by_layer) != set(P1R23_LAYER_ORDER)
        or any(
            value.ndim != 2
            or value.dtype is not torch.float32
            or not torch.isfinite(value).all()
            for value in solve_history_keys_by_layer.values()
        )
    ):
        raise ODEBFContractError("FP32 Joint-P/C solve-history inventory differs")
    with transaction:
        for ordinal, layer in enumerate(P1R23_LAYER_ORDER):
            if ordinal == 0:
                current_terminal = entry_terminal.clone()
                if solve_history_keys_by_layer is None:
                    field = entry["field"].layers[0]
                    field_source = "ENTRY_FIELD_REUSE"
                else:
                    field, _ = build_current_layer_field(
                        model, hparams, projector, covariance_registry,
                        layer=layer,
                        key=entry["physical"].keys_by_layer[layer],
                        target_state=target,
                        current_terminal=current_terminal,
                        step_index=0,
                        factor_ordinal=ordinal,
                        projector_sha256=projector_sha256,
                        residual_tolerance=controller_lock.residual_tolerance,
                        q_only=True,
                        history_keys=solve_history_keys_by_layer[layer],
                    )
                    field_source = "ENTRY_CURRENT_KEY_Q_WITH_COMMITTED_CACHE"
            else:
                physical = capture_scalable_physical_state(model, capture_plan, hparams)
                capture_receipts.append(physical.raw_free_payload())
                current_terminal = physical.terminal_z.detach().cpu().float().contiguous()
                field_target = (
                    target if arm is not JointPCWriterArm.C2
                    else (current_terminal + entry_residual).contiguous()
                )
                field, _ = build_current_layer_field(
                    model, hparams, projector, covariance_registry,
                    layer=layer, key=physical.keys_by_layer[layer],
                    target_state=field_target, current_terminal=current_terminal,
                    step_index=0, factor_ordinal=ordinal,
                    projector_sha256=projector_sha256,
                    residual_tolerance=controller_lock.residual_tolerance,
                    q_only=True,
                    history_keys=(
                        solve_history_keys_by_layer[layer]
                        if solve_history_keys_by_layer is not None
                        else None
                    ),
                )
                field_source = "CURRENT_PREFIX_KEY_Q"
            if arm is JointPCWriterArm.C2:
                coefficient = pi[ordinal]
                left32 = (coefficient * entry_residual).to(
                    device=field.q.device, dtype=torch.float32
                ).contiguous()
                suffix = None
            else:
                coefficient = float(P1R23_H) * beta[ordinal]
                left32 = (coefficient * field.residual).to(
                    device=field.q.device, dtype=torch.float32
                ).contiguous()
                suffix = math.fsum(pi[ordinal:])
            weight_name, parameter = _bindings(model, hparams)[layer]
            construction = build_prepared_low_rank_update(
                construction_id=f"{ROLE}-{arm.value}-layer-{layer}",
                layer=layer, weight_name=weight_name,
                parameter_shape=tuple(parameter.shape),
                beta_applied_left32=left32.to(parameter.device),
                q_side32=field.q.detach().to(parameter.device, torch.float32).contiguous(),
            )
            application = apply_prepared_low_rank_update(transaction, construction)
            applied.append(application)
            receipt = application.m3a_layer_receipt
            layer_rows.append({
                "layer": layer, "field_source": field_source,
                "pi": pi[ordinal], "beta": beta[ordinal] if suffix is not None else None,
                "suffix_mass": suffix, "applied_coefficient": coefficient,
                "entry_residual_norm": float(torch.linalg.norm(entry_residual.double())),
                "current_residual_norm": float(torch.linalg.norm((target-current_terminal).double())),
                "key_norm": float(torch.linalg.norm(field.key.double())),
                "q_norm": float(torch.linalg.norm(field.q.double())),
                "alpha_cache_history_width": (
                    int(solve_history_keys_by_layer[layer].shape[1])
                    if solve_history_keys_by_layer is not None else 0
                ),
                "alpha_cache_history_sha256": (
                    tensor_sha256(solve_history_keys_by_layer[layer])
                    if solve_history_keys_by_layer is not None else None
                ),
                "update_norm": receipt.actual_post_storage_delta32_norm,
                "update_energy": receipt.actual_post_storage_delta32_energy,
                "storage_dtype": receipt.post_storage_dtype,
                "numeric_storage_cast_count": receipt.numeric_storage_cast_count,
                "bf16_path_call_count": receipt.bf16_path_call_count,
            })
        prepared = transaction.prepare_authoritative_commit()
        transaction_receipt = transaction.commit_prepared(prepared.identity_sha256)
    post = capture_scalable_physical_state(model, capture_plan, hparams)
    total_norm = math.fsum(row["update_norm"] for row in layer_rows)
    total_energy = math.fsum(row["update_energy"] for row in layer_rows)
    for row in layer_rows:
        row["update_norm_share"] = row["update_norm"] / total_norm if total_norm else 0.0
        row["update_energy_share"] = row["update_energy"] / total_energy if total_energy else 0.0
    return {
        "arm": arm.value, "pi": list(pi), "beta": list(beta), "layers": layer_rows,
        "prefix_capture_receipts": capture_receipts,
        "transaction": transaction_receipt.raw_free_payload(),
        "post_terminal_sha256": tensor_sha256(post.terminal_z),
        "post_terminal_residual_norm": float(torch.linalg.norm((target-post.terminal_z.cpu().float()).double())),
        "total_update_norm_sum": total_norm, "total_update_energy": total_energy,
        "numeric_storage_cast_count": sum(row["numeric_storage_cast_count"] for row in layer_rows),
        "bf16_path_call_count": sum(row["bf16_path_call_count"] for row in layer_rows),
        "fp16_conversion_count": 0, "autocast_count": 0,
    }


def _native_baseline(
    model: torch.nn.Module, tokenizer: Any, requests: Sequence[Mapping[str, Any]],
    cases: Sequence[Any], freeze: EndpointActionFreeze, *, kind: str,
    hparams: Any, touched: Mapping[str, torch.nn.Parameter],
    base_values: Mapping[str, torch.Tensor], mutation_lock: Any,
    entry_contract: str, job_ledger: ComputeLedger,
) -> dict[str, Any]:
    if kind == "official-alphaedit":
        role = "native-alphaedit-sequential-cache-on-corrected"
        native_hparams = hparams
        apply = lambda: run_official_native_apply(
            model, tokenizer, requests, native_hparams, touched=touched,
            reset_cache=True, cache_history_width=0,
        )
        context = isolated_alphaedit_module_state()
    elif kind == "official-memit":
        role = "official-memit-sequential"
        native_hparams = load_official_memit_hparams()
        apply = lambda: run_official_memit_apply(
            model, tokenizer, requests, native_hparams, touched=touched,
        )
        from contextlib import nullcontext
        context = nullcontext({"restored": True})
    else:
        raise ODEBFContractError("FP32 baseline kind differs")
    with context as module_state:
        with OfficialNativeZCapture(role=role, requests=requests, hparams=native_hparams) as capture:
            counter = ModelForwardCounter(model, job_ledger)
            try:
                apply_payload, _ = apply()
            finally:
                counter.close()
        binding = capture.finalize()
        counter = ModelForwardCounter(model, job_ledger)
        try:
            z_observation, _ = evaluate_accepted_z_batch(
                model, tokenizer, requests, cases, role=role, round_index=1,
                binding=binding, committed_weight_sha256=_hashes(touched),
            )
        finally:
            counter.close()
        scores = _evaluate_w(model, tokenizer, cases, freeze=freeze, ledger=job_ledger)
        energy = fp32_weight_energy(touched, base_values)
        result = {
            "accepted_z": binding.raw_free_payload(),
            "z": {"summary": _endpoint_summary(z_observation["scores"]), "scores": z_observation["scores"]},
            "W": {"summary": _endpoint_summary(scores), "scores": scores},
            "apply": apply_payload, "update_energy": energy,
            "model_storage_dtype": "torch.float32", "numeric_storage_cast_count": 0,
            "bf16_path_call_count": 0, "fp16_conversion_count": 0,
            "module_state_restored": bool(module_state.get("restored", True)),
        }
        result["gap"] = _writer_gap(result["W"]["summary"], result["z"]["summary"])
        result["restore"] = _restore(
            touched, base_values, mutation_lock=mutation_lock, entry_contract=entry_contract,
        )
        return result


def run_joint_pc_full_fp32_b100(
    model: torch.nn.Module, tokenizer: Any, *, alias: str, role: str,
    destination: Path, raw_root: Path, stages: Any, source_head: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]], stream: Mapping[str, Any],
    hparams: Any, projector: torch.Tensor, contexts: Sequence[Sequence[str]],
    covariance_registry: Any, projector_sha256: str, controller_lock: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]], schedule: Any,
    theta0_cache: Any, dataset_path: Path, mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter], base_receipt: ArmWeightSnapshot,
    base_values: Mapping[str, torch.Tensor], job_ledger: ComputeLedger,
    request_microbatch_size: int, fp32_runtime: Any | None = None, **_: Any,
) -> dict[str, Any]:
    if (
        alias != "llama3-8b-inst" or role != ROLE or len(stream_batches) != 10
        or any(len(batch) != 100 for batch in stream_batches)
        or stream.get("root_digest") != STREAM_ROOT
        or stream.get("all_request_order_sha256") != STREAM_ORDER
    ):
        raise ODEBFContractError("FP32 Joint-P/C matrix differs")
    if fp32_runtime is None or any(
        parameter.dtype is not torch.float32 for parameter in model.parameters()
        if parameter.is_floating_point()
    ) or torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        raise ODEBFContractError("FP32 Joint-P/C model boundary differs")
    requests = tuple(stream_batches[0])
    case_root = raw_root / "case-01"
    case_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    seed_all(COMMON_SEED)
    entry_contract = _model_w0_contract(touched)
    entry_hashes = _hashes(touched)
    entry_pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    order = scalable_ordered_request_digest([str(item["request_sha256"]) for item in requests])

    target, target_public, objective_plan, capture_plan = _fp32_target_and_j0(
        model, tokenizer, requests, case_root=case_root, hparams=hparams,
        projector=projector, contexts=contexts, covariance_registry=covariance_registry,
        projector_sha256=projector_sha256, controller_lock=controller_lock,
        request_by_sha256=request_by_sha256, population_by_sha256=population_by_sha256,
        schedule=schedule, theta0_cache=theta0_cache, touched=touched,
        base_receipt=base_receipt, base_values=base_values,
        request_microbatch_size=request_microbatch_size, job_ledger=job_ledger,
    )
    target_hash = tensor_sha256(target)
    freeze = EndpointActionFreeze(
        arm=ROLE, sequential_batch=0, request_order_sha256=order,
        selected_snapshot_sha256=target_public["identity_sha256"],
        fixed_budget_slots_completed=P1R23_GRID_COUNT,
    )
    cases = load_counterfact_cases_after_freeze(
        dataset_path, requests, freeze, expected_batch_size=100
    )
    binding = r52_binding(ROLE, requests, hparams, target)
    counter = ModelForwardCounter(model, job_ledger)
    try:
        z_obs, _ = evaluate_accepted_z_batch(
            model, tokenizer, requests, cases, role=ROLE, round_index=1,
            binding=binding, committed_weight_sha256=_hashes(touched),
        )
    finally:
        counter.close()
    z_summary = _endpoint_summary(z_obs["scores"])
    j0_scores = _evaluate_w(model, tokenizer, cases, freeze=freeze, ledger=job_ledger)
    j0 = {
        "summary": _endpoint_summary(j0_scores), "scores": j0_scores,
        "update_energy": fp32_weight_energy(touched, base_values),
        "model_storage_dtype": "torch.float32", "bf16_path_call_count": 0,
    }
    j0["gap"] = _writer_gap(j0["summary"], z_summary)
    j0["restore"] = _restore(
        touched, base_values, mutation_lock=mutation_lock, entry_contract=entry_contract,
    )

    pre_scores = _evaluate_w(model, tokenizer, cases, freeze=freeze, ledger=job_ledger)
    pre_edit = {"summary": _endpoint_summary(pre_scores), "scores": pre_scores}
    official_alpha = _native_baseline(
        model, tokenizer, requests, cases, freeze, kind="official-alphaedit",
        hparams=hparams, touched=touched, base_values=base_values,
        mutation_lock=mutation_lock, entry_contract=entry_contract, job_ledger=job_ledger,
    )
    official_memit = _native_baseline(
        model, tokenizer, requests, cases, freeze, kind="official-memit",
        hparams=hparams, touched=touched, base_values=base_values,
        mutation_lock=mutation_lock, entry_contract=entry_contract, job_ledger=job_ledger,
    )

    entry = _writer_entry(
        model, tokenizer, requests, target=target, hparams=hparams,
        projector=projector, contexts=contexts, covariance_registry=covariance_registry,
        projector_sha256=projector_sha256, controller_lock=controller_lock,
        objective_plan=objective_plan, capture_plan=capture_plan,
    )
    arms: dict[str, Any] = {}
    arm_entry_hashes: dict[str, Mapping[str, str]] = {}
    for arm in (JointPCWriterArm.C0, JointPCWriterArm.C1, JointPCWriterArm.C2):
        arm_entry_hashes[arm.value] = _hashes(touched)
        writer = _run_pc_arm_fp32(
            model, arm, target=target, entry=entry, hparams=hparams,
            projector=projector, covariance_registry=covariance_registry,
            projector_sha256=projector_sha256, controller_lock=controller_lock,
            capture_plan=capture_plan,
        )
        scores = _evaluate_w(model, tokenizer, cases, freeze=freeze, ledger=job_ledger)
        summary = _endpoint_summary(scores)
        arms[arm.value] = {
            "writer": writer, "summary": summary, "scores": scores,
            "gap": _writer_gap(summary, z_summary),
            "update_energy": fp32_weight_energy(touched, base_values),
        }
        arms[arm.value]["restore"] = _restore(
            touched, base_values, mutation_lock=mutation_lock, entry_contract=entry_contract,
        )

    arm_entry_hashes["C3-DIRECT-OFFICIAL-ALPHAEDIT"] = _hashes(touched)
    with isolated_alphaedit_module_state() as alpha_state:
        with accepted_z_cache_template(
            requests, target, hparams, parent=case_root / "private"
        ) as (cache_template, bridge):
            counter = ModelForwardCounter(model, job_ledger)
            try:
                c3_apply, originals = run_official_native_apply(
                    model, tokenizer, requests, hparams, touched=touched,
                    reset_cache=True, cache_history_width=0,
                    cache_template=cache_template,
                    expected_native_compute_z_call_count=0,
                    accepted_z_source="P1R52_K8_TERMINAL_TARGET",
                )
            finally:
                counter.close()
            if any(tensor_sha256(originals[name]) != entry_hashes[name] for name in touched):
                raise ODEBFStateError("C3 Official AlphaEdit entry differs")
            c3_scores = _evaluate_w(model, tokenizer, cases, freeze=freeze, ledger=job_ledger)
            c3_summary = _endpoint_summary(c3_scores)
            arms["C3-DIRECT-OFFICIAL-ALPHAEDIT"] = {
                "writer": {
                    "official_entrypoint": c3_apply["official_entrypoint"],
                    "official_source_file": str(
                        __import__(
                            "easyeditor.models.alphaedit.AlphaEdit_main",
                            fromlist=["__file__"],
                        ).__file__
                    ),
                    "apply": c3_apply, "accepted_z_bridge": bridge,
                    "p1r52_pc_router_decision_influence_count": 0,
                    "p1r52_barrier_decision_influence_count": 0,
                    "native_compute_z_call_count": 0,
                },
                "summary": c3_summary, "scores": c3_scores,
                "gap": _writer_gap(c3_summary, z_summary),
                "update_energy": fp32_weight_energy(touched, base_values),
            }
            arms["C3-DIRECT-OFFICIAL-ALPHAEDIT"]["restore"] = _restore(
                touched, base_values, mutation_lock=mutation_lock,
                entry_contract=entry_contract,
            )
    if not alpha_state["restored"]:
        raise ODEBFStateError("C3 Official AlphaEdit module state did not restore")

    if (
        len({canonical_hash(item) for item in arm_entry_hashes.values()}) != 1
        or any(value != entry_hashes for value in arm_entry_hashes.values())
        or _hashes(touched) != entry_hashes
        or any(int(touched[name].data_ptr()) != entry_pointers[name] for name in touched)
    ):
        raise ODEBFStateError("FP32 Joint-P/C arm isolation differs")
    payload = {
        "schema": "ode-edit-s05-p1r52-joint-pc-c0-c1-c2-c3-full-fp32-b100/v1",
        "instruction_id": INSTRUCTION_ID, "method_id": METHOD_ID,
        "source_head": source_head, "request_count": 100,
        "stream_root": STREAM_ROOT, "stream_order": STREAM_ORDER,
        "request_order_sha256": order, "accepted_z_sha256": target_hash,
        "accepted_z_hash_exact_across_c0_c1_c2_c3": True,
        "writer_entry_W0_hash_exact_across_c0_c1_c2_c3": True,
        "target_compute_count": 1,
        "dtype_contract": {
            "model_storage": "torch.float32", "planner": "torch.float32",
            "solve": "torch.float32", "update": "torch.float32",
            "autocast_count": 0, "bf16_conversion_count": 0,
            "fp16_conversion_count": 0, "numeric_storage_cast_count": 0,
            "bf16_materializer_call_count": 0,
            "python_runtime_patch_compatibility": "3.12.x",
        },
        "pre_edit": pre_edit,
        "official_memit": official_memit,
        "official_alphaedit": official_alpha,
        "p1r52_accepted_z": {"binding": binding.raw_free_payload(), "summary": z_summary, "scores": z_obs["scores"]},
        "p1r52_j0": j0,
        "writer_entry": {
            "control_pi": list(entry["control"].pi),
            "joint_pi": list(entry["joint"].pi),
            "joint_route": entry["joint"].receipt.raw_free_payload(),
            "entry_nll": entry["entry_nll"], "endpoint_nll": entry["endpoint_nll"],
        },
        "arms": arms, "arm_entry_hashes": arm_entry_hashes,
        "W0_restored": True, "retry_count": 0, "imputation_count": 0,
        "job_compute": job_ledger.raw_free_payload(), "scientific_promotion": False,
    }
    payload, tensor_paths = _raw_free_json_tree(payload)
    payload["terminal_raw_free_tensor_adapter"] = {
        "schema": "ode-edit-s05-p1r52-joint-pc-terminal-tensor-adapter/v1",
        "replacement_count": len(tensor_paths),
        "paths": list(tensor_paths),
        "serialized_tensor_value_count": 0,
        "scientific_decision_influence_count": 0,
        "model_forward_backward_count": 0,
        "model_weight_mutation_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    case_sha = _atomic_write_once(case_root / "terminal.json", payload)
    terminal = {
        "schema": "ode-edit-s05-p1r52-joint-pc-c0-c1-c2-c3-full-fp32-b100-terminal/v1",
        "status": "TERMINAL_VALID", "source_head": source_head,
        "request_attempt_count": 100, "valid_request_count": 100,
        "technical_failure_count": 0, "typed_scientific_failure_count": 0,
        "case_terminal_sha256": case_sha, "W0_restored": True,
        "scientific_promotion": False,
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r52-joint-pc-c0-c1-c2-c3-full-fp32-b100-manifest/v1",
        "source_head": source_head, "terminal_sha256": terminal_sha,
        "case_terminal_sha256": case_sha, "members": ["terminal.json", "raw/case-01/terminal.json"],
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    stages.record("joint_pc_full_fp32_terminal", {"request_count": 100, "arm_count": 4, "W0_restored": True})
    return {"status": "P1R52_JOINT_PC_FULL_FP32_TERMINAL", "terminal_sha256": terminal_sha, "manifest_sha256": manifest_sha}


__all__ = [
    "INSTRUCTION_ID", "METHOD_ID", "RESULT_NAME", "ROLE",
    "is_joint_pc_full_fp32_role", "run_joint_pc_full_fp32_b100",
]
