"""Official AlphaEdit velocity plus a strength-neutral barrier correction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor.util import nethook
from easyeditor.util.device import copy_to_param, get_module_device, normalize_device
from easyeditor.models.alphaedit import AlphaEdit_main as official
from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams
from easyeditor.models.alphaedit.compute_ks import compute_ks
from easyeditor.models.alphaedit.compute_z import compute_z, get_module_input_output_at_words
from .geometry import (
    CorrectionSolution,
    FactorTerm,
    LowRankFactor,
    RightProjectionReceipt,
    apply_factor_terms_,
    factor_terms_times_keys,
    project_right,
    solve_strength_neutral_correction,
)
from .contracts import require_right_padding
from .target_path import (
    TargetEventBatch,
    TargetPathReference,
    TargetPathValues,
    build_target_event_batch,
    build_sequence_event_batch,
    capture_reference,
    evaluate_token_nll,
    evaluate_values,
    sequence_nll_by_request,
)
from .telemetry import (
    LayerTelemetry,
    NodeTelemetry,
    StateObservationTelemetry,
    WriterTelemetry,
    write_create_once,
)
from .official_state import prepare_official_state
from .endpoint_adoption import (
    adopt_exact_endpoint,
    capture_exact_endpoint,
    tensor_sha,
)


class BarrierArm(str, Enum):
    OFFICIAL = "OFFICIAL_ALPHAEDIT"
    SPLIT = "SPLIT_ALPHAEDIT"
    BARRIER = "STRENGTH_NEUTRAL_BARRIER_ODE"


@dataclass(frozen=True)
class BarrierWriterConfig:
    arm: BarrierArm
    steps: int
    telemetry_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.arm is BarrierArm.OFFICIAL and self.steps != 1:
            raise ValueError("Official AlphaEdit is the explicit N=1 bypass")
        if self.arm is not BarrierArm.OFFICIAL and self.steps < 2:
            raise ValueError("split/barrier writers require N>=2")


def _normalize_requests(requests: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = deepcopy(list(requests))
    for index, request in enumerate(normalized):
        if not request["target_new"]:
            raise ValueError("target_new is empty")
        if request["target_new"][0] != " ":
            request["target_new"] = " " + request["target_new"]
        if "{}" not in request["prompt"]:
            if request["subject"] not in request["prompt"]:
                raise ValueError(
                    f"subject {request['subject']!r} is absent from rewrite prompt"
                )
            request["prompt"] = request["prompt"].replace(request["subject"], "{}")
        print(
            "Executing strength-neutral AlphaEdit for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
        normalized[index] = request
    return normalized


def _selected_sha(weights: Dict[str, torch.Tensor]) -> str:
    digest = sha256()
    for name in sorted(weights):
        value = weights[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _weight_component_sha(name: str, value: torch.Tensor) -> str:
    digest = sha256()
    tensor = value.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _selected_component_root(components: Dict[str, str]) -> str:
    digest = sha256()
    digest.update(b"selected-weight-component-root.v1\0")
    for name in sorted(components):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(components[name].encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _assert_full_fp32(model: AutoModelForCausalLM) -> None:
    wrong = [
        (name, str(parameter.dtype))
        for name, parameter in model.named_parameters()
        if parameter.dtype != torch.float32
    ]
    if wrong:
        preview = ", ".join(f"{name}:{dtype}" for name, dtype in wrong[:8])
        raise RuntimeError(f"FULL_FP32 model gate failed ({len(wrong)}): {preview}")
    if getattr(model, "is_quantized", False):
        raise RuntimeError("quantized models are forbidden")


def _last_logits(
    model: AutoModelForCausalLM, batch: TargetEventBatch
) -> torch.Tensor:
    return model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
    ).logits[:, -1, :]


def _observe_values(
    model: AutoModelForCausalLM,
    batch: TargetEventBatch,
    reference: TargetPathReference,
) -> TargetPathValues:
    with torch.no_grad():
        return evaluate_values(_last_logits(model, batch), batch, reference)


def _source_sequence_nll(
    model: AutoModelForCausalLM, source_batch: TargetEventBatch
) -> torch.Tensor:
    with torch.no_grad():
        token_nll = evaluate_token_nll(_last_logits(model, source_batch), source_batch)
    return sequence_nll_by_request(token_nll, source_batch)


def _state_observation(
    values: TargetPathValues,
    target_batch: TargetEventBatch,
    source_sequence_nll: torch.Tensor,
) -> StateObservationTelemetry:
    target_sequence_nll = sequence_nll_by_request(values.target_nll, target_batch)
    margin = source_sequence_nll - target_sequence_nll
    return StateObservationTelemetry(
        per_event_q_kl=[float(value) for value in values.per_event_kl.tolist()],
        aggregate_q_kl=float(values.barrier.item()),
        target_token_log_odds=[
            float(value) for value in values.target_log_odds.tolist()
        ],
        target_token_nll=[float(value) for value in values.target_nll.tolist()],
        target_sequence_nll=[float(value) for value in target_sequence_nll.tolist()],
        source_sequence_nll=[float(value) for value in source_sequence_nll.tolist()],
        target_source_margin=[float(value) for value in margin.tolist()],
    )


@dataclass(frozen=True)
class _GradientObservation:
    values: TargetPathValues
    barrier_gradient: LowRankFactor
    target_gradients: tuple[LowRankFactor, ...]
    common_right: torch.Tensor


def _gradient_observation(
    model: AutoModelForCausalLM,
    module_name: str,
    weight: torch.Tensor,
    batch: TargetEventBatch,
    reference: TargetPathReference,
) -> _GradientObservation:
    original_requires_grad = bool(weight.requires_grad)
    weight.requires_grad_(True)
    try:
        with torch.enable_grad(), nethook.Trace(
            model,
            module_name,
            retain_input=True,
            retain_output=True,
            clone=False,
            detach=False,
        ) as trace:
            values = evaluate_values(_last_logits(model, batch), batch, reference)
        layer_input = trace.input
        layer_output = trace.output
        if not isinstance(layer_input, torch.Tensor) or not isinstance(
            layer_output, torch.Tensor
        ):
            raise RuntimeError("AlphaEdit rewrite module must have tensor input/output")
        barrier_output_grad = torch.autograd.grad(
            values.barrier, layer_output, retain_graph=True, create_graph=False
        )[0]
        target_output_grads = []
        for index in range(batch.event_count):
            target_output_grads.append(
                torch.autograd.grad(
                    values.target_log_odds[index],
                    layer_output,
                    retain_graph=index + 1 < batch.event_count,
                    create_graph=False,
                )[0]
            )
        common_right = layer_input.detach().reshape(-1, layer_input.shape[-1]).T.float()
        barrier_left = (
            barrier_output_grad.detach()
            .reshape(-1, barrier_output_grad.shape[-1])
            .T.float()
        )
        target_left = [
            gradient.detach().reshape(-1, gradient.shape[-1]).T.float()
            for gradient in target_output_grads
        ]
        return _GradientObservation(
            values=values,
            barrier_gradient=LowRankFactor(barrier_left, common_right),
            target_gradients=tuple(
                LowRankFactor(left, common_right) for left in target_left
            ),
            common_right=common_right,
        )
    finally:
        weight.requires_grad_(original_requires_grad)


def _native_layer_delta(
    *,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Sequence[Dict[str, Any]],
    hparams: AlphaEditHyperParams,
    context_templates: List[List[str]],
    zs: torch.Tensor,
    layer_index: int,
    layer: int,
    projector: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    layer_ks = compute_ks(
        model, tok, requests, hparams, layer, context_templates
    ).T
    z_layer = hparams.layers[-1]
    cur_zs = get_module_input_output_at_words(
        model,
        tok,
        z_layer,
        context_templates=[request["prompt"] for request in requests],
        words=[request["subject"] for request in requests],
        module_template=hparams.layer_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )[1].T
    cur_zs = cur_zs.to(device=zs.device, dtype=zs.dtype)
    targets = zs - cur_zs
    repeat_factor = layer_ks.size(1) // targets.size(1)
    if repeat_factor * targets.size(1) != layer_ks.size(1):
        raise RuntimeError("AlphaEdit key/target repeat factor is nonintegral")
    repeated = targets.repeat_interleave(repeat_factor, dim=1)
    residual = repeated / (len(hparams.layers) - layer_index)
    proj = projector.to(device).float()
    keys = layer_ks.to(device).float()
    residual = residual.to(device).float()
    history = cache.to(device).float()
    solve_matrix = proj @ (keys @ keys.T + history)
    solve_matrix = solve_matrix + hparams.L2 * torch.eye(
        keys.shape[0], dtype=torch.float32, device=device
    )
    update = torch.linalg.solve(solve_matrix, proj @ keys @ residual.T)
    update = official.upd_matrix_match_shape(update, weight.shape).float()
    return update, keys, float(torch.linalg.vector_norm(targets).item())


def _terminal_z_residual(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Sequence[Dict[str, Any]],
    hparams: AlphaEditHyperParams,
    zs: torch.Tensor,
) -> float:
    current = get_module_input_output_at_words(
        model,
        tok,
        hparams.layers[-1],
        context_templates=[request["prompt"] for request in requests],
        words=[request["subject"] for request in requests],
        module_template=hparams.layer_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )[1].T.to(zs)
    return float(torch.linalg.vector_norm(zs - current).item())


def _zero_solution(native_rate: float = 0.0) -> CorrectionSolution:
    return CorrectionSolution(
        terms=(),
        native_barrier_rate=native_rate,
        feasible_gradient_norm=0.0,
        correction_norm=0.0,
        directional_rate=native_rate,
        target_strength_inner_abs=(),
        max_strength_inner_abs=0.0,
        key_residual_max_abs=0.0,
        constraint_rank=0,
        constraint_condition=1.0,
        constraint_pinv_rtol=0.0,
        active=False,
    )


def _execute_guided(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Sequence[Dict[str, Any]],
    hparams: AlphaEditHyperParams,
    config: BarrierWriterConfig,
    *,
    cache_template: Optional[str],
) -> tuple[Dict[str, torch.Tensor], WriterTelemetry]:
    require_right_padding(tok, caller="guided AlphaEdit writer")
    device = normalize_device(getattr(hparams, "device", None))
    normalized = _normalize_requests(requests)
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    entry_weights = {name: weight.detach().clone() for name, weight in weights.items()}
    entry_sha = _selected_sha(weights)
    component_shas = {
        name: _weight_component_sha(name, weight) for name, weight in weights.items()
    }
    cache_entry = official.cache_c.detach().clone()
    event_batch = build_target_event_batch(tok, normalized, device=device)
    source_batch = build_sequence_event_batch(
        tok, normalized, target_key="target_true", device=device
    )
    with torch.no_grad():
        reference = capture_reference(_last_logits(model, event_batch), event_batch)
    telemetry = WriterTelemetry(
        schema="easyedit.alphaedit.strength-neutral-barrier.v2",
        arm=config.arm.value,
        steps=config.steps,
        request_count=len(normalized),
        target_event_count=event_batch.event_count,
        target_event_identity_sha256=event_batch.identity_sha256,
        q0_identity_sha256=reference.identity_sha256,
        fixed_z_compute_count=0,
        fixed_z_recompute_count=0,
        projector_load_count=1,
        cache_append_count=0,
        locality_controller_influence_count=0,
        rephrase_controller_influence_count=0,
        retry_count=0,
        imputation_count=0,
        dtype="torch.float32",
        w0_selected_sha256=entry_sha,
    )
    success = False
    try:
        context_templates = official.get_context_templates(model, tok)
        z_layer = hparams.layers[-1]
        z_module_name = hparams.layer_module_tmp.format(z_layer)
        z_device = get_module_device(nethook.get_module(model, z_module_name), device)
        z_list = []
        for request in normalized:
            cache_fname = (
                Path(
                    str(cache_template).format(
                        z_layer, hparams.clamp_norm_factor, request["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            loaded = False
            if cache_fname is not None and cache_fname.exists():
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(z_device))
                loaded = True
            if not loaded:
                value = compute_z(
                    model, tok, request, hparams, z_layer, context_templates
                )
                z_list.append(value)
                telemetry.fixed_z_compute_count += 1
                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(cache_fname, v_star=value.detach().cpu().numpy())
        zs = torch.stack(z_list, dim=1)

        for layer_index, layer in enumerate(hparams.layers):
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            weight = weights[weight_name]
            native_delta, keys, entry_residual = _native_layer_delta(
                model=model,
                tok=tok,
                requests=normalized,
                hparams=hparams,
                context_templates=context_templates,
                zs=zs,
                layer_index=layer_index,
                layer=layer,
                projector=official.P[layer_index],
                cache=official.cache_c[layer_index],
                weight=weight,
                device=device,
            )
            layer_record = LayerTelemetry(
                layer=layer,
                entry_z_residual_norm=entry_residual,
                terminal_z_residual_norm=float("nan"),
                terminal_barrier_q_kl=float("nan"),
                terminal_target_nll=[],
                native_delta_sha256=tensor_sha(native_delta.float()),
                layer_entry_weight_sha256=tensor_sha(weight.float()),
            )
            module_name = hparams.rewrite_module_tmp.format(layer)
            native_norm = float(torch.linalg.vector_norm(native_delta).item())
            for node in range(config.steps):
                pre_values = _observe_values(model, event_batch, reference)
                pre_step = _state_observation(
                    pre_values,
                    event_batch,
                    _source_sequence_nll(model, source_batch),
                )
                node_entry = weight.detach().clone()
                with torch.no_grad():
                    weight.add_(native_delta.to(weight), alpha=1.0 / config.steps)

                projection_receipt = RightProjectionReceipt(0, 1.0, 0.0, 0.0)
                if config.arm is BarrierArm.BARRIER:
                    observation = _gradient_observation(
                        model,
                        module_name,
                        weight,
                        event_batch,
                        reference,
                    )
                    projected_right, projection_receipt = project_right(
                        observation.common_right,
                        official.P[layer_index].to(device),
                        keys,
                    )
                    solution = solve_strength_neutral_correction(
                        barrier_gradient=observation.barrier_gradient,
                        target_gradients=observation.target_gradients,
                        native_delta=native_delta,
                        projected_right=projected_right,
                        keys=keys,
                    )
                    predictor_values = observation.values
                else:
                    predictor_values = _observe_values(model, event_batch, reference)
                    solution = _zero_solution()

                post_native_counterfactual = _state_observation(
                    predictor_values,
                    event_batch,
                    _source_sequence_nll(model, source_batch),
                )
                with torch.no_grad():
                    weight.copy_(node_entry)
                predictor_restore_pass = bool(torch.equal(weight, node_entry))
                if not predictor_restore_pass:
                    raise RuntimeError("native predictor temporary-state restore failed")

                apply_factor_terms_(
                    weight,
                    native_delta,
                    solution.terms,
                    step_scale=1.0 / config.steps,
                )
                guided_values = _observe_values(model, event_batch, reference)
                post_guided = _state_observation(
                    guided_values,
                    event_batch,
                    _source_sequence_nll(model, source_batch),
                )

                actual_dk = factor_terms_times_keys(solution.terms, keys)
                actual_dk_norm = (
                    float(torch.linalg.vector_norm(actual_dk).item())
                    if actual_dk.numel()
                    else 0.0
                )
                actual_dk_max_abs = (
                    float(actual_dk.abs().max().item()) if actual_dk.numel() else 0.0
                )
                component_shas[weight_name] = _weight_component_sha(weight_name, weight)
                endpoint_sha = _selected_component_root(component_shas)

                ratio = solution.correction_norm / native_norm if native_norm else 0.0
                layer_record.nodes.append(
                    NodeTelemetry(
                        layer=layer,
                        layer_index=layer_index,
                        node=node,
                        steps=config.steps,
                        barrier_q_kl=float(guided_values.barrier.item()),
                        target_nll=[
                            float(value) for value in guided_values.target_nll.tolist()
                        ],
                        target_log_odds=[
                            float(value)
                            for value in guided_values.target_log_odds.tolist()
                        ],
                        native_barrier_rate=solution.native_barrier_rate,
                        feasible_gradient_norm=solution.feasible_gradient_norm,
                        correction_norm=solution.correction_norm,
                        native_delta_norm=native_norm,
                        correction_native_ratio=ratio,
                        directional_rate=solution.directional_rate,
                        key_residual_max_abs=actual_dk_max_abs,
                        right_projection_key_residual_max_abs=max(
                            solution.key_residual_max_abs,
                            projection_receipt.key_residual_max_abs,
                        ),
                        max_strength_inner_abs=solution.max_strength_inner_abs,
                        key_gram_rank=projection_receipt.gram_rank,
                        key_gram_condition=projection_receipt.gram_condition,
                        key_pinv_rtol=projection_receipt.pinv_rtol,
                        constraint_rank=solution.constraint_rank,
                        constraint_condition=solution.constraint_condition,
                        constraint_pinv_rtol=solution.constraint_pinv_rtol,
                        correction_active=solution.active,
                        correction_term_count=len(solution.terms),
                        native_predictor=False,
                        pre_step=pre_step,
                        post_native_counterfactual=post_native_counterfactual,
                        post_guided=post_guided,
                        actual_dk_norm=actual_dk_norm,
                        actual_dk_max_abs=actual_dk_max_abs,
                        target_strength_constraint_residual_per_event_abs=list(
                            solution.target_strength_inner_abs
                        ),
                        target_strength_constraint_residual_max_abs=(
                            solution.max_strength_inner_abs
                        ),
                        selected_weight_endpoint_sha256=endpoint_sha,
                        predictor_restore_pass=predictor_restore_pass,
                    )
                )

            terminal = _observe_values(model, event_batch, reference)
            layer_record.terminal_barrier_q_kl = float(terminal.barrier.item())
            layer_record.terminal_target_nll = [
                float(value) for value in terminal.target_nll.tolist()
            ]
            layer_record.terminal_z_residual_norm = _terminal_z_residual(
                model, tok, normalized, hparams, zs
            )
            telemetry.layers.append(layer_record)

        for layer_index, layer in enumerate(hparams.layers):
            layer_ks = compute_ks(
                model, tok, normalized, hparams, layer, context_templates
            ).T
            official.cache_c[layer_index, :, :] += (
                layer_ks.cpu().float() @ layer_ks.cpu().float().T
            )
        telemetry.cache_append_count = 1

        endpoint = capture_exact_endpoint(weights)
        telemetry.temporary_endpoint_selected_sha256 = _selected_sha(weights)
        telemetry.temporary_endpoint_component_sha256 = {
            name: tensor_sha(value) for name, value in endpoint.items()
        }
        success = True
        telemetry.terminal_status = "TECHNICAL_PASS"
        return endpoint, telemetry
    except Exception as error:
        telemetry.terminal_status = "FAILED_BOUNDARY"
        telemetry.failure_type = type(error).__name__
        telemetry.failure_message = str(error)
        official.cache_c[...] = cache_entry
        raise
    finally:
        with torch.no_grad():
            for name, weight in weights.items():
                copy_to_param(weight, entry_weights[name])
        restored_sha = _selected_sha(weights)
        telemetry.restored_w0_selected_sha256 = restored_sha
        telemetry.w0_restore_pass = restored_sha == entry_sha
        if not telemetry.w0_restore_pass and success:
            telemetry.terminal_status = "FAILED_W0_RESTORE"
        if config.telemetry_path is not None and not success:
            write_create_once(config.telemetry_path, telemetry.to_dict())


def apply_strength_neutral_barrier_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict[str, Any]],
    hparams: AlphaEditHyperParams,
    config: BarrierWriterConfig,
    *,
    copy: bool = False,
    return_orig_weights: bool = False,
    cache_template: Optional[str] = None,
    reset_cache: bool = False,
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor], Dict[str, Any]]:
    """Apply one atomic edit and return model, old weights, and telemetry."""

    require_right_padding(tok, caller="AlphaEdit writer entry point")

    if config.arm is BarrierArm.OFFICIAL:
        model, originals = official.apply_AlphaEdit_to_model(
            model,
            tok,
            requests,
            hparams,
            copy=copy,
            return_orig_weights=return_orig_weights,
            cache_template=cache_template,
            reset_cache=reset_cache,
        )
        payload = {
            "schema": "easyedit.alphaedit.strength-neutral-barrier.v1",
            "arm": config.arm.value,
            "steps": 1,
            "official_native_bypass": True,
            "barrier_controller_influence_count": 0,
            "locality_controller_influence_count": 0,
        }
        if config.telemetry_path is not None:
            write_create_once(config.telemetry_path, payload)
        return model, originals, payload

    if copy:
        model = deepcopy(model)
    _assert_full_fp32(model)
    prepare_official_state(model, tok, hparams, reset_cache=reset_cache)
    originals: Dict[str, torch.Tensor] = {}
    if return_orig_weights:
        originals = {
            f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
                model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            )
            .detach()
            .clone()
            for layer in hparams.layers
        }
    endpoint, telemetry = _execute_guided(
        model,
        tok,
        requests,
        hparams,
        config,
        cache_template=cache_template,
    )
    authoritative_weights = {
        name: nethook.get_parameter(model, name) for name in endpoint
    }
    adoption = adopt_exact_endpoint(authoritative_weights, endpoint)
    telemetry.authoritative_endpoint_selected_sha256 = str(
        adoption["authoritative_endpoint_selected_sha256"]
    )
    telemetry.authoritative_endpoint_component_sha256 = {
        name: tensor_sha(value) for name, value in authoritative_weights.items()
    }
    telemetry.authoritative_endpoint_adoption_pass = bool(adoption["pass"])
    if config.telemetry_path is not None:
        write_create_once(config.telemetry_path, telemetry.to_dict())
    return model, originals, telemetry.to_dict()
