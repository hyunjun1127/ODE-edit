"""Model-backed binding for the two P4 target-side semantic objectives."""

from __future__ import annotations

import copy
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import P1R24KLPlan, evaluate_p1r24_kl
from .p4_fixed_target_solver import P4SolverEvaluation
from .p4_semantic_barrier import P4TargetArm, smooth_semantic_logodds_potential
from .scalable_batched_model import (
    OrdinalTargetActivationOverlay,
    ScalableObjectivePlan,
    _parameter_inventory_sha256,
    build_scalable_objective_plan,
)
from .target_new_nll import _left_padding_offsets, _score_suffix


@dataclass(frozen=True, slots=True)
class P4PairedObjectivePlans:
    new: ScalableObjectivePlan
    old: ScalableObjectivePlan
    binding: Mapping[str, Any]


def build_p4_paired_objective_plans(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    contexts: Sequence[Sequence[str]],
    request_microbatch_size: int,
    fact_token_strategy: str,
) -> P4PairedObjectivePlans:
    """Build identical-context plans for target_new and current target_true."""

    new_plan = build_scalable_objective_plan(
        model,
        tokenizer,
        requests,
        contexts=contexts,
        request_microbatch_size=request_microbatch_size,
        fact_token_strategy=fact_token_strategy,
    )
    old_requests = copy.deepcopy(list(requests))
    for source, old in zip(requests, old_requests, strict=True):
        target_true = source.get("target_true")
        if isinstance(target_true, Mapping):
            target_true = target_true.get("str")
        if not isinstance(target_true, str) or not target_true:
            raise ODEBFContractError("P4 current request target_true differs")
        old["target_new"] = target_true
    old_plan = build_scalable_objective_plan(
        model,
        tokenizer,
        old_requests,
        contexts=contexts,
        request_microbatch_size=request_microbatch_size,
        fact_token_strategy=fact_token_strategy,
    )
    fields = (
        "request_count",
        "request_sha256",
        "request_order_sha256",
        "context_sha256",
        "request_microbatch_size",
        "llama",
    )
    if any(getattr(new_plan, field) != getattr(old_plan, field) for field in fields):
        raise ODEBFContractError("P4 positive/negative non-barrier inputs differ")
    binding: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-paired-target-plans/v1",
        "request_count": new_plan.request_count,
        "request_order_sha256": new_plan.request_order_sha256,
        "context_sha256": new_plan.context_sha256,
        "context_count": 6,
        "request_microbatch_size": new_plan.request_microbatch_size,
        "fact_token_strategy": fact_token_strategy,
        "new_plan_sha256": new_plan.identity_sha256,
        "old_plan_sha256": old_plan.identity_sha256,
        "old_source": "CURRENT_REQUEST_TARGET_TRUE_ONLY",
        "historical_negative_access_count": 0,
        "heldout_access_count": 0,
    }
    binding["identity_sha256"] = canonical_hash(binding)
    return P4PairedObjectivePlans(new_plan, old_plan, binding)


def non_barrier_arm_identity(
    paired: P4PairedObjectivePlans,
    kl_plan: P1R24KLPlan,
    *,
    teacher_sha256: str,
    origin_target_sha256: str,
    terminal_target_sha256: str,
    target_layer_name: str,
    learning_rate: float,
    kl_factor: float,
    decay_factor: float,
    clamp_factor: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-za-non-barrier-arm-identity/v1",
        "paired_plan_sha256": paired.binding["identity_sha256"],
        "request_order_sha256": paired.new.request_order_sha256,
        "context_sha256": paired.new.context_sha256,
        "kl_plan_sha256": kl_plan.identity_sha256,
        "teacher_sha256": teacher_sha256,
        "origin_target_sha256": origin_target_sha256,
        "terminal_target_sha256": terminal_target_sha256,
        "target_layer_name": target_layer_name,
        "optimizer": "ADAM",
        "inner_iterations": 5,
        "moment_reset_each_outer": True,
        "learning_rate": learning_rate,
        "kl_factor": kl_factor,
        "decay_factor": decay_factor,
        "clamp_factor": clamp_factor,
        "outer_steps": 8,
        "writer_count": 0,
        "W0_shared": True,
        "heldout_decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _context_nll_for_ordinal(
    model: torch.nn.Module,
    plan: ScalableObjectivePlan,
    *,
    ordinal: int,
    target_state: torch.Tensor,
    current_terminal: torch.Tensor,
    target_layer_name: str,
    target_label: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if plan.request_microbatch_size != 1 or len(plan.batches) != plan.request_count:
        raise ODEBFContractError("P4 semantic execution batching differs")
    matches = [batch for batch in plan.batches if batch.request_ordinals == (ordinal,)]
    if len(matches) != 1:
        raise ODEBFContractError("P4 semantic request ordinal coverage differs")
    batch = matches[0]
    device = next(model.parameters()).device
    encoding = {
        name: value.to(device=device, non_blocking=True)
        for name, value in batch.prepared.encoding.items()
    }
    input_ids, left_padding = _left_padding_offsets(
        encoding, rows=len(batch.prepared.row_request_ordinals)
    )
    values: list[torch.Tensor | None] = [None] * 6
    spans: list[str | None] = [None] * 6
    with ExitStack() as stack:
        overlay = stack.enter_context(
            OrdinalTargetActivationOverlay(
                model,
                target_layer_name,
                residual=target_state
                - current_terminal.to(
                    device=target_state.device, dtype=target_state.dtype
                ),
                row_request_ordinals=batch.prepared.row_request_ordinals,
                padded_lookup_positions=batch.padded_lookup_positions,
            )
        )
        logits = model(**encoding).logits
        for row, (
            observed_ordinal,
            identity,
            prefix_length,
            tokens,
            context_ordinal,
        ) in enumerate(
            zip(
                batch.prepared.row_request_ordinals,
                batch.prepared.row_request_sha256,
                batch.prepared.row_prefix_lengths,
                batch.prepared.row_tokens,
                batch.prepared.row_context_ordinals,
                strict=True,
            )
        ):
            if observed_ordinal != ordinal:
                raise ODEBFContractError("P4 semantic row ordinal differs")
            score = _score_suffix(
                logits=logits,
                input_ids=input_ids,
                row=row,
                prefix_length=prefix_length,
                left_padding=left_padding[row],
                tokens=tokens,
                llama=plan.llama,
                request_sha256=identity,
                ordinal=ordinal,
                context_ordinal=context_ordinal,
                context_sha256=plan.context_sha256,
                target_label=target_label,
            )
            values[context_ordinal] = score.value
            spans[context_ordinal] = score.span_identity
    if any(item is None for item in values) or any(item is None for item in spans) or overlay.fire_count != 1:
        raise ODEBFContractError("P4 semantic context coverage differs")
    matrix = torch.stack([item for item in values if item is not None]).unsqueeze(0)
    return matrix.to(dtype=torch.float32), {
        "plan_sha256": plan.identity_sha256,
        "target_label": target_label,
        "span_sha256": canonical_hash(spans),
        "forward_count": 1,
        "processed_token_count": int(encoding["attention_mask"].sum().detach().cpu()),
    }


def evaluate_p4_target_objective(
    model: torch.nn.Module,
    paired: P4PairedObjectivePlans,
    kl_plan: P1R24KLPlan,
    teacher_log_probs: Sequence[torch.Tensor],
    *,
    target_state: torch.Tensor,
    current_terminal: torch.Tensor,
    target_origin: torch.Tensor,
    target_layer_name: str,
    arm: P4TargetArm | str,
    kl_factor: float,
    decay_factor: float,
) -> P4SolverEvaluation:
    """Evaluate per-request semantic + pinned KL + origin decay gradients."""

    if (
        target_state.dtype is not torch.float32
        or not target_state.requires_grad
        or target_state.shape != current_terminal.shape
        or target_state.shape != target_origin.shape
        or target_state.shape[1] != paired.new.request_count
    ):
        raise ODEBFContractError("P4 target binding geometry differs")
    before = _parameter_inventory_sha256(model)
    new_rows: list[torch.Tensor] = []
    old_rows: list[torch.Tensor] = []
    semantic_values: list[torch.Tensor] = []
    semantic_gradient = torch.zeros_like(target_state)
    new_receipts: list[dict[str, Any]] = []
    old_receipts: list[dict[str, Any]] = []
    for ordinal in range(paired.new.request_count):
        new_nll, new_row = _context_nll_for_ordinal(
            model, paired.new, ordinal=ordinal, target_state=target_state,
            current_terminal=current_terminal, target_layer_name=target_layer_name,
            target_label="target_new",
        )
        old_nll, old_row = _context_nll_for_ordinal(
            model, paired.old, ordinal=ordinal, target_state=target_state,
            current_terminal=current_terminal, target_layer_name=target_layer_name,
            target_label="target_true",
        )
        local = smooth_semantic_logodds_potential(-new_nll, -old_nll, arm=arm)
        local_gradient = torch.autograd.grad(
            local.per_request_objective.sum(), target_state,
            retain_graph=False, create_graph=False,
        )[0]
        semantic_gradient[:, ordinal] = local_gradient[:, ordinal]
        new_rows.append(new_nll.detach())
        old_rows.append(old_nll.detach())
        semantic_values.append(
            local.per_request_objective.detach().to(target_state.device)
        )
        new_receipts.append(new_row)
        old_receipts.append(old_row)
    all_new_nll = torch.cat(new_rows, dim=0)
    all_old_nll = torch.cat(old_rows, dim=0)
    semantic = smooth_semantic_logodds_potential(
        -all_new_nll, -all_old_nll, arm=arm
    )
    semantic_per_request = torch.cat(semantic_values, dim=0)

    kl_variable = target_state.detach().clone().requires_grad_(True)
    kl, _ = evaluate_p1r24_kl(
        model,
        kl_plan,
        teacher_log_probs=teacher_log_probs,
        target_state=kl_variable,
        current_terminal=current_terminal,
        target_layer_name=target_layer_name,
    )
    if kl.gradient is None:
        raise ODEBFContractError("P4 KL gradient is absent")
    request_count = target_state.shape[1]
    kl_values = torch.tensor(
        kl.per_request_values, device=target_state.device, dtype=torch.float32
    )
    kl_gradient = kl.gradient.to(target_state.device) * request_count

    decay_variable = target_state.detach().clone().requires_grad_(True)
    origin = target_origin.to(
        device=decay_variable.device, dtype=decay_variable.dtype
    )
    denominator = torch.square(torch.linalg.vector_norm(origin, dim=0))
    if bool(torch.any(denominator <= 0.0)):
        raise ODEBFContractError("P4 origin decay denominator differs")
    decay_values = decay_factor * torch.linalg.vector_norm(
        decay_variable - origin, dim=0
    ) / denominator
    decay_gradient = torch.autograd.grad(
        decay_values.sum(), decay_variable, retain_graph=False, create_graph=False
    )[0]
    values = semantic_per_request + kl_factor * kl_values + decay_values
    gradient = semantic_gradient + kl_factor * kl_gradient + decay_gradient
    if (
        values.dtype is not torch.float32
        or gradient.dtype is not torch.float32
        or not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(gradient).all())
    ):
        raise ODEBFContractError("P4 combined target objective is nonfinite")
    after = _parameter_inventory_sha256(model)
    if before != after:
        raise ODEBFStateError("P4 target objective mutated W0")
    telemetry = {
        "semantic": dict(semantic.receipt),
        "semantic_gradient_sha256": tensor_sha256(semantic_gradient),
        "semantic_gradient_norm_by_request": [
            float(item)
            for item in torch.linalg.vector_norm(semantic_gradient, dim=0)
        ],
        "new_context": new_receipts,
        "old_context": old_receipts,
        "kl": kl.raw_free_payload(),
        "decay_by_request": [float(item) for item in decay_values],
        "combined_gradient_sha256": tensor_sha256(gradient),
        "autocast_enabled": torch.is_autocast_enabled(),
        "cpu_autocast_enabled": torch.is_autocast_enabled("cpu"),
        "dtype": str(target_state.dtype),
        "heldout_decision_influence_count": 0,
    }
    return P4SolverEvaluation(values, gradient, telemetry)


__all__ = [
    "P4PairedObjectivePlans",
    "build_p4_paired_objective_plans",
    "evaluate_p4_target_objective",
    "non_barrier_arm_identity",
]
