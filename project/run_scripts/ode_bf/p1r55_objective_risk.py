"""Request-local MEAN/RMS target-new risks for P1R55.

The established scalable objective plan and its six target-new contexts are
reused verbatim.  This module only delays the context reduction until the six
graph values for one request are available.  Consequently RMS values and
gradients come from the same graph, without an extra forward, backward, or a
per-request backward loop.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .scalable_batched_model import (
    OrdinalTargetActivationOverlay,
    ScalableObjectivePlan,
    _parameter_inventory_sha256,
)
from .target_new_nll import (
    _left_padding_offsets,
    _score_suffix,
)


INSTRUCTION_ID = "ODEEDIT-S05-P1R55-REQUEST-LOCAL-RMS-PDZ-FLOORED-RATE-V1"
CONTEXT_COUNT = 6


class ObjectiveRiskPolicy(str, Enum):
    MEAN = "MEAN"
    RMS = "RMS"


@dataclass(frozen=True, slots=True)
class RequestLocalRiskResult:
    """Drop-in target-objective result with risk-specific provenance."""

    loss: float
    per_request_values: tuple[float, ...]
    request_order_sha256: str
    target_span_sha256: str
    target_span_identities: tuple[str, ...]
    suffix_token_counts: tuple[int, ...]
    model_forward_count: int
    backward_count: int
    processed_token_count: int
    padded_token_count: int
    target_gradient: torch.Tensor | None
    coefficient_gradient: torch.Tensor | None
    plan_sha256: str
    model_state_sha256: str
    identity_sha256: str
    risk_policy: ObjectiveRiskPolicy
    per_request_context_nll: tuple[tuple[float, ...], ...]
    per_request_mean_risk: tuple[float, ...]
    per_request_rms_risk: tuple[float, ...]
    context_nll_sha256: str
    risk_gradient_sha256: str | None
    batch_scaling_application_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r55-request-local-risk-objective/v1",
            "instruction_id": INSTRUCTION_ID,
            "loss": self.loss,
            "per_request_values": list(self.per_request_values),
            "request_order_sha256": self.request_order_sha256,
            "target_span_sha256": self.target_span_sha256,
            "suffix_token_counts": list(self.suffix_token_counts),
            "model_forward_count": self.model_forward_count,
            "backward_count": self.backward_count,
            "processed_token_count": self.processed_token_count,
            "padded_token_count": self.padded_token_count,
            "target_gradient_sha256": self.risk_gradient_sha256,
            "coefficient_gradient_sha256": None,
            "plan_sha256": self.plan_sha256,
            "model_state_sha256": self.model_state_sha256,
            "risk_policy": self.risk_policy.value,
            "per_request_context_nll": [list(row) for row in self.per_request_context_nll],
            "per_request_risk": list(self.per_request_values),
            "per_request_mean_risk": list(self.per_request_mean_risk),
            "per_request_rms_risk": list(self.per_request_rms_risk),
            "context_nll_sha256": self.context_nll_sha256,
            "risk_gradient_sha256": self.risk_gradient_sha256,
            "batch_scaling_application_count": self.batch_scaling_application_count,
            "extra_forward_count": 0,
            "extra_backward_count": 0,
            "per_request_backward_loop_count": 0,
            "cross_request_gradient_reduction_decision_count": 0,
            "old_target_access_count": 0,
            "generation_call_count": 0,
            "mean_of_means_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def reduce_context_risk(
    context_nll: torch.Tensor,
    policy: ObjectiveRiskPolicy | str,
) -> torch.Tensor:
    """Reduce ``[B, 6]`` graph values without a scientific epsilon."""

    selected = policy if isinstance(policy, ObjectiveRiskPolicy) else ObjectiveRiskPolicy(policy)
    if (
        context_nll.ndim != 2
        or context_nll.shape[1] != CONTEXT_COUNT
        or not torch.isfinite(context_nll).all()
        or bool(torch.any(context_nll < 0.0))
    ):
        raise ODEBFContractError("P1R55 context-NLL geometry differs")
    if selected is ObjectiveRiskPolicy.MEAN:
        return context_nll.mean(dim=1)
    square_mean = torch.square(context_nll).mean(dim=1)
    # sqrt(0) has an undefined derivative.  The natural all-zero boundary is
    # exactly risk=0/gradient=0, so detach only those exact zero rows.
    positive = square_mean > 0.0
    safe_root = torch.sqrt(
        torch.where(positive, square_mean, torch.ones_like(square_mean))
    )
    return torch.where(positive, safe_root, square_mean)


def _score_context_graph(
    model: torch.nn.Module,
    batch: Any,
    plan: ScalableObjectivePlan,
    *,
    device: torch.device,
) -> tuple[list[tuple[int, torch.Tensor, int, str]], int]:
    encoding = {
        name: value.to(device=device, non_blocking=True)
        for name, value in batch.prepared.encoding.items()
    }
    input_ids, left_padding = _left_padding_offsets(
        encoding, rows=len(batch.prepared.row_request_ordinals)
    )
    logits = model(**encoding).logits
    if not isinstance(logits, torch.Tensor):
        raise ODEBFContractError("P1R55 cached logits differ")
    scores: dict[int, list[tuple[int, Any]]] = {
        ordinal: [] for ordinal in batch.prepared.request_ordinals
    }
    for row, (ordinal, identity, prefix_length, tokens, context_ordinal) in enumerate(
        zip(
            batch.prepared.row_request_ordinals,
            batch.prepared.row_request_sha256,
            batch.prepared.row_prefix_lengths,
            batch.prepared.row_tokens,
            batch.prepared.row_context_ordinals,
            strict=True,
        )
    ):
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
            target_label="target_new",
        )
        scores[ordinal].append((context_ordinal, score))
    result: list[tuple[int, torch.Tensor, int, str]] = []
    for ordinal in batch.prepared.request_ordinals:
        selected = sorted(scores[ordinal], key=lambda item: item[0])
        if (
            [item[0] for item in selected] != list(range(CONTEXT_COUNT))
            or len({item[1].token_count for item in selected}) != 1
        ):
            raise ODEBFContractError("P1R55 context score coverage differs")
        result.append(
            (
                ordinal,
                torch.stack([item[1].value for item in selected]),
                selected[0][1].token_count,
                canonical_hash(
                    {
                        "schema": "ode-edit-s05-target-new-nll-request-span/v1",
                        "context_sha256": plan.context_sha256,
                        "contexts": [
                            canonical_hash(
                                {
                                    "context_ordinal": index,
                                    "target_new": item[1].span_identity,
                                    "target_true": None,
                                }
                            )
                            for index, item in enumerate(selected)
                        ],
                    }
                ),
            )
        )
    processed = int(encoding["attention_mask"].sum().item())
    return result, processed


def evaluate_request_local_risk(
    model: torch.nn.Module,
    plan: ScalableObjectivePlan,
    *,
    risk_policy: ObjectiveRiskPolicy | str,
    target_state: torch.Tensor | None = None,
    current_terminal: torch.Tensor | None = None,
    target_layer_name: str | None = None,
    target_gradient_required: bool = True,
) -> RequestLocalRiskResult:
    """Evaluate MEAN/RMS with the established forward/backward clock."""

    policy = (
        risk_policy
        if isinstance(risk_policy, ObjectiveRiskPolicy)
        else ObjectiveRiskPolicy(risk_policy)
    )
    target_mode = target_state is not None
    if target_mode != (current_terminal is not None and target_layer_name is not None):
        raise ODEBFContractError("P1R55 target objective inputs differ")
    if target_mode and (
        target_state is None
        or current_terminal is None
        or target_state.ndim != 2
        or target_state.shape != current_terminal.shape
        or target_state.shape[1] != plan.request_count
        or (target_gradient_required and not target_state.requires_grad)
        or not torch.isfinite(target_state).all()
        or not torch.isfinite(current_terminal).all()
    ):
        raise ODEBFContractError("P1R55 target objective geometry differs")
    device = next(model.parameters()).device
    before = _parameter_inventory_sha256(model)
    contexts: list[torch.Tensor | None] = [None] * plan.request_count
    counts: list[int | None] = [None] * plan.request_count
    spans: list[str | None] = [None] * plan.request_count
    target_gradient = (
        torch.zeros_like(target_state, dtype=torch.float64, device="cpu")
        if target_mode and target_gradient_required
        else None
    )
    backward_count = 0
    processed = 0
    overlay_fire_count = 0
    for batch in plan.batches:
        with ExitStack() as stack:
            overlay: OrdinalTargetActivationOverlay | None = None
            if target_mode:
                assert target_state is not None and current_terminal is not None
                assert target_layer_name is not None
                residual = target_state - current_terminal.to(
                    device=target_state.device, dtype=target_state.dtype
                )
                overlay = stack.enter_context(
                    OrdinalTargetActivationOverlay(
                        model,
                        target_layer_name,
                        residual=residual,
                        row_request_ordinals=batch.prepared.row_request_ordinals,
                        padded_lookup_positions=batch.padded_lookup_positions,
                    )
                )
            observed, batch_processed = _score_context_graph(
                model, batch, plan, device=device
            )
            context_graph = torch.stack([item[1] for item in observed])
            risk_graph = reduce_context_risk(context_graph, policy)
            if target_mode and target_gradient_required:
                assert target_state is not None and target_gradient is not None
                gradient = torch.autograd.grad(
                    risk_graph.sum(), target_state, retain_graph=False, create_graph=False
                )[0]
                target_gradient.add_(gradient.detach().to(device="cpu", dtype=torch.float64))
                backward_count += 1
                assert overlay is not None
                overlay_fire_count += overlay.fire_count
            elif target_mode:
                assert overlay is not None
                overlay_fire_count += overlay.fire_count
            for position, (ordinal, values, count, span) in enumerate(observed):
                if contexts[ordinal] is not None:
                    raise ODEBFContractError("P1R55 objective duplicated a request")
                contexts[ordinal] = values.detach().to(device="cpu", dtype=torch.float64)
                counts[ordinal] = int(count)
                spans[ordinal] = str(span)
            processed += batch_processed
    if (
        any(item is None for item in contexts)
        or any(item is None for item in counts)
        or any(item is None for item in spans)
        or processed != plan.processed_token_count
        or (target_mode and overlay_fire_count != len(plan.batches))
    ):
        raise ODEBFContractError("P1R55 objective coverage differs")
    context_cpu = torch.stack([item for item in contexts if item is not None])
    mean_risk = context_cpu.mean(dim=1)
    rms_risk = reduce_context_risk(context_cpu, ObjectiveRiskPolicy.RMS)
    selected_risk = mean_risk if policy is ObjectiveRiskPolicy.MEAN else rms_risk
    if target_gradient is not None:
        target_gradient.div_(plan.request_count)
        target_gradient = target_gradient.to(dtype=torch.float32).contiguous()
    final_values = tuple(float(item) for item in selected_risk)
    loss = math.fsum(final_values) / plan.request_count
    final_spans = tuple(str(item) for item in spans if item is not None)
    final_counts = tuple(int(item) for item in counts if item is not None)
    after = _parameter_inventory_sha256(model)
    if after != before:
        raise ODEBFStateError("P1R55 objective mutated model state")
    context_hash = tensor_sha256(context_cpu.contiguous())
    gradient_hash = None if target_gradient is None else tensor_sha256(target_gradient)
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r55-request-local-risk-objective/v1",
        "instruction_id": INSTRUCTION_ID,
        "loss": loss,
        "per_request_values": list(final_values),
        "request_order_sha256": plan.request_order_sha256,
        "target_span_sha256": canonical_hash(list(final_spans)),
        "suffix_token_counts": list(final_counts),
        "model_forward_count": len(plan.batches),
        "backward_count": backward_count,
        "processed_token_count": processed,
        "padded_token_count": plan.padded_token_count,
        "target_gradient_sha256": gradient_hash,
        "coefficient_gradient_sha256": None,
        "plan_sha256": plan.identity_sha256,
        "model_state_sha256": after,
        "risk_policy": policy.value,
        "per_request_context_nll": context_cpu.tolist(),
        "per_request_risk": list(final_values),
        "per_request_mean_risk": mean_risk.tolist(),
        "per_request_rms_risk": rms_risk.tolist(),
        "context_nll_sha256": context_hash,
        "risk_gradient_sha256": gradient_hash,
        "batch_scaling_application_count": 1 if target_gradient is not None else 0,
        "extra_forward_count": 0,
        "extra_backward_count": 0,
        "per_request_backward_loop_count": 0,
        "cross_request_gradient_reduction_decision_count": 0,
    }
    identity = canonical_hash(payload)
    return RequestLocalRiskResult(
        loss,
        final_values,
        plan.request_order_sha256,
        payload["target_span_sha256"],
        final_spans,
        final_counts,
        len(plan.batches),
        backward_count,
        processed,
        plan.padded_token_count,
        target_gradient,
        None,
        plan.identity_sha256,
        after,
        identity,
        policy,
        tuple(tuple(float(value) for value in row) for row in context_cpu),
        tuple(float(value) for value in mean_risk),
        tuple(float(value) for value in rms_risk),
        context_hash,
        gradient_hash,
        1 if target_gradient is not None else 0,
    )


class RequestLocalRiskEvaluator:
    """Typed callable injected only into the P1R55 target controller."""

    def __init__(self, risk_policy: ObjectiveRiskPolicy | str) -> None:
        self.risk_policy = (
            risk_policy
            if isinstance(risk_policy, ObjectiveRiskPolicy)
            else ObjectiveRiskPolicy(risk_policy)
        )
        self.call_count = 0

    def __call__(self, model: torch.nn.Module, plan: ScalableObjectivePlan, **kwargs: Any) -> RequestLocalRiskResult:
        result = evaluate_request_local_risk(
            model, plan, risk_policy=self.risk_policy, **kwargs
        )
        self.call_count += 1
        return result


__all__ = [
    "CONTEXT_COUNT",
    "INSTRUCTION_ID",
    "ObjectiveRiskPolicy",
    "RequestLocalRiskEvaluator",
    "RequestLocalRiskResult",
    "evaluate_request_local_risk",
    "reduce_context_risk",
]
