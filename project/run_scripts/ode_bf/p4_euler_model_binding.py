"""Checkpointed model-backed value binding for P4 Euler calibration.

The binding builds one differentiable aggregate graph while bounding activation
memory through non-reentrant checkpoint recomputation.  It never calls
``autograd.grad`` or ``backward``; the pure Euler integrator remains the sole
gradient-call owner.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .contracts import ODEBFContractError, canonical_hash
from .p1r24_atomic_strength import P1R24KLPlan
from .p4_euler_binding import P4EulerObjectiveTerms
from .p4_target_solver_binding import P4PairedObjectivePlans, _context_nll_for_ordinal
from .scalable_batched_model import OrdinalTargetActivationOverlay


def _checkpointed_context_nll(
    model: torch.nn.Module,
    plan: Any,
    *,
    ordinal: int,
    target_state: torch.Tensor,
    current_terminal: torch.Tensor,
    target_layer_name: str,
    target_label: str,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    holder: list[Mapping[str, Any]] = []

    def score(variable: torch.Tensor) -> torch.Tensor:
        value, receipt = _context_nll_for_ordinal(
            model,
            plan,
            ordinal=ordinal,
            target_state=variable,
            current_terminal=current_terminal,
            target_layer_name=target_layer_name,
            target_label=target_label,
        )
        if not holder:
            holder.append(receipt)
        return value

    value = checkpoint(score, target_state, use_reentrant=False, preserve_rng_state=True)
    if len(holder) != 1:
        raise ODEBFContractError("P4 Euler checkpointed context receipt differs")
    return value, holder[0]


def _checkpointed_kl_values(
    model: torch.nn.Module,
    plan: P1R24KLPlan,
    teacher_log_probs: Sequence[torch.Tensor],
    *,
    target_state: torch.Tensor,
    current_terminal: torch.Tensor,
    target_layer_name: str,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    if len(teacher_log_probs) != plan.request_count:
        raise ODEBFContractError("P4 Euler KL teacher inventory differs")
    values: list[torch.Tensor | None] = [None] * plan.request_count
    processed = 0
    padded = 0
    device = next(model.parameters()).device
    for batch in plan.batches:
        encoding = {name: value.to(device=device) for name, value in batch.encoding.items()}

        def evaluate(
            variable: torch.Tensor,
            *,
            bound_batch: Any = batch,
            bound_encoding: Mapping[str, torch.Tensor] = encoding,
        ) -> torch.Tensor:
            with ExitStack() as stack:
                overlay = stack.enter_context(
                    OrdinalTargetActivationOverlay(
                        model,
                        target_layer_name,
                        residual=variable
                        - current_terminal.to(variable.device, variable.dtype),
                        row_request_ordinals=bound_batch.request_ordinals,
                        padded_lookup_positions=bound_batch.padded_lookup_positions,
                    )
                )
                logits = model(**bound_encoding).logits
                selected = torch.stack(
                    [
                        logits[row, position, :].float()
                        for row, position in enumerate(
                            bound_batch.padded_lookup_positions
                        )
                    ]
                )
                log_probs = torch.log_softmax(selected, dim=1)
                teacher = torch.stack(
                    [
                        teacher_log_probs[ordinal].to(
                            device=device, dtype=torch.float32
                        )
                        for ordinal in bound_batch.request_ordinals
                    ]
                )
                result = F.kl_div(
                    teacher,
                    log_probs,
                    log_target=True,
                    reduction="none",
                ).sum(dim=1)
                if overlay.fire_count != 1:
                    raise ODEBFContractError("P4 Euler KL overlay firing differs")
                return result

        batch_values = checkpoint(
            evaluate, target_state, use_reentrant=False, preserve_rng_state=True
        )
        for row, ordinal in enumerate(batch.request_ordinals):
            values[ordinal] = batch_values[row]
        processed += int(encoding["attention_mask"].sum().detach().cpu())
        padded += int(encoding["attention_mask"].numel())
    if any(value is None for value in values):
        raise ODEBFContractError("P4 Euler KL request coverage differs")
    stacked = torch.stack([value for value in values if value is not None])
    receipt: dict[str, Any] = {
        "plan_sha256": plan.identity_sha256,
        "logical_forward_count": len(plan.batches),
        "checkpoint_recompute_forward_count": len(plan.batches),
        "actual_forward_count_after_gradient": 2 * len(plan.batches),
        "processed_token_count_per_pass": processed,
        "padded_token_count_per_pass": padded,
        "teacher_refresh_count": 0,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return stacked, receipt


def evaluate_checkpointed_model_terms(
    model: torch.nn.Module,
    paired: P4PairedObjectivePlans,
    kl_plan: P1R24KLPlan,
    teacher_log_probs: Sequence[torch.Tensor],
    *,
    target_state: torch.Tensor,
    current_terminal: torch.Tensor,
    target_origin: torch.Tensor,
    target_layer_name: str,
) -> P4EulerObjectiveTerms:
    """Build differentiable semantic/KL/decay values without gradient calls."""

    if (
        target_state.dtype is not torch.float32
        or not target_state.requires_grad
        or target_state.shape != current_terminal.shape
        or target_state.shape != target_origin.shape
        or target_state.shape[1] != paired.new.request_count
        or paired.new.request_microbatch_size != 1
        or paired.old.request_microbatch_size != 1
    ):
        raise ODEBFContractError("P4 Euler model-backed target geometry differs")
    new_rows: list[torch.Tensor] = []
    old_rows: list[torch.Tensor] = []
    new_receipts: list[Mapping[str, Any]] = []
    old_receipts: list[Mapping[str, Any]] = []
    for ordinal in range(paired.new.request_count):
        new_nll, new_receipt = _checkpointed_context_nll(
            model,
            paired.new,
            ordinal=ordinal,
            target_state=target_state,
            current_terminal=current_terminal,
            target_layer_name=target_layer_name,
            target_label="target_new",
        )
        old_nll, old_receipt = _checkpointed_context_nll(
            model,
            paired.old,
            ordinal=ordinal,
            target_state=target_state,
            current_terminal=current_terminal,
            target_layer_name=target_layer_name,
            target_label="target_true",
        )
        new_rows.append(new_nll)
        old_rows.append(old_nll)
        new_receipts.append(new_receipt)
        old_receipts.append(old_receipt)
    all_new_nll = torch.cat(new_rows, dim=0)
    all_old_nll = torch.cat(old_rows, dim=0)
    kl_values, kl_receipt = _checkpointed_kl_values(
        model,
        kl_plan,
        teacher_log_probs,
        target_state=target_state,
        current_terminal=current_terminal,
        target_layer_name=target_layer_name,
    )
    origin = target_origin.to(device=target_state.device, dtype=torch.float32)
    denominator = torch.square(torch.linalg.vector_norm(origin, dim=0))
    if bool(torch.any(denominator <= 0.0)):
        raise ODEBFContractError("P4 Euler origin decay denominator differs")
    decay = torch.linalg.vector_norm(target_state - origin, dim=0) / denominator
    logical = len(new_receipts) + len(old_receipts) + len(kl_plan.batches)
    telemetry: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-checkpointed-model-terms/v1",
        "new_nll_by_request": [
            float(value) for value in torch.mean(all_new_nll, dim=1).detach().cpu()
        ],
        "old_nll_by_request": [
            float(value) for value in torch.mean(all_old_nll, dim=1).detach().cpu()
        ],
        "new_minus_old_margin_by_request": [
            float(value)
            for value in torch.mean(-all_new_nll + all_old_nll, dim=1)
            .detach()
            .cpu()
        ],
        "new_context": list(new_receipts),
        "old_context": list(old_receipts),
        "kl": kl_receipt,
        "logical_model_forward_count": logical,
        "checkpoint_recompute_forward_count": logical,
        "actual_model_forward_count_after_gradient": 2 * logical,
        "autograd_grad_call_count_owned_here": 0,
        "loss_backward_call_count": 0,
        "parameter_gradient_count": 0,
    }
    telemetry["identity_sha256"] = canonical_hash(telemetry)
    return P4EulerObjectiveTerms(-all_new_nll, -all_old_nll, kl_values, decay, telemetry)


@torch.no_grad()
def observe_model_terms_no_grad(
    model: torch.nn.Module,
    paired: P4PairedObjectivePlans,
    kl_plan: P1R24KLPlan,
    teacher_log_probs: Sequence[torch.Tensor],
    *,
    target_state: torch.Tensor,
    current_terminal: torch.Tensor,
    target_origin: torch.Tensor,
    target_layer_name: str,
) -> Mapping[str, Any]:
    """Endpoint value-only observation; no field or gradient is evaluated."""

    new_rows: list[torch.Tensor] = []
    old_rows: list[torch.Tensor] = []
    for ordinal in range(paired.new.request_count):
        new_nll, _ = _context_nll_for_ordinal(
            model,
            paired.new,
            ordinal=ordinal,
            target_state=target_state,
            current_terminal=current_terminal,
            target_layer_name=target_layer_name,
            target_label="target_new",
        )
        old_nll, _ = _context_nll_for_ordinal(
            model,
            paired.old,
            ordinal=ordinal,
            target_state=target_state,
            current_terminal=current_terminal,
            target_layer_name=target_layer_name,
            target_label="target_true",
        )
        new_rows.append(new_nll)
        old_rows.append(old_nll)
    new = torch.cat(new_rows, dim=0)
    old = torch.cat(old_rows, dim=0)
    payload: dict[str, Any] = {
        "new_nll_by_request": [float(value) for value in torch.mean(new, dim=1).cpu()],
        "old_nll_by_request": [float(value) for value in torch.mean(old, dim=1).cpu()],
        "new_minus_old_margin_by_request": [
            float(value) for value in torch.mean(-new + old, dim=1).cpu()
        ],
        "semantic_forward_count": 2 * paired.new.request_count,
        "kl_forward_count": 0,
        "autograd_grad_call_count": 0,
        "field_evaluation_count": 0,
        "decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "evaluate_checkpointed_model_terms",
    "observe_model_terms_no_grad",
]
