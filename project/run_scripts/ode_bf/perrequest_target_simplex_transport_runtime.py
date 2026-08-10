"""Model-facing P1R15 target/factor group and matrix-free field adapter.

The target group retains the ten gradients that the locked six-context
TARGET_NEW_NLL operation already computes.  It adds no model forward or
backward family to the P1R14 matrix-free 13-group field contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from .cold_start_target import _rng_identity
from .common_cold_coordinate import evaluate_common_cold_objective
from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import WaypointFactor, tensor_sha256
from .integrated_physical_writer import (
    PHYSICAL_WRITER_CONTEXT_COUNT,
    PHYSICAL_WRITER_LAYER_ORDER,
    PHYSICAL_WRITER_MICROBATCH_GROUPS,
)
from .integrated_physical_writer_runtime import (
    _DynamicTargetResidualOverlay,
    _StreamingWriterKeyCapture,
)
from .p1_backend import _virtual_context
from .perrequest_target_simplex_transport import (
    P1R15_SCHEMA,
    PerRequestTargetStep,
    build_per_request_gdual_target_step,
)
from .request_digest import ordered_request_digest_v1
from .target_new_nll import (
    RoutingObjective,
    _is_llama,
    _locked_context_templates,
    _model_device,
    _model_state,
    _ordered_batch,
    _score_request,
    _temporary_left_padding,
)


@dataclass(frozen=True, slots=True)
class PerRequestTargetFactorGroupReceipt:
    request_order_sha256: str
    context_sha256: str
    target_state_sha256: str
    terminal_current_z_sha256: str
    residual_sha256: str
    desired_target_sha256: str
    loss_by_request: tuple[float, ...]
    gradient_by_request_sha256: str
    request_gradient_sha256: tuple[str, ...]
    off_request_gradient_nonzero_count: tuple[int, ...]
    target_step: PerRequestTargetStep
    goal_target_new_nll: float
    key_sha256_by_layer: tuple[tuple[int, str], ...]
    layer_hook_call_count: tuple[tuple[int, int], ...]
    logical_forward_groups: int
    model_forward_calls: int
    physical_target_graphs: int
    autograd_backend_invocations: int
    backward_calls: int
    processed_token_count: int
    old_target_access_count: int
    heldout_access_count: int
    native_or_direct_z_access_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_step"] = self.target_step.raw_free_payload()
        return payload


def perrequest_dynamic_target_factor_group(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    *,
    target_state: torch.Tensor,
    z_base: torch.Tensor,
    target_layer_name: str,
    lookup_positions: Sequence[int],
    rewrite_module_template: str,
    fact_token_strategy: str,
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    step_index: int,
    cumulative_g_path_before: Sequence[float],
) -> tuple[
    dict[int, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    PerRequestTargetFactorGroupReceipt,
]:
    """Build the exact per-request G-dual target and five keys in one group."""

    from easyeditor.models.rome import repr_tools

    batch, identities, order = _ordered_batch(requests)
    if order != ordered_request_digest_v1(identities):
        raise ODEBFContractError("P1R15 target order differs")
    templates, group_sizes, context_sha256 = _locked_context_templates(
        RoutingObjective.TARGET_NEW_NLL, contexts
    )
    if group_sizes != (1, 5) or len(templates) != PHYSICAL_WRITER_MICROBATCH_GROUPS:
        raise ODEBFContractError("P1R15 target contexts differ")
    rendered_templates: list[str] = []
    words: list[str] = []
    for request in batch:
        for template in templates:
            assert template is not None
            rendered_templates.append(template.format(str(request["prompt"])))
            words.append(str(request["subject"]))
    strategy = str(fact_token_strategy)
    if not strategy.startswith("subject_"):
        raise ODEBFContractError("P1R15 writer lookup strategy differs")
    writer_lookup = repr_tools.get_words_idxs_in_templates(
        tokenizer,
        rendered_templates,
        words,
        strategy[len("subject_") :],
    )
    if len(writer_lookup) != PHYSICAL_WRITER_CONTEXT_COUNT:
        raise ODEBFContractError("P1R15 writer lookup count differs")

    device = _model_device(model)
    target_flat = (
        target_state.detach()
        .to(device=device, dtype=torch.float32)
        .contiguous()
        .view(-1)
        .requires_grad_(True)
    )
    target_variable = target_flat.view(target_state.shape)
    before = _model_state(model)
    before_rng = _rng_identity()
    capture = _StreamingWriterKeyCapture(
        model,
        {
            layer: rewrite_module_template.format(layer)
            for layer in PHYSICAL_WRITER_LAYER_ORDER
        },
        writer_lookup,
    )
    overlay = _DynamicTargetResidualOverlay(
        model,
        target_layer_name,
        target_variable,
        lookup_positions,
    )
    llama = _is_llama(model)
    loss_values: list[torch.Tensor] = []
    gradient_columns: list[torch.Tensor] = []
    request_gradient_hashes: list[str] = []
    off_request_nonzero: list[int] = []
    processed_tokens = 0
    with _virtual_context(model, cumulative_factors_by_weight):
        with capture, overlay, _temporary_left_padding(tokenizer):
            for request_ordinal, (request, identity) in enumerate(
                zip(batch, identities, strict=True)
            ):
                context_values: list[torch.Tensor] = []
                for context_ordinal, template in enumerate(templates):
                    value, _new, true, processed = _score_request(
                        model,
                        tokenizer,
                        request,
                        objective=RoutingObjective.TARGET_NEW_NLL,
                        request_sha256=identity,
                        ordinal=request_ordinal,
                        device=device,
                        llama=llama,
                        context_template=template,
                        context_ordinal=context_ordinal,
                        context_sha256=context_sha256,
                    )
                    if true is not None:
                        raise ODEBFContractError("P1R15 target observed old target")
                    context_values.append(value)
                    processed_tokens += processed
                request_value = torch.stack(context_values).mean()
                request_gradient = torch.autograd.grad(
                    request_value,
                    target_flat,
                    retain_graph=False,
                    create_graph=False,
                )[0].view(target_state.shape)
                if not bool(torch.isfinite(request_gradient.detach()).all()):
                    raise ODEBFContractError("P1R15 request gradient is non-finite")
                detached = request_gradient.detach().to(
                    device="cpu", dtype=torch.float64
                )
                off_mask = torch.ones(BATCH_SIZE, dtype=torch.bool)
                off_mask[request_ordinal] = False
                off_count = int(torch.count_nonzero(detached[:, off_mask]).item())
                if off_count != 0:
                    raise ODEBFContractError(
                        "P1R15 request gradient crosses request columns"
                    )
                gradient_columns.append(detached[:, request_ordinal])
                request_gradient_hashes.append(tensor_sha256(detached))
                off_request_nonzero.append(off_count)
                loss_values.append(request_value.detach().to(device="cpu"))

    keys = capture.keys()
    current, residual = overlay.detached_state()
    expected_residual = (
        target_state.detach().to(device="cpu", dtype=torch.float32) - current
    ).contiguous()
    if not torch.equal(residual, expected_residual):
        raise ODEBFContractError("P1R15 full-current residual differs")
    losses = torch.stack(loss_values).to(dtype=torch.float64)
    gradient = torch.stack(gradient_columns, dim=1).contiguous()
    target_step = build_per_request_gdual_target_step(
        target_state=target_state,
        z_base=z_base,
        loss_by_request=losses,
        gradient_by_request=gradient,
        step_index=step_index,
        cumulative_g_path_before=cumulative_g_path_before,
    )
    desired = target_step.desired_target
    desired_residual = (desired - current).contiguous()
    with _virtual_context(model, cumulative_factors_by_weight):
        goal = evaluate_common_cold_objective(
            model,
            tokenizer,
            batch,
            contexts,
            layer_name=target_layer_name,
            lookup_positions=lookup_positions,
            residual=desired_residual,
            require_gradient=False,
        )
    if _model_state(model) != before or _rng_identity() != before_rng:
        raise ODEBFStateError("P1R15 target/factor group mutated model or RNG")
    payload = {
        "schema": f"{P1R15_SCHEMA}-target-factor-group",
        "request_order_sha256": order,
        "context_sha256": context_sha256,
        "target_state_sha256": tensor_sha256(target_state),
        "terminal_current_z_sha256": tensor_sha256(current),
        "residual_sha256": tensor_sha256(residual),
        "desired_target_sha256": tensor_sha256(desired),
        "loss_by_request": [float(item) for item in losses],
        "gradient_by_request_sha256": tensor_sha256(gradient),
        "request_gradient_sha256": request_gradient_hashes,
        "off_request_gradient_nonzero_count": off_request_nonzero,
        "target_step": target_step.raw_free_payload(),
        "goal_target_new_nll": float(goal.value),
        "key_sha256_by_layer": [
            [layer, tensor_sha256(keys[layer])]
            for layer in PHYSICAL_WRITER_LAYER_ORDER
        ],
        "layer_hook_call_count": [
            [layer, len(capture.values[layer])]
            for layer in PHYSICAL_WRITER_LAYER_ORDER
        ],
        "logical_forward_groups": 1,
        "model_forward_calls": 120,
        "physical_target_graphs": BATCH_SIZE,
        "autograd_backend_invocations": BATCH_SIZE,
        "backward_calls": BATCH_SIZE,
        "processed_token_count": processed_tokens + goal.processed_token_count,
        "old_target_access_count": 0,
        "heldout_access_count": 0,
        "native_or_direct_z_access_count": 0,
    }
    observed_forward_calls = PHYSICAL_WRITER_CONTEXT_COUNT + goal.model_forward_count
    if observed_forward_calls != 120:
        raise ODEBFContractError("P1R15 target/factor forward accounting differs")
    receipt = PerRequestTargetFactorGroupReceipt(
        order,
        context_sha256,
        tensor_sha256(target_state),
        tensor_sha256(current),
        tensor_sha256(residual),
        tensor_sha256(desired),
        tuple(float(item) for item in losses),
        tensor_sha256(gradient),
        tuple(request_gradient_hashes),
        tuple(off_request_nonzero),
        target_step,
        float(goal.value),
        tuple(
            (layer, tensor_sha256(keys[layer]))
            for layer in PHYSICAL_WRITER_LAYER_ORDER
        ),
        tuple(
            (layer, len(capture.values[layer]))
            for layer in PHYSICAL_WRITER_LAYER_ORDER
        ),
        1,
        observed_forward_calls,
        BATCH_SIZE,
        BATCH_SIZE,
        BATCH_SIZE,
        int(payload["processed_token_count"]),
        0,
        0,
        0,
        canonical_hash(payload),
    )
    return keys, current, residual, desired, receipt


__all__ = [
    "PerRequestTargetFactorGroupReceipt",
    "perrequest_dynamic_target_factor_group",
]
