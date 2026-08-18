"""Model-backed streaming primitives for the P1R23 scalable runtime.

The legacy controller helpers are intentionally B10-shaped.  P1R23 keeps
their tokenizer/span convention but owns an arbitrary-B plan whose row
ordinals survive deterministic length bucketing.  The same plan is used for
the physical W-only world and for the target-intervention world, so a partial
microbatch can never be mistaken for a new scientific batch.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterator, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .scalable_batched_runtime import scalable_ordered_request_digest
from .scalable_batched_runtime import (
    P1R23_CONTEXTS_PER_REQUEST,
    P1R23_LAYER_ORDER,
    StreamingBatchPlan,
    StreamingCaptureMicrobatch,
    StreamingPhysicalCapture,
    accumulate_streaming_capture,
    build_streaming_batch_plan,
)
from .target_new_nll import (
    RoutingObjective,
    _PreparedTargetNewBatch,
    _encoding_sha256,
    _input_ids,
    _is_llama,
    _left_padding_offsets,
    _locked_context_templates,
    _score_prepared_target_new_batch,
    _suffix_tokens,
    _surface,
    _temporary_left_padding,
    _tokenize_left,
)


def _parameter_inventory_sha256(model: torch.nn.Module) -> str:
    """Bind state identity without hashing every frozen model byte per pass."""

    return canonical_hash(
        {
            "parameters": [
                [
                    name,
                    int(value.data_ptr()),
                    int(value._version),
                    str(value.dtype),
                    list(value.shape),
                    bool(value.requires_grad),
                ]
                for name, value in model.named_parameters()
            ],
            "buffers": [
                [name, int(value.data_ptr()), int(value._version), str(value.dtype)]
                for name, value in model.named_buffers()
            ],
            "training": [
                [name, bool(module.training)]
                for name, module in model.named_modules()
            ],
        }
    )


def _tensor_identity(value: torch.Tensor) -> str:
    observed = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(observed.dtype).encode("utf-8"))
    digest.update(str(tuple(observed.shape)).encode("utf-8"))
    digest.update(observed.numpy().tobytes())
    return digest.hexdigest()


def _request_identities(
    requests: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...], str]:
    batch = tuple(requests)
    if not batch:
        raise ODEBFContractError("P1R23 objective batch is empty")
    identities: list[str] = []
    for request in batch:
        if not isinstance(request, Mapping):
            raise ODEBFContractError("P1R23 objective request is not a mapping")
        identity = request.get("request_sha256")
        if not isinstance(identity, str) or len(identity) != 64:
            raise ODEBFContractError("P1R23 objective request identity differs")
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ODEBFContractError("P1R23 objective request identity is duplicated")
    ordered = tuple(identities)
    return batch, ordered, scalable_ordered_request_digest(ordered)


def _normalized_lookup_position(
    tokenizer: Any,
    *,
    prompt_template: str,
    subject: str,
    fact_token_strategy: str,
    prefix_ids: Sequence[int],
) -> int:
    from easyeditor.models.alphaedit.compute_z import find_fact_lookup_idx

    raw = int(
        find_fact_lookup_idx(
            prompt_template,
            subject,
            tokenizer,
            fact_token_strategy,
            verbose=False,
        )
    )
    length = len(prefix_ids)
    position = raw if raw >= 0 else length + raw
    if position < 0 or position >= length:
        raise ODEBFContractError("P1R23 lookup position is out of prefix bounds")
    return position


@dataclass(frozen=True, slots=True)
class ScalableObjectiveMicrobatch:
    prepared: _PreparedTargetNewBatch
    padded_lookup_positions: tuple[int, ...]
    raw_lookup_positions: tuple[int, ...]
    input_sha256: str
    attention_sha256: str
    suffix_mask_sha256: str
    lookup_sha256: str

    @property
    def request_ordinals(self) -> tuple[int, ...]:
        return self.prepared.request_ordinals


@dataclass(frozen=True, slots=True)
class ScalableObjectivePlan:
    request_count: int
    request_sha256: tuple[str, ...]
    request_order_sha256: str
    context_sha256: str
    request_microbatch_size: int
    llama: bool
    batches: tuple[ScalableObjectiveMicrobatch, ...]
    length_bucket_request_order: tuple[int, ...]
    processed_token_count: int
    padded_token_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r23-streaming-objective-plan/v1",
            "request_count": self.request_count,
            "request_order_sha256": self.request_order_sha256,
            "context_sha256": self.context_sha256,
            "context_count": P1R23_CONTEXTS_PER_REQUEST,
            "request_microbatch_size": self.request_microbatch_size,
            "llama": self.llama,
            "length_bucket_request_order": list(self.length_bucket_request_order),
            "batch_request_ordinals": [
                list(item.request_ordinals) for item in self.batches
            ],
            "encoding_sha256": [
                item.prepared.encoding_sha256 for item in self.batches
            ],
            "input_sha256": [item.input_sha256 for item in self.batches],
            "attention_sha256": [item.attention_sha256 for item in self.batches],
            "suffix_mask_sha256": [
                item.suffix_mask_sha256 for item in self.batches
            ],
            "lookup_sha256": [item.lookup_sha256 for item in self.batches],
            "processed_token_count": self.processed_token_count,
            "padded_token_count": self.padded_token_count,
            "old_target_access_count": 0,
            "mean_of_means_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def build_scalable_objective_plan(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    contexts: Sequence[Sequence[str]],
    request_microbatch_size: int,
    fact_token_strategy: str,
) -> ScalableObjectivePlan:
    """Tokenize the full B objective once and preserve global row ordinals."""

    batch, identities, request_order = _request_identities(requests)
    request_count = len(batch)
    if (
        isinstance(request_microbatch_size, bool)
        or not isinstance(request_microbatch_size, int)
        or request_microbatch_size <= 0
        or request_microbatch_size > request_count
    ):
        raise ODEBFContractError("P1R23 objective microbatch size differs")
    context_templates, group_sizes, context_sha = _locked_context_templates(
        RoutingObjective.TARGET_NEW_NLL, contexts
    )
    if group_sizes != (1, 5) or len(context_templates) != 6:
        raise ODEBFContractError("P1R23 objective context geometry differs")
    llama = _is_llama(model)
    request_rows: dict[
        int, list[tuple[str, tuple[int, ...], int, int, int]]
    ] = {}
    request_lengths: list[int] = []
    with _temporary_left_padding(tokenizer):
        for ordinal, (request, identity) in enumerate(
            zip(batch, identities, strict=True)
        ):
            rows: list[tuple[str, tuple[int, ...], int, int, int]] = []
            maximum_length = 0
            for context_ordinal, context_template in enumerate(context_templates):
                prefix, target_new = _surface(
                    request,
                    "target_new",
                    context_template=context_template,
                )
                prefix_ids = _input_ids(
                    _tokenize_left(tokenizer, [prefix]),
                    label="P1R23 objective prefix",
                    one_row=True,
                )[0]
                tokens = _suffix_tokens(tokenizer, target_new, llama=llama)
                prompt_template = (
                    str(request["prompt"])
                    if context_template is None
                    else str(context_template).format(str(request["prompt"]))
                )
                raw_lookup = _normalized_lookup_position(
                    tokenizer,
                    prompt_template=prompt_template,
                    subject=str(request["subject"]),
                    fact_token_strategy=fact_token_strategy,
                    prefix_ids=prefix_ids,
                )
                text = f"{prefix} {target_new}"
                maximum_length = max(
                    maximum_length, len(prefix_ids) + len(tokens)
                )
                rows.append(
                    (text, tokens, len(prefix_ids), context_ordinal, raw_lookup)
                )
            request_rows[ordinal] = rows
            request_lengths.append(maximum_length)
        plan = build_streaming_batch_plan(
            identities,
            request_lengths,
            microbatch_size=request_microbatch_size,
        )
        prepared_batches: list[ScalableObjectiveMicrobatch] = []
        processed_total = 0
        padded_total = 0
        for microbatch in plan.batches:
            texts: list[str] = []
            row_ordinals: list[int] = []
            row_identities: list[str] = []
            prefix_lengths: list[int] = []
            row_tokens: list[tuple[int, ...]] = []
            context_ordinals: list[int] = []
            raw_lookups: list[int] = []
            for ordinal in microbatch.request_ordinals:
                for text, tokens, prefix_length, context_ordinal, raw_lookup in request_rows[ordinal]:
                    texts.append(text)
                    row_ordinals.append(ordinal)
                    row_identities.append(identities[ordinal])
                    prefix_lengths.append(prefix_length)
                    row_tokens.append(tokens)
                    context_ordinals.append(context_ordinal)
                    raw_lookups.append(raw_lookup)
            encoded_raw = _tokenize_left(
                tokenizer, texts, padding=True, return_tensors="pt"
            )
            encoding = {
                str(name): value.detach().to(device="cpu").contiguous().clone()
                for name, value in encoded_raw.items()
                if isinstance(value, torch.Tensor)
            }
            if set(encoding) < {"input_ids", "attention_mask"}:
                raise ODEBFContractError("P1R23 objective encoding differs")
            input_ids, left_padding = _left_padding_offsets(
                encoding, rows=len(row_ordinals)
            )
            padded_lookup = tuple(
                int(left_padding[row] + raw_lookups[row])
                for row in range(len(raw_lookups))
            )
            for row, position in enumerate(padded_lookup):
                if position < 0 or position >= input_ids.shape[1]:
                    raise ODEBFContractError("P1R23 padded lookup is out of range")
            suffix_mask = torch.zeros_like(encoding["attention_mask"], dtype=torch.uint8)
            for row, (prefix_length, tokens) in enumerate(
                zip(prefix_lengths, row_tokens, strict=True)
            ):
                start = left_padding[row] + prefix_length
                stop = start + len(tokens)
                if start < 0 or stop > suffix_mask.shape[1]:
                    raise ODEBFContractError("P1R23 suffix mask geometry differs")
                suffix_mask[row, start:stop] = 1
            prepared = _PreparedTargetNewBatch(
                tuple(microbatch.request_ordinals),
                tuple(identities[item] for item in microbatch.request_ordinals),
                tuple(row_ordinals),
                tuple(row_identities),
                tuple(prefix_lengths),
                tuple(row_tokens),
                tuple(context_ordinals),
                encoding,
                _encoding_sha256(encoding),
            )
            processed = int(encoding["attention_mask"].sum().item())
            padded = int(encoding["attention_mask"].numel())
            processed_total += processed
            padded_total += padded
            prepared_batches.append(
                ScalableObjectiveMicrobatch(
                    prepared,
                    padded_lookup,
                    tuple(raw_lookups),
                    _tensor_identity(encoding["input_ids"]),
                    _tensor_identity(encoding["attention_mask"]),
                    _tensor_identity(suffix_mask),
                    canonical_hash(
                        {
                            "row_request_ordinals": row_ordinals,
                            "row_context_ordinals": context_ordinals,
                            "raw": raw_lookups,
                            "padded": list(padded_lookup),
                        }
                    ),
                )
            )
    payload = {
        "schema": "ode-edit-s05-p1r23-streaming-objective-plan/v1",
        "request_count": request_count,
        "request_order_sha256": request_order,
        "context_sha256": context_sha,
        "request_microbatch_size": request_microbatch_size,
        "length_bucket_request_order": list(plan.bucket_order),
        "batch_request_ordinals": [
            list(item.request_ordinals) for item in prepared_batches
        ],
        "encoding_sha256": [
            item.prepared.encoding_sha256 for item in prepared_batches
        ],
        "lookup_sha256": [item.lookup_sha256 for item in prepared_batches],
        "processed_token_count": processed_total,
        "padded_token_count": padded_total,
        "old_target_access_count": 0,
    }
    return ScalableObjectivePlan(
        request_count,
        identities,
        request_order,
        context_sha,
        request_microbatch_size,
        llama,
        tuple(prepared_batches),
        plan.bucket_order,
        processed_total,
        padded_total,
        canonical_hash(payload),
    )


def _batch_axis_zero(value: torch.Tensor, rows: int) -> torch.Tensor:
    if value.ndim != 3:
        raise ODEBFContractError("P1R23 hooked activation rank differs")
    if value.shape[0] == rows:
        return value
    if value.shape[1] == rows:
        return value.transpose(0, 1)
    raise ODEBFContractError("P1R23 hooked activation batch axis differs")


def _restore_batch_axis(value: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    return value if original.shape[0] == value.shape[0] else value.transpose(0, 1)


def _unwrap(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise ODEBFContractError("P1R23 hooked activation differs")


class OrdinalTargetActivationOverlay:
    """Add request-indexed residuals at bucketed objective rows."""

    def __init__(
        self,
        model: torch.nn.Module,
        module_name: str,
        *,
        residual: torch.Tensor,
        row_request_ordinals: Sequence[int],
        padded_lookup_positions: Sequence[int],
    ) -> None:
        self.model = model
        self.module_name = module_name
        self.residual = residual
        self.row_request_ordinals = tuple(int(item) for item in row_request_ordinals)
        self.padded_lookup_positions = tuple(
            int(item) for item in padded_lookup_positions
        )
        self._handle: Any | None = None
        self.fire_count = 0

    def __enter__(self) -> "OrdinalTargetActivationOverlay":
        module = self.model.get_submodule(self.module_name)

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            raw = _unwrap(output)
            rows = len(self.row_request_ordinals)
            batch_first = _batch_axis_zero(raw, rows).clone()
            if (
                len(self.padded_lookup_positions) != rows
                or self.residual.ndim != 2
                or self.residual.shape[1] <= max(self.row_request_ordinals)
                or self.residual.shape[0] != batch_first.shape[2]
            ):
                raise ODEBFContractError("P1R23 target overlay geometry differs")
            selected = self.residual[:, list(self.row_request_ordinals)].T.to(
                device=batch_first.device, dtype=batch_first.dtype
            )
            for row, position in enumerate(self.padded_lookup_positions):
                if position < 0 or position >= batch_first.shape[1]:
                    raise ODEBFContractError("P1R23 target overlay lookup differs")
                batch_first[row, position, :] = (
                    batch_first[row, position, :] + selected[row]
                )
            replaced = _restore_batch_axis(batch_first, raw)
            self.fire_count += 1
            if isinstance(output, tuple):
                return (replaced, *output[1:])
            return replaced

        self._handle = module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._handle = None


@dataclass(frozen=True, slots=True)
class ScalableObjectiveResult:
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

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r23-streaming-target-new-objective/v1",
            "loss": self.loss,
            "per_request_values": list(self.per_request_values),
            "request_order_sha256": self.request_order_sha256,
            "target_span_sha256": self.target_span_sha256,
            "suffix_token_counts": list(self.suffix_token_counts),
            "model_forward_count": self.model_forward_count,
            "backward_count": self.backward_count,
            "processed_token_count": self.processed_token_count,
            "padded_token_count": self.padded_token_count,
            "target_gradient_sha256": (
                None if self.target_gradient is None else tensor_sha256(self.target_gradient)
            ),
            "coefficient_gradient_sha256": (
                None
                if self.coefficient_gradient is None
                else tensor_sha256(self.coefficient_gradient)
            ),
            "plan_sha256": self.plan_sha256,
            "model_state_sha256": self.model_state_sha256,
            "old_target_access_count": 0,
            "generation_call_count": 0,
            "mean_of_means_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def evaluate_scalable_target_new_objective(
    model: torch.nn.Module,
    plan: ScalableObjectivePlan,
    *,
    target_state: torch.Tensor | None = None,
    current_terminal: torch.Tensor | None = None,
    target_layer_name: str | None = None,
    coefficient_layers: Sequence[Any] | None = None,
    coefficients: torch.Tensor | None = None,
    target_gradient_required: bool = True,
    request_weights: Sequence[float] | torch.Tensor | None = None,
) -> ScalableObjectiveResult:
    """Evaluate the exact global-B target-new NLL and optional one VJP."""

    target_mode = target_state is not None
    coefficient_mode = coefficients is not None
    weights_cpu: torch.Tensor | None = None
    if request_weights is not None:
        weights_cpu = torch.as_tensor(
            request_weights, device="cpu", dtype=torch.float64
        ).detach().contiguous()
        if (
            weights_cpu.ndim != 1
            or weights_cpu.numel() != plan.request_count
            or not torch.isfinite(weights_cpu).all()
            or bool(torch.any(weights_cpu < 0.0))
            or abs(float(weights_cpu.sum()) - 1.0) > 1.0e-12
        ):
            raise ODEBFContractError("P1R23 request weights differ")
    if target_mode and coefficient_mode:
        raise ODEBFContractError("P1R23 objective worlds were mixed")
    if target_mode != (current_terminal is not None and target_layer_name is not None):
        raise ODEBFContractError("P1R23 target objective inputs differ")
    if coefficient_mode != (coefficient_layers is not None):
        raise ODEBFContractError("P1R23 coefficient objective inputs differ")
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
        raise ODEBFContractError("P1R23 target objective geometry differs")
    if coefficient_mode and (
        coefficients is None
        or coefficient_layers is None
        or coefficients.ndim != 1
        or coefficients.numel() != len(coefficient_layers)
        or coefficients.numel() <= 0
        or coefficients.numel() > len(P1R23_LAYER_ORDER)
        or len({int(item.layer) for item in coefficient_layers})
        != len(coefficient_layers)
        or any(
            int(item.layer) not in P1R23_LAYER_ORDER
            for item in coefficient_layers
        )
        or not coefficients.requires_grad
        or not torch.isfinite(coefficients).all()
    ):
        raise ODEBFContractError("P1R23 coefficient objective geometry differs")

    from .fixed_e8_runtime import _CoefficientOverlay

    device = next(model.parameters()).device
    before = _parameter_inventory_sha256(model)
    values: list[torch.Tensor | None] = [None] * plan.request_count
    counts: list[int | None] = [None] * plan.request_count
    spans: list[str | None] = [None] * plan.request_count
    target_gradient = (
        torch.zeros_like(target_state, dtype=torch.float64, device="cpu")
        if target_mode and target_gradient_required
        else None
    )
    coefficient_gradient = (
        torch.zeros_like(coefficients, dtype=torch.float64, device="cpu")
        if coefficient_mode
        else None
    )
    backward_count = 0
    processed = 0
    overlay_fire_count = 0
    for batch in plan.batches:
        with ExitStack() as stack:
            target_overlay: OrdinalTargetActivationOverlay | None = None
            if target_mode:
                assert target_state is not None
                assert current_terminal is not None
                assert target_layer_name is not None
                residual = target_state - current_terminal.to(
                    device=target_state.device, dtype=target_state.dtype
                )
                target_overlay = stack.enter_context(
                    OrdinalTargetActivationOverlay(
                        model,
                        target_layer_name,
                        residual=residual,
                        row_request_ordinals=batch.prepared.row_request_ordinals,
                        padded_lookup_positions=batch.padded_lookup_positions,
                    )
                )
            if coefficient_mode:
                assert coefficient_layers is not None and coefficients is not None
                stack.enter_context(
                    _CoefficientOverlay(model, coefficient_layers, coefficients)
                )
            observed, batch_processed = _score_prepared_target_new_batch(
                model,
                batch.prepared,
                device=device,
                llama=plan.llama,
                context_sha256=plan.context_sha256,
            )
            request_sum = (
                torch.stack([item[1] for item in observed]).sum()
                if weights_cpu is None
                else torch.stack(
                    [
                        item[1]
                        * weights_cpu[item[0]].to(
                            device=item[1].device, dtype=item[1].dtype
                        )
                        for item in observed
                    ]
                ).sum()
            )
            if target_mode and target_gradient_required:
                assert target_state is not None and target_gradient is not None
                gradient = torch.autograd.grad(
                    request_sum, target_state, retain_graph=False, create_graph=False
                )[0]
                target_gradient.add_(gradient.detach().to(device="cpu", dtype=torch.float64))
                backward_count += 1
                assert target_overlay is not None
                overlay_fire_count += target_overlay.fire_count
            elif target_mode:
                assert target_overlay is not None
                overlay_fire_count += target_overlay.fire_count
            elif coefficient_mode:
                assert coefficients is not None and coefficient_gradient is not None
                gradient = torch.autograd.grad(
                    request_sum, coefficients, retain_graph=False, create_graph=False
                )[0]
                coefficient_gradient.add_(
                    gradient.detach().to(device="cpu", dtype=torch.float64)
                )
                backward_count += 1
            for ordinal, value, count, span in observed:
                if values[ordinal] is not None:
                    raise ODEBFContractError("P1R23 objective duplicated a request")
                values[ordinal] = value.detach().to(device="cpu", dtype=torch.float64)
                counts[ordinal] = int(count)
                spans[ordinal] = str(span)
            processed += int(batch_processed)
    if (
        any(item is None for item in values)
        or any(item is None for item in counts)
        or any(item is None for item in spans)
        or processed != plan.processed_token_count
    ):
        raise ODEBFContractError("P1R23 objective request coverage differs")
    if target_mode and overlay_fire_count != len(plan.batches):
        raise ODEBFContractError("P1R23 target overlay firing count differs")
    if target_gradient is not None:
        if weights_cpu is None:
            target_gradient.div_(plan.request_count)
        target_gradient = target_gradient.to(dtype=torch.float32).contiguous()
    if coefficient_gradient is not None:
        if weights_cpu is None:
            coefficient_gradient.div_(plan.request_count)
        coefficient_gradient = coefficient_gradient.to(dtype=torch.float32).contiguous()
    final_values = tuple(float(item) for item in values if item is not None)
    loss = (
        math.fsum(final_values) / plan.request_count
        if weights_cpu is None
        else math.fsum(
            float(weights_cpu[index]) * value
            for index, value in enumerate(final_values)
        )
    )
    final_spans = tuple(str(item) for item in spans if item is not None)
    final_counts = tuple(int(item) for item in counts if item is not None)
    after = _parameter_inventory_sha256(model)
    if after != before:
        raise ODEBFStateError("P1R23 objective mutated model state")
    payload = {
        "loss": loss,
        "per_request_values": list(final_values),
        "request_order_sha256": plan.request_order_sha256,
        "target_span_sha256": canonical_hash(list(final_spans)),
        "suffix_token_counts": list(final_counts),
        "model_forward_count": len(plan.batches),
        "backward_count": backward_count,
        "processed_token_count": processed,
        "padded_token_count": plan.padded_token_count,
        "target_gradient_sha256": (
            None if target_gradient is None else tensor_sha256(target_gradient)
        ),
        "coefficient_gradient_sha256": (
            None if coefficient_gradient is None else tensor_sha256(coefficient_gradient)
        ),
        "plan_sha256": plan.identity_sha256,
        "model_state_sha256": after,
    }
    if weights_cpu is not None:
        payload.update(
            {
                "request_weights": [float(item) for item in weights_cpu],
                "request_weight_sum": float(weights_cpu.sum()),
                "request_weight_sha256": tensor_sha256(weights_cpu),
                "request_weight_detached": True,
            }
        )
    return ScalableObjectiveResult(
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
        coefficient_gradient,
        plan.identity_sha256,
        after,
        canonical_hash(payload),
    )


@dataclass(frozen=True, slots=True)
class _CapturePreparedBatch:
    request_ordinals: tuple[int, ...]
    encoding: Mapping[str, torch.Tensor]
    padded_lookup_positions: tuple[int, ...]
    row_request_ordinals: tuple[int, ...]
    row_context_ordinals: tuple[int, ...]
    encoding_sha256: str


@dataclass(frozen=True, slots=True)
class ScalableCapturePlan:
    streaming_plan: StreamingBatchPlan
    batches: tuple[_CapturePreparedBatch, ...]
    request_order_sha256: str
    context_sha256: str
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r23-streaming-capture-plan/v1",
            "request_count": self.streaming_plan.request_count,
            "microbatch_size": self.streaming_plan.microbatch_size,
            "request_order_sha256": self.request_order_sha256,
            "context_sha256": self.context_sha256,
            "bucket_order": list(self.streaming_plan.bucket_order),
            "batch_request_ordinals": [
                list(item.request_ordinals) for item in self.batches
            ],
            "encoding_sha256": [item.encoding_sha256 for item in self.batches],
            "lookup_sha256": [
                canonical_hash(
                    {
                        "row_request_ordinals": list(item.row_request_ordinals),
                        "row_context_ordinals": list(item.row_context_ordinals),
                        "padded_lookup_positions": list(item.padded_lookup_positions),
                    }
                )
                for item in self.batches
            ],
            "identity_sha256": self.identity_sha256,
        }


def build_scalable_capture_plan(
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    contexts: Sequence[Sequence[str]],
    request_microbatch_size: int,
    fact_token_strategy: str,
) -> ScalableCapturePlan:
    batch, identities, request_order = _request_identities(requests)
    groups = tuple(tuple(group) for group in contexts)
    if tuple(len(group) for group in groups) != (1, 5):
        raise ODEBFContractError("P1R23 capture contexts differ")
    templates = tuple(item for group in groups for item in group)
    context_sha = canonical_hash(
        {"group_sizes": [1, 5], "templates": list(templates)}
    )
    rows_by_request: dict[int, list[tuple[str, int, int]]] = {}
    lengths: list[int] = []
    original_padding = tokenizer.padding_side
    if original_padding not in ("left", "right"):
        raise ODEBFContractError("P1R23 capture padding side differs")
    try:
        tokenizer.padding_side = "left"
        for ordinal, request in enumerate(batch):
            rows: list[tuple[str, int, int]] = []
            maximum = 0
            for context_ordinal, context in enumerate(templates):
                prompt_template = context.format(str(request["prompt"]))
                rendered = prompt_template.format(str(request["subject"]))
                ids = _input_ids(
                    tokenizer([rendered]),
                    label="P1R23 capture prompt",
                    one_row=True,
                )[0]
                lookup = _normalized_lookup_position(
                    tokenizer,
                    prompt_template=prompt_template,
                    subject=str(request["subject"]),
                    fact_token_strategy=fact_token_strategy,
                    prefix_ids=ids,
                )
                rows.append((rendered, lookup, context_ordinal))
                maximum = max(maximum, len(ids))
            rows_by_request[ordinal] = rows
            lengths.append(maximum)
        streaming = build_streaming_batch_plan(
            identities,
            lengths,
            microbatch_size=request_microbatch_size,
        )
        prepared: list[_CapturePreparedBatch] = []
        for microbatch in streaming.batches:
            texts: list[str] = []
            raw_positions: list[int] = []
            row_ordinals: list[int] = []
            context_ordinals: list[int] = []
            for ordinal in microbatch.request_ordinals:
                for rendered, lookup, context_ordinal in rows_by_request[ordinal]:
                    texts.append(rendered)
                    raw_positions.append(lookup)
                    row_ordinals.append(ordinal)
                    context_ordinals.append(context_ordinal)
            encoded_raw = tokenizer(texts, padding=True, return_tensors="pt")
            encoding = {
                str(name): value.detach().to(device="cpu").contiguous().clone()
                for name, value in encoded_raw.items()
                if isinstance(value, torch.Tensor)
            }
            _, left_padding = _left_padding_offsets(
                encoding, rows=len(row_ordinals)
            )
            padded_positions = tuple(
                int(left_padding[row] + raw_positions[row])
                for row in range(len(raw_positions))
            )
            prepared.append(
                _CapturePreparedBatch(
                    tuple(microbatch.request_ordinals),
                    encoding,
                    padded_positions,
                    tuple(row_ordinals),
                    tuple(context_ordinals),
                    _encoding_sha256(encoding),
                )
            )
    finally:
        tokenizer.padding_side = original_padding
    payload = {
        "request_order_sha256": request_order,
        "context_sha256": context_sha,
        "streaming_plan_sha256": streaming.identity_sha256,
        "encoding_sha256": [item.encoding_sha256 for item in prepared],
        "batch_request_ordinals": [list(item.request_ordinals) for item in prepared],
    }
    return ScalableCapturePlan(
        streaming,
        tuple(prepared),
        request_order,
        context_sha,
        canonical_hash(payload),
    )


def _select_rows(value: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    batch_first = _batch_axis_zero(value, len(indices))
    selected: list[torch.Tensor] = []
    for row, position in enumerate(indices):
        if position < 0 or position >= batch_first.shape[1]:
            raise ODEBFContractError("P1R23 capture lookup is out of range")
        selected.append(batch_first[row, position, :])
    return torch.stack(selected)


def capture_scalable_physical_state(
    model: torch.nn.Module,
    plan: ScalableCapturePlan,
    hparams: Any,
) -> StreamingPhysicalCapture:
    """Capture all five key layers and canonical terminal by microbatch."""

    from easyeditor.util import nethook

    layers = tuple(int(item) for item in hparams.layers)
    if layers != P1R23_LAYER_ORDER:
        raise ODEBFContractError("P1R23 capture layer order differs")
    module_names = tuple(hparams.rewrite_module_tmp.format(layer) for layer in layers)
    terminal_module_name = hparams.layer_module_tmp.format(layers[-1])
    trace_names = tuple(dict.fromkeys((*module_names, terminal_module_name)))
    prepared_by_ordinals = {
        item.request_ordinals: item for item in plan.batches
    }
    original_padding = getattr(hparams, "_p1r23_tokenizer_padding_side", None)
    del original_padding  # hparams must not carry mutable tokenizer state.

    def forward(microbatch: Any) -> StreamingCaptureMicrobatch:
        prepared = prepared_by_ordinals[microbatch.request_ordinals]
        before = _parameter_inventory_sha256(model)
        device = next(model.parameters()).device
        encoding = {
            name: value.to(device=device, non_blocking=True)
            for name, value in prepared.encoding.items()
        }
        with torch.no_grad():
            with nethook.TraceDict(
                model,
                trace_names,
                retain_input=True,
                retain_output=True,
                stop=True,
            ) as traces:
                model(**encoding)
        after = _parameter_inventory_sha256(model)
        selected_by_layer: dict[int, torch.Tensor] = {}
        local_count = len(microbatch.request_ordinals)
        for layer, name in zip(layers, module_names, strict=True):
            selected = _select_rows(
                _unwrap(traces[name].input), prepared.padded_lookup_positions
            )
            rows = selected.reshape(
                local_count, P1R23_CONTEXTS_PER_REQUEST, selected.shape[-1]
            )
            selected_by_layer[layer] = rows.detach().to(
                device="cpu", dtype=torch.float32
            )
        terminal_rows = _select_rows(
            _unwrap(traces[terminal_module_name].output),
            prepared.padded_lookup_positions,
        ).reshape(local_count, P1R23_CONTEXTS_PER_REQUEST, -1)
        canonical = terminal_rows[:, 0, :].detach().to(
            device="cpu", dtype=torch.float32
        )
        processed = int(encoding["attention_mask"].sum().detach().cpu().item())
        padded = int(encoding["attention_mask"].numel())
        return StreamingCaptureMicrobatch(
            tuple(microbatch.request_ordinals),
            selected_by_layer,
            canonical,
            before,
            after,
            processed,
            padded,
        )

    observed = accumulate_streaming_capture(plan.streaming_plan, forward)
    payload = observed.raw_free_payload()
    payload["capture_plan_sha256"] = plan.identity_sha256
    payload["request_order_sha256"] = plan.request_order_sha256
    payload["dynamic_state_capture"] = True
    return StreamingPhysicalCapture(
        observed.keys_by_layer,
        observed.terminal_z,
        observed.batch_plan_sha256,
        observed.model_state_sha256,
        observed.physical_forward_count,
        observed.processed_token_count,
        observed.padded_token_count,
        canonical_hash(payload),
    )


__all__ = [
    "OrdinalTargetActivationOverlay",
    "ScalableCapturePlan",
    "ScalableObjectivePlan",
    "ScalableObjectiveResult",
    "build_scalable_capture_plan",
    "build_scalable_objective_plan",
    "capture_scalable_physical_state",
    "evaluate_scalable_target_new_objective",
]
