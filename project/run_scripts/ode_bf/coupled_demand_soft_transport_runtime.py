"""P1R16 context-batched target capture and coupled physical demand."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from .cold_start_target import _rewrap_layer_output, _rng_identity, _unwrap_layer_output
from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import WaypointFactor, tensor_sha256
from .integrated_physical_writer import (
    PHYSICAL_WRITER_CONTEXT_COUNT,
    PHYSICAL_WRITER_H,
    PHYSICAL_WRITER_LAYER_ORDER,
    PHYSICAL_WRITER_MICROBATCH_GROUPS,
)
from .p1_backend import _virtual_context
from .coupled_demand_soft_transport import (
    P1R16_SCHEMA,
    PerRequestTargetStep,
    build_per_request_gdual_target_step,
)
from .request_digest import ordered_request_digest_v1
from .target_new_nll import (
    RoutingObjective,
    _input_ids,
    _is_llama,
    _left_padding_offsets,
    _locked_context_templates,
    _model_device,
    _model_state,
    _move_encoding,
    _ordered_batch,
    _score_suffix,
    _suffix_tokens,
    _surface,
    _temporary_left_padding,
    _tokenize_left,
)


@dataclass(frozen=True, slots=True)
class CoupledDemandReceipt:
    current_z_sha256: str
    target_state_sha256: str
    desired_target_sha256: str
    old_residual_sha256: str
    target_velocity_sha256: str
    factor_demand_sha256: str
    factor_left_identity: str
    division_by_h_count: int
    applied_h_count: int
    composition_max_abs_error: float
    old_z_omission_count: int
    native_or_direct_z_access_count: int
    factor_demand: torch.Tensor
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("factor_demand")
        return payload


def build_coupled_physical_demand(
    *,
    current_z: torch.Tensor,
    target_state: torch.Tensor,
    desired_target: torch.Tensor,
) -> CoupledDemandReceipt:
    """Build d=(z_des-H_T(W))/h with one and only one division by h."""

    if (
        not isinstance(current_z, torch.Tensor)
        or not isinstance(target_state, torch.Tensor)
        or not isinstance(desired_target, torch.Tensor)
        or current_z.shape != target_state.shape
        or desired_target.shape != target_state.shape
        or current_z.ndim != 2
        or current_z.shape[1] != BATCH_SIZE
    ):
        raise ODEBFContractError("P1R16 coupled demand geometry differs")
    current64 = current_z.detach().to(device="cpu", dtype=torch.float64)
    target64 = target_state.detach().to(device="cpu", dtype=torch.float64)
    desired64 = desired_target.detach().to(device="cpu", dtype=torch.float64)
    if not bool(
        torch.isfinite(current64).all()
        and torch.isfinite(target64).all()
        and torch.isfinite(desired64).all()
    ):
        raise ODEBFContractError("P1R16 coupled demand is non-finite")
    old_residual64 = target64 - current64
    target_velocity64 = (desired64 - target64) / PHYSICAL_WRITER_H
    demand64 = (desired64 - current64) / PHYSICAL_WRITER_H
    composed64 = target_velocity64 + old_residual64 / PHYSICAL_WRITER_H
    error = float(torch.max(torch.abs(demand64 - composed64)))
    scale = max(float(torch.max(torch.abs(demand64))), 1.0)
    if error > 1.0e-12 * scale:
        raise ODEBFContractError("P1R16 coupled demand composition differs")
    demand = demand64.to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(demand).all()):
        raise ODEBFContractError("P1R16 model-facing demand is non-finite")
    payload = {
        "schema": f"{P1R16_SCHEMA}-coupled-demand",
        "current_z_sha256": tensor_sha256(current_z),
        "target_state_sha256": tensor_sha256(target_state),
        "desired_target_sha256": tensor_sha256(desired_target),
        "old_residual_sha256": tensor_sha256(
            old_residual64.to(dtype=torch.float32).contiguous()
        ),
        "target_velocity_sha256": tensor_sha256(
            target_velocity64.to(dtype=torch.float32).contiguous()
        ),
        "factor_demand_sha256": tensor_sha256(demand),
        "factor_left_identity": "d=(z_des-H_T(W_k))/h",
        "division_by_h_count": 1,
        "applied_h_count": 0,
        "composition_max_abs_error": error,
        "old_z_omission_count": 0,
        "native_or_direct_z_access_count": 0,
    }
    return CoupledDemandReceipt(
        payload["current_z_sha256"],
        payload["target_state_sha256"],
        payload["desired_target_sha256"],
        payload["old_residual_sha256"],
        payload["target_velocity_sha256"],
        payload["factor_demand_sha256"],
        payload["factor_left_identity"],
        1,
        0,
        error,
        0,
        0,
        demand,
        canonical_hash(payload),
    )


def padded_subject_positions(
    raw_positions: Sequence[int],
    left_padding: Sequence[int],
    input_ids: torch.Tensor,
    unpadded_input_ids: Sequence[Sequence[int]] | None = None,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Map raw request coordinates into a canonical left-padded B10 batch."""

    raw = tuple(int(item) for item in raw_positions)
    pads = tuple(int(item) for item in left_padding)
    if (
        len(raw) != BATCH_SIZE
        or len(pads) != BATCH_SIZE
        or input_ids.ndim != 2
        or input_ids.shape[0] != BATCH_SIZE
        or any(item < 0 for item in raw)
        or any(item < 0 for item in pads)
        or (
            unpadded_input_ids is not None
            and len(tuple(unpadded_input_ids)) != BATCH_SIZE
        )
    ):
        raise ODEBFContractError("P1R16 padded subject coordinate differs")
    padded: list[int] = []
    token_hashes: list[str] = []
    for row, (position, pad) in enumerate(zip(raw, pads, strict=True)):
        width = int(input_ids.shape[1]) - pad
        shifted = position + pad
        if position >= width or shifted >= input_ids.shape[1]:
            raise ODEBFContractError("P1R16 padded subject coordinate is out of span")
        raw_token = int(input_ids[row, pad + position].detach().to(device="cpu"))
        padded_token = int(input_ids[row, shifted].detach().to(device="cpu"))
        singleton_row = (
            None
            if unpadded_input_ids is None
            else tuple(int(item) for item in unpadded_input_ids[row])
        )
        if singleton_row is not None and position >= len(singleton_row):
            raise ODEBFContractError("P1R16 singleton subject span differs")
        singleton_token = raw_token if singleton_row is None else singleton_row[position]
        if raw_token != padded_token or padded_token != singleton_token:
            raise ODEBFContractError("P1R16 padded subject token identity differs")
        padded.append(shifted)
        token_hashes.append(
            canonical_hash(
                {
                    "row": row,
                    "raw_position": position,
                    "left_padding": pad,
                    "padded_position": shifted,
                    "token_id": padded_token,
                    "singleton_token_exact": True,
                }
            )
        )
    return tuple(padded), tuple(token_hashes)


class _BatchedTargetOverlay(AbstractContextManager["_BatchedTargetOverlay"]):
    def __init__(
        self,
        model: torch.nn.Module,
        layer_name: str,
        target_state: torch.Tensor,
        raw_positions_by_context: Sequence[Sequence[int]],
    ) -> None:
        if (
            target_state.ndim != 2
            or target_state.shape[1] != BATCH_SIZE
            or not target_state.requires_grad
            or len(raw_positions_by_context) != PHYSICAL_WRITER_MICROBATCH_GROUPS
            or any(len(row) != BATCH_SIZE for row in raw_positions_by_context)
        ):
            raise ODEBFContractError("P1R16 batched overlay inventory differs")
        self.model = model
        self.layer_name = layer_name
        self.target_state = target_state
        self.raw_positions = tuple(tuple(int(item) for item in row) for row in raw_positions_by_context)
        self.context: int | None = None
        self.padded_positions: tuple[int, ...] | None = None
        self.calls = 0
        self.current: torch.Tensor | None = None
        self.residual: torch.Tensor | None = None
        self.coordinate_hashes: list[str] = []
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def prepare(
        self,
        context_ordinal: int,
        left_padding: Sequence[int],
        input_ids: torch.Tensor,
        unpadded_input_ids: Sequence[Sequence[int]],
    ) -> None:
        if self.context is not None or context_ordinal != self.calls:
            raise ODEBFContractError("P1R16 batched overlay call order differs")
        positions, hashes = padded_subject_positions(
            self.raw_positions[context_ordinal],
            left_padding,
            input_ids,
            unpadded_input_ids,
        )
        self.context = context_ordinal
        self.padded_positions = positions
        self.coordinate_hashes.extend(hashes)

    def _hook(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> Any:
        if self.context is None or self.padded_positions is None:
            raise ODEBFContractError("P1R16 batched overlay was not prepared")
        activation = _unwrap_layer_output(output)
        if activation.ndim != 3:
            raise ODEBFContractError("P1R16 batched overlay activation rank differs")
        if activation.shape[0] != BATCH_SIZE:
            if activation.shape[1] == BATCH_SIZE:
                activation = activation.transpose(0, 1)
            else:
                raise ODEBFContractError("P1R16 batched overlay layout differs")
        selected = torch.stack(
            [activation[row, position, :] for row, position in enumerate(self.padded_positions)],
            dim=0,
        )
        if self.context == 0:
            current = selected.detach().to(device="cpu", dtype=torch.float32).T.contiguous()
            self.current = current
            self.residual = (
                self.target_state
                - current.to(device=self.target_state.device, dtype=torch.float32)
            ).contiguous()
        if self.residual is None:
            raise ODEBFContractError("P1R16 batched overlay lacks canonical residual")
        patched = activation.clone()
        residual_rows = self.residual.T.to(device=activation.device, dtype=torch.float32)
        for row, position in enumerate(self.padded_positions):
            patched[row, position, :] = selected[row] + residual_rows[row].to(dtype=selected.dtype)
        self.calls += 1
        self.context = None
        self.padded_positions = None
        return _rewrap_layer_output(output, patched)

    def __enter__(self) -> "_BatchedTargetOverlay":
        self._handle = self.model.get_submodule(self.layer_name).register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if exc is None and (
            self.calls != PHYSICAL_WRITER_MICROBATCH_GROUPS
            or self.context is not None
            or self.current is None
            or self.residual is None
        ):
            raise ODEBFContractError("P1R16 batched overlay capture is incomplete")
        return False


class _BatchedWriterKeyCapture(AbstractContextManager["_BatchedWriterKeyCapture"]):
    def __init__(
        self,
        model: torch.nn.Module,
        module_names: Mapping[int, str],
        raw_indices_by_context: Sequence[Sequence[Sequence[int]]],
    ) -> None:
        if (
            tuple(sorted(module_names)) != PHYSICAL_WRITER_LAYER_ORDER
            or len(raw_indices_by_context) != PHYSICAL_WRITER_MICROBATCH_GROUPS
            or any(len(rows) != BATCH_SIZE for rows in raw_indices_by_context)
        ):
            raise ODEBFContractError("P1R16 batched key inventory differs")
        self.model = model
        self.module_names = dict(module_names)
        self.raw_indices = tuple(
            tuple(tuple(int(item) for item in values) for values in rows)
            for rows in raw_indices_by_context
        )
        self.context: int | None = None
        self.left_padding: tuple[int, ...] | None = None
        self.values: dict[int, list[torch.Tensor]] = {layer: [] for layer in PHYSICAL_WRITER_LAYER_ORDER}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def prepare(self, context_ordinal: int, left_padding: Sequence[int]) -> None:
        if self.context is not None or context_ordinal != len(self.values[PHYSICAL_WRITER_LAYER_ORDER[0]]):
            raise ODEBFContractError("P1R16 batched key call order differs")
        pads = tuple(int(item) for item in left_padding)
        if len(pads) != BATCH_SIZE or any(item < 0 for item in pads):
            raise ODEBFContractError("P1R16 batched key padding differs")
        self.context = context_ordinal
        self.left_padding = pads

    def _hook(self, layer: int):
        def apply(_module: torch.nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            if self.context is None or self.left_padding is None or not inputs or not isinstance(inputs[0], torch.Tensor):
                raise ODEBFContractError("P1R16 batched key hook contract differs")
            raw = inputs[0]
            if raw.ndim != 3:
                raise ODEBFContractError("P1R16 batched key rank differs")
            if raw.shape[0] != BATCH_SIZE:
                if raw.shape[1] == BATCH_SIZE:
                    raw = raw.transpose(0, 1)
                else:
                    raise ODEBFContractError("P1R16 batched key layout differs")
            selected: list[torch.Tensor] = []
            for row, raw_indices in enumerate(self.raw_indices[self.context]):
                padded = tuple(index + self.left_padding[row] for index in raw_indices)
                if not padded or min(padded) < 0 or max(padded) >= raw.shape[1]:
                    raise ODEBFContractError("P1R16 batched key position differs")
                selected.append(raw[row, list(padded), :].mean(dim=0))
            self.values[layer].append(
                torch.stack(selected).detach().to(device="cpu", dtype=torch.float32)
            )
        return apply

    def complete_context(self) -> None:
        if self.context is None or any(len(self.values[layer]) != self.context + 1 for layer in PHYSICAL_WRITER_LAYER_ORDER):
            raise ODEBFContractError("P1R16 batched key context is incomplete")
        self.context = None
        self.left_padding = None

    def __enter__(self) -> "_BatchedWriterKeyCapture":
        try:
            for layer in PHYSICAL_WRITER_LAYER_ORDER:
                module = self.model.get_submodule(self.module_names[layer])
                if type(module) is not torch.nn.Linear:
                    raise ODEBFContractError("P1R16 batched key target is not Linear")
                self._handles.append(module.register_forward_hook(self._hook(layer)))
        except BaseException:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        if exc is None and (
            self.context is not None
            or any(len(self.values[layer]) != PHYSICAL_WRITER_MICROBATCH_GROUPS for layer in PHYSICAL_WRITER_LAYER_ORDER)
        ):
            raise ODEBFContractError("P1R16 batched key capture is incomplete")
        return False

    def keys(self) -> dict[int, torch.Tensor]:
        if self._handles or any(len(self.values[layer]) != 6 for layer in PHYSICAL_WRITER_LAYER_ORDER):
            raise ODEBFContractError("P1R16 batched keys are incomplete")
        result: dict[int, torch.Tensor] = {}
        for layer in PHYSICAL_WRITER_LAYER_ORDER:
            result[layer] = aggregate_context_batched_keys(self.values[layer])
        return result


def aggregate_context_batched_keys(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Match the frozen canonical-plus-semantic writer-key aggregation."""

    rows = tuple(values)
    if (
        len(rows) != PHYSICAL_WRITER_MICROBATCH_GROUPS
        or any(
            not isinstance(item, torch.Tensor)
            or item.ndim != 2
            or item.shape[0] != BATCH_SIZE
            or item.shape != rows[0].shape
            or not bool(torch.isfinite(item).all())
            for item in rows
        )
    ):
        raise ODEBFContractError("P1R16 batched key aggregation differs")
    canonical = rows[0]
    semantic = torch.stack(rows[1:], dim=0).mean(dim=0)
    return torch.stack((canonical, semantic), dim=0).mean(dim=0).T.contiguous()


def retain_requestwise_batched_gradients(
    request_losses: Sequence[torch.Tensor],
    target_flat: torch.Tensor,
    target_shape: torch.Size | tuple[int, ...],
) -> tuple[torch.Tensor, tuple[str, ...], tuple[int, ...]]:
    """Retain ten independent gradient columns from shared batched graphs."""

    losses = tuple(request_losses)
    shape = tuple(int(item) for item in target_shape)
    if (
        len(losses) != BATCH_SIZE
        or not isinstance(target_flat, torch.Tensor)
        or not target_flat.requires_grad
        or len(shape) != 2
        or shape[1] != BATCH_SIZE
        or target_flat.numel() != shape[0] * shape[1]
    ):
        raise ODEBFContractError("P1R16 batched gradient inventory differs")
    gradients: list[torch.Tensor] = []
    request_hashes: list[str] = []
    off_request: list[int] = []
    for row, loss in enumerate(losses):
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ODEBFContractError("P1R16 request loss differs")
        gradient = torch.autograd.grad(
            loss,
            target_flat,
            retain_graph=row < BATCH_SIZE - 1,
            create_graph=False,
        )[0].view(shape)
        detached = gradient.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(detached).all()):
            raise ODEBFContractError("P1R16 request gradient is non-finite")
        mask = torch.ones(BATCH_SIZE, dtype=torch.bool)
        mask[row] = False
        off_count = int(torch.count_nonzero(detached[:, mask]))
        if off_count != 0:
            raise ODEBFContractError("P1R16 request gradient crosses columns")
        gradients.append(detached[:, row])
        request_hashes.append(tensor_sha256(detached))
        off_request.append(off_count)
    return (
        torch.stack(gradients, dim=1).contiguous(),
        tuple(request_hashes),
        tuple(off_request),
    )


@dataclass(frozen=True, slots=True)
class PerRequestTargetFactorGroupReceipt:
    request_order_sha256: str
    context_sha256: str
    target_state_sha256: str
    terminal_current_z_sha256: str
    old_residual_sha256: str
    desired_target_sha256: str
    coupled_demand_sha256: str
    loss_by_request: tuple[float, ...]
    gradient_by_request_sha256: str
    request_gradient_sha256: tuple[str, ...]
    off_request_gradient_nonzero_count: tuple[int, ...]
    target_step: PerRequestTargetStep
    goal_target_new_nll: float
    controller_requested_reduction: float
    coordinate_identity_sha256: str
    coordinate_exact_count: int
    coordinate_mismatch_count: int
    left_padding_nonzero_count_canonical: int
    bf16_singleton_batch_exact_gate_count: int
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


def _request_context_raw_positions(
    values: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    if len(values) != PHYSICAL_WRITER_CONTEXT_COUNT:
        raise ODEBFContractError("P1R16 lookup plan count differs")
    return tuple(
        tuple(int(values[row * 6 + context]) for row in range(BATCH_SIZE))
        for context in range(6)
    )


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
    torch.Tensor,
    CoupledDemandReceipt,
    PerRequestTargetFactorGroupReceipt,
]:
    """Run six context-wise B10 graphs and retain all ten target gradients."""

    from easyeditor.models.rome import repr_tools

    batch, identities, order = _ordered_batch(requests)
    if order != ordered_request_digest_v1(identities):
        raise ODEBFContractError("P1R16 target order differs")
    templates, group_sizes, context_sha256 = _locked_context_templates(
        RoutingObjective.TARGET_NEW_NLL, contexts
    )
    if group_sizes != (1, 5) or len(templates) != 6:
        raise ODEBFContractError("P1R16 target contexts differ")
    rendered_templates: list[str] = []
    words: list[str] = []
    for request in batch:
        for template in templates:
            assert template is not None
            rendered_templates.append(template.format(str(request["prompt"])))
            words.append(str(request["subject"]))
    strategy = str(fact_token_strategy)
    if not strategy.startswith("subject_"):
        raise ODEBFContractError("P1R16 writer lookup strategy differs")
    writer_lookup = repr_tools.get_words_idxs_in_templates(
        tokenizer, rendered_templates, words, strategy[len("subject_") :]
    )
    if len(writer_lookup) != PHYSICAL_WRITER_CONTEXT_COUNT:
        raise ODEBFContractError("P1R16 writer lookup count differs")
    writer_by_context = tuple(
        tuple(tuple(int(item) for item in writer_lookup[row * 6 + context]) for row in range(BATCH_SIZE))
        for context in range(6)
    )
    target_by_context = _request_context_raw_positions(lookup_positions)

    device = _model_device(model)
    target_flat = target_state.detach().to(device=device, dtype=torch.float32).contiguous().view(-1).requires_grad_(True)
    target_variable = target_flat.view(target_state.shape)
    before = _model_state(model)
    before_rng = _rng_identity()
    capture = _BatchedWriterKeyCapture(
        model,
        {layer: rewrite_module_template.format(layer) for layer in PHYSICAL_WRITER_LAYER_ORDER},
        writer_by_context,
    )
    overlay = _BatchedTargetOverlay(model, target_layer_name, target_variable, target_by_context)
    llama = _is_llama(model)
    context_values: list[list[torch.Tensor]] = [[] for _ in range(BATCH_SIZE)]
    processed_tokens = 0
    canonical_nonzero_padding = 0
    with _virtual_context(model, cumulative_factors_by_weight):
        with capture, overlay, _temporary_left_padding(tokenizer):
            for context_ordinal, template in enumerate(templates):
                assert template is not None
                prefixes: list[str] = []
                texts: list[str] = []
                prefix_lengths: list[int] = []
                suffixes: list[tuple[int, ...]] = []
                singleton_rows: list[tuple[int, ...]] = []
                for request in batch:
                    prefix, target = _surface(request, "target_new", context_template=template)
                    prefixes.append(prefix)
                    prefix_lengths.append(
                        len(_input_ids(_tokenize_left(tokenizer, [prefix]), label="P1R16 prefix", one_row=True)[0])
                    )
                    suffixes.append(_suffix_tokens(tokenizer, target, llama=llama))
                    text = f"{prefix} {target}"
                    texts.append(text)
                    singleton_rows.append(
                        _input_ids(
                            _tokenize_left(tokenizer, [text]),
                            label="P1R16 singleton full text",
                            one_row=True,
                        )[0]
                    )
                encoded = _move_encoding(
                    _tokenize_left(tokenizer, texts, padding=True, return_tensors="pt"),
                    device,
                )
                input_ids, left_padding = _left_padding_offsets(encoded, rows=BATCH_SIZE)
                if context_ordinal == 0:
                    canonical_nonzero_padding = sum(item > 0 for item in left_padding)
                capture.prepare(context_ordinal, left_padding)
                overlay.prepare(
                    context_ordinal, left_padding, input_ids, singleton_rows
                )
                logits = model(**encoded).logits
                capture.complete_context()
                if not isinstance(logits, torch.Tensor):
                    raise ODEBFContractError("P1R16 target logits differ")
                for row, identity in enumerate(identities):
                    score = _score_suffix(
                        logits=logits,
                        input_ids=input_ids,
                        row=row,
                        prefix_length=prefix_lengths[row],
                        left_padding=left_padding[row],
                        tokens=suffixes[row],
                        llama=llama,
                        request_sha256=identity,
                        ordinal=row,
                        context_ordinal=context_ordinal,
                        context_sha256=context_sha256,
                        target_label="target_new",
                    )
                    context_values[row].append(score.value)
                attention = encoded.get("attention_mask")
                processed_tokens += int(attention.detach().to(device="cpu").sum()) if isinstance(attention, torch.Tensor) else int(input_ids.numel())

            request_losses = tuple(torch.stack(values).mean() for values in context_values)
            gradient, request_hashes, off_request = (
                retain_requestwise_batched_gradients(
                    request_losses, target_flat, target_state.shape
                )
            )

    keys = capture.keys()
    if overlay.current is None or overlay.residual is None:
        raise ODEBFContractError("P1R16 canonical target capture is absent")
    current = overlay.current
    residual = overlay.residual.detach().to(device="cpu", dtype=torch.float32).contiguous()
    expected = (target_state.detach().to(device="cpu", dtype=torch.float32) - current).contiguous()
    if not torch.equal(residual, expected):
        raise ODEBFContractError("P1R16 current residual differs")
    losses = torch.stack([item.detach().to(device="cpu") for item in request_losses]).to(dtype=torch.float64)
    target_step = build_per_request_gdual_target_step(
        target_state=target_state,
        z_base=z_base,
        loss_by_request=losses,
        gradient_by_request=gradient,
        step_index=step_index,
        cumulative_g_path_before=cumulative_g_path_before,
    )
    desired = target_step.desired_target
    demand = build_coupled_physical_demand(
        current_z=current,
        target_state=target_state,
        desired_target=desired,
    )
    if _model_state(model) != before or _rng_identity() != before_rng:
        raise ODEBFStateError("P1R16 target/factor group mutated model or RNG")
    coordinate_identity = canonical_hash(overlay.coordinate_hashes)
    goal = max(float(torch.mean(losses)) - target_step.controller_requested_reduction, 0.0)
    payload = {
        "schema": f"{P1R16_SCHEMA}-target-factor-group",
        "request_order_sha256": order,
        "context_sha256": context_sha256,
        "target_state_sha256": tensor_sha256(target_state),
        "terminal_current_z_sha256": tensor_sha256(current),
        "old_residual_sha256": tensor_sha256(residual),
        "desired_target_sha256": tensor_sha256(desired),
        "coupled_demand_sha256": demand.factor_demand_sha256,
        "loss_by_request": [float(item) for item in losses],
        "gradient_by_request_sha256": tensor_sha256(gradient),
        "request_gradient_sha256": request_hashes,
        "off_request_gradient_nonzero_count": off_request,
        "target_step": target_step.raw_free_payload(),
        "goal_target_new_nll": goal,
        "controller_requested_reduction": target_step.controller_requested_reduction,
        "coordinate_identity_sha256": coordinate_identity,
        "coordinate_exact_count": BATCH_SIZE * 6,
        "coordinate_mismatch_count": 0,
        "left_padding_nonzero_count_canonical": canonical_nonzero_padding,
        "bf16_singleton_batch_exact_gate_count": 0,
        "key_sha256_by_layer": [[layer, tensor_sha256(keys[layer])] for layer in PHYSICAL_WRITER_LAYER_ORDER],
        "layer_hook_call_count": [[layer, len(capture.values[layer])] for layer in PHYSICAL_WRITER_LAYER_ORDER],
        "logical_forward_groups": 1,
        "model_forward_calls": 6,
        "physical_target_graphs": 6,
        "autograd_backend_invocations": 10,
        "backward_calls": 10,
        "processed_token_count": processed_tokens,
        "old_target_access_count": 0,
        "heldout_access_count": 0,
        "native_or_direct_z_access_count": 0,
    }
    receipt = PerRequestTargetFactorGroupReceipt(
        order,
        context_sha256,
        tensor_sha256(target_state),
        tensor_sha256(current),
        tensor_sha256(residual),
        tensor_sha256(desired),
        demand.factor_demand_sha256,
        tuple(float(item) for item in losses),
        tensor_sha256(gradient),
        request_hashes,
        off_request,
        target_step,
        goal,
        target_step.controller_requested_reduction,
        coordinate_identity,
        60,
        0,
        canonical_nonzero_padding,
        0,
        tuple((layer, tensor_sha256(keys[layer])) for layer in PHYSICAL_WRITER_LAYER_ORDER),
        tuple((layer, len(capture.values[layer])) for layer in PHYSICAL_WRITER_LAYER_ORDER),
        1,
        6,
        6,
        10,
        10,
        processed_tokens,
        0,
        0,
        0,
        canonical_hash(payload),
    )
    return keys, current, residual, desired, demand.factor_demand, demand, receipt


def capture_canonical_b10_z_base(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    target_layer_name: str,
    canonical_lookup_positions: Sequence[int],
    canonical_context_template: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Capture the authoritative canonical B10 theta=0 target coordinates."""

    batch, identities, order = _ordered_batch(requests)
    if len(canonical_lookup_positions) != BATCH_SIZE:
        raise ODEBFContractError("P1R16 canonical z-base lookup differs")
    if (
        not isinstance(canonical_context_template, str)
        or canonical_context_template.count("{}") != 1
    ):
        raise ODEBFContractError("P1R16 canonical context template differs")
    prefixes = [
        _surface(
            request,
            "target_new",
            context_template=canonical_context_template,
        )[0]
        for request in batch
    ]
    device = _model_device(model)
    before = _model_state(model)
    before_rng = _rng_identity()
    captured: list[torch.Tensor] = []
    handle = None
    with _temporary_left_padding(tokenizer):
        encoded = _move_encoding(
            _tokenize_left(tokenizer, prefixes, padding=True, return_tensors="pt"),
            device,
        )
        input_ids, left_padding = _left_padding_offsets(encoded, rows=BATCH_SIZE)
        singleton_rows = tuple(
            _input_ids(
                _tokenize_left(tokenizer, [prefix]),
                label="P1R16 canonical singleton",
                one_row=True,
            )[0]
            for prefix in prefixes
        )
        padded, token_hashes = padded_subject_positions(
            canonical_lookup_positions,
            left_padding,
            input_ids,
            singleton_rows,
        )

        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            activation = _unwrap_layer_output(output)
            if activation.shape[0] != BATCH_SIZE:
                if activation.ndim == 3 and activation.shape[1] == BATCH_SIZE:
                    activation = activation.transpose(0, 1)
                else:
                    raise ODEBFContractError("P1R16 canonical z-base layout differs")
            captured.append(torch.stack([activation[row, pos, :] for row, pos in enumerate(padded)]).detach())

        handle = model.get_submodule(target_layer_name).register_forward_hook(hook)
        try:
            model(**encoded)
        finally:
            handle.remove()
    if len(captured) != 1:
        raise ODEBFContractError("P1R16 canonical z-base capture count differs")
    z_base = captured[0].to(device="cpu", dtype=torch.float32).T.contiguous()
    if _model_state(model) != before or _rng_identity() != before_rng:
        raise ODEBFStateError("P1R16 canonical z-base capture mutated state")
    payload = {
        "schema": f"{P1R16_SCHEMA}-canonical-b10-z-base",
        "request_order_sha256": order,
        "request_sha256": list(identities),
        "raw_positions": [int(item) for item in canonical_lookup_positions],
        "left_padding": list(left_padding),
        "padded_positions": list(padded),
        "coordinate_token_identity_sha256": canonical_hash(token_hashes),
        "coordinate_exact_count": BATCH_SIZE,
        "coordinate_mismatch_count": 0,
        "bf16_singleton_batch_exact_gate_count": 0,
        "bf16_numeric_parity_status": "OBSERVATION_ONLY_NOT_REQUIRED",
        "z_base_sha256": tensor_sha256(z_base),
        "model_forward_calls": 1,
        "native_or_direct_z_access_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return z_base, payload


__all__ = [
    "CoupledDemandReceipt",
    "PerRequestTargetFactorGroupReceipt",
    "build_coupled_physical_demand",
    "aggregate_context_batched_keys",
    "capture_canonical_b10_z_base",
    "padded_subject_positions",
    "perrequest_dynamic_target_factor_group",
    "retain_requestwise_batched_gradients",
]
