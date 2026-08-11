"""Accepted-state runtime primitives for the P1R22 atomic optimization.

The module contains no routing policy.  It only changes how an already chosen
P1R19 state is represented and how source-identical controller rows are
batched/captured.  Scientific callers remain responsible for field algebra,
target dynamics, routing, and terminal evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import WaypointFactor, assemble_effective_bf16, tensor_sha256
from .request_digest import ordered_request_digest_v1


P1R22_INSTRUCTION_ID = "ODEEDIT-S05-ODE-BF-ATOMIC-RUNTIME-OPTIMIZATION-P1R22-V1"
P1R22_METHOD_ID = "P1R19_STRENGTH_PRESERVING_ATOMIC_ACCEPTED_STATE_OPTIMIZED_V1"
P1R22_LAYER_ORDER = (4, 5, 6, 7, 8)
P1R22_CONTEXTS_PER_REQUEST = 6


@dataclass(frozen=True, slots=True)
class PhysicalStateCapture:
    """Five keys and the canonical terminal lookup from one partial forward."""

    keys_by_layer: Mapping[int, torch.Tensor]
    terminal_z: torch.Tensor
    lookup_positions: tuple[int, ...]
    request_order_sha256: str
    model_state_sha256: str
    physical_forward_count: int
    processed_token_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r22-physical-state-capture/v1",
            "layers": list(P1R22_LAYER_ORDER),
            "key_sha256": {
                str(layer): tensor_sha256(self.keys_by_layer[layer])
                for layer in P1R22_LAYER_ORDER
            },
            "terminal_z_sha256": tensor_sha256(self.terminal_z),
            "lookup_position_sha256": canonical_hash(list(self.lookup_positions)),
            "lookup_position_count": len(self.lookup_positions),
            "request_order_sha256": self.request_order_sha256,
            "model_state_sha256": self.model_state_sha256,
            "logical_key_capture_group_count": 1,
            "physical_forward_count": self.physical_forward_count,
            "processed_token_count": self.processed_token_count,
            "stopped_after_deepest_writer_layer": True,
            "identity_sha256": self.identity_sha256,
        }


def _unwrap(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise ODEBFContractError("P1R22 traced activation differs")


def _batch_axis_zero(value: torch.Tensor, rows: int) -> torch.Tensor:
    if value.ndim != 3:
        raise ODEBFContractError("P1R22 traced activation rank differs")
    if value.shape[0] == rows:
        return value
    if value.shape[1] == rows:
        return value.transpose(0, 1)
    raise ODEBFContractError("P1R22 traced activation batch axis differs")


def _select_rows(
    value: torch.Tensor,
    indices: Sequence[Sequence[int]],
) -> torch.Tensor:
    value = _batch_axis_zero(value, len(indices))
    selected: list[torch.Tensor] = []
    for row, positions in enumerate(indices):
        if len(positions) != 1:
            raise ODEBFContractError("P1R22 lookup multiplicity differs")
        position = int(positions[0])
        if position < 0:
            position += value.shape[1]
        if position < 0 or position >= value.shape[1]:
            raise ODEBFContractError("P1R22 lookup position is out of range")
        selected.append(value[row, position, :])
    return torch.stack(selected)


def capture_physical_state(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    contexts: Sequence[Sequence[str]],
) -> PhysicalStateCapture:
    """Capture all writer keys and canonical terminal z in one partial pass."""

    from easyeditor.models.rome import repr_tools
    from easyeditor.util import nethook

    batch = tuple(requests)
    if len(batch) != BATCH_SIZE:
        raise ODEBFContractError("P1R22 state capture requires joint B10")
    groups = tuple(tuple(group) for group in contexts)
    if tuple(len(group) for group in groups) != (1, 5):
        raise ODEBFContractError("P1R22 state capture contexts differ")
    layers = tuple(int(layer) for layer in hparams.layers)
    if layers != P1R22_LAYER_ORDER:
        raise ODEBFContractError("P1R22 state capture layer order differs")
    templates = [
        context.format(str(request["prompt"]))
        for request in batch
        for group in groups
        for context in group
    ]
    words = [
        str(request["subject"])
        for request in batch
        for group in groups
        for _ in group
    ]
    subtoken = str(hparams.fact_token)
    if not subtoken.startswith("subject_"):
        raise ODEBFContractError("P1R22 fact-token strategy differs")
    indices = repr_tools.get_words_idxs_in_templates(
        tokenizer,
        templates,
        words,
        subtoken[len("subject_") :],
    )
    rendered = [template.format(word) for template, word in zip(templates, words, strict=True)]
    original_padding = tokenizer.padding_side
    if original_padding not in ("left", "right"):
        raise ODEBFContractError("P1R22 source tokenizer padding differs")
    encoded = tokenizer(rendered, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    module_names = [hparams.rewrite_module_tmp.format(layer) for layer in layers]
    with torch.no_grad():
        with nethook.TraceDict(
            model,
            module_names,
            retain_input=True,
            retain_output=True,
            stop=True,
        ) as traces:
            model(**encoded)
    if tokenizer.padding_side != original_padding:
        raise ODEBFStateError("P1R22 state capture changed tokenizer padding")
    keys_by_layer: dict[int, torch.Tensor] = {}
    rows_per_request = sum(len(group) for group in groups)
    for layer, module_name in zip(layers, module_names, strict=True):
        raw = _select_rows(_unwrap(traces[module_name].input), indices)
        reduced: list[torch.Tensor] = []
        for request_ordinal in range(BATCH_SIZE):
            start = request_ordinal * rows_per_request
            group_values: list[torch.Tensor] = []
            offset = start
            for group in groups:
                stop = offset + len(group)
                group_values.append(raw[offset:stop].mean(dim=0))
                offset = stop
            reduced.append(torch.stack(group_values).mean(dim=0))
        keys_by_layer[layer] = (
            torch.stack(reduced)
            .T.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
    deepest = _select_rows(
        _unwrap(traces[module_names[-1]].output), indices
    )
    canonical_rows = tuple(
        request_ordinal * rows_per_request
        for request_ordinal in range(BATCH_SIZE)
    )
    terminal_z = (
        deepest[list(canonical_rows)]
        .T.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
    )
    processed = int(encoded["attention_mask"].detach().cpu().sum().item())
    if (
        terminal_z.ndim != 2
        or terminal_z.shape[1] != BATCH_SIZE
        or any(
            keys_by_layer[layer].ndim != 2
            or keys_by_layer[layer].shape[1] != BATCH_SIZE
            for layer in layers
        )
        or not torch.isfinite(terminal_z).all()
        or any(not torch.isfinite(keys_by_layer[layer]).all() for layer in layers)
    ):
        raise ODEBFContractError("P1R22 captured physical state differs")
    order = ordered_request_digest_v1(
        [str(request["request_sha256"]) for request in batch]
    )
    # Tensor version counters advance on an authoritative physical copy even
    # when exact W0 is restored between arms.  Full-model byte hashing here is
    # also forbidden: accepted edited-weight bytes are already bound by the
    # one-per-transition materialization receipt.  This capture identity binds
    # the stable parameter inventory/pointers and is linked to those hashes by
    # the transition receipt.
    state_payload = {
        "parameters": [
            (
                name,
                int(parameter.data_ptr()),
                str(parameter.dtype),
                list(parameter.shape),
                bool(parameter.requires_grad),
            )
            for name, parameter in model.named_parameters()
        ]
    }
    payload = {
        "layers": list(layers),
        "key_sha256": [
            (layer, tensor_sha256(keys_by_layer[layer])) for layer in layers
        ],
        "terminal_z_sha256": tensor_sha256(terminal_z),
        "lookup_positions": [int(item[0]) for item in indices],
        "request_order_sha256": order,
        "model_state_sha256": canonical_hash(state_payload),
        "physical_forward_count": 1,
        "processed_token_count": processed,
    }
    return PhysicalStateCapture(
        keys_by_layer,
        terminal_z,
        tuple(int(item[0]) for item in indices),
        order,
        payload["model_state_sha256"],
        1,
        processed,
        canonical_hash(payload),
    )


class AcceptedPhysicalStateMaterializer:
    """Materialize BF16 states from W0 plus the complete factor inventory."""

    def __init__(
        self,
        model: torch.nn.Module,
        entry_values: Mapping[str, torch.Tensor],
        *,
        row_block: int = 64,
    ) -> None:
        parameters = dict(model.named_parameters())
        if set(entry_values) - set(parameters):
            raise ODEBFContractError("P1R22 entry parameter is absent")
        self._parameters = {name: parameters[name] for name in entry_values}
        self._entry = {
            name: value.detach().to(
                device=parameters[name].device,
                dtype=torch.bfloat16,
            ).clone()
            for name, value in entry_values.items()
        }
        self._pointers = {
            name: parameter.data_ptr() for name, parameter in self._parameters.items()
        }
        self.row_block = row_block
        self.dense_assembly_count = 0
        self.full_weight_hash_count = 0
        self.hot_hook_dense_assembly_count = 0
        self.hot_hook_full_weight_hash_count = 0
        self.transition_receipts: list[dict[str, Any]] = []

    def materialize(
        self,
        factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
        *,
        transition_index: int,
    ) -> dict[str, Any]:
        if set(factors_by_weight) != set(self._parameters):
            raise ODEBFContractError("P1R22 cumulative factor inventory differs")
        hashes: dict[str, str] = {}
        factor_counts: dict[str, int] = {}
        step_energy: dict[str, float] = {}
        cumulative_energy: dict[str, float] = {}
        with torch.no_grad():
            for name in sorted(self._parameters):
                parameter = self._parameters[name]
                if parameter.data_ptr() != self._pointers[name]:
                    raise ODEBFStateError("P1R22 materialization pointer differs")
                effective, stats = assemble_effective_bf16(
                    self._entry[name],
                    factors_by_weight[name],
                    row_block=self.row_block,
                )
                step_total = 0.0
                cumulative_total = 0.0
                for start in range(0, parameter.shape[0], self.row_block):
                    end = min(start + self.row_block, parameter.shape[0])
                    step = effective[start:end].float() - parameter[start:end].float()
                    cumulative = (
                        effective[start:end].float()
                        - self._entry[name][start:end].float()
                    )
                    step_total += float(torch.sum(step.double().square()))
                    cumulative_total += float(
                        torch.sum(cumulative.double().square())
                    )
                    del step, cumulative
                parameter.copy_(effective)
                if not torch.equal(parameter, effective):
                    raise ODEBFStateError("P1R22 materialized parameter hash differs")
                hashes[name] = stats.effective_bf16_sha256
                factor_counts[name] = stats.factor_count
                step_energy[name] = step_total
                cumulative_energy[name] = cumulative_total
                self.dense_assembly_count += 1
                self.full_weight_hash_count += 1
                del effective
        payload = {
            "schema": "ode-edit-s05-p1r22-accepted-materialization/v1",
            "transition_index": transition_index,
            "weight_count": len(hashes),
            "effective_bf16_sha256": hashes,
            "factor_count": factor_counts,
            "realized_bf16_step_energy": step_energy,
            "cumulative_bf16_capacity": cumulative_energy,
            "entry_relative_rounding": True,
            "incremental_bf16_accumulation_count": 0,
            "hot_hook_dense_assembly_count": 0,
            "hot_hook_full_weight_hash_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        self.transition_receipts.append(payload)
        return payload

    def restore(self) -> dict[str, Any]:
        hashes: dict[str, str] = {}
        with torch.no_grad():
            for name in sorted(self._parameters):
                parameter = self._parameters[name]
                if parameter.data_ptr() != self._pointers[name]:
                    raise ODEBFStateError("P1R22 restore pointer differs")
                parameter.copy_(self._entry[name])
                hashes[name] = tensor_sha256(parameter)
                if hashes[name] != tensor_sha256(self._entry[name]):
                    raise ODEBFStateError("P1R22 W0 restore differs")
        payload = {
            "schema": "ode-edit-s05-p1r22-w0-restore/v1",
            "parameter_sha256": hashes,
            "pointer_identity_preserved": True,
            "persistent_commit_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "ode-edit-s05-p1r22-materializer-accounting/v1",
            "dense_assembly_count": self.dense_assembly_count,
            "full_weight_hash_count": self.full_weight_hash_count,
            "hot_hook_dense_assembly_count": self.hot_hook_dense_assembly_count,
            "hot_hook_full_weight_hash_count": self.hot_hook_full_weight_hash_count,
            "transition_count": len(self.transition_receipts),
            "transition_receipt_sha256": [
                receipt["identity_sha256"] for receipt in self.transition_receipts
            ],
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


__all__ = [
    "AcceptedPhysicalStateMaterializer",
    "P1R22_INSTRUCTION_ID",
    "P1R22_LAYER_ORDER",
    "P1R22_METHOD_ID",
    "PhysicalStateCapture",
    "capture_physical_state",
]
