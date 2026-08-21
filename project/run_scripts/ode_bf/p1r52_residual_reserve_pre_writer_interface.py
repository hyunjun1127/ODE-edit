"""Typed IL1 pre-writer bridge for future P1R52 writer integrations.

This module only exposes the already selected native P1R52 target at the
arm-specific physical writer entry.  It does not run a writer, route, model,
evaluator, materializer, or history operation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1r24_atomic_strength import P1R24TargetStep
from .p1r51_requestwise_semantic_allocation import P1R51ControllerState
from .p1r52_r42_safe_kdc import P1R52SelectedTarget
from .p1r52_target_depth import P1R52TargetDepth, P1R52TargetDepthOuter
from .scalable_batched_model import ScalableObjectiveResult
from .scalable_batched_runtime import scalable_ordered_request_digest


P1R52_PRE_WRITER_INTERFACE_STATUS = "P1R52_IL1_PRE_WRITER_INPUT_SEALED"
P1R52_PRE_WRITER_OBSERVATION_STATUS = "OBSERVATION_ONLY_NO_ACTION"
WRITER_RELEVANT_TENSOR_NAMES = (
    "target_next",
    "target_displacement",
    "required_displacement",
    "write_velocity",
    "nll_gradient",
    "combined_gradient",
)


@dataclass(frozen=True, slots=True)
class PreWriterTensorIdentity:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    sha256: str
    pointer: int
    version: int
    requires_grad: bool
    layout: str
    stride: tuple[int, ...]
    storage_offset: int

    def raw_free_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "sha256": self.sha256,
            "pointer": self.pointer,
            "version": self.version,
            "requires_grad": self.requires_grad,
            "layout": self.layout,
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
        }


@dataclass(frozen=True, slots=True)
class PreWriterWeightIdentity:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    sha256: str
    pointer: int
    version: int
    requires_grad: bool
    parameter_object_identity: int
    layout: str
    stride: tuple[int, ...]
    storage_offset: int

    def scientific_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "sha256": self.sha256,
            "requires_grad": self.requires_grad,
            "layout": self.layout,
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
        }

    def raw_free_payload(self) -> dict[str, object]:
        return {
            **self.scientific_payload(),
            "pointer": self.pointer,
            "version": self.version,
            "parameter_object_identity": self.parameter_object_identity,
        }


@dataclass(frozen=True, slots=True)
class P1R52IL1NativeSelectedBridgeReceipt:
    status: str
    step_index: int
    target_depth: str
    configured_inner_count: int
    executed_inner_count: int
    native_selected_receipt_identity: str
    native_target_step_receipt_identity: str
    outer_reassembly_receipt_identity: str
    native_tensor_identities: tuple[PreWriterTensorIdentity, ...]
    outer_tensor_identities: tuple[PreWriterTensorIdentity, ...]
    exact_six_tensor_match_count: int
    selected_endpoint_identity: str
    selected_endpoint_guard_identity: str
    endpoint_identity_match: bool
    next_state_identity: str
    next_state_match: bool
    native_component_reference_count: int
    selected_target_recompute_count: int
    target_controller_influence_count: int

    def raw_free_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "step_index": self.step_index,
            "target_depth": self.target_depth,
            "configured_inner_count": self.configured_inner_count,
            "executed_inner_count": self.executed_inner_count,
            "native_selected_receipt_identity": (
                self.native_selected_receipt_identity
            ),
            "native_target_step_receipt_identity": (
                self.native_target_step_receipt_identity
            ),
            "outer_reassembly_receipt_identity": (
                self.outer_reassembly_receipt_identity
            ),
            "native_tensor_identities": [
                item.raw_free_payload() for item in self.native_tensor_identities
            ],
            "outer_tensor_identities": [
                item.raw_free_payload() for item in self.outer_tensor_identities
            ],
            "exact_six_tensor_match_count": self.exact_six_tensor_match_count,
            "selected_endpoint_identity": self.selected_endpoint_identity,
            "selected_endpoint_guard_identity": (
                self.selected_endpoint_guard_identity
            ),
            "endpoint_identity_match": self.endpoint_identity_match,
            "next_state_identity": self.next_state_identity,
            "next_state_match": self.next_state_match,
            "native_component_reference_count": (
                self.native_component_reference_count
            ),
            "selected_target_recompute_count": self.selected_target_recompute_count,
            "target_controller_influence_count": (
                self.target_controller_influence_count
            ),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return str(self.raw_free_payload()["identity_sha256"])


@dataclass(frozen=True, slots=True)
class P1R52PreWriterInputReceipt:
    status: str
    step_index: int
    request_identities: tuple[str, ...]
    request_order_sha256: str
    native_selected_bridge_identity: str
    native_selected_receipt_identity: str
    outer_reassembly_receipt_identity: str
    entry_weight_key_inventory: tuple[str, ...]
    touched_mapping_object_identity: int
    entry_weight_identities: tuple[PreWriterWeightIdentity, ...]
    entry_weight_scientific_identity: str
    entry_weight_pointer_identity: str
    action_freeze_identity: str
    target_recompute_count: int
    model_forward_count: int
    model_backward_count: int
    writer_action_count: int
    materialization_count: int
    heldout_evaluator_count: int
    target_controller_influence_count: int

    def raw_free_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "step_index": self.step_index,
            "request_identities": list(self.request_identities),
            "request_order_sha256": self.request_order_sha256,
            "native_selected_bridge_identity": (
                self.native_selected_bridge_identity
            ),
            "native_selected_receipt_identity": (
                self.native_selected_receipt_identity
            ),
            "outer_reassembly_receipt_identity": (
                self.outer_reassembly_receipt_identity
            ),
            "entry_weight_key_inventory": list(self.entry_weight_key_inventory),
            "touched_mapping_object_identity": (
                self.touched_mapping_object_identity
            ),
            "entry_weight_identities": [
                item.raw_free_payload() for item in self.entry_weight_identities
            ],
            "entry_weight_scientific_identity": (
                self.entry_weight_scientific_identity
            ),
            "entry_weight_pointer_identity": self.entry_weight_pointer_identity,
            "action_freeze_identity": self.action_freeze_identity,
            "target_recompute_count": self.target_recompute_count,
            "model_forward_count": self.model_forward_count,
            "model_backward_count": self.model_backward_count,
            "writer_action_count": self.writer_action_count,
            "materialization_count": self.materialization_count,
            "heldout_evaluator_count": self.heldout_evaluator_count,
            "target_controller_influence_count": (
                self.target_controller_influence_count
            ),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return str(self.raw_free_payload()["identity_sha256"])


@dataclass(frozen=True, slots=True)
class P1R52PreWriterInput:
    selected_target: P1R52SelectedTarget
    bridge_receipt: P1R52IL1NativeSelectedBridgeReceipt
    receipt: P1R52PreWriterInputReceipt


@dataclass(frozen=True, slots=True)
class P1R52PreWriterObservationReceipt:
    status: str
    observer_id: str
    pre_writer_input_identity: str
    model_forward_count: int
    model_backward_count: int
    writer_action_count: int
    materialization_count: int
    heldout_evaluator_count: int
    history_consume_count: int
    history_append_count: int
    target_controller_influence_count: int

    @classmethod
    def no_action(
        cls,
        pre_writer_input: P1R52PreWriterInput,
        *,
        observer_id: str,
    ) -> "P1R52PreWriterObservationReceipt":
        if not observer_id:
            raise ODEBFContractError("P1R52 pre-writer observer identity differs")
        return cls(
            P1R52_PRE_WRITER_OBSERVATION_STATUS,
            observer_id,
            pre_writer_input.receipt.identity_sha256,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def raw_free_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "observer_id": self.observer_id,
            "pre_writer_input_identity": self.pre_writer_input_identity,
            "model_forward_count": self.model_forward_count,
            "model_backward_count": self.model_backward_count,
            "writer_action_count": self.writer_action_count,
            "materialization_count": self.materialization_count,
            "heldout_evaluator_count": self.heldout_evaluator_count,
            "history_consume_count": self.history_consume_count,
            "history_append_count": self.history_append_count,
            "target_controller_influence_count": (
                self.target_controller_influence_count
            ),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return str(self.raw_free_payload()["identity_sha256"])


class P1R52PreWriterObserver(ABC):
    """Typed no-action hook; writer takeover is intentionally not exposed."""

    @abstractmethod
    def observe(
        self, pre_writer_input: P1R52PreWriterInput
    ) -> P1R52PreWriterObservationReceipt:
        raise NotImplementedError


@dataclass(slots=True)
class _WeightRollback:
    name: str
    parameter: torch.nn.Parameter
    data_reference: torch.Tensor
    value: torch.Tensor
    pointer: int
    sha256: str
    version: int
    requires_grad: bool
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    stride: tuple[int, ...]
    storage_offset: int


@dataclass(slots=True)
class _TouchedMappingRollback:
    mapping: MutableMapping[str, torch.nn.Parameter]
    object_identity: int
    items: tuple[tuple[str, torch.nn.Parameter], ...]
    key_inventory: tuple[str, ...]


@dataclass(slots=True)
class _TargetTensorRollback:
    tensor: torch.Tensor
    data_reference: torch.Tensor
    value: torch.Tensor
    pointer: int
    sha256: str
    version: int
    requires_grad: bool
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    stride: tuple[int, ...]
    storage_offset: int


def _mapping_identity(value: Mapping[str, object], *, name: str) -> str:
    payload = dict(value)
    claimed = payload.pop("identity_sha256", None)
    if not isinstance(claimed, str) or canonical_hash(payload) != claimed:
        raise ODEBFStateError(f"P1R52 pre-writer {name} receipt identity differs")
    return claimed


def _tensor_identity(name: str, value: torch.Tensor) -> PreWriterTensorIdentity:
    if (
        not isinstance(value, torch.Tensor)
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise ODEBFContractError(f"P1R52 pre-writer tensor {name} differs")
    return PreWriterTensorIdentity(
        name,
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
        tensor_sha256(value),
        int(value.data_ptr()),
        int(value._version),
        bool(value.requires_grad),
        str(value.layout),
        tuple(value.stride()),
        int(value.storage_offset()),
    )


def _controller_identity(state: P1R51ControllerState) -> str:
    if not isinstance(state, P1R51ControllerState):
        raise ODEBFContractError("P1R52 pre-writer controller state differs")
    return canonical_hash(
        {
            "entry_semantic_gradient_norm": (
                None
                if state.entry_semantic_gradient_norm is None
                else list(state.entry_semantic_gradient_norm)
            ),
            "cumulative_accepted_activation_path": list(
                state.cumulative_accepted_activation_path
            ),
        }
    )


def _endpoint_guard(endpoint: ScalableObjectiveResult) -> str:
    if not isinstance(endpoint, ScalableObjectiveResult):
        raise ODEBFContractError("P1R52 pre-writer selected endpoint differs")
    return canonical_hash(endpoint.raw_free_payload())


def reconstruct_native_il1_selected_target(
    outer: P1R52TargetDepthOuter,
    *,
    step_index: int,
) -> tuple[P1R52SelectedTarget, P1R52IL1NativeSelectedBridgeReceipt]:
    """Recover the native selected wrapper from the preserved IL1 components."""

    if (
        not isinstance(outer, P1R52TargetDepthOuter)
        or isinstance(step_index, bool)
        or step_index < 0
        or step_index >= 8
        or len(outer.inner_steps) != 1
    ):
        raise ODEBFContractError("P1R52 pre-writer requires exact IL1 outer")
    receipt = dict(outer.receipt)
    if (
        receipt.get("depth_policy") != P1R52TargetDepth.IL1.value
        or receipt.get("configured_inner_count") != 1
        or receipt.get("executed_inner_count") != 1
        or receipt.get("outer_step_index") != step_index
        or outer.target_step.receipt is not outer.receipt
    ):
        raise ODEBFStateError("P1R52 IL1 outer reassembly binding differs")
    inner = outer.inner_steps[-1]
    if inner.inner_index != 0 or inner.global_target_update_ordinal != step_index:
        raise ODEBFStateError("P1R52 IL1 native selected ordinal differs")
    if not isinstance(inner.target_step, P1R24TargetStep):
        raise ODEBFContractError("P1R52 IL1 native target step type differs")
    native_identity = _mapping_identity(
        inner.target_step.receipt, name="native-target-step"
    )
    outer_identity = _mapping_identity(outer.receipt, name="outer-reassembly")
    native_tensors = tuple(
        _tensor_identity(name, getattr(inner.target_step, name))
        for name in WRITER_RELEVANT_TENSOR_NAMES
    )
    outer_tensors = tuple(
        _tensor_identity(name, getattr(outer.target_step, name))
        for name in WRITER_RELEVANT_TENSOR_NAMES
    )
    matches = tuple(
        left.name == right.name
        and left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and left.sha256 == right.sha256
        and torch.equal(
            getattr(inner.target_step, left.name),
            getattr(outer.target_step, right.name),
        )
        for left, right in zip(native_tensors, outer_tensors, strict=True)
    )
    if not all(matches):
        raise ODEBFStateError("P1R52 IL1 native/outer writer tensor differs")
    endpoint_identity = inner.selected_endpoint.identity_sha256
    endpoint_guard = _endpoint_guard(inner.selected_endpoint)
    endpoint_match = (
        outer.selected_endpoint is inner.selected_endpoint
        and outer.selected_endpoint.identity_sha256 == endpoint_identity
        and _endpoint_guard(outer.selected_endpoint) == endpoint_guard
    )
    next_state_identity = _controller_identity(inner.next_state)
    next_state_match = (
        outer.next_state is inner.next_state
        and _controller_identity(outer.next_state) == next_state_identity
    )
    if not endpoint_match or not next_state_match:
        raise ODEBFStateError("P1R52 IL1 endpoint/controller bridge differs")
    selected = P1R52SelectedTarget(
        inner.target_step,
        inner.selected_endpoint,
        inner.next_state,
        inner.target_step.receipt,
    )
    bridge = P1R52IL1NativeSelectedBridgeReceipt(
        P1R52_PRE_WRITER_INTERFACE_STATUS,
        step_index,
        P1R52TargetDepth.IL1.value,
        1,
        1,
        native_identity,
        native_identity,
        outer_identity,
        native_tensors,
        outer_tensors,
        sum(matches),
        endpoint_identity,
        endpoint_guard,
        endpoint_match,
        next_state_identity,
        next_state_match,
        4,
        0,
        0,
    )
    return selected, bridge


def _capture_weights(
    parameters: MutableMapping[str, torch.nn.Parameter],
) -> tuple[
    tuple[PreWriterWeightIdentity, ...],
    tuple[_WeightRollback, ...],
    _TouchedMappingRollback,
]:
    if not parameters:
        raise ODEBFContractError("P1R52 pre-writer touched weights are absent")
    identities: list[PreWriterWeightIdentity] = []
    rollbacks: list[_WeightRollback] = []
    for name, parameter in sorted(parameters.items()):
        if not isinstance(parameter, torch.nn.Parameter) or not bool(
            torch.isfinite(parameter).all()
        ):
            raise ODEBFContractError("P1R52 pre-writer parameter differs")
        sha256 = tensor_sha256(parameter)
        identities.append(
            PreWriterWeightIdentity(
                name,
                tuple(parameter.shape),
                str(parameter.dtype),
                str(parameter.device),
                sha256,
                int(parameter.data_ptr()),
                int(parameter._version),
                bool(parameter.requires_grad),
                id(parameter),
                str(parameter.layout),
                tuple(parameter.stride()),
                int(parameter.storage_offset()),
            )
        )
        rollbacks.append(
            _WeightRollback(
                name,
                parameter,
                parameter.data,
                parameter.detach().clone(),
                int(parameter.data_ptr()),
                sha256,
                int(parameter._version),
                bool(parameter.requires_grad),
                tuple(parameter.shape),
                parameter.dtype,
                parameter.device,
                parameter.layout,
                tuple(parameter.stride()),
                int(parameter.storage_offset()),
            )
        )
    items = tuple(parameters.items())
    return (
        tuple(identities),
        tuple(rollbacks),
        _TouchedMappingRollback(
            parameters,
            id(parameters),
            items,
            tuple(sorted(parameters)),
        ),
    )


def _capture_target_tensors(
    selected: P1R52SelectedTarget,
) -> tuple[_TargetTensorRollback, ...]:
    return tuple(
        _TargetTensorRollback(
            tensor=value,
            data_reference=value.data,
            value=value.detach().clone(),
            pointer=int(value.data_ptr()),
            sha256=tensor_sha256(value),
            version=int(value._version),
            requires_grad=bool(value.requires_grad),
            shape=tuple(value.shape),
            dtype=value.dtype,
            device=value.device,
            layout=value.layout,
            stride=tuple(value.stride()),
            storage_offset=int(value.storage_offset()),
        )
        for value in (
            getattr(selected.target_step, name)
            for name in WRITER_RELEVANT_TENSOR_NAMES
        )
    )


def _weight_matches(item: _WeightRollback) -> bool:
    parameter = item.parameter
    return (
        int(parameter.data_ptr()) == item.pointer
        and tensor_sha256(parameter) == item.sha256
        and int(parameter._version) == item.version
        and bool(parameter.requires_grad) == item.requires_grad
        and tuple(parameter.shape) == item.shape
        and parameter.dtype == item.dtype
        and parameter.device == item.device
        and parameter.layout == item.layout
        and tuple(parameter.stride()) == item.stride
        and int(parameter.storage_offset()) == item.storage_offset
    )


def _target_matches(item: _TargetTensorRollback) -> bool:
    tensor = item.tensor
    return (
        int(tensor.data_ptr()) == item.pointer
        and tensor_sha256(tensor) == item.sha256
        and int(tensor._version) == item.version
        and bool(tensor.requires_grad) == item.requires_grad
        and tuple(tensor.shape) == item.shape
        and tensor.dtype == item.dtype
        and tensor.device == item.device
        and tensor.layout == item.layout
        and tuple(tensor.stride()) == item.stride
        and int(tensor.storage_offset()) == item.storage_offset
    )


def _mapping_matches(
    mapping: _TouchedMappingRollback,
    weights: tuple[_WeightRollback, ...],
) -> bool:
    return (
        id(mapping.mapping) == mapping.object_identity
        and tuple(sorted(mapping.mapping)) == mapping.key_inventory
        and len(mapping.mapping) == len(mapping.items)
        and all(
            mapping.mapping.get(item.name) is item.parameter for item in weights
        )
    )


def _restore_entry(
    mapping: _TouchedMappingRollback,
    weights: tuple[_WeightRollback, ...],
    targets: tuple[_TargetTensorRollback, ...],
) -> None:
    with torch.no_grad():
        for item in weights:
            if (
                int(item.parameter.data_ptr()) != item.pointer
                or tuple(item.parameter.shape) != item.shape
                or item.parameter.dtype != item.dtype
                or item.parameter.device != item.device
                or item.parameter.layout != item.layout
                or tuple(item.parameter.stride()) != item.stride
                or int(item.parameter.storage_offset()) != item.storage_offset
            ):
                item.parameter.data = item.data_reference
            if tensor_sha256(item.parameter) != item.sha256:
                item.parameter.data.copy_(item.value)
            if bool(item.parameter.requires_grad) != item.requires_grad:
                item.parameter.requires_grad_(item.requires_grad)
        for item in targets:
            if (
                int(item.tensor.data_ptr()) != item.pointer
                or tuple(item.tensor.shape) != item.shape
                or item.tensor.dtype != item.dtype
                or item.tensor.device != item.device
                or item.tensor.layout != item.layout
                or tuple(item.tensor.stride()) != item.stride
                or int(item.tensor.storage_offset()) != item.storage_offset
            ):
                item.tensor.data = item.data_reference
            if tensor_sha256(item.tensor) != item.sha256:
                item.tensor.copy_(item.value)
            if bool(item.tensor.requires_grad) != item.requires_grad:
                item.tensor.requires_grad_(item.requires_grad)
        mapping.mapping.clear()
        for name, parameter in mapping.items:
            mapping.mapping[name] = parameter
    if any(
        int(item.parameter.data_ptr()) != item.pointer
        or tensor_sha256(item.parameter) != item.sha256
        or bool(item.parameter.requires_grad) != item.requires_grad
        or tuple(item.parameter.shape) != item.shape
        or item.parameter.dtype != item.dtype
        or item.parameter.device != item.device
        or item.parameter.layout != item.layout
        or tuple(item.parameter.stride()) != item.stride
        or int(item.parameter.storage_offset()) != item.storage_offset
        for item in weights
    ) or not _mapping_matches(mapping, weights):
        raise ODEBFStateError("P1R52 pre-writer W rollback differs")
    if any(
        int(item.tensor.data_ptr()) != item.pointer
        or tensor_sha256(item.tensor) != item.sha256
        or bool(item.tensor.requires_grad) != item.requires_grad
        or tuple(item.tensor.shape) != item.shape
        or item.tensor.dtype != item.dtype
        or item.tensor.device != item.device
        or item.tensor.layout != item.layout
        or tuple(item.tensor.stride()) != item.stride
        or int(item.tensor.storage_offset()) != item.storage_offset
        for item in targets
    ):
        raise ODEBFStateError("P1R52 pre-writer target rollback differs")


def _validate_entry_unchanged(
    mapping: _TouchedMappingRollback,
    weights: tuple[_WeightRollback, ...],
    targets: tuple[_TargetTensorRollback, ...],
) -> None:
    if not _mapping_matches(mapping, weights) or any(
        not _weight_matches(item) for item in weights
    ):
        raise ODEBFStateError("P1R52 pre-writer hook mutated physical W")
    if any(not _target_matches(item) for item in targets):
        raise ODEBFStateError("P1R52 pre-writer hook mutated selected target")


def _validate_observation_receipt(
    observation: P1R52PreWriterObservationReceipt,
    pre_writer_input: P1R52PreWriterInput,
) -> None:
    if (
        not isinstance(observation, P1R52PreWriterObservationReceipt)
        or observation.status != P1R52_PRE_WRITER_OBSERVATION_STATUS
        or not observation.observer_id
        or observation.pre_writer_input_identity
        != pre_writer_input.receipt.identity_sha256
        or any(
            int(getattr(observation, field)) != 0
            for field in (
                "model_forward_count",
                "model_backward_count",
                "writer_action_count",
                "materialization_count",
                "heldout_evaluator_count",
                "history_consume_count",
                "history_append_count",
                "target_controller_influence_count",
            )
        )
    ):
        raise ODEBFStateError("P1R52 pre-writer observation contract differs")


def observe_native_il1_pre_writer_input(
    observer: P1R52PreWriterObserver,
    *,
    outer: P1R52TargetDepthOuter,
    step_index: int,
    touched: MutableMapping[str, torch.nn.Parameter],
    request_identities: Sequence[str],
    request_order_sha256: str,
) -> P1R52PreWriterObservationReceipt:
    """Invoke one typed observation at the exact pre-J0 physical boundary."""

    if not isinstance(observer, P1R52PreWriterObserver):
        raise ODEBFContractError("P1R52 pre-writer observer type differs")
    if not isinstance(touched, MutableMapping):
        raise ODEBFContractError("P1R52 pre-writer touched mapping is not mutable")
    requests = tuple(str(item) for item in request_identities)
    if (
        not requests
        or any(len(item) != 64 for item in requests)
        or len(set(requests)) != len(requests)
        or scalable_ordered_request_digest(requests) != request_order_sha256
    ):
        raise ODEBFContractError("P1R52 pre-writer request order differs")
    selected, bridge = reconstruct_native_il1_selected_target(
        outer, step_index=step_index
    )
    if selected.target_step.target_next.shape[1] != len(requests):
        raise ODEBFContractError("P1R52 pre-writer request geometry differs")
    weight_identities, weight_rollback, mapping_rollback = _capture_weights(
        touched
    )
    target_rollback = _capture_target_tensors(selected)
    scientific_identity = canonical_hash(
        [item.scientific_payload() for item in weight_identities]
    )
    pointer_identity = canonical_hash(
        [item.raw_free_payload() for item in weight_identities]
    )
    action_freeze_identity = canonical_hash(
        {
            "native_selected_bridge_identity": bridge.identity_sha256,
            "native_selected_receipt_identity": (
                bridge.native_selected_receipt_identity
            ),
            "outer_reassembly_receipt_identity": (
                bridge.outer_reassembly_receipt_identity
            ),
            "entry_weight_scientific_identity": scientific_identity,
            "entry_weight_pointer_identity": pointer_identity,
            "request_order_sha256": request_order_sha256,
            "step_index": step_index,
        }
    )
    input_receipt = P1R52PreWriterInputReceipt(
        P1R52_PRE_WRITER_INTERFACE_STATUS,
        step_index,
        requests,
        request_order_sha256,
        bridge.identity_sha256,
        bridge.native_selected_receipt_identity,
        bridge.outer_reassembly_receipt_identity,
        mapping_rollback.key_inventory,
        mapping_rollback.object_identity,
        weight_identities,
        scientific_identity,
        pointer_identity,
        action_freeze_identity,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    pre_writer_input = P1R52PreWriterInput(selected, bridge, input_receipt)
    selected_guard = canonical_hash(
        {
            "bridge": bridge.raw_free_payload(),
            "endpoint": _endpoint_guard(selected.selected_endpoint),
            "controller": _controller_identity(selected.next_state),
        }
    )
    try:
        observation = observer.observe(pre_writer_input)
        _validate_observation_receipt(observation, pre_writer_input)
        _validate_entry_unchanged(
            mapping_rollback, weight_rollback, target_rollback
        )
        selected_after = reconstruct_native_il1_selected_target(
            outer, step_index=step_index
        )[1]
        if canonical_hash(
            {
                "bridge": selected_after.raw_free_payload(),
                "endpoint": _endpoint_guard(selected.selected_endpoint),
                "controller": _controller_identity(selected.next_state),
            }
        ) != selected_guard:
            raise ODEBFStateError(
                "P1R52 pre-writer hook mutated target/controller binding"
            )
        return observation
    except Exception:
        _restore_entry(mapping_rollback, weight_rollback, target_rollback)
        raise


__all__ = [
    "P1R52_PRE_WRITER_INTERFACE_STATUS",
    "P1R52_PRE_WRITER_OBSERVATION_STATUS",
    "P1R52IL1NativeSelectedBridgeReceipt",
    "P1R52PreWriterInput",
    "P1R52PreWriterInputReceipt",
    "P1R52PreWriterObservationReceipt",
    "P1R52PreWriterObserver",
    "PreWriterTensorIdentity",
    "PreWriterWeightIdentity",
    "WRITER_RELEVANT_TENSOR_NAMES",
    "observe_native_il1_pre_writer_input",
    "reconstruct_native_il1_selected_target",
]
