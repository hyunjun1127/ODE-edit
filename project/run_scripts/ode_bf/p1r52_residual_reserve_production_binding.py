"""Production object binding for the residual-reserve Phase-A writer.

This module contains no target, routing, writer, or evaluator policy.  It
only binds already-loaded P1R52/AlphaEdit objects to the audited residual-
reserve primitives and adapts the existing physical capture into the typed
prefix-observation interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1r52_residual_reserve_committed_state import (
    CommittedGrossLoadLedger,
    LayerCommittedStateAnchor,
    initialize_committed_gross_load_state,
)
from .p1r52_residual_reserve_fp32_transaction import (
    LayerParameterBinding,
    RESIDUAL_RESERVE_LAYER_ORDER,
)
from .p1r52_residual_reserve_nominal_shadow import (
    PrefixObservation,
    ShadowAlphaLayerContext,
    ShadowWeightStateIdentity,
    build_prefix_observation,
)
from .p1r52_residual_reserve_pc_inventory import (
    SealedPrevalidatedCovariance,
)
from .scalable_batched_model import (
    ScalableCapturePlan,
    capture_scalable_physical_state,
)
from .scalable_batched_runtime import StreamingPhysicalCapture


PRODUCTION_BINDING_STATUS = "P1R52_RR_PHASE_A_PRODUCTION_BINDING"
PREFIX_OBSERVATION_STATUS = "EXISTING_PHYSICAL_CAPTURE_READ_ONLY_PROJECTION"
PARAMETER_ORIENTATION = "ROWS_OUTPUT_COLUMNS_KEY"
PRODUCTION_ALGORITHM_DTYPE = "torch.float32"
PRODUCTION_LIVE_STORAGE_DTYPE = "torch.float32"
PREPARED_UPDATE_BINDING_STATUS = "FP32_REQUIRED_NOT_CONSTRUCTED_IN_M4B_I2"


@dataclass(frozen=True, slots=True)
class ProductionTensorIdentity:
    shape: tuple[int, ...]
    dtype: str
    device: str
    sha256: str
    pointer: int
    version: int
    requires_grad: bool

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "sha256": self.sha256,
            "pointer": self.pointer,
            "version": self.version,
            "requires_grad": self.requires_grad,
        }


@dataclass(frozen=True, slots=True)
class ProductionLayerBindingReceipt:
    layer: int
    weight_name: str
    parameter_identity: ProductionTensorIdentity
    parameter_orientation: str
    projector_input_identity: ProductionTensorIdentity
    projector_execution_identity: ProductionTensorIdentity
    covariance_input_identity: str
    covariance_execution_identity: str
    covariance_artifact_identity: str
    committed_covariance_sha256: str
    regularization_sha256: str
    pretrained_weight_norm_squared: float
    live_parameter_copy_count: int
    live_storage_dtype: str
    authoritative_prepared_update_required_dtype: str
    prepared_update_binding_status: str
    numeric_storage_cast_count: int
    bf16_path_call_count: int
    bf16_path_decision_influence_count: int
    alpha_history_consume_count: int
    alpha_history_append_count: int
    alpha_history_finalize_count: int
    post_storage_decision_influence_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "layer": self.layer,
            "weight_name": self.weight_name,
            "parameter_identity": self.parameter_identity.raw_free_payload(),
            "parameter_orientation": self.parameter_orientation,
            "projector_input_identity": self.projector_input_identity.raw_free_payload(),
            "projector_execution_identity": (
                self.projector_execution_identity.raw_free_payload()
            ),
            "covariance_input_identity": self.covariance_input_identity,
            "covariance_execution_identity": self.covariance_execution_identity,
            "covariance_artifact_identity": self.covariance_artifact_identity,
            "committed_covariance_sha256": self.committed_covariance_sha256,
            "regularization_sha256": self.regularization_sha256,
            "pretrained_weight_norm_squared": self.pretrained_weight_norm_squared,
            "live_parameter_copy_count": self.live_parameter_copy_count,
            "live_storage_dtype": self.live_storage_dtype,
            "authoritative_prepared_update_required_dtype": (
                self.authoritative_prepared_update_required_dtype
            ),
            "prepared_update_binding_status": self.prepared_update_binding_status,
            "numeric_storage_cast_count": self.numeric_storage_cast_count,
            "bf16_path_call_count": self.bf16_path_call_count,
            "bf16_path_decision_influence_count": (
                self.bf16_path_decision_influence_count
            ),
            "alpha_history_consume_count": self.alpha_history_consume_count,
            "alpha_history_append_count": self.alpha_history_append_count,
            "alpha_history_finalize_count": self.alpha_history_finalize_count,
            "post_storage_decision_influence_count": (
                self.post_storage_decision_influence_count
            ),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class ResidualReserveProductionBindingReceipt:
    status: str
    capture_plan_identity: str
    request_order_identity: str
    projector_artifact_identity: str
    layer_receipts: tuple[ProductionLayerBindingReceipt, ...]
    execution_device: str
    algorithm_dtype: str
    model_floating_parameter_count: int
    model_floating_parameter_dtype: str
    live_storage_dtypes: tuple[str, ...]
    authoritative_prepared_update_required_dtype: str
    prepared_update_binding_status: str
    numeric_storage_cast_count: int
    bf16_path_call_count: int
    bf16_path_decision_influence_count: int
    autocast_count: int
    downcast_count: int
    quantization_count: int
    gross_state_identity: str
    gross_state_decision_identity: str
    gross_state_version: int
    gross_factor_count: int
    binding_model_forward_count: int
    binding_model_backward_count: int
    materialization_count: int
    heldout_evaluator_count: int
    alpha_history_consume_count: int
    alpha_history_append_count: int
    alpha_history_finalize_count: int
    post_storage_decision_influence_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "capture_plan_identity": self.capture_plan_identity,
            "request_order_identity": self.request_order_identity,
            "projector_artifact_identity": self.projector_artifact_identity,
            "layer_receipts": [item.raw_free_payload() for item in self.layer_receipts],
            "execution_device": self.execution_device,
            "algorithm_dtype": self.algorithm_dtype,
            "model_floating_parameter_count": self.model_floating_parameter_count,
            "model_floating_parameter_dtype": self.model_floating_parameter_dtype,
            "live_storage_dtypes": list(self.live_storage_dtypes),
            "authoritative_prepared_update_required_dtype": (
                self.authoritative_prepared_update_required_dtype
            ),
            "prepared_update_binding_status": self.prepared_update_binding_status,
            "numeric_storage_cast_count": self.numeric_storage_cast_count,
            "bf16_path_call_count": self.bf16_path_call_count,
            "bf16_path_decision_influence_count": (
                self.bf16_path_decision_influence_count
            ),
            "autocast_count": self.autocast_count,
            "downcast_count": self.downcast_count,
            "quantization_count": self.quantization_count,
            "gross_state_identity": self.gross_state_identity,
            "gross_state_decision_identity": self.gross_state_decision_identity,
            "gross_state_version": self.gross_state_version,
            "gross_factor_count": self.gross_factor_count,
            "binding_model_forward_count": self.binding_model_forward_count,
            "binding_model_backward_count": self.binding_model_backward_count,
            "materialization_count": self.materialization_count,
            "heldout_evaluator_count": self.heldout_evaluator_count,
            "alpha_history_consume_count": self.alpha_history_consume_count,
            "alpha_history_append_count": self.alpha_history_append_count,
            "alpha_history_finalize_count": self.alpha_history_finalize_count,
            "post_storage_decision_influence_count": (
                self.post_storage_decision_influence_count
            ),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class ProductionPrefixObservationReceipt:
    status: str
    layer: int
    sweep_index: int
    capture_plan_identity: str
    capture_identity: str
    capture_batch_plan_identity: str
    capture_model_state_identity: str
    pre_weight_state: tuple[ShadowWeightStateIdentity, ...]
    post_weight_state: tuple[ShadowWeightStateIdentity, ...]
    prefix_observation_identity: str
    terminal_capture_sha256: str
    key_capture_sha256: str
    terminal_execution_sha256: str
    key_execution_sha256: str
    capture_device: str
    execution_device: str
    logical_capture_group_count: int
    physical_forward_count: int
    processed_token_count: int
    padded_token_count: int
    device_copy_count: int
    model_backward_count: int
    materialization_count: int
    heldout_evaluator_count: int
    action_influence_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "layer": self.layer,
            "sweep_index": self.sweep_index,
            "capture_plan_identity": self.capture_plan_identity,
            "capture_identity": self.capture_identity,
            "capture_batch_plan_identity": self.capture_batch_plan_identity,
            "capture_model_state_identity": self.capture_model_state_identity,
            "pre_weight_state": [item.raw_free_payload() for item in self.pre_weight_state],
            "post_weight_state": [item.raw_free_payload() for item in self.post_weight_state],
            "prefix_observation_identity": self.prefix_observation_identity,
            "terminal_capture_sha256": self.terminal_capture_sha256,
            "key_capture_sha256": self.key_capture_sha256,
            "terminal_execution_sha256": self.terminal_execution_sha256,
            "key_execution_sha256": self.key_execution_sha256,
            "capture_device": self.capture_device,
            "execution_device": self.execution_device,
            "logical_capture_group_count": self.logical_capture_group_count,
            "physical_forward_count": self.physical_forward_count,
            "processed_token_count": self.processed_token_count,
            "padded_token_count": self.padded_token_count,
            "device_copy_count": self.device_copy_count,
            "model_backward_count": self.model_backward_count,
            "materialization_count": self.materialization_count,
            "heldout_evaluator_count": self.heldout_evaluator_count,
            "action_influence_count": self.action_influence_count,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class _LiveParameterSnapshot:
    layer: int
    weight_name: str
    parameter: torch.nn.Parameter
    data_reference: torch.Tensor
    value: torch.Tensor
    pointer: int
    version: int
    requires_grad: bool
    sha256: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    stride: tuple[int, ...]
    storage_offset: int


@dataclass(frozen=True, slots=True)
class ResidualReserveProductionBinding:
    layer_bindings: tuple[LayerParameterBinding, ...]
    alpha_contexts: tuple[ShadowAlphaLayerContext, ...]
    covariances: tuple[SealedPrevalidatedCovariance, ...]
    gross_ledger: CommittedGrossLoadLedger
    prefix_provider: "ResidualReserveProductionPrefixObservationProvider"
    receipt: ResidualReserveProductionBindingReceipt

    def bindings_by_layer(self) -> Mapping[int, LayerParameterBinding]:
        return MappingProxyType({item.layer: item for item in self.layer_bindings})

    def contexts_by_layer(self) -> Mapping[int, ShadowAlphaLayerContext]:
        return MappingProxyType({item.layer: item for item in self.alpha_contexts})


def _tensor_identity(value: torch.Tensor) -> ProductionTensorIdentity:
    return ProductionTensorIdentity(
        shape=tuple(value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
        sha256=tensor_sha256(value),
        pointer=int(value.data_ptr()),
        version=int(value._version),
        requires_grad=bool(value.requires_grad),
    )


def _require_autocast_disabled() -> None:
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        raise ODEBFStateError("production residual-reserve autocast is enabled")


def _weight_state(
    bindings: tuple[LayerParameterBinding, ...],
) -> tuple[ShadowWeightStateIdentity, ...]:
    return tuple(
        ShadowWeightStateIdentity(
            layer=item.layer,
            weight_name=item.weight_name,
            shape=tuple(item.parameter.shape),
            dtype=str(item.parameter.dtype),
            device=str(item.parameter.device),
            sha256=tensor_sha256(item.parameter),
            pointer=int(item.parameter.data_ptr()),
        )
        for item in bindings
    )


def _snapshot_parameters(
    bindings: tuple[LayerParameterBinding, ...],
) -> tuple[_LiveParameterSnapshot, ...]:
    result = []
    for binding in bindings:
        parameter = binding.parameter
        result.append(
            _LiveParameterSnapshot(
                layer=binding.layer,
                weight_name=binding.weight_name,
                parameter=parameter,
                data_reference=parameter.data,
                value=parameter.detach().clone(),
                pointer=int(parameter.data_ptr()),
                version=int(parameter._version),
                requires_grad=bool(parameter.requires_grad),
                sha256=tensor_sha256(parameter),
                shape=tuple(parameter.shape),
                dtype=parameter.dtype,
                device=parameter.device,
                stride=tuple(parameter.stride()),
                storage_offset=int(parameter.storage_offset()),
            )
        )
    return tuple(result)


def _parameter_slot(
    model: torch.nn.Module, weight_name: str
) -> tuple[torch.nn.Module, str]:
    parent_name, separator, leaf = weight_name.rpartition(".")
    if not separator or not leaf:
        raise ODEBFContractError("residual-reserve weight name is not qualified")
    try:
        parent = model.get_submodule(parent_name)
    except (AttributeError, KeyError) as exc:
        raise ODEBFContractError("residual-reserve weight module is missing") from exc
    if leaf not in parent._parameters:
        raise ODEBFContractError("residual-reserve live parameter slot is missing")
    return parent, leaf


def _restore_parameters(
    model: torch.nn.Module,
    snapshots: tuple[_LiveParameterSnapshot, ...],
) -> None:
    with torch.no_grad():
        for snapshot in snapshots:
            parent, leaf = _parameter_slot(model, snapshot.weight_name)
            parent._parameters[leaf] = snapshot.parameter
            snapshot.parameter.data = snapshot.data_reference
            snapshot.parameter.copy_(snapshot.value)
            snapshot.parameter.requires_grad_(snapshot.requires_grad)


def _validate_live_snapshot(
    model: torch.nn.Module,
    snapshot: _LiveParameterSnapshot,
    *,
    require_version: bool,
) -> None:
    parent, leaf = _parameter_slot(model, snapshot.weight_name)
    live = parent._parameters[leaf]
    if (
        live is not snapshot.parameter
        or int(live.data_ptr()) != snapshot.pointer
        or tuple(live.shape) != snapshot.shape
        or live.dtype is not snapshot.dtype
        or live.device != snapshot.device
        or tuple(live.stride()) != snapshot.stride
        or int(live.storage_offset()) != snapshot.storage_offset
        or bool(live.requires_grad) != snapshot.requires_grad
        or tensor_sha256(live) != snapshot.sha256
        or (require_version and int(live._version) != snapshot.version)
    ):
        raise ODEBFStateError("physical capture changed a bound live parameter")


def _validate_pre_state(
    expected: tuple[ShadowWeightStateIdentity, ...],
    provided: tuple[ShadowWeightStateIdentity, ...],
) -> None:
    if provided != expected:
        raise ODEBFStateError("prefix capture pre-state differs from live weights")


class ResidualReserveProductionPrefixObservationProvider:
    """Typed read-only projection over the existing physical capture path."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        capture_plan: ScalableCapturePlan,
        hparams: Any,
        bindings: tuple[LayerParameterBinding, ...],
        contexts: tuple[ShadowAlphaLayerContext, ...],
        covariances: tuple[SealedPrevalidatedCovariance, ...],
    ) -> None:
        self._model = model
        self._capture_plan = capture_plan
        self._hparams = hparams
        self._bindings = bindings
        self._contexts = contexts
        self._covariances = covariances
        self._context_guards = tuple(
            (
                item.layer,
                _tensor_identity(item.projector32),
                _tensor_identity(item.committed_covariance32),
                _tensor_identity(item.regularization32),
            )
            for item in contexts
        )
        self._covariance_guards = tuple(
            (item.layer, item.identity_sha256, _tensor_identity(item.value))
            for item in covariances
        )
        self._next_layer_index = 0
        self._sweep_index = 0
        self._receipts: list[ProductionPrefixObservationReceipt] = []

    @property
    def receipts(self) -> tuple[ProductionPrefixObservationReceipt, ...]:
        return tuple(self._receipts)

    def _validate_fixed_inputs(self) -> None:
        for context, guard in zip(self._contexts, self._context_guards, strict=True):
            layer, projector, history, regularization = guard
            if context.layer != layer:
                raise ODEBFStateError("Alpha context layer mutated")
            for value, identity in (
                (context.projector32, projector),
                (context.committed_covariance32, history),
                (context.regularization32, regularization),
            ):
                observed = _tensor_identity(value)
                if observed != identity:
                    raise ODEBFStateError("Alpha context tensor mutated")
        for covariance, guard in zip(
            self._covariances, self._covariance_guards, strict=True
        ):
            layer, identity, tensor_identity = guard
            if (
                covariance.layer != layer
                or covariance.identity_sha256 != identity
                or _tensor_identity(covariance.value) != tensor_identity
            ):
                raise ODEBFStateError("sealed covariance mutated")

    def __call__(
        self,
        layer: int,
        pre_state: tuple[ShadowWeightStateIdentity, ...],
    ) -> PrefixObservation:
        _require_autocast_disabled()
        expected_layer = RESIDUAL_RESERVE_LAYER_ORDER[self._next_layer_index]
        if layer != expected_layer:
            raise ODEBFStateError("production prefix observation order differs")
        self._validate_fixed_inputs()
        expected_pre = _weight_state(self._bindings)
        _validate_pre_state(expected_pre, pre_state)
        snapshots = _snapshot_parameters(self._bindings)
        try:
            observed = capture_scalable_physical_state(
                self._model,
                self._capture_plan,
                self._hparams,
            )
            if not isinstance(observed, StreamingPhysicalCapture):
                raise ODEBFContractError("physical capture result type differs")
            if (
                set(observed.keys_by_layer) != set(RESIDUAL_RESERVE_LAYER_ORDER)
                or observed.physical_forward_count <= 0
                or observed.processed_token_count < 0
                or observed.padded_token_count < observed.processed_token_count
                or observed.batch_plan_sha256
                != self._capture_plan.streaming_plan.identity_sha256
            ):
                raise ODEBFContractError("physical capture receipt differs")
            terminal_capture = observed.terminal_z
            key_capture = observed.keys_by_layer[layer]
            for name, value in (
                ("terminal", terminal_capture),
                ("keys", key_capture),
            ):
                if (
                    not isinstance(value, torch.Tensor)
                    or value.dtype is not torch.float32
                    or value.device.type != "cpu"
                    or value.ndim != 2
                    or value.requires_grad
                    or not bool(torch.isfinite(value).all())
                ):
                    raise ODEBFContractError(f"physical capture {name} differs")
            context = self._contexts[self._next_layer_index]
            execution_device = context.projector32.device
            terminal32 = terminal_capture.detach().to(
                device=execution_device, dtype=torch.float32
            ).contiguous()
            keys32 = key_capture.detach().to(
                device=execution_device, dtype=torch.float32
            ).contiguous()
            if terminal32.shape[1] != keys32.shape[1]:
                raise ODEBFContractError("capture request axis differs")
            for snapshot in snapshots:
                _validate_live_snapshot(
                    self._model, snapshot, require_version=True
                )
            post_state = _weight_state(self._bindings)
            if post_state != expected_pre:
                raise ODEBFStateError("physical capture changed writer weights")
            projected = build_prefix_observation(
                layer=layer,
                pre_observation_weight_state=pre_state,
                current_terminal32=terminal32,
                joint_keys32=keys32,
                terminal_forward_count=1,
                key_forward_count=1,
                model_backward_count=0,
            )
            receipt = ProductionPrefixObservationReceipt(
                status=PREFIX_OBSERVATION_STATUS,
                layer=layer,
                sweep_index=self._sweep_index,
                capture_plan_identity=self._capture_plan.identity_sha256,
                capture_identity=observed.identity_sha256,
                capture_batch_plan_identity=observed.batch_plan_sha256,
                capture_model_state_identity=observed.model_state_sha256,
                pre_weight_state=pre_state,
                post_weight_state=post_state,
                prefix_observation_identity=projected.receipt.identity_sha256,
                terminal_capture_sha256=tensor_sha256(terminal_capture),
                key_capture_sha256=tensor_sha256(key_capture),
                terminal_execution_sha256=tensor_sha256(terminal32),
                key_execution_sha256=tensor_sha256(keys32),
                capture_device=str(terminal_capture.device),
                execution_device=str(execution_device),
                logical_capture_group_count=1,
                physical_forward_count=observed.physical_forward_count,
                processed_token_count=observed.processed_token_count,
                padded_token_count=observed.padded_token_count,
                device_copy_count=int(terminal_capture.device != execution_device)
                + int(key_capture.device != execution_device),
                model_backward_count=0,
                materialization_count=0,
                heldout_evaluator_count=0,
                action_influence_count=0,
            )
            receipt.identity_sha256
        except BaseException:
            _restore_parameters(self._model, snapshots)
            raise
        self._receipts.append(receipt)
        self._next_layer_index += 1
        if self._next_layer_index == len(RESIDUAL_RESERVE_LAYER_ORDER):
            self._next_layer_index = 0
            self._sweep_index += 1
        return projected

    def assert_complete(self, *, expected_sweeps: int) -> None:
        if (
            isinstance(expected_sweeps, bool)
            or not isinstance(expected_sweeps, int)
            or expected_sweeps < 0
            or self._next_layer_index != 0
            or self._sweep_index != expected_sweeps
            or len(self._receipts)
            != expected_sweeps * len(RESIDUAL_RESERVE_LAYER_ORDER)
        ):
            raise ODEBFStateError("production prefix capture ledger is incomplete")


def build_residual_reserve_production_binding(
    *,
    model: torch.nn.Module,
    capture_plan: ScalableCapturePlan,
    hparams: Any,
    touched_weights: Mapping[str, torch.nn.Parameter],
    projector32: torch.Tensor,
    projector_artifact_identity: str,
    covariances: tuple[SealedPrevalidatedCovariance, ...],
    gross_state_id: str,
    q_residual_tolerance: float,
    q_condition_max_dimension: int,
) -> ResidualReserveProductionBinding:
    """Bind existing runtime objects without running the model or a writer."""

    if not isinstance(model, torch.nn.Module):
        raise ODEBFContractError("production binding model type differs")
    _require_autocast_disabled()
    if not isinstance(capture_plan, ScalableCapturePlan):
        raise ODEBFContractError("production capture plan type differs")
    layers = tuple(int(item) for item in hparams.layers)
    if layers != RESIDUAL_RESERVE_LAYER_ORDER:
        raise ODEBFContractError("production binding layer order differs")
    if not isinstance(projector_artifact_identity, str) or not projector_artifact_identity:
        raise ODEBFContractError("projector artifact identity is missing")
    if (
        not isinstance(projector32, torch.Tensor)
        or projector32.dtype is not torch.float32
        or projector32.device.type != "cpu"
        or projector32.ndim != 3
        or projector32.shape[0] != len(layers)
        or projector32.requires_grad
        or not bool(torch.isfinite(projector32).all())
    ):
        raise ODEBFContractError("production projector contract differs")
    if (
        not isinstance(covariances, tuple)
        or tuple(item.layer for item in covariances) != layers
        or any(not isinstance(item, SealedPrevalidatedCovariance) for item in covariances)
    ):
        raise ODEBFContractError("production covariance inventory differs")
    if (
        isinstance(q_condition_max_dimension, bool)
        or not isinstance(q_condition_max_dimension, int)
        or q_condition_max_dimension < 0
        or not math.isfinite(float(q_residual_tolerance))
        or float(q_residual_tolerance) <= 0.0
    ):
        raise ODEBFContractError("production q certificate configuration differs")
    regularization = float(hparams.L2)
    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ODEBFContractError("production Alpha regularization differs")

    live_by_name = dict(model.named_parameters())
    floating_parameters = tuple(
        parameter
        for parameter in live_by_name.values()
        if parameter.is_floating_point()
    )
    if (
        not floating_parameters
        or len(floating_parameters) != len(live_by_name)
        or any(
            parameter.dtype is not torch.float32
            for parameter in floating_parameters
        )
    ):
        raise ODEBFContractError(
            "production model parameters are not uniformly unquantized FP32"
        )
    expected_names = tuple(
        f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in layers
    )
    if set(touched_weights) != set(expected_names):
        raise ODEBFContractError("production touched-weight inventory differs")
    resolved: list[LayerParameterBinding] = []
    for layer, name in zip(layers, expected_names, strict=True):
        parameter = live_by_name.get(name)
        if (
            not isinstance(parameter, torch.nn.Parameter)
            or touched_weights[name] is not parameter
            or parameter.ndim != 2
            or parameter.dtype is not torch.float32
            or parameter.requires_grad
            or not bool(torch.isfinite(parameter).all())
        ):
            raise ODEBFContractError("production live parameter binding differs")
        resolved.append(LayerParameterBinding(layer, name, parameter))
    bindings = tuple(resolved)
    devices = {item.parameter.device for item in bindings}
    if len(devices) != 1:
        raise ODEBFContractError("production writer parameters span devices")
    execution_device = next(iter(devices))

    contexts: list[ShadowAlphaLayerContext] = []
    execution_covariances: list[SealedPrevalidatedCovariance] = []
    anchors: list[LayerCommittedStateAnchor] = []
    layer_receipts: list[ProductionLayerBindingReceipt] = []
    zero_history_by_dimension: dict[int, torch.Tensor] = {}
    for index, (binding, covariance) in enumerate(
        zip(bindings, covariances, strict=True)
    ):
        key_dimension = int(binding.parameter.shape[1])
        projector_layer = projector32[index]
        if (
            tuple(projector_layer.shape) != (key_dimension, key_dimension)
            or tuple(covariance.value.shape) != (key_dimension, key_dimension)
        ):
            raise ODEBFContractError("production parameter orientation differs")
        projector_execution = projector_layer.detach().to(
            device=execution_device, dtype=torch.float32
        ).clone().contiguous()
        covariance_execution = SealedPrevalidatedCovariance(
            binding.layer,
            covariance.value.detach().to(
                device=execution_device, dtype=torch.float32
            ),
            covariance.artifact_identity,
        )
        committed_covariance = zero_history_by_dimension.get(key_dimension)
        if committed_covariance is None:
            committed_covariance = torch.zeros(
                (key_dimension, key_dimension),
                dtype=torch.float32,
                device=execution_device,
            )
            zero_history_by_dimension[key_dimension] = committed_covariance
        regularization32 = torch.tensor(
            regularization, dtype=torch.float32, device=execution_device
        )
        context = ShadowAlphaLayerContext(
            layer=binding.layer,
            weight_name=binding.weight_name,
            parameter_shape=tuple(binding.parameter.shape),
            projector32=projector_execution,
            committed_covariance32=committed_covariance,
            regularization32=regularization32,
            q_residual_tolerance=float(q_residual_tolerance),
            q_condition_max_dimension=q_condition_max_dimension,
            construction_id=canonical_hash(
                {
                    "schema": "p1r52-rr-production-alpha-context/v1",
                    "layer": binding.layer,
                    "weight_name": binding.weight_name,
                    "capture_plan_identity": capture_plan.identity_sha256,
                    "projector_artifact_identity": projector_artifact_identity,
                    "covariance_artifact_identity": covariance.artifact_identity,
                }
            ),
        )
        weight_norm_squared = float(
            torch.sum(
                binding.parameter.detach().to(dtype=torch.float32) ** 2,
                dtype=torch.float64,
            ).item()
        )
        if not math.isfinite(weight_norm_squared) or weight_norm_squared <= 0.0:
            raise ODEBFContractError("production pretrained weight norm differs")
        anchor = LayerCommittedStateAnchor(
            binding.layer,
            covariance_execution,
            weight_norm_squared,
            share_sealed_covariance=True,
        )
        layer_receipt = ProductionLayerBindingReceipt(
            layer=binding.layer,
            weight_name=binding.weight_name,
            parameter_identity=_tensor_identity(binding.parameter),
            parameter_orientation=PARAMETER_ORIENTATION,
            projector_input_identity=_tensor_identity(projector_layer),
            projector_execution_identity=_tensor_identity(projector_execution),
            covariance_input_identity=covariance.identity_sha256,
            covariance_execution_identity=covariance_execution.identity_sha256,
            covariance_artifact_identity=covariance.artifact_identity,
            committed_covariance_sha256=tensor_sha256(committed_covariance),
            regularization_sha256=tensor_sha256(regularization32),
            pretrained_weight_norm_squared=weight_norm_squared,
            live_parameter_copy_count=0,
            live_storage_dtype=PRODUCTION_LIVE_STORAGE_DTYPE,
            authoritative_prepared_update_required_dtype=(
                PRODUCTION_ALGORITHM_DTYPE
            ),
            prepared_update_binding_status=PREPARED_UPDATE_BINDING_STATUS,
            numeric_storage_cast_count=0,
            bf16_path_call_count=0,
            bf16_path_decision_influence_count=0,
            alpha_history_consume_count=0,
            alpha_history_append_count=0,
            alpha_history_finalize_count=0,
            post_storage_decision_influence_count=0,
        )
        layer_receipt.identity_sha256
        contexts.append(context)
        execution_covariances.append(covariance_execution)
        anchors.append(anchor)
        layer_receipts.append(layer_receipt)

    initial_state = initialize_committed_gross_load_state(
        tuple(anchors), state_id=gross_state_id
    )
    ledger = CommittedGrossLoadLedger(initial_state)
    provider = ResidualReserveProductionPrefixObservationProvider(
        model=model,
        capture_plan=capture_plan,
        hparams=hparams,
        bindings=bindings,
        contexts=tuple(contexts),
        covariances=tuple(execution_covariances),
    )
    receipt = ResidualReserveProductionBindingReceipt(
        status=PRODUCTION_BINDING_STATUS,
        capture_plan_identity=capture_plan.identity_sha256,
        request_order_identity=capture_plan.request_order_sha256,
        projector_artifact_identity=projector_artifact_identity,
        layer_receipts=tuple(layer_receipts),
        execution_device=str(execution_device),
        algorithm_dtype=PRODUCTION_ALGORITHM_DTYPE,
        model_floating_parameter_count=len(floating_parameters),
        model_floating_parameter_dtype=PRODUCTION_LIVE_STORAGE_DTYPE,
        live_storage_dtypes=tuple(str(item.parameter.dtype) for item in bindings),
        authoritative_prepared_update_required_dtype=PRODUCTION_ALGORITHM_DTYPE,
        prepared_update_binding_status=PREPARED_UPDATE_BINDING_STATUS,
        numeric_storage_cast_count=0,
        bf16_path_call_count=0,
        bf16_path_decision_influence_count=0,
        autocast_count=0,
        downcast_count=0,
        quantization_count=0,
        gross_state_identity=initial_state.identity_sha256,
        gross_state_decision_identity=initial_state.decision_identity_sha256,
        gross_state_version=initial_state.version,
        gross_factor_count=sum(
            len(item.committed_precast_factors) for item in initial_state.layers
        ),
        binding_model_forward_count=0,
        binding_model_backward_count=0,
        materialization_count=0,
        heldout_evaluator_count=0,
        alpha_history_consume_count=0,
        alpha_history_append_count=0,
        alpha_history_finalize_count=0,
        post_storage_decision_influence_count=0,
    )
    receipt.identity_sha256
    return ResidualReserveProductionBinding(
        bindings,
        tuple(contexts),
        tuple(execution_covariances),
        ledger,
        provider,
        receipt,
    )


__all__ = [
    "PARAMETER_ORIENTATION",
    "PREFIX_OBSERVATION_STATUS",
    "PREPARED_UPDATE_BINDING_STATUS",
    "PRODUCTION_ALGORITHM_DTYPE",
    "PRODUCTION_BINDING_STATUS",
    "PRODUCTION_LIVE_STORAGE_DTYPE",
    "ProductionLayerBindingReceipt",
    "ProductionPrefixObservationReceipt",
    "ProductionTensorIdentity",
    "ResidualReserveProductionBinding",
    "ResidualReserveProductionBindingReceipt",
    "ResidualReserveProductionPrefixObservationProvider",
    "build_residual_reserve_production_binding",
]
