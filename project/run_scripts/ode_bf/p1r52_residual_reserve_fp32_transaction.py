"""Official-style FP32 sequential parameter transaction for residual reserve."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256


RESIDUAL_RESERVE_LAYER_ORDER = (4, 5, 6, 7, 8)


class FP32TransactionMode(str, Enum):
    SHADOW = "SHADOW"
    AUTHORITATIVE = "AUTHORITATIVE"


@dataclass(frozen=True, slots=True)
class LayerParameterBinding:
    layer: int
    weight_name: str
    parameter: torch.nn.Parameter


@dataclass(frozen=True, slots=True)
class FP32LayerApplicationReceipt:
    layer: int
    weight_name: str
    parameter_shape: tuple[int, ...]
    entry_parameter_sha256: str
    entry_parameter_pointer: int
    entry_storage_dtype: str
    entry_storage_device: str
    pre_cast_fp32_update_sha256: str
    pre_cast_fp32_update_norm: float
    pre_cast_fp32_update_energy: float
    post_storage_parameter_sha256: str
    post_storage_parameter_pointer: int
    post_storage_dtype: str
    post_storage_device: str
    actual_post_storage_delta32_sha256: str
    actual_post_storage_delta32_norm: float
    actual_post_storage_delta32_energy: float
    storage_assignment_count: int
    storage_cast_boundary_count: int
    storage_cast_required: bool
    postcast_decision_influence_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "weight_name": self.weight_name,
            "parameter_shape": list(self.parameter_shape),
            "entry_parameter_sha256": self.entry_parameter_sha256,
            "entry_parameter_pointer": self.entry_parameter_pointer,
            "entry_storage_dtype": self.entry_storage_dtype,
            "entry_storage_device": self.entry_storage_device,
            "pre_cast_fp32_update_sha256": self.pre_cast_fp32_update_sha256,
            "pre_cast_fp32_update_norm": self.pre_cast_fp32_update_norm,
            "pre_cast_fp32_update_energy": self.pre_cast_fp32_update_energy,
            "post_storage_parameter_sha256": self.post_storage_parameter_sha256,
            "post_storage_parameter_pointer": self.post_storage_parameter_pointer,
            "post_storage_dtype": self.post_storage_dtype,
            "post_storage_device": self.post_storage_device,
            "actual_post_storage_delta32_sha256": (
                self.actual_post_storage_delta32_sha256
            ),
            "actual_post_storage_delta32_norm": (
                self.actual_post_storage_delta32_norm
            ),
            "actual_post_storage_delta32_energy": (
                self.actual_post_storage_delta32_energy
            ),
            "storage_assignment_count": self.storage_assignment_count,
            "storage_cast_boundary_count": self.storage_cast_boundary_count,
            "storage_cast_required": self.storage_cast_required,
            "postcast_decision_influence_count": (
                self.postcast_decision_influence_count
            ),
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_hash(self.raw_free_payload())


@dataclass(frozen=True, slots=True)
class FP32TransactionReceipt:
    transaction_id: str
    mode: str
    layer_order: tuple[int, ...]
    weight_names: tuple[str, ...]
    layer_receipts: tuple[FP32LayerApplicationReceipt, ...]
    native_storage_assignment_count: int
    storage_cast_boundary_count: int
    logical_outer_commit_count: int
    persistent_commit_count: int
    rollback_count: int
    restored_entry_bytes: bool
    restored_entry_pointers: bool
    final_parameter_sha256: tuple[tuple[str, str], ...]
    final_parameter_pointers: tuple[tuple[str, int], ...]
    postcast_decision_influence_count: int
    model_forward_count: int
    model_backward_count: int
    candidate_materialization_count: int
    external_materializer_call_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "transaction_id": self.transaction_id,
            "mode": self.mode,
            "layer_order": list(self.layer_order),
            "weight_names": list(self.weight_names),
            "layer_receipts": [
                item.raw_free_payload() for item in self.layer_receipts
            ],
            "native_storage_assignment_count": (
                self.native_storage_assignment_count
            ),
            "storage_cast_boundary_count": self.storage_cast_boundary_count,
            "logical_outer_commit_count": self.logical_outer_commit_count,
            "persistent_commit_count": self.persistent_commit_count,
            "rollback_count": self.rollback_count,
            "restored_entry_bytes": self.restored_entry_bytes,
            "restored_entry_pointers": self.restored_entry_pointers,
            "final_parameter_sha256": [list(item) for item in self.final_parameter_sha256],
            "final_parameter_pointers": [
                [name, pointer] for name, pointer in self.final_parameter_pointers
            ],
            "postcast_decision_influence_count": (
                self.postcast_decision_influence_count
            ),
            "model_forward_count": self.model_forward_count,
            "model_backward_count": self.model_backward_count,
            "candidate_materialization_count": self.candidate_materialization_count,
            "external_materializer_call_count": (
                self.external_materializer_call_count
            ),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class FP32PreparedCommitReceipt:
    transaction_id: str
    mode: str
    layer_order: tuple[int, ...]
    weight_names: tuple[str, ...]
    layer_receipts: tuple[FP32LayerApplicationReceipt, ...]
    native_storage_assignment_count: int
    storage_cast_boundary_count: int
    logical_outer_commit_count: int
    persistent_commit_count: int
    rollback_count: int
    prepared_parameter_sha256: tuple[tuple[str, str], ...]
    prepared_parameter_pointers: tuple[tuple[str, int], ...]
    future_final_receipt: FP32TransactionReceipt
    preparation_parameter_mutation_count: int
    postcast_decision_influence_count: int
    model_forward_count: int
    model_backward_count: int
    candidate_materialization_count: int
    external_materializer_call_count: int

    def raw_free_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "transaction_id": self.transaction_id,
            "mode": self.mode,
            "layer_order": list(self.layer_order),
            "weight_names": list(self.weight_names),
            "layer_receipts": [
                item.raw_free_payload() for item in self.layer_receipts
            ],
            "native_storage_assignment_count": (
                self.native_storage_assignment_count
            ),
            "storage_cast_boundary_count": self.storage_cast_boundary_count,
            "logical_outer_commit_count": self.logical_outer_commit_count,
            "persistent_commit_count": self.persistent_commit_count,
            "rollback_count": self.rollback_count,
            "prepared_parameter_sha256": [
                list(item) for item in self.prepared_parameter_sha256
            ],
            "prepared_parameter_pointers": [
                [name, pointer]
                for name, pointer in self.prepared_parameter_pointers
            ],
            "future_final_receipt": self.future_final_receipt.raw_free_payload(),
            "future_final_receipt_identity": (
                self.future_final_receipt.identity_sha256
            ),
            "preparation_parameter_mutation_count": (
                self.preparation_parameter_mutation_count
            ),
            "postcast_decision_influence_count": (
                self.postcast_decision_influence_count
            ),
            "model_forward_count": self.model_forward_count,
            "model_backward_count": self.model_backward_count,
            "candidate_materialization_count": self.candidate_materialization_count,
            "external_materializer_call_count": (
                self.external_materializer_call_count
            ),
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return self.raw_free_payload()["identity_sha256"]


@dataclass(frozen=True, slots=True)
class _EntryParameterSnapshot:
    layer: int
    weight_name: str
    parameter: torch.nn.Parameter
    storage_alias: torch.Tensor
    value: torch.Tensor
    sha256: str
    pointer: int
    dtype: torch.dtype
    device: torch.device
    shape: torch.Size


def _tensor_norm_and_energy(value: torch.Tensor) -> tuple[float, float]:
    flat64 = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    energy = float(torch.dot(flat64, flat64))
    norm = float(torch.sqrt(torch.tensor(energy, dtype=torch.float64)))
    return norm, energy


def _official_native_assignment(
    parameter: torch.nn.Parameter,
    matched_update32: torch.Tensor,
) -> None:
    with torch.no_grad():
        parameter[...] = parameter + matched_update32.float()


class OfficialStyleFP32SequentialTransaction(AbstractContextManager):
    """Apply FP32 updates to live storage in fixed early-to-late order."""

    def __init__(
        self,
        bindings: Mapping[int, tuple[str, torch.nn.Parameter] | LayerParameterBinding],
        *,
        mode: FP32TransactionMode,
        transaction_id: str,
    ) -> None:
        if not isinstance(mode, FP32TransactionMode):
            raise ODEBFContractError("FP32 transaction mode is invalid")
        if not isinstance(transaction_id, str) or not transaction_id:
            raise ODEBFContractError("FP32 transaction identity is empty")
        if set(bindings) != set(RESIDUAL_RESERVE_LAYER_ORDER):
            raise ODEBFContractError("FP32 transaction layer inventory differs")

        resolved: dict[int, LayerParameterBinding] = {}
        names: set[str] = set()
        parameter_ids: set[int] = set()
        for layer in RESIDUAL_RESERVE_LAYER_ORDER:
            item = bindings[layer]
            if isinstance(item, LayerParameterBinding):
                binding = item
            elif isinstance(item, tuple) and len(item) == 2:
                binding = LayerParameterBinding(layer, item[0], item[1])
            else:
                raise ODEBFContractError("FP32 layer binding is malformed")
            if binding.layer != layer:
                raise ODEBFContractError("FP32 layer binding order differs")
            if (
                not isinstance(binding.weight_name, str)
                or not binding.weight_name.endswith(".weight")
            ):
                raise ODEBFContractError("FP32 weight name is invalid")
            if not isinstance(binding.parameter, torch.nn.Parameter):
                raise ODEBFContractError("FP32 binding is not a Parameter")
            parameter = binding.parameter
            if (
                not parameter.is_floating_point()
                or parameter.ndim != 2
                or not bool(torch.isfinite(parameter).all())
            ):
                raise ODEBFContractError("FP32 transaction parameter contract differs")
            if binding.weight_name in names or id(parameter) in parameter_ids:
                raise ODEBFContractError("FP32 transaction binding is duplicated")
            names.add(binding.weight_name)
            parameter_ids.add(id(parameter))
            resolved[layer] = binding

        self.mode = mode
        self.transaction_id = transaction_id
        self._bindings = resolved
        self._entry = {
            layer: _EntryParameterSnapshot(
                layer=layer,
                weight_name=resolved[layer].weight_name,
                parameter=resolved[layer].parameter,
                storage_alias=resolved[layer].parameter.detach(),
                value=resolved[layer]
                .parameter.detach()
                .to(device="cpu")
                .clone(),
                sha256=tensor_sha256(resolved[layer].parameter),
                pointer=int(resolved[layer].parameter.data_ptr()),
                dtype=resolved[layer].parameter.dtype,
                device=resolved[layer].parameter.device,
                shape=resolved[layer].parameter.shape,
            )
            for layer in RESIDUAL_RESERVE_LAYER_ORDER
        }
        self._layer_receipts: list[FP32LayerApplicationReceipt] = []
        self._prepared_receipt: FP32PreparedCommitReceipt | None = None
        self._final_receipt: FP32TransactionReceipt | None = None
        self._finalized = False
        self._rollback_count = 0

    @property
    def applied_layers(self) -> tuple[int, ...]:
        return tuple(item.layer for item in self._layer_receipts)

    @property
    def rollback_count(self) -> int:
        return self._rollback_count

    @property
    def final_receipt(self) -> FP32TransactionReceipt | None:
        return self._final_receipt

    @property
    def prepared_receipt(self) -> FP32PreparedCommitReceipt | None:
        return self._prepared_receipt

    @property
    def finalized(self) -> bool:
        return self._finalized

    def __enter__(self) -> OfficialStyleFP32SequentialTransaction:
        if self._finalized:
            raise ODEBFStateError("FP32 transaction is already finalized")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if not self._finalized:
            self._restore_entry()
            if exc_type is None:
                raise ODEBFStateError(
                    "FP32 transaction exited without shadow finish or commit"
                )
        return False

    def _expected_hash(self, layer: int) -> str:
        for receipt in self._layer_receipts:
            if receipt.layer == layer:
                return receipt.post_storage_parameter_sha256
        return self._entry[layer].sha256

    def _validate_live_state(self) -> None:
        for layer in RESIDUAL_RESERVE_LAYER_ORDER:
            snapshot = self._entry[layer]
            parameter = self._bindings[layer].parameter
            if id(parameter) != id(snapshot.parameter):
                raise ODEBFStateError("FP32 transaction Parameter object changed")
            if (
                int(parameter.data_ptr()) != snapshot.pointer
                or parameter.dtype != snapshot.dtype
                or parameter.device != snapshot.device
                or parameter.shape != snapshot.shape
            ):
                raise ODEBFStateError("FP32 transaction parameter storage changed")
            if tensor_sha256(parameter) != self._expected_hash(layer):
                raise ODEBFStateError("FP32 transaction parameter hash changed")

    def _restore_entry(self) -> None:
        if self._finalized:
            return
        try:
            with torch.no_grad():
                for layer in RESIDUAL_RESERVE_LAYER_ORDER:
                    snapshot = self._entry[layer]
                    parameter = self._bindings[layer].parameter
                    if (
                        int(parameter.data_ptr()) != snapshot.pointer
                        or parameter.dtype != snapshot.dtype
                        or parameter.device != snapshot.device
                        or parameter.shape != snapshot.shape
                    ):
                        parameter.data = snapshot.storage_alias
                    parameter.copy_(snapshot.value)
            for layer in RESIDUAL_RESERVE_LAYER_ORDER:
                snapshot = self._entry[layer]
                parameter = self._bindings[layer].parameter
                if (
                    id(parameter) != id(snapshot.parameter)
                    or int(parameter.data_ptr()) != snapshot.pointer
                    or parameter.dtype != snapshot.dtype
                    or parameter.device != snapshot.device
                    or parameter.shape != snapshot.shape
                    or tensor_sha256(parameter) != snapshot.sha256
                ):
                    raise ODEBFStateError(
                        "FP32 transaction rollback was not pointer/byte exact"
                    )
        except BaseException:
            self._finalized = True
            raise
        self._rollback_count = 1
        self._finalized = True

    def _receipt(
        self,
        *,
        logical_outer_commit_count: int,
        persistent_commit_count: int,
        restored: bool,
    ) -> FP32TransactionReceipt:
        final_hashes = tuple(
            (
                self._bindings[layer].weight_name,
                tensor_sha256(self._bindings[layer].parameter),
            )
            for layer in RESIDUAL_RESERVE_LAYER_ORDER
        )
        final_pointers = tuple(
            (
                self._bindings[layer].weight_name,
                int(self._bindings[layer].parameter.data_ptr()),
            )
            for layer in RESIDUAL_RESERVE_LAYER_ORDER
        )
        return FP32TransactionReceipt(
            transaction_id=self.transaction_id,
            mode=self.mode.value,
            layer_order=RESIDUAL_RESERVE_LAYER_ORDER,
            weight_names=tuple(
                self._bindings[layer].weight_name
                for layer in RESIDUAL_RESERVE_LAYER_ORDER
            ),
            layer_receipts=tuple(self._layer_receipts),
            native_storage_assignment_count=len(self._layer_receipts),
            storage_cast_boundary_count=len(self._layer_receipts),
            logical_outer_commit_count=logical_outer_commit_count,
            persistent_commit_count=persistent_commit_count,
            rollback_count=self._rollback_count,
            restored_entry_bytes=restored,
            restored_entry_pointers=restored,
            final_parameter_sha256=final_hashes,
            final_parameter_pointers=final_pointers,
            postcast_decision_influence_count=0,
            model_forward_count=0,
            model_backward_count=0,
            candidate_materialization_count=0,
            external_materializer_call_count=0,
        )

    def apply_layer_fp32(
        self,
        layer: int,
        matched_update32: torch.Tensor,
    ) -> FP32LayerApplicationReceipt:
        try:
            if self._finalized:
                raise ODEBFStateError("FP32 transaction is already finalized")
            if self._prepared_receipt is not None:
                raise ODEBFStateError("prepared FP32 transaction cannot apply a layer")
            expected_layer = RESIDUAL_RESERVE_LAYER_ORDER[len(self._layer_receipts)]
            if layer != expected_layer:
                raise ODEBFContractError(
                    "FP32 transaction layer is duplicated or out of order"
                )
            self._validate_live_state()
            parameter = self._bindings[layer].parameter
            if not isinstance(matched_update32, torch.Tensor):
                raise ODEBFContractError("FP32 update is not a tensor")
            if matched_update32.dtype is not torch.float32:
                raise ODEBFContractError("scientific writer update must be FP32")
            if matched_update32.shape != parameter.shape:
                raise ODEBFContractError("FP32 update shape differs from parameter")
            if matched_update32.device != parameter.device:
                raise ODEBFContractError("FP32 update device differs from parameter")
            if not bool(torch.isfinite(matched_update32).all()):
                raise ODEBFContractError("FP32 update is non-finite")

            update_pointer = int(matched_update32.data_ptr())
            update_version = int(matched_update32._version)
            update_sha256 = tensor_sha256(matched_update32)
            update_norm, update_energy = _tensor_norm_and_energy(matched_update32)
            snapshot = self._entry[layer]
            _official_native_assignment(parameter, matched_update32)
            if (
                int(matched_update32.data_ptr()) != update_pointer
                or int(matched_update32._version) != update_version
                or tensor_sha256(matched_update32) != update_sha256
            ):
                raise ODEBFStateError("FP32 update input mutated during assignment")
            if (
                int(parameter.data_ptr()) != snapshot.pointer
                or parameter.dtype != snapshot.dtype
                or parameter.device != snapshot.device
                or parameter.shape != snapshot.shape
                or not bool(torch.isfinite(parameter).all())
            ):
                raise ODEBFStateError(
                    "native FP32 assignment changed storage identity or finiteness"
                )

            actual_delta32 = (
                parameter.detach().to(dtype=torch.float32)
                - snapshot.value.to(
                    device=parameter.device,
                    dtype=torch.float32,
                )
            ).contiguous()
            actual_norm, actual_energy = _tensor_norm_and_energy(actual_delta32)
            receipt = FP32LayerApplicationReceipt(
                layer=layer,
                weight_name=self._bindings[layer].weight_name,
                parameter_shape=tuple(parameter.shape),
                entry_parameter_sha256=snapshot.sha256,
                entry_parameter_pointer=snapshot.pointer,
                entry_storage_dtype=str(snapshot.dtype),
                entry_storage_device=str(snapshot.device),
                pre_cast_fp32_update_sha256=update_sha256,
                pre_cast_fp32_update_norm=update_norm,
                pre_cast_fp32_update_energy=update_energy,
                post_storage_parameter_sha256=tensor_sha256(parameter),
                post_storage_parameter_pointer=int(parameter.data_ptr()),
                post_storage_dtype=str(parameter.dtype),
                post_storage_device=str(parameter.device),
                actual_post_storage_delta32_sha256=tensor_sha256(actual_delta32),
                actual_post_storage_delta32_norm=actual_norm,
                actual_post_storage_delta32_energy=actual_energy,
                storage_assignment_count=1,
                storage_cast_boundary_count=1,
                storage_cast_required=parameter.dtype is not torch.float32,
                postcast_decision_influence_count=0,
            )
            self._layer_receipts.append(receipt)
            self._validate_live_state()
            return receipt
        except BaseException:
            if not self._finalized:
                self._restore_entry()
            raise

    def _require_complete(self) -> None:
        if self.applied_layers != RESIDUAL_RESERVE_LAYER_ORDER:
            raise ODEBFStateError("FP32 transaction is missing a layer application")
        self._validate_live_state()

    def finish_shadow(self) -> FP32TransactionReceipt:
        try:
            if self._finalized:
                raise ODEBFStateError("FP32 transaction is already finalized")
            if self.mode is not FP32TransactionMode.SHADOW:
                raise ODEBFContractError(
                    "authoritative FP32 transaction cannot finish as shadow"
                )
            self._require_complete()
            self._restore_entry()
            self._final_receipt = self._receipt(
                logical_outer_commit_count=0,
                persistent_commit_count=0,
                restored=True,
            )
            return self._final_receipt
        except BaseException:
            if not self._finalized:
                self._restore_entry()
            raise

    def commit_outer(
        self,
        expected_prepare_identity: str | None = None,
    ) -> FP32TransactionReceipt:
        """Finalize only an explicitly prepared authoritative transaction."""
        return self.commit_prepared(expected_prepare_identity)

    def prepare_authoritative_commit(self) -> FP32PreparedCommitReceipt:
        try:
            if self._finalized:
                raise ODEBFStateError("FP32 transaction is already finalized")
            if self._prepared_receipt is not None:
                raise ODEBFStateError("FP32 transaction is already prepared")
            if self.mode is not FP32TransactionMode.AUTHORITATIVE:
                raise ODEBFContractError("shadow FP32 transaction cannot prepare")
            self._require_complete()
            future_final_receipt = self._receipt(
                logical_outer_commit_count=1,
                persistent_commit_count=1,
                restored=False,
            )
            prepared_receipt = FP32PreparedCommitReceipt(
                transaction_id=self.transaction_id,
                mode=self.mode.value,
                layer_order=RESIDUAL_RESERVE_LAYER_ORDER,
                weight_names=tuple(
                    self._bindings[layer].weight_name
                    for layer in RESIDUAL_RESERVE_LAYER_ORDER
                ),
                layer_receipts=tuple(self._layer_receipts),
                native_storage_assignment_count=len(self._layer_receipts),
                storage_cast_boundary_count=len(self._layer_receipts),
                logical_outer_commit_count=0,
                persistent_commit_count=0,
                rollback_count=self._rollback_count,
                prepared_parameter_sha256=(
                    future_final_receipt.final_parameter_sha256
                ),
                prepared_parameter_pointers=(
                    future_final_receipt.final_parameter_pointers
                ),
                future_final_receipt=future_final_receipt,
                preparation_parameter_mutation_count=0,
                postcast_decision_influence_count=0,
                model_forward_count=0,
                model_backward_count=0,
                candidate_materialization_count=0,
                external_materializer_call_count=0,
            )
            prepared_receipt.identity_sha256
            self._prepared_receipt = prepared_receipt
            return prepared_receipt
        except BaseException:
            if not self._finalized:
                self._restore_entry()
            raise

    def commit_prepared(
        self,
        expected_prepare_identity: str,
    ) -> FP32TransactionReceipt:
        try:
            if self._finalized:
                raise ODEBFStateError("FP32 transaction is already finalized")
            if self.mode is not FP32TransactionMode.AUTHORITATIVE:
                raise ODEBFContractError("shadow FP32 transaction cannot commit")
            prepared = self._prepared_receipt
            if prepared is None:
                raise ODEBFStateError("FP32 transaction is not prepared")
            if (
                not isinstance(expected_prepare_identity, str)
                or expected_prepare_identity != prepared.identity_sha256
            ):
                raise ODEBFStateError("FP32 prepared identity differs")
            self._require_complete()
            self._final_receipt = prepared.future_final_receipt
            self._finalized = True
            return self._final_receipt
        except BaseException:
            if not self._finalized:
                self._restore_entry()
            raise

    def abort_and_rollback(self) -> FP32TransactionReceipt:
        if self._finalized:
            raise ODEBFStateError("FP32 transaction is already finalized")
        self._restore_entry()
        self._final_receipt = self._receipt(
            logical_outer_commit_count=0,
            persistent_commit_count=0,
            restored=True,
        )
        return self._final_receipt


__all__ = [
    "FP32LayerApplicationReceipt",
    "FP32PreparedCommitReceipt",
    "FP32TransactionMode",
    "FP32TransactionReceipt",
    "LayerParameterBinding",
    "OfficialStyleFP32SequentialTransaction",
    "RESIDUAL_RESERVE_LAYER_ORDER",
]
