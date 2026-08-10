"""Model-facing primitives for the P1R14 integrated physical writer.

The module contains no experiment-arm expansion.  It provides the single
forward multi-layer key capture, six M=10 no-hook writer VJPs, and six M=10
functional-P VJPs required by the sealed production ledger.  Held-out
evaluation and endpoint probes are intentionally absent from these APIs.
"""

from __future__ import annotations

import math
import hashlib
import threading
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .alpha_backend import W64_CAST_DTYPE, W64_ASSEMBLER_REFERENCE
from .cold_start_target import (
    _rewrap_layer_output,
    _rng_identity,
    _unwrap_layer_output,
)
from .common_cold_coordinate import (
    CommonColdScale,
    CommonColdScaleMetric,
    evaluate_common_cold_objective,
)
from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import WaypointFactor, assemble_effective_bf16, tensor_sha256
from .integrated_physical_writer import (
    PHYSICAL_WRITER_CONTEXT_COUNT,
    PHYSICAL_WRITER_H,
    PHYSICAL_WRITER_LAYER_ORDER,
    PHYSICAL_WRITER_MICROBATCH_GROUPS,
    PHYSICAL_WRITER_MICROBATCH_SIZE,
    TwoOutputVJP,
    batched_two_output_coefficient_vjp,
)
from .p1_backend import (
    FULL_CURRENT_RESIDUAL_DIVISOR,
    P1DynamicField,
    P1LayerField,
    PinnedCovarianceRegistry,
    SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
    SharedTerminalResidualInput,
    _virtual_context,
)
from .p1_controller import P1ControllerLock
from .p1_replay import samplewise_teacher_kl
from .routing import QuadraticBarrier, RoutingProblem
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
    evaluate_routing_objective,
)
from .transaction import AtomicBatchTransaction
from .woodbury import ProjectorCertificate, solve_alpha_woodbury


RUNTIME_SCHEMA = "ode-edit-s05-integrated-physical-writer-runtime/v1"


@dataclass(frozen=True, slots=True)
class IntegratedFieldBuildReceipt:
    """Deterministic scientific identity for a precomputed-key Alpha-WB field."""

    field_semantic_sha256: str
    request_order_sha256: str
    target_state_sha256: str
    terminal_current_z_sha256: str
    residual_sha256: str
    layer_order: tuple[int, ...]
    key_sha256_by_layer: tuple[tuple[int, str], ...]
    q_sha256_by_layer: tuple[tuple[int, str], ...]
    factor_sha256_by_layer: tuple[tuple[int, str], ...]
    covariance_cost_receipt_sha256_by_layer: tuple[tuple[int, str], ...]
    key_forward_family_count: int
    backend_terminal_recapture_count: int
    history_item_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IntegratedTechnicalProblemReceipt:
    problem: RoutingProblem
    factor_energy_raw: tuple[float, ...]
    factor_energy_step: tuple[float, ...]
    capacity_diagonal: tuple[float, ...]
    structural_p_matrix_raw: tuple[tuple[float, ...], ...]
    committed_load_by_layer: tuple[tuple[int, float], ...]
    committed_load_semantics: str
    current_virtual_factor_load_in_committed_load_count: int
    hard_h_p_budget_influence_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "problem_sha256": self.problem.identity(),
            "factor_energy_raw": list(self.factor_energy_raw),
            "factor_energy_step": list(self.factor_energy_step),
            "capacity_diagonal": list(self.capacity_diagonal),
            "structural_p_matrix_raw": [
                list(row) for row in self.structural_p_matrix_raw
            ],
            "committed_load_by_layer": [
                [layer, value] for layer, value in self.committed_load_by_layer
            ],
            "committed_load_semantics": self.committed_load_semantics,
            "current_virtual_factor_load_in_committed_load_count": (
                self.current_virtual_factor_load_in_committed_load_count
            ),
            "hard_h_p_budget_influence_count": (
                self.hard_h_p_budget_influence_count
            ),
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class SharedTargetFactorGroupReceipt:
    request_order_sha256: str
    context_sha256: str
    target_state_sha256: str
    terminal_current_z_sha256: str
    residual_sha256: str
    desired_target_sha256: str
    target_velocity_sha256: str
    current_target_new_nll: float
    goal_target_new_nll: float
    key_sha256_by_layer: tuple[tuple[int, str], ...]
    layer_hook_call_count: tuple[tuple[int, int], ...]
    logical_forward_groups: int
    model_forward_calls: int
    backward_calls: int
    processed_token_count: int
    heldout_access_count: int
    native_or_direct_z_access_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IntegratedVirtualFactorStepReceipt:
    step_index: int
    field_sha256: str
    applied_theta: tuple[float, ...]
    factor_sha256_by_layer: tuple[tuple[int, str], ...]
    accumulated_factor_count_by_layer: tuple[tuple[int, int], ...]
    h_application_count: int
    parameter_mutation_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IntegratedEndpointTransactionReceipt:
    transaction_id: str
    touched_weights: tuple[str, ...]
    accumulated_step_count: int
    factor_count_by_weight: tuple[tuple[str, int], ...]
    entry_sha256_by_weight: tuple[tuple[str, str], ...]
    candidate_sha256_by_weight: tuple[tuple[str, str], ...]
    restored_sha256_by_weight: tuple[tuple[str, str], ...]
    assembly_stats_sha256: tuple[tuple[str, str], ...]
    transaction_commit_count: int
    transaction_rollback_count: int
    postverify_count: int
    explicit_restore_count: int
    final_w0_restore_exact: bool
    pointer_restore_exact: bool
    persistent_commit_count: int
    history_append_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


class IntegratedFactorAccumulator:
    """Append one ``theta=h*v`` factor per writer and grid transition.

    The accumulator is model-free and never mutates a parameter.  A zero-write
    goal-met transition is represented by five zero-theta factors so the K8
    clock and factor inventory remain explicit and auditable.
    """

    def __init__(self) -> None:
        self._factors: dict[str, list[WaypointFactor]] = {}
        self._layers_by_weight: dict[str, int] = {}
        self._next_step = 0
        self._receipts: list[IntegratedVirtualFactorStepReceipt] = []

    @property
    def step_count(self) -> int:
        return self._next_step

    @property
    def receipts(self) -> tuple[IntegratedVirtualFactorStepReceipt, ...]:
        return tuple(self._receipts)

    def append(
        self,
        field: P1DynamicField,
        applied_theta: Sequence[float],
        *,
        step_index: int,
    ) -> IntegratedVirtualFactorStepReceipt:
        theta = tuple(float(item) for item in applied_theta)
        if (
            step_index != self._next_step
            or step_index < 0
            or step_index >= 8
            or tuple(item.layer for item in field.layers)
            != PHYSICAL_WRITER_LAYER_ORDER
            or len(theta) != len(PHYSICAL_WRITER_LAYER_ORDER)
            or any(not math.isfinite(item) or item < 0.0 for item in theta)
        ):
            raise ODEBFContractError("integrated virtual factor step differs")
        created: list[tuple[int, str]] = []
        for ordinal, (layer, coefficient) in enumerate(zip(field.layers, theta)):
            existing_layer = self._layers_by_weight.setdefault(
                layer.weight_name, layer.layer
            )
            if existing_layer != layer.layer:
                raise ODEBFContractError("integrated writer identity changed")
            factor = WaypointFactor(
                layer.weight_name,
                layer.layer,
                0,
                step_index,
                ordinal,
                coefficient,
                layer.residual.clone(),
                layer.q.clone(),
            )
            self._factors.setdefault(layer.weight_name, []).append(factor)
            created.append((layer.layer, canonical_hash({
                "weight_name_sha256": hashlib.sha256(
                    layer.weight_name.encode("utf-8")
                ).hexdigest(),
                "layer": layer.layer,
                "order_key": list(factor.order_key),
                "theta": factor.theta,
                "left_sha256": tensor_sha256(factor.left),
                "right_sha256": tensor_sha256(factor.right),
            })))
        if len(self._factors) != len(PHYSICAL_WRITER_LAYER_ORDER):
            raise ODEBFContractError("integrated factor inventory differs")
        counts = tuple(
            sorted(
                (layer, len(self._factors[name]))
                for name, layer in self._layers_by_weight.items()
            )
        )
        payload = {
            "schema": f"{RUNTIME_SCHEMA}-virtual-factor-step",
            "step_index": step_index,
            "field_sha256": field.identity_sha256,
            "applied_theta": list(theta),
            "factor_sha256_by_layer": [list(item) for item in created],
            "accumulated_factor_count_by_layer": [list(item) for item in counts],
            "h_application_count": 1,
            "parameter_mutation_count": 0,
        }
        receipt = IntegratedVirtualFactorStepReceipt(
            step_index,
            field.identity_sha256,
            theta,
            tuple(created),
            counts,
            1,
            0,
            canonical_hash(payload),
        )
        self._receipts.append(receipt)
        self._next_step += 1
        return receipt

    def factors_by_weight(self) -> dict[str, tuple[WaypointFactor, ...]]:
        return {
            name: tuple(factors)
            for name, factors in sorted(self._factors.items())
        }

    def assert_complete_k8(self) -> None:
        if self._next_step != 8 or any(
            len(factors) != 8 for factors in self._factors.values()
        ):
            raise ODEBFContractError("integrated accumulated endpoint is not K8")


def _commit_verify_restore_integrated_endpoint_locked(
    parameters: Mapping[str, torch.nn.Parameter],
    factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    *,
    accumulated_step_count: int,
    row_block: int,
    transaction_id: str,
    mutation_lock: threading.RLock,
    post_commit_verify: Callable[[], bool],
    fault_after_writes: int | None = None,
) -> IntegratedEndpointTransactionReceipt:
    """Materialize, atomically verify, then restore one accumulated endpoint."""

    names = tuple(sorted(parameters))
    if (
        not names
        or set(names) != set(factors_by_weight)
        or len(names) != len(PHYSICAL_WRITER_LAYER_ORDER)
        or accumulated_step_count < 1
        or accumulated_step_count > 8
        or not callable(post_commit_verify)
    ):
        raise ODEBFContractError("integrated endpoint inventory differs")
    snapshots = {
        name: parameters[name].detach().to(device="cpu").clone()
        for name in names
    }
    pointers = {name: parameters[name].data_ptr() for name in names}
    entry_hashes = tuple((name, tensor_sha256(snapshots[name])) for name in names)
    candidates: dict[str, torch.Tensor] = {}
    stats_hashes: list[tuple[str, str]] = []
    for name in names:
        factors = tuple(factors_by_weight[name])
        if len(factors) != accumulated_step_count:
            raise ODEBFContractError("integrated endpoint factor count differs")
        candidate, stats = assemble_effective_bf16(
            snapshots[name].to(device=parameters[name].device),
            factors,
            row_block=row_block,
        )
        candidates[name] = candidate
        stats_hashes.append((name, canonical_hash(asdict(stats))))
    candidate_hashes = tuple(
        (name, tensor_sha256(candidates[name])) for name in names
    )
    transaction = AtomicBatchTransaction(
        parameters,
        transaction_id=transaction_id,
        mutation_lock=mutation_lock,
    )
    for name in names:
        transaction.stage(name, candidates[name])

    postverify_calls = 0

    def verify() -> bool:
        nonlocal postverify_calls
        postverify_calls += 1
        observed = tuple(
            (name, tensor_sha256(parameters[name])) for name in names
        )
        return observed == candidate_hashes and bool(post_commit_verify())

    commit = transaction.commit(
        post_commit_verify=verify,
        fault_after_writes=fault_after_writes,
    )
    if postverify_calls != 1 or commit.commit_count != 1:
        raise ODEBFStateError("integrated endpoint postverify count differs")
    with mutation_lock, torch.no_grad():
        for name in names:
            parameters[name].copy_(snapshots[name].to(parameters[name].device))
    restored_hashes = tuple(
        (name, tensor_sha256(parameters[name])) for name in names
    )
    pointer_exact = all(parameters[name].data_ptr() == pointers[name] for name in names)
    restore_exact = restored_hashes == entry_hashes
    if not restore_exact or not pointer_exact:
        raise ODEBFStateError("integrated endpoint W0 restore differs")
    payload = {
        "schema": f"{RUNTIME_SCHEMA}-endpoint-transaction",
        "transaction_id": transaction_id,
        "touched_weights": list(names),
        "accumulated_step_count": accumulated_step_count,
        "factor_count_by_weight": [
            [name, len(factors_by_weight[name])] for name in names
        ],
        "entry_sha256_by_weight": [list(item) for item in entry_hashes],
        "candidate_sha256_by_weight": [list(item) for item in candidate_hashes],
        "restored_sha256_by_weight": [list(item) for item in restored_hashes],
        "assembly_stats_sha256": [list(item) for item in stats_hashes],
        "transaction_commit_count": 1,
        "transaction_rollback_count": commit.rollback_count,
        "postverify_count": postverify_calls,
        "explicit_restore_count": 1,
        "final_w0_restore_exact": restore_exact,
        "pointer_restore_exact": pointer_exact,
        "persistent_commit_count": 0,
        "history_append_count": 0,
    }
    return IntegratedEndpointTransactionReceipt(
        transaction_id,
        names,
        accumulated_step_count,
        tuple((name, len(factors_by_weight[name])) for name in names),
        entry_hashes,
        candidate_hashes,
        restored_hashes,
        tuple(stats_hashes),
        1,
        commit.rollback_count,
        postverify_calls,
        1,
        restore_exact,
        pointer_exact,
        0,
        0,
        canonical_hash(payload),
    )


def commit_verify_restore_integrated_endpoint(
    parameters: Mapping[str, torch.nn.Parameter],
    factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    *,
    accumulated_step_count: int,
    row_block: int,
    transaction_id: str,
    mutation_lock: threading.RLock,
    post_commit_verify: Callable[[], bool],
    fault_after_writes: int | None = None,
) -> IntegratedEndpointTransactionReceipt:
    """Hold one re-entrant lock from W0 snapshot through final restoration."""

    if not hasattr(mutation_lock, "__enter__"):
        raise ODEBFContractError("integrated endpoint mutation lock differs")
    with mutation_lock:
        return _commit_verify_restore_integrated_endpoint_locked(
            parameters,
            factors_by_weight,
            accumulated_step_count=accumulated_step_count,
            row_block=row_block,
            transaction_id=transaction_id,
            mutation_lock=mutation_lock,
            post_commit_verify=post_commit_verify,
            fault_after_writes=fault_after_writes,
        )


class PhysicalThetaOverlay(AbstractContextManager["PhysicalThetaOverlay"]):
    """Safely apply ``sum theta_l B_l`` at the five writer modules."""

    def __init__(
        self,
        model: torch.nn.Module,
        field: P1DynamicField,
        theta: torch.Tensor,
    ) -> None:
        if (
            tuple(int(item.layer) for item in field.layers)
            != PHYSICAL_WRITER_LAYER_ORDER
            or theta.ndim != 1
            or theta.numel() != len(PHYSICAL_WRITER_LAYER_ORDER)
            or not theta.requires_grad
            or not bool(torch.isfinite(theta.detach()).all())
        ):
            raise ODEBFContractError("physical theta overlay inventory differs")
        self.model = model
        self.field = field
        self.theta = theta
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, index: int):
        layer = self.field.layers[index]

        def apply(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> torch.Tensor:
            if (
                not inputs
                or not isinstance(inputs[0], torch.Tensor)
                or not isinstance(output, torch.Tensor)
            ):
                raise ODEBFContractError(
                    "physical theta overlay Linear contract differs"
                )
            hidden = inputs[0]
            right = layer.q.to(device=hidden.device, dtype=torch.float32)
            left = layer.residual.to(device=hidden.device, dtype=torch.float32)
            if (
                hidden.shape[-1] != right.shape[0]
                or right.shape[1] != left.shape[1]
                or output.shape[-1] != left.shape[0]
            ):
                raise ODEBFContractError(
                    "physical theta overlay geometry differs"
                )
            perturbation = (hidden.float() @ right) @ left.T
            return (
                output.float() + self.theta[index] * perturbation
            ).to(dtype=output.dtype)

        return apply

    def __enter__(self) -> "PhysicalThetaOverlay":
        try:
            for index, layer in enumerate(self.field.layers):
                module_name = layer.weight_name[: -len(".weight")]
                module = self.model.get_submodule(module_name)
                if type(module) is not torch.nn.Linear:
                    raise ODEBFContractError(
                        "physical theta overlay target is not exact Linear"
                    )
                self._handles.append(
                    module.register_forward_hook(self._hook(index))
                )
        except BaseException:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            raise
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc_type, exc, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        return False


class _TerminalLookupCapture(AbstractContextManager["_TerminalLookupCapture"]):
    def __init__(
        self,
        model: torch.nn.Module,
        layer_name: str,
        lookup_positions: Sequence[int],
        left_padding: Sequence[int] | None = None,
    ) -> None:
        positions = tuple(int(item) for item in lookup_positions)
        if len(positions) != BATCH_SIZE:
            raise ODEBFContractError("physical transport lookup count differs")
        self.model = model
        self.layer_name = layer_name
        self.lookup_positions = positions
        padding = (0,) * BATCH_SIZE if left_padding is None else tuple(
            int(item) for item in left_padding
        )
        if len(padding) != BATCH_SIZE or any(item < 0 for item in padding):
            raise ODEBFContractError("physical transport left padding differs")
        self.left_padding = padding
        self.value: torch.Tensor | None = None
        self.calls = 0
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        if self.calls != 0:
            raise ODEBFContractError("physical transport capture repeated")
        activation = _unwrap_layer_output(output)
        if activation.ndim != 3:
            raise ODEBFContractError("physical transport activation rank differs")
        if activation.shape[0] != BATCH_SIZE:
            if activation.shape[1] == BATCH_SIZE:
                activation = activation.transpose(0, 1)
            else:
                raise ODEBFContractError(
                    "physical transport activation batch differs"
                )
        selected: list[torch.Tensor] = []
        for row, raw_position in enumerate(self.lookup_positions):
            position = (
                raw_position + self.left_padding[row]
                if raw_position >= 0
                else activation.shape[1] + raw_position
            )
            if position < 0 or position >= activation.shape[1]:
                raise ODEBFContractError(
                    "physical transport lookup position differs"
                )
            selected.append(activation[row, position, :])
        self.value = torch.stack(selected, dim=0)
        self.calls += 1

    def __enter__(self) -> "_TerminalLookupCapture":
        module = self.model.get_submodule(self.layer_name)
        self._handle = module.register_forward_hook(self._hook)
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc_type, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if exc is None and (self.calls != 1 or self.value is None):
            raise ODEBFContractError(
                "physical transport capture count differs"
            )
        return False


class _MultiLayerInputCapture(AbstractContextManager["_MultiLayerInputCapture"]):
    def __init__(
        self,
        model: torch.nn.Module,
        module_names: Mapping[int, str],
    ) -> None:
        if tuple(sorted(module_names)) != PHYSICAL_WRITER_LAYER_ORDER:
            raise ODEBFContractError("physical key module inventory differs")
        self.model = model
        self.module_names = dict(module_names)
        self.values: dict[int, torch.Tensor] = {}
        self.calls: dict[int, int] = {layer: 0 for layer in module_names}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, layer: int):
        def apply(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            _output: Any,
        ) -> None:
            if (
                self.calls[layer] != 0
                or not inputs
                or not isinstance(inputs[0], torch.Tensor)
            ):
                raise ODEBFContractError("physical key hook contract differs")
            self.values[layer] = inputs[0].detach()
            self.calls[layer] += 1

        return apply

    def __enter__(self) -> "_MultiLayerInputCapture":
        try:
            for layer in PHYSICAL_WRITER_LAYER_ORDER:
                module = self.model.get_submodule(self.module_names[layer])
                if type(module) is not torch.nn.Linear:
                    raise ODEBFContractError(
                        "physical key target is not exact Linear"
                    )
                self._handles.append(
                    module.register_forward_hook(self._hook(layer))
                )
        except BaseException:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            raise
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc_type, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        if exc is None and (
            set(self.values) != set(PHYSICAL_WRITER_LAYER_ORDER)
            or any(self.calls[layer] != 1 for layer in PHYSICAL_WRITER_LAYER_ORDER)
        ):
            raise ODEBFContractError("physical key hook count differs")
        return False


@dataclass(frozen=True, slots=True)
class MultiLayerKeyReceipt:
    request_order_sha256: str
    context_sha256: str
    layer_order: tuple[int, ...]
    key_sha256_by_layer: tuple[tuple[int, str], ...]
    key_shape_by_layer: tuple[tuple[int, tuple[int, ...]], ...]
    model_forward_calls: int
    logical_forward_groups: int
    processed_token_count: int
    hook_cleanup_exact: bool
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PriorKeyParityReceipt:
    request_order_sha256: str
    shared_key_sha256_by_layer: tuple[tuple[int, str], ...]
    reference_key_sha256_by_layer: tuple[tuple[int, str], ...]
    fixed_forward_group_count: int
    model_forward_calls_by_layer: tuple[tuple[int, int], ...]
    processed_token_count_by_layer: tuple[tuple[int, int], ...]
    exact: bool
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def _process_layer_inputs(
    raw: torch.Tensor,
    lookup_indices: Sequence[Sequence[int]],
) -> torch.Tensor:
    if raw.ndim != 3:
        raise ODEBFContractError("physical key captured input rank differs")
    rows = len(lookup_indices)
    if raw.shape[0] != rows:
        if raw.shape[1] == rows:
            raw = raw.transpose(0, 1)
        else:
            raise ODEBFContractError("physical key captured batch differs")
    selected: list[torch.Tensor] = []
    for row, indices in enumerate(lookup_indices):
        normalized = tuple(int(item) for item in indices)
        if not normalized:
            raise ODEBFContractError("physical key lookup is empty")
        selected.append(raw[row, list(normalized), :].mean(dim=0))
    return torch.stack(selected, dim=0)


class _StreamingWriterKeyCapture(
    AbstractContextManager["_StreamingWriterKeyCapture"]
):
    """Capture lookup inputs at all five writers during the 60 target calls."""

    def __init__(
        self,
        model: torch.nn.Module,
        module_names: Mapping[int, str],
        lookup_indices: Sequence[Sequence[int]],
    ) -> None:
        if (
            tuple(sorted(module_names)) != PHYSICAL_WRITER_LAYER_ORDER
            or len(lookup_indices) != PHYSICAL_WRITER_CONTEXT_COUNT
        ):
            raise ODEBFContractError("shared target/factor key inventory differs")
        self.model = model
        self.module_names = dict(module_names)
        self.lookup_indices = tuple(
            tuple(int(item) for item in values) for values in lookup_indices
        )
        if any(not values for values in self.lookup_indices):
            raise ODEBFContractError("shared target/factor lookup is empty")
        self.values: dict[int, list[torch.Tensor]] = {
            layer: [] for layer in PHYSICAL_WRITER_LAYER_ORDER
        }
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, layer: int):
        def apply(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            _output: Any,
        ) -> None:
            ordinal = len(self.values[layer])
            if (
                ordinal >= PHYSICAL_WRITER_CONTEXT_COUNT
                or not inputs
                or not isinstance(inputs[0], torch.Tensor)
            ):
                raise ODEBFContractError(
                    "shared target/factor key hook contract differs"
                )
            raw = inputs[0]
            if raw.ndim != 3 or raw.shape[0] != 1:
                if raw.ndim == 3 and raw.shape[1] == 1:
                    raw = raw.transpose(0, 1)
                else:
                    raise ODEBFContractError(
                        "shared target/factor key layout differs"
                    )
            indices = self.lookup_indices[ordinal]
            if min(indices) < -raw.shape[1] or max(indices) >= raw.shape[1]:
                raise ODEBFContractError(
                    "shared target/factor key position differs"
                )
            selected = raw[0, list(indices), :].mean(dim=0)
            self.values[layer].append(
                selected.detach().to(device="cpu", dtype=torch.float32)
            )

        return apply

    def __enter__(self) -> "_StreamingWriterKeyCapture":
        try:
            for layer in PHYSICAL_WRITER_LAYER_ORDER:
                module = self.model.get_submodule(self.module_names[layer])
                if type(module) is not torch.nn.Linear:
                    raise ODEBFContractError(
                        "shared target/factor key target is not exact Linear"
                    )
                self._handles.append(
                    module.register_forward_hook(self._hook(layer))
                )
        except BaseException:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            raise
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc_type, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        if exc is None and any(
            len(self.values[layer]) != PHYSICAL_WRITER_CONTEXT_COUNT
            for layer in PHYSICAL_WRITER_LAYER_ORDER
        ):
            raise ODEBFContractError(
                "shared target/factor key call count differs"
            )
        return False

    def keys(self) -> dict[int, torch.Tensor]:
        if self._handles or any(
            len(self.values[layer]) != PHYSICAL_WRITER_CONTEXT_COUNT
            for layer in PHYSICAL_WRITER_LAYER_ORDER
        ):
            raise ODEBFContractError(
                "shared target/factor key capture is incomplete"
            )
        result: dict[int, torch.Tensor] = {}
        for layer in PHYSICAL_WRITER_LAYER_ORDER:
            stacked = torch.stack(self.values[layer], dim=0)
            request_values: list[torch.Tensor] = []
            for request_ordinal in range(BATCH_SIZE):
                start = request_ordinal * PHYSICAL_WRITER_MICROBATCH_GROUPS
                canonical = stacked[start]
                semantic = stacked[start + 1 : start + 6].mean(dim=0)
                request_values.append(torch.stack((canonical, semantic)).mean(dim=0))
            result[layer] = torch.stack(request_values, dim=0).T.contiguous()
        return result


class _DynamicTargetResidualOverlay(
    AbstractContextManager["_DynamicTargetResidualOverlay"]
):
    """Derive one canonical residual per request and reuse it over 6 contexts."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_name: str,
        target_state: torch.Tensor,
        lookup_positions: Sequence[int],
    ) -> None:
        if (
            target_state.ndim != 2
            or target_state.shape[1] != BATCH_SIZE
            or not target_state.requires_grad
            or len(lookup_positions) != PHYSICAL_WRITER_CONTEXT_COUNT
        ):
            raise ODEBFContractError(
                "shared target/factor residual overlay differs"
            )
        self.model = model
        self.layer_name = layer_name
        self.target_state = target_state
        self.lookup_positions = tuple(int(item) for item in lookup_positions)
        self.calls = 0
        self.current_columns: list[torch.Tensor | None] = [None] * BATCH_SIZE
        self.residual_columns: list[torch.Tensor | None] = [None] * BATCH_SIZE
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> Any:
        if self.calls >= PHYSICAL_WRITER_CONTEXT_COUNT:
            raise ODEBFContractError(
                "shared target/factor residual overlay repeated"
            )
        activation = _unwrap_layer_output(output)
        request_ordinal = self.calls // PHYSICAL_WRITER_MICROBATCH_GROUPS
        context_ordinal = self.calls % PHYSICAL_WRITER_MICROBATCH_GROUPS
        raw_position = self.lookup_positions[self.calls]
        if activation.ndim != 3:
            raise ODEBFContractError(
                "shared target/factor residual activation rank differs"
            )
        if activation.shape[0] == 1:
            position = (
                raw_position
                if raw_position >= 0
                else activation.shape[1] + raw_position
            )
            if position < 0 or position >= activation.shape[1]:
                raise ODEBFContractError(
                    "shared target/factor residual position differs"
                )
            before = activation[0, position, :]
            layout = 0
        elif activation.shape[1] == 1:
            position = (
                raw_position
                if raw_position >= 0
                else activation.shape[0] + raw_position
            )
            if position < 0 or position >= activation.shape[0]:
                raise ODEBFContractError(
                    "shared target/factor residual position differs"
                )
            before = activation[position, 0, :]
            layout = 1
        else:
            raise ODEBFContractError(
                "shared target/factor residual layout differs"
            )
        if context_ordinal == 0:
            current = before.detach().to(dtype=torch.float32)
            residual = self.target_state[:, request_ordinal].to(
                device=before.device, dtype=torch.float32
            ) - current
            self.current_columns[request_ordinal] = current.detach().to(
                device="cpu", dtype=torch.float32
            )
            self.residual_columns[request_ordinal] = residual
        residual = self.residual_columns[request_ordinal]
        if residual is None:
            raise ODEBFContractError(
                "shared target/factor residual was not anchored canonically"
            )
        patched = activation.clone()
        assigned = before + residual.to(dtype=before.dtype)
        if layout == 0:
            patched[0, position, :] = assigned
        else:
            patched[position, 0, :] = assigned
        self.calls += 1
        return _rewrap_layer_output(output, patched)

    def __enter__(self) -> "_DynamicTargetResidualOverlay":
        module = self.model.get_submodule(self.layer_name)
        self._handle = module.register_forward_hook(self._hook)
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc_type, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if exc is None and (
            self.calls != PHYSICAL_WRITER_CONTEXT_COUNT
            or any(value is None for value in self.current_columns)
            or any(value is None for value in self.residual_columns)
        ):
            raise ODEBFContractError(
                "shared target/factor residual capture is incomplete"
            )
        return False

    def detached_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._handle is not None or self.calls != PHYSICAL_WRITER_CONTEXT_COUNT:
            raise ODEBFContractError(
                "shared target/factor residual overlay is still active"
            )
        current = torch.stack(
            [value for value in self.current_columns if value is not None], dim=1
        ).contiguous()
        residual = torch.stack(
            [
                value.detach().to(device="cpu", dtype=torch.float32)
                for value in self.residual_columns
                if value is not None
            ],
            dim=1,
        ).contiguous()
        return current, residual


def shared_dynamic_target_factor_group(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    *,
    target_state: torch.Tensor,
    target_layer_name: str,
    lookup_positions: Sequence[int],
    rewrite_module_template: str,
    fact_token_strategy: str,
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    scale_metric: CommonColdScaleMetric,
) -> tuple[
    dict[int, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    SharedTargetFactorGroupReceipt,
]:
    """Build Dynamic-z target and all five keys in one semantic group."""

    from easyeditor.models.rome import repr_tools

    if scale_metric.scale is not CommonColdScale.BATCH_GLOBAL:
        raise ODEBFContractError("integrated physical target allocation is not BG")
    batch, identities, order = _ordered_batch(requests)
    templates, group_sizes, context_sha256 = _locked_context_templates(
        RoutingObjective.TARGET_NEW_NLL, contexts
    )
    if group_sizes != (1, 5) or len(templates) != 6:
        raise ODEBFContractError("shared target/factor contexts differ")
    rendered_templates: list[str] = []
    words: list[str] = []
    for request in batch:
        for template in templates:
            rendered_templates.append(template.format(str(request["prompt"])))
            words.append(str(request["subject"]))
    strategy = str(fact_token_strategy)
    if not strategy.startswith("subject_"):
        raise ODEBFContractError("shared target/factor token strategy differs")
    writer_lookup = repr_tools.get_words_idxs_in_templates(
        tokenizer,
        rendered_templates,
        words,
        strategy[len("subject_") :],
    )
    if len(writer_lookup) != PHYSICAL_WRITER_CONTEXT_COUNT:
        raise ODEBFContractError("shared target/factor writer lookup differs")
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
    residual_overlay = _DynamicTargetResidualOverlay(
        model,
        target_layer_name,
        target_variable,
        lookup_positions,
    )
    with _virtual_context(model, cumulative_factors_by_weight):
        with capture, residual_overlay:
            observed = evaluate_routing_objective(
                model,
                tokenizer,
                batch,
                objective=RoutingObjective.TARGET_NEW_NLL,
                contexts=contexts,
                gradient_input=target_flat,
            )
    if observed.input_gradient is None:
        raise ODEBFContractError(
            "shared target/factor target gradient is absent"
        )
    keys = capture.keys()
    current, residual = residual_overlay.detached_state()
    if not torch.equal(
        residual,
        (
            target_state.detach().to(device="cpu", dtype=torch.float32)
            - current
        ).contiguous(),
    ):
        raise ODEBFContractError(
            "shared target/factor full-current residual differs"
        )
    gradient = observed.input_gradient.view(target_state.shape).detach().to(
        device="cpu", dtype=torch.float64
    ).contiguous()
    velocity, velocity_receipt = scale_metric.velocity(gradient)
    desired = (
        target_state.detach().to(device="cpu", dtype=torch.float32)
        + PHYSICAL_WRITER_H * velocity
    ).contiguous()
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
        raise ODEBFStateError("shared target/factor group mutated model/RNG")
    payload = {
        "schema": f"{RUNTIME_SCHEMA}-shared-target-factor-group",
        "request_order_sha256": order,
        "request_sha256": list(identities),
        "context_sha256": context_sha256,
        "target_state_sha256": tensor_sha256(target_state),
        "terminal_current_z_sha256": tensor_sha256(current),
        "residual_sha256": tensor_sha256(residual),
        "desired_target_sha256": tensor_sha256(desired),
        "target_velocity_sha256": tensor_sha256(velocity),
        "target_velocity_receipt_sha256": velocity_receipt["identity_sha256"],
        "current_target_new_nll": float(observed.loss.detach().cpu()),
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
        "model_forward_calls": observed.model_forward_count + goal.model_forward_count,
        "backward_calls": observed.backward_count,
        "processed_token_count": (
            observed.processed_token_count + goal.processed_token_count
        ),
        "key_forward_family_count": 0,
        "heldout_access_count": 0,
        "native_or_direct_z_access_count": 0,
    }
    receipt = SharedTargetFactorGroupReceipt(
        order,
        context_sha256,
        tensor_sha256(target_state),
        tensor_sha256(current),
        tensor_sha256(residual),
        tensor_sha256(desired),
        tensor_sha256(velocity),
        float(observed.loss.detach().cpu()),
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
        int(payload["model_forward_calls"]),
        int(payload["backward_calls"]),
        int(payload["processed_token_count"]),
        0,
        0,
        canonical_hash(payload),
    )
    return keys, current, residual, desired, receipt


def capture_all_writer_keys_single_forward(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    *,
    rewrite_module_template: str,
    fact_token_strategy: str,
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
) -> tuple[dict[int, torch.Tensor], MultiLayerKeyReceipt]:
    """Capture all five Alpha writer keys with one model forward."""

    from easyeditor.models.rome import repr_tools

    batch, identities, order = _ordered_batch(requests)
    flat_contexts, group_sizes, context_sha256 = _locked_context_templates(
        RoutingObjective.TARGET_NEW_NLL, contexts
    )
    if group_sizes != (1, 5) or len(flat_contexts) != 6:
        raise ODEBFContractError("physical key context geometry differs")
    templates: list[str] = []
    words: list[str] = []
    for request in batch:
        for context in flat_contexts:
            if context is None:
                raise ODEBFContractError("physical key context is absent")
            templates.append(context.format(str(request["prompt"])))
            words.append(str(request["subject"]))
    if len(templates) != PHYSICAL_WRITER_CONTEXT_COUNT:
        raise ODEBFContractError("physical key flattened context count differs")
    subtoken = str(fact_token_strategy)
    if not subtoken.startswith("subject_"):
        raise ODEBFContractError("physical key fact-token strategy differs")
    lookup_indices = repr_tools.get_words_idxs_in_templates(
        tokenizer,
        templates,
        words,
        subtoken[len("subject_") :],
    )
    rendered = [
        template.format(word)
        for template, word in zip(templates, words, strict=True)
    ]
    encoded = tokenizer(rendered, padding=True, return_tensors="pt").to(
        _model_device(model)
    )
    module_names = {
        layer: rewrite_module_template.format(layer)
        for layer in PHYSICAL_WRITER_LAYER_ORDER
    }
    before = _model_state(model)
    before_rng = _rng_identity()
    capture = _MultiLayerInputCapture(model, module_names)
    with _virtual_context(model, cumulative_factors_by_weight):
        with torch.no_grad(), capture:
            model(**encoded)
    if _model_state(model) != before or _rng_identity() != before_rng:
        raise ODEBFStateError("physical key capture mutated model/RNG")
    context_len = sum(group_sizes)
    cumulative = (0, group_sizes[0], context_len)
    keys: dict[int, torch.Tensor] = {}
    for layer in PHYSICAL_WRITER_LAYER_ORDER:
        layer_values = _process_layer_inputs(
            capture.values[layer], lookup_indices
        )
        answers: list[torch.Tensor] = []
        for start in range(0, layer_values.shape[0], context_len):
            group_means = tuple(
                layer_values[start + cumulative[index] : start + cumulative[index + 1]].mean(
                    dim=0
                )
                for index in range(len(group_sizes))
            )
            answers.append(torch.stack(group_means, dim=0).mean(dim=0))
        key = (
            torch.stack(answers, dim=0)
            .T.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        if key.shape[1] != BATCH_SIZE or not bool(torch.isfinite(key).all()):
            raise ODEBFContractError("physical key geometry/value differs")
        keys[layer] = key
    attention = encoded.get("attention_mask")
    processed = (
        int(attention.detach().to(device="cpu").sum())
        if isinstance(attention, torch.Tensor)
        else int(encoded["input_ids"].numel())
    )
    payload = {
        "schema": f"{RUNTIME_SCHEMA}-multi-layer-key-capture",
        "request_order_sha256": order,
        "request_sha256": list(identities),
        "context_sha256": context_sha256,
        "layer_order": list(PHYSICAL_WRITER_LAYER_ORDER),
        "key_sha256_by_layer": [
            [layer, tensor_sha256(keys[layer])]
            for layer in PHYSICAL_WRITER_LAYER_ORDER
        ],
        "key_shape_by_layer": [
            [layer, list(keys[layer].shape)]
            for layer in PHYSICAL_WRITER_LAYER_ORDER
        ],
        "model_forward_calls": 1,
        "logical_forward_groups": 1,
        "processed_token_count": processed,
        "hook_cleanup_exact": True,
    }
    receipt = MultiLayerKeyReceipt(
        order,
        context_sha256,
        PHYSICAL_WRITER_LAYER_ORDER,
        tuple(
            (layer, tensor_sha256(keys[layer]))
            for layer in PHYSICAL_WRITER_LAYER_ORDER
        ),
        tuple(
            (layer, tuple(keys[layer].shape))
            for layer in PHYSICAL_WRITER_LAYER_ORDER
        ),
        1,
        1,
        processed,
        True,
        canonical_hash(payload),
    )
    return keys, receipt


def verify_prior_writer_key_parity_five_groups(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    *,
    hparams: Any,
    shared_keys: Mapping[int, torch.Tensor],
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
) -> PriorKeyParityReceipt:
    """Run the one-time five-family EasyEdit key parity preflight."""

    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    batch, identities, order = _ordered_batch(requests)
    if (
        tuple(int(item) for item in hparams.layers)
        != PHYSICAL_WRITER_LAYER_ORDER
        or tuple(sorted(shared_keys)) != PHYSICAL_WRITER_LAYER_ORDER
    ):
        raise ODEBFContractError("prior writer key parity inventory differs")
    references: dict[int, torch.Tensor] = {}
    calls_by_layer: list[tuple[int, int]] = []
    tokens_by_layer: list[tuple[int, int]] = []
    before = _model_state(model)
    before_rng = _rng_identity()
    with _virtual_context(model, cumulative_factors_by_weight):
        for layer in PHYSICAL_WRITER_LAYER_ORDER:
            model_calls = 0
            processed_tokens = 0

            def observe_forward(
                _module: torch.nn.Module,
                _args: tuple[Any, ...],
                kwargs: Mapping[str, Any],
            ) -> None:
                nonlocal model_calls, processed_tokens
                model_calls += 1
                attention = kwargs.get("attention_mask")
                input_ids = kwargs.get("input_ids")
                if isinstance(attention, torch.Tensor):
                    processed_tokens += int(attention.detach().sum().cpu())
                elif isinstance(input_ids, torch.Tensor):
                    processed_tokens += int(input_ids.numel())

            handle = model.register_forward_pre_hook(
                observe_forward, with_kwargs=True
            )
            try:
                value = alpha_main.compute_ks(
                    model,
                    tokenizer,
                    list(batch),
                    hparams,
                    layer,
                    list(contexts),
                ).T.detach().to(device="cpu", dtype=torch.float32).contiguous()
            finally:
                handle.remove()
            if model_calls <= 0 or processed_tokens <= 0:
                raise ODEBFContractError(
                    "prior writer key parity compute accounting differs"
                )
            calls_by_layer.append((layer, model_calls))
            tokens_by_layer.append((layer, processed_tokens))
            if value.shape != shared_keys[layer].shape:
                raise ODEBFContractError("prior writer key parity shape differs")
            references[layer] = value
    if _model_state(model) != before or _rng_identity() != before_rng:
        raise ODEBFStateError("prior writer key parity mutated model/RNG")
    exact = all(
        torch.equal(shared_keys[layer], references[layer])
        for layer in PHYSICAL_WRITER_LAYER_ORDER
    )
    if not exact:
        raise ODEBFContractError("prior writer key parity differs")
    shared_hashes = tuple(
        (layer, tensor_sha256(shared_keys[layer]))
        for layer in PHYSICAL_WRITER_LAYER_ORDER
    )
    reference_hashes = tuple(
        (layer, tensor_sha256(references[layer]))
        for layer in PHYSICAL_WRITER_LAYER_ORDER
    )
    payload = {
        "schema": f"{RUNTIME_SCHEMA}-prior-key-parity",
        "request_order_sha256": order,
        "request_sha256": list(identities),
        "shared_key_sha256_by_layer": [list(item) for item in shared_hashes],
        "reference_key_sha256_by_layer": [list(item) for item in reference_hashes],
        "fixed_forward_group_count": 5,
        "model_forward_calls_by_layer": [list(item) for item in calls_by_layer],
        "processed_token_count_by_layer": [list(item) for item in tokens_by_layer],
        "exact": exact,
    }
    return PriorKeyParityReceipt(
        order,
        shared_hashes,
        reference_hashes,
        5,
        tuple(calls_by_layer),
        tuple(tokens_by_layer),
        exact,
        canonical_hash(payload),
    )


def build_integrated_physical_field_from_keys(
    model: torch.nn.Module,
    *,
    hparams: Any,
    projector: torch.Tensor,
    precomputed_keys: Mapping[int, torch.Tensor],
    target_state: torch.Tensor,
    terminal_current_z: torch.Tensor,
    request_order_sha256: str,
    accepted_waypoint: int,
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    residual_tolerance: float,
    precomputed_residual: torch.Tensor | None = None,
) -> tuple[P1DynamicField, IntegratedFieldBuildReceipt]:
    """Build the five Alpha-WB factors without any additional model forward.

    The request-specific terminal residual is computed once and byte-copied to
    every writer layer.  Keys are supplied by the shared target/factor group;
    this function never calls ``compute_ks`` or recaptures any activation.
    """

    layers = tuple(int(item) for item in hparams.layers)
    if (
        layers != PHYSICAL_WRITER_LAYER_ORDER
        or tuple(sorted(precomputed_keys)) != PHYSICAL_WRITER_LAYER_ORDER
        or not isinstance(projector, torch.Tensor)
        or projector.dtype is not torch.float32
        or projector.ndim != 3
        or projector.shape[0] != len(layers)
        or not isinstance(accepted_waypoint, int)
        or isinstance(accepted_waypoint, bool)
        or accepted_waypoint < 0
        or not isinstance(projector_sha256, str)
        or len(projector_sha256) != 64
        or not math.isfinite(float(residual_tolerance))
        or float(residual_tolerance) <= 0.0
    ):
        raise ODEBFContractError("integrated physical field inventory differs")
    target = target_state.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    current = terminal_current_z.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    if precomputed_residual is None:
        residual = (target - current).contiguous()
    else:
        residual = precomputed_residual.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        if residual.shape != target.shape or not bool(torch.isfinite(residual).all()):
            raise ODEBFContractError(
                "integrated physical precomputed residual differs"
            )
    shared = SharedTerminalResidualInput(
        residual=residual,
        terminal_current_z=current,
        request_order_sha256=request_order_sha256,
    )
    if target.shape != current.shape:
        raise ODEBFContractError("integrated physical terminal geometry differs")
    parameters = dict(model.named_parameters())
    device = next(model.parameters()).device
    layer_fields: list[P1LayerField] = []
    covariance_cost_hashes: list[tuple[int, str]] = []
    covariance_semantics: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        key = precomputed_keys[layer].detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        if (
            key.ndim != 2
            or key.shape[1] != BATCH_SIZE
            or not bool(torch.isfinite(key).all())
        ):
            raise ODEBFContractError("integrated physical key differs")
        p_device = projector[index].to(device=device, dtype=torch.float32)
        k_device = key.to(device=device, dtype=torch.float32)
        projected_action = p_device @ k_device
        denominator = max(
            float(torch.linalg.norm(projected_action.float())),
            float(torch.finfo(torch.float32).eps),
        )
        symmetry_residual = float(
            torch.linalg.norm(projected_action - p_device.T @ k_device)
        ) / denominator
        idempotence_residual = float(
            torch.linalg.norm(p_device @ projected_action - projected_action)
        ) / denominator
        if not math.isfinite(symmetry_residual) or not math.isfinite(
            idempotence_residual
        ):
            raise ODEBFContractError("integrated projector action probe differs")
        solved = solve_alpha_woodbury(
            p_device,
            k_device,
            history_keys=torch.empty(
                (key.shape[0], 0), device=device, dtype=torch.float32
            ),
            regularization=float(hparams.L2),
            projector_certificate=ProjectorCertificate(
                projector_sha256,
                symmetry_residual,
                idempotence_residual,
                "action-probe",
                1.0e-10,
            ),
            residual_tolerance=float(residual_tolerance),
        )
        q = solved.q.detach().to(
            device="cpu", dtype=W64_CAST_DTYPE
        ).contiguous()
        projected = projected_action.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        covariance_action, covariance_gram, covariance_receipt = (
            covariance_registry.action(layer, q)
        )
        left_gram = shared.residual.T.to(dtype=torch.float64) @ shared.residual.to(
            dtype=torch.float64
        )
        right_gram = q.T.to(dtype=torch.float64) @ q.to(dtype=torch.float64)
        factor_energy = float(torch.sum(left_gram * right_gram))
        if not math.isfinite(factor_energy) or factor_energy <= 0.0:
            raise ODEBFContractError("integrated physical factor has zero capacity")
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        if weight_name not in parameters or tuple(parameters[weight_name].shape) != (
            shared.residual.shape[0],
            q.shape[0],
        ):
            raise ODEBFContractError("integrated physical writer geometry differs")
        factor = WaypointFactor(
            weight_name,
            layer,
            0,
            accepted_waypoint,
            0,
            1.0,
            shared.residual.clone(),
            q.clone(),
        )
        field = P1LayerField(
            layer,
            weight_name,
            key,
            projected,
            shared.residual.clone(),
            SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
            FULL_CURRENT_RESIDUAL_DIVISOR,
            q,
            factor,
            factor_energy,
            covariance_action,
            covariance_gram,
            covariance_receipt,
            solved.certificate,
            torch.empty((shared.residual.shape[0], 0), dtype=torch.float64),
        )
        layer_fields.append(field)
        cost_payload = asdict(covariance_receipt)
        covariance_cost_hashes.append((layer, canonical_hash(cost_payload)))
        covariance_semantics.append(
            {
                key_name: value
                for key_name, value in cost_payload.items()
                if key_name != "wall_seconds"
            }
        )
        del p_device, k_device, solved
    residual_hashes = tuple(
        tensor_sha256(item.residual) for item in layer_fields
    )
    if len(set(residual_hashes)) != 1 or residual_hashes[0] != shared.residual_sha256:
        raise ODEBFContractError("integrated physical residual sharing differs")
    semantic_payload = {
        "schema": f"{RUNTIME_SCHEMA}-field-semantic",
        "reference": "PRECOMPUTED_MULTIHOOK_ALPHA_WB",
        "assembler": W64_ASSEMBLER_REFERENCE,
        "accepted_waypoint": accepted_waypoint,
        "request_order_sha256": request_order_sha256,
        "target_state_sha256": tensor_sha256(target),
        "terminal_current_z_sha256": tensor_sha256(current),
        "residual_policy": SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1,
        "residual_divisor": 1,
        "residual_sha256": shared.residual_sha256,
        "layers": [
            {
                "layer": item.layer,
                "weight_name_sha256": hashlib.sha256(
                    item.weight_name.encode("utf-8")
                ).hexdigest(),
                "key_sha256": tensor_sha256(item.key),
                "projected_key_sha256": tensor_sha256(item.projected_key),
                "q_sha256": tensor_sha256(item.q),
                "factor_sha256": item.factor_identity(),
                "factor_frobenius_sq": item.factor_frobenius_sq,
                "covariance": covariance_semantics[index],
                "woodbury": asdict(item.woodbury_certificate),
            }
            for index, item in enumerate(layer_fields)
        ],
        "history_item_count": 0,
        "key_forward_family_count": 0,
        "backend_terminal_recapture_count": 0,
    }
    semantic_sha = canonical_hash(semantic_payload)
    dynamic = P1DynamicField(
        accepted_waypoint,
        request_order_sha256,
        target.clone(),
        current.clone(),
        tuple(layer_fields),
        semantic_sha,
        0,
        0,
    )
    receipt_payload = {
        **semantic_payload,
        "field_semantic_sha256": semantic_sha,
        "covariance_cost_receipt_sha256_by_layer": [
            list(item) for item in covariance_cost_hashes
        ],
    }
    receipt = IntegratedFieldBuildReceipt(
        semantic_sha,
        request_order_sha256,
        tensor_sha256(target),
        tensor_sha256(current),
        shared.residual_sha256,
        layers,
        tuple((item.layer, tensor_sha256(item.key)) for item in layer_fields),
        tuple((item.layer, tensor_sha256(item.q)) for item in layer_fields),
        tuple((item.layer, item.factor_identity()) for item in layer_fields),
        tuple(covariance_cost_hashes),
        0,
        0,
        0,
        canonical_hash(receipt_payload),
    )
    return dynamic, receipt


def build_integrated_technical_problem(
    field: P1DynamicField,
    *,
    a_edit: Sequence[float],
    committed_load_by_layer: Mapping[int, float],
    controller_lock: P1ControllerLock,
) -> IntegratedTechnicalProblemReceipt:
    """Construct V0 using only box, capacity, and physical write trust."""

    layers = tuple(int(item.layer) for item in field.layers)
    edit = np.asarray(tuple(float(item) for item in a_edit), dtype=np.float64)
    if (
        layers != PHYSICAL_WRITER_LAYER_ORDER
        or edit.shape != (len(layers),)
        or not np.all(np.isfinite(edit))
        or set(committed_load_by_layer) != set(layers)
    ):
        raise ODEBFContractError("integrated technical problem inventory differs")
    raw_energy = np.asarray(
        [item.factor_frobenius_sq for item in field.layers], dtype=np.float64
    )
    step_energy = (1.0 / 8.0) ** 2 * raw_energy
    load = np.asarray(
        [float(committed_load_by_layer[layer]) for layer in layers],
        dtype=np.float64,
    )
    if np.any(load < 0.0) or not np.all(np.isfinite(load)):
        raise ODEBFContractError("integrated committed load differs")
    capacity = (1.0 + load) * (step_energy + controller_lock.capacity_epsilon)
    trust_radius = controller_lock.write_trust_fraction * math.sqrt(
        float(step_energy.sum())
    )
    structural_p = np.diag(
        np.asarray(
            [
                max(
                    float(
                        torch.sum(
                            (
                                item.residual.T.to(dtype=torch.float64)
                                @ item.residual.to(dtype=torch.float64)
                            )
                            * (
                                item.q.T.to(dtype=torch.float64)
                                @ item.covariance_action.to(dtype=torch.float64)
                            )
                        )
                    ),
                    0.0,
                )
                for item in field.layers
            ],
            dtype=np.float64,
        )
    )
    if (
        not np.all(np.isfinite(raw_energy))
        or np.any(raw_energy <= 0.0)
        or not np.all(np.isfinite(capacity))
        or np.any(capacity <= 0.0)
        or not math.isfinite(trust_radius)
        or trust_radius <= 0.0
    ):
        raise ODEBFContractError("integrated technical geometry differs")
    zeros = np.zeros(len(layers), dtype=np.float64)
    zero_matrix = np.zeros((len(layers), len(layers)), dtype=np.float64)
    dummy_progress = min(float(controller_lock.minimum_progress), 1.0e-12)
    problem = RoutingProblem(
        edit,
        np.diag(capacity),
        np.diag(step_energy),
        trust_radius,
        np.minimum(
            np.full(len(layers), controller_lock.layer_velocity_cap), 1.0
        ),
        dummy_progress,
        dummy_progress,
        QuadraticBarrier(
            "historical", 0.0, zeros, zero_matrix, 0.0,
            "layer-local-diagonal",
        ),
        QuadraticBarrier(
            "pretrained", 0.0, zeros, zero_matrix, 0.0,
            "layer-local-diagonal",
        ),
    )
    payload = {
        "schema": f"{RUNTIME_SCHEMA}-technical-problem",
        "field_sha256": field.identity_sha256,
        "problem_sha256": problem.identity(),
        "factor_energy_raw": raw_energy.tolist(),
        "factor_energy_step": step_energy.tolist(),
        "capacity_diagonal": capacity.tolist(),
        "structural_p_matrix_raw": structural_p.tolist(),
        "committed_load_by_layer": [
            [layer, float(committed_load_by_layer[layer])] for layer in layers
        ],
        "committed_load_semantics": "PERSISTENT_OUTER_HISTORY_ONLY",
        "current_virtual_factor_load_in_committed_load_count": 0,
        "technical_constraints": ["BOX_0_1", "PHYSICAL_WRITE_TRUST"],
        "hard_h_p_budget_influence_count": 0,
        "overlay_progress_influence_count": 0,
        "all_five_layer_domain": True,
    }
    return IntegratedTechnicalProblemReceipt(
        problem,
        tuple(float(item) for item in raw_energy),
        tuple(float(item) for item in step_energy),
        tuple(float(item) for item in capacity),
        tuple(tuple(float(item) for item in row) for row in structural_p),
        tuple((layer, float(committed_load_by_layer[layer])) for layer in layers),
        "PERSISTENT_OUTER_HISTORY_ONLY",
        0,
        0,
        canonical_hash(payload),
    )


@dataclass(frozen=True, slots=True)
class WriterVJPMicrobatchReceipt:
    context_ordinal: int
    field_sha256: str
    request_order_sha256: str
    context_sha256: str
    target_span_sha256: str
    loss_new_contribution: float
    loss_transport_contribution: float
    a_edit_contribution: tuple[float, ...]
    a_transport_contribution: tuple[float, ...]
    model_forward_calls: int
    physical_microbatch_graphs: int
    autograd_backend_invocations: int
    backward_calls: int
    processed_token_count: int
    target_old_access_count: int
    heldout_access_count: int
    vjp: TwoOutputVJP
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["vjp"] = self.vjp.raw_free_payload()
        return payload


def writer_edit_transport_vjp_microbatch(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    contexts: Sequence[Sequence[str]],
    *,
    context_ordinal: int,
    field: P1DynamicField,
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    target_layer_name: str,
    lookup_positions: Sequence[int],
    desired_target: torch.Tensor,
    metric_norm_squared: Sequence[float],
) -> WriterVJPMicrobatchReceipt:
    """Evaluate one M=10 no-hook context and its two-output theta VJP."""

    if context_ordinal < 0 or context_ordinal >= PHYSICAL_WRITER_MICROBATCH_GROUPS:
        raise ODEBFContractError("physical writer context ordinal differs")
    batch, identities, order = _ordered_batch(requests)
    templates, group_sizes, context_sha256 = _locked_context_templates(
        RoutingObjective.TARGET_NEW_NLL, contexts
    )
    if group_sizes != (1, 5) or len(templates) != 6:
        raise ODEBFContractError("physical writer context contract differs")
    selected_template = templates[context_ordinal]
    if selected_template is None:
        raise ODEBFContractError("physical writer context is absent")
    positions = tuple(
        int(lookup_positions[ordinal * 6 + context_ordinal])
        for ordinal in range(BATCH_SIZE)
    )
    norms = torch.as_tensor(
        tuple(float(item) for item in metric_norm_squared), dtype=torch.float64
    )
    if (
        norms.shape != (BATCH_SIZE,)
        or not bool(torch.isfinite(norms).all())
        or bool(torch.any(norms <= 1.0e-12))
        or desired_target.ndim != 2
        or desired_target.shape[1] != BATCH_SIZE
        or not bool(torch.isfinite(desired_target).all())
    ):
        raise ODEBFContractError("physical transport metric/target differs")
    device = _model_device(model)
    theta = torch.zeros(
        len(PHYSICAL_WRITER_LAYER_ORDER),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    llama = _is_llama(model)
    before = _model_state(model)
    before_rng = _rng_identity()
    with _temporary_left_padding(tokenizer):
        prefixes: list[str] = []
        target_tokens: list[tuple[int, ...]] = []
        prefix_lengths: list[int] = []
        texts: list[str] = []
        for request in batch:
            prefix, target = _surface(
                request,
                "target_new",
                context_template=selected_template,
            )
            prefixes.append(prefix)
            prefix_lengths.append(
                len(
                    _input_ids(
                        _tokenize_left(tokenizer, [prefix]),
                        label="physical writer prefix",
                        one_row=True,
                    )[0]
                )
            )
            target_tokens.append(_suffix_tokens(tokenizer, target, llama=llama))
            texts.append(f"{prefix} {target}")
        encoded = _move_encoding(
            _tokenize_left(
                tokenizer, texts, padding=True, return_tensors="pt"
            ),
            device,
        )
        input_ids, left_padding = _left_padding_offsets(
            encoded, rows=BATCH_SIZE
        )
        terminal_capture = (
            _TerminalLookupCapture(
                model, target_layer_name, positions, left_padding
            )
            if context_ordinal == 0
            else None
        )
        with _virtual_context(model, cumulative_factors_by_weight):
            with PhysicalThetaOverlay(model, field, theta):
                if terminal_capture is None:
                    logits = model(**encoded).logits
                else:
                    with terminal_capture:
                        logits = model(**encoded).logits
        scores = tuple(
            _score_suffix(
                logits=logits,
                input_ids=input_ids,
                row=ordinal,
                prefix_length=prefix_lengths[ordinal],
                left_padding=left_padding[ordinal],
                tokens=target_tokens[ordinal],
                llama=llama,
                request_sha256=identities[ordinal],
                ordinal=ordinal,
                context_ordinal=context_ordinal,
                context_sha256=context_sha256,
                target_label="target_new",
            )
            for ordinal in range(BATCH_SIZE)
        )
        loss_new = torch.stack([item.value for item in scores]).mean() / 6.0
        if terminal_capture is None:
            loss_transport = theta.sum() * 0.0
        else:
            assert terminal_capture.value is not None
            observed = terminal_capture.value.to(dtype=torch.float64).T
            target = desired_target.to(device=observed.device, dtype=torch.float64)
            if observed.shape != target.shape:
                raise ODEBFContractError(
                    "physical transport target activation shape differs"
                )
            squared = torch.sum((observed - target) ** 2, dim=0)
            loss_transport = 0.5 * torch.mean(
                squared / norms.to(device=observed.device)
            )
        vjp = batched_two_output_coefficient_vjp(
            loss_new, loss_transport, theta
        )
    if _model_state(model) != before or _rng_identity() != before_rng:
        raise ODEBFStateError("physical writer VJP mutated model/RNG")
    attention = encoded.get("attention_mask")
    processed = (
        int(attention.detach().to(device="cpu").sum())
        if isinstance(attention, torch.Tensor)
        else int(input_ids.numel())
    )
    span_sha = canonical_hash([item.span_identity for item in scores])
    payload = {
        "schema": f"{RUNTIME_SCHEMA}-writer-vjp-microbatch",
        "context_ordinal": context_ordinal,
        "field_sha256": field.identity_sha256,
        "request_order_sha256": order,
        "context_sha256": context_sha256,
        "target_span_sha256": span_sha,
        "loss_new_contribution": vjp.loss_new,
        "loss_transport_contribution": vjp.loss_transport,
        "a_edit_contribution": list(vjp.a_edit),
        "a_transport_contribution": list(vjp.a_transport),
        "model_forward_calls": 1,
        "physical_microbatch_graphs": 1,
        "autograd_backend_invocations": 1,
        "backward_calls": 1,
        "processed_token_count": processed,
        "target_old_access_count": 0,
        "heldout_access_count": 0,
        "vjp_sha256": vjp.identity_sha256,
    }
    return WriterVJPMicrobatchReceipt(
        context_ordinal,
        field.identity_sha256,
        order,
        context_sha256,
        span_sha,
        vjp.loss_new,
        vjp.loss_transport,
        vjp.a_edit,
        vjp.a_transport,
        1,
        1,
        1,
        1,
        processed,
        0,
        0,
        vjp,
        canonical_hash(payload),
    )


@dataclass(frozen=True, slots=True)
class WriterVJPFieldReceipt:
    field_sha256: str
    request_order_sha256: str
    context_sha256: str
    loss_new: float
    loss_transport: float
    a_edit: tuple[float, ...]
    a_transport: tuple[float, ...]
    microbatch_receipt_sha256: tuple[str, ...]
    logical_group_count: int
    physical_microbatch_graph_count: int
    autograd_backend_invocation_count: int
    backward_call_count: int
    model_forward_call_count: int
    processed_token_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def accumulate_writer_vjp_microbatches(
    receipts: Sequence[WriterVJPMicrobatchReceipt],
) -> WriterVJPFieldReceipt:
    values = tuple(receipts)
    if (
        len(values) != PHYSICAL_WRITER_MICROBATCH_GROUPS
        or tuple(sorted(item.context_ordinal for item in values))
        != tuple(range(PHYSICAL_WRITER_MICROBATCH_GROUPS))
        or len({item.field_sha256 for item in values}) != 1
        or len({item.request_order_sha256 for item in values}) != 1
        or len({item.context_sha256 for item in values}) != 1
    ):
        raise ODEBFContractError("physical writer VJP accumulation differs")
    ordered = tuple(sorted(values, key=lambda item: item.context_ordinal))
    a_edit = tuple(
        math.fsum(item.a_edit_contribution[index] for item in ordered)
        for index in range(len(PHYSICAL_WRITER_LAYER_ORDER))
    )
    a_transport = tuple(
        math.fsum(item.a_transport_contribution[index] for item in ordered)
        for index in range(len(PHYSICAL_WRITER_LAYER_ORDER))
    )
    payload = {
        "schema": f"{RUNTIME_SCHEMA}-writer-vjp-field",
        "field_sha256": ordered[0].field_sha256,
        "request_order_sha256": ordered[0].request_order_sha256,
        "context_sha256": ordered[0].context_sha256,
        "loss_new": math.fsum(item.loss_new_contribution for item in ordered),
        "loss_transport": math.fsum(
            item.loss_transport_contribution for item in ordered
        ),
        "a_edit": list(a_edit),
        "a_transport": list(a_transport),
        "microbatch_receipt_sha256": [
            item.identity_sha256 for item in ordered
        ],
        "logical_group_count": 1,
        "physical_microbatch_graph_count": len(ordered),
        "autograd_backend_invocation_count": sum(
            item.autograd_backend_invocations for item in ordered
        ),
        "backward_call_count": sum(item.backward_calls for item in ordered),
        "model_forward_call_count": sum(
            item.model_forward_calls for item in ordered
        ),
        "processed_token_count": sum(
            item.processed_token_count for item in ordered
        ),
    }
    return WriterVJPFieldReceipt(
        payload["field_sha256"],
        payload["request_order_sha256"],
        payload["context_sha256"],
        payload["loss_new"],
        payload["loss_transport"],
        a_edit,
        a_transport,
        tuple(payload["microbatch_receipt_sha256"]),
        1,
        len(ordered),
        payload["autograd_backend_invocation_count"],
        payload["backward_call_count"],
        payload["model_forward_call_count"],
        payload["processed_token_count"],
        canonical_hash(payload),
    )


@dataclass(frozen=True, slots=True)
class FunctionalPVJPMicrobatchReceipt:
    microbatch_ordinal: int
    field_sha256: str
    anchor_order_sha256: str
    damage_value: float
    derivative: tuple[float, ...]
    model_forward_calls: int
    physical_microbatch_graphs: int
    autograd_backend_invocations: int
    backward_calls: int
    processed_token_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def functional_p_vjp_microbatch(
    model: torch.nn.Module,
    tokenizer: Any,
    anchors: Sequence[Mapping[str, Any]],
    *,
    microbatch_ordinal: int,
    field: P1DynamicField,
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    teacher_log_probs: torch.Tensor,
    entry_kl: torch.Tensor,
) -> FunctionalPVJPMicrobatchReceipt:
    """Compute one sealed M=10 functional-P theta derivative."""

    batch, identities, order = _ordered_batch(anchors)
    if (
        microbatch_ordinal < 0
        or microbatch_ordinal >= PHYSICAL_WRITER_MICROBATCH_GROUPS
        or teacher_log_probs.ndim != 2
        or teacher_log_probs.shape[0] != BATCH_SIZE
        or entry_kl.shape != (BATCH_SIZE,)
        or not bool(torch.isfinite(teacher_log_probs).all())
        or not bool(torch.isfinite(entry_kl).all())
    ):
        raise ODEBFContractError("functional P VJP input differs")
    device = _model_device(model)
    theta = torch.zeros(
        len(PHYSICAL_WRITER_LAYER_ORDER),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    prompts = [
        str(request["prompt"]).format(str(request["subject"]))
        for request in batch
    ]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    before = _model_state(model)
    before_rng = _rng_identity()
    with _virtual_context(model, cumulative_factors_by_weight):
        with PhysicalThetaOverlay(model, field, theta):
            logits = model(**encoded).logits
    attention = encoded.get("attention_mask")
    if not isinstance(attention, torch.Tensor) or attention.shape != encoded[
        "input_ids"
    ].shape:
        raise ODEBFContractError("functional P attention geometry differs")
    rows = torch.arange(BATCH_SIZE, device=device)
    positions = torch.empty(BATCH_SIZE, dtype=torch.long, device=device)
    for row in range(BATCH_SIZE):
        nonzero = torch.nonzero(attention[row], as_tuple=False).flatten()
        if nonzero.numel() == 0:
            raise ODEBFContractError("functional P anchor is empty")
        positions[row] = nonzero[-1]
    observed = torch.log_softmax(logits[rows, positions, :].float(), dim=-1)
    teacher = teacher_log_probs.to(device=device, dtype=torch.float64)
    baseline = entry_kl.to(device=device, dtype=torch.float64)
    kl = samplewise_teacher_kl(teacher, observed)
    damage = torch.clamp(kl - baseline, min=0.0).mean() / float(
        PHYSICAL_WRITER_MICROBATCH_GROUPS
    )
    gradient = torch.autograd.grad(
        damage, theta, retain_graph=False, create_graph=False
    )[0]
    if not bool(torch.isfinite(gradient).all()):
        raise ODEBFContractError("functional P VJP is non-finite")
    if _model_state(model) != before or _rng_identity() != before_rng:
        raise ODEBFStateError("functional P VJP mutated model/RNG")
    derivative = tuple(
        float(item)
        for item in gradient.detach().to(device="cpu", dtype=torch.float64)
    )
    processed = int(attention.detach().to(device="cpu").sum())
    payload = {
        "schema": f"{RUNTIME_SCHEMA}-functional-p-vjp-microbatch",
        "microbatch_ordinal": microbatch_ordinal,
        "field_sha256": field.identity_sha256,
        "anchor_order_sha256": order,
        "anchor_sha256": list(identities),
        "damage_value": float(
            damage.detach().to(device="cpu", dtype=torch.float64)
        ),
        "derivative": list(derivative),
        "model_forward_calls": 1,
        "physical_microbatch_graphs": 1,
        "autograd_backend_invocations": 1,
        "backward_calls": 1,
        "processed_token_count": processed,
        "endpoint_forward_count": 0,
        "hard_budget_veto_influence_count": 0,
    }
    return FunctionalPVJPMicrobatchReceipt(
        microbatch_ordinal,
        field.identity_sha256,
        order,
        payload["damage_value"],
        derivative,
        1,
        1,
        1,
        1,
        processed,
        canonical_hash(payload),
    )


def accumulate_functional_p_vjp(
    receipts: Sequence[FunctionalPVJPMicrobatchReceipt],
) -> tuple[tuple[float, ...], dict[str, Any]]:
    values = tuple(sorted(receipts, key=lambda item: item.microbatch_ordinal))
    if (
        len(values) != PHYSICAL_WRITER_MICROBATCH_GROUPS
        or tuple(item.microbatch_ordinal for item in values)
        != tuple(range(PHYSICAL_WRITER_MICROBATCH_GROUPS))
        or len({item.field_sha256 for item in values}) != 1
    ):
        raise ODEBFContractError("functional P VJP accumulation differs")
    derivative = tuple(
        math.fsum(item.derivative[index] for item in values)
        for index in range(len(PHYSICAL_WRITER_LAYER_ORDER))
    )
    payload = {
        "schema": f"{RUNTIME_SCHEMA}-functional-p-vjp-field",
        "field_sha256": values[0].field_sha256,
        "derivative": list(derivative),
        "damage_value": math.fsum(item.damage_value for item in values),
        "microbatch_receipt_sha256": [
            item.identity_sha256 for item in values
        ],
        "logical_group_count": 1,
        "physical_microbatch_graph_count": len(values),
        "autograd_backend_invocation_count": sum(
            item.autograd_backend_invocations for item in values
        ),
        "backward_call_count": sum(item.backward_calls for item in values),
        "model_forward_call_count": sum(
            item.model_forward_calls for item in values
        ),
        "processed_token_count": sum(
            item.processed_token_count for item in values
        ),
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return derivative, payload


__all__ = [
    "FunctionalPVJPMicrobatchReceipt",
    "IntegratedEndpointTransactionReceipt",
    "IntegratedFactorAccumulator",
    "IntegratedFieldBuildReceipt",
    "IntegratedTechnicalProblemReceipt",
    "IntegratedVirtualFactorStepReceipt",
    "MultiLayerKeyReceipt",
    "PhysicalThetaOverlay",
    "PriorKeyParityReceipt",
    "SharedTargetFactorGroupReceipt",
    "WriterVJPFieldReceipt",
    "WriterVJPMicrobatchReceipt",
    "accumulate_functional_p_vjp",
    "accumulate_writer_vjp_microbatches",
    "build_integrated_physical_field_from_keys",
    "build_integrated_technical_problem",
    "capture_all_writer_keys_single_forward",
    "commit_verify_restore_integrated_endpoint",
    "functional_p_vjp_microbatch",
    "shared_dynamic_target_factor_group",
    "verify_prior_writer_key_parity_five_groups",
    "writer_edit_transport_vjp_microbatch",
]
