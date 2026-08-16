"""Canonical entry-relative cumulative BF16 endpoint assembler and trial."""

from __future__ import annotations

import hashlib
import math
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as torch_functional

from .contracts import BATCH_SIZE, ODEBFContractError, finite


def tensor_sha256(tensor: torch.Tensor) -> str:
    logical = tensor.detach().to(device="cpu").contiguous().view(-1)
    return hashlib.sha256(logical.view(torch.uint8).numpy().tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class WaypointFactor:
    weight_name: str
    layer: int
    correction_cycle: int
    step_in_cycle: int
    factor_ordinal: int
    theta: float
    left: torch.Tensor
    right: torch.Tensor
    joint_batch: bool = True
    global_batch_size: int | None = None

    def __post_init__(self) -> None:
        if not self.weight_name.endswith(".weight"):
            raise ODEBFContractError("waypoint factor does not identify a weight")
        for name, value in (
            ("correction_cycle", self.correction_cycle),
            ("step_in_cycle", self.step_in_cycle),
            ("factor_ordinal", self.factor_ordinal),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ODEBFContractError(f"{name} is invalid")
        coefficient = finite("theta", self.theta)
        if coefficient < 0.0:
            raise ODEBFContractError("finite factor increment theta cannot be negative")
        if self.left.ndim != 2 or self.right.ndim != 2 or self.left.shape[1] != self.right.shape[1]:
            raise ODEBFContractError("low-rank waypoint factor shapes differ")
        rank = self.left.shape[1]
        if self.joint_batch:
            expected_rank = (
                BATCH_SIZE
                if self.global_batch_size is None
                else self.global_batch_size
            )
            if (
                isinstance(expected_rank, bool)
                or not isinstance(expected_rank, int)
                or expected_rank <= 0
                or rank != expected_rank
            ):
                raise ODEBFContractError(
                    "integration waypoint factor rank axis differs from global batch"
                )
        if not self.joint_batch and rank != 1:
            raise ODEBFContractError("algebra-only smoke factor must have rank one")
        if not self.joint_batch and self.global_batch_size is not None:
            raise ODEBFContractError("algebra-only factor cannot bind a global batch")
        if self.left.dtype not in (torch.float32, torch.float64) or self.right.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ODEBFContractError("waypoint factors must use FP32 or FP64")
        if not torch.isfinite(self.left).all() or not torch.isfinite(self.right).all():
            raise ODEBFContractError("waypoint factor contains non-finite values")
        object.__setattr__(self, "theta", coefficient)

    @property
    def order_key(self) -> tuple[int, int, int]:
        return self.correction_cycle, self.step_in_cycle, self.factor_ordinal


@dataclass(frozen=True, slots=True)
class AssemblyStats:
    weight_name: str
    factor_count: int
    rank_columns_total: int
    maximum_fp32_block_elements: int
    dense_fp32_full_delta_live: int
    effective_bf16_sha256: str


def assemble_effective_bf16(
    entry_weight: torch.Tensor,
    factors: Sequence[WaypointFactor],
    *,
    row_block: int,
    compute_sha256: bool = True,
) -> tuple[torch.Tensor, AssemblyStats]:
    if entry_weight.dtype is not torch.bfloat16 or entry_weight.ndim != 2:
        raise ODEBFContractError("authoritative entry weight must be a BF16 matrix")
    if isinstance(row_block, bool) or not isinstance(row_block, int) or row_block <= 0:
        raise ODEBFContractError("BF16 assembler row block is invalid")
    ordered = tuple(sorted(factors, key=lambda factor: factor.order_key))
    if not ordered:
        raise ODEBFContractError("cumulative endpoint has no factors")
    if len({factor.order_key for factor in ordered}) != len(ordered):
        raise ODEBFContractError("cumulative factor order contains a duplicate")
    weight_names = {factor.weight_name for factor in ordered}
    layers = {factor.layer for factor in ordered}
    if len(weight_names) != 1 or len(layers) != 1:
        raise ODEBFContractError("one cumulative assembler call mixed parameter arms")
    for factor in ordered:
        if tuple(entry_weight.shape) != (factor.left.shape[0], factor.right.shape[0]):
            raise ODEBFContractError("waypoint factor does not match entry weight shape")

    effective = torch.empty_like(entry_weight)
    maximum_block = 0
    with torch.no_grad():
        for start in range(0, entry_weight.shape[0], row_block):
            end = min(start + row_block, entry_weight.shape[0])
            accumulator = entry_weight.detach()[start:end].to(dtype=torch.float32)
            maximum_block = max(maximum_block, accumulator.numel())
            for factor in ordered:
                left_block = factor.left[start:end].to(
                    device=entry_weight.device,
                    dtype=torch.float32,
                )
                right = factor.right.to(device=entry_weight.device, dtype=torch.float32)
                # Preserve AlphaEdit's solve-output @ residual.T product order,
                # then transpose into the model weight orientation.
                native_order = (right @ left_block.T).T
                coefficient = torch.as_tensor(
                    factor.theta,
                    dtype=torch.float32,
                    device=entry_weight.device,
                )
                accumulator = accumulator + coefficient * native_order
                maximum_block = max(maximum_block, native_order.numel(), accumulator.numel())
                del left_block, right, native_order, coefficient
            effective[start:end].copy_(accumulator.to(dtype=torch.bfloat16))
            del accumulator
    stats = AssemblyStats(
        ordered[0].weight_name,
        len(ordered),
        sum(factor.left.shape[1] for factor in ordered),
        maximum_block,
        0,
        tensor_sha256(effective) if compute_sha256 else "NOT_COMPUTED",
    )
    return effective, stats


class CumulativeBF16FunctionalTrial(AbstractContextManager["CumulativeBF16FunctionalTrial"]):
    """Exact Linear replacement with one live effective target weight at a time."""

    verdict_eligible = True

    def __init__(
        self,
        model: torch.nn.Module,
        factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
        *,
        row_block: int,
    ) -> None:
        if not factors_by_weight:
            raise ODEBFContractError("BF16 cumulative trial target set is empty")
        self.model = model
        self.factors_by_weight = {
            name: tuple(factors) for name, factors in factors_by_weight.items()
        }
        self.row_block = row_block
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._parameters: list[tuple[str, torch.nn.Parameter, int, int, bool]] = []
        self._cpu_rng: torch.Tensor | None = None
        self._cuda_rng: dict[torch.device, torch.Tensor] = {}
        self._active_effective_weights = 0
        self.max_live_effective_weights = 0
        self.max_fp32_block_elements = 0
        self.replacement_linear_calls = 0

    @staticmethod
    def _resolve_linear(
        model: torch.nn.Module,
        weight_name: str,
    ) -> tuple[torch.nn.Linear, torch.nn.Parameter]:
        if not weight_name.endswith(".weight"):
            raise ODEBFContractError("BF16 trial target is not a weight")
        module_name = weight_name[: -len(".weight")]
        try:
            module = model.get_submodule(module_name)
            parameter = dict(model.named_parameters())[weight_name]
        except (AttributeError, KeyError) as exc:
            raise ODEBFContractError("BF16 trial target is absent") from exc
        if type(module) is not torch.nn.Linear or module.weight is not parameter:
            raise ODEBFContractError("BF16 verdict requires an exact Linear module")
        return module, parameter

    def _hook(self, factors: tuple[WaypointFactor, ...]):
        def apply(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> torch.Tensor:
            del output
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise ODEBFContractError("BF16 verdict Linear has no tensor input")
            hidden = inputs[0]
            if hidden.dtype is not torch.bfloat16:
                raise ODEBFContractError("BF16 verdict activation is not BF16")
            self._active_effective_weights += 1
            self.max_live_effective_weights = max(
                self.max_live_effective_weights,
                self._active_effective_weights,
            )
            try:
                effective, stats = assemble_effective_bf16(
                    module.weight,
                    factors,
                    row_block=self.row_block,
                )
                self.max_fp32_block_elements = max(
                    self.max_fp32_block_elements,
                    stats.maximum_fp32_block_elements,
                )
                with torch.no_grad():
                    result = torch_functional.linear(hidden, effective, module.bias)
                self.replacement_linear_calls += 1
                del effective
                return result
            finally:
                self._active_effective_weights -= 1

        return apply

    def __enter__(self) -> "CumulativeBF16FunctionalTrial":
        if self._handles:
            raise ODEBFContractError("BF16 cumulative trial is already active")
        self._cpu_rng = torch.get_rng_state().clone()
        try:
            for name, factors in self.factors_by_weight.items():
                if any(factor.weight_name != name for factor in factors):
                    raise ODEBFContractError("BF16 factor mapping key differs")
                module, parameter = self._resolve_linear(self.model, name)
                if parameter.dtype is not torch.bfloat16 or parameter.grad is not None:
                    raise ODEBFContractError("BF16 trial parameter state is invalid")
                self._parameters.append(
                    (
                        name,
                        parameter,
                        parameter.data_ptr(),
                        parameter._version,
                        parameter.requires_grad,
                    )
                )
                self._handles.append(module.register_forward_hook(self._hook(factors)))
            cuda_devices = {
                parameter.device
                for _, parameter, _, _, _ in self._parameters
                if parameter.device.type == "cuda"
            }
            self._cuda_rng = {
                device: torch.cuda.get_rng_state(device).clone()
                for device in cuda_devices
            }
        except BaseException:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            self._parameters.clear()
            self._cuda_rng.clear()
            self._cpu_rng = None
            raise
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        violations: list[str] = []
        for name, parameter, pointer, version, requires_grad in self._parameters:
            if parameter.data_ptr() != pointer:
                violations.append(f"{name}:pointer")
            if parameter._version != version:
                violations.append(f"{name}:version")
            if parameter.requires_grad != requires_grad or parameter.grad is not None:
                violations.append(f"{name}:grad")
        self._parameters.clear()
        if self._cpu_rng is not None:
            torch.set_rng_state(self._cpu_rng)
            self._cpu_rng = None
        for device, state in self._cuda_rng.items():
            torch.cuda.set_rng_state(state, device)
        self._cuda_rng.clear()
        if self._active_effective_weights != 0 or self.max_live_effective_weights > 1:
            violations.append("effective-weight-liveness")
        if violations:
            error = ODEBFContractError(
                "BF16 cumulative trial mutated state: " + ",".join(violations)
            )
            if exc is not None:
                raise error from exc
            raise error
        return False


class CachedBF16FunctionalTrial(AbstractContextManager["CachedBF16FunctionalTrial"]):
    """Entry-relative trial that assembles each effective weight once.

    Unlike :class:`CumulativeBF16FunctionalTrial`, forward hooks only apply a
    cached tensor.  This is the P1R22 fallback/diagnostic path for evaluations
    that must inspect a cumulative candidate while the accepted physical state
    is already materialized in the model.
    """

    verdict_eligible = True

    def __init__(
        self,
        model: torch.nn.Module,
        entry_weights: Mapping[str, torch.Tensor],
        factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
        *,
        row_block: int,
    ) -> None:
        if not factors_by_weight or set(entry_weights) != set(factors_by_weight):
            raise ODEBFContractError("cached BF16 trial inventory differs")
        self.model = model
        self.entry_weights = dict(entry_weights)
        self.factors_by_weight = {
            name: tuple(factors) for name, factors in factors_by_weight.items()
        }
        self.row_block = row_block
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._parameters: list[tuple[str, torch.nn.Parameter, int, int, bool]] = []
        self._effective: dict[str, torch.Tensor] = {}
        self.dense_assembly_count = 0
        self.full_weight_hash_count = 0
        self.hot_hook_dense_assembly_count = 0
        self.hot_hook_full_weight_hash_count = 0
        self.replacement_linear_calls = 0

    def _hook(self, name: str):
        def apply(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> torch.Tensor:
            del output
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise ODEBFContractError("cached BF16 trial input differs")
            hidden = inputs[0]
            effective = self._effective[name]
            if hidden.dtype is not torch.bfloat16 or effective.dtype is not torch.bfloat16:
                raise ODEBFContractError("cached BF16 trial dtype differs")
            self.replacement_linear_calls += 1
            return torch_functional.linear(hidden, effective, module.bias)

        return apply

    def __enter__(self) -> "CachedBF16FunctionalTrial":
        if self._handles:
            raise ODEBFContractError("cached BF16 trial is already active")
        try:
            for name in sorted(self.factors_by_weight):
                factors = self.factors_by_weight[name]
                module, parameter = CumulativeBF16FunctionalTrial._resolve_linear(
                    self.model, name
                )
                entry = self.entry_weights[name].detach().to(
                    device=parameter.device,
                    dtype=torch.bfloat16,
                )
                effective, _stats = assemble_effective_bf16(
                    entry,
                    factors,
                    row_block=self.row_block,
                    compute_sha256=False,
                )
                self._effective[name] = effective
                self.dense_assembly_count += 1
                self._parameters.append(
                    (
                        name,
                        parameter,
                        parameter.data_ptr(),
                        parameter._version,
                        parameter.requires_grad,
                    )
                )
                self._handles.append(
                    module.register_forward_hook(self._hook(name))
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        violations: list[str] = []
        for name, parameter, pointer, version, requires_grad in self._parameters:
            if (
                parameter.data_ptr() != pointer
                or parameter._version != version
                or parameter.requires_grad != requires_grad
                or parameter.grad is not None
            ):
                violations.append(name)
        self._parameters.clear()
        self._effective.clear()
        if violations:
            error = ODEBFContractError(
                "cached BF16 trial mutated state: " + ",".join(violations)
            )
            if exc is not None:
                raise error from exc
            raise error
        return False


def independent_native_alpha_bf16_fixture(
    entry_weight: torch.Tensor,
    factor: WaypointFactor,
) -> torch.Tensor:
    """CPU-only source-order fixture; it deliberately materializes a tiny delta."""

    if factor.theta != 1.0:
        raise ODEBFContractError("Native identity fixture requires theta=1")
    right = factor.right.detach().to(device="cpu", dtype=torch.float32)
    left = factor.left.detach().to(device="cpu", dtype=torch.float32)
    source_order = (right @ left.T).T
    return (entry_weight.detach().to(device="cpu").float() + source_order).to(torch.bfloat16)
