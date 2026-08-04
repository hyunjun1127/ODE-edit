"""Separated FP32-gradient and exact BF16-verdict functional paths."""

from __future__ import annotations

import math
from contextlib import AbstractContextManager
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .contracts import ODEAllocContractError
from .gauge import FactorPair


def quantized_effective_weight(
    base_weight: torch.Tensor,
    pair: FactorPair,
    ratio: float,
) -> torch.Tensor:
    """Materialize one effective BF16 layer, never a retained dense FP32 delta."""

    coefficient = float(ratio)
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ODEAllocContractError("allocation ratio must be finite and positive")
    if base_weight.dtype is not torch.bfloat16 or base_weight.ndim != 2:
        raise ODEAllocContractError("verdict base weight must be a BF16 matrix")
    if tuple(base_weight.shape) != (pair.left.shape[0], pair.right.shape[0]):
        raise ODEAllocContractError("verdict factor and weight shapes differ")
    with torch.no_grad():
        # This is the sole dense FP32 temporary: the effective weight itself.
        # addmm_ avoids materializing a separate full-rank delta tensor.
        effective_fp32 = base_weight.detach().to(dtype=torch.float32)
        effective_fp32.addmm_(
            pair.left.detach().to(device=base_weight.device, dtype=torch.float32),
            pair.right.detach()
            .to(device=base_weight.device, dtype=torch.float32)
            .transpose(0, 1),
            alpha=coefficient,
        )
        effective_bf16 = effective_fp32.to(dtype=torch.bfloat16)
        del effective_fp32
    return effective_bf16


class QuantizedBF16FunctionalTrial(AbstractContextManager["QuantizedBF16FunctionalTrial"]):
    """Read-only full-linear emulator with at most one effective layer live."""

    def __init__(
        self,
        model: torch.nn.Module,
        factors_by_weight: Mapping[str, FactorPair],
        ratios_by_layer: Mapping[int, float],
    ) -> None:
        if not factors_by_weight:
            raise ODEAllocContractError("BF16 trial basis is empty")
        self.model = model
        self.factors_by_weight = dict(factors_by_weight)
        self.ratios_by_layer = dict(ratios_by_layer)
        if {pair.layer for pair in self.factors_by_weight.values()} != set(self.ratios_by_layer):
            raise ODEAllocContractError("BF16 trial ratios and factor layers differ")
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._parameters: list[tuple[str, torch.nn.Parameter, int, int, bool]] = []
        self._cpu_rng: torch.Tensor | None = None
        self._cuda_rng: dict[torch.device, torch.Tensor] = {}
        self._active_effective_weights = 0
        self.max_live_effective_weights = 0

    @staticmethod
    def _module_and_parameter(
        model: torch.nn.Module,
        weight_name: str,
    ) -> tuple[torch.nn.Linear, torch.nn.Parameter]:
        if not weight_name.endswith(".weight"):
            raise ODEAllocContractError("BF16 trial target must end in .weight")
        module_name = weight_name[: -len(".weight")]
        try:
            module = model.get_submodule(module_name)
            parameter = dict(model.named_parameters())[weight_name]
        except (AttributeError, KeyError) as exc:
            raise ODEAllocContractError("BF16 trial target is absent") from exc
        if type(module) is not torch.nn.Linear or module.weight is not parameter:
            raise ODEAllocContractError("BF16 verdict requires an exact Linear target")
        return module, parameter

    def _hook(self, pair: FactorPair, ratio: float):
        def apply(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> torch.Tensor:
            del output
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise ODEAllocContractError("BF16 verdict Linear has no tensor input")
            hidden = inputs[0]
            if hidden.dtype is not torch.bfloat16:
                raise ODEAllocContractError("BF16 verdict activation is not BF16")
            self._active_effective_weights += 1
            self.max_live_effective_weights = max(
                self.max_live_effective_weights, self._active_effective_weights
            )
            try:
                effective = quantized_effective_weight(module.weight, pair, ratio)
                with torch.no_grad():
                    result = F.linear(hidden, effective, module.bias)
                del effective
                return result
            finally:
                self._active_effective_weights -= 1

        return apply

    def __enter__(self) -> "QuantizedBF16FunctionalTrial":
        if self._handles:
            raise ODEAllocContractError("BF16 trial is already active")
        self._cpu_rng = torch.get_rng_state().clone()
        try:
            for name, pair in self.factors_by_weight.items():
                module, parameter = self._module_and_parameter(self.model, name)
                if parameter.dtype is not torch.bfloat16 or parameter.grad is not None:
                    raise ODEAllocContractError("BF16 trial parameter state is invalid")
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
                    module.register_forward_hook(
                        self._hook(pair, float(self.ratios_by_layer[pair.layer]))
                    )
                )
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
            violations.append("dense-effective-weight-liveness")
        if violations:
            error = ODEAllocContractError(
                "BF16 functional trial mutated state: " + ",".join(violations)
            )
            if exc is not None:
                raise error from exc
            raise error
        return False


class FP32FactorOverlay(AbstractContextManager["FP32FactorOverlay"]):
    """Differentiable low-rank activation overlay; never valid for verdicts."""

    verdict_eligible = False

    def __init__(
        self,
        model: torch.nn.Module,
        factors_by_weight: Mapping[str, FactorPair],
        ratios_by_layer: Mapping[int, torch.Tensor],
    ) -> None:
        self.model = model
        self.factors_by_weight = dict(factors_by_weight)
        self.ratios_by_layer = dict(ratios_by_layer)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._parameters: list[tuple[str, torch.nn.Parameter, int, int, bool]] = []
        self._cpu_rng: torch.Tensor | None = None
        self._cuda_rng: dict[torch.device, torch.Tensor] = {}

    @staticmethod
    def _hook(pair: FactorPair, ratio: torch.Tensor):
        def apply(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> torch.Tensor:
            del module
            if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                raise ODEAllocContractError("FP32 overlay requires tensor Linear IO")
            hidden = inputs[0].to(dtype=torch.float32)
            left = pair.left.to(device=hidden.device, dtype=torch.float32)
            right = pair.right.to(device=hidden.device, dtype=torch.float32)
            tangent = (hidden @ right) @ left.transpose(0, 1)
            return output + (ratio.to(device=output.device, dtype=torch.float32) * tangent).to(output.dtype)

        return apply

    def __enter__(self) -> "FP32FactorOverlay":
        if self._handles:
            raise ODEAllocContractError("FP32 overlay is already active")
        if {pair.layer for pair in self.factors_by_weight.values()} != set(self.ratios_by_layer):
            raise ODEAllocContractError("FP32 overlay ratios and factors differ")
        self._cpu_rng = torch.get_rng_state().clone()
        try:
            for name, pair in self.factors_by_weight.items():
                module, parameter = QuantizedBF16FunctionalTrial._module_and_parameter(
                    self.model, name
                )
                if parameter.grad is not None:
                    raise ODEAllocContractError("FP32 overlay target .grad must be None")
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
                    module.register_forward_hook(
                        self._hook(pair, self.ratios_by_layer[pair.layer])
                    )
                )
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
        if violations:
            error = ODEAllocContractError(
                "FP32 factor overlay mutated state: " + ",".join(violations)
            )
            if exc is not None:
                raise error from exc
            raise error
        return False


def independent_native_weight_cpu_fixture(
    base_weight: torch.Tensor,
    pair: FactorPair,
) -> torch.Tensor:
    """Independent tiny CPU oracle used only to test q=0 accepted bytes."""

    if base_weight.device.type != "cpu" or base_weight.numel() > 4096:
        raise ODEAllocContractError("dense Native oracle is restricted to tiny CPU fixtures")
    with torch.no_grad():
        dense_delta = pair.left.float() @ pair.right.float().transpose(0, 1)
        result = (base_weight.float() + dense_delta).to(torch.bfloat16)
        del dense_delta
    return result
