"""Dense-weight-copy-free low-rank functional trial hooks."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from project.run_scripts.ode_edit_motivation.hooks import resolve_parameter

from .contracts import MethodContractError
from .hooks import FactorDirection


def _module_for_weight(model: torch.nn.Module, weight_name: str) -> torch.nn.Module:
    suffix = ".weight"
    if not weight_name.endswith(suffix):
        raise MethodContractError("functional trial requires a module '.weight' target")
    module_name = weight_name[: -len(suffix)]
    try:
        module = model.get_submodule(module_name)
    except AttributeError as exc:
        raise MethodContractError("functional trial target module is absent") from exc
    if getattr(module, "weight", None) is not resolve_parameter(model, weight_name):
        raise MethodContractError("functional trial module/parameter identity differs")
    return module


class LowRankFunctionalTrial:
    """Continuous-overlay reference retained for tests and diagnostics only.

    Evaluate ``W + sum y_l L_l R_l^T`` without mutating or copying W.  This
    arithmetic is not the production finite-trial backend because it does not
    reproduce parameter-dtype accepted-write quantization.

    This context is intentionally read-only.  An accepted coefficient must be
    handed to a separate exact write transaction; calling ``commit`` here is a
    contract error, preventing a functional probe from being mistaken for a
    persisted edit.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        directions: Sequence[FactorDirection],
        coefficients: Sequence[float],
    ) -> None:
        self.model = model
        self.directions = tuple(directions)
        self.applied_coefficients = tuple(float(value) for value in coefficients)
        if (
            not self.directions
            or len(self.directions) != len(self.applied_coefficients)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.applied_coefficients
            )
        ):
            raise MethodContractError("functional trial directions/coefficients are invalid")
        names = tuple(item.weight_name for item in self.directions)
        if len(names) != len(set(names)):
            raise MethodContractError("functional trial weights repeat")
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._parameters: tuple[torch.nn.Parameter, ...] = ()
        self._pointers: tuple[int, ...] = ()
        self._versions: tuple[int, ...] = ()
        self._cpu_rng: torch.Tensor | None = None
        self._cuda_rng: dict[torch.device, torch.Tensor] = {}
        self._active = False

    @staticmethod
    def _hook(
        direction: FactorDirection,
        coefficient: float,
    ) -> Any:
        def apply(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> torch.Tensor:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise MethodContractError("functional trial module has no tensor input")
            if not isinstance(output, torch.Tensor):
                raise MethodContractError("functional trial module has non-tensor output")
            hidden = inputs[0]
            tangent_dtype = (
                torch.float64
                if hidden.dtype is torch.float64 or output.dtype is torch.float64
                else torch.float32
            )
            tangent_input = hidden.to(dtype=tangent_dtype)
            left = direction.left.to(device=hidden.device, dtype=tangent_dtype)
            right = direction.right.to(device=hidden.device, dtype=tangent_dtype)
            if hidden.shape[-1] != right.shape[0] or output.shape[-1] != left.shape[0]:
                raise MethodContractError("functional trial activation/factor shapes differ")
            output_delta = (tangent_input @ right) @ left.transpose(0, 1)
            tangent_coefficient = torch.as_tensor(
                coefficient,
                device=output.device,
                dtype=tangent_dtype,
            )
            return output + (tangent_coefficient * output_delta).to(output.dtype)

        return apply

    def __enter__(self) -> "LowRankFunctionalTrial":
        if self._active:
            raise RuntimeError("functional trial is already active")
        try:
            parameters = tuple(
                resolve_parameter(self.model, direction.weight_name)
                for direction in self.directions
            )
            if any(parameter.grad is not None for parameter in parameters):
                raise MethodContractError("functional trial target weight .grad must be None")
            self._parameters = parameters
            self._pointers = tuple(parameter.data_ptr() for parameter in parameters)
            self._versions = tuple(parameter._version for parameter in parameters)
            self._cpu_rng = torch.get_rng_state().clone()
            cuda_devices = {
                parameter.device
                for parameter in parameters
                if parameter.device.type == "cuda"
            }
            self._cuda_rng = {
                device: torch.cuda.get_rng_state(device).clone()
                for device in cuda_devices
            }
            for direction, coefficient in zip(
                self.directions, self.applied_coefficients, strict=True
            ):
                parameter = resolve_parameter(self.model, direction.weight_name)
                if tuple(parameter.shape) != (
                    direction.left.shape[0],
                    direction.right.shape[0],
                ):
                    raise MethodContractError("functional trial factor/weight shapes differ")
                module = _module_for_weight(self.model, direction.weight_name)
                self._handles.append(
                    module.register_forward_hook(self._hook(direction, coefficient))
                )
        except BaseException:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            raise
        self._active = True
        return self

    def commit(self) -> None:
        raise MethodContractError(
            "functional trial is read-only; persist with an exact accepted-write transaction"
        )

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        violations = []
        for parameter, pointer, version, direction in zip(
            self._parameters,
            self._pointers,
            self._versions,
            self.directions,
            strict=True,
        ):
            if parameter.data_ptr() != pointer:
                violations.append(f"{direction.weight_name}: storage pointer changed")
            if parameter._version != version:
                violations.append(f"{direction.weight_name}: parameter version changed")
        if self._cpu_rng is not None:
            torch.set_rng_state(self._cpu_rng)
        for device, state in self._cuda_rng.items():
            torch.cuda.set_rng_state(state, device)
        self._parameters = ()
        self._pointers = ()
        self._versions = ()
        self._cpu_rng = None
        self._cuda_rng = {}
        self._active = False
        if violations:
            raise MethodContractError("functional trial mutation: " + "; ".join(violations))
        return False


class QuantizedRowBlockFunctionalTrial:
    """Read-only emulator of the quantized accepted-write Linear semantics.

    The continuous FP32 tangent is deliberately *not* reused for a finite
    coefficient.  Instead, every output-row block repeats the accepted writer's
    product/multiply/cast/add order and immediately evaluates that effective
    block with :func:`torch.nn.functional.linear`.  No full effective weight or
    dense update is materialized and the target parameter is never mutated.

    This primitive is wired as simple-T for adaptive finite candidates in the
    Session 02 technical/method runtime.  That wiring is not itself evidence of
    scientific superiority.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        directions: Sequence[FactorDirection],
        coefficients: Sequence[float],
        *,
        row_block: int = 64,
    ) -> None:
        self.model = model
        self.directions = tuple(directions)
        self.applied_coefficients = tuple(float(value) for value in coefficients)
        self.row_block = row_block
        if (
            not self.directions
            or len(self.directions) != len(self.applied_coefficients)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.applied_coefficients
            )
            or isinstance(row_block, bool)
            or not isinstance(row_block, int)
            or row_block <= 0
        ):
            raise MethodContractError(
                "quantized trial directions/coefficients/row block are invalid"
            )
        names = tuple(direction.weight_name for direction in self.directions)
        if len(names) != len(set(names)):
            raise MethodContractError("quantized trial weights repeat")
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._parameters: tuple[torch.nn.Parameter, ...] = ()
        self._requires_grad: tuple[bool, ...] = ()
        self._pointers: tuple[int, ...] = ()
        self._versions: tuple[int, ...] = ()
        self._cpu_rng: torch.Tensor | None = None
        self._cuda_rng: dict[torch.device, torch.Tensor] = {}
        self._active = False
        self._max_effective_weight_elements = 0
        self._max_output_block_elements = 0

    @property
    def temporary_shape_contract(self) -> dict[str, int]:
        """Return observed block bounds without retaining any trial tensor."""

        return {
            "row_block": self.row_block,
            "max_effective_weight_elements": self._max_effective_weight_elements,
            "max_output_block_elements": self._max_output_block_elements,
        }

    def _hook(self, direction: FactorDirection, coefficient: float) -> Any:
        def apply(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> torch.Tensor:
            if type(module) is not torch.nn.Linear:
                raise MethodContractError(
                    "quantized trial requires an exact torch.nn.Linear module"
                )
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise MethodContractError("quantized trial module has no tensor input")
            if not isinstance(output, torch.Tensor):
                raise MethodContractError("quantized trial module has non-tensor output")
            hidden = inputs[0]
            weight = module.weight
            bias = module.bias
            if (
                weight.ndim != 2
                or hidden.shape[-1] != weight.shape[1]
                or output.shape[-1] != weight.shape[0]
                or (bias is not None and tuple(bias.shape) != (weight.shape[0],))
            ):
                raise MethodContractError("quantized trial Linear geometry differs")
            if tuple(weight.shape) != (
                direction.left.shape[0],
                direction.right.shape[0],
            ):
                raise MethodContractError("quantized trial factor/weight shapes differ")

            compute_dtype = (
                torch.float64 if weight.dtype is torch.float64 else torch.float32
            )
            left = direction.left.detach().to(
                device=weight.device,
                dtype=compute_dtype,
            )
            right_t = direction.right.detach().to(
                device=weight.device,
                dtype=compute_dtype,
            ).transpose(0, 1)
            detached_weight = weight.detach()
            detached_bias = None if bias is None else bias.detach()
            block_outputs: list[torch.Tensor] = []
            for start in range(0, int(weight.shape[0]), self.row_block):
                end = min(start + self.row_block, int(weight.shape[0]))
                update_block = left[start:end] @ right_t
                update_block.mul_(coefficient)
                update_block = update_block.to(dtype=weight.dtype)
                effective_weight_block = detached_weight[start:end] + update_block
                bias_block = (
                    None
                    if detached_bias is None
                    else detached_bias[start:end]
                )
                output_block = F.linear(
                    hidden,
                    effective_weight_block,
                    bias_block,
                )
                self._max_effective_weight_elements = max(
                    self._max_effective_weight_elements,
                    effective_weight_block.numel(),
                )
                self._max_output_block_elements = max(
                    self._max_output_block_elements,
                    output_block.numel(),
                )
                block_outputs.append(output_block)
                del update_block, effective_weight_block
            replacement = torch.cat(block_outputs, dim=-1)
            if replacement.shape != output.shape or replacement.dtype != output.dtype:
                raise MethodContractError("quantized trial output identity differs")
            if coefficient == 0.0 and not torch.equal(
                replacement.detach(),
                output.detach(),
            ):
                raise MethodContractError(
                    "quantized trial coefficient=0 changed Linear output"
                )
            return replacement

        return apply

    def __enter__(self) -> "QuantizedRowBlockFunctionalTrial":
        if self._active:
            raise RuntimeError("quantized row-block trial is already active")
        parameters = tuple(
            resolve_parameter(self.model, direction.weight_name)
            for direction in self.directions
        )
        if any(parameter.grad is not None for parameter in parameters):
            raise MethodContractError("quantized trial target weight .grad must be None")
        self._parameters = parameters
        self._requires_grad = tuple(
            parameter.requires_grad for parameter in parameters
        )
        self._pointers = tuple(parameter.data_ptr() for parameter in parameters)
        self._versions = tuple(parameter._version for parameter in parameters)
        self._cpu_rng = torch.get_rng_state().clone()
        cuda_devices = {
            parameter.device
            for parameter in parameters
            if parameter.device.type == "cuda"
        }
        self._cuda_rng = {
            device: torch.cuda.get_rng_state(device).clone()
            for device in cuda_devices
        }
        try:
            for direction, coefficient in zip(
                self.directions,
                self.applied_coefficients,
                strict=True,
            ):
                module = _module_for_weight(self.model, direction.weight_name)
                if type(module) is not torch.nn.Linear:
                    raise MethodContractError(
                        "quantized trial requires an exact torch.nn.Linear module"
                    )
                self._handles.append(
                    module.register_forward_hook(self._hook(direction, coefficient))
                )
        except BaseException:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            self._parameters = ()
            self._requires_grad = ()
            self._pointers = ()
            self._versions = ()
            self._cpu_rng = None
            self._cuda_rng = {}
            raise
        self._active = True
        return self

    def commit(self) -> None:
        raise MethodContractError(
            "quantized trial is read-only; persist with the accepted-write transaction"
        )

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        violations = []
        for parameter, requires_grad, pointer, version, direction in zip(
            self._parameters,
            self._requires_grad,
            self._pointers,
            self._versions,
            self.directions,
            strict=True,
        ):
            if parameter.data_ptr() != pointer:
                violations.append(f"{direction.weight_name}: storage pointer changed")
            if parameter._version != version:
                violations.append(f"{direction.weight_name}: parameter version changed")
            if parameter.grad is not None:
                violations.append(f"{direction.weight_name}: materialized .grad")
            if parameter.requires_grad != requires_grad:
                violations.append(f"{direction.weight_name}: requires_grad changed")
        if self._cpu_rng is not None:
            torch.set_rng_state(self._cpu_rng)
        for device, state in self._cuda_rng.items():
            torch.cuda.set_rng_state(state, device)
        self._parameters = ()
        self._requires_grad = ()
        self._pointers = ()
        self._versions = ()
        self._cpu_rng = None
        self._cuda_rng = {}
        self._active = False
        if violations:
            raise MethodContractError(
                "quantized trial mutation: " + "; ".join(violations)
            )
        return False
