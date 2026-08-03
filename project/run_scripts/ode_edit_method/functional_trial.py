"""Dense-weight-copy-free low-rank functional trial hooks."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

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
    """Evaluate ``W + sum y_l L_l R_l^T`` without mutating or copying W.

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
            left = direction.left.to(device=hidden.device, dtype=hidden.dtype)
            right = direction.right.to(device=hidden.device, dtype=hidden.dtype)
            if hidden.shape[-1] != right.shape[0] or output.shape[-1] != left.shape[0]:
                raise MethodContractError("functional trial activation/factor shapes differ")
            delta = (hidden @ right) @ left.transpose(0, 1)
            return output + delta.to(output.dtype) * coefficient

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
