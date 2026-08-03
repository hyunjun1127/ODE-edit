"""One-backward all-layer low-rank directional derivatives.

The dense target-weight gradient implementation is a correctness oracle only.
The primary implementation captures module inputs and output gradients, then
contracts them directly with low-rank actuator factors.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch

from project.run_scripts.ode_edit_motivation.hooks import resolve_parameter

from .contracts import MethodContractError, canonical_hash
from .hooks import FactorDirection
from .instrumentation import EditInstrumentation


@dataclass(frozen=True, slots=True)
class DirectionalDerivative:
    layer: int
    direction_id: str
    event_derivative: float
    progress_slope: float


@dataclass(frozen=True, slots=True)
class DirectionalField:
    values: tuple[DirectionalDerivative, ...]
    backward_calls: int
    backend: str
    field_id: str

    @property
    def slopes_by_layer(self) -> dict[int, float]:
        return {value.layer: value.progress_slope for value in self.values}


@contextmanager
def directional_gradient_scope(
    model: torch.nn.Module,
    directions: Sequence[FactorDirection],
) -> Iterator[None]:
    """Enable target gradients for the dense correctness oracle only."""

    locked = tuple(directions)
    if not locked:
        raise MethodContractError("directional gradient scope has no directions")
    parameters = tuple(resolve_parameter(model, item.weight_name) for item in locked)
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise MethodContractError("directional gradient weights repeat")
    flags = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(True)
        yield
    finally:
        for parameter, flag in zip(parameters, flags, strict=True):
            parameter.requires_grad_(flag)


def _low_rank_inner_product(
    gradient: torch.Tensor,
    direction: FactorDirection,
) -> torch.Tensor:
    """Compute ``<gradient, left @ right.T>`` without a dense update."""

    left = direction.left.to(device=gradient.device, dtype=gradient.dtype)
    right = direction.right.to(device=gradient.device, dtype=gradient.dtype)
    if tuple(gradient.shape) != (left.shape[0], right.shape[0]):
        raise MethodContractError("directional derivative weight/factor shapes differ")
    return torch.sum((gradient.transpose(0, 1) @ left) * right)


def all_layer_directional_derivatives(
    smooth_event: torch.Tensor,
    model: torch.nn.Module,
    directions: Sequence[FactorDirection],
    *,
    instrumentation: EditInstrumentation | None = None,
    retain_graph: bool = False,
) -> DirectionalField:
    """Dense-gradient correctness oracle; never a production backend.

    ``torch.autograd.grad`` materializes one target-weight gradient tensor per
    layer.  It is retained only to check the actuator-hook identity on small
    CPU fixtures.
    """

    locked = tuple(directions)
    if smooth_event.ndim != 0 or not smooth_event.requires_grad:
        raise MethodContractError("smooth rewrite event must be a grad-enabled scalar")
    if not locked or len({item.layer for item in locked}) != len(locked):
        raise MethodContractError("directional field layers are empty or repeat")
    parameters = tuple(resolve_parameter(model, item.weight_name) for item in locked)
    if any(not parameter.requires_grad for parameter in parameters):
        raise MethodContractError(
            "target weights require directional_gradient_scope before the forward"
        )
    timer = (
        instrumentation.component("field")
        if instrumentation is not None
        else nullcontext()
    )
    with timer:
        gradients = torch.autograd.grad(
            smooth_event,
            parameters,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=False,
        )
        values = []
        for direction, gradient in zip(locked, gradients, strict=True):
            derivative = float(_low_rank_inner_product(gradient, direction).detach().cpu())
            if not math.isfinite(derivative):
                raise MethodContractError("directional derivative is non-finite")
            values.append(
                DirectionalDerivative(
                    layer=direction.layer,
                    direction_id=direction.direction_id,
                    event_derivative=derivative,
                    progress_slope=max(0.0, -derivative),
                )
            )
    if instrumentation is not None:
        instrumentation.increment("N_bw")
    payload = [
        {
            "layer": value.layer,
            "direction_id": value.direction_id,
            "event_derivative": value.event_derivative,
            "progress_slope": value.progress_slope,
        }
        for value in values
    ]
    return DirectionalField(
        values=tuple(values),
        backward_calls=1,
        backend="dense-target-gradient-reference-only",
        field_id=canonical_hash(payload),
    )


@dataclass(slots=True)
class _ActivationRecord:
    module_input: torch.Tensor
    module_output: torch.Tensor
    input_version: int


def _weight_module(model: torch.nn.Module, weight_name: str) -> torch.nn.Module:
    if not weight_name.endswith(".weight"):
        raise MethodContractError("actuator hook target is not a module weight")
    module_name = weight_name[: -len(".weight")]
    try:
        module = model.get_submodule(module_name)
    except AttributeError as exc:
        raise MethodContractError("actuator hook target module is absent") from exc
    if getattr(module, "weight", None) is not resolve_parameter(model, weight_name):
        raise MethodContractError("actuator hook module/weight identity differs")
    return module


def _activation_low_rank_contraction(
    module_input: torch.Tensor,
    grad_output: torch.Tensor,
    direction: FactorDirection,
) -> torch.Tensor:
    """Compute ``<g, x (U V^T)^T>`` without dense ``B`` or ``grad_W``."""

    if module_input.shape[:-1] != grad_output.shape[:-1]:
        raise MethodContractError("actuator hook input/gradient panels differ")
    compute_dtype = (
        torch.float64
        if module_input.dtype is torch.float64 or grad_output.dtype is torch.float64
        else torch.float32
    )
    x = module_input.reshape(-1, module_input.shape[-1]).to(dtype=compute_dtype)
    g = grad_output.reshape(-1, grad_output.shape[-1]).to(dtype=compute_dtype)
    left = direction.left.to(device=g.device, dtype=compute_dtype)
    right = direction.right.to(device=x.device, dtype=compute_dtype)
    if x.shape[1] != right.shape[0] or g.shape[1] != left.shape[0]:
        raise MethodContractError("actuator hook activation/factor geometry differs")
    # For B = U V^T and a linear module y = x W^T,
    # dPhi/dalpha = sum_rows (g U) * (x V).
    return torch.sum((g @ left) * (x @ right))


class ActuatorDirectionalHook:
    """Capture ``x`` and ``g`` for every actuator in one rewrite backward.

    Target weights have ``requires_grad=False`` during the forward/backward;
    their ``.grad`` slots, storage pointers, and versions must remain unchanged.
    Repeated module calls (for multiple allowed-context panels) are summed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        directions: Sequence[FactorDirection],
        *,
        instrumentation: EditInstrumentation | None = None,
    ) -> None:
        self.model = model
        self.directions = tuple(directions)
        if not self.directions:
            raise MethodContractError("actuator hook has no directions")
        layers = tuple(item.layer for item in self.directions)
        names = tuple(item.weight_name for item in self.directions)
        if layers != tuple(sorted(layers)) or len(set(layers)) != len(layers):
            raise MethodContractError("actuator hook layers must be unique and ascending")
        if len(set(names)) != len(names):
            raise MethodContractError("actuator hook target weights repeat")
        self.instrumentation = instrumentation
        self._records = {name: [] for name in names}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._parameters: tuple[torch.nn.Parameter, ...] = ()
        self._requires_grad: tuple[bool, ...] = ()
        self._data_ptrs: tuple[int, ...] = ()
        self._versions: tuple[int, ...] = ()
        self._active = False
        self._computed = False

    def _forward_hook(self, weight_name: str) -> Any:
        def capture(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> torch.Tensor | None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise MethodContractError("actuator module has no tensor input")
            if not isinstance(output, torch.Tensor):
                raise MethodContractError("actuator module has non-tensor output")
            observed = output
            replacement: torch.Tensor | None = None
            if not observed.requires_grad:
                replacement = observed.detach().requires_grad_(True)
                observed = replacement
            self._records[weight_name].append(
                _ActivationRecord(
                    module_input=inputs[0].detach(),
                    module_output=observed,
                    input_version=inputs[0]._version,
                )
            )
            return replacement

        return capture

    def __enter__(self) -> "ActuatorDirectionalHook":
        if self._active:
            raise RuntimeError("actuator directional hook is already active")
        parameters = tuple(
            resolve_parameter(self.model, item.weight_name) for item in self.directions
        )
        if any(parameter.grad is not None for parameter in parameters):
            raise MethodContractError("target weight .grad must be None before actuator hook")
        self._parameters = parameters
        self._requires_grad = tuple(parameter.requires_grad for parameter in parameters)
        self._data_ptrs = tuple(parameter.data_ptr() for parameter in parameters)
        self._versions = tuple(parameter._version for parameter in parameters)
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            for direction in self.directions:
                module = _weight_module(self.model, direction.weight_name)
                self._handles.append(
                    module.register_forward_hook(
                        self._forward_hook(direction.weight_name)
                    )
                )
        except BaseException:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            for parameter, flag in zip(
                parameters, self._requires_grad, strict=True
            ):
                parameter.requires_grad_(flag)
            raise
        self._active = True
        return self

    def compute(
        self,
        smooth_event: torch.Tensor,
        *,
        retain_graph: bool = False,
    ) -> DirectionalField:
        if not self._active or self._computed:
            raise MethodContractError("actuator hook compute requires one active capture")
        if smooth_event.ndim != 0 or not smooth_event.requires_grad:
            raise MethodContractError("smooth rewrite event must be a grad-enabled scalar")
        records: list[tuple[FactorDirection, _ActivationRecord]] = []
        for direction in self.directions:
            observed = self._records[direction.weight_name]
            if not observed:
                raise MethodContractError("target actuator was not called by the state forward")
            records.extend((direction, record) for record in observed)
        timer = (
            self.instrumentation.component("field")
            if self.instrumentation is not None
            else nullcontext()
        )
        with timer:
            gradients = torch.autograd.grad(
                smooth_event,
                tuple(record.module_output for _direction, record in records),
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=False,
            )
            totals = {direction.layer: 0.0 for direction in self.directions}
            for (direction, record), gradient in zip(records, gradients, strict=True):
                if record.module_input._version != record.input_version:
                    raise MethodContractError("captured actuator input changed in place")
                contribution = _activation_low_rank_contraction(
                    record.module_input,
                    gradient,
                    direction,
                )
                totals[direction.layer] += float(contribution.detach().cpu())
        if self.instrumentation is not None:
            self.instrumentation.increment("N_bw")
        values = tuple(
            DirectionalDerivative(
                layer=direction.layer,
                direction_id=direction.direction_id,
                event_derivative=totals[direction.layer],
                progress_slope=max(0.0, -totals[direction.layer]),
            )
            for direction in self.directions
        )
        if any(not math.isfinite(value.event_derivative) for value in values):
            raise MethodContractError("actuator directional derivative is non-finite")
        payload = [
            {
                "layer": value.layer,
                "direction_id": value.direction_id,
                "event_derivative": value.event_derivative,
                "progress_slope": value.progress_slope,
            }
            for value in values
        ]
        self._computed = True
        return DirectionalField(
            values=values,
            backward_calls=1,
            backend="activation-actuator-hook",
            field_id=canonical_hash(payload),
        )

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        violations: list[str] = []
        for parameter, flag, pointer, version, direction in zip(
            self._parameters,
            self._requires_grad,
            self._data_ptrs,
            self._versions,
            self.directions,
            strict=True,
        ):
            if parameter.grad is not None:
                violations.append(f"{direction.weight_name}: materialized .grad")
            if parameter.data_ptr() != pointer:
                violations.append(f"{direction.weight_name}: storage pointer changed")
            if parameter._version != version:
                violations.append(f"{direction.weight_name}: parameter version changed")
            parameter.requires_grad_(flag)
        self._active = False
        if violations:
            raise MethodContractError("actuator hook mutation: " + "; ".join(violations))
        return False
