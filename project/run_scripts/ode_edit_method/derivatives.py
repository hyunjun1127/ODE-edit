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
        instrumentation.component("backward_hook")
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
    """Contract the canonical direct output-space continuous tangent.

    Actual BF16/FP16 model activations use a float32 tangent.  A float64
    tangent is retained only for the CPU correctness oracle.  This creates an
    output-panel temporary, never a dense target-weight tensor or gradient.
    """

    if module_input.shape[:-1] != grad_output.shape[:-1]:
        raise MethodContractError("actuator hook input/gradient panels differ")
    compute_dtype = (
        torch.float64
        if module_input.dtype is torch.float64 or grad_output.dtype is torch.float64
        else torch.float32
    )
    tangent_input = module_input.reshape(-1, module_input.shape[-1]).to(
        dtype=compute_dtype
    )
    tangent_gradient = grad_output.reshape(-1, grad_output.shape[-1]).to(
        dtype=compute_dtype
    )
    left = direction.left.to(device=tangent_gradient.device, dtype=compute_dtype)
    right = direction.right.to(device=tangent_input.device, dtype=compute_dtype)
    if (
        tangent_input.shape[1] != right.shape[0]
        or tangent_gradient.shape[1] != left.shape[0]
    ):
        raise MethodContractError("actuator hook activation/factor geometry differs")
    projected_input = tangent_input @ right
    output_delta = projected_input @ left.transpose(0, 1)
    return torch.sum(tangent_gradient * output_delta)


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
            self.instrumentation.component("backward_hook")
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


class ScalarGateDirectionalReference:
    """Independent low-rank functional-graph derivative oracle.

    Each layer owns one float32 scalar ``alpha``.  The forward hook reproduces
    the canonical continuous tangent at ``alpha=0``: activations and factors
    use float32 for BF16/FP16 models (float64 only in the CPU oracle), alpha is
    multiplied before one final output-dtype cast, and the zero delta is added
    to the module output.  One
    ``torch.autograd.grad`` call then obtains all ``dPhi/dalpha`` values.

    This path never differentiates a target weight and shares neither captured
    activation gradients nor the contraction implementation used by
    :class:`ActuatorDirectionalHook`.
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
            raise MethodContractError("scalar-gate reference has no directions")
        names = tuple(direction.weight_name for direction in self.directions)
        layers = tuple(direction.layer for direction in self.directions)
        if len(set(names)) != len(names) or len(set(layers)) != len(layers):
            raise MethodContractError("scalar-gate weights or layers repeat")
        if layers != tuple(sorted(layers)):
            raise MethodContractError("scalar-gate layers must be ascending")
        self.instrumentation = instrumentation
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._parameters: tuple[torch.nn.Parameter, ...] = ()
        self._requires_grad: tuple[bool, ...] = ()
        self._pointers: tuple[int, ...] = ()
        self._versions: tuple[int, ...] = ()
        self._alphas: tuple[torch.Tensor, ...] = ()
        self._active = False
        self._computed = False

    @staticmethod
    def _hook(direction: FactorDirection, alpha: torch.Tensor) -> Any:
        def apply(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> torch.Tensor:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise MethodContractError("scalar-gate module has no tensor input")
            if not isinstance(output, torch.Tensor):
                raise MethodContractError("scalar-gate module has non-tensor output")
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
                raise MethodContractError("scalar-gate activation/factor shapes differ")
            output_delta = (tangent_input @ right) @ left.transpose(0, 1)
            tangent_alpha = alpha.to(device=output.device, dtype=tangent_dtype)
            gated = output + (tangent_alpha * output_delta).to(dtype=output.dtype)
            if gated.dtype != output.dtype or not torch.equal(
                gated.detach(), output.detach()
            ):
                raise MethodContractError(
                    "scalar-gate alpha=0 changed module output identity"
                )
            return gated

        return apply

    def __enter__(self) -> "ScalarGateDirectionalReference":
        if self._active:
            raise RuntimeError("scalar-gate reference is already active")
        parameters = tuple(
            resolve_parameter(self.model, direction.weight_name)
            for direction in self.directions
        )
        if any(parameter.grad is not None for parameter in parameters):
            raise MethodContractError("target weight .grad must be None before scalar gate")
        self._parameters = parameters
        self._requires_grad = tuple(parameter.requires_grad for parameter in parameters)
        self._pointers = tuple(parameter.data_ptr() for parameter in parameters)
        self._versions = tuple(parameter._version for parameter in parameters)
        self._alphas = tuple(
            torch.zeros(
                (),
                device=parameter.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            for parameter in parameters
        )
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            for direction, alpha in zip(
                self.directions, self._alphas, strict=True
            ):
                module = _weight_module(self.model, direction.weight_name)
                self._handles.append(
                    module.register_forward_hook(self._hook(direction, alpha))
                )
        except BaseException:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            for parameter, flag in zip(
                parameters, self._requires_grad, strict=True
            ):
                parameter.requires_grad_(flag)
            self._alphas = ()
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
            raise MethodContractError("scalar-gate compute requires one active capture")
        if smooth_event.ndim != 0 or not smooth_event.requires_grad:
            raise MethodContractError("smooth rewrite event must be a grad-enabled scalar")
        timer = (
            self.instrumentation.component("reference_gate")
            if self.instrumentation is not None
            else nullcontext()
        )
        with timer:
            gradients = torch.autograd.grad(
                smooth_event,
                self._alphas,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=False,
            )
        if self.instrumentation is not None:
            self.instrumentation.increment("N_reference_gate_bw")
        values = []
        for direction, gradient in zip(self.directions, gradients, strict=True):
            derivative = float(gradient.detach().cpu())
            if not math.isfinite(derivative):
                raise MethodContractError("scalar-gate derivative is non-finite")
            values.append(
                DirectionalDerivative(
                    layer=direction.layer,
                    direction_id=direction.direction_id,
                    event_derivative=derivative,
                    progress_slope=max(0.0, -derivative),
                )
            )
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
            values=tuple(values),
            backward_calls=1,
            backend="scalar-gate-functional-graph-reference-only",
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
            self._pointers,
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
            if parameter.requires_grad is not False:
                violations.append(f"{direction.weight_name}: requires_grad changed")
            parameter.requires_grad_(flag)
        if any(alpha.grad is not None for alpha in self._alphas):
            violations.append("scalar alpha .grad was materialized")
        self._parameters = ()
        self._requires_grad = ()
        self._pointers = ()
        self._versions = ()
        self._alphas = ()
        self._active = False
        if violations:
            raise MethodContractError(
                "scalar-gate reference mutation: " + "; ".join(violations)
            )
        return False


def assert_scalar_gate_matches_hook(
    primary: DirectionalField,
    reference: DirectionalField,
    *,
    abs_tol: float,
    rel_tol: float,
) -> tuple[dict[str, float | int | str], ...]:
    """Fail closed when independent A/B directional paths disagree."""

    if (
        not math.isfinite(abs_tol)
        or not math.isfinite(rel_tol)
        or abs_tol <= 0.0
        or rel_tol <= 0.0
    ):
        raise MethodContractError("scalar-gate identity tolerances are invalid")
    if len(primary.values) != len(reference.values):
        raise MethodContractError("scalar-gate field cardinality differs")
    rows = []
    for hook_value, gate_value in zip(
        primary.values, reference.values, strict=True
    ):
        if (
            hook_value.layer != gate_value.layer
            or hook_value.direction_id != gate_value.direction_id
        ):
            raise MethodContractError("scalar-gate field identity differs")
        absolute_error = abs(
            hook_value.event_derivative - gate_value.event_derivative
        )
        if not math.isclose(
            hook_value.event_derivative,
            gate_value.event_derivative,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        ):
            raise MethodContractError(
                "actuator hook differs from scalar-gate hard reference: "
                f"layer={hook_value.layer} "
                f"hook={hook_value.event_derivative:.17g} "
                f"scalar_gate={gate_value.event_derivative:.17g} "
                f"abs_error={absolute_error:.17g}"
            )
        rows.append(
            {
                "layer": hook_value.layer,
                "direction_id": hook_value.direction_id,
                "hook_derivative": hook_value.event_derivative,
                "scalar_gate_derivative": gate_value.event_derivative,
                "absolute_error": absolute_error,
            }
        )
    return tuple(rows)
