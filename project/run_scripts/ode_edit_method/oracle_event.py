"""Common direct-z oracle and uniform-mean two-constraint event.

The direct-z intervention is an activation-only calibration.  It never
mutates a parameter, and the resulting target is shared by every arm for one
outer edit.  Runtime decisions use only the two aggregate deficits; individual
context margins remain available to post-hoc observability code.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Mapping, Sequence

import torch

from project.run_scripts.ode_edit_motivation.direct_z import FrozenDirectZ
from project.run_scripts.ode_edit_motivation.hooks import resolve_module

from .contracts import EventReading, MethodContractError
from .events import (
    EVENT_MODEL_FORWARD_CALLS,
    ControllerRequest,
    DifferentiableEvent,
    build_allowed_contexts,
    build_teacher_batch,
    event_from_log_likelihoods,
    score_teacher_batch_tensor,
)


ORACLE_MEAN_EVENT_MODE = "direct-z-oracle-uniform-mean-two-constraint"
ORACLE_REALIZATION_FRACTION = 0.5
ORACLE_MODEL_FORWARD_CALLS = 2


def _finite(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodContractError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise MethodContractError(f"{name} must be finite")
    return result


def _panel_means(
    target_new: Sequence[float], target_old: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...], float, float, float]:
    new = tuple(_finite("target-new log likelihood", value) for value in target_new)
    old = tuple(_finite("target-old log likelihood", value) for value in target_old)
    if not new or len(new) != len(old):
        raise MethodContractError("oracle event panels are empty or misaligned")
    count = len(new)
    mean_new = math.fsum(new) / count
    mean_old = math.fsum(old) / count
    # The canonical relative objective is the uniform mean of context-wise
    # margins, not a differently rounded subtraction of two means.
    mean_margin = math.fsum(
        new_value - old_value
        for new_value, old_value in zip(new, old, strict=True)
    ) / count
    return new, old, mean_new, mean_old, mean_margin


@dataclass(frozen=True, slots=True)
class OracleMeanEventTarget:
    """One edit-local target calibrated from its entry and direct-z oracle."""

    context_count: int
    rho: float
    epsilon: float
    entry_mean_new: float
    entry_mean_old: float
    entry_mean_margin: float
    oracle_mean_new: float
    oracle_mean_old: float
    oracle_mean_margin: float
    required_mean_new: float
    required_mean_margin: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.context_count, bool)
            or not isinstance(self.context_count, int)
            or self.context_count <= 0
        ):
            raise MethodContractError("oracle context count is invalid")
        for name in (
            "rho",
            "epsilon",
            "entry_mean_new",
            "entry_mean_old",
            "entry_mean_margin",
            "oracle_mean_new",
            "oracle_mean_old",
            "oracle_mean_margin",
            "required_mean_new",
            "required_mean_margin",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.rho != ORACLE_REALIZATION_FRACTION:
            raise MethodContractError("oracle realization fraction is not the common lock")
        if self.epsilon <= 0.0:
            raise MethodContractError("oracle epsilon must be positive")
        if self.oracle_mean_margin <= 0.0:
            raise MethodContractError("oracle mean margin is not positive")
        if self.oracle_mean_new <= self.entry_mean_new + self.epsilon:
            raise MethodContractError("oracle target-new likelihood is insufficient")
        expected_margin = self.rho * self.oracle_mean_margin
        expected_new = self.entry_mean_new + self.rho * (
            self.oracle_mean_new - self.entry_mean_new
        )
        if not math.isclose(
            self.required_mean_margin,
            expected_margin,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            self.required_mean_new,
            expected_new,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise MethodContractError("oracle requirements differ from rho calibration")

    @classmethod
    def calibrate(
        cls,
        *,
        entry_new: Sequence[float],
        entry_old: Sequence[float],
        oracle_new: Sequence[float],
        oracle_old: Sequence[float],
        epsilon: float,
        rho: float = ORACLE_REALIZATION_FRACTION,
    ) -> "OracleMeanEventTarget":
        entry_new_values, entry_old_values, entry_mean_new, entry_mean_old, entry_margin = (
            _panel_means(entry_new, entry_old)
        )
        oracle_new_values, oracle_old_values, oracle_mean_new, oracle_mean_old, oracle_margin = (
            _panel_means(oracle_new, oracle_old)
        )
        if len(entry_new_values) != len(oracle_new_values):
            raise MethodContractError("entry and oracle context panels differ")
        del entry_old_values, oracle_old_values
        fraction = _finite("oracle rho", rho)
        threshold_epsilon = _finite("oracle epsilon", epsilon)
        return cls(
            context_count=len(entry_new_values),
            rho=fraction,
            epsilon=threshold_epsilon,
            entry_mean_new=entry_mean_new,
            entry_mean_old=entry_mean_old,
            entry_mean_margin=entry_margin,
            oracle_mean_new=oracle_mean_new,
            oracle_mean_old=oracle_mean_old,
            oracle_mean_margin=oracle_margin,
            required_mean_new=entry_mean_new
            + fraction * (oracle_mean_new - entry_mean_new),
            required_mean_margin=fraction * oracle_margin,
        )

    @property
    def margin_denominator_degenerate(self) -> bool:
        return self.oracle_mean_margin <= self.epsilon

    @property
    def new_denominator(self) -> float:
        return self.oracle_mean_new - self.entry_mean_new

    @property
    def new_denominator_degenerate(self) -> bool:
        return self.new_denominator <= self.epsilon

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ode-edit-oracle-mean-target/v1",
            "event_mode": ORACLE_MEAN_EVENT_MODE,
            "context_count": self.context_count,
            "uniform_context_weights": [
                1.0 / self.context_count for _ in range(self.context_count)
            ],
            "rho": self.rho,
            "epsilon": self.epsilon,
            "entry_mean_new": self.entry_mean_new,
            "entry_mean_old": self.entry_mean_old,
            "entry_mean_margin": self.entry_mean_margin,
            "oracle_mean_new": self.oracle_mean_new,
            "oracle_mean_old": self.oracle_mean_old,
            "oracle_mean_margin": self.oracle_mean_margin,
            "required_mean_new": self.required_mean_new,
            "required_mean_margin": self.required_mean_margin,
            "margin_denominator_degenerate": self.margin_denominator_degenerate,
            "new_denominator": self.new_denominator,
            "new_denominator_degenerate": self.new_denominator_degenerate,
            "native_nll_equality_used": False,
            "per_context_success_used": False,
        }


def oracle_mean_event_from_log_likelihoods(
    target_new: Sequence[float],
    target_old: Sequence[float],
    *,
    target: OracleMeanEventTarget,
    tau: float,
    nfe: int = 0,
) -> EventReading:
    """Build the primary event exclusively from two aggregate deficits."""

    new, old, mean_new, mean_old, mean_margin = _panel_means(
        target_new, target_old
    )
    if len(new) != target.context_count:
        raise MethodContractError("oracle runtime context count differs")
    temperature = _finite("oracle event tau", tau)
    if temperature <= 0.0:
        raise MethodContractError("oracle event tau must be positive")
    relative_deficit = target.required_mean_margin - mean_margin
    absolute_deficit = target.required_mean_new - mean_new
    deficits = (relative_deficit, absolute_deficit)
    maximum = max(deficits)
    smooth = maximum + temperature * (
        math.log(
            math.fsum(
                math.exp((value - maximum) / temperature) for value in deficits
            )
        )
        - math.log(2)
    )
    return EventReading(
        hard_phi=maximum,
        smooth_phi=smooth,
        context_margins=tuple(
            new_value - old_value
            for new_value, old_value in zip(new, old, strict=True)
        ),
        nfe=nfe,
        target_new_log_likelihoods=new,
        target_old_log_likelihoods=old,
        event_mode=ORACLE_MEAN_EVENT_MODE,
        decision_deficits=deficits,
    )


def differentiable_oracle_mean_event_from_log_likelihoods(
    target_new: torch.Tensor,
    target_old: torch.Tensor,
    *,
    target: OracleMeanEventTarget,
    tau: float,
) -> DifferentiableEvent:
    if (
        target_new.ndim != 1
        or target_old.shape != target_new.shape
        or target_new.numel() != target.context_count
    ):
        raise MethodContractError("differentiable oracle event panels differ")
    temperature = _finite("oracle event tau", tau)
    mean_new = target_new.mean()
    mean_margin = (target_new - target_old).mean()
    deficits = torch.stack(
        (
            mean_margin.new_tensor(target.required_mean_margin) - mean_margin,
            mean_new.new_tensor(target.required_mean_new) - mean_new,
        )
    )
    smooth = temperature * (
        torch.logsumexp(deficits / temperature, dim=0) - math.log(2)
    )
    new_values = tuple(float(value) for value in target_new.detach().cpu())
    old_values = tuple(float(value) for value in target_old.detach().cpu())
    reading = oracle_mean_event_from_log_likelihoods(
        new_values,
        old_values,
        target=target,
        tau=temperature,
        nfe=EVENT_MODEL_FORWARD_CALLS,
    )
    return DifferentiableEvent(
        reading=reading,
        smooth_phi=smooth,
        target_new_log_likelihoods=new_values,
        target_old_log_likelihoods=old_values,
    )


def measure_oracle_mean_event(
    model: torch.nn.Module,
    tokenizer: Any,
    request: ControllerRequest,
    context_templates: Sequence[Sequence[str]],
    *,
    target: OracleMeanEventTarget,
    tau: float,
) -> EventReading:
    contexts = build_allowed_contexts(request, context_templates)
    new_batch = build_teacher_batch(tokenizer, contexts, request.target_new)
    old_batch = build_teacher_batch(tokenizer, contexts, request.target_old)
    new = score_teacher_batch_tensor(model, new_batch, differentiable=False)
    old = score_teacher_batch_tensor(model, old_batch, differentiable=False)
    return oracle_mean_event_from_log_likelihoods(
        tuple(float(value) for value in new.cpu()),
        tuple(float(value) for value in old.cpu()),
        target=target,
        tau=tau,
        nfe=EVENT_MODEL_FORWARD_CALLS,
    )


def measure_differentiable_oracle_mean_event(
    model: torch.nn.Module,
    tokenizer: Any,
    request: ControllerRequest,
    context_templates: Sequence[Sequence[str]],
    *,
    target: OracleMeanEventTarget,
    tau: float,
) -> DifferentiableEvent:
    contexts = build_allowed_contexts(request, context_templates)
    new_batch = build_teacher_batch(tokenizer, contexts, request.target_new)
    old_batch = build_teacher_batch(tokenizer, contexts, request.target_old)
    new = score_teacher_batch_tensor(model, new_batch, differentiable=True)
    old = score_teacher_batch_tensor(model, old_batch, differentiable=True)
    return differentiable_oracle_mean_event_from_log_likelihoods(
        new, old, target=target, tau=tau
    )


def _unwrap_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise MethodContractError("oracle z-layer output layout is unsupported")


def _rewrap_output(original: Any, updated: torch.Tensor) -> Any:
    if isinstance(original, torch.Tensor):
        return updated
    values = (updated, *original[1:])
    if type(original) is tuple:
        return values
    if hasattr(original, "_fields"):
        return type(original)(*values)
    try:
        return type(original)(values)
    except TypeError as exc:
        raise MethodContractError("oracle z-layer output wrapper is unsupported") from exc


class _LookupCapture:
    def __init__(self, model: torch.nn.Module, layer_name: str, position: int) -> None:
        self.model = model
        self.layer_name = layer_name
        self.position = int(position)
        self.value: torch.Tensor | None = None
        self.calls = 0
        self._handle: Any = None

    def _hook(self, _module: torch.nn.Module, _args: Any, output: Any) -> Any:
        activation = _unwrap_output(output)
        if activation.ndim != 3 or activation.shape[0] <= 0:
            raise MethodContractError("oracle capture activation layout differs")
        position = self.position if self.position >= 0 else activation.shape[1] + self.position
        if position < 0 or position >= activation.shape[1]:
            raise MethodContractError("oracle canonical lookup position is out of range")
        self.calls += 1
        if self.calls != 1:
            raise MethodContractError("oracle canonical activation was captured twice")
        self.value = activation[0, position, :].detach().clone()
        return output

    def __enter__(self) -> "_LookupCapture":
        module = resolve_module(self.model, self.layer_name)
        self._handle = module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


class OracleActivationHook:
    """Clone-and-add one shared delta at each batch subject lookup position."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_name: str,
        positions: Sequence[int],
        delta: torch.Tensor,
    ) -> None:
        if not positions:
            raise MethodContractError("oracle hook positions are empty")
        if delta.ndim != 1 or not delta.is_floating_point() or not torch.isfinite(delta).all():
            raise MethodContractError("oracle hook delta is invalid")
        self.model = model
        self.layer_name = layer_name
        self.positions = tuple(int(value) for value in positions)
        self.delta = delta.detach().clone()
        self.calls = 0
        self._handle: Any = None

    def _hook(self, _module: torch.nn.Module, _args: Any, output: Any) -> Any:
        activation = _unwrap_output(output)
        if activation.ndim != 3 or activation.shape[0] != len(self.positions):
            raise MethodContractError("oracle patched activation layout differs")
        if activation.shape[2] != self.delta.numel():
            raise MethodContractError("oracle delta width differs from activation")
        normalized = tuple(
            value if value >= 0 else activation.shape[1] + value
            for value in self.positions
        )
        if any(value < 0 or value >= activation.shape[1] for value in normalized):
            raise MethodContractError("oracle lookup position is out of range")
        patched = activation.clone()
        rows = torch.arange(len(normalized), device=activation.device)
        columns = torch.tensor(normalized, device=activation.device, dtype=torch.long)
        delta = self.delta.to(device=activation.device, dtype=activation.dtype)
        patched[rows, columns, :] = patched[rows, columns, :] + delta
        self.calls += 1
        return _rewrap_output(output, patched)

    def __enter__(self) -> "OracleActivationHook":
        module = resolve_module(self.model, self.layer_name)
        self._handle = module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def _parameter_guard(model: torch.nn.Module) -> tuple[tuple[Any, ...], ...]:
    rows = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            raise MethodContractError(
                "oracle calibration requires every parameter grad to remain None"
            )
        rows.append(
            (
                name,
                parameter.data_ptr(),
                parameter._version,
                tuple(parameter.shape),
                str(parameter.dtype),
                str(parameter.device),
                parameter.requires_grad,
            )
        )
    return tuple(rows)


def _rng_guard() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    return (
        torch.get_rng_state().clone(),
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else (),
    )


def _assert_rng_guard(before: tuple[torch.Tensor, tuple[torch.Tensor, ...]]) -> None:
    cpu, cuda = before
    if not torch.equal(torch.get_rng_state(), cpu):
        raise MethodContractError("oracle calibration changed CPU RNG")
    current = tuple(torch.cuda.get_rng_state_all()) if cuda else ()
    if len(current) != len(cuda) or any(
        not torch.equal(left, right) for left, right in zip(current, cuda, strict=True)
    ):
        raise MethodContractError("oracle calibration changed CUDA RNG")


@dataclass(frozen=True, slots=True)
class OracleCalibration:
    target: OracleMeanEventTarget
    entry_reading: EventReading
    oracle_shadow_reading: EventReading
    entry_forward_count: int
    oracle_forward_count: int
    hook_call_count: int
    layer_name: str
    lookup_positions: tuple[int, ...]
    h0_dtype: str
    delta_dtype: str
    delta_l2: float
    parameter_guard_exact: bool
    rng_guard_exact: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ode-edit-oracle-calibration/v1",
            "target": self.target.to_dict(),
            "entry_forward_count": self.entry_forward_count,
            "oracle_forward_count": self.oracle_forward_count,
            "hook_call_count": self.hook_call_count,
            "layer_name": self.layer_name,
            "lookup_positions": list(self.lookup_positions),
            "h0_dtype": self.h0_dtype,
            "delta_dtype": self.delta_dtype,
            "delta_l2": self.delta_l2,
            "parameter_guard_exact": self.parameter_guard_exact,
            "rng_guard_exact": self.rng_guard_exact,
            "weight_mutation": False,
            "extra_backward": 0,
        }


def _allowed_lookup_templates(
    request: ControllerRequest,
    context_templates: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    values: list[str] = []
    for group in context_templates:
        for template in group:
            prefix = template.format(request.prompt)
            if prefix.count("{}") != 1:
                raise MethodContractError("oracle lookup template lost subject field")
            values.append(prefix)
    if not values:
        raise MethodContractError("oracle lookup templates are empty")
    return tuple(values)


def calibrate_oracle_mean_target(
    model: torch.nn.Module,
    tokenizer: Any,
    request: ControllerRequest,
    context_templates: Sequence[Sequence[str]],
    *,
    direct_z: FrozenDirectZ,
    bindings: Any,
    hparams: Any,
    tau: float,
    epsilon: float,
    entry_forward_scope: ContextManager[Any] | None = None,
    oracle_forward_scope: ContextManager[Any] | None = None,
) -> OracleCalibration:
    """Measure entry and direct-z oracle with exactly four total forwards."""

    if direct_z.values.shape[1] != 1 or direct_z.z_layer != int(hparams.layers[-1]):
        raise MethodContractError("oracle direct-z shape/layer differs")
    contexts = build_allowed_contexts(request, context_templates)
    templates = _allowed_lookup_templates(request, context_templates)
    if len(contexts) != len(templates):
        raise MethodContractError("oracle context/template count differs")
    positions = tuple(
        int(
            bindings.compute_z.find_fact_lookup_idx(
                template,
                request.subject,
                tokenizer,
                hparams.fact_token,
                verbose=False,
            )
        )
        for template in templates
    )
    new_batch = build_teacher_batch(tokenizer, contexts, request.target_new)
    old_batch = build_teacher_batch(tokenizer, contexts, request.target_old)
    layer_name = str(hparams.layer_module_tmp).format(direct_z.z_layer)
    parameters_before = _parameter_guard(model)
    rng_before = _rng_guard()
    capture = _LookupCapture(model, layer_name, positions[0])
    entry_scope = entry_forward_scope or nullcontext()
    oracle_scope = oracle_forward_scope or nullcontext()
    with entry_scope:
        with capture:
            entry_new_tensor = score_teacher_batch_tensor(
                model, new_batch, differentiable=False
            )
        entry_old_tensor = score_teacher_batch_tensor(
            model, old_batch, differentiable=False
        )
    if capture.calls != 1 or capture.value is None:
        raise MethodContractError("oracle canonical h0 capture count differs")
    z_value = direct_z.values[:, 0].to(device=capture.value.device)
    delta = z_value - capture.value.to(dtype=z_value.dtype)
    patch = OracleActivationHook(model, layer_name, positions, delta)
    with oracle_scope, patch:
        oracle_new_tensor = score_teacher_batch_tensor(
            model, new_batch, differentiable=False
        )
        oracle_old_tensor = score_teacher_batch_tensor(
            model, old_batch, differentiable=False
        )
    if patch.calls != ORACLE_MODEL_FORWARD_CALLS:
        raise MethodContractError("oracle activation hook call count differs")
    if _parameter_guard(model) != parameters_before:
        raise MethodContractError("oracle calibration changed parameter identity")
    _assert_rng_guard(rng_before)
    entry_new = tuple(float(value) for value in entry_new_tensor.cpu())
    entry_old = tuple(float(value) for value in entry_old_tensor.cpu())
    oracle_new = tuple(float(value) for value in oracle_new_tensor.cpu())
    oracle_old = tuple(float(value) for value in oracle_old_tensor.cpu())
    target = OracleMeanEventTarget.calibrate(
        entry_new=entry_new,
        entry_old=entry_old,
        oracle_new=oracle_new,
        oracle_old=oracle_old,
        epsilon=epsilon,
    )
    entry_reading = oracle_mean_event_from_log_likelihoods(
        entry_new,
        entry_old,
        target=target,
        tau=tau,
        nfe=EVENT_MODEL_FORWARD_CALLS,
    )
    oracle_shadow = event_from_log_likelihoods(
        oracle_new,
        oracle_old,
        tau=tau,
        nfe=ORACLE_MODEL_FORWARD_CALLS,
    )
    return OracleCalibration(
        target=target,
        entry_reading=entry_reading,
        oracle_shadow_reading=oracle_shadow,
        entry_forward_count=EVENT_MODEL_FORWARD_CALLS,
        oracle_forward_count=ORACLE_MODEL_FORWARD_CALLS,
        hook_call_count=patch.calls,
        layer_name=layer_name,
        lookup_positions=positions,
        h0_dtype=str(capture.value.dtype),
        delta_dtype=str(delta.dtype),
        delta_l2=float(torch.linalg.vector_norm(delta.float()).item()),
        parameter_guard_exact=True,
        rng_guard_exact=True,
    )


__all__ = [
    "ORACLE_MEAN_EVENT_MODE",
    "ORACLE_MODEL_FORWARD_CALLS",
    "ORACLE_REALIZATION_FRACTION",
    "OracleActivationHook",
    "OracleCalibration",
    "OracleMeanEventTarget",
    "calibrate_oracle_mean_target",
    "differentiable_oracle_mean_event_from_log_likelihoods",
    "measure_differentiable_oracle_mean_event",
    "measure_oracle_mean_event",
    "oracle_mean_event_from_log_likelihoods",
]
