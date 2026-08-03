"""V3 direct-z oracle event with an absolute-new and mean-margin target.

The direct-z oracle calibrates only the absolute target-new likelihood floor.
Its positive mean margin is a validity check and a diagnostic denominator, not
a fractional decision requirement.  Runtime decisions therefore consume the
same two aggregate deficits everywhere: ``-mean_margin`` and
``required_mean_new - mean_new``.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Sequence

import torch

from project.run_scripts.ode_edit_motivation.direct_z import FrozenDirectZ

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
from .oracle_event import (
    ORACLE_MODEL_FORWARD_CALLS,
    ORACLE_REALIZATION_FRACTION,
    OracleActivationHook,
    _LookupCapture,
    _allowed_lookup_templates,
    _assert_rng_guard,
    _finite,
    _panel_means,
    _parameter_guard,
    _rng_guard,
)


ORACLE_ABSOLUTE_MEAN_MARGIN_EVENT_MODE = (
    "direct-z-oracle-absolute-new-uniform-mean-margin-v3"
)


@dataclass(frozen=True, slots=True)
class OracleAbsoluteMeanMarginTarget:
    """Edit-local V3 target: mean margin >= 0 and oracle absolute floor."""

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
    required_mean_margin: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.context_count, bool)
            or not isinstance(self.context_count, int)
            or self.context_count <= 0
        ):
            raise MethodContractError("V3 oracle context count is invalid")
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
            raise MethodContractError("V3 oracle realization fraction differs")
        if self.epsilon <= 0.0:
            raise MethodContractError("V3 oracle epsilon must be positive")
        if self.oracle_mean_margin <= 0.0:
            raise MethodContractError("V3 oracle mean margin is not positive")
        if self.oracle_mean_new <= self.entry_mean_new + self.epsilon:
            raise MethodContractError("V3 oracle target-new likelihood is insufficient")
        expected_new = self.entry_mean_new + self.rho * (
            self.oracle_mean_new - self.entry_mean_new
        )
        if self.required_mean_margin != 0.0 or not math.isclose(
            self.required_mean_new,
            expected_new,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise MethodContractError("V3 oracle requirements differ from the lock")

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
    ) -> "OracleAbsoluteMeanMarginTarget":
        entry_new_values, _, entry_mean_new, entry_mean_old, entry_margin = (
            _panel_means(entry_new, entry_old)
        )
        oracle_new_values, _, oracle_mean_new, oracle_mean_old, oracle_margin = (
            _panel_means(oracle_new, oracle_old)
        )
        if len(entry_new_values) != len(oracle_new_values):
            raise MethodContractError("V3 entry and oracle context panels differ")
        fraction = _finite("V3 oracle rho", rho)
        threshold_epsilon = _finite("V3 oracle epsilon", epsilon)
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
            required_mean_margin=0.0,
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
            "schema_version": "ode-edit-oracle-absolute-mean-margin-target/v3",
            "event_mode": ORACLE_ABSOLUTE_MEAN_MARGIN_EVENT_MODE,
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
            "oracle_margin_fraction_used": False,
            "native_nll_equality_used": False,
            "per_context_success_used": False,
        }


def oracle_absolute_mean_margin_event_from_log_likelihoods(
    target_new: Sequence[float],
    target_old: Sequence[float],
    *,
    target: OracleAbsoluteMeanMarginTarget,
    tau: float,
    nfe: int = 0,
) -> EventReading:
    """Build V3 from the absolute floor and zero uniform-mean margin floor."""

    new, old, mean_new, _mean_old, mean_margin = _panel_means(
        target_new, target_old
    )
    if len(new) != target.context_count:
        raise MethodContractError("V3 runtime context count differs")
    temperature = _finite("V3 event tau", tau)
    if temperature <= 0.0:
        raise MethodContractError("V3 event tau must be positive")
    deficits = (-mean_margin, target.required_mean_new - mean_new)
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
        event_mode=ORACLE_ABSOLUTE_MEAN_MARGIN_EVENT_MODE,
        decision_deficits=deficits,
    )


def differentiable_oracle_absolute_mean_margin_event_from_log_likelihoods(
    target_new: torch.Tensor,
    target_old: torch.Tensor,
    *,
    target: OracleAbsoluteMeanMarginTarget,
    tau: float,
) -> DifferentiableEvent:
    if (
        target_new.ndim != 1
        or target_old.shape != target_new.shape
        or target_new.numel() != target.context_count
    ):
        raise MethodContractError("differentiable V3 oracle event panels differ")
    temperature = _finite("V3 event tau", tau)
    mean_new = target_new.mean()
    mean_margin = (target_new - target_old).mean()
    deficits = torch.stack(
        (-mean_margin, mean_new.new_tensor(target.required_mean_new) - mean_new)
    )
    smooth = temperature * (
        torch.logsumexp(deficits / temperature, dim=0) - math.log(2)
    )
    new_values = tuple(float(value) for value in target_new.detach().cpu())
    old_values = tuple(float(value) for value in target_old.detach().cpu())
    reading = oracle_absolute_mean_margin_event_from_log_likelihoods(
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


def measure_oracle_absolute_mean_margin_event(
    model: torch.nn.Module,
    tokenizer: Any,
    request: ControllerRequest,
    context_templates: Sequence[Sequence[str]],
    *,
    target: OracleAbsoluteMeanMarginTarget,
    tau: float,
) -> EventReading:
    contexts = build_allowed_contexts(request, context_templates)
    new_batch = build_teacher_batch(tokenizer, contexts, request.target_new)
    old_batch = build_teacher_batch(tokenizer, contexts, request.target_old)
    new = score_teacher_batch_tensor(model, new_batch, differentiable=False)
    old = score_teacher_batch_tensor(model, old_batch, differentiable=False)
    return oracle_absolute_mean_margin_event_from_log_likelihoods(
        tuple(float(value) for value in new.cpu()),
        tuple(float(value) for value in old.cpu()),
        target=target,
        tau=tau,
        nfe=EVENT_MODEL_FORWARD_CALLS,
    )


def measure_differentiable_oracle_absolute_mean_margin_event(
    model: torch.nn.Module,
    tokenizer: Any,
    request: ControllerRequest,
    context_templates: Sequence[Sequence[str]],
    *,
    target: OracleAbsoluteMeanMarginTarget,
    tau: float,
) -> DifferentiableEvent:
    contexts = build_allowed_contexts(request, context_templates)
    new_batch = build_teacher_batch(tokenizer, contexts, request.target_new)
    old_batch = build_teacher_batch(tokenizer, contexts, request.target_old)
    new = score_teacher_batch_tensor(model, new_batch, differentiable=True)
    old = score_teacher_batch_tensor(model, old_batch, differentiable=True)
    return differentiable_oracle_absolute_mean_margin_event_from_log_likelihoods(
        new, old, target=target, tau=tau
    )


@dataclass(frozen=True, slots=True)
class OracleAbsoluteMeanMarginCalibration:
    target: OracleAbsoluteMeanMarginTarget
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
            "schema_version": "ode-edit-oracle-absolute-calibration/v3",
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


def calibrate_oracle_absolute_mean_margin_target(
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
) -> OracleAbsoluteMeanMarginCalibration:
    """Measure the V3 entry and direct-z oracle with four total forwards."""

    if direct_z.values.shape[1] != 1 or direct_z.z_layer != int(hparams.layers[-1]):
        raise MethodContractError("V3 oracle direct-z shape/layer differs")
    contexts = build_allowed_contexts(request, context_templates)
    templates = _allowed_lookup_templates(request, context_templates)
    if len(contexts) != len(templates):
        raise MethodContractError("V3 oracle context/template count differs")
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
        raise MethodContractError("V3 oracle canonical h0 capture count differs")
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
        raise MethodContractError("V3 oracle activation hook call count differs")
    if _parameter_guard(model) != parameters_before:
        raise MethodContractError("V3 oracle calibration changed parameter identity")
    _assert_rng_guard(rng_before)
    entry_new = tuple(float(value) for value in entry_new_tensor.cpu())
    entry_old = tuple(float(value) for value in entry_old_tensor.cpu())
    oracle_new = tuple(float(value) for value in oracle_new_tensor.cpu())
    oracle_old = tuple(float(value) for value in oracle_old_tensor.cpu())
    target = OracleAbsoluteMeanMarginTarget.calibrate(
        entry_new=entry_new,
        entry_old=entry_old,
        oracle_new=oracle_new,
        oracle_old=oracle_old,
        epsilon=epsilon,
    )
    entry = oracle_absolute_mean_margin_event_from_log_likelihoods(
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
    return OracleAbsoluteMeanMarginCalibration(
        target=target,
        entry_reading=entry,
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
    "ORACLE_ABSOLUTE_MEAN_MARGIN_EVENT_MODE",
    "OracleAbsoluteMeanMarginCalibration",
    "OracleAbsoluteMeanMarginTarget",
    "calibrate_oracle_absolute_mean_margin_target",
    "differentiable_oracle_absolute_mean_margin_event_from_log_likelihoods",
    "measure_differentiable_oracle_absolute_mean_margin_event",
    "measure_oracle_absolute_mean_margin_event",
    "oracle_absolute_mean_margin_event_from_log_likelihoods",
]
