"""Pure tensor math for ODE-Edit motivation diagnostics.

No function in this module loads a model or requires CUDA.  In particular,
low-rank comparisons never materialize a dense weight update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Mapping, Protocol, Sequence, TypeVar

import torch

from .contracts import ContractError, LowRankFactor


class MetricOperator(Protocol):
    """Right-side metric action used by the C inner product."""

    def matmat(self, value: torch.Tensor) -> torch.Tensor:
        ...


class LinearSolveBackend(Protocol):
    """Backend contract for a matrix or operator linear solve."""

    name: str

    def solve(self, system: Any, rhs: torch.Tensor) -> torch.Tensor:
        ...


class MemitSystemSolver(Protocol):
    """Solver contract for ``(lambda*C + K K.T) X = K``."""

    name: str

    def adjusted_keys(
        self,
        covariance: Any,
        keys: torch.Tensor,
        covariance_weight: float,
    ) -> torch.Tensor:
        ...


@dataclass(frozen=True, slots=True)
class DenseMetricOperator:
    matrix: torch.Tensor

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ContractError("metric matrix must be square")

    def matmat(self, value: torch.Tensor) -> torch.Tensor:
        return self.matrix @ value


@dataclass(frozen=True, slots=True)
class CallableMetricOperator:
    action: Callable[[torch.Tensor], torch.Tensor]

    def matmat(self, value: torch.Tensor) -> torch.Tensor:
        return self.action(value)


def _metric_action(
    metric: torch.Tensor | MetricOperator | Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
) -> torch.Tensor:
    if isinstance(metric, torch.Tensor):
        result = metric @ value
    elif hasattr(metric, "matmat"):
        result = metric.matmat(value)
    elif callable(metric):
        result = metric(value)
    else:
        raise TypeError("metric must be a tensor, MetricOperator, or callable")
    if result.shape != value.shape:
        raise ContractError(
            f"metric action returned shape {tuple(result.shape)}, expected {tuple(value.shape)}"
        )
    return result


def c_inner_product(
    first: LowRankFactor,
    second: LowRankFactor,
    metric: torch.Tensor | MetricOperator | Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute ``tr(A C B.T)`` from factors, without constructing A or B.

    ``A = first.left @ first.right.T`` and similarly for ``B``.  The largest
    intermediates are a metric-applied right factor and rank-by-rank Gram
    matrices.
    """

    if first.weight_shape != second.weight_shape:
        raise ContractError(
            f"C-inner product shape mismatch: {first.weight_shape} != {second.weight_shape}"
        )
    metric_second_right = _metric_action(metric, second.right)
    left_gram = first.left.transpose(0, 1) @ second.left
    right_gram = first.right.transpose(0, 1) @ metric_second_right
    return torch.sum(left_gram * right_gram)


def c_squared_norm(
    factor: LowRankFactor,
    metric: torch.Tensor | MetricOperator | Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    value = c_inner_product(factor, factor, metric)
    tolerance = 100 * torch.finfo(value.dtype).eps
    if bool(value < -tolerance):
        raise ContractError("metric produced a negative squared C-norm")
    return value.clamp_min(0)


def c_norm(
    factor: LowRankFactor,
    metric: torch.Tensor | MetricOperator | Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    return torch.sqrt(c_squared_norm(factor, metric))


def c_cosine_similarity(
    first: LowRankFactor,
    second: LowRankFactor,
    metric: torch.Tensor | MetricOperator | Callable[[torch.Tensor], torch.Tensor],
    *,
    epsilon: float = 0.0,
) -> torch.Tensor:
    denominator = c_norm(first, metric) * c_norm(second, metric)
    if epsilon:
        denominator = denominator.clamp_min(epsilon)
    elif bool(denominator == 0):
        raise ContractError("C-cosine is undefined for a zero-norm factor")
    return c_inner_product(first, second, metric) / denominator


@dataclass(frozen=True, slots=True)
class TorchLinearSolveBackend:
    """Native ``torch.linalg.solve`` backend."""

    name: str = "torch.linalg.solve"

    def solve(self, system: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if not isinstance(system, torch.Tensor):
            raise TypeError("native torch backend requires a dense tensor system")
        return torch.linalg.solve(system, rhs)


@dataclass(frozen=True, slots=True)
class CallableLinearSolveBackend:
    """Adapter for iterative, sharded, or external linear solvers."""

    callback: Callable[[Any, torch.Tensor], torch.Tensor]
    name: str = "callable-linear-solve"

    def solve(self, system: Any, rhs: torch.Tensor) -> torch.Tensor:
        result = self.callback(system, rhs)
        if result.shape != rhs.shape:
            raise ContractError(
                f"linear solve returned {tuple(result.shape)}, expected {tuple(rhs.shape)}"
            )
        return result


@dataclass(frozen=True, slots=True)
class NativeMemitSolver:
    """Reference MEMIT solve, matching EasyEdit's dense native expression."""

    backend: LinearSolveBackend = TorchLinearSolveBackend()
    name: str = "native-memit"

    def adjusted_keys(
        self,
        covariance: torch.Tensor,
        keys: torch.Tensor,
        covariance_weight: float,
    ) -> torch.Tensor:
        _validate_memit_inputs(covariance, keys, covariance_weight)
        system = covariance_weight * covariance + keys @ keys.transpose(0, 1)
        return self.backend.solve(system, keys)


@dataclass(frozen=True, slots=True)
class WoodburyMemitSolver:
    """MEMIT solve using covariance solves plus a rank-sized system.

    ``covariance_backend`` may be replaced by a backend for a factored,
    iterative, or implicit covariance operator.  This keeps the high-dimensional
    linear solve pluggable while the Woodbury correction remains small.
    """

    covariance_backend: LinearSolveBackend = TorchLinearSolveBackend()
    small_backend: LinearSolveBackend = TorchLinearSolveBackend()
    name: str = "woodbury-memit"

    def adjusted_keys(
        self,
        covariance: Any,
        keys: torch.Tensor,
        covariance_weight: float,
    ) -> torch.Tensor:
        if not math.isfinite(covariance_weight) or covariance_weight <= 0:
            raise ContractError("covariance_weight must be finite and positive")
        if keys.ndim != 2 or keys.shape[1] == 0:
            raise ContractError("keys must be a non-empty matrix")
        covariance_inverse_keys = self.covariance_backend.solve(covariance, keys)
        base = covariance_inverse_keys / covariance_weight
        gram = torch.eye(
            keys.shape[1],
            dtype=keys.dtype,
            device=keys.device,
        ) + keys.transpose(0, 1) @ base
        # X = base @ gram^{-1}; transpose to express the right solve through
        # the same backend contract.
        return self.small_backend.solve(gram.transpose(0, 1), base.transpose(0, 1)).transpose(0, 1)


def _validate_memit_inputs(
    covariance: torch.Tensor,
    keys: torch.Tensor,
    covariance_weight: float,
) -> None:
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ContractError("covariance must be square")
    if keys.ndim != 2 or keys.shape[0] != covariance.shape[0] or keys.shape[1] == 0:
        raise ContractError("keys must be [covariance_dim, nonzero_rank]")
    if not math.isfinite(covariance_weight) or covariance_weight <= 0:
        raise ContractError("covariance_weight must be finite and positive")


def solve_memit_factor(
    *,
    covariance: Any,
    keys: torch.Tensor,
    targets: torch.Tensor,
    covariance_weight: float,
    remaining_layers: int,
    solver: MemitSystemSolver,
    weight_name: str,
    weight_shape: Sequence[int],
    expected_weight_sha256: str,
) -> LowRankFactor:
    """Build one canonical proposal factor through a selected solve backend."""

    if targets.ndim != 2 or targets.shape[1] != keys.shape[1]:
        raise ContractError("targets and keys must have the same proposal rank")
    if remaining_layers <= 0:
        raise ContractError("remaining_layers must be positive")
    adjusted = solver.adjusted_keys(covariance, keys, covariance_weight)
    residual = targets / remaining_layers
    from .contracts import orient_easyedit_factor

    return orient_easyedit_factor(
        adjusted,
        residual,
        weight_name=weight_name,
        weight_shape=weight_shape,
        expected_weight_sha256=expected_weight_sha256,
    )


@dataclass(frozen=True, slots=True)
class ExactTop1Stop:
    target_ids: tuple[int, ...]
    predicted_ids: tuple[int, ...]
    margins: tuple[float, ...]
    satisfied: tuple[bool, ...]

    @property
    def all_satisfied(self) -> bool:
        return all(self.satisfied)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_ids": list(self.target_ids),
            "predicted_ids": list(self.predicted_ids),
            "margins": list(self.margins),
            "satisfied": list(self.satisfied),
            "all_satisfied": self.all_satisfied,
        }


def _flatten_targets(logits: torch.Tensor, target_ids: int | torch.Tensor) -> torch.Tensor:
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ContractError("logits must have a vocabulary dimension of at least two")
    prefix_shape = logits.shape[:-1]
    target = torch.as_tensor(target_ids, device=logits.device, dtype=torch.long)
    if target.ndim == 0:
        target = target.expand(prefix_shape)
    if tuple(target.shape) != tuple(prefix_shape):
        raise ContractError(
            f"target shape {tuple(target.shape)} does not match logits prefix {tuple(prefix_shape)}"
        )
    if bool((target < 0).any()) or bool((target >= logits.shape[-1]).any()):
        raise ContractError("target id is outside the vocabulary")
    return target.reshape(-1)


def exact_top1_stop(
    logits: torch.Tensor,
    target_ids: int | torch.Tensor,
    *,
    margin_threshold: float = 0.0,
) -> ExactTop1Stop:
    """Detached exact top-1 decision; never use this as an optimization loss."""

    if not math.isfinite(margin_threshold):
        raise ContractError("margin_threshold must be finite")
    if not logits.is_floating_point():
        raise ContractError("logits must be floating-point")
    flat_logits = logits.detach().reshape(-1, logits.shape[-1])
    flat_targets = _flatten_targets(logits, target_ids)
    rows = torch.arange(flat_logits.shape[0], device=flat_logits.device)
    target_logits = flat_logits[rows, flat_targets]
    non_target = flat_logits.clone()
    non_target[rows, flat_targets] = -torch.inf
    best_other = non_target.max(dim=-1).values
    predicted = flat_logits.argmax(dim=-1)
    margins = target_logits - best_other
    satisfied = (predicted == flat_targets) & (margins >= margin_threshold)
    return ExactTop1Stop(
        target_ids=tuple(int(value) for value in flat_targets.cpu().tolist()),
        predicted_ids=tuple(int(value) for value in predicted.cpu().tolist()),
        margins=tuple(float(value) for value in margins.cpu().tolist()),
        satisfied=tuple(bool(value) for value in satisfied.cpu().tolist()),
    )


def smooth_target_utility(
    logits: torch.Tensor,
    target_ids: int | torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Differentiable soft target-vs-rest margin.

    As ``temperature`` approaches zero this approaches the exact target logit
    minus the best non-target logit, while retaining a useful gradient at
    ordinary positive temperatures.
    """

    if not math.isfinite(temperature) or temperature <= 0:
        raise ContractError("temperature must be finite and positive")
    if not logits.is_floating_point():
        raise ContractError("logits must be floating-point")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = _flatten_targets(logits, target_ids)
    rows = torch.arange(flat_logits.shape[0], device=flat_logits.device)
    target_logits = flat_logits[rows, flat_targets]
    non_target = flat_logits.clone()
    non_target[rows, flat_targets] = -torch.inf
    soft_best_other = temperature * torch.logsumexp(non_target / temperature, dim=-1)
    return (target_logits - soft_best_other).reshape(logits.shape[:-1])


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ContractError(f"{name} must be scalar")
        value = value.detach().cpu().item()
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class FiniteDifferenceSample:
    step: float
    negative_value: float
    positive_value: float
    central_slope: float
    absolute_error: float
    relative_error: float | None


@dataclass(frozen=True, slots=True)
class FiniteDifferenceCalibration:
    baseline: float
    predicted_slope: float
    samples: tuple[FiniteDifferenceSample, ...]

    @property
    def best_absolute_error(self) -> float:
        return min(sample.absolute_error for sample in self.samples)


def finite_difference_calibration(
    evaluate: Callable[[float], float | torch.Tensor],
    *,
    predicted_slope: float | torch.Tensor,
    steps: Sequence[float],
    zero_tolerance: float = 1e-12,
) -> FiniteDifferenceCalibration:
    """Compare an analytic local slope with central finite differences."""

    predicted = _finite_float("predicted_slope", predicted_slope)
    baseline = _finite_float("evaluate(0)", evaluate(0.0))
    if not steps:
        raise ContractError("finite-difference steps must not be empty")
    samples: list[FiniteDifferenceSample] = []
    for raw_step in steps:
        step = _finite_float("finite-difference step", raw_step)
        if step <= 0:
            raise ContractError("finite-difference steps must be positive")
        negative = _finite_float(f"evaluate(-{step})", evaluate(-step))
        positive = _finite_float(f"evaluate({step})", evaluate(step))
        slope = (positive - negative) / (2 * step)
        absolute_error = abs(slope - predicted)
        denominator = abs(predicted)
        relative_error = None if denominator <= zero_tolerance else absolute_error / denominator
        samples.append(
            FiniteDifferenceSample(
                step=step,
                negative_value=negative,
                positive_value=positive,
                central_slope=slope,
                absolute_error=absolute_error,
                relative_error=relative_error,
            )
        )
    return FiniteDifferenceCalibration(
        baseline=baseline,
        predicted_slope=predicted,
        samples=tuple(samples),
    )


@dataclass(frozen=True, slots=True)
class JointAdditivity:
    baseline: float
    individual_values: tuple[float, ...]
    predicted_joint: float
    actual_joint: float
    interaction: float
    relative_interaction: float | None


def joint_additivity(
    *,
    baseline: float | torch.Tensor,
    individual_values: Sequence[float | torch.Tensor],
    joint_value: float | torch.Tensor,
    zero_tolerance: float = 1e-12,
) -> JointAdditivity:
    """Measure departure from the sum of individually observed changes."""

    base = _finite_float("baseline", baseline)
    individuals = tuple(
        _finite_float(f"individual_values[{index}]", value)
        for index, value in enumerate(individual_values)
    )
    if not individuals:
        raise ContractError("individual_values must not be empty")
    actual = _finite_float("joint_value", joint_value)
    predicted = base + sum(value - base for value in individuals)
    interaction = actual - predicted
    predicted_effect = predicted - base
    relative = (
        None
        if abs(predicted_effect) <= zero_tolerance
        else interaction / abs(predicted_effect)
    )
    return JointAdditivity(
        baseline=base,
        individual_values=individuals,
        predicted_joint=predicted,
        actual_joint=actual,
        interaction=interaction,
        relative_interaction=relative,
    )


@dataclass(frozen=True, slots=True)
class RankTurnover:
    k: int
    before_top: tuple[Hashable, ...]
    after_top: tuple[Hashable, ...]
    retained: tuple[Hashable, ...]
    turnover_fraction: float
    mean_normalized_displacement: float


K = TypeVar("K", bound=Hashable)


def rank_turnover(
    before_scores: Mapping[K, float | torch.Tensor],
    after_scores: Mapping[K, float | torch.Tensor],
    *,
    k: int | None = None,
) -> RankTurnover:
    """Top-k membership churn plus rank displacement for retained entries."""

    if set(before_scores) != set(after_scores) or not before_scores:
        raise ContractError("rank score mappings must contain the same non-empty key set")
    count = len(before_scores)
    chosen_k = count if k is None else int(k)
    if chosen_k <= 0 or chosen_k > count:
        raise ContractError(f"k must be in [1, {count}]")

    before_values = {key: _finite_float(f"before_scores[{key!r}]", value) for key, value in before_scores.items()}
    after_values = {key: _finite_float(f"after_scores[{key!r}]", value) for key, value in after_scores.items()}
    before_order = tuple(sorted(before_values, key=lambda key: (-before_values[key], repr(key))))
    after_order = tuple(sorted(after_values, key=lambda key: (-after_values[key], repr(key))))
    before_top = before_order[:chosen_k]
    after_top = after_order[:chosen_k]
    retained_set = set(before_top) & set(after_top)
    retained = tuple(key for key in before_top if key in retained_set)
    turnover = 1.0 - len(retained) / chosen_k
    if retained:
        before_rank = {key: index for index, key in enumerate(before_order)}
        after_rank = {key: index for index, key in enumerate(after_order)}
        normalizer = max(1, count - 1)
        displacement = sum(
            abs(before_rank[key] - after_rank[key]) / normalizer for key in retained
        ) / len(retained)
    else:
        displacement = 1.0
    return RankTurnover(
        k=chosen_k,
        before_top=before_top,
        after_top=after_top,
        retained=retained,
        turnover_fraction=turnover,
        mean_normalized_displacement=displacement,
    )


@dataclass(frozen=True, slots=True)
class Reroutability:
    baseline: float
    direct_value: float
    best_route: Hashable
    best_rerouted_value: float
    recovered_fraction: float | None
    residual_effect: float


def reroutability(
    *,
    baseline: float | torch.Tensor,
    direct_value: float | torch.Tensor,
    rerouted_values: Mapping[K, float | torch.Tensor],
    zero_tolerance: float = 1e-12,
) -> Reroutability:
    """Quantify how much of a direct effect survives through the best route."""

    if not rerouted_values:
        raise ContractError("rerouted_values must not be empty")
    base = _finite_float("baseline", baseline)
    direct = _finite_float("direct_value", direct_value)
    values = {
        key: _finite_float(f"rerouted_values[{key!r}]", value)
        for key, value in rerouted_values.items()
    }
    direct_effect = direct - base
    if direct_effect >= 0:
        best_route = max(values, key=lambda key: (values[key], repr(key)))
    else:
        best_route = min(values, key=lambda key: (values[key], repr(key)))
    best = values[best_route]
    rerouted_effect = best - base
    recovered = (
        None
        if abs(direct_effect) <= zero_tolerance
        else rerouted_effect / direct_effect
    )
    return Reroutability(
        baseline=base,
        direct_value=direct,
        best_route=best_route,
        best_rerouted_value=best,
        recovered_fraction=recovered,
        residual_effect=direct_effect - rerouted_effect,
    )
