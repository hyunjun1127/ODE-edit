"""Bounded global-scalar proximal gradient in v=D/entry_anchor coordinates.

The physical oracle is responsible for a complete logical-batch smooth loss
and gradient.  Candidate payloads (materialized weights and native commit
statistics) must be independent, detached snapshots.  Only the exact latest
accepted payload is retained; the ledger never stores candidate tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Hashable, Mapping

import torch


TensorMap = Mapping[Hashable, torch.Tensor]
Oracle = Callable[[TensorMap, bool], Mapping[str, Any]]


@dataclass(frozen=True)
class SolverConfig:
    # REQUIRED: bind this from fixed-candidate runtime qualification, not
    # quality results or an adaptive line-search tolerance policy.
    epsilon_num: float
    lambda_norm: float = 0.5
    native_clamp: float = 0.75
    candidate_cap: int = 25

    def __post_init__(self) -> None:
        for name in ("epsilon_num", "lambda_norm", "native_clamp"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.native_clamp <= 0:
            raise ValueError("native_clamp must be positive")
        if (isinstance(self.candidate_cap, bool) or not isinstance(self.candidate_cap, int)
                or not 2 <= self.candidate_cap <= 25):
            raise ValueError("candidate_cap must be an integer in [2,25]")

    @property
    def nu(self) -> float:
        return self.native_clamp / 4.0


@dataclass
class SolverResult:
    v: dict[Hashable, torch.Tensor]
    payload: Any
    smooth: float
    grad: dict[Hashable, torch.Tensor] | None
    ledger: list[dict[str, Any]]
    calls: int
    backward_calls: int
    accepted_updates: int
    rejected_trials: int
    stop_reason: str
    accepted_tau: float | None
    accepted_evaluation: int

    @property
    def no_change(self) -> bool:
        return all(not bool(value.detach().count_nonzero()) for value in self.v.values())


def _finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"nonfinite {name}")


def _anchors(initial_v: TensorMap, anchor_norms: TensorMap) -> dict[Hashable, torch.Tensor]:
    if not initial_v or set(initial_v) != set(anchor_norms):
        raise ValueError("initial_v and anchor_norms must contain the same nonempty eligible set")
    B = None
    result = {}
    for layer, value in initial_v.items():
        if (not isinstance(value, torch.Tensor) or value.ndim != 2 or min(value.shape) <= 0
                or value.dtype not in {torch.float32, torch.float64}):
            raise ValueError("v must have floating shape [d_out,actual_B]")
        if B is None:
            B = value.shape[1]
        if value.shape[1] != B:
            raise ValueError("every layer must contain the same actual_B")
        _finite(value, "initial_v")
        if bool(value.detach().count_nonzero()):
            raise ValueError("v5 initializes every eligible group at zero")
        anchor = torch.as_tensor(anchor_norms[layer], device=value.device, dtype=torch.float64).detach().clone()
        if anchor.shape != (B,) or bool((anchor <= 0).any()):
            raise ValueError("each anchor must have positive shape [actual_B]")
        _finite(anchor, "anchor")
        result[layer] = anchor
    return result


@torch.no_grad()
def group_prox(v: TensorMap, grad: TensorMap, anchor_norms: TensorMap, tau: float,
               *, lambda_norm: float, native_clamp: float) -> dict[Hashable, torch.Tensor]:
    """Exact radial norm-plus-ball prox; never normalize the group gradient.

Columns are groups. Arithmetic is FP64 before casting to each coordinate
tensor's dtype.  A zero column of y maps to exactly zero, without division by
zero.  Every column remains present, including currently inactive groups.
"""
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if not math.isfinite(lambda_norm) or lambda_norm < 0 or not math.isfinite(native_clamp) or native_clamp <= 0:
        raise ValueError("invalid norm or ball coefficient")
    if set(v) != set(grad) or set(v) != set(anchor_norms):
        raise ValueError("prox maps must contain identical eligible groups")
    result = {}
    for layer, value in v.items():
        y = value.detach().double() - tau * grad[layer].detach().to(device=value.device, dtype=torch.float64)
        norms = torch.linalg.vector_norm(y, dim=0)
        anchor = torch.as_tensor(anchor_norms[layer], device=value.device, dtype=torch.float64)
        radius = (norms - tau * lambda_norm / anchor).clamp(min=0, max=native_clamp)
        scale = torch.zeros_like(norms)
        nonzero = norms > 0
        scale[nonzero] = radius[nonzero] / norms[nonzero]
        result[layer] = (y * scale.unsqueeze(0)).to(value.dtype)
        _finite(result[layer], "proximal candidate")
    return result


def norm_penalty(v: TensorMap, anchor_norms: TensorMap, lambda_norm: float) -> float:
    return sum(float((torch.linalg.vector_norm(value.detach().double(), dim=0)
                      * (lambda_norm / anchor_norms[layer])).sum()) for layer, value in v.items())


def _max_group_norm(grad: TensorMap) -> float:
    return max(float(torch.linalg.vector_norm(value.detach().double(), dim=0).max())
               for value in grad.values())


def _payload_stats(payload: Any) -> dict[str, Any]:
    """Copy only JSON scalar accounting, never a candidate tensor/cache."""
    if not isinstance(payload, Mapping) or 'stats' not in payload:
        return {}

    def copy_scalar(value):
        if isinstance(value, Mapping):
            return {str(key): copy_scalar(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [copy_scalar(item) for item in value]
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise TypeError('payload.stats must contain finite JSON scalars only, never tensors')

    result = copy_scalar(payload['stats'])
    if not isinstance(result, dict):
        raise TypeError('payload.stats must be a scalar accounting mapping')
    return result


def solve(oracle: Oracle, initial_v: TensorMap, anchor_norms: TensorMap,
          config: SolverConfig, *,
          on_trial: Callable[[dict[str, Any]], None] | None = None) -> SolverResult:
    """Optimize all layer/request groups, returning the exact accepted state.

``oracle(v, backward)`` returns a mapping with ``smooth`` (request-SUM smooth
loss), ``grad`` (relative-v gradients for every layer, or None on the final
forward-only trial), and ``payload`` (any detached physical candidate/cache).
Exceptions and nonfinite values are technical errors and propagate directly;
they are not converted into line-search rejections or hidden retries.

The zero candidate is evaluation 1 and is differentiable. Evaluations 2
through cap-1 request gradients; evaluation cap is forward-only. Rejected
gradients and payloads are released before the next oracle invocation. There
is no return reevaluation, accepted-update quota, or result-based tolerance.
"""
    if not isinstance(config, SolverConfig):
        raise TypeError("config must be an immutable SolverConfig")
    anchors = _anchors(initial_v, anchor_norms)
    accepted_v = {layer: value.detach().clone().requires_grad_(True) for layer, value in initial_v.items()}
    calls = backward_calls = accepted_updates = rejected_trials = 0
    ledger: list[dict[str, Any]] = []

    def evaluate(candidate: dict[Hashable, torch.Tensor], backward: bool):
        nonlocal calls, backward_calls
        calls += 1
        backward_calls += int(backward)
        response = oracle(candidate, backward)
        smooth = float(response["smooth"])
        if not math.isfinite(smooth):
            raise FloatingPointError("nonfinite oracle smooth loss")
        raw_grad = response["grad"]
        if backward:
            if raw_grad is None or set(raw_grad) != set(candidate):
                raise ValueError("oracle must return gradients for every eligible layer")
            grad = {}
            for layer, value in candidate.items():
                derivative = raw_grad[layer]
                if not isinstance(derivative, torch.Tensor) or derivative.shape != value.shape:
                    raise ValueError("oracle relative-v gradient shape mismatch")
                _finite(derivative, "oracle gradient")
                grad[layer] = derivative.detach().to(device=value.device, dtype=value.dtype).clone()
                _finite(grad[layer], "converted oracle gradient")
        else:
            if raw_grad is not None:
                raise ValueError("the final forward-only oracle must return grad=None")
            grad = None
        return smooth, grad, response["payload"]

    accepted_f, accepted_g, accepted_payload = evaluate(accepted_v, True)
    accepted_evaluation = 1
    accepted_tau = None
    ledger.append(dict(evaluation=1, kind="initial_zero", backward=True, accepted=True,
                       smooth=accepted_f, norm=0.0, composite=accepted_f, tau=None,
                       epsilon_num=config.epsilon_num, stats=_payload_stats(accepted_payload)))
    if on_trial is not None:
        on_trial(dict(ledger[-1]))
    max_grad = _max_group_norm(accepted_g)
    if max_grad == 0.0:
        stop_reason = "zero_proximal_mapping"
    else:
        tau = config.nu / max_grad
        stop_reason = "candidate_budget"
        while calls < config.candidate_cap:
            trial_v = group_prox(accepted_v, accepted_g, anchors, tau,
                                 lambda_norm=config.lambda_norm, native_clamp=config.native_clamp)
            if all(torch.equal(trial_v[layer], accepted_v[layer].detach()) for layer in accepted_v):
                stop_reason = "identical_v"
                break
            backward = calls + 1 < config.candidate_cap
            for value in trial_v.values():
                value.requires_grad_(backward)
            inner = distance_sq = 0.0
            for layer, value in accepted_v.items():
                delta = trial_v[layer].detach().double() - value.detach().double()
                inner += float((accepted_g[layer].double() * delta).sum())
                distance_sq += float(delta.square().sum())
            rhs = accepted_f + inner + distance_sq / (2.0 * tau) + config.epsilon_num
            trial_f, trial_g, trial_payload = evaluate(trial_v, backward)
            trial_norm = norm_penalty(trial_v, anchors, config.lambda_norm)
            accepted = trial_f <= rhs
            ledger.append(dict(evaluation=calls, kind="trial", backward=backward, accepted=accepted,
                               smooth=trial_f, norm=trial_norm, composite=trial_f + trial_norm,
                               tau=tau, majorization_rhs=rhs, majorization_gap=trial_f - rhs,
                               displacement_l2=math.sqrt(distance_sq), epsilon_num=config.epsilon_num,
                               stats=_payload_stats(trial_payload)))
            if on_trial is not None:
                on_trial(dict(ledger[-1]))
            if accepted:
                accepted_updates += 1
                accepted_v, accepted_f, accepted_g, accepted_payload = trial_v, trial_f, trial_g, trial_payload
                accepted_evaluation, accepted_tau = calls, tau
                if calls < config.candidate_cap:
                    max_grad = _max_group_norm(accepted_g)
                    tau = min(accepted_tau, config.nu / max_grad) if max_grad > 0 else accepted_tau
            else:
                rejected_trials += 1
                # Halve the currently active tau, not a newly initialized one.
                tau *= 0.5
            # Accepted objects have their independent reference above; a
            # rejected payload cannot accumulate across trials or the ledger.
            del trial_v, trial_f, trial_g, trial_payload
    return SolverResult(v=accepted_v, payload=accepted_payload, smooth=accepted_f, grad=accepted_g,
                        ledger=ledger, calls=calls, backward_calls=backward_calls,
                        accepted_updates=accepted_updates, rejected_trials=rejected_trials,
                        stop_reason=stop_reason, accepted_tau=accepted_tau,
                        accepted_evaluation=accepted_evaluation)
