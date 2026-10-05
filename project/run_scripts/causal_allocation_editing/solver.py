"""Full-gradient proximal BB1 on the actual causal candidate objective.

The engine owns RAM-only materialized payloads.  Primal trials are evaluated
once; only initial/accepted payloads receive a reverse replay.  Norms belong
solely to this proximal map, never to the engine's smooth gradient.
"""
from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Mapping

import torch


class SolverError(RuntimeError):
    def __init__(self, status, detail=None):
        self.status, self.detail = status, detail
        super().__init__(status if detail is None else f"{status}: {detail}")


@dataclass(frozen=True)
class SolverLimits:
    max_evaluations: int = 50
    max_updates: int = 24
    max_gradients: int = 25

    def validate(self):
        # Smaller limits are useful for CPU fixtures; production uses defaults.
        if not (1 <= self.max_evaluations <= 50 and 0 <= self.max_updates <= 24
                and 1 <= self.max_gradients <= 25):
            raise SolverError("INVALID_SOLVER_BUDGET")


@dataclass
class SolverResult:
    u: dict
    payload: object
    gradient: dict
    status: str
    receipt: dict

    def __getitem__(self, key):
        return getattr(self, key)


def _require(condition, status, detail=None):
    if not condition:
        raise SolverError(status, detail)


def _clone(values):
    return {layer: value.detach().clone() for layer, value in values.items()}


def _scalar(value, name):
    if isinstance(value, torch.Tensor):
        _require(value.numel() == 1, "NONSCALAR_OBJECTIVE", name)
        value = float(value.detach().double())
    else:
        value = float(value)
    _require(math.isfinite(value), "NONFINITE_OBJECTIVE", name)
    return value


def _validate_blocks(u, anchors, radius):
    _require(bool(u) and set(u) == set(anchors), "BLOCK_ANCHOR_IDENTITY")
    _require(math.isfinite(radius) and radius > 0, "INVALID_LOCAL_RADIUS")
    sizes = set()
    for layer, value in u.items():
        _require(isinstance(value, torch.Tensor) and value.ndim == 2,
                 "INVALID_BLOCK_SHAPE", layer)
        _require(value.is_floating_point() and bool(torch.isfinite(value).all()),
                 "NONFINITE_PLAN", layer)
        anchor = torch.as_tensor(anchors[layer], device=value.device)
        _require(anchor.ndim == 1 and anchor.shape[0] == value.shape[1],
                 "ANCHOR_SHAPE", layer)
        _require(bool(torch.isfinite(anchor).all()) and bool((anchor > 0).all()),
                 "INVALID_NATIVE_ANCHOR", layer)
        sizes.add(value.shape[1])
        violation = value.detach().double().norm(dim=0) - radius
        _require(not violation.numel() or float(violation.max()) <= 1e-6 * max(1., radius),
                 "REQUESTED_CAP_VIOLATION", layer)
    _require(len(sizes) == 1 and next(iter(sizes)) > 0, "LOGICAL_B_IDENTITY")


def norm_value(u, anchors, coefficient=.5):
    """G in u-coordinates; every layer/request term appears exactly once."""
    _require(math.isfinite(coefficient) and coefficient >= 0, "INVALID_NORM_PRICE")
    return sum(float((value.detach().double().norm(dim=0) *
                      (coefficient / torch.as_tensor(anchors[layer],
                       device=value.device).double())).sum()) for layer, value in u.items())


def prox(u, gradient, anchors, eta, radius=.75, coefficient=.5):
    """FP64 exact block norm+ball prox followed by the plan's stored dtype."""
    _require(math.isfinite(eta) and eta > 0, "INVALID_PROX_STEP")
    _require(set(gradient) == set(u), "GRADIENT_BLOCK_IDENTITY")
    result = {}
    for layer, value in u.items():
        g = gradient[layer]
        _require(g.shape == value.shape and bool(torch.isfinite(g).all()),
                 "NONFINITE_OR_INVALID_GRADIENT", layer)
        proposal = value.detach().double() - eta * g.detach().double()
        lengths = proposal.norm(dim=0)
        beta = coefficient / torch.as_tensor(anchors[layer], device=value.device).double()
        new_lengths = torch.minimum(torch.full_like(lengths, radius),
                                    torch.clamp(lengths - eta * beta, min=0))
        # No epsilon, no hidden additional rescaling/reprojection after cast.
        denominator = torch.where(lengths == 0, torch.ones_like(lengths), lengths)
        result[layer] = (proposal * (new_lengths / denominator)[None, :]).to(value.dtype)
    _validate_blocks(result, anchors, radius)
    return result


def mapping(u, gradient, anchors, tau, radius=.75, coefficient=.5):
    """Fixed tau=eta0 mapping; uses the already computed accepted gradient."""
    if tau is None:
        _require(all(not bool(torch.count_nonzero(g)) for g in gradient.values()),
                 "ZERO_INITIAL_BRANCH_IDENTITY")
        return dict(tau=None, eta0=None, raw_max=0., raw_rms=0., raw_frobenius=0.,
                    normalized_max=0., denominator_max=None, branch="EXACT_ZERO_INITIAL_GRADIENT")
    proposal = prox(u, gradient, anchors, tau, radius, coefficient)
    raw_max = normalized = denominator_max = squares = 0.
    components = 0
    per_layer = {}
    for layer, value in u.items():
        residual = (value.detach().double() - proposal[layer].double()) / tau
        blocks = residual.norm(dim=0)
        gnorm = gradient[layer].detach().double().norm(dim=0)
        beta = coefficient / torch.as_tensor(anchors[layer], device=value.device).double()
        denominator = torch.maximum(torch.ones_like(gnorm), torch.maximum(gnorm, beta))
        ratios = blocks / denominator
        layer_max = float(blocks.max()); layer_normalized = float(ratios.max())
        raw_max = max(raw_max, layer_max); normalized = max(normalized, layer_normalized)
        denominator_max = max(denominator_max, float(denominator.max()))
        squares += float((residual * residual).sum()); components += residual.numel()
        per_layer[str(layer)] = dict(raw_max=layer_max, normalized_max=layer_normalized,
                                    denominator_min=float(denominator.min()),
                                    denominator_max=float(denominator.max()))
    return dict(tau=tau, eta0=tau, raw_max=raw_max,
                raw_rms=math.sqrt(squares / components), raw_frobenius=math.sqrt(squares),
                normalized_max=normalized, denominator_max=denominator_max,
                per_layer=per_layer, branch="FIXED_INITIAL_TAU")


def _dot(a, b):
    return sum(float((a[l].detach().double() * b[l].detach().double()).sum()) for l in a)


def _difference(a, b):
    return {l: a[l].detach().double() - b[l].detach().double() for l in a}


def block_activity(u, previous_accepted=None, radius=.75):
    """Scalar-only activity relative to the last ACCEPTED state, not a trial."""
    result = {}
    for layer, value in u.items():
        lengths = value.detach().double().norm(dim=0)
        active = lengths > 0
        previous = (torch.zeros_like(active) if previous_accepted is None else
                    previous_accepted[layer].detach().double().norm(dim=0) > 0)
        result[str(layer)] = dict(active_blocks=int(active.sum()),
            cap_contacts=int((lengths >= radius-1e-6).sum()),
            reentry_from_previous_accepted=int((active & ~previous).sum()),
            zero_blocks=int((~active).sum()))
    return result


def bb_step(u, previous_u, gradient, previous_gradient, last_eta, eta0):
    """BB1 contains only smooth gradients and accepted states."""
    lo, hi = 1e-6 * eta0, 1e6 * eta0
    reason, curvature, squared = "LAST_ACCEPTED_ETA", None, None
    eta = last_eta
    if previous_u is not None:
        s = _difference(u, previous_u); y = _difference(gradient, previous_gradient)
        curvature, squared = _dot(s, y), _dot(s, s)
        if math.isfinite(curvature) and curvature > 0 and math.isfinite(squared):
            candidate = squared / curvature
            if math.isfinite(candidate) and candidate > 0:
                eta, reason = candidate, "POSITIVE_FINITE_BB1"
    eta = min(hi, max(lo, eta))
    _require(math.isfinite(eta) and eta > 0, "NONFINITE_BB_STEP")
    return eta, dict(reason=reason, sTy=curvature, sTs=squared, eta=eta,
                     safeguard_lower=lo, safeguard_upper=hi)


def _gradient(result, u):
    g = result.get("gradient")
    _require(isinstance(g, Mapping) and set(g) == set(u), "FULL_GRADIENT_MISSING")
    out = {}
    for layer, value in u.items():
        _require(g[layer].shape == value.shape and bool(torch.isfinite(g[layer]).all()),
                 "NONFINITE_OR_INVALID_GRADIENT", layer)
        out[layer] = g[layer].detach().clone()
    return out


def _smooth(result, lambda_Q):
    if "smooth_sum" in result:
        return _scalar(result["smooth_sum"], "smooth_sum")
    return _scalar(result["task_sum"], "task_sum") + lambda_Q * _scalar(result["Q"], "Q")


def solve(objective, u0, anchors, lambda_Q, emit=None, *, limits=None,
          radius=.75, norm_coefficient=.5):
    """Return last accepted RAM payload without a second terminal evaluation.

    Engine API: evaluate(u, gradient, lambda_Q, candidate_id) and
    backward(payload, lambda_Q).  ``backward`` replays this SAME payload, does
    not create a new logical candidate or modify persistent W/H.
    """
    _require(math.isfinite(lambda_Q) and lambda_Q > 0, "INVALID_FIXED_PRICE")
    limits = limits or SolverLimits(); limits.validate()
    _validate_blocks(u0, anchors, radius)
    _require(all(not bool(torch.count_nonzero(v)) for v in u0.values()), "NONZERO_BATCH_INITIAL_PLAN")
    u = _clone(u0); B = next(iter(u.values())).shape[1]
    started = time.monotonic(); events = []
    evaluations = gradients = accepted = rejected = 0
    previous_u = previous_g = None

    def record(event):
        events.append(event)
        if emit is not None:
            emit(event)

    initial = objective.evaluate(u, gradient=True, lambda_Q=lambda_Q, candidate_id=0)
    evaluations = gradients = 1
    g = _gradient(initial, u); smooth = _smooth(initial, lambda_Q)
    G = norm_value(u, anchors, norm_coefficient); J = smooth + G
    _require(math.isfinite(J), "NONFINITE_OBJECTIVE", "J_sum")
    payload = initial["payload"]; accepted_id = 0
    max_gradient = max(float(v.detach().double().norm(dim=0).max()) for v in g.values())
    eta0 = None if max_gradient == 0 else radius / max_gradient
    _require(eta0 is None or math.isfinite(eta0) and eta0 > 0, "INVALID_INITIAL_STEP")
    last_eta = eta0; window = deque([J], maxlen=5)
    record(dict(candidate_id=0, accepted=True, initial=True, full_gradient=True,
                smooth_sum=smooth, norm_sum=G, J_sum=J, J_mean=J / B,
                activity=block_activity(u),
                eta=None, logical_evaluations=1, full_gradients=1, accepted_updates=0))
    # Keep only the accepted payload, not an extra original result dictionary
    # pinning candidate0 after later accepts.  Peak is current+one trial.
    del initial
    status = None
    while status is None:
        terminal_mapping = mapping(u, g, anchors, eta0, radius, norm_coefficient)
        if terminal_mapping["normalized_max"] <= 1e-3:
            status = "STATIONARY_TOL"; break
        if accepted >= limits.max_updates or gradients >= limits.max_gradients:
            status = "BUDGET_NOT_STATIONARY"; break
        if evaluations >= limits.max_evaluations:
            status = "LINE_SEARCH_BUDGET" if not events[-1]["accepted"] else "BUDGET_NOT_STATIONARY"; break
        eta, bb = bb_step(u, previous_u, g, previous_g, last_eta, eta0)
        lower = 1e-6 * eta0
        previous_trial = None
        while True:
            proposal = prox(u, g, anchors, eta, radius, norm_coefficient)
            difference = _difference(proposal, u); squared_step = _dot(difference, difference)
            if all(torch.equal(proposal[l], u[l]) for l in u):
                status = "NUMERICAL_STALL"; break
            if previous_trial is not None and all(torch.equal(proposal[l], previous_trial[l]) for l in u):
                status = "NUMERICAL_STALL"; break
            candidate_id = evaluations
            trial = objective.evaluate(proposal, gradient=False, lambda_Q=lambda_Q,
                                       candidate_id=candidate_id)
            evaluations += 1
            trial_smooth = _smooth(trial, lambda_Q)
            trial_G = norm_value(proposal, anchors, norm_coefficient)
            trial_J = trial_smooth + trial_G
            _require(math.isfinite(trial_J), "NONFINITE_OBJECTIVE", "trial_J_sum")
            reference = max(window)
            bound = reference - 1e-4 * squared_step / (2 * eta)
            is_accepted = trial_J <= bound
            event = dict(candidate_id=candidate_id, accepted=is_accepted,
                         initial=False, full_gradient=is_accepted,
                         smooth_sum=trial_smooth, norm_sum=trial_G, J_sum=trial_J,
                         J_mean=trial_J / B, eta=eta, armijo_reference=reference,
                         activity=block_activity(proposal,u),
                         armijo_bound=bound, squared_step=squared_step, bb=bb,
                         logical_evaluations=evaluations,
                         full_gradients=gradients + int(is_accepted),
                         accepted_updates=accepted + int(is_accepted))
            if is_accepted:
                reverse = objective.backward(trial["payload"], lambda_Q=lambda_Q)
                new_g = _gradient(reverse, proposal)
                # The primal payload is the exact accepted state, not replay's
                # materialized tensor or a freshly solved replacement.
                previous_u, previous_g = u, g
                u, g, payload = proposal, new_g, trial["payload"]
                del trial, reverse
                smooth, G, J = trial_smooth, trial_G, trial_J
                accepted += 1; gradients += 1; accepted_id = candidate_id
                last_eta = eta; window.append(J)
                record(event)
                break
            rejected += 1; record(event)
            previous_trial = proposal
            # Explicitly release rejected RAM payload; never route its adjoint.
            del trial
            if evaluations >= limits.max_evaluations:
                status = "LINE_SEARCH_BUDGET"; break
            if eta <= lower:
                status = "NUMERICAL_STALL"; break
            eta = max(lower, .5 * eta)
    terminal_mapping = mapping(u, g, anchors, eta0, radius, norm_coefficient)
    _validate_blocks(u, anchors, radius)
    _require(evaluations <= 50 and accepted <= 24 and gradients <= 25
             and gradients == accepted + 1, "SOLVER_COUNTER_INTEGRITY")
    receipt = dict(status=status, B=B, logical_evaluations=evaluations,
                   accepted_updates=accepted, rejected_trials=rejected,
                   full_gradients=gradients, accepted_candidate_id=accepted_id,
                   last_evaluated_candidate_id=evaluations - 1,
                   terminal_gradient_reused=True, terminal_extra_gradient=0,
                   terminal_extra_evaluation=0, eta0=eta0, tau=eta0,
                   last_accepted_eta=last_eta, mapping=terminal_mapping,
                   smooth_sum=smooth, norm_sum=G, J_sum=J, J_mean=J / B,
                   lambda_Q=lambda_Q, norm_gradient_in_smooth=False,
                   norm_prox_once=True, shared_sum_cap=None, per_block_radius=radius,
                   logical_gradient_reduction="REQUEST_SUM",
                   seconds=time.monotonic() - started, events=events,
                   checkpoint_saved=False)
    return SolverResult(u, payload, g, status, receipt)
