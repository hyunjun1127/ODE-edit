"""Mechanical instrumented copy of pilot solver f5dbfe02; math unchanged.
Shared accountant supplies evaluate/remaining/final. Static routes only;
near-boundary accelerated trajectories are excluded, never free rechecked.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import torch
from .budget import BudgetAccountant


def _detach_payload(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {k: _detach_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_detach_payload(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_detach_payload(v) for v in value)
    return value


def prox_blocks(v, step, c, rho, active_mask):
    """Exact radial shrinkage then ball projection; inactive blocks are zero."""
    norms = torch.linalg.vector_norm(v, dim=1)
    targets = torch.minimum(rho, (norms - step * c).clamp_min(0))
    ratios = targets / norms.clamp_min(torch.finfo(v.dtype).tiny)
    result = v * ratios[:, None]
    return torch.where(active_mask[:, None], result, torch.zeros_like(result))


def block_kkt(x, gradient, c, rho, active_mask):
    """Signed radial KKT evidence, including singleton and inactive blocks.

At a nonzero ball boundary, radial <= 0 is necessary: the outward normal
multiplier cancels only an inward objective gradient. Reporting its absolute
value, or omitting a positive radial violation, would give a false certificate.
"""
    rows = []
    eps = torch.finfo(x.dtype).eps
    for i in range(x.shape[0]):
        v, g = x[i].double(), gradient[i].double()
        norm = float(v.norm())
        radius, coefficient = float(rho[i]), float(c[i])
        base = dict(block=i, norm=norm, radius=radius,
                    norm_over_radius=norm / radius if radius > 0 else 0.0,
                    feasibility_violation=max(0.0, norm - radius))
        if not bool(active_mask[i]) or radius == 0.0:
            base.update(state="INACTIVE" if not bool(active_mask[i]) else "FIXED_ZERO",
                        stationarity_violation=0.0, feasibility_violation=norm)
        elif norm == 0.0:
            base.update(state="ZERO", stationarity_violation=max(0.0, float(g.norm()) - coefficient),
                        gradient_norm=float(g.norm()), decay_coefficient=coefficient)
        else:
            unit = v / norm
            total = g + coefficient * unit
            radial = float(total @ unit)
            tangent = float((total - radial * unit).norm())
            boundary = abs(norm - radius) <= 8 * eps * max(norm, radius, torch.finfo(x.dtype).tiny)
            violation = max(0.0, radial) if boundary else abs(radial)
            base.update(state="BOUNDARY" if boundary else "INTERIOR", signed_radial=radial,
                        radial_violation=violation, tangential_residual=tangent,
                        clamp_multiplier=max(0.0, -radial) if boundary else 0.0,
                        stationarity_violation=math.hypot(tangent, violation))
        rows.append(base)
    return rows


def solve(
    fun: Callable,
    x0: torch.Tensor,
    c: torch.Tensor,
    rho: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    tol: float = 1e-3,
    cap: int = 80,
    max_seconds: float | None = None,
    max_trials: int = 50,
    memory: int = 10,
    sigma: float = 1e-4,
    account=None,
) -> dict:
    """Minimize smooth(x) + sum_i c_i ||x_i|| subject to block balls.

CONVERGED requires a freshly recomputed normalized unit-step proximal
residual <= tol. FP32 objective plateaus have STALLED_AT_PRECISION status.
This technical solver makes no scientific commit. Caller preserves fixed-budget policy.
Only finite initial state and nonnegative finite c/rho are accepted.
"""
    started = time.monotonic()
    if cap < 2:
        raise ValueError("RETURN_EVALUATION_RESERVE: cap must be at least 2")
    if x0.ndim != 2 or x0.dtype != torch.float32 or 0 in x0.shape:
        raise ValueError("x0 must be a nonempty FP32 (nblocks, hidden_size) tensor")
    if not math.isfinite(tol) or tol < 0 or max_trials < 1 or memory < 1 or not 0 < sigma < 1:
        raise ValueError("invalid solver controls")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds < 0):
        raise ValueError("max_seconds must be nonnegative and finite")
    c = torch.as_tensor(c, device=x0.device, dtype=x0.dtype).detach().clone()
    rho = torch.as_tensor(rho, device=x0.device, dtype=x0.dtype).detach().clone()
    active_mask = torch.as_tensor(active_mask, device=x0.device, dtype=torch.bool).detach().clone()
    if c.shape != x0.shape[:1] or rho.shape != c.shape or active_mask.shape != c.shape:
        raise ValueError("c, rho, active_mask must each have shape (nblocks,)")
    if not all(bool(torch.isfinite(a).all()) for a in (x0, c, rho)) or bool((c < 0).any()) or bool((rho < 0).any()):
        raise ValueError("initial state must be finite; c and rho must be nonnegative")

    account = account or BudgetAccountant(cap)
    if account.cap != cap or account.used: raise ValueError("FRESH_SHARED_ACCOUNT_REQUIRED")
    backs, accepted = 0, 0
    branches=[]; boundary_ambiguities=[]
    status, reason = "BUDGET_STOP", "CALL_CAP"
    scale = 1.0
    final_recomputed = False
    final_payload = None
    x = prox_blocks(x0.detach().clone(), 0.0, c, rho, active_mask)
    f, gradient = math.nan, torch.zeros_like(x)
    history = []

    def evaluate(point, final=False):
        account.charge("final" if final else ("initial" if account.used == 0 else "trial"))
        value, grad, payload = fun(point.detach().clone())
        value = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
        if not isinstance(grad, torch.Tensor) or grad.shape != x.shape:
            raise ValueError("oracle gradient must be a tensor with the same shape as x")
        grad = grad.detach().to(device=x.device, dtype=x.dtype).clone()
        if not math.isfinite(value) or not bool(torch.isfinite(grad).all()):
            raise FloatingPointError("NONFINITE_ORACLE")
        return value, grad, _detach_payload(payload)

    def decay(point):
        return float((c * torch.linalg.vector_norm(point, dim=1)).sum())

    def residual(point, grad):
        mapped = point - prox_blocks(point - grad, 1.0, c, rho, active_mask)
        return float(mapped.double().norm())

    def expired():
        return max_seconds is not None and time.monotonic() - started >= max_seconds

    try:
        f, gradient, discarded_payload = evaluate(x)
        del discarded_payload
        scale = max(1.0, float(gradient[active_mask].double().norm()))
        step = 1.0 / max(residual(x, gradient), 1e-12)
        history.append(f + decay(x))
        while account.remaining_trials > 0:
            if abs(residual(x, gradient) / scale - tol) <= 5e-6: boundary_ambiguities.append("PROX_NEAR_TOL")
            if residual(x, gradient) / scale <= tol:
                status, reason = "CONVERGED", "PROX_RESIDUAL"
                break
            if expired():
                status, reason = "BUDGET_STOP", "TIME_LIMIT"
                break
            ok = False
            for _ in range(max_trials):
                if account.remaining_trials <= 0 or expired():
                    break
                candidate = prox_blocks(x - step * gradient, step, c, rho, active_mask)
                if not bool(torch.isfinite(candidate).all()):
                    raise FloatingPointError("NONFINITE_PROX")
                candidate_f, candidate_gradient, discarded_payload = evaluate(candidate)
                del discarded_payload
                candidate_value = candidate_f + decay(candidate)
                displacement = candidate - x
                decrease = sigma / (2.0 * step) * float(displacement.double().square().sum())
                margin = max(history[-memory:]) - decrease - candidate_value
                branches.append(bool(margin >= 0))
                if abs(margin) <= 2e-3: boundary_ambiguities.append("ARMIJO_NEAR_BAND")
                if candidate_value <= max(history[-memory:]) - decrease:
                    ok = True
                    break
                backs += 1
                step *= 0.5
            if not ok:
                if expired():
                    status, reason = "BUDGET_STOP", "TIME_LIMIT"
                elif account.remaining_trials <= 0:
                    status, reason = "BUDGET_STOP", "CALL_CAP"
                else:
                    status, reason = "LINESEARCH_FAILED", "TRIAL_LIMIT"
                break
            change = candidate_gradient - gradient
            sy = float((displacement.double() * change.double()).sum())
            ss = float(displacement.double().square().sum())
            step = min(1e12, max(1e-12, ss / sy)) if sy > 0 else 1e12
            x, f, gradient = candidate, candidate_f, candidate_gradient
            accepted += 1
            history.append(candidate_value)
            if len(history) >= 7:
                window = history[-6:]
                floor = 10 * torch.finfo(x.dtype).eps * max(max(map(abs, window)), torch.finfo(x.dtype).tiny)
                adjacent = max(abs(a - b) for a, b in zip(window, window[1:]))
                improvement = window[0] - min(window)
                if adjacent <= floor and improvement <= floor:
                    status, reason = "STALLED_AT_PRECISION", "SIX_ACCEPTED_FP32_PLATEAU"
                    break
    except FloatingPointError as error:
        status, reason = "NONFINITE", str(error)

    # Even at a time limit, consume the reserved return call. A failed final
    # oracle leaves no payload that the caller could mistake for commit-ready.
    try:
        f, gradient, final_payload = evaluate(x, final=True)
        final_recomputed = True
        normalized = residual(x, gradient) / scale
        if not math.isfinite(normalized):
            raise FloatingPointError("NONFINITE_RETURN_RESIDUAL")
        if status == "CONVERGED" and normalized > tol:
            status, reason = "NOT_CONVERGED", "RETURN_RESIDUAL_CHANGED"
        elif status == "BUDGET_STOP" and normalized <= tol:
            status, reason = "CONVERGED", "PROX_RESIDUAL"
    except FloatingPointError as error:
        status, reason = "NONFINITE", str(error)
        normalized, final_payload, final_recomputed = math.inf, None, False

    assert account.used <= cap
    evidence = block_kkt(x, gradient, c, rho, active_mask) if final_recomputed else []
    penalty = decay(x)
    return dict(x=x.detach(), gradient=gradient.detach(), status=status, reason=reason, calls=account.used, account=account.report(), branches=branches, boundary_ambiguities=boundary_ambiguities,
                value=f + penalty, smooth=f, decay=penalty, normalized_residual=normalized,
                gradient_scale=scale, backtracks=backs, accepted_steps=accepted,
                elapsed=time.monotonic() - started, final_payload=final_payload,
                final_recomputed=final_recomputed, kkt=evidence, history=history,
                feasibility_violation=max((row["feasibility_violation"] for row in evidence), default=math.inf),
                commit_eligible=status == "CONVERGED" and final_recomputed)

