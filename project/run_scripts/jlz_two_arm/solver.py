"""Pinned SPG/BB mathematics with the two-arm call/finite-return policy.

The numerical algorithm comes from ``jlz_pilot.solver`` at c91962dd, whose
file SHA is f5dbfe0226b0993104367901e59bb97e173f4019eff79de1a928c0f2ec735eae.
The later ``jlz_efficiency.budget`` is accounting infrastructure, not evidence
of a different executed optimizer. No weights or history are committed here.

``fun`` must be a non-mutating oracle. If it uses a fast route, ``final_fun``
must name the original materialized oracle. There are no implicit rechecks;
explicit original-route rechecks must use ``budgeted_oracle_evaluate`` and the
same account. Every oracle invocation is charged before it starts, including
failed trials. One call is reserved for the fresh original final evaluation.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import torch

from project.run_scripts.jlz_efficiency.budget import BudgetAccountant
from project.run_scripts.jlz_pilot.solver import (
    _detach_payload, block_kkt, prox_blocks,
)


def _payload_scalar_summary(payload):
    """Keep only small loss summaries, never candidate weights or tensors."""
    summary = {}
    for key in ("nll", "kl", "general", "replay", "omega"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, (int, float)):
            summary[key] = float(value) if math.isfinite(float(value)) else "NONFINITE_RECORDED"
            continue
        if not isinstance(value, (list, tuple)):
            summary[key] = "NOT_RECORDED"
            continue
        scalars = [float(v["value"] if isinstance(v, dict) else v) for v in value]
        if not all(math.isfinite(v) for v in scalars):
            summary[key] = dict(status="NONFINITE_RECORDED", count=len(scalars))
        else:
            summary[key] = dict(count=len(scalars), sum=sum(scalars),
                                mean=sum(scalars) / len(scalars) if scalars else None,
                                minimum=min(scalars) if scalars else None,
                                maximum=max(scalars) if scalars else None)
    return summary


def budgeted_oracle_evaluate(fun, point, account, trace, *, role):
    """Debit once and evaluate without giving the oracle an alias of state.

State/CUDA/runtime/shape failures are not converted to numerical warnings.
The input-copy check also runs on exceptions: a mutated oracle input is a
structural failure, even if that oracle raised ``FloatingPointError``.
"""
    account.charge(role)
    event = dict(call=account.used, role=role, outcome="STARTED")
    trace.append(event)
    started = time.monotonic()
    argument = point.detach().clone()
    try:
        try:
            value, gradient, payload = fun(argument)
        finally:
            if not torch.equal(argument, point):
                raise RuntimeError("ORACLE_INPUT_MUTATION")
        value = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
        if not isinstance(gradient, torch.Tensor) or gradient.shape != point.shape:
            raise ValueError("oracle gradient must have the same shape as x")
        gradient = gradient.detach().to(device=point.device, dtype=point.dtype).clone()
        if not math.isfinite(value) or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("NONFINITE_ORACLE")
        # Preserve the executed sequential policy's checks on commit payloads.
        if isinstance(payload, dict):
            for name in ("nll", "kl"):
                if not all(math.isfinite(float(v)) for v in payload.get(name, [])):
                    raise FloatingPointError("NONFINITE_ORACLE_PAYLOAD_" + name)
            if not all(bool(torch.isfinite(w).all()) for w in payload.get("weights", {}).values()):
                raise FloatingPointError("NONFINITE_ORACLE_PAYLOAD_weights")
        event.update(outcome="FINITE", smooth=value)
        return value, gradient, _detach_payload(payload)
    except BaseException as error:
        event.update(outcome="ERROR", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        event["wall_seconds"] = time.monotonic() - started


def solve(
    fun: Callable,
    x0: torch.Tensor,
    c: torch.Tensor,
    rho: torch.Tensor,
    mask: torch.Tensor,
    *,
    cap: int = 120,
    final_fun: Callable | None = None,
    tol: float = 1e-4,
    max_trials: int = 50,
    memory: int = 10,
    sigma: float = 1e-4,
    account: BudgetAccountant | None = None,
) -> dict:
    """Return the freshly evaluated last accepted/initial finite candidate.

Finite budget stops, failed line searches, and precision stalls are eligible;
convergence remains an explicitly reported numerical verdict. A nonfinite
trial rejects and halves its step. Initial/final nonfinite is fatal. The
caller remains responsible for exact final weight/history/commit identity.
"""
    started = time.monotonic()
    if cap < 2:
        raise ValueError("RETURN_EVALUATION_RESERVE: cap must be at least 2")
    if x0.ndim != 2 or x0.dtype != torch.float32 or 0 in x0.shape:
        raise ValueError("x0 must be a nonempty FP32 (nblocks, hidden_size) tensor")
    if not math.isfinite(tol) or tol < 0 or max_trials < 1 or memory < 1 or not 0 < sigma < 1:
        raise ValueError("invalid solver controls")
    c = torch.as_tensor(c, device=x0.device, dtype=x0.dtype).detach().clone()
    rho = torch.as_tensor(rho, device=x0.device, dtype=x0.dtype).detach().clone()
    mask = torch.as_tensor(mask, device=x0.device, dtype=torch.bool).detach().clone()
    if c.shape != x0.shape[:1] or rho.shape != c.shape or mask.shape != c.shape:
        raise ValueError("c, rho, mask must each have shape (nblocks,)")
    if not all(bool(torch.isfinite(a).all()) for a in (x0, c, rho)) or bool((c < 0).any()) or bool((rho < 0).any()):
        raise ValueError("initial state must be finite; c and rho must be nonnegative")

    account = BudgetAccountant(cap) if account is None else account
    if account.cap != cap or account.used:
        raise ValueError("FRESH_SHARED_ACCOUNT_REQUIRED")
    final_fun = fun if final_fun is None else final_fun
    trace, history, branches, boundary_ambiguities, milestone_trace = [], [], [], [], []
    milestone_levels = (8, 16, 32)
    status, reason = "BUDGET_STOP", "CALL_CAP"
    backs, accepted, nonfinite_trials = 0, 0, 0
    x = prox_blocks(x0.detach().clone(), 0.0, c, rho, mask)

    def decay(point):
        return float((c * torch.linalg.vector_norm(point, dim=1)).sum())

    def residual(point, gradient):
        mapped = point - prox_blocks(point - gradient, 1.0, c, rho, mask)
        return float(mapped.double().norm())

    def evaluate(oracle, point, role):
        return budgeted_oracle_evaluate(oracle, point, account, trace, role=role)

    def fatal_evaluate(oracle, point, role):
        try:
            return evaluate(oracle, point, role)
        except BaseException as error:
            # A lightweight error receipt; callers can persist this in their
            # immutable failure output without serializing any solver tensor.
            error.solver_evidence = dict(phase=role, budget=account.report(), trace=trace,
                                         milestone_trace=milestone_trace)
            raise

    def record_milestone(trigger, *, fresh_original_final=False):
        if account.used not in milestone_levels or any(r["actual_calls"] == account.used for r in milestone_trace):
            return
        layer_norms = "NOT_APPLICABLE_NON_5B_FIXTURE"
        if x.shape[0] % 5 == 0:
            batch = x.shape[0] // 5
            layer_norms = [dict(layer=4 + i, R_norm=float(x[i * batch:(i + 1) * batch].double().norm()))
                           for i in range(5)]
        penalty = decay(x)
        raw_residual = residual(x, gradient)
        milestone_trace.append(dict(
            actual_calls=account.used, solver_cap=cap, trigger=trigger,
            interpretation="SAME_RUN_ACCEPTED_INCUMBENT_PREFIX_NOT_INDEPENDENT_CAP_RERUN",
            fresh_original_final=fresh_original_final,
            incumbent_evaluation_call=incumbent_evaluation_call,
            accepted_steps=accepted, total=f + penalty, smooth=f, decay=penalty,
            gradient_norm=float(gradient.double().norm()), residual=raw_residual,
            normalized_residual=raw_residual / scale, loss_summary=incumbent_scalars,
            layer_R_norms=layer_norms, extra_oracle_calls=0, saved_tensor_or_weight=False,
        ))

    f, gradient, initial_payload = fatal_evaluate(fun, x, "initial")
    incumbent_scalars = _payload_scalar_summary(initial_payload)
    incumbent_evaluation_call = account.used
    del initial_payload
    scale = max(1.0, float(gradient[mask].double().norm()))
    step = 1.0 / max(residual(x, gradient), 1e-12)
    history.append(f + decay(x))
    while account.remaining_trials > 0:
        normalized = residual(x, gradient) / scale
        if abs(normalized - tol) <= 5e-6:
            boundary_ambiguities.append("PROX_NEAR_TOL")
        if normalized <= tol:
            status, reason = "CONVERGED", "PROX_RESIDUAL"
            break
        ok = False
        for trial_index in range(max_trials):
            if account.remaining_trials <= 0:
                break
            candidate = prox_blocks(x - step * gradient, step, c, rho, mask)
            if not bool(torch.isfinite(candidate).all()):
                # No oracle was invoked for this invalid proposal.
                trace.append(dict(role="prox", outcome="NONFINITE_REJECTED",
                                  trial_index=trial_index, step=step))
                nonfinite_trials += 1
                backs += 1
                step *= 0.5
                continue
            try:
                candidate_f, candidate_gradient, candidate_payload = evaluate(fun, candidate, "trial")
            except FloatingPointError:
                trace[-1].update(decision="REJECT_NONFINITE_TRIAL", step=step)
                nonfinite_trials += 1
                backs += 1
                record_milestone("NONFINITE_TRIAL_REJECTED")
                step *= 0.5
                continue
            candidate_value = candidate_f + decay(candidate)
            displacement = candidate - x
            decrease = sigma / (2.0 * step) * float(displacement.double().square().sum())
            margin = max(history[-memory:]) - decrease - candidate_value
            branches.append(bool(margin >= 0))
            if abs(margin) <= 2e-3:
                boundary_ambiguities.append("ARMIJO_NEAR_BAND")
            trace[-1].update(step=step, armijo_margin=margin,
                             decision="ACCEPT" if margin >= 0 else "REJECT_ARMIJO")
            if candidate_value <= max(history[-memory:]) - decrease:
                candidate_scalars = _payload_scalar_summary(candidate_payload)
                del candidate_payload
                ok = True
                break
            del candidate_payload
            backs += 1
            record_milestone("ARMIJO_TRIAL_REJECTED")
            step *= 0.5
        if not ok:
            status, reason = (("BUDGET_STOP", "CALL_CAP") if account.remaining_trials <= 0
                              else ("LINESEARCH_FAILED", "TRIAL_LIMIT"))
            break
        change = candidate_gradient - gradient
        sy = float((displacement.double() * change.double()).sum())
        ss = float(displacement.double().square().sum())
        step = min(1e12, max(1e-12, ss / sy)) if sy > 0 else 1e12
        x, f, gradient = candidate, candidate_f, candidate_gradient
        incumbent_scalars = candidate_scalars
        incumbent_evaluation_call = account.used
        accepted += 1
        history.append(candidate_value)
        record_milestone("TRIAL_ACCEPTED")
        if len(history) >= 7:
            window = history[-6:]
            floor = 10 * torch.finfo(x.dtype).eps * max(max(map(abs, window)), torch.finfo(x.dtype).tiny)
            adjacent = max(abs(a - b) for a, b in zip(window, window[1:]))
            improvement = window[0] - min(window)
            if adjacent <= floor and improvement <= floor:
                status, reason = "STALLED_AT_PRECISION", "SIX_ACCEPTED_FP32_PLATEAU"
                break

    f, gradient, payload = fatal_evaluate(final_fun, x, "final")
    incumbent_scalars = _payload_scalar_summary(payload)
    incumbent_evaluation_call = account.used
    normalized = residual(x, gradient) / scale
    penalty = decay(x)
    if not all(math.isfinite(v) for v in (normalized, f + penalty, penalty)):
        error = FloatingPointError("NONFINITE_FINAL_SCALAR")
        error.solver_evidence = dict(phase="final", budget=account.report(), trace=trace)
        raise error
    if bool((x.norm(dim=1) > rho * (1 + 1e-6)).any()) or bool((x[~mask] != 0).any()):
        raise RuntimeError("FEASIBILITY_FAILED")
    if status == "CONVERGED" and normalized > tol:
        status, reason = "NOT_CONVERGED", "RETURN_RESIDUAL_CHANGED"
    elif status == "BUDGET_STOP" and normalized <= tol:
        status, reason = "CONVERGED", "PROX_RESIDUAL"
    if not bool(mask.any()):
        status, reason = "POLICY_ZERO_STEP", "ALL_COLUMNS_INACTIVE"
    record_milestone("FRESH_ORIGINAL_FINAL", fresh_original_final=True)
    evidence = block_kkt(x, gradient, c, rho, mask)
    budget = account.report()
    return dict(
        x=x.detach(), gradient=gradient.detach(), final_payload=payload,
        status=status, reason=reason, calls=account.used, trace=trace, budget=budget,
        account=budget, value=f + penalty, smooth=f, decay=penalty,
        normalized_residual=normalized, gradient_scale=scale, history=history,
        branches=branches, boundary_ambiguities=boundary_ambiguities,
        backtracks=backs, accepted_steps=accepted, nonfinite_trials=nonfinite_trials,
        elapsed=time.monotonic() - started, final_recomputed=True, kkt=evidence,
        feasibility_violation=max(row["feasibility_violation"] for row in evidence),
        commit_eligible=True, commit_authority="USER_FIXED_BUDGET_RETURN_CANDIDATE",
        numerical_fidelity_policy="RECORD_ONLY_USER_DIRECTED",
        convergence_original_verdict="PASS" if normalized <= tol else "FAIL",
        numerical_certification="NOT_ESTABLISHED",
        milestone_trace=milestone_trace,
        milestone_not_reached=[n for n in milestone_levels if n <= cap and
                               n not in {r["actual_calls"] for r in milestone_trace}],
    )
