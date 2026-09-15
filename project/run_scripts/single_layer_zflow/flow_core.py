"""SL-zFlow mathematical reference kernel (not a Llama/GPU integration).

State X: [hidden_size, number_of_requests]. Native write: dW = X @ B.
Smooth oracle: returns L(X), dL/dX, excluding the analytic quadratic cost.
Each oracle call is a COMPLETE logical loss/gradient pass, not one microbatch.

This implementation is intended for small FP64 correctness tests. In deployment,
retain the pinned model dtype and test candidate/materialized numerical parity.
The native system is generally nonsymmetric; it is solved without symmetrizing.
The small quadratic S, in contrast, must be symmetric positive semidefinite.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional
import torch

Tensor = torch.Tensor
Oracle = Callable[[Tensor], tuple[float, Tensor]]


def _number(name: str, value: float, *, positive: bool = False,
            nonnegative: bool = False) -> None:
    if (not isinstance(value, (float, int)) or isinstance(value, bool) or
        not math.isfinite(value) or (positive and value <= 0) or
        (nonnegative and value < 0)):
        raise ValueError(f'invalid {name}')


def _barrier_budget(budget: Optional[float], mode: str) -> Optional[float]:
    if mode not in ('off', 'fixed', 'exponential'):
        raise ValueError('barrier_mode must be off, fixed, or exponential')
    if budget is not None:
        _number('budget (zero is a separate no-edit policy)', budget, positive=True)
    return None if mode == 'off' else budget


def native_map(K: Tensor, P: Tensor, M: Tensor, ridge: float,
               E: Optional[Tensor] = None) -> Tensor:
    """K=[d_in,q], E=[m,q] encodes the actual native residual expansion.

    Use E=I only when there is exactly one writer key column per request.
    No covariance approximation, inverse, or native-system Cholesky is used.
    """
    if (K.ndim != 2 or min(K.shape) == 0 or
        P.shape != (K.shape[0], K.shape[0]) or M.shape != P.shape):
        raise ValueError('incompatible K/P/M shapes')
    _number('ridge', ridge, positive=True)
    if K.dtype not in (torch.float32, torch.float64):
        raise ValueError('reference geometry requires float32 or float64')
    if E is None:
        E = torch.eye(K.shape[1], dtype=K.dtype, device=K.device)
    if E.ndim != 2 or E.shape[0] == 0 or E.shape[1] != K.shape[1]:
        raise ValueError('E must encode native key-column/request binding')
    arrays = [K, P, M, E]
    if any(a.dtype != K.dtype or a.device != K.device or not torch.isfinite(a).all()
           for a in arrays):
        raise ValueError('all arrays must be finite with one dtype/device')
    system = P @ (K @ K.T + M) + ridge * torch.eye(
        K.shape[0], dtype=K.dtype, device=K.device)
    solved = torch.linalg.solve(system, P @ K)
    B = E @ solved.T
    if not torch.isfinite(B).all():
        raise FloatingPointError('nonfinite native map')
    return B


def preservation_gram(B: Tensor, M: Tensor, ridge: float) -> Tensor:
    """S gives C(X)=0.5/m * (tr(dW M dW.T)+ridge*||dW||_F^2)."""
    _number('ridge', ridge, positive=True)
    if (B.ndim != 2 or min(B.shape) == 0 or
        M.shape != (B.shape[1], B.shape[1]) or
        B.dtype not in (torch.float32, torch.float64) or
        M.dtype != B.dtype or M.device != B.device or
        not torch.isfinite(B).all() or not torch.isfinite(M).all()):
        raise ValueError('invalid preservation geometry')
    raw = (B @ M @ B.T + ridge * (B @ B.T)) / B.shape[0]
    # Mathematical S is symmetric; this is not the nonsymmetric native system.
    return (raw + raw.T) * 0.5


@dataclass
class Candidate:
    x: Tensor
    dual: float
    allowed_cost: float
    cost: float
    scalar_iterations: int


class QuadraticGeometry:
    """S=U diag(s) U.T and H=S+epsilon*I; eigendecomposition once/batch.

    In this basis scalar barrier searches take O(m) work per iteration,
    after O(d*m) column-energy reduction. No model calls occur here.
    """
    def __init__(self, S: Tensor, metric_ridge_relative: float = 1e-3,
                 psd_tolerance: float = 1e-10):
        if (S.ndim != 2 or S.shape[0] == 0 or S.shape[0] != S.shape[1] or
            S.dtype not in (torch.float32, torch.float64) or
            not torch.isfinite(S).all()):
            raise ValueError('S must be a finite square matrix')
        _number('metric_ridge_relative', metric_ridge_relative, positive=True)
        _number('psd_tolerance', psd_tolerance, nonnegative=True)
        scale = max(float(S.abs().max()), torch.finfo(S.dtype).tiny)
        if float((S-S.T).abs().max()) > psd_tolerance * scale:
            raise ValueError('S is not numerically symmetric')
        s, U = torch.linalg.eigh((S+S.T)*0.5)
        if float(s.min()) < -psd_tolerance * scale:
            raise ValueError('S is not positive semidefinite')
        self.clipped_negative_eigenvalues = int((s < 0).sum())
        self.s = s.clamp_min(0)
        if float(self.s.max()) == 0:
            raise ValueError('DEGENERATE_WRITER: all preservation eigenvalues are zero')
        self.U = U
        self.epsilon = metric_ridge_relative * float(self.s.mean())
        self.h = self.s + self.epsilon

    def _check(self, x: Tensor) -> None:
        if (not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] == 0 or
            x.shape[1] != self.s.numel() or
            x.dtype != self.U.dtype or x.device != self.U.device or
            not torch.isfinite(x).all()):
            raise ValueError('invalid state or gradient')

    def cost(self, x: Tensor) -> float:
        self._check(x)
        return float(0.5 * ((x @ self.U).square() * self.s).sum())

    def cost_grad(self, x: Tensor) -> Tensor:
        self._check(x)
        return ((x @ self.U) * self.s) @ self.U.T

    def norm_H_squared(self, dx: Tensor) -> float:
        self._check(dx)
        return float(((dx @ self.U).square() * self.h).sum())

    def stationary(self, x: Tensor, g: Tensor, price: float,
                   budget: Optional[float] = None,
                   active_tolerance: float = 1e-8) -> dict[str, float]:
        """First-order residual for the FINAL budget, not a first-hit test.

        Activity uses relative budget slack, including for tiny budgets.
        normalized_complementarity = nu/max(1,nu) * abs((b-C)/b).
        normalized_feasibility = max(0, (C-b)/b). Together with the
        stationarity residual these avoid an absolute-cost tolerance masking
        an interior point. Raw complementarity/feasibility are also reported.
        Small residuals do not certify a minimum or successful editing.
        """
        self._check(x); self._check(g)
        if x.shape != g.shape:
            raise ValueError('state and gradient shapes differ')
        _number('price', price, nonnegative=True)
        _number('active_tolerance', active_tolerance, positive=True)
        budget = _barrier_budget(budget, 'fixed')
        a = (x @ self.U) * self.s
        q = g @ self.U + price * a
        C = self.cost(x)
        nu = 0.0
        relative_slack = 0.0 if budget is None else (budget-C)/budget
        if budget is not None and relative_slack <= active_tolerance:
            den = float((a.square() / self.h).sum())
            if den > 0:
                nu = max(0.0, -float((q*a/self.h).sum()) / den)
        residual = math.sqrt(max(0.0, float(((q+nu*a).square()/self.h).sum())))
        return dict(residual=residual, normal_multiplier=nu,
                    relative_slack=relative_slack,
                    normalized_complementarity=(nu/max(1.0, nu))*abs(relative_slack),
                    normalized_feasibility=max(0.0, -relative_slack),
                    complementarity=0.0 if budget is None else nu*abs(budget-C),
                    feasibility=0.0 if budget is None else max(0.0, C-budget))

    @torch.no_grad()
    def propose(self, x: Tensor, g: Tensor, step: float, price: float,
                budget: Optional[float] = None, kappa: float = 1.0,
                scalar_tolerance: float = 1e-12,
                barrier_mode: str = 'exponential') -> Candidate:
        """Implicit-cost/explicit-loss Euler with an exact discrete barrier.

        exponential: C(Y) <= C(X)+(1-exp(-kappa*step))*(budget-C(X)).
        fixed: C(Y) <= budget. off or budget=None: no constraint.
        At X=0 this still constrains finite steps, even though grad C(0)=0.
        """
        self._check(x); self._check(g)
        if x.shape != g.shape:
            raise ValueError('state and gradient shapes differ')
        _number('step', step, positive=True)
        _number('price', price, nonnegative=True)
        _number('kappa', kappa, positive=True)
        _number('scalar_tolerance', scalar_tolerance, positive=True)
        budget = _barrier_budget(budget, barrier_mode)
        C = self.cost(x)
        cap = math.inf
        if budget is not None:
            if (C-budget)/budget > scalar_tolerance:
                raise ValueError('infeasible entry: preservation is not a recovery guarantee')
            C = min(C, budget)
            cap = (budget if barrier_mode == 'fixed' else
                   C + (-math.expm1(-kappa*step)) * (budget-C))
        numerator = (x @ self.U) * self.h - step * (g @ self.U)
        energies = numerator.square().sum(dim=0)

        def cost_at(dual: float) -> float:
            den = self.h + step*(price+dual)*self.s
            return float(0.5 * (self.s*energies/den.square()).sum())

        dual = 0.0
        iters = 0
        if cost_at(0.0) > cap:
            lo, hi = 0.0, max(1.0, price)
            for _ in range(128):
                iters += 1
                if cost_at(hi) <= cap:
                    break
                hi *= 2
            else:
                raise FloatingPointError('failed to bracket barrier multiplier')
            for _ in range(100):
                iters += 1
                mid = (lo+hi)*0.5
                if cost_at(mid) > cap:
                    lo = mid
                else:
                    hi = mid
                if hi-lo <= scalar_tolerance*max(1.0, hi):
                    break
            dual = hi  # feasible side of the bracket
        yhat = numerator / (self.h + step*(price+dual)*self.s)
        y = yhat @ self.U.T
        if not torch.isfinite(y).all():
            raise FloatingPointError('nonfinite candidate')
        cost = self.cost(y)
        # Check the reconstructed state rather than only its spectral estimate.
        feasibility_roundoff = max(scalar_tolerance, 64*torch.finfo(x.dtype).eps)
        if budget is not None and (cost-cap)/budget > feasibility_roundoff:
            raise FloatingPointError('reconstructed candidate violates barrier')
        return Candidate(y, dual, cap, cost, iters)


@dataclass
class FlowResult:
    x: Tensor
    status: str
    oracle_calls: int
    accepted_steps: int
    rejected_steps: int
    trace: list[dict[str, float]]
    terminal_stats: dict[str, float]


def integrate(oracle: Oracle, x0: Tensor, geom: QuadraticGeometry, *,
              price: float, budget: Optional[float] = None, kappa: float = 1.0,
              initial_step: float = 1.0, max_oracle_calls: int = 128,
              relative_stationarity: float = 1e-5,
              absolute_stationarity: float = 1e-9,
              complementarity_tolerance: float = 1e-7,
              armijo: float = 1e-4, minimum_step: float = 1e-12,
              barrier_mode: str = 'exponential',
              active_tolerance: float = 1e-8,
              feasibility_tolerance: float = 1e-10,
              maximum_step: Optional[float] = None,
              step_growth: float = 1.5,
              growth_after_accepts: int = 2) -> FlowResult:
    """Gradient-carry integrator; no nested z solver and no native-z warm start.

    An accepted candidate's full logical gradient is the next iteration's
    gradient. Rejected candidate F+B calls are counted and discarded. Oracle
    MUST NOT mutate weights, history, teacher, inputs or its own stochastic state.
    Consecutive accepts with decrease above rounding noise allow bounded step
    recovery. At the rounding floor an increasing residual rejects the trial;
    otherwise growing steps can amplify an already small gradient indefinitely.
    All settings and initial feasibility are checked before the first oracle.
    Resource exhaustion is never marked stationarity.
    """
    budget = _barrier_budget(budget, barrier_mode)
    _number('price', price, nonnegative=True)
    for name, value in (('kappa', kappa), ('initial_step', initial_step),
                        ('minimum_step', minimum_step),
                        ('active_tolerance', active_tolerance),
                        ('feasibility_tolerance', feasibility_tolerance),
                        ('complementarity_tolerance', complementarity_tolerance)):
        _number(name, value, positive=True)
    for name, value in (('relative_stationarity', relative_stationarity),
                        ('absolute_stationarity', absolute_stationarity)):
        _number(name, value, nonnegative=True)
    if relative_stationarity + absolute_stationarity == 0:
        raise ValueError('at least one stationarity tolerance must be positive')
    _number('armijo', armijo, positive=True)
    _number('step_growth', step_growth, positive=True)
    maximum_step = initial_step if maximum_step is None else maximum_step
    _number('maximum_step', maximum_step, positive=True)
    if (not isinstance(max_oracle_calls, int) or isinstance(max_oracle_calls, bool) or
        max_oracle_calls < 1 or not 0 < armijo < 0.5 or step_growth < 1 or
        not minimum_step <= initial_step <= maximum_step or
        not isinstance(growth_after_accepts, int) or
        isinstance(growth_after_accepts, bool) or growth_after_accepts < 1):
        raise ValueError('invalid integration settings')
    geom._check(x0)
    initial_cost = geom.cost(x0)
    if not math.isfinite(initial_cost):
        raise ValueError('nonfinite initial cost')
    # The proposal feasibility tolerance is stricter than the stopping tolerance.
    if budget is not None and (initial_cost-budget)/budget > 1e-12:
        raise ValueError('infeasible initial state')
    x = x0.detach().clone()
    L, g = oracle(x)
    geom._check(g)
    if g.shape != x.shape:
        raise ValueError('oracle gradient has the wrong shape')
    g = g.detach()
    L = float(L)
    if not math.isfinite(L):
        raise FloatingPointError('nonfinite initial oracle')
    nfe, accepted, rejected = 1, 0, 0
    step = initial_step
    r0 = geom.stationary(x, g, price, budget, active_tolerance)['residual']
    threshold = absolute_stationarity + relative_stationarity*r0
    trace: list[dict[str, float]] = []
    clean_accepts = 0

    def finish(status: str) -> FlowResult:
        terminal = geom.stationary(x, g, price, budget, active_tolerance)
        C = geom.cost(x)
        terminal.update(L=L, C=C, F=L+price*C,
                        slack=0.0 if budget is None else budget-C,
                        stationarity_threshold=threshold)
        return FlowResult(x, status, nfe, accepted, rejected, trace, terminal)

    while True:
        C = geom.cost(x)
        F = L + price*C
        stats = geom.stationary(x, g, price, budget, active_tolerance)
        if (stats['residual'] <= threshold and
            stats['normalized_complementarity'] <= complementarity_tolerance and
            stats['normalized_feasibility'] <= feasibility_tolerance):
            return finish('FIRST_ORDER_STATIONARY')
        if nfe >= max_oracle_calls:
            return finish('RESOURCE_STOP')
        if step < minimum_step:
            return finish('NUMERICAL_STOP')
        try:
            trial = geom.propose(x, g, step, price, budget, kappa,
                                 barrier_mode=barrier_mode)
        except FloatingPointError:
            return finish('NUMERICAL_STOP')
        trial_L, trial_g = oracle(trial.x)
        nfe += 1
        if (not isinstance(trial_g, Tensor) or trial_g.shape != x.shape or
            trial_g.dtype != x.dtype or trial_g.device != x.device):
            raise ValueError('oracle gradient has the wrong shape, dtype, or device')
        trial_L = float(trial_L)
        trial_F = trial_L + price*trial.cost
        dx2 = geom.norm_H_squared(trial.x-x)
        rounding = 32*torch.finfo(x.dtype).eps*max(1.0, abs(F))
        ok = (math.isfinite(trial_L) and torch.isfinite(trial_g).all().item() and
              trial_F <= F-armijo*dx2/step+rounding)
        roundoff_limited = F-trial_F <= rounding
        if ok and roundoff_limited:
            trial_stats = geom.stationary(trial.x, trial_g, price, budget, active_tolerance)
            ok = trial_stats['residual'] <= max(stats['residual'], threshold)
        trace.append(dict(nfe=float(nfe), accepted=float(ok), step=step,
                          F_before=F, F_trial=trial_F, cost=trial.cost,
                          barrier_cap=trial.allowed_cost, barrier_dual=trial.dual,
                          stationarity_before=stats['residual'],
                          roundoff_limited=float(roundoff_limited)))
        if ok:
            x, L, g = trial.x.detach(), float(trial_L), trial_g.detach()
            accepted += 1
            clean_accepts = 0 if roundoff_limited else clean_accepts+1
            if clean_accepts >= growth_after_accepts:
                step = min(maximum_step, step*step_growth)
                clean_accepts = 0
        else:
            rejected += 1
            step *= 0.5
            clean_accepts = 0
