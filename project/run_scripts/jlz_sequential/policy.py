"""User-authorized admission; optimizer equations remain pinned pilot equations."""
import math
import torch
from project.run_scripts.jlz_pilot.solver import solve as pilot_solve

ALLOWED = {'CONVERGED', 'POLICY_ZERO_STEP', 'BUDGET_STOP', 'STALLED_AT_PRECISION',
           'NOT_CONVERGED', 'LINESEARCH_FAILED'}

def admit(result, rho, mask):
    if result['status'] not in ALLOWED or result.get('any_nonfinite', False):
        raise FloatingPointError('NONFINITE_OR_INVALID_SOLVER_STATUS: '+result['status'])
    if not result.get('final_recomputed') or result.get('final_payload') is None:
        raise RuntimeError('FRESH_FINAL_EVALUATION_REQUIRED')
    x, g = result['x'], result['gradient']
    if not all(bool(torch.isfinite(v).all()) for v in (x, g, rho)):
        raise FloatingPointError('NONFINITE_RETURN')
    if not all(math.isfinite(float(result[k])) for k in ('value', 'smooth', 'decay', 'normalized_residual')):
        raise FloatingPointError('NONFINITE_RETURN_SCALAR')
    if bool((x.norm(dim=1) > rho*(1+1e-6)).any()) or bool((x[~mask] != 0).any()):
        raise RuntimeError('FEASIBILITY_FAILED')
    result['commit_eligible'] = True
    result['commit_authority'] = 'USER_FIXED_BUDGET_RETURN_CANDIDATE'
    return result

def solve(fun, x0, c, rho, mask, *, cap=120, tol=1e-4, **kwargs):
    """Reserve final call through pinned solver; any bad trial remains fatal."""
    bad = False
    def checked(x):
        nonlocal bad
        try:
            f, g, payload = fun(x)
            finite = math.isfinite(float(f)) and bool(torch.isfinite(g).all())
            if payload:
                for key in ('nll', 'kl'):
                    finite &= all(math.isfinite(float(v)) for v in payload.get(key, []))
                finite &= all(bool(torch.isfinite(w).all()) for w in payload.get('weights', {}).values())
            if not finite:
                raise FloatingPointError('NONFINITE_ORACLE_PAYLOAD')
            return f, g, payload
        except FloatingPointError:
            bad = True
            raise
    result = pilot_solve(checked, x0, c, rho, mask, cap=cap, tol=tol,
                         max_seconds=None, **kwargs)
    result['pilot_convergence_eligible'] = result.pop('commit_eligible')
    result['any_nonfinite'] = bad or result['status'] == 'NONFINITE'
    if not bool(mask.any()) and not result['any_nonfinite']:
        result['status'] = 'POLICY_ZERO_STEP'
    return admit(result, rho, mask)
