"""Raw native ridge geometry and compact Q; no symmetrization or norm term."""
import torch
from project.run_scripts.jlz_realized_subject.geometry import (
    mean_keys, ridge, PriorProduct, prior, inverse_apply,
)
from project.run_scripts.jlz_native_writer_aware.routes import cached_vjp


def compact_cost(R, P, A):
    """trace(R P.T A P R.T), including raw A's nonsymmetric VJP."""
    rd = R.double()
    G = P.T @ PriorProduct.apply(A, P)
    return ((rd.T @ rd) * G.T).sum()


def cost_adjoint(R, P, A):
    r = R.detach().requires_grad_(True)
    p = P.detach().requires_grad_(True)
    q = compact_cost(r, p, A)
    dr, dp = torch.autograd.grad(q, (r, p))
    return dr.detach(), dp.detach(), float(q.detach())


def effective_cost(weight, entry, A, chunk=256):
    """RAM-only effective FP32 subtraction, CPU blocks avoid dense GPU delta."""
    result = 0.0
    wc, we = weight.detach().cpu(), entry.detach().cpu()
    for start in range(0, wc.shape[0], chunk):
        delta = wc[start:start+chunk].double() - we[start:start+chunk].double()
        result += float(((delta @ A) * delta).sum())
    return result


def solve_vjp(K, P, gp, factor, cached=True):
    if cached:
        return cached_vjp(K, P, gp, factor)
    k = K.detach().requires_grad_(True)
    # Independent native full-system transpose VJP, not a detached upper key.
    p = torch.linalg.solve(factor['A'].to(k.device) + k @ k.T, k)
    gk = torch.autograd.grad(p, k, gp)[0]
    return gk, dict(backend='native_full_system_autograd', transpose_solve=1)
