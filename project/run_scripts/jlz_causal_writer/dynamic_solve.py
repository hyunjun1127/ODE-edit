"""FP64 whole-context differentiable dual. Only the batch prior is frozen."""
import time
import torch
from project.run_scripts.jlz_writer_coupled.geometry import build_prior_from_npz
from .common import require

@torch.no_grad()
def prior(path, history, device, coefficient=15000.):
    start = time.monotonic()
    A = build_prior_from_npz(path, history, lambda_c=coefficient, device=device)
    L, info = torch.linalg.cholesky_ex(A)
    require(int(info) == 0 and bool(torch.isfinite(L).all()), 'PRIOR_NOT_SPD_NO_JITTER')
    return L, dict(seconds=time.monotonic()-start, d_in=L.shape[0],
                   normalization='FP32_mom2/count_then_FP64', factorization_count=1,
                   factor_diagonal_min=float(L.diagonal().min()), jitter=0)

def solve(K, L, alpha, owners, B, residual_tolerance=1e-8):
    """Every upper call retains K -> F/S/V/T/P, including T geometry adjoints."""
    start = time.monotonic()
    K = K.double()
    alpha = alpha.to(device=K.device, dtype=torch.float64)
    owners = owners.to(device=K.device, dtype=torch.long)
    require(K.ndim == 2 and K.shape[1] == len(alpha) == len(owners), 'KEY_COLUMN_SCHEMA')
    require(bool(torch.isfinite(K).all()) and bool(torch.isfinite(alpha).all()) and
            bool((alpha >= 0).all()) and bool(((owners >= 0) & (owners < B)).all()), 'KEY_IDENTITY_FINITE')
    Z = torch.nn.functional.one_hot(owners, B).T.double()
    root = alpha.sqrt()
    F = torch.linalg.solve_triangular(L, K * root, upper=False)
    S = torch.eye(K.shape[1], device=K.device, dtype=torch.float64) + F.T @ F
    chol, info = torch.linalg.cholesky_ex(S)
    require(int(info) == 0, 'CONTEXT_NOT_SPD_NO_JITTER')
    V = torch.cholesky_solve(root[:, None] * Z.T, chol)
    T = F @ V
    P = torch.linalg.solve_triangular(L.T, T, upper=True)
    G = T.T @ T
    R = P.T @ K - Z
    E = (R * alpha) @ R.T
    with torch.no_grad():
        kbar = (K * alpha) @ Z.T
        residual = L @ (L.T @ P) + (K * alpha) @ (K.T @ P) - kbar
        measured = float(residual.norm() / kbar.norm().clamp_min(1e-12))
        identity = (Z * alpha) @ Z.T - kbar.T @ P
        meta = dict(seconds=time.monotonic()-start, relative_residual=measured,
                    residual_tolerance=residual_tolerance, actual_B=B,
                    context_columns=K.shape[1], geometry_identity_max=float((G+E-identity).abs().max()),
                    stored_context_mass_min=float(((Z*alpha)@Z.T).diagonal().min()),
                    grad_K_enabled=K.requires_grad, grad_P_enabled=P.requires_grad)
    require(bool(torch.isfinite(P).all()) and measured <= residual_tolerance,
            'SPD_RESIDUAL:'+str(meta))
    return dict(P=P, T=T, G=G, E=E, K=K, metadata=meta)
