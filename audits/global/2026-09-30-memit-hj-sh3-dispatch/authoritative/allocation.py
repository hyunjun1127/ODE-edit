"""MEMIT-HJ layer allocation -- reference implementation (NumPy, float64).

This module is the executable specification of the only writer change in MEMIT-HJ v1:
the per-layer residual target of the pinned MEMIT-H loop. Everything else (compute_z,
compute_ks, the FP64 solve, history append) stays the pinned code path.

Shapes follow the MEMIT code:
  R        (d_out, B)    residual at the top edit layer, R = Z - cur_zs (re-measured each step)
  K        (d_in, B)     context-averaged subject keys of one layer (compute_ks(...).T)
  A        (d_in, d_in)  preservation Gram A = mom2_update_weight * C0 + H  (SPD)
  adj      (d_in, B)     what MEMIT-H computes: adj = (A + K K^T)^{-1} K
  update   (d_out, d_in) Delta = resid @ adj.T

MEMIT-H  : resid_i = R_rem / (n - i)
MEMIT-HJ : resid_i = R_rem (I + S_i)^{-1} (I + G_i),  S_i = sum_{j >= i} G_j,  G_j = K_j^T A_j^{-1} K_j
The production port is a line-for-line translation to torch (float64 solves as in MEMIT-H).
"""
import numpy as np


def sym(M):
    return 0.5 * (M + M.T)


def ainv_k(A, K):
    """A^{-1} K by a direct solve (production: Cholesky of A reused within a batch)."""
    return np.linalg.solve(A, K)


def capacity(K, AinvK):
    """G = K^T A^{-1} K (B x B, symmetric positive definite for full-column-rank K)."""
    return sym(K.T @ AinvK)


def memit_adj(A, K):
    """The pinned MEMIT-H factor adj = (A + K K^T)^{-1} K."""
    return np.linalg.solve(A + K @ K.T, K)


def capacity_from_adj(K, adj):
    """Recover G from MEMIT-H's own adj without a second solve.

    F = K^T adj = G (I + G)^{-1}  =>  G = F (I - F)^{-1}.
    """
    F = sym(K.T @ adj)
    I = np.eye(F.shape[0])
    return sym(F @ np.linalg.inv(I - F))


def resid_divisor(R_rem, i, n):
    """MEMIT-H target for the i-th of n layers (0-based, ascending)."""
    return R_rem / (n - i)


def resid_joint(R_rem, G_i, G_upper_sum):
    """MEMIT-HJ target: R_rem (I + S_i)^{-1} (I + G_i) with S_i = G_i + sum of upper-layer G."""
    I = np.eye(G_i.shape[0])
    S = G_i + G_upper_sum
    return R_rem @ np.linalg.solve(I + S, I + G_i)


def update(resid, adj):
    """Delta = resid adj^T, exactly the MEMIT-H expression."""
    return resid @ adj.T


def joint_ols(R, Ks, As):
    """Closed form of  min_{Delta_l} ||sum_l Delta_l K_l - R||^2 + sum_l tr(Delta_l A_l Delta_l^T).

    Returns updates Delta_l = R (I+G)^{-1} (A_l^{-1} K_l)^T, per-layer realized displacements
    D_l = Delta_l K_l = R (I+G)^{-1} G_l, capacities G_l and the unrealized part R (I+G)^{-1}.
    """
    AK = [ainv_k(A, K) for A, K in zip(As, Ks)]
    Gs = [capacity(K, a) for K, a in zip(Ks, AK)]
    I = np.eye(R.shape[1])
    M = np.linalg.inv(I + sum(Gs))
    return dict(deltas=[R @ M @ a.T for a in AK], displacements=[R @ M @ g for g in Gs],
                capacities=Gs, unrealized=R @ M)


def joint_objective(R, Ks, As, deltas, propagation=None):
    """Joint objective, optionally with linear propagation operators J_l acting on displacements."""
    J = propagation or [None] * len(Ks)
    fit = -R.copy()
    for D, K, P in zip(deltas, Ks, J):
        fit = fit + (D @ K if P is None else P @ (D @ K))
    return float(np.sum(fit ** 2) + sum(np.trace(D @ A @ D.T) for D, A in zip(deltas, As)))


def sequential(R0, Ks, As, mode="joint", propagation=None):
    """Sequential writer under a linear surrogate of the network.

    After writing layer i the residual measured at the top layer becomes R_rem - J_i (Delta_i K_i)
    (J_i = identity is MEMIT's additivity assumption). Keys are held fixed here; the production
    writer recomputes keys at the current state before each step.
    mode: "divisor" (MEMIT-H) or "joint" (MEMIT-HJ). Returns updates, trajectory of ||R_rem||.
    """
    n = len(Ks)
    J = propagation or [None] * n
    R = R0.copy()
    deltas, traj = [], [float(np.linalg.norm(R))]
    Gs = [capacity(K, ainv_k(A, K)) for K, A in zip(Ks, As)]
    for i in range(n):
        adj = memit_adj(As[i], Ks[i])
        if mode == "divisor":
            r = resid_divisor(R, i, n)
        elif mode == "joint":
            upper = sum(Gs[i + 1:]) if i + 1 < n else np.zeros_like(Gs[i])
            r = resid_joint(R, Gs[i], upper)
        else:
            raise ValueError(mode)
        D = update(r, adj)
        deltas.append(D)
        realized = D @ Ks[i]
        R = R - (realized if J[i] is None else J[i] @ realized)
        traj.append(float(np.linalg.norm(R)))
    return dict(deltas=deltas, residual_norms=traj, final_residual=R)


def min_cost_realization(d, k, A):
    """Cheapest single-key update realizing displacement d at key k under cost tr(Delta A Delta^T).

    Delta* = d (A^{-1}k)^T / kappa,  kappa = k^T A^{-1} k,  cost = ||d||^2 / kappa.
    """
    a = np.linalg.solve(A, k)
    kappa = float(k @ a)
    return np.outer(d, a) / kappa, kappa


def layer_shares(Gs):
    """Capacity shares tr(G_l)/tr(sum G) and, for B=1, the realized shares kappa_l/(1+sum kappa)."""
    tr = np.array([np.trace(G) for G in Gs])
    out = dict(capacity_share=tr / tr.sum())
    if Gs[0].shape == (1, 1):
        k = np.array([G[0, 0] for G in Gs])
        out["realized_fraction"] = k / (1.0 + k.sum())
    return out
