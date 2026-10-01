"""One JLZ batch problem (reference, NumPy float64).

Variables  R_l (d, B): local-z target displacement of request r at edit layer l (native compute_z delta).
Realization Delta_l = R_l adj_l^T,  adj_l = (lam C0_l + H_l + K_l K_l^T)^{-1} K_l   (MEMIT-H solve factor).
Objective  sum_r [ NLL_r + kl_factor KL_r ] + sum_{l,r} wd ||R_l[:, r]|| / ||h_{l,r}||^2,
           ||R_l[:, r]|| <= clamp ||h_{l,r}||;  requests with NLL_r(0) < stop are fixed at zero.
NLL_r: mean over the request's context prompts of -log p(target); KL_r: KL(p_entry || p) on its KL prompt.
"""
import numpy as np

from .toy import log_softmax


class JLZProblem:
    def __init__(self, model, batch, layers, contexts, C0, H, lam, wd=0.5, clamp=0.75, kl_factor=0.0625,
                 stop=5e-2, free_layers=None):
        self.model, self.layers, self.B, self.C = model, list(layers), len(batch), len(contexts)
        self.wd, self.clamp, self.kl_factor, self.stop = wd, clamp, kl_factor, stop
        self.rw = np.stack([q["h0"] + c for q in batch for c in contexts])   # request-major
        self.klp = np.stack([q["h_kl"] for q in batch])
        self.t_new = np.repeat([q["t_new"] for q in batch], self.C)
        entry = model.forward(self.rw)
        kl_entry = model.forward(self.klp)
        B, C = self.B, self.C
        # P1 keys (context-averaged) and anchors (canonical prompt = context 0) at entry
        self.K = {l: entry["xs"][l].reshape(B, C, -1).mean(axis=1).T for l in self.layers}
        hn = {l: np.linalg.norm(entry["hs"][l + 1].reshape(B, C, -1)[:, 0, :], axis=1) for l in self.layers}
        self.rho = {l: clamp * hn[l] for l in self.layers}
        self.c = {l: wd / hn[l] ** 2 for l in self.layers}
        self.logp0 = log_softmax(kl_entry["logits"])
        # P2 realization factor (MEMIT-H expression, all layers at entry)
        self.A = {l: lam * C0[l] + H[l] for l in self.layers}
        self.adj = {l: np.linalg.solve(self.A[l] + self.K[l] @ self.K[l].T, self.K[l]) for l in self.layers}
        # P3 native zero-step screening on the entry loss (delta = 0: KL = decay = 0)
        nll0 = self._nll(entry["logits"])
        self.nll_entry = nll0
        self.active = nll0 >= stop
        free = set(self.layers if free_layers is None else free_layers)
        self.free = {(l, r): (l in free and bool(self.active[r])) for l in self.layers for r in range(B)}
        self.blocks = [k for k in sorted(self.free) if self.free[k]]
        self.d = model.d

    # ----- packing -----
    def zeros(self):
        return {l: np.zeros((self.d, self.B)) for l in self.layers}

    def pack(self, R):
        return np.concatenate([R[l][:, r] for (l, r) in self.blocks]) if self.blocks else np.zeros(0)

    def unpack(self, x):
        R = self.zeros()
        for i, (l, r) in enumerate(self.blocks):
            R[l][:, r] = x[i * self.d:(i + 1) * self.d]
        return R

    def block_params(self):
        return (np.array([self.c[l][r] for (l, r) in self.blocks]), np.array([self.rho[l][r] for (l, r) in self.blocks]))

    # ----- realization and loss -----
    def deltas(self, R):
        return {l: R[l] @ self.adj[l].T for l in self.layers}

    def _nll(self, logits):
        lp = log_softmax(logits)
        per_prompt = -lp[np.arange(len(self.t_new)), self.t_new]
        return per_prompt.reshape(self.B, self.C).mean(axis=1)

    def terms(self, deltas):
        """Per-request NLL and KL with the given weight deltas (None = model as is)."""
        rw = self.model.forward(self.rw, deltas)
        kl = self.model.forward(self.klp, deltas)
        lp = log_softmax(kl["logits"])
        KL = np.sum(np.exp(self.logp0) * (self.logp0 - lp), axis=1)
        return self._nll(rw["logits"]), KL, rw, kl

    def smooth(self, R):
        """Value and gradient of sum_active (NLL_r + kl_factor KL_r); gradient restricted to free blocks."""
        deltas = self.deltas(R)
        nll, KL, rw, kl = self.terms(deltas)
        a = self.active.astype(float)
        f = float(np.sum(a * (nll + self.kl_factor * KL)))
        p = np.exp(log_softmax(rw["logits"]))
        g_rw = p.copy()
        g_rw[np.arange(len(self.t_new)), self.t_new] -= 1.0
        g_rw *= np.repeat(a, self.C)[:, None] / self.C
        g_kl = self.kl_factor * a[:, None] * (np.exp(log_softmax(kl["logits"])) - np.exp(self.logp0))
        gd_rw = self.model.backward(rw, g_rw, self.layers)
        gd_kl = self.model.backward(kl, g_kl, self.layers)
        G = {l: (gd_rw[l] + gd_kl[l]) @ self.adj[l] for l in self.layers}
        for (l, r), free in self.free.items():
            if not free:
                G[l][:, r] = 0.0
        return f, G, dict(nll=nll, kl=KL)

    def decay(self, R):
        return float(sum(self.c[l][r] * np.linalg.norm(R[l][:, r]) for (l, r) in self.blocks))

    # ----- reports -----
    def kkt(self, R, G):
        """Per-block optimality evidence for the composite objective (smooth + norm decay + ball)."""
        out = []
        for (l, r) in self.blocks:
            v, g, c, rho = R[l][:, r], G[l][:, r], self.c[l][r], self.rho[l][r]
            n = float(np.linalg.norm(v))
            if n == 0.0:
                out.append(dict(layer=l, request=r, state="ZERO", norm_over_radius=0.0,
                                subgradient_ratio=float(np.linalg.norm(g)) / c))
                continue
            u = v / n
            total = g + c * u
            radial = float(total @ u)
            tangential = float(np.linalg.norm(total - radial * u))
            boundary = abs(n / rho - 1.0) <= 1e-9
            out.append(dict(layer=l, request=r, state="BOUNDARY" if boundary else "INTERIOR",
                            norm_over_radius=n / rho, tangential_residual=tangential,
                            radial_residual=0.0 if boundary else abs(radial),
                            clamp_multiplier=max(0.0, -radial) if boundary else 0.0))
        return out

    def allocation(self, R):
        """Per-layer target and realized displacement (at entry keys), plus own-key realization fractions."""
        rows = []
        for l in self.layers:
            D = R[l] @ self.adj[l].T @ self.K[l]
            Gm = self.K[l].T @ np.linalg.solve(self.A[l], self.K[l])
            frac = np.diag(Gm @ np.linalg.inv(np.eye(self.B) + Gm))
            rows.append(dict(layer=l, target_norm=float(np.linalg.norm(R[l])), realized_norm=float(np.linalg.norm(D)),
                             active_blocks=int(sum(np.linalg.norm(R[l][:, r]) > 0 for r in range(self.B))),
                             own_key_fraction=frac.tolist()))
        tot = sum(x["realized_norm"] ** 2 for x in rows) or 1.0
        for x in rows:
            x["realized_energy_share"] = x["realized_norm"] ** 2 / tot
        return rows


def prox_blocks(V, t, c, rho, d):
    """Exact prox of t*(c||v|| + indicator(||v|| <= rho)) per block: radial shrink, then clip."""
    out = np.zeros_like(V)
    for i in range(len(c)):
        v = V[i * d:(i + 1) * d]
        n = float(np.linalg.norm(v))
        if n > 0.0:
            out[i * d:(i + 1) * d] = v * (min(rho[i], max(0.0, n - t * c[i])) / n)
    return out
