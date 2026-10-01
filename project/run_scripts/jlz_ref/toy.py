"""Toy stand-in for the transformer: one position per prompt, residual MLP blocks.

h_{l+1} = h_l + W_l tanh(U_l h_l + b_l),  logits = E h_L.
x_l = tanh(U_l h_l + b_l) is the key of layer l (down_proj input) and W_l plays down_proj.
Weight edits are passed as deltas {layer: (d, m)} and enter exactly as W_l + Delta_l, which is
the same expression the commit uses, so a fitted forward and the committed model coincide.
"""
import numpy as np


def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class ToyLM:
    def __init__(self, d=48, m=24, V=40, L=9, seed=0, w_scale=0.6, e_scale=3.0):
        rng = np.random.default_rng(seed)
        self.d, self.m, self.V, self.L = d, m, V, L
        self.U = [rng.standard_normal((m, d)) / np.sqrt(d) for _ in range(L)]
        self.b = [0.1 * rng.standard_normal(m) for _ in range(L)]
        self.W = [rng.standard_normal((d, m)) * (w_scale / np.sqrt(m)) for _ in range(L)]
        self.E = rng.standard_normal((V, d)) * (e_scale / np.sqrt(d))

    def weight(self, l, deltas):
        return self.W[l] + deltas[l] if deltas is not None and l in deltas else self.W[l]

    def forward(self, H0, deltas=None):
        H, xs, hs = H0, [], [H0]
        for l in range(self.L):
            X = np.tanh(H @ self.U[l].T + self.b[l])
            H = H + X @ self.weight(l, deltas).T
            xs.append(X)
            hs.append(H)
        return dict(xs=xs, hs=hs, logits=H @ self.E.T, deltas=deltas)

    def backward(self, cache, g_logits, layers):
        """dL/dDelta_l for l in layers, given dL/dlogits."""
        gH = g_logits @ self.E
        out = {}
        for l in reversed(range(self.L)):
            X = cache["xs"][l]
            if l in layers:
                out[l] = gH.T @ X
            gA = (gH @ self.weight(l, cache["deltas"])) * (1.0 - X * X)
            gH = gH + gA @ self.U[l]
        return out
