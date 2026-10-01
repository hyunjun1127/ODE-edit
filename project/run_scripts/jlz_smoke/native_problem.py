"""Separate native-KL adapter; the attached jlz_ref sources remain unchanged.

Native F.kl_div(entry_logp, current_logp, log_target=True) computes
KL(current || entry). Its logits derivative is p * (log(p/q) - KL(p||q)).
All geometry, screening, constraints and decay are inherited unchanged.
"""
import numpy as np

from project.run_scripts.jlz_ref.problem import JLZProblem
from project.run_scripts.jlz_ref.toy import log_softmax


class NativeKLProblem(JLZProblem):
    def terms(self, deltas):
        nll, _, rw, kl = super().terms(deltas)
        lp = log_softmax(kl["logits"])
        KL = np.sum(np.exp(lp) * (lp - self.logp0), axis=1)
        return nll, KL, rw, kl

    def smooth(self, R):
        deltas = self.deltas(R)
        nll, KL, rw, kl = self.terms(deltas)
        a = self.active.astype(float)
        f = float(np.sum(a * (nll + self.kl_factor * KL)))
        g_rw = np.exp(log_softmax(rw["logits"]))
        g_rw[np.arange(len(self.t_new)), self.t_new] -= 1.0
        g_rw *= np.repeat(a, self.C)[:, None] / self.C
        lp = log_softmax(kl["logits"])
        g_kl = self.kl_factor * a[:, None] * np.exp(lp) * (lp - self.logp0 - KL[:, None])
        gd_rw = self.model.backward(rw, g_rw, self.layers)
        gd_kl = self.model.backward(kl, g_kl, self.layers)
        G = {l: (gd_rw[l] + gd_kl[l]) @ self.adj[l] for l in self.layers}
        for (l, r), free in self.free.items():
            if not free:
                G[l][:, r] = 0.0
        return f, G, dict(nll=nll, kl=KL)
