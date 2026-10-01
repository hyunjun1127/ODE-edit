"""One JLZ batch write on the toy model (reference pipeline P1-P9 of the method document)."""
import numpy as np

from .problem import JLZProblem, prox_blocks
from .prox import native_adam, solve


def write_batch(model, state, batch, cfg, solver="prox", free_layers=None):
    """state: dict(C0={l: (m,m)}, H={l: (m,m)}). Mutates model.W and state['H']. Returns a receipt."""
    layers = cfg["layers"]
    prob = JLZProblem(model, batch, layers, cfg["contexts"], state["C0"], state["H"], cfg["lam"],
                      wd=cfg.get("wd", 0.5), clamp=cfg.get("clamp", 0.75), kl_factor=cfg.get("kl_factor", 0.0625),
                      stop=cfg.get("stop", 5e-2), free_layers=free_layers)
    receipt = dict(zero_step=[int(r) for r in np.where(~prob.active)[0]], blocks=len(prob.blocks))
    c, rho = prob.block_params()
    d = prob.d

    def fun(x):
        f, G, _ = prob.smooth(prob.unpack(x))
        return f, prob.pack(G)

    if not prob.blocks:
        R, G = prob.zeros(), prob.zeros()
        receipt["solver"] = dict(status="POLICY_ZERO_STEP", calls=0)
    elif solver == "prox":
        x, g, info = solve(fun, lambda x: prob.decay(prob.unpack(x)), lambda v, t: prox_blocks(v, t, c, rho, d),
                           np.zeros(len(prob.blocks) * d), tol=cfg.get("tol", 1e-8), cap=cfg.get("cap", 2000))
        R, G = prob.unpack(x), prob.unpack(g)
        receipt["solver"] = info
    elif solver == "native_adam":
        x, info = native_adam(fun, c, rho, d, np.zeros(len(prob.blocks) * d),
                              stop=cfg.get("stop", 5e-2) * int(prob.active.sum()))
        R = prob.unpack(x)
        G = prob.unpack(fun(x)[1])
        receipt["solver"] = info
    else:
        raise ValueError(solver)
    # P6 the update the oracle used is the update that is committed
    deltas = prob.deltas(R)
    nll_fit, kl_fit, _, _ = prob.terms(deltas)
    for l in layers:
        model.W[l] = model.W[l] + deltas[l]
    nll_post, kl_post, rw_post, _ = prob.terms(None)
    receipt["consistency"] = float(max(np.max(np.abs(nll_post - nll_fit)), np.max(np.abs(kl_post - kl_fit))))
    # P7 history append with post-write keys; lowest edited layer keys must be unchanged
    K_post = {l: rw_post["xs"][l].reshape(prob.B, prob.C, -1).mean(axis=1).T for l in layers}
    low = min(layers)
    receipt["lowest_layer_key_change"] = float(np.max(np.abs(K_post[low] - prob.K[low])))
    for l in layers:
        state["H"][l] = state["H"][l] + K_post[l] @ K_post[l].T
    receipt.update(kkt=prob.kkt(R, G), allocation=prob.allocation(R), nll_entry=prob.nll_entry.tolist(),
                   nll_fit=nll_fit.tolist(), kl_fit=kl_fit.tolist(), decay=prob.decay(R))
    receipt["_R"], receipt["_deltas"], receipt["_problem"] = R, deltas, prob
    return receipt
