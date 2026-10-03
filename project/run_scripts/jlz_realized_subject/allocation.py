"""G-only A=sum(layer roots), B=root(sum layer squares), zero derivative at 0."""
import torch
from .common import require


class Root(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        root = value.clamp_min(0).sqrt(); ctx.save_for_backward(root); return root

    @staticmethod
    def backward(ctx, grad):
        root, = ctx.saved_tensors
        derivative = torch.zeros_like(root); nz = root > 0
        derivative[nz] = .5 / root[nz]
        return grad * derivative


def loss(R, geometry, anchors, arm):
    require(arm in ('A','B'), 'ARM')
    costs = []; details = {}
    for l, r in R.items():
        terms = (r.double().T @ r.double()) * geometry[l]['G'].T
        Q = terms.sum(); limit = 1e-12 * max(1.,float(terms.detach().abs().sum()))
        require(bool(torch.isfinite(Q)) and float(Q.detach()) >= -limit, 'INVALID_G_ENERGY')
        scale = r.shape[1] * anchors[l].double().square().mean()
        cost = Q.clamp_min(0) / scale
        costs.append(cost)
        details[str(l)] = dict(Q=float(Q.detach()), c=float(Root.apply(cost).detach()),
                               sigma2=float(anchors[l].double().square().mean()), roundoff_limit=limit)
    total = Root.apply(torch.stack(costs)).sum() if arm=='A' else Root.apply(torch.stack(costs).sum())
    return total, details
