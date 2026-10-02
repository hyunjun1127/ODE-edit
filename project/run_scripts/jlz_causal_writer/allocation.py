"""Current graph trace costs. Zero selected subgradient, no smoothed norm."""
import torch
from .common import require

class Root(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q):
        out = q.clamp_min(0).sqrt()
        ctx.save_for_backward(out)
        return out
    @staticmethod
    def backward(ctx, incoming):
        out, = ctx.saved_tensors
        scale = torch.zeros_like(out)
        nz = out > 0
        scale[nz] = .5 / out[nz]
        return incoming * scale

def quadratic(D, Q):
    parts = (D.double().T @ D.double()) * Q.T
    q = parts.sum()
    limit = 1e-12 * max(1., float(parts.detach().abs().sum()))
    require(float(q.detach()) >= -limit, 'NEGATIVE_GEOMETRY_OUTSIDE_ROUNDOFF')
    return q.clamp_min(0), dict(negative=float(min(0., float(q.detach()))), limit=limit)

def loss(D, geometry, anchors, arm):
    require(arm in ('A', 'B'), 'ARM')
    qs = [[], []]; telemetry = {}
    for l, d in D.items():
        scale = d.shape[1] * anchors[l].double().square().mean()
        g, rg = quadratic(d, geometry[l]['G'])
        e, re = quadratic(d, geometry[l]['E'])
        qs[0].append(g / scale); qs[1].append(e / scale)
        telemetry[str(l)] = dict(g=float(Root.apply(g/scale).detach()),
                                 e=float(Root.apply(e/scale).detach()), roundoff_G=rg, roundoff_E=re)
    terms = [(Root.apply(torch.stack(q)).sum() if arm == 'A'
              else Root.apply(torch.stack(q).sum())) for q in qs]
    return .1 * sum(terms), telemetry
