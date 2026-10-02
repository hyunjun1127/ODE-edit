"""V9 one combined ridge-energy norm, all off-diagonal/causal gradients."""
import torch
from .common import require

class Root(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q):
        out=q.clamp_min(0).sqrt();ctx.save_for_backward(out);return out
    @staticmethod
    def backward(ctx,g):
        out,=ctx.saved_tensors;s=torch.zeros_like(out);nz=out>0;s[nz]=.5/out[nz]
        return g*s

def quadratic(D,Q):
    terms=(D.double().T@D.double())*Q.T;q=terms.sum()
    limit=1e-12*max(1.,float(terms.detach().abs().sum()))
    require(float(q.detach())>=-limit,'NEGATIVE_GEOMETRY_OUTSIDE_ROUNDOFF')
    return q.clamp_min(0),dict(original=float(q.detach()),limit=limit)

def loss(D,geometry,anchors,arm):
    require(arm in ('A','B'),'ARM');costs=[];details={}
    for l,d in D.items():
        scale=d.shape[1]*anchors[l].double().square().mean()
        g,roundoff=quadratic(d,geometry[l]['G'])
        residual=geometry[l]['M']-torch.eye(d.shape[1],device=d.device,dtype=torch.float64) if 'M' in geometry[l] else geometry[l]['P'].T@geometry[l]['K']-torch.eye(d.shape[1],device=d.device,dtype=torch.float64)
        e=(d.double()@residual).square().sum();c=(g+e)/scale;costs.append(c)
        details[str(l)]=dict(g_energy=float(g.detach()),e_energy=float(e.detach()),g=float(Root.apply(g/scale).detach()),
            e=float(Root.apply(e/scale).detach()),c=float(Root.apply(c).detach()),roundoff=roundoff)
    value=Root.apply(torch.stack(costs)).sum() if arm=='A' else Root.apply(torch.stack(costs).sum())
    return .1*value,details
