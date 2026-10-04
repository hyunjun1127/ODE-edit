"""Request-local FP32 EfficiencyAdam and exact FP64 shared projection."""
import torch
from .common import require

def norm_gradient(u, price):
    n = u.norm()
    return price*u/n if bool(n > 0) else torch.zeros_like(u)

@torch.no_grad()
def project(blocks, radius=.75):
    require(radius >= 0 and blocks, 'PROJECTION_INPUT')
    double = [u.double() for u in blocks]
    n = torch.stack([u.norm() for u in double])
    require(bool(torch.isfinite(n).all()), 'PROJECTION_NONFINITE')
    tau = n.new_zeros(())
    if bool(n.sum() > radius):
        sorted_n = n.sort(descending=True).values
        thresholds = (sorted_n.cumsum(0)-radius)/torch.arange(1,len(n)+1,device=n.device,dtype=n.dtype)
        selected = sorted_n > thresholds
        tau = thresholds[torch.nonzero(selected)[-1,0]] if radius else sorted_n[0]
    scales = torch.where(n > 0, (n-tau).clamp_min(0)/torch.where(n>0,n,torch.ones_like(n)), torch.zeros_like(n))
    projected64 = [u*s for u,s in zip(double,scales)]
    stored = [v.to(u.dtype) for v,u in zip(projected64,blocks)]
    budget = sum(float(v.double().norm()) for v in stored)
    require(budget <= radius+1e-6*max(1,radius), 'PROJECTION_CASTBACK')
    return stored, dict(tau=float(tau),proposal_norm=n.tolist(),
        projected64_norm=[float(v.norm()) for v in projected64],stored_norm=[float(v.double().norm()) for v in stored],
        stored_budget=budget,violation=max(0.,budget-radius))

class EfficiencyAdam:
    def __init__(self, blocks, anchor_star, lr=.1, eps=1e-8, betas=(.9,.999)):
        require(bool(torch.isfinite(anchor_star)) and float(anchor_star)>0, 'ANCHOR_STAR')
        self.anchor = float(anchor_star);self.lr=lr/self.anchor;self.eps=eps*self.anchor
        self.b1,self.b2=betas;self.t=0
        self.m=[torch.zeros_like(u) for u in blocks];self.v=[torch.zeros_like(u) for u in blocks]
        self.s=blocks[0].new_zeros(len(blocks))

    @torch.no_grad()
    def step(self, blocks, gradients, radius=.75):
        require(len(blocks)==len(gradients)==len(self.m), 'OPT_SHAPES')
        require(all(bool(torch.isfinite(g).all()) for g in gradients), 'NONFINITE_GRADIENT')
        self.t+=1;b1t=1-self.b1**self.t;b2t=1-self.b2**self.t
        for i,g in enumerate(gradients):
            self.m[i].mul_(self.b1).add_(g,alpha=1-self.b1)
            self.v[i].mul_(self.b2).addcmul_(g,g,value=1-self.b2)
            self.s[i]=self.b2*self.s[i]+(1-self.b2)*g.square().mean()
        shat=self.s/b2t;average=shat.mean()
        gamma=(shat/average).sqrt() if bool(average>0) else torch.ones_like(shat)
        proposal=[u-self.lr*gm*(m/b1t)/((v/b2t).sqrt()+self.eps)
                  for u,gm,m,v in zip(blocks,gamma,self.m,self.v)]
        require(all(bool(torch.isfinite(u).all()) for u in proposal), 'OPT_NONFINITE')
        stored,receipt=project(proposal,radius)
        receipt.update(gamma=gamma.tolist(),s_hat=shat.tolist(),mean_s_hat=float(average),
            step_norm=[float((new-old).norm()) for new,old in zip(stored,blocks)],
            update=self.t,lr=self.lr,eps=self.eps)
        return stored,receipt
