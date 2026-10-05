"""Request-local FP32 EfficiencyAdam and exact FP64 shared projection."""
import torch
from .common import require

def norm_gradient(u, price):
    n = u.norm()
    return price*u/n if bool(n > 0) else torch.zeros_like(u)

@torch.no_grad()
def project(blocks, radius=1.5, local_radius=.75):
    """Euclidean group projection with common tau and separate local caps."""
    require(radius >= 0 and local_radius >= 0 and blocks, 'PROJECTION_INPUT')
    double=[u.double() for u in blocks];n=torch.stack([u.norm() for u in double])
    require(bool(torch.isfinite(n).all()), 'PROJECTION_NONFINITE')
    vals=n.tolist();tau_value=0.
    if sum(min(local_radius,x) for x in vals)>radius:
        lo=0.;hi=max(vals)
        for _ in range(64):
            mid=(lo+hi)/2
            if sum(min(local_radius,max(x-mid,0.)) for x in vals)>radius:lo=mid
            else:hi=mid
        tau_value=hi
    tau=n.new_tensor(tau_value)
    sizes=(n-tau).clamp(0,local_radius)
    scales=torch.where(n>0,sizes/n.clamp_min(torch.finfo(n.dtype).tiny),torch.zeros_like(n))
    projected64=[u*s for u,s in zip(double,scales)]
    stored=[v.to(u.dtype) for v,u in zip(projected64,blocks)]
    norms=[float(v.double().norm()) for v in stored];budget=sum(norms)
    require(budget<=radius+1e-6 and max(norms)<=local_radius+1e-6,'PROJECTION_CASTBACK')
    return stored,dict(tau=float(tau),proposal_norm=n.tolist(),projected64_norm=sizes.tolist(),
        stored_norm=norms,stored_budget=budget,shared_radius=radius,local_radius=local_radius,
        violation=max(0.,budget-radius),local_violation=max(0.,max(norms)-local_radius),
        algorithm='capped_group_Euclidean_common_tau_FP64_FP32storage')

class EfficiencyAdam:
    def __init__(self, blocks, anchor_star, lr=.1, eps=1e-8, betas=(.9,.999)):
        require(bool(torch.isfinite(anchor_star)) and float(anchor_star)>0, 'ANCHOR_STAR')
        self.anchor = float(anchor_star);self.lr=lr/self.anchor;self.eps=eps*self.anchor
        self.b1,self.b2=betas;self.t=0
        self.m=[torch.zeros_like(u) for u in blocks];self.v=[torch.zeros_like(u) for u in blocks]
        self.s=blocks[0].new_zeros(len(blocks))

    @torch.no_grad()
    def step(self, blocks, gradients, radius=1.5, local_radius=.75):
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
        stored,receipt=project(proposal,radius,local_radius)
        receipt.update(gamma=gamma.tolist(),s_hat=shat.tolist(),mean_s_hat=float(average),
            step_norm=[float((new-old).norm()) for new,old in zip(stored,blocks)],
            update=self.t,lr=self.lr,eps=self.eps)
        return stored,receipt
