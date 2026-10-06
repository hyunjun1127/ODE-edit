"""Unchanged absolute-R EfficiencyAdam, with weighted group-L1 projection."""
import time
import torch
from project.run_scripts.jlz_v12r.optimizer import analytic_norm
from .projection import project_capped_weighted_l1


class EfficiencyAdamAbs:
    def __init__(self,template,lr=.1,eps=1e-8,betas=(.9,.999)):
        self.layers=tuple(template);first=template[self.layers[0]];self.B=first.shape[1]
        self.lr,self.eps,self.betas=float(lr),float(eps),tuple(betas)
        self.m={l:torch.zeros_like(template[l],dtype=torch.float32) for l in self.layers}
        self.v={l:torch.zeros_like(template[l],dtype=torch.float32) for l in self.layers}
        self.s=torch.zeros((len(self.layers),self.B),dtype=torch.float32,device=first.device)
        self.t=torch.zeros(self.B,dtype=torch.int64,device=first.device)

    @torch.no_grad()
    def step(self,R,gradients,active_mask,caps,weights,beta):
        started=time.monotonic();active=torch.as_tensor(active_mask,device=self.t.device,dtype=torch.bool)
        if active.shape!=(self.B,) or tuple(R)!=self.layers or tuple(gradients)!=self.layers:raise RuntimeError('ADAM_SHAPE')
        if not bool(active.any()) or bool((self.t[active]>=24).any()):raise RuntimeError('ADAM_UPDATE_BUDGET')
        b1,b2=self.betas;self.t[active]+=1;proposed={l:R[l].clone() for l in self.layers}
        for i,l in enumerate(self.layers):
            g=gradients[l].float()
            if g.shape!=R[l].shape or not bool(torch.isfinite(g).all()):raise RuntimeError('ADAM_GRADIENT')
            self.m[l][:,active]=b1*self.m[l][:,active]+(1-b1)*g[:,active]
            self.v[l][:,active]=b2*self.v[l][:,active]+(1-b2)*g[:,active].square()
            self.s[i,active]=b2*self.s[i,active]+(1-b2)*g[:,active].square().mean(0)
        counter=self.t[active].double();bc1=(1-b1**counter).float();bc2=(1-b2**counter).float()
        shat=self.s[:,active]/bc2[None,:];mean_s=shat.mean(0)
        denom=torch.where(mean_s>0,mean_s,torch.ones_like(mean_s))
        gamma_active=torch.sqrt(shat/denom[None,:]);gamma_active[:,mean_s==0]=1
        gamma=torch.ones_like(self.s);gamma[:,active]=gamma_active
        for i,l in enumerate(self.layers):
            mhat=self.m[l][:,active]/bc1[None,:];vhat=self.v[l][:,active]/bc2[None,:]
            proposed[l][:,active]-=self.lr*gamma_active[i][None,:]*mhat/(vhat.sqrt()+self.eps)
        adam_seconds=time.monotonic()-started
        projected,receipt=project_capped_weighted_l1(proposed,caps,weights,beta)
        for l in self.layers:
            if not torch.equal(projected[l][:,~active],R[l][:,~active]):raise RuntimeError('INACTIVE_REQUEST_CHANGED')
        receipt.update(gamma=gamma.cpu().tolist(),adam_updates=self.t.cpu().tolist(),lr=self.lr,eps=self.eps,
                       moment_reset=False,adam_seconds=adam_seconds,projection_seconds=receipt['seconds'])
        return projected,receipt
