"""Irreversible request activation with dimensionless priced-budget expansion."""
import torch


class RequestController:
    def __init__(self,anchors,anchor_star,layers,prices,n_exp=4,grace=12,threshold=.05,c=.75):
        self.layers=tuple(layers);self.anchors=torch.as_tensor(anchors).double()
        self.anchor_star=torch.as_tensor(anchor_star,device=self.anchors.device).double()
        self.prices=torch.as_tensor(prices,device=self.anchors.device,dtype=torch.float64)
        B=self.anchor_star.numel()
        if self.anchors.shape!=(len(self.layers),B) or self.prices.shape!=self.anchors.shape:raise RuntimeError('CONTROLLER_SCHEMA')
        if not bool(torch.isfinite(self.anchors).all() and torch.isfinite(self.anchor_star).all() and torch.isfinite(self.prices).all()
                    and (self.anchors>0).all() and (self.anchor_star>0).all() and (self.prices>=1).all()):raise RuntimeError('CONTROLLER_DOMAIN')
        self.weights=self.prices/self.anchors;self.caps=c*self.anchors
        self.base=torch.full((B,),c,device=self.anchors.device,dtype=torch.float64)
        self.maximum=c*self.prices.max(0).values;self.beta=self.base.clone()
        self.n_exp,self.grace,self.threshold=int(n_exp),int(grace),float(threshold)
        if self.n_exp<0 or self.grace<0:raise RuntimeError('CONTROLLER_PROFILE')
        self.expansion=torch.zeros(B,dtype=torch.int64,device=self.anchors.device);self.t=torch.zeros_like(self.expansion)
        self.active=torch.zeros(B,dtype=torch.bool,device=self.anchors.device);self.last_candidate=-1

    def _loss(self,F):
        F=torch.as_tensor(F,device=self.anchors.device)
        if F.shape!=self.active.shape or not bool(torch.isfinite(F).all()):raise RuntimeError('CONTROLLER_LOSS')
        return F

    def observe(self,F,candidate):
        F=self._loss(F)
        if candidate!=self.last_candidate+1 or not 0<=candidate<=24:raise RuntimeError('CONTROLLER_SEQUENCE')
        self.active|=F>=self.threshold;self.last_candidate=candidate
        return self.active.clone(),candidate==24 or not bool(self.active.any())

    def before_update(self,F):
        if not 0<=self.last_candidate<24 or not bool(self.active.any()):raise RuntimeError('TERMINAL_EXPANSION_FORBIDDEN')
        F=self._loss(F)
        if self.n_exp:
            expand=self.active&(self.t>=self.grace)&(F>=self.threshold)&(self.expansion<self.n_exp)&(self.base<self.maximum)
            self.expansion[expand]+=1
            self.beta=self.base*torch.exp((self.expansion.double()/self.n_exp)*torch.log(self.maximum/self.base))
            self.beta=torch.where(self.expansion==self.n_exp,self.maximum,self.beta)
        else:self.beta=self.base.clone()
        return self.beta.clone()

    def record_update(self,active):
        mask=torch.as_tensor(active,device=self.active.device,dtype=torch.bool)
        if not torch.equal(mask,self.active) or self.last_candidate==24:raise RuntimeError('CONTROLLER_UPDATE_MASK')
        self.t[mask]+=1
        if bool((self.t>24).any()):raise RuntimeError('CONTROLLER_UPDATE_BUDGET')

    def terminal_states(self,F):
        values=self._loss(F).cpu().tolist();states=[]
        for r,f in enumerate(values):
            if not bool(self.active[r]):states.append('ZERO_STEP')
            elif f<self.threshold:states.append('SATISFIED_EXPANDED' if int(self.expansion[r]) else 'SATISFIED_BASE')
            else:states.append('UNSATISFIED_MAX' if bool(self.beta[r]>=self.maximum[r]) else 'UNSATISFIED')
        return states

    def receipt(self):
        # All anchors, caps, prices and endpoints live only in entry-price.json.
        return dict(update_counts=self.t.cpu().tolist(),expansion=self.expansion.cpu().tolist(),
                    active=self.active.cpu().tolist(),beta=self.beta.cpu().tolist())
