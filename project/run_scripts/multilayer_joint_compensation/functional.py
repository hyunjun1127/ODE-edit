"""Full-sequence output-space gradients and matrix-free generalized Gauss–Newton.

The caller owns pure full-model logits callbacks and exact packing/teachers.
Only prediction logits are returned by the callback; all sequence positions
must experience every selected weight. Functional layer cross blocks are not
separated. No writer/key proxy, truncated rank, dense Hessian or mutable model.
"""
from dataclasses import dataclass, field
from typing import Callable
import math
import torch
from .linear_solve import WeightTree, zeros, add, finite


@dataclass
class OutputBatch:
    logits_fn: Callable[[WeightTree],torch.Tensor]
    target_ids: torch.Tensor                 # [T]
    context_index: torch.Tensor              # [T], local contiguous 0..C-1
    token_mean_weights: torch.Tensor         # [T], 1/(tokens in this context)
    context_weights: torch.Tensor            # [C], GLOBAL request/context weight
    teacher_logp: torch.Tensor               # [T,V], We or fixed Current path
    reference_nll: torch.Tensor              # [C], same fixed teacher
    identity: str = ''
    input_tokens: int = 0

    def validate(self):
        t=self.target_ids.numel();c=self.context_weights.numel()
        if t==0 or c==0:raise ValueError('EMPTY_MICROBATCH_USE_EMPTY_PANEL')
        if any(x.shape!=(t,) for x in [self.context_index,self.token_mean_weights]):raise ValueError('TOKEN_METADATA_SHAPE')
        if self.reference_nll.shape!=(c,) or self.teacher_logp.shape[0]!=t:raise ValueError('TEACHER_METADATA_SHAPE')
        if set(self.context_index.tolist())!=set(range(c)):raise ValueError('CONTEXT_INDEX_COVERAGE')
        sums=torch.zeros_like(self.context_weights).scatter_add(0,self.context_index,self.token_mean_weights.to(self.context_weights))
        if not torch.allclose(sums,torch.ones_like(sums),rtol=1e-6,atol=1e-7):raise ValueError('TOKEN_MEAN_NORMALIZATION')
        if not bool((self.context_weights>0).all() and (self.token_mean_weights>0).all()):raise ValueError('INVALID_MEAN_WEIGHTS')
        if self.teacher_logp.requires_grad or self.reference_nll.requires_grad:raise ValueError('TEACHER_MUST_BE_FROZEN')
        if not all(bool(torch.isfinite(t).all()) for t in (self.token_mean_weights,self.context_weights,self.teacher_logp,self.reference_nll)):
            raise FloatingPointError('NONFINITE_PANEL_METADATA')


def psi(value,tau=.1):
    return torch.where(value<=0,torch.zeros_like(value),torch.where(value<tau,value.square()/(2*tau),value-tau/2))


def psi_derivatives(value,tau=.1):
    first=torch.where(value<=0,torch.zeros_like(value),torch.where(value<tau,value/tau,torch.ones_like(value)))
    # At the two C1 breakpoints use the outside-region curvature convention.
    second=((value>0)&(value<tau)).to(value.dtype)/tau
    return first,second


def context_sum(value,batch):
    return value.new_zeros(batch.context_weights.numel()).scatter_add(0,batch.context_index,value)


def context_nll(logp,batch):
    token=-logp.gather(1,batch.target_ids[:,None]).squeeze(1)
    return context_sum(token*batch.token_mean_weights,batch)


def values(logits,batch,role,tau):
    logp=torch.log_softmax(logits,dim=-1);nll=context_nll(logp,batch)
    kl_tokens=(batch.teacher_logp.exp()*(batch.teacher_logp-logp)).sum(-1)
    kl=context_sum(kl_tokens*batch.token_mean_weights,batch)
    if role=='base':per=kl
    elif role=='past':per=psi(nll-batch.reference_nll,tau)
    elif role=='current':per=kl/tau+(nll-batch.reference_nll).square()/(2*tau*tau)
    else:raise ValueError('UNKNOWN_FUNCTIONAL_ROLE')
    return (per*batch.context_weights).sum(),(nll*batch.context_weights).sum(),per,nll


def logit_ggn_action(logits,direction,batch,role,tau):
    """PSD output curvature action, excluding model second derivatives.

    Current NLL-square contributes its profile-Jacobian outer product, not
    residual times the NLL Hessian. Past psi contributes BOTH psi'' times
    the NLL-Jacobian outer product AND psi' times CE Fisher.
    """
    p=torch.softmax(logits,dim=-1)
    token_weight=batch.token_mean_weights*batch.context_weights[batch.context_index]
    fisher=p*(direction-(p*direction).sum(-1,keepdim=True))
    if role=='base':return fisher*token_weight[:,None]
    grad_token=p-torch.nn.functional.one_hot(batch.target_ids,p.shape[-1]).to(p)
    dnll=context_sum((grad_token*direction).sum(-1)*batch.token_mean_weights,batch)
    if role=='current':
        diagonal=torch.full_like(batch.context_weights,1/tau)
        outer=torch.full_like(batch.context_weights,1/(tau*tau))
    elif role=='past':
        delta=context_nll(torch.log_softmax(logits,-1),batch)-batch.reference_nll
        diagonal,outer=psi_derivatives(delta,tau)
    else:raise ValueError('UNKNOWN_FUNCTIONAL_ROLE')
    return token_weight[:,None]*(diagonal[batch.context_index,None]*fisher+
        (outer*dnll)[batch.context_index,None]*grad_token)


@dataclass
class PanelLinearization:
    value: float
    mean_nll: float
    gradient: WeightTree
    nll_gradient: WeightTree
    context_rows: list[dict]
    ledger: dict


class FunctionalPanel:
    def __init__(self,batches:list[OutputBatch],role:str,*,tau=.1):
        if role not in ('current','base','past') or tau<=0:raise ValueError('FUNCTIONAL_PANEL_POLICY')
        self.batches=batches;self.role=role;self.tau=tau
        self.counts=dict(logits_forward=0,forward_input_tokens=0,gradient_backward=0,
                         jvp_calls=0,jvp_vectors=0,vjp_calls=0,vjp_vectors=0,ggn_matvecs=0)
        for b in batches:b.validate()
        total=sum(float(b.context_weights.sum()) for b in batches)
        if batches and abs(total-1)>1e-6:raise ValueError(('LOGICAL_REQUEST_MEAN_NOT_ONE',total))

    def _logits(self,weights,b):
        self.counts['logits_forward']+=1;self.counts['forward_input_tokens']+=b.input_tokens
        return b.logits_fn(weights)

    def linearize(self,weights:WeightTree,*,need_nll_gradient=False):
        params=tuple(w.detach().requires_grad_(True) for w in weights)
        gradient=zeros(weights);ngrad=zeros(weights);value=0.;nll=0.;rows=[]
        before=dict(self.counts)
        for b in self.batches:
            logits=self._logits(params,b);loss,nll_loss,per,nlls=values(logits,b,self.role,self.tau)
            grad=torch.autograd.grad(loss,params,retain_graph=need_nll_gradient,allow_unused=True)
            self.counts['gradient_backward']+=1
            gradient=add(gradient,tuple(torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,grad)))
            if need_nll_gradient:
                ng=torch.autograd.grad(nll_loss,params,allow_unused=True)
                self.counts['gradient_backward']+=1
                ngrad=add(ngrad,tuple(torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,ng)))
            value+=float(loss.detach());nll+=float(nll_loss.detach())
            rows.append(dict(identity=b.identity,values=per.detach().cpu().tolist(),nll=nlls.detach().cpu().tolist(),weights=b.context_weights.detach().cpu().tolist()))
        if not math.isfinite(value) or not math.isfinite(nll):raise FloatingPointError('NONFINITE_PANEL_OBSERVATION')
        return PanelLinearization(value,nll,finite(gradient),finite(ngrad),rows,{k:self.counts[k]-before[k] for k in before})

    def ggn(self,weights:WeightTree,direction:WeightTree)->WeightTree:
        params=tuple(w.detach().requires_grad_(True) for w in weights)
        out=zeros(weights);self.counts['ggn_matvecs']+=1
        for b in self.batches:
            # Reverse graph and forward AD are separate full-model passes. Both
            # count in the ledger; no claim that one small QP costs one backward.
            logits=self._logits(params,b)
            _,jv=torch.func.jvp(lambda *ws:self._logits(ws,b),params,direction)
            self.counts['jvp_calls']+=1;self.counts['jvp_vectors']+=1
            cotangent=logit_ggn_action(logits.detach(),jv.detach(),b,self.role,self.tau)
            grads=torch.autograd.grad(logits,params,grad_outputs=cotangent,allow_unused=True)
            self.counts['vjp_calls']+=1;self.counts['vjp_vectors']+=1
            out=add(out,tuple(torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,grads)))
        return finite(out)

    def value(self,weights):
        return self.observe(weights)['value']

    def observe(self,weights):
        """One actual no-grad panel pass; scalar and raw-context observations."""
        total=0.;nll=0.;rows=[]
        with torch.no_grad():
            for b in self.batches:
                loss,nll_loss,per,nlls=values(self._logits(weights,b),b,self.role,self.tau)
                total+=float(loss);nll+=float(nll_loss)
                rows.append(dict(identity=b.identity,values=per.cpu().tolist(),nll=nlls.cpu().tolist(),weights=b.context_weights.cpu().tolist()))
        if not math.isfinite(total) or not math.isfinite(nll):raise FloatingPointError('NONFINITE_PANEL_OBSERVATION')
        return dict(value=total,mean_nll=nll,context_rows=rows)
