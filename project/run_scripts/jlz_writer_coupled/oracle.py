"""SUM native objective on the physical all-token joint model."""
import time
import torch
from .common import require
from .physical import materialize

class Oracle:
    def __init__(self,a,entry,residents,reference_groups,eta,route='direct',cached=True):
        self.a,self.entry,self.residents,self.reference_groups=a,entry,residents,reference_groups
        self.eta,self.route,self.cached=eta,route,cached
        self.calls=0;self.backward_calls=0

    def __call__(self,v,backward):
        start=time.monotonic();a=self.a;e=self.entry;B=e['pack']['n_requests'];n_rw=e['pack']['n_rw']
        D={l:(v[l].detach()*e['anchors'][l][None,:]).requires_grad_(backward) for l in a.sites}
        with torch.set_grad_enabled(backward and self.route=='dense'):
            weights={l:materialize(e['entry_weights'][l],D[l],e['P'][l]) for l in a.sites}
        grad={l:torch.zeros_like(D[l]) for l in a.sites};past_grad={l:torch.zeros_like(D[l]) for l in a.sites};rawkeys={l:[] for l in a.sites}
        nll=torch.zeros((B,n_rw),dtype=torch.float64);kl=torch.zeros(B,dtype=torch.float64)
        rk=re=0.;loss_total=0.;token_count=0
        work=[(False,g) for g in e['groups']]
        if self.eta:work += [(True,g) for g in self.reference_groups]
        with torch.set_grad_enabled(backward):
            for index,(past,group) in enumerate(work):
                captures={}
                def capture(l,x):
                    if not past:captures[l]=x.detach()
                with a.install(D,e['P'],weights,self.route,capture):
                    if self.cached:nh,fh=a.cached(group['cache'])
                    else:nh,fh=a.full({k:x.to(a.device) for k,x in group['tokens'].items()})
                # One batched full-vocabulary head for required positions only.
                hidden=[];spans=[]
                for j,r in enumerate(group['rows']):
                    pos=torch.nonzero(r['target']!=-100).flatten().to(a.device) if r['kind']=='rewrite' else torch.tensor([r['lookup']],device=a.device)
                    spans.append((len(hidden),len(pos)))
                    hidden.extend((nh if r['kind']=='rewrite' else fh)[j,pos].unbind(0))
                logits=a.head(torch.stack(hidden));logp=logits.log_softmax(-1);cursor=0
                loss=logits.reshape(-1)[0]*0
                for j,r in enumerate(group['rows']):
                    count=spans[j][1];lp=logp[cursor:cursor+count];cursor+=count;request=r['request'];token_count+=count
                    if r['kind']=='rewrite':
                        target=r['target'][r['target']!=-100].to(a.device)
                        value=-lp.gather(1,target[:,None]).mean()
                        if past:
                            baseline=self.residents[request]['context_nll'][r['context_index']]
                            term=.5*torch.relu(value-baseline).square()/r['n_rw']/len(self.residents)
                            re+=float(term.detach());loss=loss+self.eta*B*a.profile['preservation_E']*term
                        else:
                            context=r['global_row']%(n_rw+1)
                            nll[request,context]=float(value.detach());loss=loss+value/n_rw
                            for l in a.sites:rawkeys[l].append(captures[l][j,r['lookup']].cpu().clone())
                    else:
                        teacher=(self.residents[request]['teacher'] if past else e['teachers'][request]).to(a.device)
                        value=(lp.exp()*(lp-teacher)).sum()
                        if past:
                            term=value/len(self.residents);rk+=float(term.detach());loss=loss+self.eta*B*a.profile['preservation_K']*term
                        else:
                            kl[request]=float(value.detach());loss=loss+a.profile['kl_factor']*value
                require(bool(torch.isfinite(loss)),'NONFINITE_SMOOTH')
                loss_total+=float(loss.detach())
                if backward:
                    partial=torch.autograd.grad(loss,tuple(D.values()),retain_graph=self.route=='dense' and index<len(work)-1,allow_unused=False)
                    for l,g in zip(a.sites,partial):
                        require(bool(torch.isfinite(g).all()),'NONFINITE_GRADIENT');grad[l].add_(g.detach())
                        if past:past_grad[l].add_(g.detach())
                # Do not hold one dense MB's graph while forwarding the next.
                del loss,logits,logp,nh,fh,captures,hidden,lp,value
                if past:del term
        self.calls+=1;self.backward_calls+=int(backward)
        payload=dict(weights={l:w.detach() for l,w in weights.items()},keys={l:torch.stack(k).T for l,k in rawkeys.items()},
            context_nll=nll,native_kl=kl,stats=dict(smooth=loss_total,native_nll_sum=float(nll.mean(1).sum()),
            native_kl_sum=float(kl.sum()),reference_K=rk,reference_E=re,actual_B=B,reference_count=len(self.residents),
            seconds=time.monotonic()-start,prediction_tokens=token_count,backward=bool(backward),route=self.route,cache=self.cached,
            past_gradient_norm={str(l):float((g*e['anchors'][l][None,:]).double().norm()) for l,g in past_grad.items()},
            whole_gradient_norm={str(l):float((g*e['anchors'][l][None,:]).double().norm()) for l,g in grad.items()}))
        return dict(smooth=loss_total,grad={l:grad[l]*e['anchors'][l][None,:] for l in a.sites} if backward else None,payload=payload)
