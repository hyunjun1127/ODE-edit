"""Frozen-entry full-block absolute subject deltas and native SUM loss."""
import time
import torch
from .common import require

def row_logprobs(a, rows, nll, final):
    selected=[]; widths=[]
    for j,r in enumerate(rows):
        pos=(torch.nonzero(r['target']!=-100).flatten().to(a.device) if r['kind']=='rewrite'
             else torch.tensor([r['lookup']],device=a.device))
        widths.append(len(pos)); selected.extend((nll if r['kind']=='rewrite' else final)[j,pos].unbind(0))
    lp=a.head(torch.stack(selected)).log_softmax(-1)
    return list(lp.split(widths))

def native(a,entry,D,backward,teacher_requests=()):
    start=time.monotonic();pack=entry['pack'];B=pack['n_requests'];n_rw=pack['n_rw']
    nll=torch.zeros((B,n_rw),dtype=torch.float64);kl=torch.zeros(B,dtype=torch.float64)
    teachers={};selected=set(teacher_requests);tokens=0
    with torch.set_grad_enabled(backward):
        for group in entry['groups']:
            capture=any(r['request'] in selected and r['kind']=='rewrite' for r in group['rows'])
            nh,fh,hidden=a.native(group,D,capture)
            probs=row_logprobs(a,group['rows'],nh,fh);loss=nh.reshape(-1)[0]*0
            for j,(r,lp) in enumerate(zip(group['rows'],probs)):
                req=r['request'];tokens+=len(lp)
                if r['kind']=='rewrite':
                    targets=r['target'][r['target']!=-100].to(a.device)
                    value=-lp.gather(1,targets[:,None]).mean()
                    context=r['global_row']%(n_rw+1)
                    nll[req,context]=float(value.detach());loss=loss+value/n_rw
                    if req in selected:
                        teachers[r['global_row']]=dict(logp=lp.detach().cpu().clone(),
                            hidden={l:h[j].detach().cpu().clone() for l,h in hidden.items()})
                else:
                    target=entry['teachers'][req].to(a.device)
                    value=(lp.exp()*(lp-target)).sum()
                    kl[req]=float(value.detach());loss=loss+a.profile['kl_factor']*value
            require(bool(torch.isfinite(loss)),'NONFINITE_NATIVE')
            if backward: loss.backward()
            del loss,probs,nh,fh,hidden,lp,value
        norm=.5*sum((torch.linalg.vector_norm(d,dim=0)/entry['anchors'][l].square()).sum() for l,d in D.items())
        require(bool(torch.isfinite(norm)),'NONFINITE_NATIVE_NORM')
        if backward:norm.backward()
    return dict(nll=nll,kl=kl,norm_sum=float(norm.detach()),teachers=teachers,
                native_sum=float(nll.mean(1).sum()+a.profile['kl_factor']*kl.sum())+float(norm.detach()),
                seconds=time.monotonic()-start,prediction_tokens=tokens,
                rows=sum(len(g['rows']) for g in entry['groups']))

