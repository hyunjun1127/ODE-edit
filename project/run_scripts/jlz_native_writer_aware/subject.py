"""Native SUM gradient; context-v leaves get one masked adjoint per row."""
import time
import torch
from project.run_scripts.jlz_realized_subject.subject import row_logprobs
from .common import require

def evaluate(a,entry,built,backward=False,capture=False,dense_R=None):
    start=time.monotonic();B=entry['pack']['n_requests'];n=entry['pack']['n_rw']
    v={l:t.to(a.device) if dense_R is not None else t.to(a.device).detach().requires_grad_(backward) for l,t in built['v'].items()}
    targets=dense_R if dense_R is not None else v
    adj={l:torch.zeros_like(t) for l,t in targets.items()};nll=torch.zeros(B,n,dtype=torch.float64);kl=torch.zeros(B,dtype=torch.float64)
    bases={l:[] for l in a.sites};tokens=0
    for group in entry['groups']:
        with torch.set_grad_enabled(backward):
            nh,fh,_,base=a.masked(group,v,capture=capture)
            probs=row_logprobs(a,group['rows'],nh,fh)
            loss=nh.reshape(-1)[0]*0
            for row,lp in zip(group['rows'],probs):
                req=row['request'];c=row['global_row']%(n+1);tokens+=len(lp)
                if row['kind']=='rewrite':
                    labels=row['target'][row['target']!=-100].to(a.device)
                    value=-lp.gather(1,labels[:,None]).mean()
                    loss=loss+value/n;nll[req,c]=float(value.detach())
                else:
                    teacher=entry['teachers'][req].to(a.device)
                    value=(lp.exp()*(lp-teacher)).sum()
                    loss=loss+.0625*value;kl[req]=float(value.detach())
            require(bool(torch.isfinite(loss)),'NONFINITE_NATIVE')
            if backward:
                grads=torch.autograd.grad(loss,tuple(targets.values()),retain_graph=dense_R is not None)
                for dst,g in zip(adj.values(),grads):dst.add_(g)
        if capture:
            for l in a.sites:bases[l].append(base[l].detach().cpu())
    return dict(F=nll.mean(1)+.0625*kl,nll=nll,kl=kl,adjoint={l:t.cpu() for l,t in adj.items()} if backward else None,
                masked_bases={l:torch.cat(t) for l,t in bases.items()} if capture else None,
                seconds=time.monotonic()-start,tokens=tokens,backward_groups=len(entry['groups']) if backward else 0,
                reduction='request_SUM_gradient_request_MEAN_report')
