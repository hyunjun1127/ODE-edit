"""Native SUM gradient; context-v leaves get one masked adjoint per row."""
import time
import torch
from project.run_scripts.jlz_realized_subject.subject import row_logprobs
from .common import require
from .routes import annotate

def evaluate(a,entry,built,backward=False,capture=False,dense_R=None,group_indices=None):
    annotate(entry)
    start=time.monotonic();B=entry['pack']['n_requests'];n=entry['pack']['n_rw']
    v={l:t.to(a.device) if dense_R is not None else t.to(a.device).detach().requires_grad_(backward) for l,t in built['v'].items()}
    targets=dense_R if dense_R is not None else v
    adj={l:torch.zeros_like(t) for l,t in targets.items()};nll=torch.zeros(B,n,dtype=torch.float64);kl=torch.zeros(B,dtype=torch.float64)
    bases={l:[] for l in a.sites};tokens=0;padded=0;valid=0;seen=set();processed=0
    for gi,group in enumerate(entry['groups']):
        if group_indices is not None and gi not in group_indices:continue
        processed+=1;padded+=group['tokens']['input_ids'].numel();valid+=int(group['tokens']['attention_mask'].sum())
        with torch.set_grad_enabled(backward):
            nh,fh,_,base=a.masked(group,v,capture=capture)
            probs=row_logprobs(a,group['rows'],nh,fh)
            loss=nh.reshape(-1)[0]*0
            for row,lp in zip(group['rows'],probs):
                req=row['request'];c=row['reduction_index'];tokens+=len(lp);seen.add(row['global_row'])
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
    complete=[all(r['global_row'] in seen for g in entry['groups'] for r in g['rows'] if r['request']==owner) for owner in range(B)]
    require(group_indices is not None or all(complete),'INCOMPLETE_NATIVE_LOSS')
    return dict(complete_owner=complete,seen_rows=len(seen),forward_groups=processed,valid_tokens=valid,padded_tokens=padded,F=nll.mean(1)+.0625*kl,nll=nll,kl=kl,adjoint={l:t.cpu() for l,t in adj.items()} if backward else None,
                masked_bases={l:torch.cat(t) for l,t in bases.items()} if capture else None,
                seconds=time.monotonic()-start,tokens=tokens,backward_groups=processed if backward else 0,
                reduction='request_SUM_gradient_request_MEAN_report')
