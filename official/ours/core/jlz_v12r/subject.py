"""Same-mask request SUM subject loss; owner groups backward immediately."""
import time
import torch
from official.ours.core.jlz_realized_subject.subject import row_logprobs
from official.ours.common import require
from official.ours.core.jlz_native_writer_aware.routes import annotate


def evaluate(a,entry,built,backward=False,capture=False,active_previous=None,
             fixed_mask=None,terminal=False,owner_subset=None):
    annotate(entry);start=time.monotonic();B=entry['pack']['n_requests'];n=entry['pack']['n_rw']
    previous=torch.zeros(B,dtype=torch.bool) if active_previous is None else torch.as_tensor(active_previous,dtype=torch.bool).cpu().clone()
    require(previous.shape==(B,),'ACTIVE_MASK_SHAPE')
    fixed=None if fixed_mask is None else torch.as_tensor(fixed_mask,dtype=torch.bool).cpu()
    require(fixed is None or fixed.shape==(B,),'FIXED_MASK_SHAPE')
    v={l:t.to(a.device).detach().requires_grad_(backward and not terminal) for l,t in built['v'].items()}
    adj={l:torch.zeros_like(t) for l,t in v.items()};nll=torch.zeros(B,n,dtype=torch.float64)
    kl=torch.zeros(B,dtype=torch.float64);F=torch.zeros(B,dtype=torch.float64);active=previous.clone()
    bases={l:[] for l in a.sites};seen=set();calls=0;backs=0;tokens=valid=padded=0;masked_sum=0.
    for group in entry['groups']:
        owners={r['request'] for r in group['rows']};require(len(owners)==1,'IMMEDIATE_BACKWARD_COMPLETE_OWNER')
        owner=next(iter(owners))
        if owner_subset is not None and owner not in owner_subset:continue
        calls+=1;padded+=group['tokens']['input_ids'].numel();valid+=int(group['tokens']['attention_mask'].sum())
        with torch.set_grad_enabled(backward and not terminal):
            nh,fh,_,base=a.masked(group,v,capture=capture);probs=row_logprobs(a,group['rows'],nh,fh)
            loss=nh.reshape(-1)[0]*0
            for row,lp in zip(group['rows'],probs):
                require(row['request']==owner,'OWNER_LOSS_IDENTITY');seen.add(row['global_row']);tokens+=len(lp)
                if row['kind']=='rewrite':
                    labels=row['target'][row['target']!=-100].to(a.device)
                    value=-lp.gather(1,labels[:,None]).mean();loss=loss+value/n
                    nll[owner,row['reduction_index']]=float(value.detach())
                else:
                    teacher=entry['teachers'][owner].to(a.device)
                    value=(lp.exp()*(lp-teacher)).sum();loss=loss+.0625*value;kl[owner]=float(value.detach())
            require(bool(torch.isfinite(loss)),'NONFINITE_NATIVE_SUBJECT');F[owner]=float(loss.detach())
            active[owner]=bool(fixed[owner]) if fixed is not None else bool(previous[owner] or F[owner]>=.05)
            if active[owner]:masked_sum+=float(loss.detach())
            if backward and not terminal and active[owner]:
                gradients=torch.autograd.grad(loss,tuple(v.values()),allow_unused=False)
                for dst,g in zip(adj.values(),gradients):dst.add_(g)
                backs+=1
        if capture:
            for l in a.sites:bases[l].append(base[l].detach().cpu())
    complete=[all(r['global_row'] in seen for g in entry['groups'] for r in g['rows'] if r['request']==owner) for owner in range(B)]
    require(owner_subset is not None or all(complete),'NATIVE_ALL_OWNER_FORWARD_COVERAGE')
    return dict(F=F,nll=nll,kl=kl,active_mask=active,task_sum=float(F.sum()),masked_backward_sum=masked_sum,
        adjoint={l:t.detach().cpu() for l,t in adj.items()} if backward and not terminal else None,
        masked_bases={l:torch.cat(parts) for l,parts in bases.items()} if capture else None,
        forward_groups=calls,backward_groups=backs,logical_forward=1,logical_backward=int(backs>0),
        complete_owner=complete,seen_rows=len(seen),valid_tokens=valid,padded_tokens=padded,
        prediction_tokens=tokens,seconds=time.monotonic()-start,terminal_no_backward=terminal,
        reduction='SUM_active_request_losses_no_extra_B_or_context_reduction')


@torch.no_grad()
def pullback(built,adjoint,R,blind=False):
    """Lambda M.T, full B columns; row adjoints are already native-weighted."""
    start=time.monotonic();result={};rows=built['rows'];B=next(iter(R.values())).shape[1]
    owners=torch.tensor([r['request'] for r in rows],dtype=torch.long)
    for l,value in R.items():
        lam=adjoint[l].double();raw=built['raw'][l].double();P=built['P'][l].double()
        require(lam.shape[0]==raw.shape[0]==len(rows) and P.shape[1]==B,'RESPONSE_PULLBACK_SHAPE')
        if blind:
            gradient=torch.zeros(value.shape,dtype=torch.float64)
            gradient.index_add_(1,owners,lam.T)
        else:
            gradient=(lam.to(P.device).T@(raw.to(P.device)@P)).cpu()
        require(bool(torch.isfinite(gradient).all()),'NONFINITE_SAME_LAYER_PULLBACK')
        result[l]=gradient.to(value.device)
    return result,dict(seconds=time.monotonic()-start,FP64=True,whole_B=B,off_owner_coupling=not blind,
        backend='same_owner_adjoint_sum' if blind else 'Lambda_M_transpose',builder_reverse_calls=0,
        solve_VJP_calls=0,native_row_reduction_applied_again=False)
