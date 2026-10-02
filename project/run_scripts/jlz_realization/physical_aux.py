"""Current-only physical pulse and terminal observer; no replay objects."""
import time
import torch
from .common import require
from .subject import row_logprobs
from .physical_linear import materialize
from .geometry import mean_keys
from .telemetry import context_decomposition

def subset_group(group,selected):
    ix=[i for i,r in enumerate(group['rows']) if r['request'] in selected]
    if not ix:return None
    def select(t):
        return t[ix] if isinstance(t,torch.Tensor) and t.ndim>0 and t.shape[0]==len(group['rows']) and t.shape[0]!=1 else t
    kw={k:tuple(select(t) for t in v) if isinstance(v,tuple) else select(v) for k,v in group['cache']['kwargs'].items()}
    return dict(rows=[group['rows'][i] for i in ix],tokens={k:v[ix] for k,v in group['tokens'].items()},
        cache=dict(key=group['cache']['key'][ix],residual=group['cache']['residual'][ix],kwargs=kw))

def actual(a,entry,builder,D,teacher,current_ids,backward,terminal=False,route='direct',diagnostics=True):
    start=time.monotonic();B=entry['pack']['n_requests'];n_rw=entry['pack']['n_rw']
    ds={l:d.detach().requires_grad_(backward) for l,d in D.items()}
    ps={l:p.detach().requires_grad_(backward) for l,p in builder['P'].items()}
    gd={l:torch.zeros_like(d) for l,d in ds.items()};gp={l:torch.zeros_like(p) for l,p in ps.items()}
    nll=torch.zeros((B,n_rw),dtype=torch.float64);kl=torch.zeros(B,dtype=torch.float64)
    rawkeys={l:[] for l in a.sites};key_rows=[];decomposition=[]
    stats=dict(subject=0.,distillation=0.,current_kl=0.);total=0.;tokens=0;rows=0
    with torch.set_grad_enabled(backward):
        for group in entry['groups']:
            g=subset_group(group,set(current_ids))
            if g is None:continue
            weights=({l:materialize(entry['entry_weights'][l],ds[l],ps[l]) for l in a.sites} if route=='dense' else builder['weights'])
            nh,fh,hidden,keys=a.actual(g,ds,ps,weights,route,capture=True)
            probs=row_logprobs(a,g['rows'],nh,fh);loss=nh.reshape(-1)[0]*0
            for j,(r,lp) in enumerate(zip(g['rows'],probs)):
                req=r['request'];tokens+=len(lp);rows+=1
                if r['kind']=='rewrite':
                    labels=r['target'][r['target']!=-100].to(a.device)
                    value=-lp.gather(1,labels[:,None]).mean()
                    context=r['global_row']%(n_rw+1);nll[req,context]=float(value.detach())
                    if terminal:
                        key_rows.append(r)
                        for l in a.sites:rawkeys[l].append(keys[l][j].detach().cpu().clone())
                    else:
                        t=teacher[r['global_row']]
                        ch=.5*sum((hidden[l][j]-t['hidden'][l].to(a.device)).square().sum()/entry['anchors'][l][req].square() for l in a.sites)/n_rw
                        tlp=t['logp'].to(a.device);cd=(tlp.exp()*(tlp-lp)).sum(-1).mean()/n_rw
                        # SUM gradient = B * |I|/B * cohort mean: no extra factor.
                        loss=loss+.1*(ch+cd);stats['subject']+=float(ch.detach());stats['distillation']+=float(cd.detach())
                else:
                    target=entry['teachers'][req].to(a.device);value=(lp.exp()*(lp-target)).sum()
                    kl[req]=float(value.detach());stats['current_kl']+=float(value.detach())
                    if not terminal:loss=loss+a.profile['kl_factor']*value
            require(bool(torch.isfinite(loss)),'NONFINITE_PHYSICAL')
            if diagnostics:
                decomposition.append(context_decomposition(a,entry,builder,D,g['rows'],hidden,keys,teacher))
            total+=float(loss.detach())
            if backward:
                gradients=torch.autograd.grad(loss,tuple(ds.values())+tuple(ps.values()),allow_unused=True)
                for l,grad in zip(a.sites,gradients[:len(ds)]):
                    if grad is not None:gd[l].add_(grad.detach())
                for l,grad in zip(a.sites,gradients[len(ds):]):
                    if grad is not None:gp[l].add_(grad.detach())
            del nh,fh,hidden,keys,loss,probs,lp,value
    payload=dict(context_nll=nll,native_kl=kl,weights={l:w.detach() for l,w in builder['weights'].items()})
    require(bool(torch.isfinite(nll).all()) and bool(torch.isfinite(kl).all()),'NONFINITE_PHYSICAL_OBSERVATIONS')
    if terminal:
        payload['keys']={l:mean_keys(torch.stack(v).T,key_rows,entry['pack']) for l,v in rawkeys.items()}
    return dict(grad_D=gd,grad_P=gp,payload=payload,loss_sum=total,stats=stats,decomposition=decomposition,
        seconds=time.monotonic()-start,prediction_tokens=tokens,rows=rows)
