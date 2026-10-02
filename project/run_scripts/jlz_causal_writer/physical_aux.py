"""Chunk adjoints are returned to the SAME candidate's causal P graph."""
import time
import torch
from .common import require
from .subject import row_logprobs
from .profile import move
from .physical_linear import materialize

def subset_group(group, selected):
    ix=[i for i,r in enumerate(group['rows']) if r['request'] in selected]
    if not ix:return None
    out=dict(rows=[group['rows'][i] for i in ix])
    cache=group['cache'];kw=dict(cache['kwargs'])
    def select(t):
        return t[ix] if isinstance(t,torch.Tensor) and t.ndim>0 and t.shape[0]==len(group['rows']) and t.shape[0]!=1 else t
    for k,v in kw.items():kw[k]=tuple(select(t) for t in v) if isinstance(v,tuple) else select(v)
    out['cache']=dict(key=cache['key'][ix],residual=cache['residual'][ix],kwargs=kw)
    out['tokens']={k:v[ix] for k,v in group['tokens'].items()}
    return out

def actual(a,entry,builder,D,teacher,current_ids,residents,reference_groups,past_ids,
           backward,terminal=False,route='direct'):
    start=time.monotonic();B=entry['pack']['n_requests'];n_rw=entry['pack']['n_rw']
    # Leaves permit bounded MB graphs; every collected P adjoint is sent back by optimize.
    ds={l:d.detach().requires_grad_(backward) for l,d in D.items()}
    ps={l:p.detach().requires_grad_(backward) for l,p in builder['P'].items()}
    gd={l:torch.zeros_like(d) for l,d in ds.items()};gp={l:torch.zeros_like(p) for l,p in ps.items()}
    nll=torch.zeros((B,n_rw),dtype=torch.float64);kl=torch.zeros(B,dtype=torch.float64)
    rawkeys={l:[] for l in a.sites};stats=dict(subject=0.,distillation=0.,current_kl=0.,past_kl=0.,past_degradation=0.)
    total=0.;tokens=0;rows=0
    work=[(False,subset_group(g,set(current_ids))) for g in entry['groups']]
    work += [(True,subset_group(g,set(past_ids))) for g in reference_groups]
    with torch.set_grad_enabled(backward):
        for past,g in work:
            if g is None:continue
            weights=({l:materialize(entry['entry_weights'][l],ds[l],ps[l]) for l in a.sites}
                     if route=='dense' else builder['weights'])
            nh,fh,hidden,keys=a.actual(g,ds,ps,weights,route,capture=not past)
            probs=row_logprobs(a,g['rows'],nh,fh);loss=nh.reshape(-1)[0]*0
            for j,(r,lp) in enumerate(zip(g['rows'],probs)):
                req=r['request'];tokens+=len(lp);rows+=1
                if r['kind']=='rewrite':
                    labels=r['target'][r['target']!=-100].to(a.device)
                    value=-lp.gather(1,labels[:,None]).mean()
                    if past:
                        term=.5*torch.relu(value-residents[req]['context_nll'][r['context_index']]).square()/r['n_rw']
                        weighted=(B/len(residents))*term
                        loss=loss+weighted;stats['past_degradation']+=float(weighted.detach())
                    else:
                        context=r['global_row']%(n_rw+1);nll[req,context]=float(value.detach())
                        if terminal:
                            for l in a.sites:rawkeys[l].append(keys[l][j].detach().cpu().clone())
                        else:
                            t=teacher[r['global_row']]
                            ch=.5*sum((hidden[l][j]-t['hidden'][l].to(a.device)).square().sum()/entry['anchors'][l][req].square() for l in a.sites)/n_rw
                            tlp=t['logp'].to(a.device)
                            cd=(tlp.exp()*(tlp-lp)).sum(-1).mean()/n_rw
                            loss=loss+.1*(ch+cd)
                            stats['subject']+=float(ch.detach());stats['distillation']+=float(cd.detach())
                else:
                    target=(residents[req]['teacher'] if past else entry['teachers'][req]).to(a.device)
                    value=(lp.exp()*(lp-target)).sum()
                    if past:
                        weighted=B/len(residents)*a.profile['kl_factor']*value
                        loss=loss+weighted;stats['past_kl']+=float(weighted.detach())
                    else:
                        kl[req]=float(value.detach())
                        if not terminal:loss=loss+a.profile['kl_factor']*value
                        stats['current_kl']+=float(value.detach())
            require(bool(torch.isfinite(loss)),'NONFINITE_PHYSICAL')
            total+=float(loss.detach())
            if backward:
                gradients=torch.autograd.grad(loss,tuple(ds.values())+tuple(ps.values()),allow_unused=True)
                for l,gradient in zip(a.sites,gradients[:len(ds)]):
                    if gradient is not None:gd[l].add_(gradient.detach())
                for l,gradient in zip(a.sites,gradients[len(ds):]):
                    if gradient is not None:gp[l].add_(gradient.detach())
            del nh,fh,hidden,keys,loss,probs,lp,value
    payload=dict(context_nll=nll,native_kl=kl,weights={l:w.detach() for l,w in builder['weights'].items()})
    if terminal:payload['keys']={l:torch.stack(v).T for l,v in rawkeys.items()}
    return dict(grad_D=gd,grad_P=gp,payload=payload,loss_sum=total,
                stats=stats,seconds=time.monotonic()-start,prediction_tokens=tokens,rows=rows)
