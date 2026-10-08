"""Native row reductions and the sole v-adjoint bridge to the causal graph."""
import time
import torch
from official.ours.config import require_config
from official.ours.common import require
from .allocation import loss as allocation

def coefficients(config):
    config=require_config(config)
    return dict(nll=1.,kl=config['lambda_KL'],norm=config['lambda_N'],allocation=.1)


def row_logprobs(a, rows, nll, final):
    selected = []; widths = []
    for j, r in enumerate(rows):
        pos = (torch.nonzero(r['target']!=-100).flatten().to(a.device) if r['kind']=='rewrite'
               else torch.tensor([r['lookup']], device=a.device))
        widths.append(len(pos)); selected.extend((nll if r['kind']=='rewrite' else final)[j,pos].unbind(0))
    return list(a.head(torch.stack(selected)).log_softmax(-1).split(widths))


def row_terms(a, entry, group, v):
    nh, fh, _, _ = a.masked(group, v)
    probs = row_logprobs(a, group['rows'], nh, fh)
    zero = nh.reshape(-1)[0]*0
    terms = dict(nll=zero, kl=zero, norm=zero); B=entry['pack']['n_requests']; n=entry['pack']['n_rw']
    weights = entry['pack']['key_context_weights']; values = []; tokens = 0
    for r, lp in zip(group['rows'], probs):
        req = r['request']; c = r['global_row'] % (n+1); idx=r['global_row']; tokens+=len(lp)
        if r['kind']=='rewrite':
            labels=r['target'][r['target']!=-100].to(a.device)
            value=-lp.gather(1,labels[:,None]).mean()
            terms['nll']=terms['nll']+value/(B*n)
            w=weights[req*n+c]
            terms['norm']=terms['norm']+sum(w*v[l][idx].norm()/entry['anchors'][l][req].square()/B for l in a.sites)
        else:
            teacher=entry['teachers'][req].to(a.device)
            value=(lp.exp()*(lp-teacher)).sum()
            terms['kl']=terms['kl']+value/B
        values.append((r['kind'],req,c,float(value.detach())))
    return terms, values, tokens


def evaluate(a, entry, R, built, arm, components=False, dense=False):
    """Return gradients of full request MEAN; optimizer applies B and scale once."""
    coef=coefficients(a.profile)
    start=time.monotonic(); sites=tuple(R); originals=[built['v'][l] for l in sites]
    v={l:x if dense else x.detach().requires_grad_(True) for l,x in zip(sites,originals)}
    leaves=tuple(v.values()); totals=dict(nll=0.,kl=0.,norm=0.); B=entry['pack']['n_requests']; n=entry['pack']['n_rw']
    nll=torch.zeros(B,n,dtype=torch.float64); kl=torch.zeros(B,dtype=torch.float64)
    adj={name:[torch.zeros_like(x) for x in leaves] for name in (('nll','kl','norm') if components else ('total',))}
    tokens=0; physical_backward=0; dense_loss=None
    for group in entry['groups']:
        terms, values, nt = row_terms(a,entry,group,v); tokens+=nt
        for name,t in terms.items():
            require(bool(torch.isfinite(t)), 'NONFINITE_NATIVE_'+name);totals[name]+=float(t.detach())
        for kind,req,c,value in values:
            if kind=='rewrite':nll[req,c]=value
            else:kl[req]=value
        if dense:
            piece=sum(coef[k]*value for k,value in terms.items())
            dense_loss=piece if dense_loss is None else dense_loss+piece
        else:
            names=('nll','kl','norm') if components else ('total',)
            for i,name in enumerate(names):
                scalar=terms[name] if name!='total' else sum(coef[k]*value for k,value in terms.items())
                gs=torch.autograd.grad(scalar,leaves,retain_graph=i+1<len(names),allow_unused=True)
                for dst,g in zip(adj[name],gs):
                    if g is not None:dst.add_(g)
                physical_backward+=1
    alloc,energy=allocation(R,built['geometry'],entry['anchors'],arm)
    totals['allocation']=float(alloc.detach())
    if dense:
        total=torch.autograd.grad(dense_loss+coef['allocation']*alloc,tuple(R.values()),retain_graph=False)
        component=None; physical_backward+=1; bridge_count=0
    else:
        seeds=adj['total'] if not components else [sum(coef[k]*adj[k][i] for k in adj) for i in range(len(sites))]
        component={}
        if components:
            for name in ('nll','kl','norm'):
                component[name]=torch.autograd.grad(originals,tuple(R.values()),adj[name],retain_graph=True,allow_unused=False)
            component['allocation']=torch.autograd.grad(alloc,tuple(R.values()),retain_graph=True)
        total=torch.autograd.grad(originals+[alloc],tuple(R.values()),seeds+[alloc.new_tensor(coef['allocation'])],retain_graph=False)
        bridge_count=1 # component diagnostics are separate, never accumulated into optimizer.
    require(all(bool(torch.isfinite(x).all()) for x in total),'NONFINITE_TOTAL_GRADIENT')
    return dict(gradient=dict(zip(sites,total)), components={k:dict(zip(sites,v)) for k,v in component.items()} if component else None,
        losses=totals,total_mean=sum(coef[k]*totals[k] for k in totals),energy=energy,nll=nll,kl=kl,
        seconds=time.monotonic()-start,prediction_tokens=tokens,masked_rows=sum(len(g['rows']) for g in entry['groups']),
        masked_backward_calls=physical_backward,builder_optimizer_bridges=bridge_count,
        component_builder_backwards=4 if components else 0)
