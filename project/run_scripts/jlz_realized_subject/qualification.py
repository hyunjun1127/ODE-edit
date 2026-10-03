"""Bounded actual-model qualification of the production Tprime graph."""
from contextlib import contextmanager
import torch
from .common import require,write
from .causal_builder import build
from .subject import evaluate
from .entry import prepare_entry


def fixed(a,entry,scale=.025):
    B=entry['pack']['n_requests'];out={}
    for l in a.sites:
        x=torch.arange(a.dims[l][0]*B,device=a.device,dtype=torch.float32).reshape(a.dims[l][0],B)
        x=torch.sin(x+l+1);x=x/x.norm(dim=0)
        out[l]=(x*(scale*entry['anchors'][l])).detach().requires_grad_(True)
    return out


def compare(left,right):
    out={}
    for l in left:
        x,y=left[l].double(),right[l].double()
        err=float((x-y).square().mean().sqrt());ref=float(y.square().mean().sqrt())
        out[str(l)]=dict(error_RMS=err,reference_RMS=ref,limit=1e-6+1e-3*ref,passed=err<=1e-6+1e-3*ref)
    return out


def singletons(entry):
    groups=[]
    for g in reversed(entry['groups']):
        for i in reversed(range(len(g['rows']))):
            def select(t):
                return t[i:i+1] if isinstance(t,torch.Tensor) and t.ndim>=2 and t.shape[0]==len(g['rows']) and t.shape[0]!=1 else t
            kw={k:tuple(select(t) for t in v) if isinstance(v,tuple) else select(v) for k,v in g['cache']['kwargs'].items()}
            groups.append(dict(rows=[g['rows'][i]],tokens={k:v[i:i+1] for k,v in g['tokens'].items()},
                cache=dict(key=g['cache']['key'][i:i+1],residual=g['cache']['residual'][i:i+1],kwargs=kw)))
    return dict(entry,groups=groups,first_geometry={})


@contextmanager
def full_masked(a):
    original=a.masked
    def call(group,v,capture=False):
        handles=[];ix=torch.arange(len(group['rows']),device=a.device)
        pos=torch.tensor([r['lookup'] for r in group['rows']],device=a.device)
        rid=torch.tensor([r['global_row'] for r in group['rows']],device=a.device)
        for l in a.sites:
            def hook(m,args,out,l=l):
                h=a.unwrap(out).clone();h[ix,pos]=h[ix,pos]+v[l][rid]
                return (h,)+out[1:] if isinstance(out,tuple) else h
            handles.append(a.blocks[l].register_forward_hook(hook))
        try:
            nh,fh=a.full({k:x.to(a.device) for k,x in group['tokens'].items()})
            return nh,fh,{},{}
        finally:
            for h in handles:h.remove()
    a.masked=call
    try:yield
    finally:a.masked=original


def one(a,entry,route='direct',dense=False,stop_geometry=False):
    R=fixed(a,entry);built=build(a,entry,R,0,route,stop_geometry=stop_geometry)
    result=evaluate(a,entry,R,built,'A',dense=dense)
    return dict(loss=result['total_mean'],gradient={l:g.detach().clone() for l,g in result['gradient'].items()},
                keys={l:g['K'].detach().clone() for l,g in built['geometry'].items()},
                residual={str(l):g['metadata']['relative_residual'] for l,g in built['geometry'].items()})


def qualify(a,entry,out):
    results={}
    baseline=one(a,entry)
    for name,entry2,route,dense in [('dense',entry,'dense',True),('reversed_MB1',singletons(entry),'direct',False)]:
        a.checkpoint_enabled=not dense
        other=one(a,entry2,route,dense)
        gradients=compare(baseline['gradient'],other['gradient'])
        err=abs(baseline['loss']-other['loss']);limit=1e-5+1e-4*abs(other['loss'])
        results[name]=dict(loss_error=err,limit=limit,gradients=gradients,solve=other['residual'])
        write(out/(name+'.json'),results[name])
        require(err<=limit and all(x['passed'] for x in gradients.values()),'TPRIME_'+name+'_PARITY')
    a.checkpoint_enabled=True
    with full_masked(a):full=one(a,entry)
    gradients=compare(baseline['gradient'],full['gradient']);err=abs(baseline['loss']-full['loss'])
    results['full_model_hooks']=dict(loss_error=err,limit=1e-5+1e-4*abs(full['loss']),gradients=gradients)
    write(out/'full-model-hooks.json',results['full_model_hooks'])
    require(err<=results['full_model_hooks']['limit'] and all(x['passed'] for x in gradients.values()),'FULL_MASKED_PARITY')
    stopped=one(a,entry,stop_geometry=True)
    write(out/'stop-geometry-negative-control.json',dict(loss_difference=abs(stopped['loss']-baseline['loss']),
        gradient_difference=compare(stopped['gradient'],baseline['gradient']),production_uses_full_graph=True))
    result=dict(status='QUALIFIED_BOUNDED',actual_model=True,logical_B=entry['pack']['n_requests'],
        production_builder='whole_B_all_RW_KL',dense_reference=True,microbatch_permutation=True,
        native_full_block_hook_reference=True,new_fit_candidates=0,new_optimizer_updates=0,
        comparison_candidates=5,main_PASS=False,tolerances=dict(forward_atol=1e-5,forward_rtol=1e-4,gradient_RMS_atol=1e-6,gradient_RMS_rtol=1e-3))
    write(out/'receipt.json',result);return result


def shape_check(a,bench,records,history,config,out):
    entry=prepare_entry(a,bench,bench.prepare(records),history,config['stats'],config['settings']['fit_microbatch'])
    result=one(a,entry)
    write(out/'receipt.json',dict(actual_B=len(records),fixed_candidate=True,fit_candidates=0,updates=0,
        loss=result['loss'],gradient_norm={str(l):float(g.norm()) for l,g in result['gradient'].items()},
        solve=result['residual'],actual_model=True,commit=False))


@torch.no_grad()
def actual_commit_probe(a,entry,payload,out):
    from .subject import row_logprobs
    from .geometry import mean_keys
    raw={l:[] for l in a.sites};rows=[];measured=torch.zeros_like(payload['context_nll'])
    for group in entry['groups']:
        captured={};handles=[]
        for l in a.sites:
            handles.append(a.blocks[l].mlp.down_proj.register_forward_pre_hook(lambda m,args,l=l:captured.update({l:args[0]})))
        try:
            nh,fh=a.full({k:v.to(a.device) for k,v in group['tokens'].items()})
            probs=row_logprobs(a,group['rows'],nh,fh)
            for j,(r,lp) in enumerate(zip(group['rows'],probs)):
                if r['kind']!='rewrite':continue
                rows.append(r);targets=r['target'][r['target']!=-100].to(a.device)
                measured[r['request'],r['global_row']%(entry['pack']['n_rw']+1)]=float(-lp.gather(1,targets[:,None]).mean())
                for l in a.sites:raw[l].append(captured[l][j,r['lookup']].cpu())
        finally:
            for h in handles:h.remove()
    evidence={};passed=True
    for l,parts in raw.items():
        k=mean_keys(torch.stack(parts).T,rows,entry['pack']);ref=payload['keys'][l];err=(k-ref).abs()
        ok=bool((err<=1e-5+1e-4*ref.abs()).all());passed &= ok
        evidence[str(l)]=dict(max_error=float(err.max()),passed=ok)
    error=(measured-payload['context_nll']).abs();ok=bool((error<=1e-5+1e-4*payload['context_nll'].abs()).all())
    write(out/'actual-commit-probe.json',dict(keys=evidence,nll_error_max=float(error.max()),passed=bool(passed and ok)))
    require(passed and ok,'SAME_EVALUATED_COMMIT_MODEL_PARITY')
