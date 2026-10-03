"""Bounded zero/nonzero full-reference and native-key alignment on dev BS2."""
import torch
from project.run_scripts.jlz_realization.qualification import fixed, singleton_permuted
from .subject import native
from .geometry import allocation
from .common import require,write

def gradients(a,b):
    rows={}
    for l in a:
        ref=b[l].double();diff=a[l].double()-ref;n=float(ref.norm())
        error=float(diff.abs().max()) if n<=1e-8 else float(diff.norm()/ref.norm())
        limit=1e-6 if n<=1e-8 else 2e-3
        rows[l]=dict(error=error,limit=limit,reference_norm=n,pass_=error<=limit)
    return rows

def qualify(adapter,entry,frozen,out):
    receipts=[];B=entry['pack']['n_requests']
    for index,scale in enumerate((0.,.025)):
        results=[]
        for route in ('cached','full'):
            adapter.native_route=route;D=fixed(adapter,entry,scale)
            value=native(adapter,entry,D,True)
            alloc,_=allocation(D,frozen,.1);(B*alloc).backward()
            results.append(dict(mean=value['mean']+float(alloc.detach()),gradient={l:d.grad.clone() for l,d in D.items()}))
        left,right=results;err=abs(left['mean']-right['mean']);limit=2e-5+2e-4*abs(right['mean'])
        check=gradients(left['gradient'],right['gradient'])
        receipt=dict(scale=scale,loss_error=err,loss_limit=limit,gradient=check,reference='full_model_hooks',
                     candidate_fit=0,optimizer_updates=0)
        write(out/f'full-reference-{index}.json',receipt)
        require(err<=limit and all(x['pass_'] for x in check.values()),'ACTUAL_NATIVE_REFERENCE')
        receipts.append(receipt)
    adapter.native_route='cached'
    D=fixed(adapter,entry,.025);normal=native(adapter,entry,D,True);g={l:d.grad.clone() for l,d in D.items()}
    D=fixed(adapter,entry,.025);permuted=native(adapter,singleton_permuted(entry),D,True)
    check=gradients({l:d.grad for l,d in D.items()},g)
    error=abs(normal['mean']-permuted['mean']);limit=2e-5+2e-4*abs(normal['mean'])
    write(out/'microbatch-reference.json',dict(loss_error=error,limit=limit,gradients=check,logical_B=B,MB1_reverse=True))
    require(error<=limit and all(x['pass_'] for x in check.values()),'MICROBATCH_GRADIENT')
    write(out/'ready.json',dict(status='QUALIFIED_BOUNDED_BS2',source='actual model',no_main_PASS=True,
          fixed_candidates=6,fit_candidates=0,backward_comparisons=6))

@torch.no_grad()
def native_keys(adapter,bench,entry,bound,out):
    module,hp,_,_=bound
    measured={}
    for l in adapter.sites:
        ref=module.compute_ks(adapter.model,bench.tokenizer,entry['pack']['requests'],hp,l,bench.contexts).T
        own=entry['mean_keys'][l].to(ref.device)
        error=(own-ref).abs();limit=2e-5+2e-4*ref.abs()
        measured[l]=dict(max_error=float(error.max()),limit_excess=float((error-limit).max()),shape=list(ref.shape),
                         source='BLUE compute_ks original function; same tokens/context/lookup')
    write(out/'native-key-reference.json',measured)
    require(all(r['limit_excess']<=0 for r in measured.values()),'NATIVE_KEY_REFERENCE')
