"""Terminal fixed-D increments, actual upper-key refresh, final CPU32 H once."""
import time
import torch
from project.run_scripts.jlz_realization.geometry import mean_keys, ridge
from project.run_scripts.jlz_realization.profile import move
from project.run_scripts.jlz_realization.subject import row_logprobs
from project.run_scripts.jlz_realization.writer import Transaction, rng_snapshot, rng_equal
from .common import require, tensor_sha, write

@torch.no_grad()
def capture(adapter, entry, layers,hidden_out=None):
    raw={l:[] for l in layers}; ordered=[]; means={}; nll=0.;kl=0.
    blocks={l:[] for l in layers}
    for group in entry['groups']:
        found={};hidden={};handles=[]
        for l in layers:
            handles.append(adapter.blocks[l].mlp.down_proj.register_forward_pre_hook(
                lambda m,args,l=l:found.update({l:args[0]})))
            if hidden_out is not None:
                handles.append(adapter.blocks[l].register_forward_hook(lambda m,args,out,l=l:hidden.update({l:out})))
        try:
            nh,fh=adapter.full(move(group['tokens'],adapter.device))
            for j,(r,lp) in enumerate(zip(group['rows'],row_logprobs(adapter,group['rows'],nh,fh))):
                if r['kind']=='rewrite':
                    ordered.append(r)
                    for l in layers:raw[l].append(found[l][j,r['lookup']].float().cpu().clone())
                    if hidden_out is not None:
                        for l in layers:blocks[l].append(hidden[l][j,r['lookup']].float().cpu().clone())
                    target=r['target'][r['target']!=-100].to(adapter.device)
                    nll+=float(-lp.gather(1,target[:,None]).mean())/entry['pack']['n_rw']
                else:
                    teacher=entry['teachers'][r['request']].to(adapter.device)
                    kl+=float((lp.exp()*(lp-teacher)).sum())
        finally:
            for h in handles:h.remove()
    for l in layers:means[l]=mean_keys(torch.stack(raw[l]).T,ordered,entry['pack'])
    if hidden_out is not None:
        hidden_out.update({l:dict(hidden=torch.stack(blocks[l]).T,keys=torch.stack(raw[l]).T) for l in layers})
    return means,dict(nll_mean=nll/entry['pack']['n_requests'],KL_mean=kl/entry['pack']['n_requests'],
                      forward_groups=len(entry['groups']),subject_delta_hook=False)

@torch.no_grad()
def apply(adapter,entry,D,history,out):
    start=time.monotonic();receipt={};evaluated={};B=entry['pack']['n_requests']
    virtual={l:[] for l in adapter.sites}
    for group in entry['groups']:
        _,_,h=adapter.native(group,D,True)
        for j,r in enumerate(group['rows']):
            if r['kind']=='rewrite':
                for l in adapter.sites:virtual[l].append(h[l][j].cpu().clone())
    virtual={l:torch.stack(v).T.double() for l,v in virtual.items()}
    for l in adapter.sites:
        before={}
        keys,_=capture(adapter,entry,(l,),before) # actual model already contains lower writes
        K=keys[l].to(adapter.device);geo=ridge(K,entry['factors'][l])
        U=D[l].double()@geo['P'].T
        W=entry['entry_weights'][l]+U.float()
        require(torch.isfinite(W).all(),'NONFINITE_INCREMENT')
        expected=tensor_sha(W);adapter.weights[l].copy_(W)
        require(tensor_sha(adapter.weights[l])==expected,'COMMIT_WEIGHT_IDENTITY')
        actual_delta=(W.double()-entry['entry_weights'][l].double())@K.double()
        planned=D[l].double();difference=actual_delta-planned
        norms=planned.norm(dim=0);realnorm=actual_delta.norm(dim=0)
        cosine=(planned*actual_delta).sum(0)/norms.square().clamp_min(1e-300)
        direct=(W.double()-entry['entry_weights'][l].double())@before[l]['keys'].to(adapter.device).double()
        direct=direct.cpu();inherited=before[l]['hidden'].double()-entry['entry_hidden'][l].double()
        actual=before[l]['hidden'].double()+direct;virtual_gap=actual-virtual[l]
        receipt[l]=dict(source='actual_lower_write',key_sha=tensor_sha(K),entry_key_sha=tensor_sha(entry['mean_keys'][l]),
            key_drift_norm=float((K.cpu()-entry['mean_keys'][l]).norm()),solve=geo['metadata'],
            D_norm=norms.tolist(),U_norm=float(U.norm()),actual_write_norm=float((W-entry['entry_weights'][l]).norm()),
            gamma=[None if float(n)==0 else float(g) for n,g in zip(norms,cosine)],realized_norm=realnorm.tolist(),
            realization_error_norm=difference.norm(dim=0).tolist(),
            actual_G_cost=float(((planned.T@planned)*geo['G'].T).sum()),
            actual_E_cost=float((planned@(geo['M']-torch.eye(B,device=adapter.device))).square().sum()),
            context_direct_norm=direct.norm(dim=0).tolist(),context_inherited_norm=inherited.norm(dim=0).tolist(),
            context_direct_inherited_dot=(direct*inherited).sum(0).tolist(),
            context_actual_virtual_gap=virtual_gap.norm(dim=0).tolist(),
            exact_committed_SHA=expected,no_divisor=True,no_absolute_target=True)
        evaluated[l]=expected
        del U,W,geo,K,actual_delta,planned,difference
    final_keys,observation=capture(adapter,entry,adapter.sites)
    require({l:tensor_sha(w) for l,w in adapter.weights.items()}==evaluated,'POST_COMMIT_OBSERVER_CHANGED_W')
    for l,k in final_keys.items():
        require(k.dtype==torch.float32 and k.device.type=='cpu' and torch.isfinite(k).all(),'HISTORY_KEYS')
        native_k=k.T.contiguous().T # original native stack(requests).T storage/order
        history[l].add_(native_k@native_k.T)
        require(torch.isfinite(history[l]).all(),'NONFINITE_HISTORY')
    result=dict(layers=receipt,history_appends=len(adapter.sites),final_key_hash={l:tensor_sha(k) for l,k in final_keys.items()},
        post_commit_native=observation,seconds=time.monotonic()-start,checkpoint_saved=False,
        terminal_virtual_diagnostic_forward_groups=len(entry['groups']),actual_writer_forward_groups=6*len(entry['groups']),
        exact_resume='NOT_AVAILABLE',actual_upper_keys_refreshed=True,terminal_D_fixed=True)
    write(out/'writer.json',result)
    return result
