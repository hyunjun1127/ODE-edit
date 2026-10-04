"""Branch-local ascending write and explicit final rewrite-only history."""
import time
import torch
from project.run_scripts.jlz_writer_coupled.geometry import build_prior_from_npz
from project.run_scripts.jlz_realization.geometry import mean_keys
from .common import require,write,tensor_sha,digest
from .capture import capture_native_sites
from .geometry import prepare_metric,solve_realized,targets,owner_weights,action_parity
from .telemetry import vector_metrics,shares

@torch.no_grad()
def apply_sequential(a,entry,plan,virtual,initial,H,branch,out):
    start=time.monotonic();B=entry['pack']['n_requests'];stored={};summaries={};calls=0
    guard=a.guard();before_H={l:tensor_sha(h) for l,h in H.items()}
    for l in a.sites:
        layerstart=time.monotonic();pre=capture_native_sites(a,entry,(l,));calls+=pre['calls']
        rows=pre['rows'];owners=torch.tensor([r['request'] for r in rows],device=a.device)
        canonical=[j for j,r in enumerate(rows) if r['global_row'] in entry['pack']['canonical_rows']]
        require(len(canonical)==B,'CANONICAL_COUNT')
        raw=pre['keys'][l].to(a.device).double();mean=pre['mean'][l].to(a.device).double()
        D=plan['D'][l];z=plan['z'][l].to(a.device);h=pre['hidden'][l][:,canonical].to(a.device)
        # Preserve native FP32 residual subtraction, then FP64 writer product.
        T=targets(branch,D,z,h,owners).double()
        K=raw if branch=='CD' else mean
        weights=owner_weights(owners,B) if branch=='CD' else torch.full((B,),1/B,dtype=torch.float64,device=a.device)
        rawA=build_prior_from_npz(entry['stats'][str(l)],H[l],lambda_c=15000.,device=a.device)
        factorstart=time.monotonic();A,L,metric=prepare_metric(rawA);del rawA
        metric['factor_seconds']=time.monotonic()-factorstart
        U,diag=solve_realized(A,L,K,T,weights,'ridge' if branch in ('RT','RD') else 'equality')
        diag.update(metric=metric,branch=branch,layer=l,key_hash=tensor_sha(K),target_hash=tensor_sha(T),
            weight_hash=tensor_sha(weights),row_order=digest([(r['global_row'],r['request'],r['kind'],r['lookup']) for r in rows]),
            original_columns_retained=True,branch_fresh_upper_capture=True)
        write(out/f'layer-{l}-solve.json',diag)
        require(diag['numerical_projection_verified'],'NUMERICAL_REALIZATION_FAILURE')
        old=a.weights[l].detach().clone();new=old+U.float()
        require(bool(torch.isfinite(new).all()),'MATERIALIZED_NONFINITE')
        effective=new.double()-old.double();cast=U.float().double()
        ideal_action=U@raw;cast_action=cast@raw;effective_action=effective@raw
        parity=action_parity(effective_action,ideal_action)
        # P1 requires effective Q even for soft-ridge branches.
        energies=dict(ideal_Q=diag['ideal_Q'],effective_Q=float((effective@L).square().sum()),
            cast_update_norm=float(cast.norm()),effective_update_norm=float(effective.norm()))
        write(out/f'layer-{l}-materialization.json',dict(**energies,ideal_effective_parity=parity))
        require(parity['pass_'],'FP32_IDEAL_EFFECTIVE_PARITY')
        a.weights[l].copy_(new);require(tensor_sha(a.weights[l])==tensor_sha(new),'EXACT_EVALUATED_COPY')
        ownD=D[:,owners];anchor=entry['anchors'][l][owners]
        action_rows={name:vector_metrics(action,ownD,anchor) for name,action in
            [('ideal',ideal_action),('cast',cast_action),('effective',effective_action)]}
        mean_rows={name:vector_metrics(update@mean,D,entry['anchors'][l]) for name,update in [('ideal',U),('cast',cast),('effective',effective)]}
        stored[l]=dict(pre=pre['hidden'][l],effective=effective_action.cpu(),rows=rows,
            actions=action_rows,mean_actions=mean_rows,effective_mean=(effective@mean).cpu(),
            target=T.cpu(),canonical=canonical)
        summaries[l]=dict(**energies,solver=diag,ideal_effective_parity=parity,
            seconds=time.monotonic()-layerstart,weight_after=tensor_sha(a.weights[l]))
        del U,A,L,K,T,raw,mean,D,z,h,new,old,effective,cast,ideal_action,cast_action,effective_action,pre
    final=capture_native_sites(a,entry,a.sites);calls+=final['calls']
    canonical=stored[a.first]['canonical'];rowmeta=stored[a.first]['rows']
    allrows=[];localchecks={};direct_canonical={};direct_mean={};direct_context={}
    for l in a.sites:
        saved=stored[l];local=final['hidden'][l].double()-saved['pre'].double()
        parity=action_parity(local,saved['effective']);localchecks[l]=parity
        owners=torch.tensor([r['request'] for r in rowmeta]);D=plan['D'][l].cpu().double()[:,owners]
        virtualpost=virtual[l].double();virtualpre=virtualpost-D
        inherited=saved['pre'].double()-virtualpre
        directerror=local-D;finalgap=final['hidden'][l].double()-virtualpost
        net=final['hidden'][l].double()-initial['hidden'][l].double()
        require(bool(torch.allclose(inherited+directerror,finalgap,atol=1e-12,rtol=1e-12)),'VECTOR_GAP_IDENTITY')
        actual=vector_metrics(local,D,entry['anchors'][l].cpu()[owners])
        rw_indices=[j for j,r in enumerate(rowmeta) if r['kind']=='rewrite']
        rw_rows=[rowmeta[j] for j in rw_indices]
        localmean=mean_keys(local[:,rw_indices].float(),rw_rows,entry['pack'])
        saved['mean_actions']['actual']=vector_metrics(localmean,plan['D'][l].cpu(),entry['anchors'][l].cpu())
        gapmetrics=vector_metrics(net,D,entry['anchors'][l].cpu()[owners])
        for j,row in enumerate(rowmeta):
            allrows.append(dict(branch=branch,layer=l,owner=row['request'],case_id=entry['pack']['record_ids'][row['request']],
                global_row=row['global_row'],kind=row['kind'],canonical=j in canonical,
                ideal=saved['actions']['ideal'][j],cast=saved['actions']['cast'][j],effective=saved['actions']['effective'][j],
                actual=actual[j],net_entry_change=gapmetrics[j],inherited_gap_norm=float(inherited[:,j].norm()),
                direct_error_norm=float(directerror[:,j].norm()),final_virtual_gap_norm=float(finalgap[:,j].norm()),
                inherited_direct_error_dot=float((inherited[:,j]*directerror[:,j]).sum())))
        direct_canonical[l]=saved['effective'][:,canonical];direct_mean[l]=saved['effective_mean'];direct_context[l]=saved['effective']
    write(out/'local-additivity.json',dict(checks=localchecks,all_native_rows=True,no_delta_hook=True))
    write(out/'actions.json',dict(rows=allrows,mean={l:s['mean_actions'] for l,s in stored.items()}))
    write(out/'shares.json',dict(currency='effective_FP32_direct_action_over_entry_anchor',
        canonical=shares({l:v.cpu() for l,v in plan['u'].items()},direct_canonical,entry['anchors'],a.sites),
        mean=shares({l:v.cpu() for l,v in plan['u'].items()},direct_mean,entry['anchors'],a.sites),
        contexts=shares({l:v.cpu()[:,owners] for l,v in plan['u'].items()},direct_context,
            {l:v.cpu()[owners] for l,v in entry['anchors'].items()},a.sites),
        context_owners=owners.tolist(),context_roles=[r['kind'] for r in rowmeta]))
    require(all(c['pass_'] for c in localchecks.values()),'ACTUAL_LOCAL_ADDITIVITY_PARITY')
    require(a.guard()==guard,'NONSELECTED_MUTATION')
    history=[]
    for l in a.sites:
        k=final['mean'][l];require(k.dtype==torch.float32 and k.device.type=='cpu' and k.shape[1]==B,'FINAL_NATIVE_KEYS')
        require(tensor_sha(H[l])==before_H[l],'PREMATURE_HISTORY')
        native=k.T.contiguous().T;H[l].add_(native@native.T)
        require(bool(torch.isfinite(H[l]).all()),'HISTORY_NONFINITE')
        history.append(dict(layer=l,append_count=1,columns=B,key_sha256=tensor_sha(k),before=before_H[l],after=tensor_sha(H[l]),
                            rewrite_only=True,KL_in_history=False,CPU_FP32=True))
    result=dict(branch=branch,layers=summaries,history=history,history_appends=len(history),
        capture_forward_calls=calls,seconds=time.monotonic()-start,checkpoint_saved=False)
    write(out/'writer.json',result);return result
