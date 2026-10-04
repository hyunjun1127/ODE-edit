"""Actual target tracking, fresh K/h per layer, native final-key history once."""
import time
import torch
from project.run_scripts.jlz_realization.geometry import mean_keys
from project.run_scripts.jlz_realization.profile import move
from project.run_scripts.jlz_writer_coupled.geometry import build_prior_from_npz
from .common import require,tensor_sha,digest,write
from .telemetry import ratio,cosine

@torch.no_grad()
def capture(a,entry,layers):
    keys={l:[] for l in layers};hidden={l:[] for l in layers};ordered=[]
    for group in entry['groups']:
        k={};h={};handles=[]
        for l in layers:
            handles.append(a.blocks[l].mlp.down_proj.register_forward_pre_hook(lambda m,args,l=l:k.update({l:args[0]})))
            handles.append(a.blocks[l].register_forward_hook(lambda m,args,out,l=l:h.update({l:out})))
        try:
            a.full(move(group['tokens'],a.device))
            for j,row in enumerate(group['rows']):
                if row['kind']!='rewrite':continue
                ordered.append(row)
                for l in layers:
                    keys[l].append(k[l][j,row['lookup']].detach().cpu().clone())
                    hidden[l].append(h[l][j,row['lookup']].detach().cpu().clone())
        finally:
            for handle in handles:handle.remove()
    raw={l:torch.stack(v).T for l,v in keys.items()}
    return dict(keys=raw,hidden={l:torch.stack(v).T for l,v in hidden.items()},rows=ordered,
                mean={l:mean_keys(raw[l],ordered,entry['pack']) for l in layers})

@torch.no_grad()
def solve(stats,H,K,device):
    # Native full system, no symmetry repair/jitter/approximate solver.
    A=build_prior_from_npz(stats,H,lambda_c=15000.,device=device)
    K=K.to(device).double();system=A+K@K.T
    P=torch.linalg.solve(system,K)
    residual=float((system@P-K).norm()/K.norm()) if bool(K.norm()>0) else float((system@P-K).norm())
    require(bool(torch.isfinite(P).all()) and residual<=1e-8,'NATIVE_SOLVE_RESIDUAL:'+str(residual))
    return P,residual

@torch.no_grad()
def apply(a,entry,plan,H,events,out):
    start=time.monotonic();B=entry['pack']['n_requests'];ids=entry['pack']['record_ids'];nctx=entry['pack']['n_rw']
    stored={};real_relative={};summaries={};guard=a.guard();h_before={l:tensor_sha(h) for l,h in H.items()}
    for l in a.sites:
        prewrite_id=digest({j:tensor_sha(w) for j,w in a.weights.items()})
        before=capture(a,entry,(l,));K=before['mean'][l];raw=before['keys'][l].to(a.device).double()
        canon=[i for i,r in enumerate(before['rows']) if r['global_row'] in entry['pack']['canonical_rows']]
        require(len(canon)==B,'CANONICAL_COVERAGE')
        hcur=before['hidden'][l][:,canon];R=plan['z'][l].to(a.device)-hcur.to(a.device)
        P,residual=solve(entry['stats'][str(l)],H[l],K,a.device)
        U=R.double()@P.T;old=entry['entry_weights'][l]
        W=old+U.float();expected=tensor_sha(W)
        require(bool(torch.isfinite(W).all()),'WRITE_NONFINITE')
        a.weights[l].copy_(W);require(tensor_sha(a.weights[l])==expected,'WRITE_EXACT_COPY')
        applied=W.double()-old.double();ideal=U@raw;effective=applied@raw
        idealmean=U@K.to(a.device).double();effectivemean=applied@K.to(a.device).double()
        own=plan['D'][l];inh=R-own;rows=[]
        for r in range(B):
            ci=canon[r];planned=own[:,r];res=R[:,r];inherit=inh[:,r]
            indices=[j for j,row in enumerate(before['rows']) if row['request']==r]
            actions=[]
            for j in indices:
                context=before['rows'][j]['global_row']%(nctx+1)
                gi=next(i for i,(lo,hi) in enumerate(entry['pack']['context_group_slices']) if lo<=context<hi)
                lo,hi=entry['pack']['context_group_slices'][gi]
                actions.append(dict(context_id=str(before['rows'][j]['global_row']),native_group_id=str(gi),
                    is_canonical=j==ci,native_key_mean_weight=1/len(entry['pack']['context_group_slices'])/(hi-lo),
                    ideal_direct_l2=float(ideal[:,j].norm()),effective_direct_l2=float(effective[:,j].norm()),
                    effective_direct_to_plan_cosine=cosine(effective[:,j],planned),
                    effective_direct_to_residual_cosine=cosine(effective[:,j],res)))
            rows.append(dict(accepted_plan_candidate=plan['terminal'][r]['candidate'],entry_anchor_l2=float(entry['anchors'][l][r]),
                plan_delta_l2=float(planned.norm()),plan_relative_l2=float(plan['u'][l][:,r].norm()),
                residual_l2=float(res.norm()),inherited_mismatch_l2=float(inherit.norm()),
                residual_decomposition_error_l2=float((res.double()-planned.double()-inherit.double()).norm()),
                plan_inherited_cosine=cosine(planned,inherit),virtual_target_canonical_l2=float(plan['z'][l][:,r].norm()),
                actual_prewrite_canonical_l2=float(hcur[:,r].norm()),
                ideal_direct_native_mean_l2=float(idealmean[:,r].norm()),effective_direct_native_mean_l2=float(effectivemean[:,r].norm()),
                ideal_direct_canonical_l2=float(ideal[:,ci].norm()),effective_direct_canonical_l2=float(effective[:,ci].norm()),
                effective_direct_canonical_relative_l2=float(effective[:,ci].norm()/entry['anchors'][l][r]),
                realized_share_currency='EFFECTIVE_CANONICAL_DIRECT_L2_OVER_ENTRY_ANCHOR',
                final_capture_source='FINAL_HISTORY_KEY_FORWARD',context_direct_actions=actions))
        real_relative[l]=[r['effective_direct_canonical_relative_l2'] for r in rows]
        stored[l]=dict(rows=rows,hcur=hcur,effective=effective[:,canon].cpu())
        summaries[l]=dict(actual_request_count=B,native_mean_key_columns=B,
            prewrite_model_state_id=prewrite_id,postwrite_model_state_id=digest({j:tensor_sha(w) for j,w in a.weights.items()}),key_hcur_same_state=True,
            refreshed_after_lower_writes=True,solve_kind='NATIVE_RIDGE_FULL_BATCH',solve_relative_residual=residual,
            ideal_update_frobenius=float(U.norm()),effective_update_frobenius=float(applied.norm()),
            ideal_effective_update_relative_difference=ratio(float((applied-U).norm()),float(U.norm())),
            native_orientation_cast_add_verified=True,plan_support_used_to_skip_writer=False)
        events.emit('write_layer_summary',summaries[l],layer=l)
        del U,W,P,applied,ideal,effective,raw,R,own,inh,idealmean,effectivemean,before
    final=capture(a,entry,a.sites);final_id=digest({l:tensor_sha(w) for l,w in a.weights.items()})
    require(a.guard()==guard,'NONSELECTED_CHANGED')
    canon=[i for i,r in enumerate(final['rows']) if r['global_row'] in entry['pack']['canonical_rows']]
    for l in a.sites:
        for r,row in enumerate(stored[l]['rows']):
            actual=final['hidden'][l][:,canon[r]];local=actual.double()-stored[l]['hcur'][:,r].double()
            row.update(actual_final_canonical_l2=float(actual.norm()),actual_local_canonical_change_l2=float(local.norm()),
                actual_net_canonical_drift_l2=float((actual.double()-entry['entry_hidden'][l][:,r].double()).norm()),
                local_additivity_error_l2=float((local-stored[l]['effective'][:,r]).norm()),
                effective_direct_realized_share=ratio(real_relative[l][r],sum(v[r] for v in real_relative.values()),'ZERO_REALIZED_TOTAL'))
            events.emit('write_layer_request',row,ids[r],plan['terminal'][r]['candidate'],l)
        k=final['mean'][l];require(k.device.type=='cpu' and k.dtype==torch.float32,'HISTORY_DTYPE')
        require(tensor_sha(H[l])==h_before[l],'PREMATURE_HISTORY')
        native_k=k.T.contiguous().T;H[l].add_(native_k@native_k.T)
        require(bool(torch.isfinite(H[l]).all()),'HISTORY_NONFINITE')
        events.emit('history_commit',dict(append_count=1,request_count=B,final_key_columns=B,final_keys_identity=tensor_sha(k),
            final_model_state_id=final_id,history_before_id=h_before[l],history_after_id=tensor_sha(H[l]),
            history_dtype='cpu_float32',all_original_request_columns_retained=True,successful_requests_only_filter=False),layer=l)
    result=dict(history_appends=len(a.sites),seconds=time.monotonic()-start,
        capture_forward_calls=(len(a.sites)+1)*len(entry['groups']),layers=summaries,
        target_tracking=True,no_divisor=True,checkpoint_saved=False,final_keys={l:tensor_sha(k) for l,k in final['mean'].items()})
    write(out/'writer.json',result);return result
