"""Scalar realization diagnostics only; no reconstructible payload is persisted."""
import time
import torch
import torch.nn.functional as F
from project.run_scripts.jlz_realized_writer.telemetry import vector_metrics,shares
from project.run_scripts.jlz_realized_subject.geometry import mean_keys
from project.run_scripts.jlz_realized_subject.subject import row_logprobs
from .common import require,write,state,tensor_sha

def candidate_summary(a,entry,R,built):
    out={}
    for l in a.sites:
        req=torch.tensor([r['request'] for r in built['rows']])
        target=R[l].detach().cpu()[:,req];action=built['v'][l].T
        metrics=vector_metrics(action,target,entry['anchors'][l].cpu()[req])
        defined=[m for m in metrics if m['directional_ratio'] is not None]
        out[str(l)]=dict(requested_norm=float(R[l].norm()),actual_context_norm=float(action.norm()),
            directional_ratio_mean=sum(m['directional_ratio'] for m in defined)/len(defined) if defined else None,
            error_RMS=float((action.double()-target.double()).square().mean().sqrt()),
            zero_owners=sum(float(R[l][:,r].norm())==0 for r in range(R[l].shape[1])),
            actual_RHS_identity=tensor_sha(R[l]),rows=len(metrics))
    return out

@torch.no_grad()
def commit_measure(a,entry,H,plan,out):
    start=time.monotonic();built=plan['built'];R=plan['R'];B=entry['pack']['n_requests'];n=entry['pack']['n_rw']
    before=state(a,H);weight_hash={}
    for l,w in a.weights.items():
        weight_hash[l]=tensor_sha(built['weights'][l])
        w.copy_(built['weights'][l]);require(torch.equal(w,built['weights'][l]),'EXACT_EVALUATED_COMMIT')
    raw={l:[] for l in a.sites};hidden={l:[] for l in a.sites};rows=[]
    nll=torch.zeros(B,n,dtype=torch.float64);kl=torch.zeros(B,dtype=torch.float64)
    for group in entry['groups']:
        captured={};hs={};handles=[]
        for l in a.sites:
            handles.append(a.blocks[l].mlp.down_proj.register_forward_pre_hook(lambda m,args,l=l:captured.update({l:args[0]})))
            handles.append(a.blocks[l].register_forward_hook(lambda m,args,value,l=l:hs.update({l:a.unwrap(value)})))
        try:
            nh,fh=a.full({k:v.to(a.device) for k,v in group['tokens'].items()})
            probs=row_logprobs(a,group['rows'],nh,fh)
            for j,(r,lp) in enumerate(zip(group['rows'],probs)):
                rows.append(r);req=r['request'];c=r['global_row']%(n+1)
                if r['kind']=='rewrite':
                    labels=r['target'][r['target']!=-100].to(a.device)
                    nll[req,c]=float(-lp.gather(1,labels[:,None]).mean())
                else:kl[req]=float((lp.exp()*(lp-entry['teachers'][req].to(a.device))).sum())
                for l in a.sites:
                    raw[l].append(captured[l][j,r['lookup']].cpu().clone())
                    hidden[l].append(hs[l][j,r['lookup']].cpu().clone())
        finally:
            for h in handles:h.remove()
    rw=[i for i,r in enumerate(rows) if r['kind']=='rewrite'];rwrows=[rows[i] for i in rw]
    layers={};action_rows=[];means={};canonical_actions={};history=[]
    for l in a.sites:
        keys=torch.stack(raw[l]).T;reference=built['raw'][l].T
        err=(keys-reference).abs();limit=1e-5+1e-4*reference.abs()
        require(bool((err<=limit).all()),'TERMINAL_NATIVE_KEY_PARITY')
        K=mean_keys(keys[:,rw],rwrows,entry['pack'])
        owners=torch.tensor([r['request'] for r in rows]);target=R[l].cpu()[:,owners]
        key=keys.to(a.device);w0=entry['entry_weights'][l];w=built['weights'][l]
        actual=(F.linear(key.T,w)-F.linear(key.T,w0)).T.cpu()
        U=R[l].double()@built['P'][l].T
        effective=w.double()-w0.double()
        ideal=(U@key.double()).cpu();eff=(effective@key.double()).cpu()
        local_error=(actual.double()-eff).abs()
        require(bool((local_error<=1e-5+1e-4*eff.abs()).all()),'ACTUAL_LOCAL_ACTION_PARITY')
        ideal_error=(ideal-eff).abs()
        require(bool((ideal_error<=1e-5+1e-4*ideal.abs()).all()),'NATIVE_CAST_ACTION_PARITY')
        anchor=entry['anchors'][l].cpu()[owners]
        am=vector_metrics(actual,target,anchor);im=vector_metrics(ideal,target,anchor);em=vector_metrics(eff,target,anchor)
        for i,row in enumerate(rows):
            action_rows.append(dict(layer=l,owner=row['request'],row=row['global_row'],kind=row['kind'],
                canonical=row['global_row'] in entry['pack']['canonical_rows'],actual=am[i],ideal=im[i],effective=em[i]))
        mean_actual=(F.linear(K.T.to(a.device),w)-F.linear(K.T.to(a.device),w0)).T.cpu()
        means[str(l)]=vector_metrics(mean_actual,R[l].cpu(),entry['anchors'][l].cpu())
        canonical_actions[l]=actual[:,entry['pack']['canonical_rows']]
        A=entry['factors'][l]['A'].to(a.device);AP=A@built['P'][l]
        ideal_Q=float(((R[l].double().T@R[l].double())*(built['P'][l].T@AP).T).sum())
        effective_Q=0.
        for j in range(0,effective.shape[0],128):
            e=effective[j:j+128];effective_Q+=float(((e@A)*e).sum())
        gap=plan['observed']['masked_bases'][l]-built['prebase'][l]
        layers[str(l)]=dict(ideal_Q=ideal_Q,effective_Q=effective_Q,ideal_update_norm=float(U.norm()),
            cast_update_norm=float(U.float().norm()),effective_update_norm=float(effective.norm()),
            native_key_max_error=float(err.max()),cast_action_max_error=float(ideal_error.max()),
            local_action_max_error=float(local_error.max()),masked_actual_prestate_gap_RMS=float(gap.double().square().mean().sqrt()),
            weight_sha=weight_hash[l],actual_RHS_sha=tensor_sha(R[l]),solve=built['metadata'][l])
        # Final-model rewrite-only native CPU FP32 history, exactly once.
        require(K.dtype==torch.float32 and H[l].device.type=='cpu','HISTORY_DTYPE')
        old=tensor_sha(H[l]);H[l].add_(K@K.T)
        require(bool(torch.isfinite(H[l]).all()),'NONFINITE_HISTORY')
        history.append(dict(layer=l,append_count=1,columns=B,rewrite_only=True,key_sha=tensor_sha(K),before=old,after=tensor_sha(H[l])))
        del U,effective,A,AP,key
    write(out/'actions.json',dict(rows=action_rows,mean=means))
    write(out/'shares.json',dict(rows=shares({l:x.cpu() for l,x in plan['u'].items()},canonical_actions,entry['anchors'],a.sites)))
    receipt=dict(before=before,after=state(a,H),history=history,history_appends=len(history),layers=layers,
        accepted_weight_copy_exact=True,terminal_candidate=built['candidate'],actual_native_nll=nll.tolist(),
        actual_native_KL=kl.tolist(),masked_native_nll=plan['observed']['nll'].tolist(),
        masked_native_KL=plan['observed']['kl'].tolist(),seconds=time.monotonic()-start,postfit_commits=1,
        checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    write(out/'writer.json',receipt)
    return receipt
