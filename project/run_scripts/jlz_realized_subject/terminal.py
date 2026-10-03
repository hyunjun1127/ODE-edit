"""All-token observation only; no pulse/loss gradient or writer reconstruction."""
import time
import torch
import torch.nn.functional as F
from .subject import row_logprobs
from .geometry import mean_keys
from .common import require,tensor_sha
from .telemetry import effective_energy


@torch.no_grad()
def observe(a,entry,R,built):
    start=time.monotonic();B=entry['pack']['n_requests'];n=entry['pack']['n_rw']
    nll=torch.zeros(B,n,dtype=torch.float64);kl=torch.zeros(B,dtype=torch.float64);targetkl=torch.zeros_like(nll)
    key_rows=[];raw={l:[] for l in a.sites};details=[];tokens=0
    for group in entry['groups']:
        nh,fh,actual_h,actual_k=a.actual(group,R,built['P'],built['weights'],capture=True)
        sn,sf,fit_k,fit_pre=a.masked(group,built['v'],capture=True)
        probs=row_logprobs(a,group['rows'],nh,fh);sprobs=row_logprobs(a,group['rows'],sn,sf)
        for j,(r,lp,slp) in enumerate(zip(group['rows'],probs,sprobs)):
            req=r['request'];c=r['global_row']%(n+1);idx=r['global_row'];tokens+=len(lp)*2
            if r['kind']=='rewrite':
                labels=r['target'][r['target']!=-100].to(a.device)
                nll[req,c]=float(-lp.gather(1,labels[:,None]).mean())
                targetkl[req,c]=float((slp.exp()*(slp-lp)).sum(-1).mean())
                key_rows.append(r)
                for l in a.sites:raw[l].append(actual_k[l][j].cpu())
            else:kl[req]=float((lp.exp()*(lp-entry['teachers'][req].to(a.device))).sum())
            row=dict(request=req,context=c,kind=r['kind'],layer={})
            for l in a.sites:
                ka=actual_k[l][j];ks=fit_k[l][j];va=built['v'][l][idx]
                vs=F.linear(ks,built['weights'][l])-F.linear(ks,entry['entry_weights'][l])
                base=built['prebase'][l][idx]-fit_pre[l][j]
                dk=ka-ks;ideal=(R[l].double()@built['P'][l].T@dk.double())
                keyerr=ka-built['actual_keys'][l][idx]
                reconstruction=built['prebase'][l][idx]+va
                err=actual_h[l][j]-reconstruction
                require(bool((keyerr.abs()<=1e-5+1e-4*ka.abs()).all()),'ACTUAL_BUILDER_KEY_PARITY')
                require(bool((err.abs()<=1e-5+1e-4*actual_h[l][j].abs()).all()),'LOCAL_ADDITIVE_ADAPTER_PARITY')
                row['layer'][str(l)]=dict(key_abs=float(dk.norm()),key_relative=float(dk.norm()/ka.norm().clamp_min(1e-30)),
                    v_actual_norm=float(va.norm()),v_fit_norm=float(vs.norm()),v_actual_minus_v_fit=float((va-vs).norm()),
                    ideal_U_key_gap=float(ideal.norm()),prebase_gap=float(base.norm()),
                    additive_reconstruction_max=float(err.abs().max()),builder_key_max=float(keyerr.abs().max()))
            details.append(row)
    keys={l:mean_keys(torch.stack(v).T,key_rows,entry['pack']) for l,v in raw.items()}
    comparisons={}
    for l,key in keys.items():
        ref=built['geometry'][l]['K'].detach().cpu().float();err=(key-ref).abs()
        require(bool((err<=1e-5+1e-4*ref.abs()).all()),'TERMINAL_KEY_PARITY')
        comparisons[str(l)]=dict(error_max=float(err.max()),terminal_key_sha=tensor_sha(key),builder_key_sha=tensor_sha(ref))
    payload=dict(weights={l:w.detach() for l,w in built['weights'].items()},keys=keys,context_nll=nll,native_kl=kl)
    return payload,dict(actual_context_nll=nll,actual_native_kl=kl,fit_to_actual_target_kl=targetkl,
        decomposition=details,key_comparison=comparisons,effective_energy=effective_energy(entry,built),
        seconds=time.monotonic()-start,prediction_tokens=tokens,weights_rebuilt=False,fit_feedback=False)
