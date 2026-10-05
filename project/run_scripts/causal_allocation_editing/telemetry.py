"""Scalar-only requested/action/cost diagnostics, no checkpoint equivalent."""
import time
import torch
from .geometry import compact_cost, effective_cost


def summarize(actual, requested, weights=None):
    x,y=actual.double(),requested.double()
    nx=x.norm(dim=0);ny=y.norm(dim=0);dot=(x*y).sum(0);error=(x-y).norm(dim=0)
    valid=ny>0;cosvalid=valid&(nx>0)
    weights=torch.ones_like(nx) if weights is None else weights.double()
    def weighted(values,mask):
        mass=weights[mask].sum()
        return None if not bool(mask.any()) or float(mass)==0 else float((values[mask]*weights[mask]).sum()/mass)
    ratio=torch.zeros_like(nx);direction=torch.zeros_like(nx);cos=torch.zeros_like(nx)
    ratio[valid]=nx[valid]/ny[valid];direction[valid]=dot[valid]/ny[valid].square()
    cos[cosvalid]=dot[cosvalid]/(nx[cosvalid]*ny[cosvalid])
    return dict(columns=x.shape[1],zero_requested=int((~valid).sum()),zero_actual=int((nx==0).sum()),
        requested_norm=float(y.norm()),actual_norm=float(x.norm()),error_norm=float((x-y).norm()),
        norm_ratio=weighted(ratio,valid),directional_ratio=weighted(direction,valid),
        cosine=weighted(cos,cosvalid),absolute_error_mean=weighted(error,torch.ones_like(valid)),
        relative_error=weighted(error/torch.where(valid,ny,torch.ones_like(ny)),valid),
        zero_owner_action_norm=float(x[:,~valid].norm()) if bool((~valid).any()) else 0.,
        undefined_reason='NO_REQUESTED_ACTION' if not bool(valid.any()) else None,
        aggregation='explicit_column_weighted_nonzero_reference_mean; norms are Frobenius')


def candidate_summary(a,entry,payload,terminal=False,full_gradient=False,lambda_Q=None):
    start=time.monotonic();rows=payload['rows'];owners=torch.tensor([r['request'] for r in rows])
    rw=torch.tensor([r['kind']=='rewrite' for r in rows]);kl=~rw
    canonical_ids=set(entry['pack'].get('canonical_rows',[]))
    canonical=torch.tensor([r['global_row'] in canonical_ids for r in rows])
    layers={};requested_norms=[];realized_norms=[];B=entry['pack']['n_requests']
    for l in a.sites:
        W=payload['weights'][l].cpu();W0=entry['entry_weights'][l].detach().cpu()
        R=payload['R'][l].detach().cpu().double();K=payload['K'][l].detach().cpu().double()
        raw=payload['raw'][l].cpu().double();A=entry['factors'][l]['A']
        # Applied subtraction is FP64(W_FP32)-FP64(Wentry_FP32), never cast(U).
        effective=W.double()-W0.double()
        mean_action=effective@K;context_action=effective@raw.T
        owner_target=R[:,owners]
        q=float(payload['Q_by_layer'][l]);P=payload['P'][l]
        H=entry.get('history_entry',{}).get(l)
        if H is None:
            qc0=qh=None;decomposition='NOT_MEASURED_HISTORY_ENTRY_MISSING'
        else:
            hd=H.detach().cpu().double()
            qh=float(compact_cost(payload['R'][l],P,hd))
            qc0=float(compact_cost(payload['R'][l],P,A-hd))
            decomposition='same raw stored A minus exact frozen CPUFP32 H promotion'
        req=float(R.norm());real=float(mean_action.norm());requested_norms.append(req);realized_norms.append(real)
        eq=effective_cost(W,W0,A) if terminal else None
        eqh=(0. if not bool(torch.count_nonzero(H)) else effective_cost(W,W0,H.double())) if terminal and H is not None else None
        layers[str(l)]=dict(mean=summarize(mean_action,R),
            canonical=summarize(context_action[:,canonical],owner_target[:,canonical]) if bool(canonical.any()) else dict(status='NOT_MEASURED_CANONICAL_MAPPING_MISSING'),
            rewrite=summarize(context_action[:,rw],owner_target[:,rw]),
            KL=summarize(context_action[:,kl],owner_target[:,kl]),
            requested_R_norm=req,actual_direct_mean_norm=real,
            u_norm_max=float(payload['u'][l].detach().double().norm(dim=0).max()),
            per_layer_cap=.75,active_blocks=int((payload['u'][l].detach().double().norm(dim=0)>0).sum()),
            cap_contacts=int((payload['u'][l].detach().double().norm(dim=0)>=.75-1e-6).sum()),
            ideal_Q=q,ideal_Q_C0=qc0,ideal_Q_H=qh,Q_decomposition=decomposition,
            effective_Q=eq,effective_Q_H=eqh,effective_Q_C0=None if eqh is None else eq-eqh,
            effective_Q_status='MEASURED' if terminal else 'TERMINAL_ONLY',
            effective_update_norm=float(effective.norm()),
            solve=payload['geometry'][l],actual_update='double(W_candidate_FP32)-double(W_entry_FP32)',
            requested_reference='native returned exact R' if payload.get('native_reference_exact_R') else 'a_FP32*u_FP32')
        del effective,context_action,mean_action,raw
    rs=sum(requested_norms);ys=sum(realized_norms)
    for index,l in enumerate(a.sites):
        layers[str(l)]['requested_share']=None if rs==0 else requested_norms[index]/rs
        layers[str(l)]['actual_direct_share']=None if ys==0 else realized_norms[index]/ys
    return dict(candidate=payload['candidate'],terminal=terminal,gradient_measured=full_gradient,
        task_sum=payload['task_sum'],native_NLL_sum=payload['nll_sum'],native_KL_sum=payload['kl_sum'],
        norm_sum=payload['G'],ideal_Q_sum=payload['Q'],lambda_Q=lambda_Q,B=B,layers=layers,
        requested_zero_reason='NO_PLANNED_EDIT' if rs==0 else None,
        actual_zero_reason='ZERO_REALIZED_TOTAL' if ys==0 else None,
        no_extra_model_forward=True,no_extra_backward=True,no_durable_tensors=True,
        scope='local own-write action at actual candidate keys; net/inherited/task effects not conflated',
        seconds=time.monotonic()-start)
