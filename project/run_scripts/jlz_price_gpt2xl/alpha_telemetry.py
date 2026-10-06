"""Compact action/share/cost diagnostics from already evaluated payload only."""
import math
import time
import torch
import torch.nn.functional as F
from project.run_scripts.jlz_native_writer_aware.common import require


def _shares(values):
    totals=values.sum(0)
    return [[float(values[i,r]/totals[r]) if totals[r]>0 else None for r in range(values.shape[1])]
            for i in range(values.shape[0])]


def _vectors(action,target,anchor):
    x=action.double();y=target.double();xn=x.norm(dim=0);yn=y.norm(dim=0);dot=(x*y).sum(0)
    return dict(normratio=[float(xn[r]/yn[r]) if yn[r]>0 else None for r in range(len(yn))],
        directionalratio=[float(dot[r]/yn[r].square()) if yn[r]>0 else None for r in range(len(yn))],
        cosine=[float(dot[r]/(xn[r]*yn[r])) if xn[r]>0 and yn[r]>0 else None for r in range(len(yn))],
        relative_error=[float((x[:,r]-y[:,r]).norm()/yn[r]) if yn[r]>0 else None for r in range(len(yn))],
        zero_target_leakage=[float(xn[r]/anchor[r]) if yn[r]==0 else None for r in range(len(yn))],
        actual_norm=xn.tolist(),target_norm=yn.tolist(),anchor=anchor.tolist(),
        target_norm_over_anchor=(yn/anchor).tolist(),actual_norm_over_anchor=(xn/anchor).tolist(),
        action_squared_norm=x.square().sum(0).tolist(),target_squared_norm=y.square().sum(0).tolist(),
        action_target_dot=dot.tolist(),residual_squared_norm=(x-y).square().sum(0).tolist())


def _history_product(H,p):
    """FP64 CPU product without a new full-sized FP64 history allocation."""
    result=torch.empty((H.shape[0],p.shape[1]),dtype=torch.float64,device='cpu')
    for begin in range(0,H.shape[0],128):result[begin:begin+128]=H[begin:begin+128].double()@p
    return result


def _history_energy(H,block):
    # Bound transient history casting to128rows and the output to the rowchunk.
    product=torch.zeros_like(block)
    for begin in range(0,H.shape[0],128):
        product.add_(block[:,begin:begin+128]@H[begin:begin+128].double())
    return float((product*block).sum())


@torch.no_grad()
def _ideal_cost(entry,built,R,l):
    p=built['P'][l];C0=entry['factors'][l]['C0'];H=entry['factors'][l]['H']
    gram=R[l].double().T@R[l].double()
    C0P=_history_product(C0,p.cpu()).to(p.device)
    HP=_history_product(H,p.cpu()).to(p.device)
    covariance=float((gram*(p.T@C0P).T).sum())
    history=float((gram*(p.T@HP).T).sum());total=covariance+history
    require(math.isfinite(total),'NONFINITE_TELEMETRY_Q')
    if history is not None:require(math.isfinite(history),'NONFINITE_TELEMETRY_Q_H')
    # ||R P.T||_F without materializing a second full-sized update matrix.
    update2=float((gram*(p.T@p).T).sum())
    return dict(ideal_Q=total,Q_H=history,Q_C0=covariance,C0_scale=1.,
                Q_definition='trace(U C0 U.T)+trace(U H U.T); diagnostic only, not Alpha A0 energy',
                Q_split_status='MEASURED' if history is not None else 'NOT_AVAILABLE_NO_HISTORY_BINDING',
                ideal_update_squared_norm_trace=update2,
                ideal_update_norm=update2**.5 if update2>=0 else None,
                ideal_update_norm_status='MEASURED_COMPACT_TRACE' if update2>=0 else 'NOT_AVAILABLE_TRACE_ROUNDOFF')


@torch.no_grad()
def candidate(a,entry,R,built,F,controller):
    start=time.monotonic();sites=tuple(a.sites);norm=torch.stack([R[l].double().norm(dim=0).cpu() for l in sites])
    normalized=norm/controller.anchors.cpu();spend=controller.prices.cpu()*normalized
    rawshare=_shares(normalized);pricedshare=_shares(spend);energyshare=_shares(norm.square())
    canonical=entry['pack']['canonical_rows'];actual=torch.stack([built['v'][l][canonical].double().norm(dim=1) for l in sites])
    realizedshare=_shares(actual.square());layers={};Q=0.
    for i,l in enumerate(sites):
        costs=_ideal_cost(entry,built,R,l);Q+=costs['ideal_Q']
        metrics=_vectors(built['v'][l][canonical].T,R[l].cpu(),entry['anchors'][l].cpu().double())
        layers[str(l)]=dict(requested_norm=norm[i].tolist(),normalized_norm=normalized[i].tolist(),
            raw_relative_norm_share=rawshare[i],priced_spend_share=pricedshare[i],requested_energy_share=energyshare[i],
            realized_canonical_energy_share=realizedshare[i],canonical=metrics,**costs,
            solve=built['metadata'][l],raw_A_asymmetry=entry['factors'][l]['asymmetry_max'])
    return dict(candidate=built['candidate'],layers=layers,weighted_spend=spend.sum(0).tolist(),
        spend_slack=(controller.beta.cpu()-spend.sum(0)).tolist(),
        exact_support=(norm>0).sum(0).tolist(),tiny_diagnostic_count=((normalized>0)&(normalized<=1e-12)).sum(0).tolist(),
        above_diagnostic_count=(normalized>1e-12).sum(0).tolist(),cap_mode=controller.cap_mode,ideal_Q=Q,
        price_fixed=True,no_extra_forward=True,no_extra_backward=True,no_extra_solve=True,
        seconds=time.monotonic()-start,
        cost_policy='C0@Q and H@Q measured separately; diagnostic only, no nonsymmetric A0 energy')


@torch.no_grad()
def terminal(a,entry,R,built,observed,controller=None):
    start=time.monotonic();rows=built['rows'];owners=torch.tensor([r['request'] for r in rows]);layers={}
    B=entry['pack']['n_requests'];canonical=entry['pack']['canonical_rows'];totalQ=0.
    for l in a.sites:
        costs=_ideal_cost(entry,built,R,l);totalQ+=costs['ideal_Q']
        W=built['weights'][l];W0=entry['entry_weights'][l];C0=entry['factors'][l]['C0'];H=entry['factors'][l]['H']
        effectiveC0=effectiveH=norm2=0.
        for begin in range(0,W.shape[1],128):
            block=(W[:,begin:begin+128].T.double()-W0[:,begin:begin+128].T.double()).cpu()
            effectiveC0+=_history_energy(C0,block);norm2+=float(block.square().sum())
            if H is not None:effectiveH+=_history_energy(H,block)
        effectiveQ=effectiveC0+effectiveH
        groups={};target=R[l].cpu().double()
        for role,indices in (('canonical',canonical),('rewrite',[i for i,r in enumerate(rows) if r['kind']=='rewrite']),
                             ('KL',[i for i,r in enumerate(rows) if r['kind']=='kl'])):
            values=[]
            for owner in range(B):
                ix=[i for i in indices if rows[i]['request']==owner]
                if not ix:raise RuntimeError('TERMINAL_ROLE_OWNER_COVERAGE')
                x=built['v'][l][ix].T.double();y=target[:,owner,None].expand(-1,len(ix))
                anchor=entry['anchors'][l][owner].cpu().double().expand(len(ix));metric=_vectors(x,y,anchor)
                summary={}
                for name,items in metric.items():
                    selected=[v for v in items if v is not None]
                    summary[name]=None if not selected else sum(selected)/len(selected)
                summary.update(owner=owner,rows=len(ix),unit='owner_mean_context_ratio',
                    sufficient_statistics={k:sum(metric[k]) for k in ('action_squared_norm','target_squared_norm','action_target_dot','residual_squared_norm')},
                    valid_counts={k:sum(v is not None for v in metric[k]) for k in ('normratio','directionalratio','cosine','relative_error')},
                    target_class='exact_zero' if float(target[:,owner].norm())==0 else
                        'positive_tiny_diagnostic' if float(target[:,owner].norm()/anchor[0])<=1e-12 else 'positive_above_diagnostic_threshold',
                    ratio_undefined_reason='ZERO_TARGET' if float(target[:,owner].norm())==0 else None)
                values.append(summary)
            groups[role]=values
        K=built['K'][l];mean=(a.local_linear(l,K.T.float(),W)-a.local_linear(l,K.T.float(),W0)).T.cpu()
        groups['mean']=_vectors(mean,target,entry['anchors'][l].cpu().double())
        gap=None if observed.get('masked_bases') is None else observed['masked_bases'][l].double()-built['prebase'][l].double()
        layers[str(l)]=dict(**groups,**costs,effective_Q=effectiveQ,
            effective_Q_H=effectiveH,effective_Q_C0=effectiveC0,
            effective_update_norm=norm2**.5,effective_definition='double(Wcandidate_FP32)-double(Wentry_FP32)',
            masked_pre_minus_actual_pre_RMS=None if gap is None else float(gap.square().mean().sqrt()),
            requested_actual_comparison='effective FP32 local action vs requested R, not exact full causal gradient',
            solve=built['metadata'][l],raw_A_asymmetry=entry['factors'][l]['asymmetry_max'])
        require(math.isfinite(effectiveQ),'NONFINITE_TERMINAL_EFFECTIVE_Q')
    return dict(schema='PRICE_CAP_BASE_REALIZATION_V1',diagnostic_threshold=1e-12,threshold_usage='REPORT_ONLY',
        sufficient_statistics_unit='sum_over_actual_role_context_rows_per_owner',mean_unit='one_mean_key_action_per_owner',
        candidate=built['candidate'],B=B,layers=layers,ideal_Q=totalQ,
        subject_F=observed['F'].tolist(),subject_nll=observed['nll'].tolist(),subject_KL=observed['kl'].tolist(),
        effective_Q_measured=True,diagnostic_inverse_solves=0,terminal_extra_planner_forward=0,
        terminal_extra_planner_backward=0,no_durable_tensors=True,seconds=time.monotonic()-start,
        actual_alltoken_loss='PAID_COMMIT_OBSERVER_NOT_THIS_SUBJECT_VALUE')
