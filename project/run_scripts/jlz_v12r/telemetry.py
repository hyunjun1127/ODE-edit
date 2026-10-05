"""Scalar-only requested/effective action diagnostics; no model or backward."""
import time
import torch
import torch.nn.functional as F
from project.run_scripts.jlz_realized_writer.telemetry import vector_metrics
from project.run_scripts.jlz_realized_subject.geometry import inverse_apply
from project.run_scripts.jlz_native_writer_aware.common import require


@torch.no_grad()
def candidate(a,entry,R,built,F,controller=None):
    start=time.monotonic()
    norms={str(l):R[l].double().norm(dim=0).cpu().tolist() for l in a.sites}
    energies=torch.stack([R[l].double().square().sum(0).cpu() for l in a.sites])
    denominator=energies.sum(0)
    shares=[[float(energies[i,r]/denominator[r]) if denominator[r]>0 else None
             for r in range(denominator.numel())] for i in range(len(a.sites))]
    layers={};Q=0.
    for i,l in enumerate(a.sites):
        p=built['P'][l];A=entry['factors'][l]['A'];AP=(A@p.cpu()).to(p.device)
        gamma=p.T@AP;q=float(((R[l].double().T@R[l].double())*gamma.T).sum());Q+=q
        layers[str(l)]=dict(requested_norm=norms[str(l)],requested_energy=energies[i].tolist(),
            requested_energy_share=shares[i],ideal_Q=q,solve=built['metadata'][l],
            ideal_update_norm=float((R[l].double()@p.T).norm()),
            SPD_energy_bound_applicable=bool(entry['factors'][l]['SPD'] and entry['factors'][l]['asymmetry_max']==0.),
            raw_A_asymmetry=entry['factors'][l]['asymmetry_max'])
    return dict(candidate=built['candidate'],F=F.tolist(),layers=layers,requested_energy=float(denominator.sum()),
        ideal_Q=Q,requested_energy_quarter=float(denominator.sum())/4,
        forward_effective_response=True,production_builder_reverse=0,
        no_extra_forward=True,no_extra_backward=True,seconds=time.monotonic()-start,
        physical_cost='CPU/GPU scalar matmul/reduction; no extra model forward or backward')


@torch.no_grad()
def terminal(a,entry,R,built,observed,controller=None):
    start=time.monotonic();rows=built['rows'];owners=torch.tensor([r['request'] for r in rows]);layers={}
    canonical=entry['pack']['canonical_rows'];requested=torch.stack([R[l].double().norm(dim=0).cpu() for l in a.sites])
    realized=[];totalQ=0.;energy=float(requested.square().sum())
    for l in a.sites:
        P=built['P'][l];A=entry['factors'][l]['A'];K=built['K'][l]
        W=built['weights'][l];W0=entry['entry_weights'][l]
        effective=W.double()-W0.double();U=R[l].double()@P.T
        AP=(A@P.cpu()).to(P.device);G=P.T@AP
        idealQ=float(((R[l].double().T@R[l].double())*G.T).sum());totalQ+=idealQ
        effQ=0.;metric=A.to(effective.device)
        for begin in range(0,effective.shape[0],128):
            block=effective[begin:begin+128];effQ+=float(((block@metric)*block).sum())
        action=built['v'][l].T.double();target=R[l].detach().cpu().double()[:,owners]
        anchor=entry['anchors'][l].detach().cpu()[owners]
        rowmetrics=vector_metrics(action,target,anchor)
        groups={}
        for name,ix in (('canonical',canonical),('rewrite',[i for i,r in enumerate(rows) if r['kind']=='rewrite']),
                        ('KL',[i for i,r in enumerate(rows) if r['kind']=='kl'])):
            groups[name]=[dict(owner=rows[i]['request'],global_row=rows[i]['global_row'],**rowmetrics[i]) for i in ix]
        mean_action=(F.linear(K.T.float(),W)-F.linear(K.T.float(),W0)).T.cpu().double()
        groups['mean']=[dict(owner=r,**m) for r,m in enumerate(vector_metrics(mean_action,R[l].cpu(),entry['anchors'][l].cpu()))]
        M=P.T@built['raw'][l].to(P.device).double().T
        own=R[l].double().cpu()[:,owners]*M.cpu()[owners,torch.arange(len(rows))][None,:]
        cross=R[l].double().cpu()@M.cpu()-own
        capacity=(K.T@inverse_apply(entry['factors'][l],K)).diagonal().cpu()
        raw_symmetric=entry['factors'][l]['asymmetry_max']==0.
        applicability=bool(entry['factors'][l]['SPD'] and raw_symmetric)
        pre_gap=None if observed.get('masked_bases') is None else observed['masked_bases'][l].double()-built['prebase'][l].double()
        layers[str(l)]=dict(**groups,ideal_Q=idealQ,effective_Q=effQ,
            effective_definition='double(Wcandidate_FP32)-double(Wentry_FP32)',
            ideal_update_norm=float(U.norm()),cast_update_norm=float(U.float().norm()),
            effective_update_norm=float(effective.norm()),capacity=capacity.tolist(),
            ideal_cross_owner_norm=cross.norm(dim=0).tolist(),
            actual_minus_ideal_self_norm=(action-own).norm(dim=0).tolist(),
            cross_owner_policy='ideal linear owner decomposition; effective FP32 rounding is not owner-separable',
            zero_owner_actual_norm=[dict(owner=r,global_row=i,norm=float(action[:,i].norm())) for i,r in enumerate(owners.tolist())
                                    if not bool(torch.count_nonzero(R[l][:,r]))],
            masked_pre_minus_actual_pre_RMS=None if pre_gap is None else float(pre_gap.square().mean().sqrt()),
            SPD_energy_bound_applicable=applicability,raw_A_asymmetry=entry['factors'][l]['asymmetry_max'],
            requested_energy=float(R[l].double().square().sum()),energy_quarter_bound=float(R[l].double().square().sum())/4,
            terminal_gradient=None,terminal_gradient_status='NO_BACKWARD_TERMINAL',solve=built['metadata'][l])
        realized.append(action[:,canonical].norm(dim=0))
        del effective,U,AP,G,metric
    realized=torch.stack(realized);reqe=requested.square();reale=realized.square()
    owner_shares=[]
    for r in range(requested.shape[1]):
        den=float(reqe[:,r].sum());actual=float(reale[:,r].sum())
        owner_shares.append(dict(owner=r,requested_energy_share=(reqe[:,r]/den).tolist() if den else None,
            realized_canonical_energy_share=(reale[:,r]/actual).tolist() if actual else None,
            requested_status='DEFINED' if den else 'NO_PLANNED_EDIT',
            realized_status='DEFINED' if actual else 'ZERO_REALIZED_TOTAL'))
    require(all(torch.isfinite(torch.tensor(row['ideal_Q'])) for row in layers.values()),'NONFINITE_TELEMETRY_Q')
    return dict(candidate=built['candidate'],B=entry['pack']['n_requests'],layers=layers,shares=owner_shares,
        subject_F=observed['F'].tolist(),subject_nll=observed['nll'].tolist(),subject_KL=observed['kl'].tolist(),
        requested_energy=energy,ideal_Q=totalQ,energy_quarter_bound=energy/4,
        terminal_extra_planner_forward=0,terminal_extra_planner_backward=0,
        diagnostic_inverse_solves=len(a.sites),effective_Q_measured=True,no_durable_tensors=True,
        seconds=time.monotonic()-start,actual_alltoken_loss='PAID_COMMIT_OBSERVER_NOT_THIS_SUBJECT_VALUE')
