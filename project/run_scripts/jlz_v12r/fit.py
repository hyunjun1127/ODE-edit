"""At most25 last-evaluated candidates; owner-local activation/energy control."""
import time
import torch
from project.run_scripts.jlz_native_writer_aware.common import require
from .engine import CandidateObjective
from .controller import RequestController
from .optimizer import EfficiencyAdamAbs,analytic_norm
from .telemetry import candidate as candidate_telemetry


def emit(events,kind,payload):
    if events is None:return
    if hasattr(events,'emit'):events.emit(kind,payload)
    else:events(kind,payload)


def fit(a,entry,arm_profile,events=None):
    start=time.monotonic();engine=CandidateObjective(a,entry,events);R=engine.zeros()
    sites=tuple(a.sites);anchor_star=entry['anchors'][a.profile['anchor_layer']]
    controller=RequestController(torch.stack([entry['anchors'][l] for l in sites]),anchor_star,sites,
        n_exp=arm_profile.get('n_exp',4),base_multiplier=arm_profile.get('base_multiplier',1.),
        grace=arm_profile.get('K_grace',12),threshold=arm_profile.get('tau_F',.05))
    optimizer=EfficiencyAdamAbs(R,lr=.1,eps=1e-8,betas=(.9,.999));ledger=[]
    logical_backwards=updates=0;optimizer_seconds=pullback_seconds=build_seconds=subject_seconds=telemetry_seconds=0.
    for k in range(25):
        # Capture pre-injection bases in this SAME forward so candidate0 no-op
        # terminal and candidate24 require no diagnostic/stop witness forward.
        result=engine.evaluate(R,k,active_previous=controller.active,terminal=k==24,
            capture=True,blind=arm_profile.get('blind',False))
        F=result['F'];active,terminal=controller.observe(F,k)
        require(torch.equal(active.cpu(),result['active_mask']),'ACTIVE_FORWARD_BACKWARD_MASK_IDENTITY')
        built=result['built'];observed=result['observed'];norm,gn=analytic_norm(R,anchor_star,active)
        scalar_telemetry=candidate_telemetry(a,entry,R,built,F);telemetry_seconds+=scalar_telemetry['seconds']
        row=dict(candidate=k,ordinal=k+1,F=F.tolist(),nll=observed['nll'].tolist(),KL=observed['kl'].tolist(),
            full_task_sum=float(F.sum()),masked_backward_sum=observed['masked_backward_sum'],
            norm=norm.cpu().tolist(),J_mean=float((F.to(norm.device)+norm).mean()),
            active_mask=active.cpu().tolist(),terminal=terminal,
            backward=bool(observed['logical_backward']),controller=controller.receipt(),
            telemetry=scalar_telemetry,
            gradient_status='NO_BACKWARD_TERMINAL' if terminal else 'MASKED_REQUEST_SUM_MEASURED')
        build_seconds+=built['seconds'];subject_seconds+=observed['seconds']
        logical_backwards+=observed['logical_backward']
        if terminal:
            require(observed['backward_groups']==0,'NO_TERMINAL_BACKWARD')
            row['states']=controller.terminal_states(F);ledger.append(row);emit(events,'candidate',row);break
        require(result['gradient'] is not None,'MISSING_SAME_LAYER_GRADIENT')
        pullback_seconds+=0. if result['pullback'] is None else result['pullback']['seconds']
        gradients={l:result['gradient'][l]+gn[l] for l in sites}
        # Full pullback may include inactive columns through off-owner M, but
        # those columns do not enter Adam until their own irreversible activation.
        controller.before_update(F);update_start=time.monotonic()
        Rnext,projection=optimizer.step(R,gradients,active,caps=controller.caps,radii=controller.radii)
        controller.record_update(active)
        require(torch.equal(optimizer.t,controller.t),'REQUEST_BIAS_CONTROLLER_COUNTER_JOIN')
        optimizer_seconds+=time.monotonic()-update_start;updates+=1
        row['projection']=projection;row['post_update_controller']=controller.receipt()
        ledger.append(row);emit(events,'candidate',row);R=Rnext
    require(len(ledger)<=25 and updates<=24 and logical_backwards<=24,'25_24_BUDGET')
    require(ledger[-1]['terminal'] and ledger[-1]['backward'] is False,'LAST_EVALUATED_TERMINAL')
    receipt=dict(status='LAST_EVALUATED_FINITE_TERMINAL',arm=arm_profile.get('arm','MAIN'),
        candidates=len(ledger),updates=updates,logical_builds=engine.calls['logical_builds'],
        logical_subject_forwards=engine.calls['logical_subject_forwards'],
        logical_subject_backwards=logical_backwards,request_updates=int(optimizer.t.sum()),
        terminal_candidate=built['candidate'],terminal_no_backward=True,terminal_extra_forward=0,
        requested_gradient='FP64_Lambda_M_T' if not arm_profile.get('blind',False) else 'same_owner_adjoint_sum',
        active_loss_mask='current_F_threshold_with_irreversible_active_state',
        norm_analytic_once=True,production_builder_reverse=0,production_solve_VJP=0,
        controller=controller.receipt(),terminal_states=controller.terminal_states(F),events=ledger,
        call_counts=dict(engine.calls),build_seconds=build_seconds,subject_seconds=subject_seconds,
        pullback_seconds=pullback_seconds,optimizer_projection_seconds=optimizer_seconds,
        candidate_scalar_telemetry_seconds=telemetry_seconds,
        seconds=time.monotonic()-start,timing_policy='exclusive stages plus inclusive overall fit; do not add overall again',
        checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    return dict(built=built,R=R,receipt=receipt,F=F,mask=active,observed=observed,
        engine=engine,controller=controller)
