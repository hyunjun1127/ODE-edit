"""Single c0 BUILD, fixed own-entry prices, authoritative candidate stream once."""
import hashlib
import json
import time
import torch
from project.run_scripts.jlz_native_writer_aware.common import require
from .engine import CandidateObjective
from .price import initialize
from project.run_scripts.jlz_interference_l1.cap_controller import RequestController
from project.run_scripts.jlz_interference_l1.cap_optimizer import EfficiencyAdamAbs,analytic_norm
from .telemetry import candidate as candidate_telemetry


def emit(events,kind,payload):
    if events is None:return
    if hasattr(events,'emit'):events.emit(kind,payload)
    else:events(kind,payload)


def fit(a,entry,arm_profile,events=None,built0=None,price=None,event_sink=None):
    start=time.monotonic();events=events if event_sink is None else event_sink
    mark=events.mark() if events is not None and hasattr(events,'mark') else None
    engine=CandidateObjective(a,entry,events);R=engine.zeros();sites=tuple(a.sites)
    arm=arm_profile.get('arm','PRICE');anchor_star=entry['anchors'][a.profile['anchor_layer']]
    if built0 is None:built0=engine.build(R,0)
    else:
        require(built0['candidate']==0,'C0_REUSE_CANDIDATE');engine.calls['logical_builds']=1
        engine.calls['builder_stage_groups']=len(entry['groups'])*max(0,len(sites)-1)
    if price is None:price=initialize(a,entry,built0,arm,batch=entry.get('batch',getattr(events,'batch',arm_profile.get('batch',1))))
    emit(events,'entry_price',dict(**price['record'],record_sha256=price['sha256']))
    controller=RequestController(torch.stack([entry['anchors'][l] for l in sites]),anchor_star,sites,
        price['effective_pi'],n_exp=arm_profile.get('n_exp',4),grace=arm_profile.get('K_grace',12),threshold=arm_profile.get('tau_F',.05),c=arm_profile['c'],
        beta_base=arm_profile['beta_base'],cap_mode=arm_profile['cap_mode'],beta_max_native_scale=arm_profile['beta_max_native_scale'])
    optimizer=EfficiencyAdamAbs(R);logical_backwards=updates=candidates=0;digest=hashlib.sha256()
    optimizer_seconds=projection_seconds=pullback_seconds=build_seconds=subject_seconds=telemetry_seconds=io_seconds=0.
    for k in range(25):
        result=engine.evaluate(R,k,active_previous=controller.active,terminal=k==24,capture=True,
                               built=built0 if k==0 else None,blind=False)
        if k==0:built0=None  # Do not retain a second full FP32 payload through the remaining candidates.
        F=result['F'];active,terminal=controller.observe(F,k)
        require(torch.equal(active.cpu(),result['active_mask']),'ACTIVE_FORWARD_BACKWARD_MASK_IDENTITY')
        built=result['built'];observed=result['observed'];norm,gn=analytic_norm(R,anchor_star,active)
        scalar=candidate_telemetry(a,entry,R,built,F,controller);telemetry_seconds+=scalar['seconds']
        row=dict(candidate=k,ordinal=k+1,entry_price_sha256=price['sha256'],F=F.tolist(),
            nll=observed['nll'].tolist(),KL=observed['kl'].tolist(),full_task_sum=float(F.sum()),
            masked_backward_sum=observed['masked_backward_sum'],norm=norm.cpu().tolist(),
            J_mean=float((F.to(norm.device)+norm).mean()),active_mask=active.cpu().tolist(),terminal=terminal,
            backward=bool(observed['logical_backward']),controller=controller.receipt(),telemetry=scalar,
            gradient_status='NO_BACKWARD_TERMINAL' if terminal else 'MASKED_REQUEST_SUM_MEASURED')
        build_seconds+=built['seconds'];subject_seconds+=observed['seconds'];logical_backwards+=observed['logical_backward']
        if terminal:
            require(observed['backward_groups']==0,'NO_TERMINAL_BACKWARD');row['states']=controller.terminal_states(F)
        else:
            require(result['gradient'] is not None,'MISSING_SAME_LAYER_GRADIENT')
            pullback_seconds+=0. if result['pullback'] is None else result['pullback']['seconds']
            gradients={l:result['gradient'][l]+gn[l] for l in sites};controller.before_update(F)
            Rnext,projection=optimizer.step(R,gradients,active,caps=controller.caps,weights=controller.weights,beta=controller.beta)
            controller.record_update(active);require(torch.equal(optimizer.t.cpu(),controller.t.cpu()),'REQUEST_COUNTER_JOIN')
            optimizer_seconds+=projection['adam_seconds'];projection_seconds+=projection['projection_seconds'];updates+=1
            pre_support=torch.stack([R[l].double().norm(dim=0).cpu()>0 for l in sites])
            post_support=torch.stack([Rnext[l].double().norm(dim=0).cpu()>0 for l in sites])
            row['support_transition']=dict(zero_to_positive=(~pre_support&post_support).tolist(),
                positive_to_zero=(pre_support&~post_support).tolist(),
                defined_by_exact_norm_zero=True,diagnostic_threshold_not_used=True)
            row['projection']=projection;row['post_update_controller']=controller.receipt()
        encoded=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n';digest.update(encoded)
        io_start=time.monotonic();emit(events,'candidate',row);io_seconds+=time.monotonic()-io_start;candidates+=1
        if terminal:break
        R=Rnext
    require(candidates<=25 and updates<=24 and logical_backwards<=24,'25_24_BUDGET')
    require(terminal and not row['backward'],'LAST_EVALUATED_TERMINAL')
    stream=events.reference_since(mark) if events is not None and hasattr(events,'reference_since') else dict(
        path=None,line_count=candidates,record_sha256=digest.hexdigest(),status='NOT_PERSISTED_NO_EVENT_SINK')
    require(stream['line_count']==candidates,'AUTHORITATIVE_CANDIDATE_STREAM_COUNT')
    receipt=dict(status='LAST_EVALUATED_FINITE_TERMINAL',arm=arm,candidates=candidates,updates=updates,
        logical_builds=engine.calls['logical_builds'],logical_subject_forwards=engine.calls['logical_subject_forwards'],
        logical_subject_backwards=logical_backwards,request_updates=int(optimizer.t.sum()),terminal_candidate=built['candidate'],
        terminal_no_backward=True,terminal_extra_forward=0,requested_gradient='FP64_Lambda_M_rows_T',
        active_loss_mask='current_F_threshold_with_irreversible_active_state',norm_analytic_once=True,
        production_builder_reverse=0,production_solve_VJP=0,controller=controller.receipt(),
        terminal_states=controller.terminal_states(F),entry_price_sha256=price['sha256'],candidate_stream=stream,
        candidate_record_digest=digest.hexdigest(),call_counts=dict(engine.calls),build_seconds=build_seconds,
        subject_seconds=subject_seconds,pullback_seconds=pullback_seconds,optimizer_seconds=optimizer_seconds,
        projection_seconds=projection_seconds,candidate_scalar_telemetry_seconds=telemetry_seconds,
        price_seconds=price['price_seconds'],loo_seconds=price['loo_seconds'],candidate_io_seconds=io_seconds,
        seconds=time.monotonic()-start,timing_policy='exclusive stages plus inclusive overall fit; do not add overall again',
        checkpoint_saved=False,exact_resume='NOT_AVAILABLE',events_embedded=False)
    return dict(built=built,R=R,receipt=receipt,F=F,mask=active,observed=observed,
                engine=engine,controller=controller,price=price)
