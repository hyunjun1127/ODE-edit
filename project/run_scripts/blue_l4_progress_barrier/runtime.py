"""Fixed 12-trajectory experiment; reused immutable native inputs only."""
import argparse
import json
import os
from pathlib import Path
import resource
import time
import traceback
import torch
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.single_layer_cumulative_risk import binding
from project.run_scripts.single_layer_cumulative_risk.records import save,tensor_save,tensor_sha,Ledger,digest
from project.run_scripts.single_layer_cumulative_risk.evaluation import measure,materialized,generation
from project.run_scripts.single_layer_cumulative_risk.runtime import covariance
from .objective import ProgressObjective
from .algebra import calibrate,correction,risk,dot,groups
from .transfer import DEST,sha

def emit(output,stage,**kw):
    save(output/'progress'/f'{time.time_ns()}.json',dict(stage=stage,**kw))
    print('L4_EP',stage,json.dumps(kw),flush=True)

def geometry(p,u,c0,ledger):
    with ledger.time('geometry'),torch.no_grad():
        ud=u.double();m=p['M'][0].cuda().double();delta=p['native_delta'].cuda().double()
        mu=m@ud+ud;hr=ud.T@mu;cn=delta@mu
        # Scalar original normalizer is preserved; the exact-Gram expansion
        # and We-relative cross term are new authoritative reduced quantities.
        jn=float(p['J_native']);jn64=dot(delta@m,delta)+dot(delta,delta)
        del m,mu
        c=c0.double();cu=c@ud;hc=ud.T@cu
        d=(p['WN'].cuda()-p['W0'].cuda()).double();cc=d@cu
        risk0=dot(d@c,d)/2
        z=ud.T@p['K'].cuda().double();hn=z@z.T+hr
        sf=dot(delta,delta)/2
        if sf<=0 or jn<=0:raise RuntimeError('ZERO_NATIVE_ACTION_INPUT')
        return dict(hr=hr,cn=cn,hc=hc,cc=cc,hn=hn,sf=sf,jn=jn,risk0=risk0,
                    normalizer_fp64_check=float(jn64),u_gram=ud.T@ud)

def gate(model,obj,x,w,wn,we,ledger):
    inputs=obj.pack([obj.training[i][0][0] for i in (0,1)])
    with torch.no_grad(),ledger.time('actual_weight_gate'):
        overlay=obj.forward(x,**inputs)
        with materialized(w,wn,ledger):actual=model(**inputs,use_cache=False).logits
        assert torch.equal(overlay,actual),'OVERLAY_MATERIALIZED_MISMATCH'
        assert torch.equal(w,we),'TEACHER_ENTRY_RESTORE_MISMATCH'
        gap=float((overlay-actual).abs().max())
    return dict(status='ACTUAL_APPLICATION_PASS',max_logit_difference=gap,exact_WN=True,
                teacher_We=True,only_L4_editable=True,additional_forward=2,additional_backward=0)

def run_entry(args,model,tok,evaltok,records,w,w0,ledger):
    entry=args.entry;root=DEST/'imports';lock=json.loads((root/'input.lock.json').read_text())
    inputlock=json.loads((DEST/'input.lock.json').read_text());spec=inputlock['entries'][entry]
    pp=Path(spec['prepared']);assert sha(pp)==spec['prepared_sha']
    p=torch.load(pp,map_location='cpu',weights_only=True,mmap=True)
    assert tensor_sha(w0)==spec['W0_sha'] and torch.equal(p['W0'],w0.cpu())
    binding.ROOT=root
    panel=lock['entries'][entry]['panels']
    assert digest(panel)==spec['panel_sha']
    current=[records[i] for i in panel['panels']['Current100']]
    requests=[dict(r['requested_rewrite'],case_id=r['case_id']) for r in current]
    for r in requests:
        if not r['target_new']['str'].startswith(' '):r['target_new']=dict(r['target_new'],str=' '+r['target_new']['str'])
    group=groups([r['case_id'] for r in current])
    we=p['We'].cuda();wn=p['WN'].cuda();u=p['Ub'].cuda()
    with torch.no_grad():w.copy_(we)
    c0=covariance();geo=geometry(p,u,c0,ledger)
    obj=ProgressObjective(model,tok,requests,p['contexts'],binding.kernel().find_fact_lookup_idx,wn,u,geo['hr'],geo['jn'],ledger,
            penalty_cross=geo['cn'],penalty_constant=geo['jn'],accumulate_weight_gradient=True)
    assert len(obj.contexts)==6
    x0=torch.zeros((4096,u.shape[1]),device='cuda',dtype=torch.float32)
    output=args.output;save(output/'application-gate.json',gate(model,obj,x0,w,wn,we,ledger))
    teacher_sha=digest([tensor_sha(t) for t in obj.teacher])
    common=output/'common.pt'
    if args.common:
        initial=torch.load(args.common,weights_only=True,map_location='cuda')
        assert initial['entry']==entry and initial['prepared_sha']==spec['prepared_sha']
        assert initial['teacher_sha']==teacher_sha and initial['groups']==group
        terms0,g0,js0,pg0=initial['terms'],initial['g'],initial['js'],initial['request_gradients']
        ledger.add('initial_gradient_sweep_reused')
    else:
        terms0,g0,js0=obj.grouped(x0,group)
        pg0=obj.last_request_gradients
        tensor_save(common,dict(entry=entry,prepared_sha=spec['prepared_sha'],teacher_sha=teacher_sha,
                    groups=group,terms=terms0,g=g0.cpu(),js=js0.cpu(),request_gradients=pg0.cpu()))
        ledger.add('gradient_sweeps')
    nu,broot,binv,eps,cal=calibrate(g0,geo['hn'],u,float(p['native_norm']),geo['cc']/geo['sf'],geo['hc']/geo['sf'])
    save(output/'calibration.json',dict(**cal,epsilon=float(eps),native_normalizer=geo['jn'],
          native_normalizer_fp64_check=geo['normalizer_fp64_check'],teacher_sha=teacher_sha,
          groups=group,initial_objective=terms0,source_head=os.environ.get('L4_EP_SOURCE_HEAD'),
          prepared_sha=spec['prepared_sha'],input_lock_sha=sha(DEST/'input.lock.json'),
          U_gram_identity_error=float((geo['u_gram']-torch.eye(u.shape[1],device='cuda')).norm())))
    emit(output,'INITIAL_VALID',entry=entry,compute=ledger.receipt(),calibration=cal)
    if not args.common:
        r,a=risk(x0.double(),geo['cc'],geo['hc'],geo['sf']);v0=-nu*torch.linalg.solve(geo['hn'],g0.T).T
        ep,rc=correction(v0,a,js0.mean(0)[None],broot,binv,eps,-r,'EP')
        t=dot(a,v0-ep);qall=dot(a@binv,a@binv)
        vm=v0-(t/qall)*((a@binv)@binv) if qall>0 else v0
        xp=(vm/8).float();state=wn+xp@u.T
        probe_terms,_,_=obj.grouped(xp,group,False)
        with materialized(w,state,ledger):measure(model,evaltok,records,panel,False,ledger,output/'matched-risk-curve.json')
        tensor_save(output/'matched-risk.pt',dict(X=xp.cpu(),prepared_sha=spec['prepared_sha']))
        save(output/'matched-risk.json',dict(terms=probe_terms,risk=float(risk(xp.double(),geo['cc'],geo['hc'],geo['sf'])[0]),
             matched_velocity_risk_removal=float(t),h=.125,actual_update_risk_prediction=float(dot(a,xp)),
             nominal_edit_derivative=float(dot(js0.mean(0),v0)),matched_edit_derivative=float(dot(js0.mean(0),vm)),
             ep_edit_derivative=float(dot(js0.mean(0),ep)),additional_backward=0,trajectory_carry=False))
    h1=None
    for arm in args.arms:
        n=16 if arm=='EP-N16' else 8;h=1/n;dest=output/arm;dest.mkdir(exist_ok=False)
        x=x0.clone();terms,g,js,pg=terms0,g0,js0,pg0;last=wn.clone()
        tensor_save(dest/'snapshot-000.pt',dict(X=x.cpu(),prepared_path=str(pp),prepared_sha=spec['prepared_sha'],step=0))
        for k in range(n):
            if k:
                terms,g,js=obj.grouped(x,group);ledger.add('gradient_sweeps')
                pg=obj.last_request_gradients
                if arm=='H' and k==1:h1=terms['request_edit_nll']
            r,a=risk(x.double(),geo['cc'],geo['hc'],geo['sf'])
            v0=-nu*torch.linalg.solve(geo['hn'],g.T).T
            j=js if arm=='EP-J4' else js.mean(0)[None]
            velocity,control=correction(v0,a,j,broot,binv,eps,-r,arm)
            shadow=None
            if arm=='EP' and k in (0,3,7):
                if k==0:
                    assert h1 is not None
                    shadow=h1;ledger.add('nominal_shadow_sweep_reused')
                else:
                    nominal_x=(x.double()+h*v0).float()
                    shadow=obj.grouped(nominal_x,group,False,True)[0]['request_edit_nll']
                    ledger.add('nominal_shadow_sweeps')
            next_x=(x.double()+h*velocity).float()
            state=wn+next_x@u.T
            if not torch.isfinite(state).all():raise FloatingPointError('NONFINITE_PHYSICAL_UPDATE')
            dx=next_x.double()-x.double();actual=state-last
            # FP32 materialization can add an off-support rounding remainder;
            # record its size, do not silently call it exact coefficient equality.
            effective=torch.linalg.solve(geo['u_gram'],(actual.double()@u.double()).T).T
            projected=effective@u.double().T
            newrisk,_=risk(next_x.double(),geo['cc'],geo['hc'],geo['sf'])
            with torch.no_grad():
                d=(state-w0).double();actual_risk=(dot(d@c0.double(),d)/2-geo['risk0'])/geo['sf']
            row=dict(entry=entry,arm=arm,pre_step=k,step=k+1,time=(k+1)*h,h=h,pre_terms=terms,control=control,
                 nominal_edit_prediction=(js*v0).flatten(1).sum(1).mul(h).tolist(),
                 corrected_edit_prediction=(js*velocity).flatten(1).sum(1).mul(h).tolist(),
                 applied_coefficient_group_prediction=(js*dx).flatten(1).sum(1).tolist(),
                 projected_actual_group_prediction=(js*effective).flatten(1).sum(1).tolist(),
                 request_nominal_prediction=(pg.double()*v0).flatten(1).sum(1).mul(h).tolist(),
                 request_corrected_prediction=(pg.double()*velocity).flatten(1).sum(1).mul(h).tolist(),
                 request_applied_prediction=(pg.double()*effective).flatten(1).sum(1).tolist(),
                 request_observer_added_backward=0,
                 nominal_shadow_request_nll=shadow,risk_pre=float(r),risk_post=float(newrisk),
                 risk_actual_fp32=float(actual_risk),risk_rounding_difference=float(actual_risk-newrisk),
                 risk_linear=float(dot(a,dx)),risk_quadratic=float(.5*dot(dx@geo['hc'],dx)/geo['sf']),
                 actual_step_norm=float(actual.double().norm()),actual_step_energy=float(dot(actual,actual)),
                 materialization_off_support_norm=float((actual.double()-projected).norm()),
                 coefficient_update_rounding_norm=float((dx-h*velocity).norm()),
                 X_norm=float(next_x.double().norm()),selected_weight_sha=tensor_sha(state),compute=ledger.receipt())
            save(dest/f'step-{k+1:03d}.json',row)
            x=next_x;last=state
            milestones=(2,4,8,16) if n==16 else (1,2,4,8)
            if k+1 in milestones:
                tensor_save(dest/f'snapshot-{k+1:03d}.pt',dict(X=x.cpu(),prepared_path=str(pp),prepared_sha=spec['prepared_sha'],step=k+1))
                with materialized(w,state,ledger):
                    measure(model,evaltok,records,panel,k+1==n,ledger,dest/f'eval-{k+1:03d}.json')
            emit(output,'STEP',entry=entry,arm=arm,step=k+1,n=n,objective=terms['objective'],risk=float(newrisk))
        terminal_terms,_,_=obj.grouped(x,group,False)
        if entry=='Middle' and arm in ('H','R','EP'):
            with materialized(w,state,ledger):generation(model,evaltok,records,panel,ledger,dest/'generation.json')
        save(dest/'terminal.json',dict(status='TERMINAL_VALID',entry=entry,arm=arm,steps=n,terms=terminal_terms,
             WN_start_sha=spec['WN_sha'],selected_weight_sha=tensor_sha(state),risk=float(newrisk),
             risk_actual_fp32=float(actual_risk),inner_compute_z=0,inner_native_calls=0,history_append=0,
             compute=ledger.receipt(),scientific_promotion=False))
        emit(output,'ARM_TERMINAL',entry=entry,arm=arm,compute=ledger.receipt())
    with torch.no_grad():w.copy_(w0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--entry',choices=['Early','Middle','Late'],required=True)
    ap.add_argument('--arms',nargs='+',required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--common',type=Path);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);ledger=Ledger();w=w0=None
    try:
        from .seal import verify
        verify()
        records=load_prefix(binding.DATA,10000)
        model,tok,evaltok=binding.load_model(ledger);w=dict(model.named_parameters())[binding.WEIGHT];w0=w.detach().clone()
        guards={n:(p.data_ptr(),p._version) for n,p in model.named_parameters() if n!=binding.WEIGHT}
        def count(module,aa,kw):
            ledger.add('actual_model_forward_invocations');ids=kw.get('input_ids')
            if ids is not None:
                ledger.add('actual_model_forward_sequences',ids.shape[0]);ledger.add('actual_model_forward_tokens',int(kw['attention_mask'].sum()))
        model.register_forward_pre_hook(count,with_kwargs=True)
        import transformers
        save(args.output/'runtime.json',dict(source_head=os.environ.get('L4_EP_SOURCE_HEAD'),
             input_lock_sha=sha(DEST/'input.lock.json'),torch=torch.__version__,transformers=transformers.__version__,
             model_revision=Path(binding.MODEL).name,GPU=torch.cuda.get_device_name(),job=os.environ.get('SLURM_JOB_ID'),
             fp32_all=all(p.dtype==torch.float32 for p in model.parameters()),attention=model.config._attn_implementation,
             tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
             autocast=torch.is_autocast_enabled(),W0_sha=tensor_sha(w0)))
        run_entry(args,model,tok,evaltok,records,w,w0,ledger)
        assert torch.equal(w,w0)
        assert all((p.data_ptr(),p._version)==guards[n] for n,p in model.named_parameters() if n!=binding.WEIGHT)
        save(args.output/'terminal.json',dict(status='TERMINAL_VALID',entry=args.entry,arms=args.arms,W0_restore=True,
             other_parameter_mutation=0,compute=ledger.receipt(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
             peak_host_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scientific_promotion=False))
    except BaseException as exc:
        if w is not None and w0 is not None:
            with torch.no_grad():w.copy_(w0)
        save(args.output/'failure.json',dict(status='TECHNICAL_FAILURE',error=repr(exc),traceback=traceback.format_exc(),
             compute=ledger.receipt(),W0_restore=bool(torch.equal(w,w0)) if w is not None else None))
        raise

if __name__=='__main__':main()
