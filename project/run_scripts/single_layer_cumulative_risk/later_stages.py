"""B direction trials and C frozen/refresh probes; gated by prior-stage completion."""
import json
from pathlib import Path
import torch
from .algebra import reduced_metric,momentum_update
from .binding import kernel
from .directions import group_indices,normalized_rows,soft_filter,physical_unit,operator_gradient,frobenius_finite_change,product_resolution
from .evaluation import measure,materialized,generation
from .import_assets import sha
from .objective import DirectObjective,WEIGHT
from .panels import select
from .records import save,tensor_save,tensor_sha

def require_previous(path,stage):
    if path is None:raise RuntimeError('PRIOR_STAGE_COMPLETION_REQUIRED')
    receipt=json.loads(Path(path).read_text())
    if receipt.get('stage')!=stage or receipt.get('status')!='COMPLETE' or receipt.get('remaining_mandatory')!=0:
        raise RuntimeError('PRIOR_STAGE_NOT_COMPLETE')
    return sha(path)

def common(prepared_path,entry,cp,records,model,tok,w,ledger):
    r=json.loads(prepared_path.with_name('prepared-receipt.json').read_text())
    assert sha(prepared_path)==r['sha256']
    data=torch.load(prepared_path,map_location='cpu',weights_only=True,mmap=True)
    assert data['entry']==entry and torch.equal(data['We'],cp['weights'][WEIGHT])
    assert torch.equal(data['M'],cp['cache_c'])
    panel=select(records,entry);requests=[]
    for i in panel['panels']['Current100']:
        row=records[i];rw=dict(row['requested_rewrite'],case_id=row['case_id'])
        if not rw['target_new']['str'].startswith(' '):rw['target_new']=dict(rw['target_new'],str=' '+rw['target_new']['str'])
        requests.append(rw)
    we,wn,u,m=(data[k].cuda() for k in ['We','WN','Q','M']);m=m[0]
    delta=wn-we;metric=reduced_metric(u,m)
    cross=(delta@m+delta)@u
    with torch.no_grad():w.copy_(we)
    objective=DirectObjective(model,tok,requests,data['contexts'],kernel().find_fact_lookup_idx,
              wn,u,metric,data['J_native'],ledger,penalty_cross=cross,penalty_constant=data['J_native'],accumulate_weight_gradient=True)
    groups=group_indices([r['case_id'] for r in requests])
    return data,panel,we,wn,u,m,objective,groups

def build_j(objective,a,groups,output):
    objective.ledger.add('group_jacobian_build')
    gradients,values=objective.group_edit_gradients(a,groups)
    j,receipt=normalized_rows(gradients)
    # Keep actual gradients locally for auditing; never put tensors into Git.
    tensor_save(output.with_suffix('.pt'),dict(group_gradients=gradients.cpu(),J=j.cpu(),groups=groups))
    save(output.with_suffix('.json'),dict(**receipt,group_edit_nll=values,groups=groups))
    mean=gradients.mean(0);del gradients
    return j,mean

def risk_direction(w,reference,u,j,ledger):
    with ledger.time('risk_gradient_pullback'):raw=(w-reference)@u
    ledger.add('risk_gradient_pullback')
    with ledger.time('small_filter_solve'):filtered,receipt=soft_filter(raw,j)
    ledger.add('small_filter_solve')
    direction,norm=physical_unit(-filtered,u,product_resolution(w-reference,u))
    return direction,dict(**receipt,normalization=norm,final_Jd=(j@direction.flatten()).cpu().tolist())

def run_b(entry,prepared_path,output,model,tok,evaltok,records,cp,w,w0,ledger):
    from .runtime import covariance,structural,progress
    data,panel,we,wn,u,m,objective,groups=common(prepared_path,entry,cp,records,model,tok,w,ledger)
    a=torch.zeros((wn.shape[0],u.shape[1]),device=w.device,requires_grad=True)
    j,edit=build_j(objective,a,groups,output/'group-jacobian')
    c0=covariance();global_d=wn-w0;local_d=wn-we
    with ledger.time('risk_gradients'):
        op,opreceipt=operator_gradient(global_d)
        cov_product=global_d@c0
        raw={'GFminus':global_d@u,'LFminus':local_d@u,'OPminus':op@u,'COVminus':cov_product@u}
        uncertainty={'GFminus':product_resolution(global_d,u),'LFminus':product_resolution(local_d,u),
             'OPminus':product_resolution(op,u),'COVminus':product_resolution(global_d,c0)+product_resolution(cov_product,u)}
        for name,seed in [('Random1',20260910),('Random2',20260911)]:
            gen=torch.Generator(device=w.device);gen.manual_seed(seed)
            raw[name]=torch.randn(a.shape,device=w.device,generator=gen)
    directions={};probes={}
    for name,g in raw.items():
        with ledger.time('small_filter_solve'):filtered,r=soft_filter(g,j)
        ledger.add('small_filter_solve')
        if not name.startswith('Random'):filtered=-filtered
        d,nr=physical_unit(filtered,u,uncertainty.get(name,0.));directions[name]=d
        probes[name]=dict(**r,normalization=nr,final_Jd=(j@d.flatten()).cpu().tolist(),
              raw_edit_cosine=float((g.double()*edit.double()).sum()/(g.double().norm()*edit.double().norm())) if g.norm()>0 and edit.norm()>0 else None)
    directions['GFplus']=-directions['GFminus'];probes['GFplus']=dict(exact_opposite_of='GFminus',final_Jd=(j@directions['GFplus'].flatten()).cpu().tolist())
    save(output/'direction-probes.json',dict(probes=probes,operator=opreceipt,
           GF_LF_cosine=float((raw['GFminus'].double()*raw['LFminus'].double()).sum()/(raw['GFminus'].double().norm()*raw['LFminus'].double().norm())),
           native_norm=data['native_norm'],Q_rank=u.shape[1],source='common native WN',gamma=.1))
    del raw,edit
    tensor_save(output/'directions.pt',dict(directions={k:v.cpu() for k,v in directions.items()},J=j.cpu(),Q_sha=tensor_sha(u)))
    for name in ['GFminus','GFplus','LFminus','Random1','Random2','OPminus','COVminus']:
        d=directions[name]
        for amplitude in [.03,.1,.3]:
            trial=output/f'{name}-amplitude-{amplitude}';trial.mkdir(exist_ok=False)
            delta=amplitude*data['native_norm']*(d@u.T);state=wn+delta
            if not torch.isfinite(state).all():raise FloatingPointError('NONFINITE_B_TRIAL')
            with materialized(w,state,ledger):
                measure(model,evaltok,records,panel,amplitude==.1,ledger,trial/'eval.json')
                structure=structural(state,w0,we,m,c0,data['K'].cuda())
            tensor_save(trial/'endpoint.pt',dict(W=state.cpu(),M=data['MN'],entry=entry,direction=name,amplitude=amplitude))
            save(trial/'receipt.json',dict(status='TERMINAL_VALID',direction=name,amplitude=amplitude,
                 intended_extra_norm=float(delta.double().norm()),actual_extra_norm=float((state-wn).double().norm()),
                 global_risk_change=frobenius_finite_change(wn,w0,delta),local_risk_change=frobenius_finite_change(wn,we,delta),
                 structure=structure,compute=ledger.receipt(),endpoint_sha=sha(trial/'endpoint.pt')))
            progress(output,'B_TRIAL_SAVED',entry=entry,direction=name,amplitude=amplitude)

def run_c(entry,prepared_path,selection_path,output,model,tok,evaltok,records,cp,w,w0,ledger):
    from .runtime import covariance,structural,progress
    assert entry=='Middle'
    selection=json.loads(selection_path.read_text());assert selection['support']=='C' and selection['entry']=='Middle'
    assert sha(selection['endpoint_path'])==selection['endpoint_sha']
    eta=selection['eta']
    data,panel,we,wn,u,m,objective,groups=common(prepared_path,entry,cp,records,model,tok,w,ledger)
    c0=covariance();correction_length=.1*data['native_norm']/8
    initial_a=torch.zeros((wn.shape[0],u.shape[1]),device=w.device,requires_grad=True)
    j0,_=build_j(objective,initial_a,groups,output/'initial-group-jacobian')
    frozen,frozen_receipt=risk_direction(wn,w0,u,j0,ledger)
    raw_global=(wn-w0)@u;den=eta*float((raw_global@u.T).double().norm())
    soft_lambda=correction_length/den if den>0 else None
    save(output/'controller.json',dict(eta=eta,eta_source=str(selection_path),selection_sha=sha(selection_path),
        correction_length=correction_length,soft_lambda=soft_lambda,soft_calibration_status='DEFINED' if soft_lambda is not None else 'UNDEFINED_ZERO_DENOMINATOR',
        frozen_receipt=frozen_receipt,nominal_momentum=.9,penalty_in_momentum=False,steps=8))
    for arm in ['Continue','FrozenGlobal','RefreshedGlobal','RefreshedLocal','SoftGlobal']:
        directory=output/arm;directory.mkdir(exist_ok=False)
        if arm=='SoftGlobal' and soft_lambda is None:
            save(directory/'scientific-boundary.json',dict(status='UNDEFINED_ZERO_CALIBRATION',
                 eta=eta,denominator=den,completed_steps=0,arbitrary_lambda=0,fallback=0,
                 other_arms_continue=True))
            continue
        a=torch.zeros_like(initial_a,requires_grad=True);velocity=None;last=wn.clone();path=0.;correction_path=0.
        tensor_save(directory/'snapshot-000.pt',dict(W=wn.cpu(),A=a.detach().cpu(),step=0))
        terms=objective.evaluate(a,True)
        save(directory/'step-000.json',dict(step=0,**terms))
        for step in range(1,9):
            # The nominal gradient is saved before group-gradient calls clear a.grad.
            nominal_gradient=a.grad.detach().clone();current=wn+a.detach()@u.T;probe={}
            if arm=='Continue':correction=torch.zeros_like(a)
            elif arm=='FrozenGlobal':correction=correction_length*frozen
            elif arm=='SoftGlobal':
                correction=-eta*soft_lambda*((current-w0)@u)
                probe=dict(calibration_status='DEFINED')
            else:
                if step==1:j=j0
                else:j,_=build_j(objective,a,groups,directory/f'group-jacobian-{step:03d}')
                direction,probe=risk_direction(current,w0 if arm=='RefreshedGlobal' else we,u,j,ledger)
                correction=correction_length*direction
            with torch.no_grad(),ledger.time('C_optimizer_update'):
                new,velocity=momentum_update(a,nominal_gradient,velocity,eta)
                nominal_state=wn+new@u.T
                a.copy_(new+correction);state=wn+a@u.T
                if not torch.isfinite(state).all():raise FloatingPointError('NONFINITE_C_WEIGHT')
                stepnorm=float((state-last).double().norm());path+=stepnorm;last=state.clone()
                intended_correction_norm=float((correction@u.T).double().norm())
                correction_norm=float((state-nominal_state).double().norm());correction_path+=correction_norm
                nominal_step_norm=float((nominal_state-current).double().norm())
            terms=objective.evaluate(a,step<8)
            save(directory/f'step-{step:03d}.json',dict(step=step,**terms,probe=probe,
               actual_step_norm=stepnorm,path_length=path,correction_path=correction_path,
               correction_norm=correction_norm,extra_net_norm=float((state-wn).double().norm()),
               intended_correction_norm=intended_correction_norm,actual_nominal_step_norm=nominal_step_norm,
               correction_rounding_gap=correction_norm-intended_correction_norm,
               batch_net_norm=float((state-we).double().norm()),global_net_norm=float((state-w0).double().norm()),
               selected_weight_sha=tensor_sha(state),compute=ledger.receipt()))
            if step in [1,2,4,8]:tensor_save(directory/f'snapshot-{step:03d}.pt',dict(W=state.cpu(),A=a.detach().cpu(),momentum=velocity.cpu(),step=step))
            if step in [2,4,8]:
                with materialized(w,state,ledger):
                    measure(model,evaltok,records,panel,step==8,ledger,directory/f'eval-{step:03d}.json')
                    save(directory/f'structure-{step:03d}.json',structural(state,w0,we,m,c0,data['K'].cuda()))
            progress(output,'C_STEP_SAVED',arm=arm,step=step)
        with materialized(w,state,ledger):generation(model,evaltok,records,panel,ledger,directory/'generation.json')
        tensor_save(directory/'endpoint.pt',dict(W=state.cpu(),M=data['MN'],entry=entry,arm=arm,step=8))
        save(directory/'terminal.json',dict(status='TERMINAL_VALID',completed_steps=8,endpoint_sha=sha(directory/'endpoint.pt'),compute=ledger.receipt()))
