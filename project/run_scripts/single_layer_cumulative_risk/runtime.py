"""A native preparation / direct candidates, with independent exact entry restore."""
import argparse
import importlib
import json
import os
from pathlib import Path
import time
import traceback
import torch
from scripts.fixed_counterfact import load_prefix
from .import_assets import ROOT,sha
from .binding import DATA,PROJECTOR,COVARIANCE,MODEL,load_entry,load_model,native_write,kernel,restore_rng,historical
from .objective import WEIGHT,DirectObjective
from .panels import select,ENTRIES,choose_alpha
from .algebra import action,orthobasis,projector_basis,reduced_metric,calibrated_eta,momentum_update
from .records import save,tensor_save,tensor_sha,Ledger,digest
from .evaluation import measure,materialized,generation

def finite_state(w):
    if not torch.isfinite(w).all():raise FloatingPointError('NONFINITE_WEIGHT')

def structural(w,w0,we,m,c0,k=None):
    # FP64 scalars/chunk reductions; parameter storage remains FP32.
    d=(w-w0);local=w-we
    def norm2(x):return float(x.double().square().sum())
    with torch.no_grad():
        cov=float(((d@c0).double()*d.double()).sum())*.5
        history=float(((local@m).double()*local.double()).sum())
        def spectral(x):
            estimates=[]
            for seed in [20260910,20260911]:
                gen=torch.Generator(device=w.device);gen.manual_seed(seed)
                v=torch.randn(x.shape[1],generator=gen,device=w.device);v=v/v.norm()
                for _ in range(30):
                    y=x.T@(x@v);norm=y.norm()
                    if norm==0:break
                    v=y/norm
                eigen=(x@v).square().sum();res=(x.T@(x@v)-eigen*v).norm()
                estimates.append(dict(singular_estimate=float(eigen.sqrt()),eigen_residual=float(res),seed=seed))
            return estimates
        estimates=spectral(d);local_estimates=spectral(local)
        local_cov=float(((local@c0).double()*local.double()).sum())*.5
    return dict(global_frobenius_sq=norm2(d),local_frobenius_sq=norm2(local),global_covariance_risk=cov,
                local_history_action=history,local_L2_action=norm2(local),operator_estimates=estimates,
                local_operator_estimates=local_estimates,local_covariance_risk=local_cov,
                native_action=history+norm2(local),current_key_action=norm2(local@k) if k is not None else None,
                current_key_action_in_native_penalty=False,operator_is_upper_bound=False)

def covariance():
    # Same serialized native second moment, not checkpoint's empty covariance map.
    import numpy as np
    with np.load(COVARIANCE,allow_pickle=False) as d:
        # Native SecondMoment.moment performs Torch FP32 division, not NumPy's
        # float64 scalar-promotion path. The source asset bytes stay untouched.
        moment=torch.from_numpy(d['mom2.mom2'])/int(d['mom2.count'])
    return moment.float().cuda()

def progress(output,stage,**fields):
    save(output/'progress'/f'{time.time_ns()}.json',dict(stage=stage,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields))
    print('CUMRISK_PROGRESS',stage,json.dumps(fields),flush=True)

def native_run(entry,output,model,tok,evaltok,records,cp,targets,current,w,w0,ledger):
    panel=select(records,entry);we=cp['weights'][WEIGHT].cuda();m=cp['cache_c'][0].cuda()
    with ledger.time('projector_load'):p=torch.load(PROJECTOR,map_location='cpu',weights_only=True,mmap=True)[0].clone().cuda()
    with ledger.time('covariance_load'):c0=covariance()
    wn,mn,k,hp,native=native_write(model,tok,cp,targets,current,p,ledger)
    delta=wn-we
    with ledger.time('native_support'):
        system=p@(k@k.T+m)+torch.eye(k.shape[0],device='cuda')
        bn=torch.linalg.solve(system,p@k)
        ub,ub_receipt=orthobasis(bn)
        del system
    with ledger.time('full_projector_basis'):q,q_receipt=projector_basis(p)
    with ledger.time('native_metric'):jn=action(delta,m)
    normalizer_status='NATIVE_ASSISTED_RATIO'
    if not torch.isfinite(jn):raise FloatingPointError('NONFINITE_NATIVE_NORMALIZER')
    if jn==0:
        # Preserve an honest typed dependency; no invented large epsilon.
        raise RuntimeError('NATIVE_NORMALIZER_ZERO_REQUIRES_DECLARED_PHYSICAL_SCALING')
    prepared=dict(entry=entry,W0=w0.cpu(),We=we.cpu(),WN=wn.cpu(),M=cp['cache_c'],MN=mn,
                  native_delta=delta.cpu(),native_input_basis=bn.cpu(),Ub=ub.cpu(),Q=q.cpu(),K=k.cpu(),
                  native_targets=targets,contexts=cp['metadata']['contexts'],metadata=cp['metadata'],
                  J_native=float(jn),native_norm=float(delta.norm()),normalizer_status=normalizer_status,
                  projector_basis=q_receipt,native_basis=ub_receipt,input_lock_sha=sha(ROOT/'input.lock.json'))
    tensor_save(output/'prepared.pt',prepared)
    save(output/'prepared-receipt.json',dict(path=str(output/'prepared.pt'),sha256=sha(output/'prepared.pt'),
         WN_sha=tensor_sha(wn),We_sha=tensor_sha(we),M_sha=tensor_sha(m),MN_sha=tensor_sha(mn),
         native_cache_hits=100,native_compute_z=0,direct_z_influence=0,Q=q_receipt,Ub=ub_receipt,
         native_action=float(jn),native_norm=float(delta.norm()),compute=ledger.receipt()))
    progress(output,'NATIVE_PREPARED',entry=entry,rankB=ub.shape[1],rankC=q.shape[1])
    for name,state in [('W0',w0),('ENTRY',we),('N',wn)]:
        with materialized(w,state,ledger):
            measure(model,evaltok,records,panel,True,ledger,output/f'{name}-full.json')
            save(output/f'{name}-structure.json',structural(state,w0,we,m,c0,k))
            if name=='N':generation(model,evaltok,records,panel,ledger,output/'N-generation.json')
        progress(output,'FULL_EVALUATION_SAVED',entry=entry,endpoint=name)
    for scale in [.25,.5,.75,1.25]:
        state=we+scale*delta
        with materialized(w,state,ledger):
            measure(model,evaltok,records,panel,False,ledger,output/f'native-scale-{scale}-curve.json')
            save(output/f'native-scale-{scale}-structure.json',structural(state,w0,we,m,c0,k))
        progress(output,'NATIVE_SCALING_SAVED',entry=entry,scale=scale)

def direct_run(entry,support,alphas,prepared_path,output,model,tok,evaltok,records,cp,w,w0,ledger):
    receipt=json.loads(prepared_path.with_name('prepared-receipt.json').read_text())
    assert sha(prepared_path)==receipt['sha256']
    data=torch.load(prepared_path,map_location='cpu',weights_only=True,mmap=True)
    assert data['entry']==entry and torch.equal(data['We'],cp['weights'][WEIGHT]) and torch.equal(data['M'],cp['cache_c'])
    panel=select(records,entry);current=[records[i] for i in panel['panels']['Current100']]
    requests=[dict(r['requested_rewrite'],case_id=r['case_id']) for r in current]
    for r in requests:
        if not r['target_new']['str'].startswith(' '):r['target_new']=dict(r['target_new'],str=' '+r['target_new']['str'])
    we=data['We'].cuda();u=data['Ub' if support=='B' else 'Q'].cuda();m=data['M'][0].cuda()
    metric=reduced_metric(u,m);c0=covariance();k=data['K'].cuda()
    with torch.no_grad():w.copy_(we)
    native=kernel()
    objective=DirectObjective(model,tok,requests,data['contexts'],native.find_fact_lookup_idx,we,u,metric,data['J_native'],ledger,
                              accumulate_weight_gradient=True)
    failures=[];all_history={};endpoints={}
    for alpha in alphas:
        candidate=output/f'{support}-alpha-{alpha}'
        candidate.mkdir(parents=True,exist_ok=False)
        a=torch.zeros((we.shape[0],u.shape[1]),device='cuda',dtype=torch.float32,requires_grad=True)
        velocity=None;history=[];path_length=0.;last=we.clone()
        try:
            terms=objective.evaluate(a,True);eta,eta_status=calibrated_eta(alpha,data['native_norm'],a.grad)
            save(candidate/'optimizer.json',dict(alpha=alpha,eta=eta,calibration=eta_status,momentum=.9,weight_decay=0,
                  initial_gradient_norm=float(a.grad.norm()),native_norm=data['native_norm'],J_native=data['J_native'],
                  support=support,rank=u.shape[1],objective_state='post-update Wk',source_z_in_direct=0,
                  essence='KL(student||entry_teacher)',normalization_status=data['normalizer_status'],
                  gradient_accumulation='dense-weight sum then one coefficient pullback; FP32 rounding order disclosed'))
            save(candidate/'step-000.json',dict(step=0,**terms))
            tensor_save(candidate/'snapshot-000.pt',dict(W=we.cpu(),A=a.detach().cpu(),step=0))
            for step in range(1,33):
                with ledger.time('optimizer_update'),torch.no_grad():
                    next_a,velocity=momentum_update(a,a.grad,velocity,eta);a.copy_(next_a)
                    state=we+a@u.T;finite_state(state)
                    actual_step=state-last;stepnorm=float(actual_step.norm());path_length+=stepnorm;last=state.clone()
                terms=objective.evaluate(a,step<32)
                row=dict(step=step,**terms,step_norm=stepnorm,step_squared_norm=float(actual_step.double().square().sum()),
                         path_length=path_length,batch_net_norm=float((state-we).norm()),
                         coefficient_norm=float(a.detach().norm()),selected_weight_sha=tensor_sha(state),compute=ledger.receipt())
                history.append(row);save(candidate/f'step-{step:03d}.json',row)
                if step in [1,2,4,8,16,24,32]:tensor_save(candidate/f'snapshot-{step:03d}.pt',dict(W=state.cpu(),A=a.detach().cpu(),momentum=velocity.cpu(),step=step))
                if step in [4,8,16,24,32]:
                    with materialized(w,state,ledger):
                        measure(model,evaltok,records,panel,step==32,ledger,candidate/f'eval-{step:03d}.json')
                        save(candidate/f'structure-{step:03d}.json',structural(state,w0,we,m,c0,k))
                progress(output,'DIRECT_STEP',entry=entry,support=support,alpha=alpha,step=step,objective=terms['objective'])
                if step==1:save(candidate/'initial-execution.json',dict(input_applied=True,gradient_finite=True,step_weight_changed=stepnorm>0,
                       zero_update_is_observation=True,persistent_training_mutation=0,logical_requests=100,contexts_per_request=len(objective.contexts),
                       next_step_gradient_saved=a.grad is not None,compute=ledger.receipt()))
            tensor_save(candidate/'endpoint.pt',dict(W=state.cpu(),M=data['M'],A=a.detach().cpu(),eta=eta,step=32,
                                                     entry=entry,support=support,alpha=alpha,input_lock_sha=data['input_lock_sha']))
            save(candidate/'terminal.json',dict(status='TERMINAL_VALID',completed_steps=32,finite=True,
                 endpoint_sha256=sha(candidate/'endpoint.pt'),endpoint_exists_verified=True,
                 mean_last_four_objective=sum(r['objective'] for r in history[-4:])/4,compute=ledger.receipt()))
            all_history[alpha]=history
            endpoints[alpha]=json.loads((candidate/'terminal.json').read_text())
        except BaseException as exc:
            failures.append(dict(alpha=alpha,error=repr(exc)))
            save(candidate/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed_steps=len(history),
                    finite_endpoint_eligible=False,restore_entry_exact=torch.equal(w,we),compute=ledger.receipt()))
            if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
        finally:
            with torch.no_grad():w.copy_(we)
            assert torch.equal(w,we)
            del a,velocity
            torch.cuda.empty_cache()
    if failures:save(output/'candidate-failures.json',dict(failures=failures))
    if all_history:
        selected,scores=choose_alpha(all_history,endpoints)
        chosen=output/f'{support}-alpha-{selected}'
        endpoint=torch.load(chosen/'endpoint.pt',map_location='cpu',weights_only=True)
        with materialized(w,endpoint['W'].cuda(),ledger):
            generation(model,evaltok,records,panel,ledger,output/'selected-generation.json')
        save(output/'selection.json',dict(alpha=selected,scores=scores,support=support,entry=entry,
              criterion='mean common post-update objective steps29..32',PS_NS_influence=0,
              endpoint_path=str(chosen/'endpoint.pt'),endpoint_sha=sha(chosen/'endpoint.pt'),eta=endpoint['eta']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['native','direct','B','C'],required=True);p.add_argument('--entry',choices=list(ENTRIES),required=True)
    p.add_argument('--support',choices=['B','C']);p.add_argument('--alpha',type=float,nargs='+');p.add_argument('--prepared',type=Path)
    p.add_argument('--prior-stage-receipt',type=Path);p.add_argument('--selection',type=Path)
    p.add_argument('--repair-r1-covariance',action='store_true')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    output=args.output.absolute();output.mkdir(parents=True,exist_ok=False)
    ledger=Ledger();w=w0=None;stage='INPUT'
    try:
        if args.repair_r1_covariance:
            assert args.mode=='native'
            from .covariance_repair import repair
            repair(ROOT/'A/Middle/native-r1',[ROOT/'A/Middle/direct-B-r1',ROOT/'A/Middle/direct-C-r2'],output/'Middle-structure-diagnostic-repair')
        if args.mode in ['B','C']:
            from .later_stages import require_previous
            require_previous(args.prior_stage_receipt,'A' if args.mode=='B' else 'B')
        lock=json.loads((ROOT/'input.lock.json').read_text());records=load_prefix(DATA,10000)
        cp,targets,current=load_entry(args.entry,records)
        model,tok,evaltok=load_model(ledger);w=dict(model.named_parameters())[WEIGHT];w0=w.detach().clone()
        original_pointer=w.data_ptr()
        def forward_counter(module,args,kwargs):
            ledger.add('actual_model_forward_invocations')
            ids=kwargs.get('input_ids')
            if ids is not None:
                ledger.add('actual_model_forward_sequences',ids.shape[0])
                mask=kwargs.get('attention_mask')
                from .token_accounting import count_tokens
                count_tokens(ledger,ids,mask)
        model.register_forward_pre_hook(forward_counter,with_kwargs=True)
        restore_rng(cp)
        save(output/'runtime.json',dict(arguments={k:str(v) for k,v in vars(args).items()},input_lock_sha=sha(ROOT/'input.lock.json'),
                  source_head=os.environ.get('CUMRISK_SOURCE_HEAD'),model_revision=Path(MODEL).name,
                  parameter_dtype_counts={'torch.float32':len(list(model.parameters()))},
                  autocast=torch.is_autocast_enabled(),attention=model.config._attn_implementation,
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  W0_sha=tensor_sha(w0),GPU=torch.cuda.get_device_name(),slurm_job=os.environ.get('SLURM_JOB_ID'),
                  original_cuda_rng_preserved_in_checkpoint=True,runtime_cuda_seed=20260907,
                  stochastic_direct_paths=0,compute=ledger.receipt(),scientific_promotion=False))
        progress(output,'MODEL_ENTRY_BOUND',entry=args.entry,mode=args.mode)
        stage=args.mode
        if args.mode=='native':native_run(args.entry,output,model,tok,evaltok,records,cp,targets,current,w,w0,ledger)
        elif args.mode=='direct':direct_run(args.entry,args.support,args.alpha,args.prepared,output,model,tok,evaltok,records,cp,w,w0,ledger)
        elif args.mode=='B':
            from .later_stages import run_b
            run_b(args.entry,args.prepared,output,model,tok,evaltok,records,cp,w,w0,ledger)
        else:
            from .later_stages import run_c
            run_c(args.entry,args.prepared,args.selection,output,model,tok,evaltok,records,cp,w,w0,ledger)
        with torch.no_grad():w.copy_(w0)
        assert torch.equal(w,w0) and w.data_ptr()==original_pointer
        save(output/'terminal.json',dict(status='TERMINAL_VALID' if not (output/'candidate-failures.json').exists() else 'PARTIAL_TECHNICAL',
             stage=stage,W0_restored=True,compute=ledger.receipt(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
             peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),scientific_promotion=False))
    except BaseException as exc:
        restored=None
        if w is not None and w0 is not None:
            with torch.no_grad():w.copy_(w0)
            restored=torch.equal(w,w0)
        save(output/'failure.json',dict(stage=stage,error=repr(exc),traceback=traceback.format_exc(),W0_restored=restored,compute=ledger.receipt()))
        raise

if __name__=='__main__':main()
