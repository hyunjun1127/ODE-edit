"""Read-only observations around the unchanged author writer; latest-only storage.

No target computation, solve, coefficient modification or forward is added to
the editing path. Common evaluation is observational and restores editing RNG.
"""
import atexit
import json
import math
import os
import sys
import time
import functools
from pathlib import Path

import torch
from omegaconf import OmegaConf
from official.experiments.checkpoint import rng_snapshot, rng_restore
from official.experiments.prepare import digest, write_new
from official.runners.server1.common import read, verify, member, factual_payload
from official.runners.fe_original import cf_payload, versions
from official.diagnostics.fe_qwen_current.storage import save_latest, finite_tensor
from official.evaluation.factual import evaluate_counterfact, _plan, validate_retained_observation
from official.tracking import init
from official.tracking.schema import bind_job_identity, job_identity
from official.diagnostics.fe_qwen_current.contract import SPEC, tracking_config

_STATE = None
NUMERIC_CODES={'NONFINITE_SELECTED_WEIGHTS','NONFINITE_NATIVE_HISTORY','NONFINITE_WRITER_OBSERVATION',
               'NONFINITE_CURRENT_BATCH_TARGETS','NONFINITE_FIRST_LAYER_FIT',
               'FACTUAL_NONFINITE_LOGITS','FACTUAL_NONFINITE_NLL'}

def failure_hook(kind, error, tb):
    if _STATE is not None:
        code=str(error)
        write_new(_STATE['out']/'FAILURE.json',dict(status='NUMERICAL_FAILURE' if code in NUMERIC_CODES else 'TECHNICAL_FAILURE',
            error_type=kind.__name__,code=code,identity=_STATE['identity'],last_committed_batch=_STATE['last_batch'],
            W20_complete=False,source_preserved=True))
    sys.__excepthook__(kind,error,tb)

def record_failure(function):
    @functools.wraps(function)
    def call(*args,**kwargs):
        try:return function(*args,**kwargs)
        except Exception as error:
            # Hydra may render an exception without calling sys.excepthook.
            if _STATE is not None and not (_STATE['out']/'FAILURE.json').exists():
                code=str(error)
                write_new(_STATE['out']/'FAILURE.json',dict(status='NUMERICAL_FAILURE' if code in NUMERIC_CODES else 'TECHNICAL_FAILURE',
                    error_type=type(error).__name__,code=code,identity=_STATE['identity'],
                    last_committed_batch=_STATE['last_batch'],W20_complete=False,source_preserved=True))
            raise
    return call

def model_signature(model):
    return dict(parameters=versions(model),
        buffers={n:(v._version,tuple(v.shape),str(v.dtype)) for n,v in model.named_buffers()},
        modules={n:(m.training,tuple(m._forward_hooks),tuple(m._forward_pre_hooks))
                 for n,m in model.named_modules()})

def preserved_evaluation(model,H,callback):
    before=model_signature(model);h_version=H._version;rng=rng_snapshot()
    try:return callback()
    finally:
        rng_restore(rng)
        if before!=model_signature(model) or H._version!=h_version:
            raise ValueError('EVALUATION_MUTATED_EDITING_STATE')

def _state(cfg,model,tok,requests):
    global _STATE
    if _STATE is not None:return _STATE
    c=read(cfg.working_config)
    if c['config_sha256']!=digest({k:v for k,v in c.items() if k!='config_sha256'}):
        raise ValueError('CONFIG_HASH')
    if c['selection_contract_sha256']!=digest(SPEC) or c['L2'] not in SPEC['L2_arms']:
        raise ValueError('SELECTION_CONTRACT')
    if cfg.get('diagnostic_z_reference','batch_entry')!='batch_entry':raise ValueError('W0_FALLBACK_FORBIDDEN')
    expected=dict(clamp_norm_factor=1,v_num_grad_steps=35,v_lr=.5,v_weight_decay=.001,
                  kl_factor=.0625,v_loss_layer=27)
    if (any(cfg.llms[k]!=v for k,v in expected.items()) or list(cfg.llms.layers)!=SPEC['layers']
        or list(cfg.llms.mom2_update_weight)!=[15000]*5 or cfg.algs.L2!=c['L2']
        or not cfg.algs.add_old_keys or cfg.model_dtype!='bfloat16'
        or (cfg.bs,cfg.num_edits,cfg.seed)!=(100,2000,0)):
        raise ValueError('RESOLVED_AUTHOR_SETTINGS')
    rows=read(verify(c['stream']))
    if len(rows)!=2000 or [r['case_id'] for r in rows]!=[r['case_id'] for r in requests]:
        raise ValueError('NATIVE_COMMON_STREAM_ORDER')
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'runtime.json').exists():raise ValueError('EXISTING_ATTEMPT_NOT_RESUMED_OR_OVERWRITTEN')
    lock=read(os.environ['FE_WORKING_LOCK'])
    if not Path(__file__).resolve().is_relative_to(Path(lock['source_directory']).resolve()):
        raise ValueError('FROZEN_OBSERVER_PATH')
    values=bind_job_identity(tracking_config(c,lock['source_commit']))
    job=job_identity(values)
    if job['execution_backend']=='local':raise ValueError('ACTUAL_EXECUTION_JOB_REQUIRED')
    tracker=init(env_file=c['tracking_env_file'],spool=out/'tracking',config=values)
    identity=dict(config_sha256=c['config_sha256'],source=lock['source_commit'],
        author=SPEC['author_commit'],stream_sha256=c['stream']['sha256'],
        split='heldout_fixed10k_4000_6000',selection_contract_sha256=digest(SPEC),
        z_reference='batch_entry',L2=c['L2'],dtype='bfloat16',job_id=job['job_id'],job_identity=job)
    names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
    _STATE=dict(c=c,rows=rows,out=out,tracker=tracker,identity=identity,values=values,
                model=model,tok=tok,excluded=names,nonedited=versions(model,names),last_batch=-1,finished=False)
    write_new(out/'runtime.json',dict(identity=identity,resolved=OmegaConf.to_container(cfg,resolve=True),
        torch=torch.__version__,transformers=__import__('transformers').__version__,
        attention_backend=model.config._attn_implementation,tracking_config=values,
        native_writer_unchanged=True,full_model_export=False,checkpoint='latest_only_no_W0'))
    atexit.register(_finish_error)
    sys.excepthook=failure_hook
    return _STATE

def _finish_error():
    if _STATE is not None and not _STATE['finished']:
        _STATE['tracker'].finish(exit_code=1,timeout=45)

@record_failure
def snapshot(cfg,model,tok,cache_c,contexts,batch,requests):
    s=_state(cfg,model,tok,requests)
    if batch!=s['last_batch']+1 or not 0<=batch<=20:raise ValueError('COMMIT_CURSOR')
    if versions(model,s['excluded'])!=s['nonedited']:raise ValueError('NONEDITED_PARAMETER_CHANGED')
    parameters=dict(model.named_parameters())
    if any(not torch.isfinite(parameters[n]).all().item() for n in s['excluded']):
        raise ValueError('NONFINITE_SELECTED_WEIGHTS')
    if not finite_tensor(cache_c):raise ValueError('NONFINITE_NATIVE_HISTORY')
    if batch==0:
        write_new(s['out']/'contexts.json',dict(contexts=contexts,sha256=digest(contexts)))
    elif digest(contexts)!=read(s['out']/'contexts.json')['sha256']:
        raise ValueError('CONTEXT_CHANGED')
    endpoints=[('W0_first2000',s['rows'],'W0')] if batch==0 else [
        ('current/post',s['rows'][(batch-1)*100:batch*100],f'B{batch:02d}-current')]
    if batch in SPEC['milestones']:
        endpoints.append(('all_seen/post',s['rows'][:batch*100],f'B{batch:02d}-all-seen'))
    receipts=[]
    for endpoint,rows,label in endpoints:
        started=time.monotonic()
        result=preserved_evaluation(model,cache_c,lambda:evaluate_counterfact(
            model,tok,rows,identity=dict(s['identity'],endpoint=endpoint,edits=batch*100),
            batch_size=16,device=str(next(model.parameters()).device)))
        if len(result['cases'])!=len(rows) or result['summary']['requests']!=len(rows):
            raise ValueError('ENDPOINT_COVERAGE')
        path=s['out']/'factual'/f'{label}.json';write_new(path,result);receipts.append(member(path))
        # CPU-only tokenizer query plan retained for an independent post-run reducer.
        signatures=_plan(rows,'cf',tok)[2]
        querypath=s['out']/'queries'/f'{label}.json';write_new(querypath,signatures)
        audit=validate_retained_observation(result,'cf',input_signature=signatures)
        write_new(s['out']/'raw-audits'/f'{label}.json',audit)
        payload=factual_payload(result,endpoint,0) if batch==0 else cf_payload(result,endpoint,batch*100)
        payload['batch']=batch
        if s['tracker'].log(payload) is False:raise ValueError('TRACKING_NOT_ACCEPTED')
        write_new(s['out']/'evaluations'/f'{label}.json',dict(raw=receipts[-1],elapsed_sec=time.monotonic()-started,
            endpoint=endpoint,requests=len(rows),batch=batch,identity=s['identity'],queries=member(querypath)))
    checkpoint=None
    if batch in SPEC['milestones']:
        checkpoint=preserved_evaluation(model,cache_c,lambda:save_latest(s['out']/'checkpoints',
            {n:parameters[n] for n in s['excluded']},cache_c,contexts,
            dict(policy='batch_entry_current100',durable_target_cache=False),batch,s['identity'],
            dict(edits=batch*100,case_ids=[r['case_id'] for r in s['rows'][:batch*100]],
                 raw=receipts,native_config=OmegaConf.to_container(cfg,resolve=True)),
            lock_path=s['c']['checkpoint_lock'],reserve_bytes=s['c']['reserve_bytes']))
    write_new(s['out']/'commits'/f'B{batch:02d}.json',dict(batch=batch,edits=batch*100,
        raw=receipts,checkpoint=checkpoint,identity=s['identity'],nonedited_guard=True))
    s['last_batch']=batch

@record_failure
def observed_targets(supplier,model,tok,requests,cfg,contexts,**kwargs):
    fit=kwargs['compute_z'];fits=[]
    def observed_fit(*args,**kw):
        value=fit(*args,**kw)
        if not torch.isfinite(value).all().item():raise ValueError('NONFINITE_FIRST_LAYER_FIT')
        fits.append(dict(target_norm=float(torch.linalg.vector_norm(value.detach().float()).item()),finite=True))
        return value
    result=supplier(model,tok,requests,cfg,contexts,**dict(kwargs,compute_z=observed_fit))
    if len(fits)!=100:raise ValueError('NATIVE_FIT_COUNT')
    write_new(_STATE['out']/'fits'/f"B{_STATE['last_batch']+1:02d}.json",dict(
        first_layer=4,requests=100,reference='batch_entry',rows=fits,later_layers='same_entry_firstforward'))
    return result

@record_failure
def observed_write(writer,cfg,model,tok,requests,device,cache_c,zs_all,batch):
    if _STATE is None or _STATE['last_batch']!=batch-1:raise ValueError('WRITER_ENTRY_CURSOR')
    if len(requests)!=100 or any(v.ndim!=2 or v.shape[1]!=100 for v in zs_all.values()):
        raise ValueError('BATCH_ENTRY_TARGET_SHAPE')
    if any(not torch.isfinite(v).all().item() for v in zs_all.values()):
        raise ValueError('NONFINITE_CURRENT_BATCH_TARGETS')
    if sys.gettrace() is not None:raise ValueError('EXISTING_TRACE_NOT_REPLACED')
    observations=[];seen=set();started=time.monotonic()
    def trace(frame,event,arg):
        if frame.f_code is not writer.__code__:return None
        values=frame.f_locals
        # At the native print("upd norm", ...) statement, before weight assignment.
        if event=='line' and all(k in values for k in ('layer','upd_matrix','weight_name','weights','targets')):
            layer=int(values['layer'])
            # Locals retain the previous layer's update until weight_name is reassigned.
            name=cfg.llms.rewrite_module_tmp.format(layer)+'.weight'
            if values['weight_name']==name and layer not in seen:
                seen.add(layer)
                scalars=dict(layer=layer,weight_norm=float(torch.linalg.vector_norm(values['weights'][name].detach().double()).item()),
                    update_norm=float(torch.linalg.vector_norm(values['upd_matrix'].detach()).item()),
                    residual_norm=float(torch.linalg.vector_norm(values['targets'].detach()).item()),
                    target_norm=float(torch.linalg.vector_norm(zs_all[layer].detach().float()).item()))
                if not all(math.isfinite(v) for v in scalars.values()):raise ValueError('NONFINITE_WRITER_OBSERVATION')
                observations.append(scalars)
        return trace
    sys.settrace(trace)
    try:writer(cfg,model,tok,requests,device,cache_c,zs_all)
    finally:sys.settrace(None)
    if seen!=set(SPEC['layers']):raise ValueError('LAYER_OBSERVATION_COVERAGE')
    write_new(_STATE['out']/'writer'/f'B{batch:02d}.json',dict(batch=batch,layers=observations,
        fit=dict(requests=100,first_layer=4,later_layers='native_firstforward',finite_targets=True,
                 z_reference='batch_entry',solver_status='returned',native_fit_trace='local_stdout'),
        elapsed_sec=time.monotonic()-started,L2=float(cfg.algs.L2)))

def finish(cfg):
    s=_STATE
    if s is None or s['last_batch']!=20:raise ValueError('W20_NOT_COMMITTED')
    write_new(s['out']/'COMPLETE.json',dict(status='W20_COMPLETE',edits=2000,batches=20,
        identity=s['identity'],checkpoint=member(s['out']/'checkpoints/latest.pt'),
        endpoint=member(s['out']/'factual/B20-all-seen.json'),scientific_recovery_claim=False))
    s['finished']=True
    s['tracker'].finish(exit_code=0,timeout=45)
