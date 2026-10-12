"""Rescue-only observation/quality exit. Native writer function remains unchanged."""
import math
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from official.experiments.prepare import write_new
from official.runners.server1.common import read,member,require
from official.diagnostics.fe_qwen_current import observer as base
from official.diagnostics.fe_qwen_current.rescue import SPEC,tracking_config,quality_stop

class QualityEarlyStop(RuntimeError):pass


def snapshot(cfg,model,tok,H,contexts,batch,requests):
    with patch.object(base,'SPEC',SPEC),patch.object(base,'tracking_config',tracking_config):
        base.snapshot(cfg,model,tok,H,contexts,batch,requests)
    if batch==0 or base._STATE['c'].get('stage')=='final_benchmark':return
    s=base._STATE;raw=s['out']/'factual'/f'B{batch:02d}-current.json'
    failed=quality_stop(read(raw)['summary'])
    if not failed:return
    receipt=dict(status='QUALITY_EARLY_STOP',not_numerical_failure=True,identity=s['identity'],
        last_committed_batch=batch,current=member(raw),thresholds=SPEC['early_stop'],failed=failed,
        W20_complete=False,source_preserved=True,last_scheduled_checkpoint_keep=True)
    write_new(s['out']/'QUALITY_EARLY_STOP.json',receipt)
    # Avoid the generic failure hook relabeling an intentional quality stop.
    s['finished']=True;s['tracker'].finish(exit_code=2,timeout=45)
    sys.excepthook=sys.__excepthook__
    raise QualityEarlyStop('QUALITY_EARLY_STOP')


def norm(tensor,cast=None):
    value=tensor.detach();squared=0.0
    # Slice before any conversion: transposed native deltas must not trigger a
    # full contiguous reshape/FP64 temporary just for an observation.
    width=math.prod(value.shape[1:]) if value.ndim>1 else 1
    step=max(1,262144//width)
    if value.ndim==0:value=value.reshape(1)
    for start in range(0,value.shape[0],step):
        part=value[start:start+step]
        if cast is not None:part=part.to(cast)
        squared+=float(torch.sum(part.double().square()).item())
    return math.sqrt(squared)


def solve_diagnostics(A,B,X):
    """Four fixed RHS columns, row-chunked residual; NOT a full solve residual.

    Uses the actual native matrix/RHS/solution while alive, no reconstruction,
    decomposition, eigenscan, extra solve, RNG or model forward.
    """
    cols=sorted(set([0,B.shape[1]//3,2*B.shape[1]//3,B.shape[1]-1]))
    selected=X.detach()[:,cols];err=0.0;rhs=0.0
    for start in range(0,A.shape[0],128):
        b=B.detach()[start:start+128,cols]
        residual=A.detach()[start:start+128]@selected-b
        err+=float(residual.double().square().sum().item());rhs+=float(b.double().square().sum().item())
    return dict(rhs_frobenius=norm(B),solve_residual_sampled_columns=cols,
        solve_relative_residual_sampled=math.sqrt(err/rhs) if rhs else None,
        solve_absolute_residual_sampled=math.sqrt(err),full_solve_residual_measured=False)


@base.record_failure
def observed_write(writer,cfg,model,tok,requests,device,H,zs_all,batch):
    s=base._STATE
    require(s is not None and s['last_batch']==batch-1,'WRITER_ENTRY_CURSOR')
    require(len(requests)==100 and all(v.ndim==2 and v.shape[1]==100 for v in zs_all.values()),'BATCH_ENTRY_TARGET_SHAPE')
    require(all(torch.isfinite(v).all().item() for v in zs_all.values()),'NONFINITE_CURRENT_BATCH_TARGETS')
    require(sys.gettrace() is None,'EXISTING_TRACE_NOT_REPLACED')
    target_rows={}
    for layer in (4,5):
        values=[norm(zs_all[layer][:,i]) for i in range(100)]
        target_rows[str(layer)]=dict(column_norms=values,
            top_local_indices=sorted(range(100),key=lambda i:(-values[i],i))[:10])
    observations=[];seen=set();solves=[];native_solve=torch.linalg.solve
    def solve(A,B,*args,**kwargs):
        result=native_solve(A,B,*args,**kwargs)
        solves.append(solve_diagnostics(A,B,result))
        return result
    def trace(frame,event,arg):
        if frame.f_code is not writer.__code__:return None
        v=frame.f_locals
        if event=='line' and all(k in v for k in ('layer','upd_matrix','weight_name','weights','targets','layer_ks')):
            layer=int(v['layer']);name=cfg.llms.rewrite_module_tmp.format(layer)+'.weight'
            if v['weight_name']==name and layer not in seen:
                seen.add(layer)
                require(len(solves)==len(seen),'ONE_NATIVE_SOLVE_PER_LAYER')
                values=dict(layer=layer,weight_name=name,pre_weight_norm=norm(v['weights'][name]),
                    key_frobenius=norm(v['layer_ks']),target_norm=norm(zs_all[layer]),
                    residual_frobenius=norm(v['targets']),raw_delta_frobenius=norm(v['upd_matrix']),
                    bf16_cast_delta_frobenius=norm(v['upd_matrix'],torch.bfloat16),**solves[-1])
                require(all(math.isfinite(x) for x in values.values() if type(x) is float),'NONFINITE_WRITER_OBSERVATION')
                observations.append(values)
        return trace
    with patch.object(torch.linalg,'solve',solve):
        sys.settrace(trace)
        try:writer(cfg,model,tok,requests,device,H,zs_all)
        finally:sys.settrace(None)
    require(seen==set(SPEC['layers']),'LAYER_OBSERVATION_COVERAGE')
    parameters=dict(model.named_parameters())
    for row in observations:row['post_weight_norm']=norm(parameters[row['weight_name']])
    write_new(s['out']/'writer'/f'B{batch:02d}.json',dict(batch=batch,layers=observations,L2=100,
        entry_targets=target_rows,local_indices_are_current_request_order=True,
        author_hparam_tuned=True,native_writer_AST_unchanged=True,extra_forward=0,
        diagnostic_scope='actual_native_solve_rhs_four_columns_and_existing_tensors',
        fit_trace='existing_native_stdout_per_case_init_delta_clamp_lookup_probability'))


observed_targets=base.observed_targets
finish=base.finish
