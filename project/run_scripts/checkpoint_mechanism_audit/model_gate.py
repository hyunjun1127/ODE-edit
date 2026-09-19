"""Actual physical-model C00/C01 gate; no editing optimizer or saved weights."""
import argparse
import gc
import os
import time
import traceback
from pathlib import Path
import torch
from .common import *
from . import model_runtime as rt
from . import validation as val
from .operators import native_dense


def tensor_artifact(path,value):
    # Analysis key/h0/factor artifacts ONLY, never a restorable W/M bundle.
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f: torch.save(value,f)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path))


def compare_evaluation(new,old):
    out={}
    for tag in ('RS','PS','NS'):
        a=new['metrics'][tag]['rows'];b=old['metrics'][tag]['rows']
        assert [r['identity'] for r in a]==[r['identity'] for r in b], 'EVAL_IDENTITY_ORDER'
        out[tag]=val.evaluation_rows_gate(
            torch.tensor([r['new_nll'] for r in a],dtype=torch.float64),
            torch.tensor([r['true_nll'] for r in a],dtype=torch.float64),
            torch.tensor([r['new_nll'] for r in b],dtype=torch.float64),
            torch.tensor([r['true_nll'] for r in b],dtype=torch.float64),[tag]*len(a))
    return out


def repeat_norm(error,reference,relative_tolerance,columnwise=False):
    from .operators import norm_discrepancy
    r=norm_discrepancy(error,reference,relative_tolerance,columnwise=columnwise)
    # The absolute near-zero rule must shrink by the same 10% repeat fraction.
    r['zero_reference_absolute_atol']=1e-8
    r['failed_indices']=[i for i,(e,n) in enumerate(zip(r['absolute_errors'],r['reference_norms']))
        if e>(1e-8 if n<=1e-12 else relative_tolerance*n)]
    r['passed']=not r['failed_indices']
    return r


def gate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();stage='SOURCE';results={};model=None;w0=None
    try:
        bindings=rt.bind_sources();rows,probe=rt.stream_and_probe();mapping=source_map()
        contexts=read(mapped(S4CELL+'/B001/contexts.json',mapping))
        assert [len(g) for g in contexts]==[1,5]
        stage='LOAD';model,writer,evaltok,w0,runtime=rt.load();nonselected=rt.pointer_versions(model)
        write_json(output/'runtime.json',runtime)
        stage='C00_PREFIX'
        for label,selected in [('B001',rows[:100]),('B002',rows[100:200]),('geometry32',probe[:32])]:
            full=rt.capture(model,writer,selected,contexts,bindings,early=False)
            early=rt.capture(model,writer,selected,contexts,bindings,early=True)
            repeat=rt.capture(model,writer,selected,contexts,bindings,early=True)
            local={}
            for key in ('K','bare_K','h0'):
                local[key]=val.elementwise_gate(early[key],full[key])
                local['repeat_'+key]=val.repeat_spread_gate(torch.stack([early[key],repeat[key]]),
                    1e-6+1e-5*early[key].abs())
            results[label]=local
            tensor_artifact(output/(label+'-keys.pt'),early)
            print('C00_CAPTURE',label,{k:v['passed'] for k,v in local.items()},flush=True)
            del full,early,repeat
        c00=all(v['passed'] for r in results.values() for v in r.values())
        write_json(output/'C00.json',dict(status='PASS' if c00 else 'FAILED',comparisons=results))
        # Independent physical endpoint evaluation proceeds even if operator
        # reconstruction fails; it is not silently counted as operator PASS.
        stage='W0_EVALUATION';ev0=rt.evaluate(model,evaltok,rows[:100],bindings)
        write_json(output/'W0-first100.json',ev0)
        cp=rt.checkpoint(1);w1=cp['weights'][WEIGHT];m1=cp['cache_c'][0]
        capture=torch.load(output/'B001-keys.pt',weights_only=True,map_location='cpu')
        k=capture['K'];bare=capture['bare_K'];h0=capture['h0']
        stage='B1_PHYSICAL_RESTORE';rt.set_weight(model,w1)
        h1=rt.capture(model,writer,rows[:100],contexts,bindings,early=True)
        delta=w1.double()-w0.double()
        affine=(h0.double()+delta@bare.double())
        reconstruction={'keys_invariant':val.elementwise_gate(h1['K'],k),
            'physical_block_affine':val.elementwise_gate(h1['h0'],affine),
            'M1_gram':val.elementwise_gate(k@k.T,m1)}
        stage='B1_EVALUATION';ev1=rt.evaluate(model,evaltok,rows[:100],bindings)
        write_json(output/'B1-first100.json',ev1)
        repeat_eval=rt.evaluate(model,evaltok,rows[:100],bindings)
        archived=read(mapped(S4CELL+'/B001/current.json',mapping))
        eval_gate=compare_evaluation(ev1,archived)
        eval_repeat={}
        for tag in ('RS','PS','NS'):
            assert ev1['metrics'][tag]['denominator']=={'RS':100,'PS':200,'NS':1000}[tag]
            x=torch.tensor([[v for r in z['metrics'][tag]['rows'] for v in (r['new_nll'],r['true_nll'],r['true_nll']-r['new_nll'])]
                            for z in (ev1,repeat_eval)],dtype=torch.float64)
            eval_repeat[tag]=val.repeat_spread_gate(x,1e-4)
        write_json(output/'evaluation-parity.json',dict(archive=eval_gate,repeated_rows=eval_repeat,
            expected_counts={'RS':100,'PS':190,'NS':867},observed_counts={t:ev1['metrics'][t]['numerator'] for t in eval_gate}))
        del ev0,ev1,repeat_eval,h1;gc.collect();torch.cuda.empty_cache()
        stage='B1_DENSE'
        p_all=torch.load(CONTRACT['paths']['projector'],map_location='cpu',weights_only=True,mmap=True)
        p=p_all[0].contiguous();assert tensor_sha(p)==CONTRACT['identity']['projector_selected_sha256']
        targets=torch.load(mapped(S4CELL+'/B001/native-targets.pt',mapping),weights_only=True,map_location='cpu')
        assert [x['case_id'] for x in targets['identities']]==[r['case_id'] for r in rows[:100]]
        assert [tensor_sha(x) for x in targets['values']]==[x['sha256'] for x in targets['identities']]
        z=torch.stack(targets['values'],1);residual=z-h0
        pg=p.cuda();kg=k.cuda();rg=residual.cuda();mg=torch.zeros_like(pg)
        reconstructed,solve=native_dense(pg,mg,kg,rg)
        reconstructed_cpu=reconstructed.cpu()
        reconstruction['dense_solve']=solve
        reconstruction['actual_delta']=val.update_gate(reconstructed_cpu,delta)
        reconstruction['actual_response']=val.response_gate(reconstructed_cpu,delta,k)
        # A repeat is a genuine same-operand solve; use tensor differences and
        # the same actual-delta/request-response denominators at 10% thresholds.
        repeated,repeat_solve=native_dense(pg,mg,kg,rg)
        diff=repeated.cpu().double()-reconstructed_cpu.double()
        reconstruction['repeat_dense_delta']=repeat_norm(diff,delta,1e-4)
        reconstruction['repeat_dense_response']=repeat_norm(diff@k.double(),delta@k.double(),1e-4,columnwise=True)
        if torch.equal(repeated,reconstructed):
            reconstruction['repeat_residual_vector']=dict(passed=True,exact_repeated_solution=True,residual_vector_spread=0.0)
        else:
            matrix=pg@(kg@kg.T+mg)+torch.eye(pg.shape[0],device=pg.device,dtype=pg.dtype)
            rhs=(pg@kg)@rg.T
            reconstruction['repeat_residual_vector']=repeat_norm(matrix.double()@diff.T.cuda(),rhs.double(),1e-6,columnwise=True)
            del matrix,rhs
        reconstruction['repeat_dense_solve']=repeat_solve
        write_json(output/'reconstruction-parity.json',reconstruction)
        del pg,kg,rg,mg,reconstructed,repeated;gc.collect();torch.cuda.empty_cache()
        # No new W/M/Delta checkpoint is persisted. Only keys/h0/targets binding.
        rt.set_weight(model,w0)
        assert tensor_sha(model.get_parameter(WEIGHT))==tensor_sha(w0)
        assert nonselected==rt.pointer_versions(model)
        checks=[c00,*[r['passed'] for r in eval_gate.values()],*[r['passed'] for r in eval_repeat.values()]]
        checks += [reconstruction[k]['passed'] for k in ('keys_invariant','physical_block_affine','M1_gram','actual_delta','actual_response','repeat_dense_delta','repeat_dense_response','repeat_residual_vector')]
        checks += [solve['residual']['passed'],repeat_solve['residual']['passed']]
        passed=all(checks)
        result=dict(status='PASS' if passed else 'FAILED',failure_type=None if passed else 'NUMERICAL_CONTRACT_UNRESOLVED',
            C00='PASS' if c00 else 'FAILED',C01='PASS' if passed else 'FAILED',
            W0_restore=True,nonselected_pointer_versions=True,nonselected_full_bytes='NOT_CLAIMED',
            checkpoint_saved=False,z_optimization_calls=0,history_append=0,
            elapsed_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            source_map_sha256=sha256(ATTEMPT/'inputs/source-map.json'),
            members=[dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file()])
        write_json(output/'terminal.json',result)
        print('GATE_TERMINAL',result['status'],result['elapsed_seconds'],flush=True)
    except BaseException as e:
        restored=None
        if model is not None and w0 is not None:
            try: rt.set_weight(model,w0);restored=tensor_sha(model.get_parameter(WEIGHT))==tensor_sha(w0)
            except BaseException: restored=False
        write_json(output/'failure.json',dict(status='FAILED_TECHNICAL',stage=stage,error=repr(e),
            traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start,W0_restore=restored,
            allocated_cost_must_be_preserved=True,z_optimization_calls=0,history_append=0))
        raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);args=a.parse_args();gate(args.output)
