"""Integrated BS100 preflight + one cold first1000 ten-batch RAM chain."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_pilot.prompts import prepare
from project.run_scripts.jlz_pilot.run import (LAYERS, MODEL, STATS, capture_entry,
    capture_keys, geometry, file_sha, require, serial)
from .policy import solve
from .oracle import Oracle, actual_preflight
from .state import Transaction, state_hash, weights, digest, tensor_sha
from .observation import observe

CONTRACT = Path(__file__).resolve().parents[3]/'plans/global/2026-10-01-jlz-sequential-bs100x10-v1/contract.json'

def write_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x') as f:
        json.dump(serial(data),f,ensure_ascii=False,sort_keys=True,allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def reconcile_ledger(out, ledger):
    """Exclude a rolled-back candidate even after ambiguous publication IO."""
    try:
        write_json(out/'ledger.json',ledger)
        return json.loads((out/'ledger.json').read_text())==serial(ledger)
    except Exception:
        return False

def batch(model,tok,records,all_records,contexts,history,ledger,number,out,oracle_route):
    started=time.monotonic()
    txn=Transaction(model,history,contexts,ledger)
    row={'batch':number,'entry':txn.before,'case_ids':[r['case_id'] for r in records],
         'geometry':[],'commit':False,'history_appends':0}
    if ledger: require(row['entry']==ledger[-1]['post'],'CHAIN_LINK_MISMATCH')
    write_json(out/f'B{number:03d}-entry.json',row)
    if number==2:
        write_json(out/'INITIAL_VALID.json',{'status':'B1_COMMIT_5_APPEND_OBSERVER_B2_ENTRY',
            'B1_receipt_sha256':file_sha(out/'B001-committed.json'),
            'B2_entry':row['entry'],'same_as_B1_post':True,'not_full_ten_complete':True,
            'ledger_sha256':file_sha(out/'ledger.json')})
    oracle=None; before_calls=0; science_started=False
    try:
        with txn:
            t=time.monotonic()
            requests=[r['requested_rewrite']|{'case_id':r['case_id']} for r in records]
            spec=prepare(tok,requests,contexts,'cuda')
            require(spec['tokens']['input_ids'].shape[0]==700,'BS100_SPEC_REQUIRED')
            require(spec['tokens']['input_ids'].shape[1]<=model.config.max_position_embeddings,'LENGTH_OVERFLOW')
            anchors,teacher,nll=capture_entry(model,spec,2)
            active=nll>=.05
            keys=capture_keys(model,spec,contexts,2)
            row.update(input_identity=spec['identity'],entry_nll=nll.cpu().tolist(),active=active.cpu().tolist(),
                       teacher_sha256=tensor_sha(teacher),key_sha256={str(l):tensor_sha(k) for l,k in keys.items()},
                       context_sha256=digest(contexts),capture_seconds=time.monotonic()-t)
            adj=geometry(keys,history,row['geometry'],float('inf'))
            for l in LAYERS:
                require(row['geometry'][LAYERS.index(l)]['count']>0,'COVARIANCE_COUNT_INVALID')
            oracle=Oracle(model,spec,teacher,keys,adj,active,2)
            x0=torch.zeros(500,model.config.hidden_size,device='cuda')
            norms=torch.cat([anchors[l].norm(dim=0) for l in LAYERS])
            require(bool(torch.isfinite(norms).all()) and bool((norms>0).all()),'ANCHOR_INVALID')
            c,rho,mask=.5/norms.square(),.75*norms,active.repeat(5)
            if number==1:
                row['actual_BS100_preflight']=actual_preflight(oracle,x0,rho,mask)
                oracle_route=oracle.route
                write_json(out/'actual-preflight.json',row['actual_BS100_preflight'])
            else: oracle.route=oracle_route
            before_calls=oracle.calls; fit_start=time.monotonic()
            science_started=True
            result=solve(oracle,x0,c,rho,mask,cap=120,tol=1e-4)
            payload=result.pop('final_payload'); x=result.pop('x'); grad=result.pop('gradient')
            require(oracle.calls-before_calls==result['calls']<=120,'CALL_ACCOUNTING_FAILED')
            row['solver']=result; row['fit_seconds']=time.monotonic()-fit_start
            row['fit']={k:payload[k] for k in ('nll','kl')}
            row['oracle']={'whole_batch_science':result['calls'],'whole_batch_preflight':before_calls,
                           'forward_chunks':oracle.forward_calls,'backward_chunks':oracle.backward_calls,
                           'new_route_token_work':oracle.token_work,'selected_route':oracle_route,
                           'materializations_new_route':oracle.materializations}
            row['initial_gradient_price_ratio']=(oracle.initial_gradient.norm(dim=1)/c).cpu().tolist()
            row['block_norm_over_radius']=(x.norm(dim=1)/rho).cpu().tolist()
            row['layer_allocation']=[]
            for i,l in enumerate(LAYERS):
                R=x[i*100:(i+1)*100].T
                realized=R.double()@(adj[l].T@keys[l].double())
                row['layer_allocation'].append({'layer':l,'R_norm':float(R.norm()),
                    'realized_norm':float(realized.norm()),'nonzero_blocks':int((R.norm(dim=0)>0).sum()),
                    'physical_delta_norm':float((payload['weights'][l]-oracle.entry_w[l]).norm())})
            t=time.monotonic()
            with torch.no_grad():
                for l,w in weights(model).items(): w.copy_(payload['weights'][l])
            require(all(torch.equal(w,payload['weights'][l]) for l,w in weights(model).items()),'MATERIALIZATION_MISMATCH')
            nr,kr=oracle.committed_losses()
            error=max(float((nr-torch.tensor(payload['nll'],device='cuda')).abs().max()),
                      float((kr-torch.tensor(payload['kl'],device='cuda')).abs().max()))
            require(error<=1e-3,'COMMIT_PARITY_FAILED')
            post_keys=capture_keys(model,spec,contexts,2)
            require(torch.equal(post_keys[4],keys[4]),'LOWEST_KEY_MISMATCH')
            row['commit_gate']={'materialization_bitwise':True,'NLL_KL_maxabs':error,'L4_key_bitwise':True}
            for l in LAYERS: txn.append(l,post_keys[l])
            row['history_appends']=len(txn.appended)
            row['commit_history_seconds']=time.monotonic()-t
            row['post']=state_hash(model,history)
            # Free optimizer/prefix/geometry before official observations.
            del payload,x,grad,adj,keys,post_keys,anchors,teacher,spec,nr,kr,realized,R
            oracle=None
            gc.collect(); torch.cuda.empty_cache()
            observation=observe(model,tok,all_records[:number*100],history,number,(number-1)*100)
            observation_path=out/f'W{number:02d}-observations.json'
            write_json(observation_path,observation)
            row['observation']={'path':str(observation_path),'sha256':file_sha(observation_path),
                                'current':observation['current'],'recorded_panels':observation['summary'],
                                'N_scope':'all_seen' if number in (5,10) else 'current100_only',
                                'seconds':observation['seconds'],'forward_microbatches':observation['forward_microbatches']}
            require(state_hash(model,history)==row['post'],'POST_OBSERVER_HASH_MISMATCH')
            row.update(commit=True,seconds=time.monotonic()-started,
                       peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_saved=False,
                       nonselected_guard='pointer/version/source scope; not all-byte certificate')
            def publish(new_ledger):
                write_json(out/f'B{number:03d}-committed.json',row)
                # ledger is the last publication and canonical commit marker.
                write_json(out/'ledger.json',new_ledger)
            txn.finish(row,publish)
    except Exception:
        trace=traceback.format_exc()
        if oracle is not None:
            row['attempt_oracle_counters']={'total':oracle.calls,'science':max(0,oracle.calls-before_calls) if science_started else 0,
                'preflight':before_calls if science_started else oracle.calls,'forward_chunks':oracle.forward_calls,
                'backward_chunks':oracle.backward_calls}
        row['attempt_seconds']=time.monotonic()-started
        # An os.replace may succeed before fsync raises. RAM rollback alone is
        # insufficient: re-publish the restored prefix and record ambiguity if
        # storage cannot confirm it. A failed candidate is never canonical.
        reconciled=reconcile_ledger(out,ledger)
        write_json(out/f'B{number:03d}-failure.json',{'status':'TECHNICAL_FAILURE','error':traceback.format_exc(),
            'rollback_verified':txn.restored,'completed_prefix':len(ledger),'row':row,
            'original_traceback':trace,'disk_ledger_reconciled_to_RAM':reconciled,
            'candidate_excluded':True,'commit_receipt_if_present':'NONCANONICAL_FAILED_ATTEMPT',
            'checkpoint_saved':False,'exact_resume':'NOT_AVAILABLE'})
        raise
    print({'event':'batch_committed','batch':number,'solver':result['status'],'calls':result['calls'],
           'seconds':row['seconds']},flush=True)
    return oracle_route

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    root=args.run;out=root/'output';out.mkdir(exist_ok=False)
    start=time.monotonic(); ledger=[]
    receipt={'status':'STARTED','checkpoint_saved':False,'exact_resume':'NOT_AVAILABLE',
             'native_fit_calls':0,'job_id':os.getenv('SLURM_JOB_ID'),'batches':10}
    try:
        lock=json.loads((root/'execution.lock.json').read_text())
        for r in lock['source_members']+lock['inputs']:
            require(Path(r['path']).stat().st_size==r['bytes'] and file_sha(r['path'])==r['sha256'],'LOCK_MISMATCH '+r['path'])
        contract=json.loads(CONTRACT.read_text())
        torch.set_num_threads(8); random.seed(20261001);np.random.seed(20261001);torch.manual_seed(20261001)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'ONE_GPU_REQUIRED')
        require(str(torch.__version__)==contract['runtime']['torch'] and transformers.__version__==contract['runtime']['transformers'],'RUNTIME_IDENTITY')
        records=load_prefix(Path(contract['data']['path']).parent,1000)
        require(digest([r['case_id'] for r in records])==contract['data']['first1000_case_ids_sha256'],'ORDER_MISMATCH')
        contexts=json.loads(Path(contract['data']['contexts_path']).read_text())
        require([len(x) for x in contexts]==[1,5],'CONTEXT_GROUPS')
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        tokenizer.padding_side='right';tokenizer.pad_token=tokenizer.eos_token
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32,
                   device_map={'':'cuda:0'},attn_implementation='eager').eval()
        model.config.use_cache=False
        for parameter in model.parameters():
            require(parameter.dtype==torch.float32,'FULL_FP32_REQUIRED');parameter.requires_grad_(False)
        history={l:torch.zeros(model.config.intermediate_size,model.config.intermediate_size) for l in LAYERS}
        receipt.update(load_seconds=time.monotonic()-start,torch=str(torch.__version__),transformers=transformers.__version__,
                       gpu=torch.cuda.get_device_name(),contract_sha256=file_sha(CONTRACT),W0_H0=state_hash(model,history))
        write_json(out/'runtime.json',receipt)
        w0=observe(model,tokenizer,records,history,0,w0=True)
        write_json(out/'W00-observations.json',w0);del w0
        oracle_route='shared_selected'
        for number in range(1,11):
            oracle_route=batch(model,tokenizer,records[(number-1)*100:number*100],records,
                               contexts,history,ledger,number,out,oracle_route)
        receipt['status']='TEN_BATCHES_COMMITTED_EVALUATED'
    except Exception as exc:
        receipt.update(status='TECHNICAL_FAILURE',error=str(exc),traceback=traceback.format_exc())
        print(receipt['traceback'],flush=True)
    finally:
        receipt.update(total_seconds=time.monotonic()-start,commits=len(ledger),history_appends=sum(r['history_appends'] for r in ledger),
                       committed_prefix_science_oracle_calls=sum(r['solver']['calls'] for r in ledger),
                       failed_attempt_costs='B*-failure.json; not included in committed-prefix counter',
                       peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None)
        write_json(out/'terminal.json',receipt)
    return 0 if receipt['status']=='TEN_BATCHES_COMMITTED_EVALUATED' else 2

if __name__=='__main__': raise SystemExit(main())
