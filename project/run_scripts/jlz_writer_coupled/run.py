"""Sealed v5 qualification and independent pilot/timing/cold-main workers.

No scheduler calls, retries, checkpoint or cross-arm learned-state sharing.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import re
import resource
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from .common import INSTRUCTION,require,write,sha,member,digest,state,tensor_sha
from .physical import LlamaAdapter
from .inputs import CounterFactAdapter,fact_identity,make_rows,batches
from .entry import prepare_entry,prepare_reference,cpu
from .oracle import Oracle
from .memory import NativeMemory
from .solver import solve,SolverConfig
from .writer import Transaction,commit
from .observe import observe

def seed(config):
    n=config['settings']['seed'];random.seed(n);np.random.seed(n);torch.manual_seed(n)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(n)

def verify(config):
    require(config['instruction_id']==INSTRUCTION,'INSTRUCTION')
    require(not config['settings']['save_checkpoints'],'NO_CP')
    require(torch.__version__==config['runtime']['torch'] and transformers.__version__==config['runtime']['transformers'],'RUNTIME_VERSION')
    for row in config['assets']:
        s=Path(row['path']).stat()
        require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_CHANGED:'+row['path'])

def setup(config,out):
    verify(config);seed(config);torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    a=LlamaAdapter(model,config['profile']);bench=CounterFactAdapter(tokenizer,json.loads(Path(config['contexts']).read_text()))
    history={l:torch.zeros(shape[1],shape[1],dtype=torch.float32) for l,shape in a.dims.items()}
    data=json.loads(Path(config['stream']).read_text());require(len(data)>=2004,'SHORT_STREAM')
    write(out/'runtime.json',dict(instruction=INSTRUCTION,job_id=os.environ.get('SLURM_JOB_ID'),device=torch.cuda.get_device_name(),
        versions=config['runtime'],model=config['model'],profile=a.profile,checkpoint_saved=False))
    imported={}
    for name,module in list(sys.modules.items()):
        p=getattr(module,'__file__',None)
        if p and Path(p).is_file() and name.startswith(('project.run_scripts.jlz_writer_coupled','project.run_scripts.jlz_pilot.prompts','transformers.models.llama','transformers.masking_utils')):
            imported[name]=member(p)
    write(out/'actual-imports.json',imported)
    return a,bench,data,history

def blank(a,B):return {l:torch.zeros(shape[0],B,device=a.device) for l,shape in a.dims.items()}

def fixed_candidate(a,B,scale):
    # Deterministic, feasible and outcome-independent; no global RNG advance.
    result={}
    for l,shape in a.dims.items():
        x=torch.arange(shape[0]*B,device=a.device,dtype=torch.float32).reshape(shape[0],B)
        x=torch.sin(x+float(l+1));result[l]=scale*x/x.norm(dim=0).clamp_min(1e-12)
    return result

def difference(left,right):
    g={}
    for l in left['grad']:
        x,y=left['grad'][l].double(),right['grad'][l].double()
        g[str(l)]=dict(relative=float((x-y).norm()/y.norm().clamp_min(1e-12)),
                      absolute_max=float((x-y).abs().max()),reference_norm=float(y.norm()),
                      columns=[dict(request=j,relative=float((x[:,j]-y[:,j]).norm()/y[:,j].norm().clamp_min(1e-12)),
                       absolute_max=float((x[:,j]-y[:,j]).abs().max()),reference_norm=float(y[:,j].norm())) for j in range(y.shape[1])])
    return dict(loss_absolute=abs(left['smooth']-right['smooth']),gradient=g,
        gradient_relative_max=max(c['relative'] for v in g.values() for c in v['columns']),
        left_stats=left['payload']['stats'],right_stats=right['payload']['stats'])

def qualification(a,bench,data,history,config,out,config_path):
    before=state(a,history);pack=bench.prepare(data[2000:2002])
    entry=prepare_entry(a,bench,pack,history,config['stats'],config['settings']['fit_microbatch'])
    write(out/'entry.json',dict(input_identity=pack['identity'],geometry=entry['geometry'],state=before))
    comparisons=[]
    for number,scale in enumerate((0.,.05),1):
        v=fixed_candidate(a,2,scale)
        left=Oracle(a,entry,[],[],0,'direct',True)(v,True)
        right=Oracle(a,entry,[],[],0,'dense',False)(v,True)
        d=difference(left,right);d.update(pair=number,scale=scale,left='direct_cached',right='dense_full')
        write(out/f'probe-{number}.json',d);comparisons.append(d);del left,right
    # Fixed same-candidate full-reference MB1 vs MB4 numerical envelope.
    v=fixed_candidate(a,2,.05);results=[]
    for mb in (1,4):
        other=dict(entry);other['groups']=[dict(rows=g,tokens=cpu(t),cache=None) for g,t in batches(make_rows(pack),mb,bench.tokenizer.pad_token_id,a.device)]
        results.append(Oracle(a,other,[],[],0,'dense',False)(v,True))
    d=difference(*results);d.update(pair=3,scale=.05,left='dense_full_MB1',right='dense_full_MB4')
    write(out/'probe-3.json',d);comparisons.append(d);del results
    require(d['loss_absolute']<=5e-5 and d['gradient_relative_max']<=2e-3,
            'DENSE_REFERENCE_MICROBATCH_QUALIFICATION_FAILED')
    require(all(x['loss_absolute']<=5e-4 and x['gradient_relative_max']<=2e-2 for x in comparisons[:2]),
            'DIRECT_CACHE_SEMANTIC_OR_GRADIENT_DISCREPANCY')
    direct_ok=all(d['loss_absolute']<=5e-5 and d['gradient_relative_max']<=2e-3 for d in comparisons[:2])
    # Freeze based exclusively on predetermined numerical probes, in mean
    # units; each solver receives actual_B times this value for SUM units.
    epsilon=max(1e-7,2*d['loss_absolute']/2)
    require(state(a,history)==before,'QUALIFICATION_MUTATION')
    receipt=dict(status='QUALIFIED',instruction=INSTRUCTION,source_commit=os.environ.get('ODEEDIT_SOURCE_COMMIT'),
        config_sha256=sha(config_path),epsilon_num_per_request=epsilon,
        epsilon_num_units='request_mean; solver uses actual_B multiplier',
        epsilon_rule='max(1e-7,2*qualified_dense_samecandidate_MB1_MB4_SUM_difference/actual_B)',
        route='direct' if direct_ok else 'dense',cached=direct_ok,
        route_verdict='DIRECT_QUALIFIED' if direct_ok else 'DENSE_FULL_REFERENCE_FALLBACK',
        comparisons=comparisons,qualification_pair_count=3,physical_forward=6,physical_backward=6,
        actual_probe_limit_total=6,initial_state=before,quality_selection=False,baseline_runs=0)
    write(out/'qualification.json',receipt)
    return receipt

@torch.no_grad()
def actual_commit_probe(a,entry,payload,out):
    measured=torch.zeros_like(payload['context_nll']);maxkey=0.
    for group in entry['groups']:
        captured={};hooks=[]
        for l in a.sites:
            hooks.append(a.blocks[l].mlp.down_proj.register_forward_pre_hook(lambda m,args,l=l:captured.update({l:args[0]})))
        try:
            nh,_=a.full({k:x.to(a.device) for k,x in group['tokens'].items()})
            for j,r in enumerate(group['rows']):
                if r['kind']!='rewrite':continue
                target=r['target'][r['target']!=-100].to(a.device);pos=torch.nonzero(r['target']!=-100).flatten().to(a.device)
                lp=a.head(nh[j,pos]).log_softmax(-1);context=r['global_row']%(entry['pack']['n_rw']+1)
                measured[r['request'],context]=float(-lp.gather(1,target[:,None]).mean())
                c=r['request']*entry['pack']['n_rw']+context
                for l in a.sites:maxkey=max(maxkey,float((captured[l][j,r['lookup']].cpu()-payload['keys'][l][:,c]).abs().max()))
        finally:
            for h in hooks:h.remove()
    require(torch.isfinite(measured).all(),'ACTUAL_COMMIT_NONFINITE')
    write(out/'actual-commit-probe.json',dict(expected=payload['context_nll'],actual=measured,
        nll_max_absolute=float((measured-payload['context_nll']).abs().max()),key_max_absolute=maxkey,
        exact_weight_copy=True,physical_forward=1,physical_backward=0,rounding_recorded=True))
    require(float((measured-payload['context_nll']).abs().max())<=5e-5 and maxkey<=1e-4,
            'ACTUAL_COMMIT_CAUSAL_KEY_PARITY_FAILED')

def batch(a,bench,records,history,memory,config,eta,out,qualified,number,entry_callback=None,probe=False):
    started=time.monotonic();pack=bench.prepare(records);before=state(a,history);mb=config['settings']['fit_microbatch']
    write(out/'input.json',dict(input_identity=pack['identity'],ids=pack['record_ids'],actual_B=len(records)))
    with Transaction(a,history,memory) as tx:
        refs=memory.sample([fact_identity(r) for r in records],len(records));mem_before=memory.summary()
        entry=prepare_entry(a,bench,pack,history,config['stats'],mb)
        write(out/'entry.json',dict(state=before,memory=mem_before,input_identity=pack['identity'],geometry=entry['geometry'],
            anchors={l:x.cpu().tolist() for l,x in entry['anchors'].items()},entry_seconds=entry['seconds']))
        if entry_callback:entry_callback(before,mem_before,pack)
        references=prepare_reference(a,bench,refs,mb) if eta else []
        oracle=Oracle(a,entry,refs,references,eta,qualified['route'],qualified['cached'])
        result=solve(oracle,blank(a,len(records)),entry['anchors'],SolverConfig(
            epsilon_num=qualified['epsilon_num_per_request']*len(records),
            lambda_norm=a.profile['norm_factor'],native_clamp=a.profile['clamp_factor']),
            on_trial=lambda row:write(out/'fit'/f"candidate-{row['evaluation']:02d}.json",row))
        require(state(a,history)==before,'ORACLE_MUTATED_WEIGHTS_OR_HISTORY')
        applied=commit(a,history,memory,entry,result.payload,records)
        if probe:actual_commit_probe(a,entry,result.payload,out)
        receipt=dict(batch=number,actual_B=len(records),current_ids=pack['record_ids'],before=before,after=applied['after'],
            candidate_count=result.calls,backward_count=result.backward_calls,accepted_updates=result.accepted_updates,
            rejected_trials=result.rejected_trials,accepted_evaluation=result.accepted_evaluation,
            stop_reason=result.stop_reason,history_appends=applied['history_appends'],
            memory_before=mem_before,memory_after=memory.summary(),reference_fact_ids=[r['fact_id'] for r in refs],
            reference_count=len(refs),accepted_stats=result.payload['stats'],history=applied,
            allocation={str(l):dict(relative_group_norm=result.v[l].norm(dim=0).cpu().tolist(),
                nominal_D_norm=(result.v[l]*entry['anchors'][l][None,:]).norm(dim=0).cpu().tolist(),
                actual_weight_delta_norm=float((result.payload['weights'][l]-entry['entry_weights'][l]).double().norm())) for l in a.sites},
            seconds=time.monotonic()-started,source=os.environ.get('ODEEDIT_SOURCE_COMMIT'),
            config=digest(config),no_checkpoint=True)
        write(out/'commit.json',receipt);tx.finish()
    del entry,result,oracle,references;gc.collect();torch.cuda.empty_cache()
    return receipt

def cold(a,history,memory,W0,config):
    with torch.no_grad():
        for l,w in a.weights.items():w.copy_(W0[l])
        for h in history.values():h.zero_()
    seed(config);return NativeMemory(seed=config['settings']['seed'])

def timing(a,bench,data,history,memory,config,eta,out,qualified):
    before=state(a,history);mem=memory.snapshot();ms=memory.summary()
    records=data[:100];pack=bench.prepare(records)
    refs=memory.sample([fact_identity(r) for r in records],100)
    entry=prepare_entry(a,bench,pack,history,config['stats'],config['settings']['fit_microbatch'])
    ref=prepare_reference(a,bench,refs,config['settings']['fit_microbatch']) if eta else []
    oracle=Oracle(a,entry,refs,ref,eta,qualified['route'],qualified['cached'])
    for i,scale in enumerate((0.,.025,.05),1):
        torch.cuda.synchronize();start=time.monotonic()
        result=oracle(fixed_candidate(a,100,scale),True);torch.cuda.synchronize()
        write(out/f'candidate-{i}.json',dict(candidate=i,role='warmup' if i==1 else 'measured',
            scale=scale,seconds=time.monotonic()-start,stats=result['payload']['stats'],
            available_pilot_reference=len(refs),main_reference_cap16_measured=False,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_host_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        del result
    memory.restore(mem)
    require(state(a,history)==before and memory.summary()==ms,'TIMING_MUTATION')
    write(out/'receipt.json',dict(candidate_count=3,backward_count=3,writes=0,available_pilot_reference=len(refs),
        geometry=entry['geometry'],entry_seconds=entry['seconds'],main_carryover=False))
    del entry,oracle,ref;gc.collect();torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['qualification','arm'],required=True)
    p.add_argument('--arm',choices=['A','B']);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--attempt',type=Path,required=True);p.add_argument('--qualification',type=Path);args=p.parse_args()
    config=json.loads(args.config.read_text());out=args.attempt/('qualification' if args.phase=='qualification' else 'arm-'+str(args.arm))
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();status='TECHNICAL_FAILED';commits=[]
    try:
        source=os.environ.get('ODEEDIT_SOURCE_COMMIT','')
        require(re.fullmatch('[0-9a-f]{40}',source),'SOURCE_COMMIT_ENV')
        lock=json.loads((args.attempt/'execution.lock.json').read_text())
        require(lock['source_commit']==source and lock['config_sha256']==sha(args.config),'EXECUTION_LOCK_BINDING')
        for item in lock['source_members']+lock['runtime_sources']:
            require(Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'],'SOURCE_MEMBER_CHANGED')
        if args.phase=='arm':
            seal=json.loads((args.attempt/'qualification-seal.json').read_text())
            require(str(args.qualification.resolve())==lock['qualification_path']==seal['receipt']['path'], 'QUALIFICATION_EXACT_PATH')
            require(sha(args.qualification)==seal['receipt']['sha256'], 'QUALIFICATION_SEAL_SHA')
            require(seal['source_commit']==source and seal['config_sha256']==sha(args.config), 'QUALIFICATION_SEAL_BINDING')
        a,bench,data,history=setup(config,out);initial=state(a,history);write(out/'initial-state.json',initial)
        reuse=config['w0_reuse']['receipt'];require(sha(reuse['path'])==reuse['sha256'],'W0_REUSE_RECEIPT')
        bridge=json.loads(Path(reuse['path']).read_text())
        require(initial==bridge['state'],'W0_METRICS_MODEL_STATE_MISMATCH')
        write(out/'W0-reuse.json',dict(mode='REUSED_HISTORICAL_NO_NEW_FORWARD',receipt=reuse,cold_state_exact=True,
            numerical_bitwise_equivalence='NOT_ESTABLISHED',old_task_resumed=False))
        if args.phase=='qualification':
            qualification(a,bench,data,history,config,out,args.config)
        else:
            q=json.loads(args.qualification.read_text())
            require(q['status']=='QUALIFIED' and q['instruction']==INSTRUCTION and q['config_sha256']==sha(args.config),'QUALIFICATION_BINDING')
            require(q['source_commit']==os.environ.get('ODEEDIT_SOURCE_COMMIT'),'QUALIFICATION_SOURCE')
            require(initial==q['initial_state'],'COLD_W0_H0')
            write(out/'qualification-reuse.json',member(args.qualification))
            W0={l:w.detach().cpu().clone() for l,w in a.weights.items()};memory=NativeMemory(seed=config['settings']['seed']);eta=int(args.arm=='B')
            with Transaction(a,history,memory) as tx:
                with torch.no_grad():a.weights[a.first].view(-1)[0].add_(1);history[a.first].view(-1)[0].add_(1)
            require(tx.rollback_verified,'ACTUAL_RAM_ROLLBACK');write(out/'rollback.json',dict(exact=True,new_forward=0))
            for b in (1,2):
                records=data[2000+(b-1)*2:2000+b*2]
                batch(a,bench,records,history,memory,config,eta,out/'pilot'/f'batch-{b:02d}',q,b,probe=b==1)
            pilot_memory=memory
            # Timing cold weights/history but isolated already-observed pilot
            # memory. Its anchor count is disclosed, not a fictitious cap16.
            cold(a,history,memory,W0,config)
            timing(a,bench,data,history,pilot_memory,config,eta,out/'timing',q)
            memory=cold(a,history,pilot_memory,W0,config)
            require(state(a,history)==initial and len(memory)==0,'MAIN_NOT_COLD')
            write(out/'main'/'cold.json',dict(state=initial,memory=memory.summary(),pilot_carryover=False))
            schedule=config['evaluation_schedule']['endpoints']
            for b in range(1,21):
                current=data[(b-1)*100:b*100]
                if commits:require(state(a,history)==commits[-1]['after'],'MAIN_PREVIOUS_COMMIT_ENTRY')
                def entry_callback(st,mem,pack):
                    if b==2:
                        write(out/'initial.json',dict(main_B1_committed=True,history_appends=len(a.sites),
                            observer_no_mutation=True,main_B2_entry=True,state=st,memory=mem,
                            own_entry_teacher=True,input_identity=pack['identity'],representative_only=True))
                rec=batch(a,bench,current,history,memory,config,eta,out/'main'/f'batch-{b:02d}',q,b,entry_callback)
                commits.append(rec);ep=schedule[b];lo,hi=ep['ordinal_slice']
                obs=observe(a,bench,data[:b*100],data[lo:hi],history,b,out/'main'/f'observe-W{b:02d}',
                    config['settings']['observer_microbatch'],[r['case_id'] for r in current],memory=memory)
                require({k:v['denominator'] for k,v in obs['summary'].items()}=={k:ep[k] for k in ('R','P','N')},'EVAL_SCHEDULE')
        status='COMPLETED'
    except BaseException as exc:
        write(out/'first-error.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
            new_checkpoint=False,exact_resume='NOT_AVAILABLE'));raise
    finally:
        write(out/'terminal.json',dict(status=status,phase=args.phase,arm=args.arm,main_commits=len(commits),
            instruction=INSTRUCTION,source=os.environ.get('ODEEDIT_SOURCE_COMMIT'),config_sha256=sha(args.config),
            seconds=time.monotonic()-start,job_id=os.environ.get('SLURM_JOB_ID'),checkpoint_saved=False,
            peak_host_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None))

if __name__=='__main__':main()
