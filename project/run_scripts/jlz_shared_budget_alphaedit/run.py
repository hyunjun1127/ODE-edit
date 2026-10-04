"""One cold chain. No scheduler calls, continuation callbacks, or checkpoints."""
import argparse
import gc
import json
import os
import random
import resource
import shutil
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from .entry import prepare_entry,Adapter
from .telemetry import Events
from .events import metadata,batch_entry,evaluation
from .w0 import rows_from
from project.run_scripts.jlz_realization.observe import observe,reduce_rows
from project.run_scripts.jlz_realization.writer import Transaction,rng_snapshot,rng_equal
from project.run_scripts.jlz_two_arm.baseline_pilot import _rng_hash
from .optimize import fit
from . import writer,qualification
from .common import *

def locked(attempt):
    lock=json.loads((attempt/'execution.lock.json').read_text());config=json.loads((attempt/'config.json').read_text())
    require(lock['instruction_id']==config['instruction_id']==NONCE and config['task_id']==TASK,'AUTHORITY')
    require(config['settings']['chains']==['V12_ALPHAEDIT'] and config['alphaedit']['L2']==10 and config['alphaedit']['blue'] is False,'WRITER_AUTHORITY')
    require(config['alphaedit']['dtype']=='float32' and config['profile']['lambda_C']==0,'NO_RIDGE_MAIN')
    require(os.environ.get('ODEEDIT_SOURCE_COMMIT')==lock['source_commit'] and sha(attempt/'config.json')==lock['config_sha256'],'FROZEN_SOURCE')
    for row in lock['source_members']+lock['dependency_sources']+lock['runtime_sources']+lock['native_reference']:verify(row)
    verify(lock['native_hparams']);verify(config['native_input_alignment'])
    for row in config['assets']:
        stat=Path(row['path']).stat();require((stat.st_size,stat.st_ino,stat.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_CHANGED')
    return lock,config

def tokenizer_metadata(tokenizer):
    # Generic fast tokenizers encode BOS behavior in their postprocessor and
    # need not expose the Llama-specific attribute. This is reporting only.
    return dict(tokenizer_type=type(tokenizer).__name__,
        add_bos=getattr(tokenizer,'add_bos_token',None),
        add_bos_attribute_available=hasattr(tokenizer,'add_bos_token'),
        bos_token_id=tokenizer.bos_token_id)

def setup(config,out):
    require(shutil.disk_usage(out).free>=config['resources']['reserve_bytes'],'RESOURCE_BLOCKED_STORAGE')
    require(torch.__version__==config['runtime']['torch'] and transformers.__version__==config['runtime']['transformers'],'RUNTIME_VERSION')
    records=load_prefix(Path(config['stream']).parent,2004)
    torch.set_num_threads(8);random.seed(20261002);np.random.seed(20261002);torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,dtype=torch.float32,
        attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    adapter=Adapter(model,config['profile']);bench=CounterFactAdapter(tokenizer,json.loads(Path(config['contexts']).read_text()))
    history={l:torch.zeros(shape[1],shape[1],dtype=torch.float32) for l,shape in adapter.dims.items()}
    require(all(p.dtype==torch.float32 for p in model.parameters()) and not torch.is_autocast_enabled(),'MODEL_PRECISION')
    write(out/'runtime.json',dict(nonce=NONCE,source=os.environ['ODEEDIT_SOURCE_COMMIT'],job=os.environ.get('SLURM_JOB_ID'),
        torch=torch.__version__,transformers=transformers.__version__,device=torch.cuda.get_device_name(),
        cold_W0_H0=True,model=config['model'],**tokenizer_metadata(tokenizer),
        eager=True,tf32=False,autocast=False,noCP=True,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        GPU_properties=str(torch.cuda.get_device_properties(0))))
    write(out/'initial-state.json',state(adapter,history))
    pi_config=config['alphaedit']['projector']
    p=torch.load(pi_config['path'],map_location='cpu',weights_only=True,mmap=True)
    require(p.dtype==torch.float32 and list(p.shape)==[5,14336,14336],'PROJECTOR_HEADER')
    require(config['alphaedit']['layer_mapping']=={str(l):l-4 for l in adapter.sites},'PROJECTOR_LAYER_MAPPING')
    adapter.alpha_projectors={l:p[l-4].to(adapter.device) for l in adapter.sites}
    return adapter,bench,records,history

def observer(a,bench,all_records,selected,H,endpoint,out,config,current=None):
    rng=rng_snapshot();before=state(a,H)
    result=observe(a,bench,all_records,selected,H,endpoint,out,config['settings']['observer_microbatch'],current)
    require(rng_equal(rng) and state(a,H)==before,'OBSERVER_RNG_STATE')
    signed=[]
    for row in rows_from(out):
        signed.append(dict(identity=row['identity'],case_id=row['case_id'],kind=row['kind'],
            margin_true_minus_new=row['true_nll']-row['new_nll'],margin_new_minus_true=row['new_nll']-row['true_nll']))
    write(out/'signed-margins.json',dict(rows=signed,derived_without_forward=True))
    return result

def w0(a,bench,records,H,attempt,out,config):
    before=state(a,H)
    if config['W0']['mode']=='EXACT_STATIC_W0_REUSE':
        receipt=json.loads(verify(config['W0']['receipt']).read_text())
        for row in receipt['rows']+[receipt['binding'],receipt['summary']]:verify(row)
        require(before==receipt['state'],'COLD_W0_REUSE_STATE')
        write(out/'W0-reuse.json',dict(receipt=config['W0']['receipt'],runtime_cold_state_match=True,new_forward=0))
        return Path(receipt['folder'])
    folder=attempt/'shared-W0'
    if not folder.exists():
        result=observer(a,bench,records[:2000],records[:2000],H,0,folder,config)
        write(folder/'binding.json',dict(source=os.environ['ODEEDIT_SOURCE_COMMIT'],config=digest(config),state=before,
            observer_identity=config['observer_identity'],completed=True,requests=2000,no_feedback=True))
    bound=json.loads((folder/'binding.json').read_text())
    require(bound['completed'] and bound['state']==before and bound['source']==os.environ['ODEEDIT_SOURCE_COMMIT']
            and bound['config']==digest(config),'SHARED_W0_EXACT')
    write(out/'W0-reuse.json',dict(binding=member(folder/'binding.json'),summary=member(folder/'summary.json'),state=before))
    return folder

def subset_w0(folder,current,out):
    ids={r['case_id'] for r in current};rows=[]
    for path in sorted(folder.glob('chunk-*.json')):
        rows.extend(r for r in json.loads(path.read_text())['rows'] if r['case_id'] in ids)
    require(len(rows)==13*len(current),'W0_PRE_CURRENT_COVERAGE')
    summary=reduce_rows(rows)
    write(out/'rows.json',dict(rows=rows,reused_same_W0=True))
    result=dict(summary=summary,no_mutation=True,seconds=0.,row_count=len(rows),reuse=str(folder))
    write(out/'summary.json',result);return result

def drive(a,bench,records,H,config,chain,out,attempt,pilot,bound=None):
    rows=records[2000:2004] if pilot else records[:2000];B=2 if pilot else 100;count=2 if pilot else 20
    source=os.environ['ODEEDIT_SOURCE_COMMIT'];previous=state(a,H);previous_rng=rng_snapshot();commits=[]
    meta=Events(out/'run-events.jsonl',source+':'+chain);metadata(meta,a,config,pilot,source)
    shared=None if pilot else w0(a,bench,records,H,attempt,out,config)
    for number in range(1,count+1):
        require(shutil.disk_usage(out).free>=10*1024**3,'RESOURCE_BLOCKED_ATOMIC_STORAGE_FLOOR')
        require(state(a,H)==previous and rng_equal(previous_rng),'OWN_W_H_RNG_JOIN')
        current=rows[(number-1)*B:number*B];seen=rows[:number*B];root=out/f'batch-{number:02d}'
        pack=bench.prepare(current)
        sealed=next(x for x in config['packing'] if x['phase']==('pilot' if pilot else 'main') and x['batch']==number)
        require(pack['identity']==sealed['identity'] and pack['record_ids']==sealed['ids'],'PACK_LOCK')
        if number==2 and not pilot:
            write(out/'initial.json',dict(main_B1_commit=True,main_B2_ownentry=True,observer_restored=True,history5=True,
                  source=source,config=digest(config),state=previous,pack=pack['identity'],ids=pack['record_ids'],no_main_completion_claim=True))
        write(root/'entry.json',dict(source=source,config=digest(config),before=previous,ids=pack['record_ids'],
              native_pack=pack['identity'],batch=number,chain=chain,phase='pilot' if pilot else 'main',
              RNG_hash=_rng_hash(previous_rng),context_hash=digest(bench.contexts)))
        started=time.monotonic();transaction=Transaction(a,H)
        events=Events(root/'events.jsonl',source+':'+chain,number)
        try:
            with transaction:
                if shared is not None and number==1:
                    pre=subset_w0(shared,current,root/'pre')
                else:pre=observer(a,bench,seen,current,H,number-1,root/'pre',config)
                entry=prepare_entry(a,bench,pack,H,config['stats'],config['settings']['fit_requests_per_group'])
                entry['projectors']=a.alpha_projectors
                entry['alpha_native_source']=config['alphaedit']['native_source']['path']
                entry['qualify_alphaedit']=pilot
                batch_entry(events,entry,previous)
                if pilot and number==1:qualification.qualify(a,bench,entry,config,root/'qualification')
                plan,fit_receipt=fit(a,entry,events,root/'fit')
                require(state(a,H)==previous,'FIT_MUTATED_ENTRY')
                if not pilot and number==1:
                    from .probe import same_plan
                    same_plan(a,bench,seen,current,H,entry,plan,config,root/'same-plan-ridge-probe',observer)
                write_receipt=writer.apply(a,entry,plan,H,events,root)
                del plan,entry;gc.collect();torch.cuda.empty_cache()
                selected=seen if (not pilot and number in MILESTONES) or (pilot and number==count) else current
                post=observer(a,bench,seen,selected,H,number,root/'post',config,[r['case_id'] for r in current])
                require({k:v['denominator'] for k,v in post['summary'].items()}==dict(R=len(selected),P=2*len(selected),N=10*len(selected)),'POST_EVAL_COVERAGE')
                after=state(a,H)
                receipt=dict(source=source,config=digest(config),chain=chain,batch=number,ids=pack['record_ids'],before=previous,after=after,
                    native_pack=pack['identity'],fit=fit_receipt,history_appends=write_receipt['history_appends'],
                    observer_no_mutation=True,pre=pre['summary'],post=post['summary'],post_current=post['current'],
                    seconds=time.monotonic()-started,checkpoint_saved=False,RNG_before=_rng_hash(previous_rng),
                    RNG_after=_rng_hash(rng_snapshot()),context_hash=digest(bench.contexts))
                evaluation(events,pre['summary'],digest(previous),config['observer_identity']['sha256'],pack['record_ids'],'PRE_EDIT')
                evaluation(events,post['summary'],digest(after),config['observer_identity']['sha256'],
                    [r['case_id'] for r in selected],'SEEN_PREFIX' if len(selected)>B else 'CURRENT_AT_WRITE')
                require(a.guard()==transaction.guard,'NONSELECTED_PARAMETER_MUTATION')
                events.emit('batch_commit',dict(entry_model_state_id=digest(previous['W']),final_model_state_id=digest(after['W']),
                    entry_history_state_id=digest(previous['H']),final_history_state_id=digest(after['H']),
                    request_count=B,layer_write_count=len(a.sites),history_append_count_per_layer=1,all_requests_terminal=True,
                    nonselected_weights_preserved=True,ledger_commit_id=digest(receipt),technical_status='COMMITTED'))
                write(root/'commit.json',receipt)
                transaction.finish()
            commits.append(receipt);previous=after;previous_rng=rng_snapshot()
            print(dict(event='V12_COMMIT',chain=chain,batch=number,phase='pilot' if pilot else 'main'),flush=True)
        except BaseException as error:
            write(root/'rollback.json',dict(verified=transaction.rollback_verified,original_state=transaction.before,
                rollback_after_SIGTERM='NOT_VERIFIED',logical_commit=False))
            events.emit('technical_failure',dict(stage='ROLLBACK',code=type(error).__name__,detail=str(error) or repr(error),
                state_before_failure_id=digest(transaction.before),rollback_required=True,
                rollback_status='RESTORED_AND_VERIFIED' if transaction.rollback_verified else 'ROLLBACK_FAILED',
                restored_entry_model_id=digest(previous['W']) if transaction.rollback_verified else None,
                restored_entry_history_id=digest(previous['H']) if transaction.rollback_verified else None,
                rng_cache_ledger_restored=True if transaction.rollback_verified else None,
                next_batch_allowed=False,performance_score_imputed=False))
            raise
    require(len(commits)==count and sum(x['history_appends'] for x in commits)==5*count,'CHAIN_COVERAGE')
    return commits

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--chain',choices=['V12_ALPHAEDIT'],default='V12_ALPHAEDIT')
    p.add_argument('--pilot',action='store_true');args=p.parse_args()
    attempt=args.attempt;out=attempt/(('pilot-' if args.pilot else 'main-')+args.chain)
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();status='TECHNICAL_FAILED';commits=[]
    try:
        lock,config=locked(attempt)
        if not args.pilot:
            ready=json.loads((attempt/'pilot-V12_ALPHAEDIT'/'ready.json').read_text())
            require(ready['status']=='CHAIN_COMPLETE' and ready['source']==lock['source_commit'] and ready['config']==digest(config),'PILOT_NOT_QUALIFIED')
        a,bench,records,H=setup(config,out)
        if args.pilot:
            before=state(a,H)
            try:
                with Transaction(a,H):
                    with torch.no_grad():
                        a.weights[a.first].view(-1)[0].add_(1);H[a.first].view(-1)[0].add_(1)
                    raise RuntimeError('BOUNDED_ROLLBACK_PROBE')
            except RuntimeError as e:require(str(e)=='BOUNDED_ROLLBACK_PROBE','ROLLBACK_PROBE_EXCEPTION')
            require(state(a,H)==before,'ACTUAL_ROLLBACK')
            write(out/'rollback-probe.json',dict(exact_W_H_RNG=True,new_fit=0))
        commits=drive(a,bench,records,H,config,args.chain,out,attempt,args.pilot)
        imported=[]
        for name,module in list(sys.modules.items()):
            path=getattr(module,'__file__',None)
            if isinstance(path,str) and Path(path).is_absolute() and Path(path).is_file() and name.startswith(('project.run_scripts.','memit.','rome.','util.','transformers.models.llama')):
                imported.append(dict(module=name,**member(path)))
        write(out/'actual-imports.json',dict(files=imported,source=lock['source_commit']))
        write(out/'ready.json',dict(status='CHAIN_COMPLETE',source=lock['source_commit'],config=digest(config),
            chain=args.chain,pilot=args.pilot,commits=len(commits),requests=4 if args.pilot else 2000))
        status='COMPLETED'
    except BaseException as error:
        write(out/'first-error.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
              original_artifacts_preserved=True));raise
    finally:
        write(out/'terminal.json',dict(status=status,source=os.environ.get('ODEEDIT_SOURCE_COMMIT'),chain=args.chain,pilot=args.pilot,
            commits=len(list(out.glob('batch-*/commit.json'))),seconds=time.monotonic()-started,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_VRAM_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            job=os.environ.get('SLURM_JOB_ID'),checkpoint_saved=False,no_B21=True))

if __name__=='__main__':main()
