"""Single cold B1 A fit -> five RAM snapshots -> D1/D2 -> exact teardown."""
import argparse
import gc
import importlib.metadata
import json
import os
import random
import resource
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from project.run_scripts.jlz_realized_subject.profile import LlamaAdapter
from project.run_scripts.jlz_realized_subject.inputs import CounterFactAdapter
from project.run_scripts.jlz_realized_subject.entry import prepare_entry
from project.run_scripts.jlz_realized_subject.optimize import fit
from project.run_scripts.jlz_realized_subject.writer import Transaction,commit,rng_snapshot,rng_equal
from project.run_scripts.jlz_realized_subject.qualification import actual_commit_probe
from project.run_scripts.jlz_realized_subject.observe import observe
from .common import *
from .instrument import RAMObserver,replace
from .qualify import qualify,mask_parity
from .masked_observer import observe_masks


def check(attempt):
    lock=json.loads((attempt/'execution.lock.json').read_text());config=json.loads((attempt/'config.json').read_text())
    require(lock['instruction']==NONCE and lock['source']==os.environ['ODEEDIT_SOURCE_COMMIT'],'SOURCE_AUTHORITY')
    for r in lock['members']+lock['runtime_sources']:
        require(Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256'],'SEALED_CLOSURE_CHANGED:'+r['path'])
    require(config['instruction']==NONCE and config['settings']==dict(seed=20261002,fit_microbatch=1,observer_microbatch=1,
        fit_count=1,B=100,batches=1,candidates=25,updates=24,arm='A',snapshots=list(CAPTURE),save_checkpoints=False,exact_resume='NOT_AVAILABLE'),'EXACT_SINGLE_B1_SETTINGS')
    for r in config['assets']:
        s=Path(r['path']).stat()
        require((s.st_size,s.st_ino,s.st_mtime_ns)==(r['bytes'],r['inode'],r['mtime_ns']),'ASSET_STAT_CHANGED')
    for name,version in config['runtime'].items():require(importlib.metadata.version(name)==version,'RUNTIME_VERSION')
    return lock,config


def raw(path):
    return [r for p in sorted(path.glob('chunk-*.json')) for r in json.loads(p.read_text())['rows']]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=Path,required=True);args=parser.parse_args()
    attempt=args.attempt;lock,config=check(attempt);out=attempt/'output';out.mkdir(exist_ok=False)
    require(os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1,'SINGLE_SLURM_GPU')
    started=time.monotonic();stage='SETUP';status='TECHNICAL_BLOCKED';cost={};observer=None;tx=None
    fit_started=False;fit_complete=False;complete_D1=[];complete_D2=[];restored=False
    try:
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        n=config['settings']['seed'];random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
        t=time.monotonic()
        model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,
            torch_dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
        require(all(p.dtype==torch.float32 for p in model.parameters()) and not torch.is_autocast_enabled(),'MODEL_PRECISION')
        tok=AutoTokenizer.from_pretrained(config['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
        a=LlamaAdapter(model,config['profile']);bench=CounterFactAdapter(tok,json.loads(Path(config['contexts']).read_text()))
        records=json.loads(Path(config['stream']).read_text())[:100];pack=bench.prepare(records)
        require(pack['identity']==config['input_identity'] and pack['record_ids']==config['ids'],'RUNTIME_PACKING')
        H={l:torch.zeros(shape[1],shape[1],dtype=torch.float32) for l,shape in a.dims.items()}
        metadata=dict(ledger=[],next_batch=1);lookup=json.loads((attempt/'lookup-local.json').read_text())
        require(sha(attempt/'lookup-local.json')==config['lookup']['sha256'],'LOOKUP_FROZEN_BEFORE_RESULTS')
        cost['load_seconds']=time.monotonic()-t
        imports={name:member(m.__file__) for name,m in list(sys.modules.items()) if getattr(m,'__file__',None)
            and Path(m.__file__).is_file() and name.startswith(('project.run_scripts.jlz_realized_subject','project.run_scripts.jlz_writer_coupled','project.run_scripts.jlz_pilot.prompts','transformers.models.llama'))}
        write(out/'runtime.json',dict(source=lock['source'],config_sha=sha(attempt/'config.json'),imports=imports,
            device=torch.cuda.get_device_name(),device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            torch=str(torch.__version__),versions=config['runtime'],TF32=False,autocast=False,attention='eager',
            reference_runtime=config['reference_runtime'],input_identity=pack['identity'],job_id=os.environ['SLURM_JOB_ID'],
            save_checkpoints=False,exact_resume='NOT_AVAILABLE',logical_B=100,fit_MB=1,observer_MB=1))
        with Transaction(a,H,metadata,bench.contexts) as tx:
            initial=state(a,H);rng=rng_snapshot();hooks=a.hook_signature();ctx=digest(bench.contexts)
            write(out/'initial-state.json',dict(state=initial,input_identity=pack['identity'],cold_W0_H0=True))
            stage='ENTRY';t=time.monotonic();entry=prepare_entry(a,bench,pack,H,config['stats'],1)
            cost['entry_seconds']=time.monotonic()-t
            write(out/'entry.json',dict(geometry=entry['geometry'],anchors=entry['anchors'],state=initial,
                input_identity=pack['identity'],teacher='same runtime own cold W0',seconds=cost['entry_seconds']))
            stage='FIXED_CANDIDATE_QUALIFICATION';qualify(a,entry,H,out/'qualification')
            require(state(a,H)==initial and rng_equal(rng) and a.hook_signature()==hooks,'PRE_FIT_STATE')
            write(out/'TECHNICAL_QUALIFIED.json',dict(status='TECHNICAL_QUALIFIED',logical_B=100,
                fits_started=0,source=lock['source'],mask_full_subset_parity='CHECKED_AFTER_FIT_NOT_YET',time=time.time()))
            stage='SINGLE_FIT';observer=RAMObserver(out/'instrument');fit_started=True;t=time.monotonic()
            payload,summary=fit(a,entry,'A',out/'fit',1,observer=observer)
            cost['fit_including_nested_observer_seconds']=time.monotonic()-t;fit_complete=True
            require(set(observer.snapshots)==set(CAPTURE),'SNAPSHOT_COVERAGE')
            require(state(a,H)==initial and rng_equal(rng) and digest(bench.contexts)==ctx,'FIT_MASTER_OR_RNG_MUTATION')
            require(all(tensor_sha(payload['weights'][l])==observer.identities[25][str(l)] for l in a.sites),'C25_CAPTURE_TERMINAL_IDENTITY')
            stage='EPHEMERAL_COMMIT';t=time.monotonic();applied=commit(a,H,entry,payload)
            actual_commit_probe(a,entry,payload,out/'qualification')
            write(out/'ephemeral-commit.json',dict(**applied,fit_summary=summary,scope='single ephemeral terminal, no continuation',
                c25_snapshot=observer.identities[25]))
            # Return to W0/H0 explicitly before all diagnostic probes.
            replace(a,tx.W)
            with torch.no_grad():
                for l,h in H.items():h.copy_(tx.H[l])
            require(state(a,H)==initial and rng_equal(rng),'POST_COMMIT_EPHEMERAL_RESET')
            write(out/'ephemeral-reset.json',dict(exact=True,state=initial,history_appends=5,no_B2=True))
            cost['commit_probe_reset_seconds']=time.monotonic()-t
            del entry,payload;gc.collect();torch.cuda.empty_cache()
            stage='D1';cost['D1']={}
            for candidate in (0,)+CAPTURE:
                target=tx.W if candidate==0 else observer.snapshots[candidate]
                replace(a,target)
                require({str(l):tensor_sha(w) for l,w in a.weights.items()}=={str(l):tensor_sha(w) for l,w in target.items()},'D1_ENDPOINT_ACTIVATION')
                result=observe(a,bench,records,records,H,candidate,out/'D1'/f'c{candidate:02d}',1)
                require({k:v['denominator'] for k,v in result['summary'].items()}==dict(R=100,P=200,N=1000),'D1_DENOMINATOR')
                complete_D1.append(candidate);cost['D1'][str(candidate)]=result['seconds'];replace(a,tx.W)
                require(state(a,H)==initial and rng_equal(rng),'D1_RESTORE')
            stage='D2';t=time.monotonic()
            d2=observe_masks(a,bench,records,H,lookup,tx.W,observer.snapshots[25],out/'D2',1)
            if d2:
                for label,c in (('NONE',0),('ALL',25)):
                    evidence=mask_parity(d2[label],raw(out/'D1'/f'c{c:02d}'),label)
                    write(out/'qualification'/f'mask-{label}.json',evidence)
                    require(evidence['passed'],'D2_'+label+'_PARITY')
                complete_D2=list(d2)
            cost['D2_seconds']=time.monotonic()-t
            require(state(a,H)==initial and rng_equal(rng) and digest(bench.contexts)==ctx,'FINAL_DIAGNOSTIC_STATE')
            write(out/'scope-complete.json',dict(D1=complete_D1,D2=complete_D2,D2_identifiable=lookup['identifiable'],
                scientific_fits=1,candidates=25,updates=24,history_appends=5,probe_history_appends=0,
                state_restored=initial,checkpoint_saved=False,other_tasks_touched=False))
        restored=tx.rollback_verified;require(restored,'FINAL_TRANSACTION_RESTORE')
        stage='DONE';status='D1_D2_COMPLETE'
    except BaseException as exc:
        if isinstance(exc,torch.cuda.OutOfMemoryError):status='TECHNICAL_CAPACITY_BLOCKED'
        write(out/'first-error.json',dict(stage=stage,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
            fit_started=fit_started,fit_complete=fit_complete,automatic_retry=False,original_outputs_preserved=True))
        raise
    finally:
        if observer is not None:observer.clear()
        if tx is not None:restored=tx.rollback_verified
        gc.collect()
        write(out/'terminal.json',dict(status=status,last_stage=stage,source=lock['source'],instruction=NONCE,
            config_sha=sha(attempt/'config.json'),job_id=os.environ['SLURM_JOB_ID'],fit_started=fit_started,fit_complete=fit_complete,
            D1_completed=complete_D1,D2_completed=complete_D2,restored=restored,cost=cost,
            seconds=time.monotonic()-started,peak_host_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(),
            checkpoint_saved=False,exact_resume='NOT_AVAILABLE',automatic_retry=False,
            allocation_GPU_seconds='COLLECTOR_ACCOUNTING',cost_note='fit timer includes nested diagnostic backward/capture; do not add twice'))


if __name__=='__main__':main()
