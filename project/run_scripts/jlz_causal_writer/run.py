"""Sealed v7 prepare/main processes. No scheduler calls or retries."""
import argparse
import gc
import json
import os
import random
import resource
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from .common import INSTRUCTION,TASK,require,write,sha,member,digest,state,tensor_sha
from .profile import LlamaAdapter
from .inputs import CounterFactAdapter,fact_identity
from .entry import prepare_entry,prepare_reference
from .memory import NativeMemory
from .writer import Transaction,commit
from .observe import observe
from .optimize import fit
from .qualification import qualify

def seed(config):
    n=config['settings']['seed'];random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)

def setup(config,out):
    require(config['instruction_id']==INSTRUCTION and not config['settings']['save_checkpoints'],'TASK_AUTHORITY_NOCP')
    require(torch.__version__==config['runtime']['torch'] and transformers.__version__==config['runtime']['transformers'],'RUNTIME_VERSION')
    for row in config['assets']:
        s=Path(row['path']).stat()
        require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_CHANGED')
    torch.set_num_threads(8);seed(config)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,
        dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    a=LlamaAdapter(model,config['profile']);bench=CounterFactAdapter(tokenizer,json.loads(Path(config['contexts']).read_text()))
    history={l:torch.zeros(shape[1],shape[1],dtype=torch.float32) for l,shape in a.dims.items()}
    data=json.loads(Path(config['stream']).read_text())
    write(out/'runtime.json',dict(instruction=INSTRUCTION,job_id=os.environ.get('SLURM_JOB_ID'),device=torch.cuda.get_device_name(),
        versions=config['runtime'],model=config['model'],profile=a.profile,checkpoint_saved=False))
    imports={name:member(m.__file__) for name,m in list(__import__('sys').modules.items())
        if getattr(m,'__file__',None) and Path(m.__file__).is_file() and
        name.startswith(('project.run_scripts.jlz_causal_writer','project.run_scripts.jlz_writer_coupled','project.run_scripts.jlz_pilot.prompts','transformers.models.llama'))}
    write(out/'actual-imports.json',imports)
    return a,bench,data,history

def batch(a,bench,records,history,memory,config,arm,out,number,qualification=False,callback=None):
    started=time.monotonic();before=state(a,history);pack=bench.prepare(records)
    write(out/'input.json',dict(identity=pack['identity'],ids=pack['record_ids'],actual_B=len(records)))
    with Transaction(a,history,memory) as tx:
        residents=memory.sample([fact_identity(r) for r in records],len(records));mem_before=memory.summary()
        entry=prepare_entry(a,bench,pack,history,config['stats'],config['settings']['fit_microbatch'])
        write(out/'entry.json',dict(state=before,memory=mem_before,input=pack['identity'],prior=entry['geometry'],
            anchors=entry['anchors'],source=os.environ.get('ODEEDIT_SOURCE_COMMIT')))
        if callback:callback(before,mem_before,pack)
        if qualification:qualify(a,entry,out/'qualification')
        references=prepare_reference(a,bench,residents,config['settings']['fit_microbatch'])
        payload,summary=fit(a,entry,residents,references,arm,out/'fit',number,config['settings']['seed'],sha(config['stream']))
        require(state(a,history)==before,'FIT_MASTER_STATE_MUTATION')
        applied=commit(a,history,memory,entry,payload,records)
        if qualification:
            # Read-only native full-model replay of evaluated committed weights.
            from project.run_scripts.jlz_writer_coupled.run import actual_commit_probe
            actual_commit_probe(a,entry,payload,out)
        receipt=dict(batch=number,actual_B=len(records),current_ids=pack['record_ids'],before=before,after=applied['after'],
            candidate_count=summary['candidates'],backward_count=summary['Adam_updates'],Adam_updates=summary['Adam_updates'],
            history=applied,history_appends=applied['history_appends'],memory_before=mem_before,memory_after=memory.summary(),
            reference_count=len(residents),reference_fact_ids=[r['fact_id'] for r in residents],
            native_terminal_nll=payload['context_nll'],terminal_gradient_measured=False,
            seconds=time.monotonic()-started,source=os.environ.get('ODEEDIT_SOURCE_COMMIT'),config=digest(config),
            exact_commit_sha={str(l):tensor_sha(w) for l,w in payload['weights'].items()},
            no_checkpoint=True,exact_resume='NOT_AVAILABLE')
        write(out/'commit.json',receipt);tx.finish()
    del entry,payload,references;gc.collect();torch.cuda.empty_cache()
    return receipt

def cold(a,history,W0,config):
    with torch.no_grad():
        for l,w in a.weights.items():w.copy_(W0[l])
        for h in history.values():h.zero_()
    seed(config);return NativeMemory(seed=config['settings']['seed'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prep','main'],required=True)
    p.add_argument('--arm',choices=['A','B'],required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--attempt',type=Path,required=True);args=p.parse_args()
    config=json.loads(args.config.read_text());out=args.attempt/(args.phase+'-'+args.arm)
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();status='TECHNICAL_FAILED';commits=[]
    try:
        lock=json.loads((args.attempt/'execution.lock.json').read_text());source=os.environ.get('ODEEDIT_SOURCE_COMMIT')
        require(source==lock['source_commit'] and sha(args.config)==lock['config_sha256'],'SOURCE_CONFIG_LOCK')
        for row in lock['source_members']+lock['runtime_sources']:
            require(Path(row['path']).stat().st_size==row['bytes'] and sha(row['path'])==row['sha256'],'SOURCE_CHANGED')
        if args.phase=='main':
            evidence=[]
            for arm in ('A','B'):
                path=args.attempt/('prep-'+arm)/'READY.json';r=json.loads(path.read_text())
                require(r['status']=='STRUCTURAL_READY' and r['instruction']==INSTRUCTION and r['source']==source
                    and r['config_sha256']==sha(args.config),'MATCHING_PREP_READY')
                evidence.append(member(path))
            write(out/'upstream.json',evidence)
        a,bench,data,history=setup(config,out);initial=state(a,history);write(out/'initial-state.json',initial)
        bridge=json.loads(Path(config['w0_reuse']['receipt']['path']).read_text())
        require(sha(config['w0_reuse']['receipt']['path'])==config['w0_reuse']['receipt']['sha256'] and initial==bridge['state'],'COLD_W0_IDENTITY')
        write(out/'W0-reuse.json',dict(new_forward=0,receipt=config['w0_reuse']['receipt'],actual_cold_state=True))
        memory=NativeMemory(seed=config['settings']['seed'])
        if args.phase=='prep':
            W0={l:w.detach().cpu().clone() for l,w in a.weights.items()}
            # Actual RAM rollback, no forward or saved state bundle.
            with Transaction(a,history,memory) as tx:
                with torch.no_grad():a.weights[a.first].view(-1)[0].add_(1);history[a.first].view(-1)[0].add_(1)
            require(tx.rollback_verified,'RAM_ROLLBACK');write(out/'rollback.json',dict(exact=True,new_forward=0))
            for b in (1,2):
                batch(a,bench,data[2000+(b-1)*2:2000+b*2],history,memory,config,args.arm,
                      out/'pilot'/f'batch-{b:02d}',b,qualification=(args.arm=='A' and b==1))
            if args.arm=='B':
                memory=cold(a,history,W0,config);require(state(a,history)==initial and len(memory)==0,'SMOKE_COLD')
                entry=prepare_entry(a,bench,bench.prepare(data[:100]),history,config['stats'],config['settings']['fit_microbatch'])
                _,smoke=fit(a,entry,[],[],args.arm,out/'B100',1,config['settings']['seed'],sha(config['stream']),5,4)
                require(state(a,history)==initial,'SMOKE_NO_COMMIT')
                write(out/'B100'/'receipt.json',dict(**smoke,no_state_to_main=True,candidate5_backward=True,
                    fifth_Adam_step=False,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                    peak_host_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            write(out/'READY.json',dict(status='STRUCTURAL_READY',instruction=INSTRUCTION,source=source,
                config_sha256=sha(args.config),pilot_commits=2,B100_required_this_lane=args.arm=='B',main_state_transfer=False))
        else:
            for b in range(1,6):
                current=data[(b-1)*100:b*100]
                if commits:require(state(a,history)==commits[-1]['after'],'OWN_ENTRY_CONTINUITY')
                def callback(st,mem,pack):
                    if b==2:write(out/'initial.json',dict(main_B1_commit=True,all5_history=True,memory=mem,
                        observer_restored=True,main_B2_ownentry=True,state=st,input=pack['identity'],
                        source=source,config_sha256=sha(args.config),representative_only=True))
                receipt=batch(a,bench,current,history,memory,config,args.arm,out/f'batch-{b:02d}',b,callback=callback)
                commits.append(receipt)
                selected=data[:500] if b==5 else current
                obs=observe(a,bench,data[:b*100],selected,history,b,out/f'observe-W{b:02d}',
                    config['settings']['observer_microbatch'],[r['case_id'] for r in current],memory)
                N=500 if b==5 else 100
                require({k:v['denominator'] for k,v in obs['summary'].items()}==dict(R=N,P=2*N,N=10*N),'EVAL_COUNTS')
        status='COMPLETED'
    except BaseException as exc:
        write(out/'first-error.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
              rollback_after_process_termination='NOT_VERIFIED',no_checkpoint=True));raise
    finally:
        write(out/'terminal.json',dict(status=status,phase=args.phase,arm=args.arm,instruction=INSTRUCTION,
            source=os.environ.get('ODEEDIT_SOURCE_COMMIT'),config_sha256=sha(args.config),main_commits=len(commits),
            job_id=os.environ.get('SLURM_JOB_ID'),seconds=time.monotonic()-started,checkpoint_saved=False,no_B6=True,
            peak_host_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None))

if __name__=='__main__':main()
