"""Sealed Q1 and cold A/B programs. No submission, retry, checkpoint or B6."""
import argparse
import gc
import hashlib
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
from .common import INSTRUCTION,require,write,sha,member,digest,state,tensor_sha
from .profile import LlamaAdapter
from .inputs import CounterFactAdapter
from .entry import prepare_entry
from .writer import Transaction,commit,rng_snapshot,rng_equal
from .observe import observe
from .optimize import fit
from .qualification import qualify,shape_check,actual_commit_probe


def aux_identity(bench,metadata):
    rng=rng_snapshot();h=hashlib.sha256()
    h.update(repr(rng[0]).encode());h.update(repr((rng[1][0],rng[1][2:])).encode());h.update(rng[1][1].tobytes())
    h.update(rng[2].numpy().tobytes())
    for value in rng[3]:h.update(value.cpu().numpy().tobytes())
    return dict(rng=h.hexdigest(),contexts=digest(bench.contexts),ledger=digest(metadata))


def setup(config,out):
    require(config['instruction_id']==INSTRUCTION and config['settings']['save_checkpoints'] is False,'AUTHORITY_NOCP')
    require(str(torch.__version__)==config['runtime']['torch'] and transformers.__version__==config['runtime']['transformers'],'VERSION')
    for row in config['assets']:
        st=Path(row['path']).stat();require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_CHANGED')
    n=config['settings']['seed'];random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,torch_dtype=torch.float32,
        attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
    require(all(p.dtype==torch.float32 for p in model.parameters()) and not torch.is_autocast_enabled(),'MODEL_FP32')
    tok=AutoTokenizer.from_pretrained(config['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    a=LlamaAdapter(model,config['profile']);bench=CounterFactAdapter(tok,json.loads(Path(config['contexts']).read_text()))
    H={l:torch.zeros(shape[1],shape[1],dtype=torch.float32) for l,shape in a.dims.items()}
    w0=config['W0_reuse'];require(sha(w0['path'])==w0['sha256'],'W0_REUSE_SHA')
    reference=json.loads(Path(w0['path']).read_text());comparison={}
    for l,w in a.weights.items():
        x=w.detach().contiguous().cpu();h=hashlib.sha256(str((str(x.dtype),list(x.shape))).encode());h.update(memoryview(x.numpy()).cast('B'))
        comparison[str(l)]=h.hexdigest();require(h.hexdigest()==reference['original_weight_state'][f'model.layers.{l}.mlp.down_proj.weight'],'W0_WEIGHTS')
    write(out/'runtime.json',dict(instruction=INSTRUCTION,job_id=os.environ.get('SLURM_JOB_ID'),device=torch.cuda.get_device_name(),
        versions=config['runtime'],model=config['model'],profile=a.profile,FP32=True,eager=True,TF32=False,autocast=False,
        tokenizer=dict(class_name=type(tok).__name__,BOS=tok.bos_token_id,EOS=tok.eos_token_id,padding=tok.padding_side,actual_probe_ids=tok('JLZ v10')['input_ids']),
        checkpoint_saved=False,W0_weight_sha=comparison))
    data=json.loads(Path(config['stream']).read_text())
    imports={name:member(m.__file__) for name,m in list(__import__('sys').modules.items())
        if getattr(m,'__file__',None) and Path(m.__file__).is_file() and
        name.startswith(('project.run_scripts.jlz_realized_subject','project.run_scripts.jlz_writer_coupled','project.run_scripts.jlz_pilot.prompts','transformers.models.llama'))}
    write(out/'actual-imports.json',imports)
    write(out/'W0-reuse.json',dict(source=w0,first500_RPN=True,new_forward=0,actual_W0_weights_verified=True))
    return a,bench,data,H


def batch(a,bench,records,H,metadata,config,arm,out,number,qualification=False,initial_previous=None):
    start=time.monotonic();before=state(a,H);before_aux=aux_identity(bench,metadata);pack=bench.prepare(records)
    if len(records)==100:
        require(pack['identity']==config['packed_native'][number-1]['identity'],'PACKED_RUNTIME_IDENTITY')
    write(out/'input.json',dict(identity=pack['identity'],ids=pack['record_ids'],actual_B=len(records)))
    with Transaction(a,H,metadata,bench.contexts) as tx:
        entry=prepare_entry(a,bench,pack,H,config['stats'],config['settings']['fit_microbatch'])
        require(aux_identity(bench,metadata)==before_aux,'ENTRY_CONTEXT_RNG_MUTATION')
        write(out/'entry.json',dict(before=before,aux=before_aux,input=pack['identity'],prior=entry['geometry'],anchors=entry['anchors']))
        if initial_previous is not None:
            require(before==initial_previous['after'] and before_aux==initial_previous['after_aux'],'B1_B2_CONTINUITY')
            write(out.parent/'initial.json',dict(status='MAIN_INITIAL_PASS',source=os.environ['ODEEDIT_SOURCE_COMMIT'],
                config_digest=digest(config),B1_same_evaluated_weight_commit=True,H_once_per_layer=True,
                B1_current_observer_complete=True,B2_own_entry=True,state=before,aux=before_aux,checkpoint_saved=False))
        if qualification:
            qualify(a,entry,out/'qualification')
            entry['first_geometry']={}  # Qualification solve costs remain separate.
        payload,summary=fit(a,entry,arm,out/'fit',number)
        require(state(a,H)==before and aux_identity(bench,metadata)==before_aux,'CANDIDATE_MASTER_MUTATION')
        applied=commit(a,H,entry,payload)
        if qualification:actual_commit_probe(a,entry,payload,out)
        metadata['ledger'].extend(pack['record_ids']);metadata['next_batch']=number+1
        receipt=dict(batch=number,actual_B=len(records),ids=pack['record_ids'],before=before,after=applied['after'],
            before_aux=before_aux,after_aux=aux_identity(bench,metadata),history=applied,candidates=summary['candidates'],
            updates=summary['Adam_updates'],terminal_gradient_measured=True,seconds=time.monotonic()-start,
            source=os.environ['ODEEDIT_SOURCE_COMMIT'],config=digest(config),checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
        write(out/'commit.json',receipt);tx.finish()
    del entry,payload;gc.collect();torch.cuda.empty_cache();return receipt


def check_lock(attempt,config_path):
    lock=json.loads((attempt/'execution.lock.json').read_text())
    require(lock['instruction']==INSTRUCTION and lock['source_commit']==os.environ['ODEEDIT_SOURCE_COMMIT'],'SOURCE_LOCK')
    require(sha(config_path)==lock['config_sha256'],'CONFIG_LOCK')
    for row in lock['source_members']+lock['runtime_sources']+lock['native_reference']:
        require(Path(row['path']).stat().st_size==row['bytes'] and sha(row['path'])==row['sha256'],'SOURCE_IMPORT_CHANGED:'+row['path'])
    return lock


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['q1','main'],required=True);p.add_argument('--arm',choices=['A','B'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True);args=p.parse_args()
    require(args.phase=='q1' or args.arm in ('A','B'),'MAIN_ARM')
    config=json.loads(args.config.read_text());lock=check_lock(args.attempt,args.config)
    out=args.attempt/(args.phase if args.phase=='q1' else 'main-'+args.arm);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();status='TECHNICAL_FAILED';allcommits=[]
    try:
        if args.phase=='main':
            path=args.attempt/'q1/READY.json';ready=json.loads(path.read_text())
            require(ready['status']=='GPU_QUALIFIED' and ready['source']==lock['source_commit'] and ready['config_sha256']==sha(args.config),'Q1_READY_SOURCE_CONFIG')
            write(out/'upstream.json',member(path))
        for arm in (('A','B') if args.phase=='q1' else (args.arm,)):
            lane=out/arm if args.phase=='q1' else out
            lane.mkdir(parents=True,exist_ok=True);a,bench,data,H=setup(config,lane)
            metadata=dict(ledger=[],next_batch=1);write(lane/'initial-state.json',dict(state=state(a,H),aux=aux_identity(bench,metadata)))
            commits=[]
            if args.phase=='q1':
                with Transaction(a,H,metadata,bench.contexts) as tx:
                    with torch.no_grad():a.weights[a.first].view(-1)[0].add_(1);H[a.first].view(-1)[0].add_(1)
                    metadata['ledger'].append('fault');random.random();torch.rand(1,device=a.device)
                require(tx.rollback_verified,'ROLLBACK_PROBE');write(lane/'rollback.json',dict(exact=True,actual_model=True))
                batches=[data[2000:2002],data[2002:2004]]
            else:batches=[data[i:i+100] for i in range(0,500,100)]
            for number,records in enumerate(batches,1):
                if commits:require(state(a,H)==commits[-1]['after'] and aux_identity(bench,metadata)==commits[-1]['after_aux'],'OWN_ENTRY_CONTINUITY')
                receipt=batch(a,bench,records,H,metadata,config,arm,lane/f'batch-{number:02d}',number,
                    qualification=args.phase=='q1' and arm=='A' and number==1,
                    initial_previous=commits[-1] if args.phase=='main' and arm=='A' and number==2 else None)
                commits.append(receipt);allcommits.append((arm,number))
                if args.phase=='main':
                    selected=data[:500] if number==5 else records
                    ob=observe(a,bench,data[:number*100],selected,H,number,lane/f'observe-W{number:02d}',16,[r['case_id'] for r in records])
                    N=500 if number==5 else 100
                    require({k:v['denominator'] for k,v in ob['summary'].items()}==dict(R=N,P=2*N,N=10*N),'OBSERVER_DENOMINATORS')
                    require(aux_identity(bench,metadata)==receipt['after_aux'],'OBSERVER_AUX_MUTATION')
            if args.phase=='q1' and arm=='A':
                for B in (1,3):shape_check(a,bench,data[2000:2000+B],H,config,lane/f'fixed-B{B}')
            write(lane/'chain-complete.json',dict(arm=arm,batches=len(commits),candidates=25*len(commits),updates=24*len(commits),
                endpoint='W5' if args.phase=='main' else 'Q1_B2',state=state(a,H),actual_science=args.phase=='main'))
            del a,bench,H;gc.collect();torch.cuda.empty_cache()
        if args.phase=='q1':write(out/'READY.json',dict(status='GPU_QUALIFIED',instruction=INSTRUCTION,source=lock['source_commit'],
            config_sha256=sha(args.config),candidates=100,updates=96,main_state_transfer=False,main_PASS=False))
        status='COMPLETED'
    except BaseException as exc:
        write(out/'first-error.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
            no_checkpoint=True,completed_commits=allcommits));raise
    finally:
        write(out/'terminal.json',dict(status=status,phase=args.phase,arm=args.arm,instruction=INSTRUCTION,
            source=lock['source_commit'],config_sha256=sha(args.config),commits=allcommits,job_id=os.environ.get('SLURM_JOB_ID'),
            seconds=time.monotonic()-start,checkpoint_saved=False,no_B6=True,
            peak_host_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None))


if __name__=='__main__':main()
