"""Cold Qwen PRICE eval-2K final chain on server3 H200: Q3 knobs, position-0 rule, B1-B20.

Same execution path as qwen_price_hparam_tier2.run (the runner that tuned Q3),
extended to the baseline CF stream first 2000 with all-seen W5/W10/W15/W20 and a
final editable-weight snapshot for the later W20 generation evaluation.
"""
import argparse
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from official.ours.config import resolve,profile,plain
from official.ours.core.jlz_interference_l1.cap_adapter import Adapter
from official.ours.core.jlz_v12r.entry import prepare_entry
from official.ours.core.jlz_interference_l1.cap_fit import fit
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.observe import reduce_rows
from project.run_scripts.jlz_interference_l1 import cap_run as parent
from project.run_scripts.jlz_interference_l1.cap_common import verify,require,state,digest,member,rows_from,validate_rows
from project.run_scripts.jlz_interference_l1.cap_storage import write,guard,Events,ConsoleBudget
from project.run_scripts.jlz_interference_l1.cap_tracking import batch_values,w0_subset,safe_log
from project.run_scripts.jlz_interference_l1.comparison_bridge import metric_row
from project.run_scripts.qwen_price_hparam_tier2.run import TrackedEvents
from scripts.fixed_counterfact import load_prefix
from . import TASK,eot
from .prepare import CELL,ARM,BATCHES,ALL_SEEN,runtime_profile
from .tracking import install,SCHEMA,W0_NAME

ENV='QWEN_FINAL_SOURCE'


def read(path):return json.loads(Path(path).read_text())


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<24),b''):h.update(block)
    return h.hexdigest()


def execute(root):
    root=Path(root);lock=read(root/'execution.lock.json');prep=read(verify(lock['preparation']))
    from project.run_scripts.jlz_interference_l1 import cap_storage
    cap_storage.TASK=parent.TASK=TASK
    require(os.environ.get(ENV)==lock['source_commit'],'FROZEN_SOURCE_ENV')
    for row in lock['source_members']+lock['input_members']:verify(row)
    c=read(verify(prep['parent_config']))['cells'][CELL]
    require(torch.__version__==c['runtime']['torch'] and transformers.__version__==c['runtime']['transformers'],'RUNTIME')
    for row in c['assets']:
        require(Path(row['path']).stat().st_size==row['bytes'] and sha256_file(row['path'])==row['sha256'],'ASSET_CONTENT')
    out=root/'final';out.mkdir(exist_ok=False)
    ConsoleBudget(out/'console-bound-failure.json').install()
    guard(out,lock['storage_reserve_bytes'])
    resolved=resolve('qwen25','qwen25-'+ARM);p=profile(resolved,runtime_profile(c))
    require(digest(plain(p))==digest(read(verify(prep['resolved']))['profile']),'PROFILE_IDENTITY')
    write(out/'resolved-profile.json',plain(p))
    cfg=dict(server='server3',task_id=TASK,arm='QWEN_final_'+ARM+'_pos0',
        attempt=root.name,source_sha=lock['source_commit'],config_sha=digest(plain(p)),
        model='qwen',model_family='QWEN',writer='memit',role='scientific',
        metric_schema=SCHEMA,cohort_role='eval2k_final',tier='final',slice_identity=prep['slice_identity'],
        resolved_config=plain(resolved),resolved_config_sha256=resolved['sha256'],M1=False,preset='override',
        position0_rule=prep['position0_rule']['prepare_source_sha256'])
    tracker=None;status='TECHNICAL_FAILED';started=time.monotonic()
    try:
        tracker=install()(env_file=c['tracking']['env_file'],spool=out/'tracking',config=cfg)
        print(json.dumps(dict(tracking='STARTUP_REMOTE_IDENTITY_VERIFIED',run_id=tracker.run_id,url=tracker.startup['url'])),flush=True)
        torch.set_num_threads(8);random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        require('H200' in torch.cuda.get_device_name(),'GPU_FAMILY')
        model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,dtype=torch.float32,
            attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
        tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
        tok.pad_token=tok.eos_token;tok.padding_side='right'
        rule=eot.install(tok)
        require(rule['prepare_source_sha256']==prep['position0_rule']['prepare_source_sha256'],'POS0_RULE_IDENTITY')
        a=Adapter(model,p);bench=CounterFactAdapter(tok,read(c['contexts']))
        H={l:torch.zeros(d[1],d[1],dtype=torch.float32) for l,d in a.dims.items()}
        cold=state(a,H);require(cold==c['cold_W0_H0']==prep['W0_cold'],'COLD_W0_H0')
        records=load_prefix(Path(c['stream']).parent,2000)
        require(digest(records)==prep['slice_identity'],'EVAL_ORDER')
        identities=read(verify(prep['observer_identity']))['rows']
        write(out/'initial.json',dict(cold=cold,RNG=parent.rng_identity(),profile_sha=digest(plain(p)),
            source=lock['source_commit'],slice=prep['slice_identity'],position0_rule=rule,job_id=os.environ['SLURM_JOB_ID']))
        # W0 is measured here on the same H200 runtime as the edited chain (no cross-GPU reuse).
        ids=[r['case_id'] for r in records]
        w0result=parent.observer(a,bench,records,records,H,'W0',out/'W0',c,identities,ids)
        require(state(a,H)==cold,'W0_NO_MUTATION')
        w0=rows_from(out/'W0',cold);w0summary=validate_rows(w0,identities,ids,'W0')
        require(w0summary==w0result['summary'],'W0_RAW_REDUCER')
        write(out/'W0-result.json',dict(status='W0_COMPLETE',summary=w0summary,state=cold,device=torch.cuda.get_device_name()))
        safe_log(tracker,lambda:dict(edits=0,batch=0,**metric_row(W0_NAME,w0summary,2000)),'W0_measured')
        previous=cold;previous_rng=parent.rng_identity();cursor=[];commits=[]
        for number in range(1,BATCHES+1):
            batch_started=time.monotonic()
            guard(out,lock['storage_reserve_bytes'])
            current=records[(number-1)*100:number*100];seen=records[:number*100]
            folder=out/f'batch-{number:02d}';folder.mkdir()
            require(state(a,H)==previous and parent.rng_identity()==previous_rng,'W_H_RNG_JOIN')
            pack=bench.prepare(current);expected=prep['packs'][number-1]
            require(pack['identity']==expected['identity'] and pack['record_ids']==expected['ids'],'PACK_IDENTITY')
            require(all(x>0 for x in pack['lookup']),'POS0_ROW_REACHED_ENTRY')
            if number==1:pre=reduce_rows([r for r in w0 if r['case_id'] in set(pack['record_ids'])])
            else:pre=parent.observer(a,bench,seen,current,H,f'B{number}_PRE',folder/'pre',c,identities,pack['record_ids'])['summary']
            with parent.BatchTransaction(a,H,bench,cursor) as tx:
                entry=prepare_entry(a,bench,pack,H,c['stats'],requests_per_group=1)
                require(not entry['anchor_guard_diagnostics'],'M1_MUST_BE_INERT')
                entry.update(batch=number,entry_state=previous,entry_state_sha256=digest(previous),
                    source_binding=dict(source=lock['source_commit'],config_sha256=digest(plain(p)),
                        slice_identity=prep['slice_identity'],batch=number,arm=ARM,pack=pack['identity']))
                anchor=entry['anchors'][a.profile['anchor_layer']].detach().cpu()
                rows=[r for r in prep['position0_rows'] if r['batch']==number]
                write(folder/'position0.json',dict(rows=rows,anchor_star=anchor.tolist(),
                    anchor_star_median=float(anchor.median()),anchor_star_max=float(anchor.max()),
                    owners={str(r['owner']):float(anchor[r['owner']]) for r in rows}))
                events=TrackedEvents(Events(folder/'events.jsonl',ARM,number),tracker)
                plan=fit(a,entry,p,events=events)
                require(state(a,H)==previous and parent.rng_identity()==previous_rng,'FIT_NONMUTATION')
                write(folder/'fit.json',plan['receipt']);write(folder/'size-diagnostics.json',events.diagnostics(),limit=1024**2)
                writer=parent.commit_measure(a,entry,H,plan,folder/'writer')
                del entry,plan;gc.collect();torch.cuda.empty_cache()
                require(writer['history_appends']==5,'H_ONCE')
                selected=seen if number in ALL_SEEN else current
                post=parent.observer(a,bench,seen,selected,H,f'W{number}',folder/'post',c,identities,pack['record_ids'])
                cursor.extend(pack['record_ids'])
                after=state(a,H);after_rng=parent.rng_identity()
                require(after_rng==previous_rng,'RNG_NONMUTATION')
                commit=dict(batch=number,arm=ARM,before=previous,after=after,
                    RNG_before=previous_rng,RNG_after=after_rng,source=lock['source_commit'],
                    config_sha=digest(plain(p)),pack=pack['identity'],history_appends=5,
                    pre=pre,post=post,writer=member(folder/'writer/commit.json'),
                    seconds=time.monotonic()-batch_started,checkpoint_saved=False)
                write(folder/'commit.json',commit);tx.finish()
            previous,previous_rng=after,after_rng;commits.append(commit)
            def values():
                value=batch_values(pre,post['current'],post['summary'] if number in ALL_SEEN else None,number)
                value.update(w0_subset(w0,pack['record_ids'],'w0/current/N'))
                if number in ALL_SEEN:value.update(w0_subset(w0,[r['case_id'] for r in seen],'w0/all_seen/N'))
                return value
            safe_log(tracker,values,'batch_commit')
            print(json.dumps(dict(event='BATCH_COMMIT',batch=number,seconds=round(commit['seconds']))),flush=True)
        snapshot=out/'W20-editable-weights.pt'
        torch.save({str(l):a.weights[l].detach().cpu().clone() for l in a.sites},snapshot)
        write(out/'result.json',dict(status='COMPLETE',arm=ARM,batches=BATCHES,source=lock['source_commit'],
            config_sha=digest(plain(p)),slice_identity=prep['slice_identity'],final=commits[-1]['post'],
            state=previous,W20_snapshot=member(snapshot),W20_snapshot_scope='editable down_proj L4-L8 FP32',
            position0_rule=rule,exact_resume='NOT_AVAILABLE'))
        status='COMPLETE'
    except BaseException as error:
        write(out/'first-error.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False))
        raise
    finally:
        write(out/'terminal.json',dict(status=status,seconds=time.monotonic()-started,source=lock['source_commit'],
            GPU_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,automatic_retry=False))
        if tracker is not None:tracker.finish(exit_code=0 if status=='COMPLETE' else 1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    execute(p.parse_args().attempt)
