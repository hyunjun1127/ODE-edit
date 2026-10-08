"""One cold Qwen arm per process, exact old inputs and task-local scalar logging."""
import argparse
import gc
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from official.ours.config import resolve,profile,plain,PRICE_KEYS
from official.ours.core.jlz_interference_l1.cap_adapter import Adapter
from official.ours.core.jlz_v12r.entry import prepare_entry
from official.ours.core.jlz_interference_l1.cap_fit import fit
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.observe import reduce_rows
from project.run_scripts.jlz_realization.writer import rng_equal,rng_snapshot
from project.run_scripts.jlz_interference_l1 import cap_run as parent
from project.run_scripts.jlz_interference_l1.cap_common import verify,require,state,digest,member,rows_from,validate_rows
from project.run_scripts.jlz_interference_l1.cap_storage import write,guard,Events,ConsoleBudget
from project.run_scripts.jlz_interference_l1.cap_tracking import candidate_metrics,batch_values,w0_subset,safe_log
from project.run_scripts.jlz_interference_l1.comparison_bridge import metric_row
from scripts.fixed_counterfact import load_prefix
from . import TASK
from .prepare import OLD
from .tracking import install,SCHEMA


def read(path):return json.loads(Path(path).read_text())


def resolved_profile(c,arm):
    base=c['arm_profiles']['QWEN_M1_CAP075']
    remove=PRICE_KEYS|{'beta_max_native_scale','clamp_factor','kl_factor','norm_factor','K_eval',
                     'eligible_layers','anchor_layer','nll_layer','lambda_C'}
    runtime={k:v for k,v in base.items() if k not in remove}
    resolved=(resolve('qwen25','qwen25-Q4-beta400',override={'lambda_N':0.0})
              if arm=='Q4-beta400-lamN0' else resolve('qwen25','qwen25-'+arm))
    return profile(resolved,runtime),resolved


class TrackedEvents:
    def __init__(self,events,tracker):
        self.events,self.tracker=events,tracker
        self.last=None;self.projection=None;self.first_binding=[None]*100
    def __getattr__(self,name):return getattr(self.events,name)
    def emit(self,event,payload):
        self.events.emit(event,payload)
        if event=='candidate':
            self.last=payload
            if 'projection' in payload:self.projection=payload['projection']
            for i,(spend,beta) in enumerate(zip(payload['telemetry']['weighted_spend'],payload['controller']['beta'])):
                if self.first_binding[i] is None and spend>=.99*beta:self.first_binding[i]=payload['candidate']
            safe_log(self.tracker,lambda:candidate_metrics(payload,self.events.batch),'candidate')
    def diagnostics(self):
        row=self.last;projection=self.projection
        active=[] if projection is None else projection['shared_active']
        layers=row['telemetry']['layers']
        normalized=[sum(layers[l]['normalized_norm'][i] for l in layers) for i in range(100)]
        return dict(last_shared_active_fraction=sum(active)/len(active) if active else 0.,
            last_shared_active_basis='last proposal that produced terminal R; zero updates means identity',
            first_candidate_spend_ge_99pct_beta=self.first_binding,
            F_quantiles=dict(zip(('min','p25','median','p75','max'),np.quantile(row['F'],[0,.25,.5,.75,1]).tolist())),
            sum_requested_norm_over_anchor=normalized,KL=row['KL'],terminal_states=row['states'],
            last_candidate=row['candidate'],projection=projection,extra_forward=0)


def execute(root,stage,arm,batches):
    root=Path(root);lock=read(root/'execution.lock.json');prep=read(verify(lock['preparation']))
    from project.run_scripts.jlz_interference_l1 import cap_storage
    cap_storage.TASK=parent.TASK=TASK
    require(os.environ.get('QWEN_TUNING_SOURCE')==lock['source_commit'],'FROZEN_SOURCE_ENV')
    for row in lock['source_members']+lock['input_members']:verify(row)
    c=read(verify(prep['parent_config']))['cells']['QWEN_M1_CAP075']
    require(torch.__version__==c['runtime']['torch'] and transformers.__version__==c['runtime']['transformers'],'RUNTIME')
    for row in c['assets']:
        st=Path(row['path']).stat()
        require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_CHANGED')
    smoke=stage=='smoke';isw0=stage=='w0'
    require((stage in ('smoke','tier1') and batches==1) or (stage=='tier2' and batches==5)
            or (isw0 and batches==0),'SEALED_STAGE_HORIZON')
    if stage!='smoke':require(read(root/'smoke-Q0/result.json')['smoke_WH_equal'],'SMOKE_REQUIRED')
    out=root/(stage+'-'+arm);out.mkdir(exist_ok=False)
    ConsoleBudget(out/'console-bound-failure.json').install()
    guard(out,lock['storage_reserve_bytes'])
    p,resolved=resolved_profile(c,arm);write(out/'resolved-profile.json',plain(p))
    cfg=dict(server='server4',task_id=TASK,arm='QWEN_'+stage+'_'+arm,
        attempt=root.name,source_sha=lock['source_commit'],config_sha=digest(plain(p)),
        model='qwen',model_family='QWEN',writer='memit',role='validation' if smoke else 'scientific',
        metric_schema=SCHEMA,cohort_role='validation' if smoke else 'heldout_tuning',tier=stage,
        slice_identity=digest(prep['smoke_pack']) if smoke else prep['slice_identity'],
        resolved_config=plain(resolved),resolved_config_sha256=resolved['sha256'],M1=True,
        preset='native' if arm=='Q7-native' else 'base' if arm=='Q0' else 'override')
    tracker=None;status='TECHNICAL_FAILED';started=time.monotonic()
    try:
        tracker=install()(env_file=c['tracking']['env_file'],spool=out/'tracking',config=cfg)
        print(json.dumps(dict(stage=stage,arm=arm,tracking='STARTUP_REMOTE_IDENTITY_VERIFIED',
            run_id=tracker.run_id,url=tracker.startup['url'])),flush=True)
        torch.set_num_threads(8);random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        require('RTX PRO 6000 Blackwell Server Edition' in torch.cuda.get_device_name(),'GPU_FAMILY')
        model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,dtype=torch.float32,
            attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
        tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
        tok.pad_token=tok.eos_token;tok.padding_side='right'
        a=Adapter(model,p);bench=CounterFactAdapter(tok,read(c['contexts']))
        H={l:torch.zeros(d[1],d[1],dtype=torch.float32) for l,d in a.dims.items()}
        cold=state(a,H);require(cold==c['cold_W0_H0'],'COLD_W0_H0')
        records=load_prefix(Path(c['stream']).parent,2500)
        records=records[:100] if smoke else records[2000:2500]
        if not smoke:require(digest(records)==prep['slice_identity'],'HELDOUT_ORDER')
        identities=read(verify(c['observer_identity'] if smoke else prep['observer_identity']))['rows']
        write(out/'initial.json',dict(cold=cold,RNG=parent.rng_identity(),profile_sha=digest(plain(p)),
            source=lock['source_commit'],slice=cfg['slice_identity'],job_id=os.environ['SLURM_JOB_ID']))
        if isw0:
            result=parent.observer(a,bench,records,records,H,'W0',out/'W0',c,identities,[r['case_id'] for r in records])
            safe_log(tracker,lambda:dict(edits=0,batch=0,**metric_row('W0_first500',result['summary'],500)),'W0_first500')
            write(out/'result.json',dict(status='W0_COMPLETE',summary=result['summary'],state=cold,
                slice_identity=prep['slice_identity'],source=lock['source_commit']))
            status='COMPLETE';return
        w0=[]
        if not smoke:
            w0result=read(root/'w0-Q0/result.json')
            require(w0result['state']==cold and w0result['slice_identity']==prep['slice_identity'],'W0_MATCHED_IDENTITY')
            w0=rows_from(root/'w0-Q0/W0',cold)
            require(validate_rows(w0,identities,[r['case_id'] for r in records],'W0')==w0result['summary'],'W0_RAW_REDUCER')
            safe_log(tracker,lambda:dict(edits=0,batch=0,**metric_row('W0_first500',w0result['summary'],500)),'W0_reuse')
        previous=cold;previous_rng=parent.rng_identity();cursor=[];commits=[]
        for number in range(1,batches+1):
            batch_started=time.monotonic()
            guard(out,lock['storage_reserve_bytes'])
            current=records[(number-1)*100:number*100];seen=records[:number*100]
            folder=out/f'batch-{number:02d}';folder.mkdir()
            require(state(a,H)==previous and parent.rng_identity()==previous_rng,'W_H_RNG_JOIN')
            pack=bench.prepare(current);expected=prep['smoke_pack'] if smoke else prep['packs'][number-1]
            require(pack['identity']==expected['identity'] and pack['record_ids']==expected['ids'],'PACK_IDENTITY')
            pre=None
            if not smoke:
                if number==1:
                    pre=reduce_rows([r for r in w0 if r['case_id'] in set(pack['record_ids'])])
                else:pre=parent.observer(a,bench,seen,current,H,f'B{number}_PRE',folder/'pre',c,identities,pack['record_ids'])['summary']
            with parent.BatchTransaction(a,H,bench,cursor) as tx:
                entry=prepare_entry(a,bench,pack,H,c['stats'],requests_per_group=1)
                entry.update(batch=number,entry_state=previous,entry_state_sha256=digest(previous),
                    source_binding=dict(source=lock['source_commit'],config_sha256=digest(plain(p)),
                        slice_identity=cfg['slice_identity'],batch=number,arm=arm,pack=pack['identity']))
                write(folder/'m1-anchor.json',dict(affected=entry['anchor_guard_diagnostics'],extra_forward=0))
                events=TrackedEvents(Events(folder/'events.jsonl',arm,number),tracker)
                plan=fit(a,entry,p,events=events)
                require(state(a,H)==previous and parent.rng_identity()==previous_rng,'FIT_NONMUTATION')
                write(folder/'fit.json',plan['receipt']);write(folder/'size-diagnostics.json',events.diagnostics(),limit=1024**2)
                writer=parent.commit_measure(a,entry,H,plan,folder/'writer')
                del entry,plan;gc.collect();torch.cuda.empty_cache()
                require(writer['history_appends']==5,'H_ONCE')
                if smoke:
                    old=read(verify(prep['smoke_reference']))
                    equal=writer['weight_hashes']==old['weight_hashes'] and writer['after']['H']==old['after']['H']
                    write(out/'reproduction.json',dict(same_W=writer['weight_hashes']==old['weight_hashes'],
                        same_H=writer['after']['H']==old['after']['H'],reference=prep['smoke_reference'],actual=writer['after']))
                    require(equal,'SMOKE_W_H_HASH_MISMATCH_SWEEP_BLOCKED')
                    post=None
                else:
                    selected=seen if number==5 else current
                    post=parent.observer(a,bench,seen,selected,H,f'W{number}',folder/'post',c,identities,pack['record_ids'])
                cursor.extend(pack['record_ids'])
                after=state(a,H);after_rng=parent.rng_identity()
                require(after_rng==previous_rng,'RNG_NONMUTATION')
                commit=dict(batch=number,arm=arm,before=previous,after=after,
                    RNG_before=previous_rng,RNG_after=after_rng,source=lock['source_commit'],
                    config_sha=digest(plain(p)),pack=pack['identity'],history_appends=5,
                    pre=pre,post=post,writer=member(folder/'writer/commit.json'),
                    seconds=time.monotonic()-batch_started,checkpoint_saved=False)
                write(folder/'commit.json',commit);tx.finish()
            previous,previous_rng=after,after_rng;commits.append(commit)
            if not smoke:
                def values():
                    value=batch_values(pre,post['current'],post['summary'] if number==5 else None,number)
                    value.update(w0_subset(w0,pack['record_ids'],'w0/current/N'))
                    if number==5:value.update(w0_subset(w0,[r['case_id'] for r in seen],'w0/all_seen/N'))
                    return value
                safe_log(tracker,values,'batch_commit')
            print(json.dumps(dict(event='BATCH_COMMIT',stage=stage,arm=arm,batch=number)),flush=True)
        write(out/'result.json',dict(status='COMPLETE',stage=stage,arm=arm,batches=batches,
            source=lock['source_commit'],config_sha=digest(plain(p)),slice_identity=cfg['slice_identity'],
            smoke_WH_equal=smoke,final=commits[-1]['post'],state=previous,
            checkpoint_saved=False,exact_resume='NOT_AVAILABLE',cold_B1_refit_for_tier2=True))
        status='COMPLETE'
    except BaseException as error:
        write(out/'first-error.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False))
        raise
    finally:
        write(out/'terminal.json',dict(status=status,stage=stage,arm=arm,seconds=time.monotonic()-started,
            source=lock['source_commit'],GPU_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            no_checkpoint=True,archive='NOT_APPLICABLE_NO_CHECKPOINT',automatic_retry=False))
        if tracker is not None:tracker.finish(exit_code=0 if status=='COMPLETE' else 1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--stage',choices=['smoke','w0','tier1','tier2'],required=True)
    p.add_argument('--arm',required=True);p.add_argument('--batches',type=int,required=True)
    args=p.parse_args();execute(args.attempt,args.stage,args.arm,args.batches)
