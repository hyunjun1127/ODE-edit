"""Persistent W0 frozen-target pool then one explicitly locked own W/H chain."""
import argparse
import gc
import json
import os
import random
import resource
import shutil
import time
import traceback
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.observe import observe,active_flags,reduce_rows
from . import *
from .adapter import Adapter,solve_write
from .profile import profile
from . import telemetry

def seed():
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def rng():return digest([repr(random.getstate()),repr(np.random.get_state()),tensor_sha(torch.get_rng_state()),[tensor_sha(s) for s in torch.cuda.get_rng_state_all()]])

def rows(folder):
    return [r for p in sorted(Path(folder).glob('chunk-*.json')) for r in json.loads(p.read_text())['rows']]

def observer(a,bench,seen,selected,H,name,out,c,identities):
    before=rng();result=observe(a,bench,seen,selected,H,name,out,c['settings']['observer_microbatch'])
    actual=rows(out);ids={r['case_id'] for r in selected};expected=[r for r in identities if r['case_id'] in ids]
    require(len(actual)==len(expected)==13*len(selected),'OBSERVER_DENOMINATOR')
    for x,y in zip(actual,expected):
        require(all(x[k]==y[k] for k in y),'OBSERVER_ROW_TOKEN_IDENTITY')
    require(before==rng(),'OBSERVER_RNG_MUTATION')
    telemetry.evaluation(getattr(a,'tracking',None),result,len(seen)//100 if name!='W0' else 0)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);attempt=p.parse_args().attempt
    tracking=None;horizon=None
    out=attempt/'main';out.mkdir(exist_ok=False);start=time.monotonic();status='TECHNICAL_FAILED';commits=[];c=None
    try:
        c=json.loads((attempt/'config.json').read_text());lock=json.loads((attempt/'execution.lock.json').read_text())
        horizon=profile(c['settings']['requests']);require(c['profile']==horizon,'PROFILE_IDENTITY')
        require(c.get('tracking',{}).get('integration')=='SH1_SHARED_HELPER_BOUND','LOGGING_BLOCKED_BEFORE_MODEL')
        require(c['instruction_id']==NONCE and os.environ['FE_SOURCE_COMMIT']==lock['source_commit'] and sha(attempt/'config.json')==lock['config_sha256'],'FROZEN_SOURCE')
        for key in ('source_members','launchers'):
            for row in lock[key]:verify(row)
        for row in c['upstream_members']+c['runtime']['source_members']+c['authority_members']+[c['specs'],c['observer_identity'],c['cpu_preflight']]:verify(row)
        for row in c['assets']:
            st=Path(row['path']).stat();require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_CHANGED')
        verify(c['tracking']['env_member'])
        for row in c['tracking']['helper_members']:verify(row)
        tracking=telemetry.start(attempt,c,lock,os.environ['SLURM_JOB_ID'])
        tracking.log({'phase_id':0,'resource/gpus':1,'resource/cpus':6})
        records=load_prefix(Path(c['stream']).parent,horizon['requests'])
        require(digest([digest(r) for r in records])==c['record_root'],'FULL_RECORD_ORDER')
        specs=json.loads(verify(c['specs']).read_text());identities=json.loads(verify(c['observer_identity']).read_text())
        torch.set_num_threads(6);seed();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        require(torch.__version__==c['runtime']['torch'] and transformers.__version__==c['runtime']['transformers'],'RUNTIME')
        model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
        tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
        contexts=json.loads(Path(c['contexts']).read_text());a=Adapter(model,tok,contexts,c['upstream']);bench=CounterFactAdapter(tok,contexts)
        a.tracking=tracking
        H={l:torch.zeros(14336,14336,dtype=torch.float32) for l in LAYERS};cold=state(a,H)
        require(cold==c['cold_W0_H0'],'ACTUAL_COLD_W0_H0')
        write(out/'runtime.json',dict(source=lock['source_commit'],config=digest(c),torch=torch.__version__,transformers=transformers.__version__,
            device=torch.cuda.get_device_name(),cold=cold,FP32=True,geometry_FP64=True,eager=True,TF32=False,cpu_threads=6))
        tracking.log({'phase_id':1,'edits':0})
        observer(a,bench,records,records,H,'W0',out/'W0',c,identities)
        table=torch.empty(5,4096,horizon['requests'],dtype=torch.float32);pool_start=time.monotonic();pool_guard=a.versions()
        for index,record in enumerate(records):
            table[0,:,index]=a.fit(record,index,out/'targets'/f'{index:04d}-fit.json',specs[index]['pack'])
            if index==0:
                first=json.loads((out/'targets/0000-fit.json').read_text())
                write(out/'first-target.json',dict(status='FIRST_REAL_FIT_REUSED',fit=member(out/'targets/0000-fit.json'),source=lock['source_commit'],
                    naive_all_target_seconds=horizon['requests']*first['seconds'],planned_targets=horizon['requests'],estimate_scope='first request only, not representative; excludes replay/writer/eval/IO',
                    physical_forwards=first['physical_forwards'],activation_recompute=first['checkpoint_recompute']))
            if (index+1)%50==0:print(json.dumps(dict(event='W0_TARGET_FIT',completed=index+1,seconds=time.monotonic()-pool_start)),flush=True)
        fit_seconds=time.monotonic()-pool_start;replay_start=time.monotonic()
        tracking.log({'phase_id':3,'time/phase_seconds':fit_seconds})
        for index,record in enumerate(records):
            result=a.replay(record,table[0,:,index])
            for j,l in enumerate(LAYERS):table[j,:,index]=result[l]
            require(a.versions()==pool_guard,'W0_REPLAY_MUTATION')
            write(out/'targets'/f'{index:04d}-replay.json',dict(index=index,case_id=record['case_id'],record=digest(record),
                hashes={l:tensor_sha(v) for l,v in result.items()},canonical=True,absolute_z4=True,online_refit=False))
            if (index+1)%50==0:tracking.log({'phase_id':3,'candidate':index+1,'time/phase_seconds':time.monotonic()-replay_start})
        require(state(a,H)==cold and bool(torch.isfinite(table).all()),'FROZEN_TARGET_W0_STATE')
        write(out/'target-pool.json',dict(targets=horizon['requests'],layers=list(LAYERS),shape=list(table.shape),bytes=table.numel()*4,
            hash=tensor_sha(table),W0=cold,fit_seconds=fit_seconds,replay_seconds=time.monotonic()-replay_start,
            fit_calls=dict(a.calls),table_storage='CPU_RAM_ONLY',target_fits=horizon['requests'],extra_fits=0))
        seed();C0={}
        for l in LAYERS:
            with np.load(c['stats'][str(l)],allow_pickle=False) as z:
                C0[l]=torch.from_numpy(z['mom2.mom2']);count=float(z['mom2.count'].item())
            require(C0[l].dtype==torch.float32 and C0[l].shape==H[l].shape and count>0 and count.is_integer(),'C0_IDENTITY')
            C0[l].div_(int(count));require(bool(torch.isfinite(C0[l]).all()),'NONFINITE_C0')
        previous=cold;previous_rng=rng();guard=a.guard();previous_commit=None
        for batch in range(1,horizon['batches']+1):
            folder=out/f'batch-{batch:02d}';folder.mkdir();current=records[(batch-1)*100:batch*100];seen=records[:batch*100]
            require(shutil.disk_usage(out).free>c['resources']['reserve_bytes'],'DISK_RESERVE')
            require(state(a,H)==previous and rng()==previous_rng and a.guard()==guard,'OWN_W_H_RNG_LINK')
            write(folder/'entry.json',dict(batch=batch,before=previous,RNG=previous_rng,parent=previous_commit,
                occurrence_indices=list(range((batch-1)*100,batch*100)),target_anchor='FROZEN_W0'))
            if batch==1:
                wanted={r['case_id'] for r in current};subset=[dict(r,endpoint='B1_PRE',active_at_endpoint=active_flags(seen)[r['case_id']]) for r in rows(out/'W0') if r['case_id'] in wanted]
                write(folder/'pre/chunk-0000.json',dict(rows=subset,state=previous,optimizer_feedback=False))
                write(folder/'pre/summary.json',dict(summary=reduce_rows(subset),seconds=0,new_forwards=0,reused='W0'))
            else:observer(a,bench,seen,current,H,f'B{batch}_PRE',folder/'pre',c,identities)
            Wcopy={l:w.detach().clone() for l,w in a.weights.items()};Hcopy={l:h.clone() for l,h in H.items()}
            elapsed=time.monotonic();write_calls=[]
            tracking.log({'phase_id':4,'batch':batch,'edits':(batch-1)*100})
            try:
                for j,l in enumerate(LAYERS):
                    torch.cuda.synchronize();key_start=time.monotonic();K,h=a.capture(current,l);torch.cuda.synchronize();key_seconds=time.monotonic()-key_start
                    z=table[j,:,(batch-1)*100:batch*100].to(a.device);R=z-h
                    receipt=solve_write(a.weights[l],H[l],C0[l],K,R);receipt.update(layer=l,key_seconds=key_seconds,target_anchor='W0',sample_offset=(batch-1)*100)
                    write(folder/f'layer-{l}.json',receipt);write_calls.append(receipt);del K,h,z,R;gc.collect();torch.cuda.empty_cache()
                require(a.guard()==guard and rng()==previous_rng,'NONSELECTED_OR_RNG_MUTATION')
                after=state(a,H);write(folder/'commit.json',dict(batch=batch,commits=1,history_appends=5,before=previous,after=after,
                    parent=previous_commit,RNG=previous_rng,ids=[r['case_id'] for r in current],seconds=time.monotonic()-elapsed,
                    target_pool_hash=sha(out/'target-pool.json'),checkpoint_saved=False))
            except BaseException:
                with torch.no_grad():
                    for l in LAYERS:a.weights[l].copy_(Wcopy[l]);H[l].copy_(Hcopy[l])
                write(folder/'rollback.json',dict(restored=state(a,H)==previous,checkpoint_saved=False));raise
            finally:del Wcopy,Hcopy
            commits.append(batch);previous=after;previous_commit=member(folder/'commit.json')
            selected=seen if batch in horizon['milestones'] else current
            observer(a,bench,seen,selected,H,f'W{batch}',folder/'post',c,identities)
            print(json.dumps(dict(event='FE_COMMIT',batch=batch,history_appends=5,requests=batch*100)),flush=True)
        require(len(commits)==horizon['batches'] and a.calls['replay']==horizon['requests'],'CHAIN_COMPLETENESS');status=f"W{horizon['batches']}_COMPLETE"
    except BaseException as e:
        write(out/'first-error.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),original_KEEP=True));raise
    finally:
        write(out/'terminal.json',dict(status=status,source=os.environ.get('FE_SOURCE_COMMIT'),job=os.environ.get('SLURM_JOB_ID'),
            commits=len(commits),seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_VRAM_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            checkpoint_saved=False,exact_resume='NOT_AVAILABLE',planned_profile=horizon))
        if tracking is not None:
            tracking.log({'phase_id':6,'status_code':0 if status.endswith('_COMPLETE') else 1,
                'batch':len(commits),'time/elapsed_seconds':time.monotonic()-start})
            # Bounded network finish cannot replace the original scientific error.
            try:tracking.finish(exit_code=0 if status.endswith('_COMPLETE') else 1)
            except Exception:pass

if __name__=='__main__':main()
