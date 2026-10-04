"""One fit/commit B1, technical checks then CPU collector; no retry/controller."""
import argparse,gc,json,os,random,resource,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.observe import observe
from project.run_scripts.jlz_realization.writer import Transaction,rng_snapshot,rng_equal
from project.run_scripts.jlz_realized_subject.entry import prepare_entry
from .common import *
from .physical import Adapter
from .qualification import check
from .optimize import fit
from .telemetry import commit_measure

def locked(attempt):
    c=json.loads((attempt/'config.json').read_text());lock=json.loads((attempt/'execution.lock.json').read_text())
    require(c['instruction_id']==lock['instruction_id']==NONCE and c['task_id']==TASK,'AUTHORITY')
    require(os.environ.get('ODEEDIT_SOURCE_COMMIT')==lock['source_commit'] and sha(attempt/'config.json')==lock['config_sha256'],'FROZEN_SOURCE')
    for row in lock['source_members']+lock['runtime_sources']+lock['native_reference']+lock['dependency_sources']:verify(row)
    verify(c['native_input_alignment']);verify(c['observer_identity']);verify(lock['native_hparams'])
    for row in c['assets']:
        st=Path(row['path']).stat();require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_CHANGED')
    require(c['settings']['B']==100 and c['settings']['batches']==1 and c['settings']['fit_count']==1,'B1_ONLY')
    return c,lock

def observer(a,bench,records,H,name,out,c):
    rng=rng_snapshot();before=state(a,H)
    result=observe(a,bench,records,records,H,name,out,c['settings']['observer_microbatch'])
    require(rng_equal(rng) and state(a,H)==before,'OBSERVER_STATE_RNG')
    require({k:v['denominator'] for k,v in result['summary'].items()}==dict(R=100,P=200,N=1000),'ENDPOINT_DENOMINATORS')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);args=p.parse_args()
    attempt=args.attempt;out=attempt/'B1';out.mkdir(exist_ok=False);started=time.monotonic();status='TECHNICAL_BLOCKED';fits=commits=0
    try:
        c,lock=locked(attempt)
        require(shutil.disk_usage(out).free>=c['resources']['reserve_bytes'],'RESOURCE_BLOCKED_STORAGE')
        require(torch.__version__==c['runtime']['torch'] and transformers.__version__==c['runtime']['transformers'],'RUNTIME')
        torch.set_num_threads(8);random.seed(20261002);np.random.seed(20261002);torch.manual_seed(20261002)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,dtype=torch.float32,
            attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
        tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
        a=Adapter(model,c['profile']);bench=CounterFactAdapter(tok,json.loads(Path(c['contexts']).read_text()))
        H={l:torch.zeros(d[1],d[1],dtype=torch.float32) for l,d in a.dims.items()}
        records=load_prefix(Path(c['stream']).parent,100);pack=bench.prepare(records)
        require(pack['identity']==c['packing']['identity'] and pack['record_ids']==c['packing']['ids'],'INPUT_PACK_LOCK')
        cold=state(a,H)
        write(out/'runtime.json',dict(source=lock['source_commit'],device=torch.cuda.get_device_name(),torch=torch.__version__,
            model=c['model'],cold_W0_H0=True,FP32=True,geometry_FP64=True,eager=True,TF32=False,autocast=False,B=100,fit_count_authorized=1))
        tx=Transaction(a,H)
        with tx:
            with torch.no_grad():a.weights[a.first].view(-1)[0].add_(1);H[a.first].view(-1)[0].add_(1)
        require(tx.rollback_verified and state(a,H)==cold,'ACTUAL_RAM_RESTORE')
        write(out/'rollback-probe.json',dict(verified=True,fit_count=0))
        # Input-bound fixed candidate checks only. No BS2 sequential fit.
        for count in c['qualification']['B1_subset_shapes']:
            small=prepare_entry(a,bench,bench.prepare(records[:count]),H,c['stats'],2)
            check(a,small,out/'qualification'/f'B{count}',zero=count==1)
            del small;gc.collect();torch.cuda.empty_cache()
            require(state(a,H)==cold,'QUALIFICATION_MUTATION')
        write(out/'initial.json',dict(actual_B1_subset_checks=True,fit_count=0,main_complete=False))
        observer(a,bench,records,H,'W0',out/'W0',c)
        entry=prepare_entry(a,bench,pack,H,c['stats'],c['settings']['fit_microbatch'])
        tx=Transaction(a,H)
        with tx:
            fits+=1;plan=fit(a,entry,out/'fit')
            require(state(a,H)==cold,'FIT_MUTATED_ENTRY')
            writer=commit_measure(a,entry,H,plan,out/'V14_RD');commits+=1
            observed=observer(a,bench,records,H,'V14_RD',out/'V14_RD'/'evaluation',c)
            write(out/'V14_RD'/'complete.json',dict(state=state(a,H),history_appends=writer['history_appends'],
                metrics=observed['summary'],observer_no_mutation=True,commit_count=1,terminal_candidate=plan['built']['candidate']))
            tx.finish()
        status='B1_COMPLETE'
    except BaseException as error:
        write(out/'first-error.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),sourceKEEP=True));raise
    finally:
        write(out/'terminal.json',dict(status=status,fit_calls=fits,commit_calls=commits,seconds=time.monotonic()-started,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_VRAM_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            job=os.environ.get('SLURM_JOB_ID'),no_B2=True,checkpoint_saved=False,automatic_retry=False))

if __name__=='__main__':main()
