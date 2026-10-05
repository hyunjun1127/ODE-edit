"""Frozen bounded qualification then one cold, uninterrupted B100 x20 chain."""
import argparse,gc,json,os,random,resource,shutil,time,traceback,hashlib
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
from .routes import annotate
from .qualification_minimal import check
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
    require(c['settings']['B']==100 and c['settings']['batches']==20 and c['settings']['fit_count']==20 and not c['settings']['save_checkpoints'],'CONTINUOUS20_ONLY')
    require(c['settings']['shared_radius']==1.5 and c['settings']['local_radius']==.75,'BUDGET_AUTHORITY')
    return c,lock

def rng_id():
    r=rng_snapshot()
    return digest([repr(r[0]),r[1][0],r[1][1].tolist(),repr(r[1][2:]),tensor_sha(r[2]),[tensor_sha(t) for t in r[3]]])

def observer(a,bench,records,H,name,out,c,current_ids=None):
    rng=rng_snapshot();before=state(a,H)
    result=observe(a,bench,records,records,H,name,out,c['settings']['observer_microbatch'],current_ids)
    require(rng_equal(rng) and state(a,H)==before,'OBSERVER_STATE_RNG')
    n=len(records);require({k:v['denominator'] for k,v in result['summary'].items()}==dict(R=n,P=2*n,N=10*n),'ENDPOINT_DENOMINATORS')
    return result

def entry_for(a,bench,records,H,c):
    e=annotate(prepare_entry(a,bench,bench.prepare(records),H,c['stats'],c['settings']['fit_microbatch']))
    limits=c['resources']['physical_caps']
    require(all(len(g['global_rows'])<=limits['rows'] and g['padded_tokens']<=limits['padded_tokens'] and g['boundary_bytes']<=limits['boundary_bytes'] for g in e['row_schedule']),'PHYSICAL_RESOURCE_CAP')
    return e

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=Path,required=True);args=parser.parse_args()
    attempt=args.attempt;out=attempt/'main';out.mkdir(exist_ok=False)
    started=time.monotonic();status='TECHNICAL_BLOCKED';fits=commits=0;entry=None;plan=None
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
        records=load_prefix(Path(c['stream']).parent,2000);cold=state(a,H);context=digest(bench.contexts)
        write(out/'runtime.json',dict(source=lock['source_commit'],device=torch.cuda.get_device_name(),torch=torch.__version__,
            model=c['model'],cold_W0_H0=True,FP32=True,geometry_FP64=True,eager=True,TF32=False,autocast=False,B=100,fit_count_authorized=20))
        # One B2 fixed candidate, no sequential pilot or extra optimization fit.
        tx=Transaction(a,H)
        with tx:
            small=entry_for(a,bench,records[:2],H,c)
            qualification=check(a,small,out/'qualification')
            with torch.no_grad():a.weights[a.first].view(-1)[0].add_(1);H[a.first].view(-1)[0].add_(1)
        require(tx.rollback_verified and state(a,H)==cold,'ACTUAL_RAM_RESTORE')
        del small;gc.collect();torch.cuda.empty_cache()
        write(out/'READY.json',dict(status='TECHNICAL_READY',source=lock['source_commit'],config_sha=lock['config_sha256'],
            qualification=member(out/'qualification/qualification.json'),routes=qualification['routes'],
            cold_state=cold,actual_RAM_restore=True,extra_fits=0))
        if c['W0']['mode']=='EXACT_REUSE':
            for m in c['W0']['members']:verify(m)
            summary=json.loads(Path(c['W0']['summary']['path']).read_text())
            require(summary['state']==cold,'W0_COLD_STATE_REUSE')
            write(out/'W0-reuse.json',dict(**c['W0'],actual_cold_state_verified=True,new_forward=0))
        else:observer(a,bench,records,H,'W0',out/'W0',c)
        previous=None;ledger=[]
        for batch in range(1,21):
            folder=out/f'batch-{batch:02d}';group=records[(batch-1)*100:batch*100];before=state(a,H)
            pack=bench.prepare(group);expected=c['packing'][batch-1]
            require(pack['identity']==expected['identity'] and pack['record_ids']==expected['ids'],'INPUT_PACK_LOCK')
            boundary=dict(batch=batch,state=before,rng=rng_id(),context=context,ledger=digest(ledger),rows=pack['identity'])
            if previous is not None:
                require(all(boundary[k]==previous[k] for k in ('state','rng','context','ledger')),'OWN_CHAIN_ENTRY_JOIN')
            write(folder/'entry.json',dict(**boundary,previous_commit=member(out/f'batch-{batch-1:02d}/commit.json') if batch>1 else None,
                previous_join_verified=batch>1,fresh_entry_teachers_anchors_factors=True,zero_history_only_at_cold=batch==1))
            if batch==2:write(out/'initial-handoff.json',dict(status='ACTUAL_B1_COMMIT_OBSERVER_B2_ENTRY_PASS',
                B1=member(out/'batch-01/commit.json'),B2=member(folder/'entry.json'),no_checkpoint=True))
            entry=entry_for(a,bench,group,H,c)
            write(folder/'entry-runtime.json',dict(seconds=entry['seconds'],geometry=entry['geometry'],row_schedule=entry['row_schedule'],
                pack_identity=entry['pack']['identity'],factor_lifetime='this batch only'))
            tx=Transaction(a,H)
            with tx:
                fits+=1;plan=fit(a,entry,folder/'fit',qualification['routes'])
                require(state(a,H)==before,'FIT_MUTATED_ENTRY')
                writer=commit_measure(a,entry,H,plan,folder/'writer')
                observed=observer(a,bench,records[:batch*100],H,f'W{batch}',folder/'evaluation',c,[r['case_id'] for r in group])
                require(digest(bench.contexts)==context,'CONTEXT_MUTATION')
                ledger.extend(r['case_id'] for r in group)
                terminal=dict(batch=batch,state=state(a,H),rng=rng_id(),context=context,ledger=digest(ledger),
                    history_appends=writer['history_appends'],observer_no_mutation=True,metrics=observed['summary'],
                    current=observed['current'],terminal_candidate=plan['built']['candidate'],checkpoint_saved=False)
                require(a.guard()==tx.guard,'NONSELECTED_PARAMETER_MUTATION')
                write(folder/'commit.json',terminal)
                tx.finish()
            # Only published logical commits are eligible for the next entry.
            commits+=1;previous=terminal
            print(json.dumps(dict(event='BATCH_COMMIT',batch=batch,commits=commits)),flush=True)
            del entry,plan;entry=plan=None;gc.collect();torch.cuda.empty_cache()
        require(fits==commits==20 and len(ledger)==2000,'ENDPOINT20_COVERAGE');status='W20_COMPLETE'
    except BaseException as error:
        write(out/'first-error.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            committed_prefix=commits,sourceKEEP=True,automatic_retry=False));raise
    finally:
        write(out/'terminal.json',dict(status=status,fit_calls=fits,commit_calls=commits,history_appends=commits*5,
            entry_joins=max(0,commits-1),seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_VRAM_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            job=os.environ.get('SLURM_JOB_ID'),no_B21=True,checkpoint_saved=False,automatic_retry=False))

if __name__=='__main__':main()
