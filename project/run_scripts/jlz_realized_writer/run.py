"""One B1 fit, five branch transactions, no scheduler/retry/continuation code."""
import argparse,gc,json,os,random,resource,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.observe import observe
from project.run_scripts.jlz_realization.writer import Transaction,rng_snapshot,rng_equal
from project.run_scripts.jlz_shared_budget.entry import prepare_entry
from project.run_scripts.jlz_shared_budget.optimize import fit
from project.run_scripts.jlz_shared_budget.telemetry import Events
from project.run_scripts.jlz_shared_budget.qualification import qualify
from project.run_scripts.jlz_two_arm.baseline_pilot import _rng_hash
from .common import *
from .capture import Adapter,capture_native_sites,virtual_terminal,cache_identity
from .writer import apply_sequential

def locked(attempt):
    c=json.loads((attempt/'config.json').read_text());lock=json.loads((attempt/'execution.lock.json').read_text())
    require(c['instruction_id']==lock['instruction_id']==NONCE and c['task_id']==TASK,'AUTHORITY')
    require(os.environ.get('ODEEDIT_SOURCE_COMMIT')==lock['source_commit'] and sha(attempt/'config.json')==lock['config_sha256'],'FROZEN_SOURCE')
    for row in lock['source_members']+lock['runtime_sources']+lock['native_reference']+lock['dependency_sources']:verify(row)
    verify(c['native_input_alignment']);verify(c['observer_identity']);verify(lock['native_hparams'])
    for row in c['assets']:
        st=Path(row['path']).stat();require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_CHANGED')
    require(c['settings']['B']==100 and c['settings']['batches']==1 and c['settings']['fit_count']==1 and tuple(c['settings']['branches'])==BRANCHES,'B1_ONLY')
    return c,lock

def setup(c,out):
    require(shutil.disk_usage(out).free>=c['resources']['reserve_bytes'],'RESOURCE_BLOCKED_STORAGE')
    require(torch.__version__==c['runtime']['torch'] and transformers.__version__==c['runtime']['transformers'],'RUNTIME_VERSION')
    torch.set_num_threads(8);random.seed(20261002);np.random.seed(20261002);torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,dtype=torch.float32,
        attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
    tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    a=Adapter(model,c['profile']);bench=CounterFactAdapter(tok,json.loads(Path(c['contexts']).read_text()))
    H={l:torch.zeros(dim[1],dim[1],dtype=torch.float32) for l,dim in a.dims.items()}
    records=load_prefix(Path(c['stream']).parent,100);pack=bench.prepare(records)
    require(pack['identity']==c['packing']['identity'] and pack['record_ids']==c['packing']['ids'],'INPUT_PACK_LOCK')
    write(out/'runtime.json',dict(source=os.environ['ODEEDIT_SOURCE_COMMIT'],job=os.environ.get('SLURM_JOB_ID'),
        device=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,
        cold_W0_H0=True,FP32=True,geometry_FP64=True,eager=True,TF32=False,autocast=False,fit_count_authorized=1,
        B=100,batches=1,no_B2=True,checkpoint_saved=False))
    return a,bench,records,pack,H

def observer(a,bench,records,H,name,out,c):
    rng=rng_snapshot();before=state(a,H)
    result=observe(a,bench,records,records,H,name,out,c['settings']['observer_microbatch'])
    require(rng_equal(rng) and state(a,H)==before,'OBSERVER_STATE_RNG')
    require({k:v['denominator'] for k,v in result['summary'].items()}==dict(R=100,P=200,N=1000),'ENDPOINT_DENOMINATORS')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);args=p.parse_args()
    attempt=args.attempt;out=attempt/'B1';out.mkdir(exist_ok=False);started=time.monotonic();status='TECHNICAL_BLOCKED';completed=[];a=None
    try:
        c,lock=locked(attempt);a,bench,records,pack,H=setup(c,out);cold=state(a,H)
        # Deliberate RAM-only rollback probe: no optimizer or target fit.
        transaction=Transaction(a,H)
        with transaction:
            with torch.no_grad():a.weights[a.first].view(-1)[0].add_(1);H[a.first].view(-1)[0].add_(1)
        require(transaction.rollback_verified and state(a,H)==cold,'ACTUAL_RAM_RESTORE')
        write(out/'rollback-probe.json',dict(verified=True,additional_fit=0))
        # Fixed same-candidate checks on only the first two B1 inputs; not a BS2 chain.
        small=prepare_entry(a,bench,bench.prepare(records[:c['qualification']['B1_subset_requests']]),H,c['stats'],1)
        qualify(a,bench,small,c,out/'qualification');del small;gc.collect();torch.cuda.empty_cache()
        require(state(a,H)==cold,'QUALIFICATION_MUTATION')
        write(out/'initial.json',dict(actual_B1_input_checks=True,fit_count=0,branch_endpoints=0,
            GPU_qualification_scope='fixed same-candidate planner/native keys and RAM rollback; new writer actual checks occur per branch',
            source=lock['source_commit'],new_BS2_chain=False))
        observer(a,bench,records,H,'W0',out/'W0',c)
        entry=prepare_entry(a,bench,pack,H,c['stats'],1);initial=capture_native_sites(a,entry,a.sites)
        events=Events(out/'fit-events.jsonl',lock['source_commit']+':V13_PLAN',1)
        a.capture_virtual=True
        plan,fit_receipt=fit(a,entry,events,out/'fit')
        a.capture_virtual=False;virtual=virtual_terminal(a,entry,plan)
        require(state(a,H)==cold,'FIT_MUTATED_W_H')
        # Full raw plan/targets never leave RAM; only hashes in receipts.
        plan_id=digest({name:{l:tensor_sha(v) for l,v in plan[name].items()} for name in ('u','D','z')})
        virtual_id=digest({l:tensor_sha(v) for l,v in virtual.items()})
        cache=cache_identity(entry);context=digest(bench.contexts);baseline_rng=rng_snapshot();guard=a.guard();hooks=a.hook_signature()
        write(out/'plan.json',dict(identity=plan_id,terminal_z_same_forward=True,fit_count=1,receipt=fit_receipt,
            layers=a.sites,ids=pack['record_ids'],native_pack=pack['identity'],virtual_all_rows_identity=virtual_id,persisted_plan_tensors=False))
        for branch in BRANCHES:
            root=out/branch;before=state(a,H)
            require(before==cold and rng_equal(baseline_rng) and cache_identity(entry)==cache and digest(bench.contexts)==context,'BRANCH_ENTRY_RESTORE')
            tx=Transaction(a,H);logical_ledger=tuple(completed)
            try:
                with tx:
                    writer=apply_sequential(a,entry,plan,virtual,initial,H,branch,root)
                    observed=observer(a,bench,records,H,branch,root/'evaluation',c)
                    endpoint=state(a,H)
                    write(root/'endpoint.json',dict(branch=branch,state=endpoint,plan_identity=plan_id,fit_count=0,
                        history_appends=writer['history_appends'],metrics=observed['summary'],observer_no_mutation=True))
                    # Intentionally do not finish: restore branch after its observation.
            finally:
                restored=(tx.rollback_verified and state(a,H)==cold and rng_equal(baseline_rng)
                    and cache_identity(entry)==cache and digest(bench.contexts)==context and a.guard()==guard
                    and a.hook_signature()==hooks and tuple(completed)==logical_ledger
                    and digest({l:tensor_sha(v) for l,v in virtual.items()})==virtual_id
                    and digest({name:{l:tensor_sha(v) for l,v in plan[name].items()} for name in ('u','D','z')})==plan_id)
                write(root/'restore.json',dict(verified=restored,W_H_RNG=True if restored else 'NOT_VERIFIED',
                    plan_cache_context_ledger_nonselected=restored,checkpoint_saved=False))
                require(restored,'BRANCH_RESTORE_FAILURE')
            completed.append(branch)
            write(root/'complete.json',dict(branch=branch,restore_verified=True,endpoint=member(root/'endpoint.json')))
            print(dict(event='V13_BRANCH_COMPLETE',branch=branch,fit_count=1),flush=True)
        require(tuple(completed)==BRANCHES,'FIVE_BRANCH_COVERAGE');status='B1_FIVE_WRITERS_COMPLETE'
        imported=[]
        for name,module in list(sys.modules.items()):
            path=getattr(module,'__file__',None)
            if isinstance(path,str) and Path(path).is_file() and name.startswith(('project.run_scripts.','memit.','rome.','util.','transformers.models.llama')):
                imported.append(dict(module=name,**member(path)))
        write(out/'actual-imports.json',dict(files=imported))
    except BaseException as error:
        write(out/'first-error.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),sourceKEEP=True));raise
    finally:
        write(out/'terminal.json',dict(status=status,branches_completed=completed,fit_count=1 if (out/'plan.json').exists() else 'NOT_COMPLETED',
            seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_VRAM_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            job=os.environ.get('SLURM_JOB_ID'),adapter_calls=None if a is None else a.calls,
            call_units='full_adapter_forward includes native_full; cached native separate. layer_function entries count pure checkpoint function entries, not whole-model F. Observer forwards tracked by original rows/microbatch.',
            no_B2=True,checkpoint_saved=False))

if __name__=='__main__':main()
