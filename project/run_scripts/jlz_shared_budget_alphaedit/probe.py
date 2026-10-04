"""One same-plan ridge observation; RAM transaction restores before AlphaEdit continuation."""
import gc,time,os
from pathlib import Path
import torch
from project.run_scripts.jlz_realization.writer import Transaction,rng_snapshot,rng_equal
from project.run_scripts.jlz_shared_budget.writer import apply as ridge_apply
from project.run_scripts.jlz_shared_budget.telemetry import Events as RidgeEvents
from .common import write,state,digest,tensor_sha,require

def identity(value):
    if isinstance(value,torch.Tensor):return tensor_sha(value)
    if isinstance(value,dict):return {str(k):identity(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [identity(v) for v in value]
    return value

def same_plan(a,bench,records,current,H,entry,plan,config,root,observer):
    start=time.monotonic();before=state(a,H);rng=rng_snapshot()
    if a.device.type=='cuda':
        resident=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        gpu_free,_=torch.cuda.mem_get_info(a.device)
        allowance=config['resources']['host_mib']*1024**2
        # Additional inner W/H snapshot4.93GiB, prior/temporary3.1GiB, hashing/IO margin5GiB.
        if resident+14*1024**3>allowance or gpu_free<14*1024**3:
            write(root/'unavailable.json',dict(status='PROBE_UNAVAILABLE',reason='BOUNDED_RAM_GPU_HEADROOM',
                resident_bytes=resident,host_limit_bytes=allowance,GPU_free_bytes=gpu_free,reserve_bytes=14*1024**3,
                attempted=False,extra_fit=0,main_continuation='QUALIFIED_ALPHAEDIT'))
            return
    # Entry tensors are read-only by the imported writer. Bind cache/context/plan too.
    cache=digest(identity(entry));plan_hash=digest(identity(plan));contexts=digest(bench.contexts)
    tr=Transaction(a,H)
    with tr:
        events=RidgeEvents(root/'events.jsonl','SAME_PLAN_RIDGE_PROBE',1)
        receipt=ridge_apply(a,entry,plan,H,events,root)
        observation=observer(a,bench,records,current,H,1,root/'post',config)
        # Intentionally no finish: exact W/H/RNG rollback, including temporary H.
    require(tr.rollback_verified and state(a,H)==before and rng_equal(rng),'PROBE_RESTORE')
    require(digest(identity(entry))==cache and digest(identity(plan))==plan_hash and digest(bench.contexts)==contexts,'PROBE_CACHE_CONTEXT_PLAN')
    write(root/'restore.json',dict(status='EXACT_RAM_RESTORED',before=before,after=state(a,H),
        entry_cache_hash=cache,context_hash=contexts,plan_hash=plan_hash,extra_fit=0,extra_solves=5,
        temporary_history_appends=5,persistent_main_history_appends=0,seconds=time.monotonic()-start,
        writer_seconds=receipt['seconds'],observer_seconds=observation['seconds'],noCP=True,
        comparison='own causal upper K/h for ridge, followed by fresh own causal AlphaEdit K/h'))
    del tr;gc.collect();torch.cuda.empty_cache()
