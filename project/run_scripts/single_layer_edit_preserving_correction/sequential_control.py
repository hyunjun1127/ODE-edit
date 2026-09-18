"""User-directed four-arm S source freeze and held/inspected cap2 admission."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
from .common import ROOT,member,sha,digest,write
from .control import call,PACKAGE,project_queue
from .admission import freeze_source,external_recheck
from .sequential_state import S_ARMS
from .sequential_runner import validate_lock

def authority_check(a):
    if (tuple(a['arms'])!=S_ARMS or a['requests_per_chain']!=1000 or a['batches']!=10 or
        a['start']!='W0_ZERO_M4' or a['cap']!=2 or a['M_dependency'] is not None or a['R_L_allowed'] or
        a['T']!='SKIPPED_USER_DIRECTED' or a['full_numerical_validation']!='NOT_ESTABLISHED'):
        raise ValueError('USER_S_FOUR_SCOPE')

def freeze(repo,prior,authority,attempt):
    a=json.loads(authority.read_text());authority_check(a)
    old=json.loads(prior.read_text());external_recheck(old)
    same={}
    for name in ('runtime.py','binding.py','geometry.py','alltoken.py','optimizer.py','observer.py','retained_native.py'):
        original=Path(old['source_root'])/PACKAGE/name;current=repo/PACKAGE/name
        if sha(original)!=sha(current):raise ValueError('UNCHANGED_CORE_'+name)
        same[name]=dict(sha256=sha(current),prior=member(original),status='EXACT_BYTES_NOT_NUMERICAL_PASS')
    for key in ('technical_evidence','P_star_basis','cold_capsule','teacher_manifest'):
        if member(old[key]['path'])!=old[key]:raise ValueError('ASSET_'+key)
    b1=Path(old['M_root'])/'b001/attempt-v1'
    names=dict(provenance='protected-provenance.json',native_objective='native-objective.json',
        EN_S_unused='native-quality.json')
    names.update({f'{name}_{kind}':f'geometry/{name}'+('-factors.pt' if kind=='factors' else '.json')
                  for name in ('EN-S','EN-F') for kind in ('factors','receipt')})
    reused=dict(members={k:member(b1/v) for k,v in names.items()},WN=old['reused_native_binding']['b1_endpoint_verified'],
        prior_M_lock=member(prior),source_core=same,not_M_completion=True)
    parent=ROOT/'S'/attempt
    if parent.exists():raise ValueError('CREATE_ONCE_S_ATTEMPT')
    execution=freeze_source(repo,parent)
    frozen=parent/'user-authority.json'
    with frozen.open('xb') as f:f.write(authority.read_bytes())
    frozen.chmod(0o400)
    # Whitelist environment/input bindings, not the M-only validation or resource route.
    keys=('snapshot','config4','blue_root','dataset_root','projector','reference_root','cold_capsule','teacher_manifest',
          'editor_sha256','historical_evaluator_root','helper_scripts_root','records_digest','sample_order',
          'torch','transformers','numpy','scipy','seed','TF32_matmul','TF32_cudnn','loss_microbatch','guard_microbatch',
          'official_observer_microbatch','external_members','reused_native_b1','reused_native_binding','observer_reuse_binding',
          'P_star_basis','technical_evidence','full_read','numeric_contract')
    lock={k:old[k] for k in keys}
    disk=shutil.disk_usage(ROOT)
    # Upper physical components, not a hard reserved allocation or user-cleanup claim.
    W=4096*14336*4;M=14336*14336*4;G=2*W
    storage=dict(free_bytes=disk.free,free_inodes=os.statvfs(ROOT).f_favail,
        checkpoint_W_M_bytes=40*(W+M),separate_final_W_bytes=40*W,ideal_delta_bytes=40*G,
        maximum_distinct_gradient_bytes=70*G,maximum_blocked_factor_bytes=40*14336*14336*8,
        new_native_preview_estimate_bytes=36*252213779,raw_temporary_serialization_additional=True,
        original_M_reserve=old['storage_admission'],cleanup='USER_PLANNED_NOT_VERIFIED',SH4_delete_move=False,
        no_checkpoint_waiver_inherited=False,actual_IO_errors_fatal=True)
    lock.update(stage='S_FOUR_USER_DIRECTED',allowed_stages=['S_FOUR_USER_DIRECTED'],attempt=attempt,
        authority=member(frozen),S_arms=list(S_ARMS),S_batches=10,S_root=str(parent/'arms'),
        execution=execution,source_root=execution['source_root'],prior_M_lock=member(prior),
        B1_completed_calculations=reused,T_status='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED',
        M_to_S='USER_DIRECTED_NOT_ESTABLISHED',M_completion_dependency=None,R_L_allowed=False,
        source_unchanged=same,storage=storage,resources=dict(gpus=1,cpus=8,mem_MiB=60416,wall_hours=72,project_cap=2,hour_hardcap=None),
        cost_plan=dict(new_native_fit_max=36,new_native_target_max=3600,history_appends=40,
            gradient_sweeps_upper=70,trial_slots_upper=480,own_state_B2plus=True,planned_not_measured=True,
            walltime_reason='10 sequential correction/observer batches; EN-F4 up to24 trials/batch;72h ceiling not expected usage'),
        checkpoint_policy='all40 selected W4/M4/context/RNG/received ledger/next-index retained; GPUcontinuation NOT_TESTED',
        evaluation_plan='entry/native/selected current after seal; fullseen+Dev B5/B10; original greedy32; noReport256/R/L',
        initial_pause='ACTUAL_S_INITIAL_OR_VERIFIED_MAIN_GPU_RESOURCE_PENDING')
    lock['lock_identity']=digest(lock)
    ref=write(parent/'execution.lock.json',lock);validate_lock(lock)
    print(json.dumps(ref));return ref

def inspect(text,lock,path,throttle):
    required=('UserId=janghj','JobState=PENDING','JobHeldUser','NumCPUs=8','Requeue=0','server4',
        'Command='+lock['source_root']+'/'+PACKAGE+'sequential.sbatch',str(path),
        'ArrayTaskId=0-3','ArrayTaskThrottle='+str(throttle),'Dependency=(null)','TimeLimit=3-00:00:00')
    for item in required:
        if item not in text:raise ValueError('HELD_INSPECTION_'+item)
    if ('mem=59G' not in text and 'mem=60416M' not in text) or 'gres/gpu=1' not in text:
        raise ValueError('RESOURCE_INSPECTION')

def submit(path):
    lock=json.loads(path.read_text());validate_lock(lock);parent=path.parent
    if (parent/'held-inspection.json').exists() or (parent/'submission.json').exists():
        raise ValueError('NO_DUPLICATE_S_SUBMISSION')
    external_recheck(lock)
    queue=project_queue();rows=queue.splitlines()
    if rows:raise ValueError('CAP2_REQUIRES_EXACT_OTHER_ADMISSION_ACCOUNTING:'+queue)
    # The prior two M cancellations must be terminal before freeing their slots.
    prior=call(['sacct','-n','-X','-j','50050_0,50050_1','--format=JobID,State,ElapsedRaw,AllocTRES','-P'])
    if any(x in prior for x in ('RUNNING','PENDING','COMPLETING','CONFIGURING')):
        raise ValueError('M_CANCEL_RELEASE_NOT_CONFIRMED')
    audit=write(parent/'resource-admission.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        project_queue=queue,old_M=prior,node=call(['scontrol','show','node','server4']),
        free_disk=shutil.disk_usage(ROOT).free,free_inodes=os.statvfs(ROOT).f_favail,
        meminfo=Path('/proc/meminfo').read_text(),cap=2,new_array_throttle=2,total_admitted_capacity=2,
        no_M_T_dependency=True,actual_IO_errors_fatal=True,source_archive_lock_writable=True))
    (parent/'logs').mkdir()
    args=['sbatch','--parsable','--hold','--job-name=odeedit_enfc_S4_s4','--array=0-3%2',
        f'--output={parent}/logs/%A_%a.out',f'--error={parent}/logs/%A_%a.err',
        str(Path(lock['source_root'])/PACKAGE/'sequential.sbatch'),lock['source_root'],str(path)]
    job=call(args).split(';')[0];text=call(['scontrol','show','job',job])
    write(parent/'held-inspection.json',dict(job=job,args=args,inspection=text,resource=audit,lock=member(path)))
    inspect(text,lock,path,2)
    call(['scontrol','release',job])
    result=write(parent/'submission.json',dict(job=job,mapping={str(i):a for i,a in enumerate(S_ARMS)},
        args=args,release_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        registered=4,chains=4,batches_per_chain=10,requested_unique=1000,arm_requests=4000,
        cap=2,throttle=2,T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED',
        M_to_S='USER_DIRECTED_NOT_ESTABLISHED',M_dependency=None,R_L_registered=0,initial_observed=False))
    print(json.dumps(result));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','submit'])
    p.add_argument('--worktree',type=Path);p.add_argument('--prior',type=Path);p.add_argument('--authority',type=Path)
    p.add_argument('--attempt');p.add_argument('--lock',type=Path);a=p.parse_args()
    if a.action=='freeze':freeze(a.worktree,a.prior,a.authority,a.attempt)
    else:submit(a.lock)

if __name__=='__main__':main()
