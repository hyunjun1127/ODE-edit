"""Explicit user-directed T waiver; M-only fresh attempt, no T watcher/dependency."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
from .common import ROOT, member, sha, digest, write
from .control import call, PACKAGE, project_queue
from .admission import freeze_source, external_recheck, inspect_M
from .validation_route import check_waiver, validation_binding, SKIPPED


def freeze(repo, prior_lock, override, attempt):
    authority=json.loads(override.read_text());check_waiver(authority)
    old=json.loads(prior_lock.read_text())
    if old['execution']['head']!='76bb90372b6ddc05f53374812bfc2df90153601e':
        raise ValueError('PRIOR_M_SOURCE_IDENTITY')
    # Reuse CPU82 only for unchanged model/method/helper modules; routing tested separately.
    same={}
    for name in ('runtime.py','alltoken.py','geometry.py','binding.py','optimizer.py',
                 'observer.py','reuse_bindings.py','technical.py','run.sbatch'):
        prior=Path(old['source_root'])/PACKAGE/name;current=repo/PACKAGE/name
        if sha(prior)!=sha(current):raise ValueError('METHOD_SOURCE_CHANGED:'+name)
        same[name]=dict(prior=member(prior),current=member(current),status='EXACT_BYTES_NOT_GPU_PASS')
    external_recheck(old)
    for key in ('reuse_matrix','reuse_plan','P_star_basis'):
        if member(old[key]['path'])!=old[key]:raise ValueError('REUSE_IDENTITY:'+key)
    for m in (old['reused_native_binding']['native'],old['observer_reuse_binding']['N4_B1'],
              old['observer_reuse_binding']['W0_first1000']):
        if member(m['path'])!=m:raise ValueError('REUSED_INPUT_IDENTITY')
    prior_binding=json.loads(Path(old['technical_evidence']['path']).read_text())
    if member(old['technical_evidence']['path'])!=old['technical_evidence']:
        raise ValueError('PRIOR_BINDING_IDENTITY')
    disk=shutil.disk_usage(ROOT)
    if disk.free<old['disk']['reserve_bytes']:raise ValueError('M_STORAGE_RESERVE_UNAVAILABLE')
    parent=ROOT/'M'/attempt
    if parent.exists():raise ValueError('CREATE_ONCE_M_ATTEMPT_EXISTS')
    execution=freeze_source(repo,parent)
    sealed=parent/'skip-t-all-m-override.json'
    with sealed.open('xb') as f:f.write(override.read_bytes())
    sealed.chmod(0o400)
    inherited=write(parent/'validation-waiver-binding.json',dict(status=SKIPPED,
        full_numerical_validation='NOT_ESTABLISHED',identity=prior_binding['identity'],
        EN_COV_resolution=prior_binding['EN_COV_resolution'],
        prior_partial_evidence=prior_binding['partial_evidence'],prior_binding=old['technical_evidence'],
        waiver=member(sealed),P_star_basis=old['P_star_basis'],
        scope='prior identity and completed roundoff asset reuse only; no new T or full numerical PASS',
        T_partial_projector_null_validation='NOT_RECORDED',FD='NOT_RUN',
        no_new_teacher=True,no_new_T=True))
    lock=dict(old,stage='M',allowed_stages=['M'],attempt=attempt,execution=execution,
        source_root=execution['source_root'],M_root=str(parent/'episodes'),output=str(parent/'unused-shared-output'),
        technical_evidence=inherited,technical_validation_status=SKIPPED,
        full_numerical_validation='NOT_ESTABLISHED',skip_T_override=member(sealed),
        T_job=None,T_failure_path=None,parallel_override=None,old_T_failcancel=False,
        failcancel_responsibility='NOT_APPLICABLE_USER_DIRECTED_SKIP_NEW_ATTEMPT',
        prior_attempt_lock=member(prior_lock),unchanged_method_source=same,
        prior_paired_stop=member(ROOT/'receipts/paired-stop-r1/scheduler-terminal.json'),
        prior_CPU82=member(ROOT/'receipts/cpu-preflight-paired-stop-r1.json'),
        disk=dict(old['disk'],available=disk.free),
        initial_pause='ACTUAL_M_INITIAL_OR_VERIFIED_MAIN_GPU_RESOURCE_PENDING',
        partial_cancelled_geometry_reuse=False,
        partial_cancelled_geometry_reason='no complete factor tensor closure retained; safe W0/native reuse only')
    lock.pop('lock_identity',None);lock['lock_identity']=digest(lock)
    ref=write(parent/'execution.lock.json',lock)
    validation_binding(lock,0)
    print(json.dumps(ref));return ref


def submit(lockpath):
    lock=json.loads(lockpath.read_text());parent=lockpath.parent
    validation_binding(lock,0)
    if lock['technical_validation_status']!=SKIPPED:raise ValueError('WAIVER_ROUTE_ONLY')
    if (parent/'held-inspection.json').exists() or (parent/'submission.json').exists():
        raise ValueError('NO_DUPLICATE_M_SUBMISSION')
    for m in lock['execution']['members']:
        if sha(m['path'])!=m['sha256']:raise ValueError('FROZEN_SOURCE_DRIFT')
    external_recheck(lock)
    queue=project_queue();active=queue.splitlines()
    # Conservatively count all owned server4 admissions, not merely odeedit labels.
    if any('_' in x.split('|')[0] or '[' in x.split('|')[0] for x in active):
        raise ValueError('EXISTING_ARRAY_REQUIRES_EXACT_CAP_ACCOUNTING:'+queue)
    if len(active)>=2:raise ValueError('CAP2_NO_ADMISSION_CAPACITY:'+queue)
    throttle=2-len(active)
    disk=shutil.disk_usage(ROOT)
    if disk.free<lock['disk']['reserve_bytes']:raise ValueError('M_DISK_RESERVE_CHANGED')
    audit=write(parent/'resource-admission.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        queue=queue,node=call(['scontrol','show','node','server4']),available_disk=disk.free,
        free_inodes=os.statvfs(ROOT).f_favail,meminfo=Path('/proc/meminfo').read_text(),
        existing_project_slots=len(active),new_array_throttle=throttle,total_admitted_capacity=2,
        cap=2,hour_hardcap=None,old_terminal_evidence=lock['prior_paired_stop'],
        T_dependency=False,T_monitor=False,T_failcancel=False))
    (parent/'logs').mkdir()
    args=['sbatch','--parsable','--hold','--job-name=odeedit_enfc_M_s4','--array=0-9%'+str(throttle),
        '--time='+str(lock['resources']['wall_hours'])+':00:00',
        f'--output={parent}/logs/%A_%a.out',f'--error={parent}/logs/%A_%a.err',
        lock['source_root']+'/'+PACKAGE+'run.sbatch',lock['source_root'],str(lockpath),'M']
    job=call(args).split(';')[0]
    inspection=call(['scontrol','show','job',job])
    write(parent/'held-inspection.json',dict(job=job,args=args,inspection=inspection,lock=member(lockpath),resource=audit))
    inspect_M(inspection,lock,lockpath,throttle)
    hours=lock['resources']['wall_hours'];days,remainder=divmod(hours,24)
    expected_time=(str(days)+'-' if days else '')+f'{remainder:02d}:00:00'
    if 'Dependency=(null)' not in inspection or 'TimeLimit='+expected_time not in inspection:
        raise ValueError('M_DEPENDENCY_OR_TIME_INSPECTION')
    validation_binding(lock,0)
    call(['scontrol','release',job])
    result=write(parent/'submission.json',dict(job=job,args=args,inspection=inspection,
        mapping={str(i):dict(episode=f'b{i+1:03d}',ordinal=[100*i,100*(i+1)],arms=8,
            native='REUSE' if i==0 else 'RUN_MISSING',N4_pair_eval='REUSE' if i==0 else 'RUN_MISSING') for i in range(10)},
        release_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='M',
        scope='ALL10_M_REGISTERED_RELEASED_NOT_COMPLETION',array_throttle=throttle,S_R_L_registered=0,
        initial_gate_observed=False,T_job=None,T_dependency=False,T_failcancel=False,
        validation=SKIPPED,full_numerical_validation='NOT_ESTABLISHED'))
    print(json.dumps(result));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','submit'])
    p.add_argument('--worktree',type=Path);p.add_argument('--prior-lock',type=Path)
    p.add_argument('--override',type=Path);p.add_argument('--attempt');p.add_argument('--lock',type=Path)
    a=p.parse_args()
    if a.action=='freeze':freeze(a.worktree,a.prior_lock,a.override,a.attempt)
    else:submit(a.lock)


if __name__=='__main__':main()
