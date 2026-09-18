"""Create-once continuation/M closure and inspected M-only array admission.

No job is submitted by a model process. No S/R/L or automatic next-stage hook.
"""
import argparse
import ast
import datetime
import json
import os
from pathlib import Path
import shutil
import tarfile
from .common import ROOT,member,sha,digest,write
from .control import call,source_closure,PACKAGE,PYTHON
from .technical import NUMERIC


def function_hash(path,name):
    tree=ast.parse(Path(path).read_bytes())
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==name:
            return digest(ast.dump(node,include_attributes=False))
    raise ValueError('MISSING_FUNCTION:'+name)


def applicability(repo,prior_source):
    """Exact numerical modules and unaffected model methods, not whole-source PASS."""
    a=Path(prior_source)/PACKAGE;b=repo/PACKAGE;checks={}
    for name in ('alltoken.py','geometry.py','binding.py'):
        if sha(a/name)!=sha(b/name):raise ValueError('T_CORE_SOURCE_CHANGED:'+name)
        checks[name]=dict(prior=member(a/name),current=member(b/name),status='EXACT_BYTES')
    methods=('__init__','guard','sync_oracles','copy_weight','reset','requests','reference_oracle',
             'protected_oracle','factor_A','covariance','invariant')
    for name in methods:
        old=function_hash(a/'runtime.py',name);new=function_hash(b/'runtime.py',name)
        if old!=new:raise ValueError('T_MODEL_METHOD_CHANGED:'+name)
        checks['Runtime.'+name]=dict(prior_AST=old,current_AST=new,status='AST_EXACT')
    return dict(checks=checks,scope='model/alltoken/geometry/guard numerical route',
        native_reuse_branch='new M-only retained-file/hash/order checks; native fitter and new-fit branch unchanged',
        not_claimed='new M orchestration/observer whole-program GPU parity; actual M initial still required')


def freeze_source(repo,parent):
    relatives=source_closure(repo)
    if call(['git','status','--porcelain','--']+relatives,repo):raise ValueError('UNCOMMITTED_SOURCE_CLOSURE')
    source=parent/'source';source.mkdir(parents=True,exist_ok=False);members=[]
    for rel in relatives:
        data=(repo/rel).read_bytes()
        if rel.endswith('.py'):ast.parse(data)
        target=source/rel;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(data)
        target.chmod(0o400);members.append(dict(relative=rel,**member(target)))
    call(['bash','-n',str(source/PACKAGE/'run.sbatch')])
    archive=parent/'source.tar'
    with archive.open('xb') as f,tarfile.open(fileobj=f,mode='w') as tar:
        for rel in relatives:
            info=tar.gettarinfo(str(source/rel),arcname=rel)
            info.uid=info.gid=info.mtime=0;info.uname=info.gname='';info.mode=0o400
            with (source/rel).open('rb') as src:tar.addfile(info,src)
    return dict(head=call(['git','rev-parse','HEAD'],repo),tree=call(['git','rev-parse','HEAD^{tree}'],repo),
        source_root=str(source),archive=member(archive),members=members)


def external_recheck(prior):
    for m in prior['external_members']:
        p=Path(m['path']);s=p.stat()
        if s.st_size!=m['bytes'] or [s.st_dev,s.st_ino,s.st_mtime_ns]!=m['stat']:
            raise ValueError('IMMUTABLE_EXTERNAL_STAT_CHANGED:'+str(p))


def freeze_resume(repo,prior_lock,attempt):
    old=json.loads(prior_lock.read_text());output=Path(old['output'])
    failure=json.loads((output/'failure.json').read_text())
    if not (failure['stage']=='full_token_nullspace' and 'multiple values' in failure['error'] and 'status' in failure['error']):
        raise ValueError('NOT_RECEIPT_ONLY_RESUME_BOUNDARY')
    if (output/'READY.json').exists():raise ValueError('ALREADY_READY_NO_REPLAY')
    required=['failure.json','native-repeat.json','teacher-fixed-binding.json','noop-repeat.json',
        'direct-cached-gradient.json','all-token-forward-stationarity.json','COV-resolution.json',
        'nonselected-before.json','native1/native-capsule.pt','protected-provenance.json',
        'cached-gradient.pt','physical-gradient.pt','P-star-basis.pt','P-star.json','EN-F-space.json']
    evidence=[member(output/name) for name in required]
    source_match=applicability(repo,old['source_root']);external_recheck(old)
    parent=ROOT/'T'/attempt
    if parent.exists():raise ValueError('CREATE_ONCE_ATTEMPT_ALREADY_EXISTS')
    execution=freeze_source(repo,parent)
    lock=dict(old,execution=execution,source_root=execution['source_root'],output=str(parent/'output'),attempt=attempt,
        numeric_contract=NUMERIC,resume_prior=dict(output=str(output),lock=member(prior_lock),members=evidence,
            source_applicability=source_match,native_fit_new=0,prior_cost_separate=True),
        inherited_lock_immutable=member(prior_lock))
    lock.pop('lock_identity',None);lock['lock_identity']=digest(lock)
    result=write(parent/'execution.lock.json',lock);print(json.dumps(result));return result


def freeze_M(repo,technical_lock,attempt,wall_hours,parallel_override=None):
    tech=json.loads(technical_lock.read_text());out=Path(tech['output'])
    if (out/'failure.json').exists():raise ValueError('T_FAILED_NO_M_SUBMISSION')
    provisional=parallel_override is not None and not (out/'READY.json').exists()
    if provisional:
        override=json.loads(parallel_override.read_text())
        if override['nonce']!='ODEEDIT-GH-SH4-ENFC-T-M-PARALLEL-FAILCANCEL-20260918-R1' or override['M_waits_for_T_ready'] is not False:
            raise ValueError('PARALLEL_AUTHORITY_MISMATCH')
        state=call(['sacct','-n','-X','-j','49928','--format=JobID,User,State','-P'])
        if '49928|janghj|RUNNING' not in state:raise ValueError('T_NOT_RUNNING_FOR_PARALLEL:'+state)
        for name in ('native-repeat','teacher-fixed-binding','noop-repeat','direct-cached-gradient','all-token-forward-stationarity'):
            if json.loads((out/(name+'.json')).read_text())['status']!='PASS':raise ValueError('M_REQUIRED_AVAILABLE_BINDING:'+name)
        ready=dict(status='PROVISIONAL_T_UNRESOLVED',identity=json.loads((out/'runtime-load.json').read_text())['identity'],
            EN_COV_resolution=json.loads((out/'COV-resolution.json').read_text())['resolution'],
            technical_contract=digest(NUMERIC),source=tech['execution'],wall_seconds=None,
            timing=json.loads((out/'runtime-load.json').read_text())['timing'],oracle_work='T_IN_PROGRESS_NOT_FINAL',
            unvalidated=['full T projector/FD/actual nullspace/terminal restore'],
            T_job=49928,T_output=str(out),override=member(parallel_override),
            partial_evidence=[member(out/(n+'.json')) for n in ('runtime-load','native-repeat','teacher-fixed-binding',
                'noop-repeat','direct-cached-gradient','all-token-forward-stationarity','COV-resolution','P-star')])
    else:
        ready=json.loads((out/'READY.json').read_text())
        if ready['status']!='T_READY' or ready['technical_contract']!=digest(NUMERIC):raise ValueError('T_NOT_READY')
    base_source=tech.get('resume_prior',{}).get('source_applicability')
    source_match=applicability(repo,tech['source_root']);external_recheck(tech)
    native=json.loads((ROOT/'reuse/b001-native-binding.json').read_text())
    observer=json.loads((ROOT/'reuse/observer-binding-r1.json').read_text())
    for m in (native['native'],observer['N4_B1'],observer['W0_first1000']):
        if member(m['path'])!=m:raise ValueError('M_REUSE_MEMBER_DRIFT')
    basis=ready.get('P_star_basis',member(out/'P-star-basis.pt') if (out/'P-star-basis.pt').exists() else None)
    if basis is None or member(basis['path'])!=basis:raise ValueError('M_MISSING_VALID_P_STAR')
    disk=shutil.disk_usage(ROOT);reserve=72*(1<<30)
    if disk.free<reserve:raise ValueError('M_DISK_RESERVE_72GiB_UNAVAILABLE')
    if not 1<=wall_hours<=72:raise ValueError('WALLTIME_NOT_BOUNDED')
    parent=ROOT/'M'/attempt
    if parent.exists():raise ValueError('CREATE_ONCE_M_ALREADY_EXISTS')
    execution=freeze_source(repo,parent)
    technical_binding=write(parent/'technical-provisional-binding.json',ready) if provisional else member(out/'READY.json')
    lock=dict(tech,source_root=execution['source_root'],execution=execution,stage='M',attempt=attempt,
        output=str(parent/'unused-shared-output'),M_root=str(parent/'episodes'),allowed_stages=['T','M'],
        technical_evidence=technical_binding,technical_validation_status=ready['status'],
        technical_lock=member(technical_lock),P_star_basis=basis,
        T_failure_path=str(out/'failure.json'),T_job=49928,
        parallel_override=None if parallel_override is None else member(parallel_override),
        failcancel_responsibility='root active bounded observation until T terminal; exact linked M only; no daemon/no auto M retry',
        reused_native_binding=native,observer_reuse_binding=observer,source_applicability=source_match,
        prior_T_source_applicability=base_source,
        reuse_matrix=member(ROOT/'reuse/m-reuse-decisions.json'),reuse_plan=member(ROOT/'reuse/m-execution-plan.csv'),
        resources=dict(gpus=1,cpus=8,mem_MiB=60416,wall_hours=wall_hours,hour_hardcap=None,project_cap=2),
        disk=dict(available=disk.free,reserve_bytes=reserve,final_L4_raw_bytes=80*4096*14336*4,
            maximum_unique_gradient_bytes=50*4096*14336*8,
            upper_ENF_blocked_bytes=10*14336*14336*8,additional='native+small factors+teacher references+receipts, no teacher duplication',
            estimate_not_measured=True),
        cost_plan=dict(T_actual_wall_seconds=ready['wall_seconds'],T_timing=ready['timing'],
            T_oracle_work=ready['oracle_work'],native_fit_new_max=9,new_targets_max=900,
            arm_episodes=80,independent_cold_episodes=10,W0_observer_reuse=True,B1_N4_pair_observer_reuse=True,
            resources_are_not_scientific_threshold=True),
        evaluation_plan='R/P/N plus greedy32 and Dev128 after selection; Report256/S/R/L NOT_ALLOWED')
    lock.pop('resume_prior',None);lock.pop('lock_identity',None);lock['lock_identity']=digest(lock)
    result=write(parent/'execution.lock.json',lock);print(json.dumps(result));return result


def inspect_M(text,lock,lockpath,throttle):
    required=('UserId=janghj','JobState=PENDING','JobHeldUser','NumCPUs=8','Requeue=0','server4',
              'Command='+lock['source_root']+'/'+PACKAGE+'run.sbatch',str(lockpath)+' M',
              'ArrayTaskId=0-9','ArrayTaskThrottle='+str(throttle))
    for item in required:
        if item not in text:raise ValueError('M_HELD_INSPECTION:'+item)
    if 'mem=59G' not in text and 'mem=60416M' not in text:raise ValueError('M_MEMORY')
    if 'gres/gpu=1' not in text:raise ValueError('M_GPU')


def submit_M(lockpath):
    lock=json.loads(lockpath.read_text());parent=lockpath.parent
    if lock['stage']!='M' or lock['allowed_stages']!=['T','M']:raise ValueError('M_SCOPE')
    if (parent/'held-inspection.json').exists() or (parent/'submission.json').exists():raise ValueError('NO_DUPLICATE_M_SUBMISSION')
    if member(lock['technical_evidence']['path'])!=lock['technical_evidence']:raise ValueError('T_EVIDENCE_DRIFT')
    if Path(lock['T_failure_path']).exists():raise ValueError('T_FAILED_NO_M_SUBMISSION')
    Tstate=call(['sacct','-n','-X','-j',str(lock['T_job']),'--format=JobID,User,State','-P'])
    if not any(s in Tstate for s in ('|RUNNING','|COMPLETED')):raise ValueError('T_ABNORMAL_NO_M_SUBMISSION:'+Tstate)
    for m in lock['execution']['members']:
        if sha(m['path'])!=m['sha256']:raise ValueError('SOURCE_DRIFT')
    external_recheck(lock)
    queue=call(['squeue','-h','-u','janghj','-w','server4','-o','%i|%j|%T|%b|%R'])
    active=[x for x in queue.splitlines() if 'odeedit_' in x]
    # Conservative admission: an existing array is not guessed to be one slot.
    if any('[' in x.split('|')[0] or '_' in x.split('|')[0] for x in active):
        raise ValueError('EXISTING_ARRAY_NEEDS_EXACT_CAP_ACCOUNTING:'+queue)
    if len(active)>=2:raise ValueError('CAP2_NO_NEW_CAPACITY:'+queue)
    throttle=2-len(active)
    disk=shutil.disk_usage(ROOT)
    if disk.free<lock['disk']['reserve_bytes']:raise ValueError('M_STORAGE_CAPACITY_CHANGED')
    audit=write(parent/'resource-admission.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        queue=queue,node=call(['scontrol','show','node','server4']),available_disk=disk.free,
        free_inodes=os.statvfs(ROOT).f_favail,meminfo=Path('/proc/meminfo').read_text(),
        existing_project_slots=len(active),new_array_throttle=throttle,total_admitted_capacity=2,cap=2,
        stage_scope='M independent cold10 only; no S/R/L'))
    (parent/'logs').mkdir()
    args=['sbatch','--parsable','--hold','--job-name=odeedit_enfc_M_s4','--array=0-9%'+str(throttle),
        '--time='+str(lock['resources']['wall_hours'])+':00:00',f'--output={parent}/logs/%A_%a.out',f'--error={parent}/logs/%A_%a.err',
        lock['source_root']+'/'+PACKAGE+'run.sbatch',lock['source_root'],str(lockpath),'M']
    job=call(args).split(';')[0];inspection=call(['scontrol','show','job',job])
    write(parent/'held-inspection.json',dict(job=job,args=args,inspection=inspection,lock=member(lockpath),resource=audit))
    inspect_M(inspection,lock,lockpath,throttle)
    if Path(lock['T_failure_path']).exists():
        call(['scancel',job])
        write(parent/'paired-stop-during-held.json',dict(job=job,T_failure=member(lock['T_failure_path']),
            action='exact newly held M array cancelled before release',rollback_NOT_VERIFIED=True))
        raise ValueError('T_FAILED_DURING_HELD_M_CANCELLED')
    call(['scontrol','release',job])
    result=write(parent/'submission.json',dict(job=job,args=args,inspection=inspection,
        mapping={str(i):dict(episode=f'b{i+1:03d}',ordinal=[100*i,100*(i+1)],arms=8,
            native='REUSE' if i==0 else 'RUN_MISSING',N4_pair_eval='REUSE' if i==0 else 'RUN_MISSING') for i in range(10)},
        release_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='M',scope='ALL_NEEDED_M_REGISTERED_NOT_STARTED_CLAIM',
        array_throttle=throttle,S_R_L_registered=0,initial_gate_observed=False,T_job=lock['T_job'],
        validation=lock['technical_validation_status'],cancel_allowlist=[job+'_'+str(i) for i in range(10)]))
    print(json.dumps(result))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze-resume','freeze-M','submit-M'])
    p.add_argument('--worktree',type=Path);p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--attempt');p.add_argument('--wall-hours',type=int);p.add_argument('--parallel-override',type=Path);a=p.parse_args()
    if a.action=='freeze-resume':freeze_resume(a.worktree,a.lock,a.attempt)
    elif a.action=='freeze-M':freeze_M(a.worktree,a.lock,a.attempt,a.wall_hours,a.parallel_override)
    else:submit_M(a.lock)


if __name__=='__main__':main()
