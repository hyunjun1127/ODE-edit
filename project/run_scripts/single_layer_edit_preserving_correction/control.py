"""CPU preflight, immutable source closure, held T submission and inspection."""
import argparse
import ast
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from .common import ROOT,STEM,member,sha,write,digest
from .technical import NUMERIC

PACKAGE='project/run_scripts/single_layer_edit_preserving_correction/'
PYTHON='/data/janghj/EasyEdit/.venv/bin/python'

def call(args,cwd=None):return subprocess.check_output(args,cwd=cwd,text=True).strip()

def source_closure(repo):
    sources=[p for p in call(['git','ls-files',PACKAGE],repo).splitlines() if p.endswith(('.py','.sbatch'))]
    dependencies=['scripts/fixed_counterfact.py','project/run_scripts/single_layer_zflow/native_binding.py',
        'project/run_scripts/baseline_mechanism_first/fixtures.py','project/run_scripts/baseline_mechanism_first/evaluation.py',
        'project/run_scripts/low_cost_write_donor_pilot/fitting.py','project/run_scripts/bg_tw_reference/ep_tw/model_adapter.py',
        'project/run_scripts/alphaedit_strength_neutral_barrier/contracts.py']
    # Namespace/package initializers are bound as real source when present.
    for relative in list(sources)+dependencies:
        for parent in Path(relative).parents:
            p=parent/'__init__.py'
            if (repo/p).is_file():dependencies.append(str(p))
    return sorted(set(sources+dependencies))

def freeze(repo,attempt):
    base=json.loads((ROOT/'inputs/base-binding.json').read_text())
    source=ROOT/'T'/attempt/'source';source.mkdir(parents=True,exist_ok=False)
    relative=source_closure(repo)
    dirty=call(['git','status','--porcelain','--']+relative,repo)
    if dirty:raise ValueError('UNCOMMITTED_SOURCE_CLOSURE: '+dirty)
    members=[]
    for rel in relative:
        p=repo/rel;data=p.read_bytes()
        if p.suffix=='.py':ast.parse(data)
        out=source/rel;out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('xb') as f:f.write(data)
        out.chmod(0o400);members.append(dict(relative=rel,**member(out)))
    call(['bash','-n',str(source/PACKAGE/'run.sbatch')])
    archive=source.parent/'source.tar'
    with archive.open('xb') as f,tarfile.open(fileobj=f,mode='w') as tar:
        for rel in relative:
            info=tar.gettarinfo(str(source/rel),arcname=rel);info.uid=info.gid=info.mtime=0;info.uname=info.gname='';info.mode=0o400
            with (source/rel).open('rb') as src:tar.addfile(info,src)
    prior=json.loads(Path(base['historical_assets_prior_fullsha']['path']).read_text())
    roots=[Path(base[k]) for k in ('snapshot','projector','config4','blue_root','reference_root','dataset_root','historical_evaluator_root','helper_scripts_root')]
    roots.append(Path(base['teacher_manifest']['path']).parent)
    external=[]
    for m in prior['members']:
        p=Path(m['path'])
        if not any(p==r or p.is_relative_to(r) for r in roots):continue
        stat=p.stat()
        if stat.st_size!=m['bytes']:raise ValueError('PRIOR_ASSET_SIZE:'+str(p))
        if m.get('stat'):
            if [stat.st_dev,stat.st_ino,stat.st_mtime_ns]!=m['stat']:raise ValueError('PRIOR_ASSET_STAT:'+str(p))
        elif sha(p)!=m['sha256']:raise ValueError('ASSET_SHA:'+str(p))
        external.append(dict(m,verification='PRIOR_FULLSHA_CURRENT_STABLE_STAT',stat=[stat.st_dev,stat.st_ino,stat.st_mtime_ns]))
    if not external:raise ValueError('EMPTY_EXTERNAL_CLOSURE')
    import torch,transformers,scipy,numpy
    if torch.__version__!=base['torch'] or transformers.__version__!=base['transformers']:raise ValueError('DEPENDENCY_VERSION')
    disk=shutil.disk_usage(ROOT)
    if disk.free<48*(1<<30):raise ValueError('DISK_RESERVE_UNAVAILABLE')
    execution=dict(head=call(['git','rev-parse','HEAD'],repo),tree=call(['git','rev-parse','HEAD^{tree}'],repo),
        source_root=str(source),archive=member(archive),members=members)
    lock=dict(base,source_root=str(source),execution=execution,external_members=external,
        output=str(source.parent/'output'),numeric_contract=NUMERIC,stage='T',attempt=attempt,
        python=call([PYTHON,'--version']),scipy=scipy.__version__,numpy=numpy.__version__,
        reuse_matrix=member(ROOT/'reuse/m-reuse-decisions.json'),reuse_plan=member(ROOT/'reuse/m-execution-plan.csv'),
        final_L4_policy='ALL_M_FINAL_ENDPOINTS; M_HISTORY0_PER_PHASE_CELLS; no S/R/L',
        resources=dict(gpus=1,cpus=8,mem_MiB=60416,wall_hours=12,hour_hardcap=None,project_cap=2),
        disk=dict(available=disk.free,reserve_bytes=48*(1<<30),all_M_L4_raw_bytes=80*4096*14336*4),
        initial_pause='ACTUAL_M_INITIAL_OR_PROVEN_M_GPU_RESOURCE_SHORTAGE_ONLY',S_R_L_registered=False)
    lock['lock_identity']=digest(lock)
    ref=write(source.parent/'execution.lock.json',lock)
    print(json.dumps(ref));return ref

def submit_T(lockpath):
    lock=json.loads(Path(lockpath).read_text());parent=Path(lockpath).parent
    if (parent/'submission.json').exists():raise ValueError('EXISTING_SUBMISSION_NO_DUPLICATE')
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    queue=call(['squeue','-h','-u','janghj','-w','server4','-o','%i|%j|%T|%b|%R'])
    active=[x for x in queue.splitlines() if 'odeedit_' in x]
    # Conservative: no guessed dependency/throttle accounting.
    if len(active)>=2:raise ValueError('PROJECT_CAP_ADMISSION_REQUIRES_EXPLICIT_SERIAL_DEPENDENCY:'+queue)
    disk=shutil.disk_usage(ROOT)
    if disk.free<48*(1<<30):raise ValueError('DISK_RESERVE_UNAVAILABLE')
    audit=write(parent/'resource-admission.json',dict(time=now,queue=queue,node=call(['scontrol','show','node','server4']),
        available_disk=disk.free,inodes=os.statvfs(ROOT).f_favail,meminfo=Path('/proc/meminfo').read_text(),
        project_concurrent_capacity_before=len(active),after=len(active)+1,cap=2))
    (parent/'logs').mkdir()
    args=['sbatch','--parsable','--hold','--job-name=odeedit_enfc_T_s4',f'--output={parent}/logs/%j.out',f'--error={parent}/logs/%j.err',
        lock['source_root']+'/'+PACKAGE+'run.sbatch',lock['source_root'],str(lockpath),'T']
    job=call(args).split(';')[0]
    inspection=call(['scontrol','show','job',job])
    write(parent/'held-inspection.json',dict(job=job,arguments=args,inspection=inspection,lock=member(lockpath),resource=audit))
    for required in ('UserId=janghj','JobState=PENDING','JobHeldUser','NumCPUs=8','Requeue=0','server4'):
        if required not in inspection:raise ValueError('HELD_INSPECTION_'+required)
    if 'mem=60416M' not in inspection and 'mem=59G' not in inspection:raise ValueError('HELD_MEMORY_NOT_60416_MiB')
    if 'gres/gpu=1' not in inspection and 'gres/gpu:rtx_pro_6000=1' not in inspection:raise ValueError('GPU_REQUEST')
    call(['scontrol','release',job])
    write(parent/'submission.json',dict(job=job,args=args,inspection=inspection,release_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        stage='T',M_registered=0,technical_or_pending_NOT_stop=True))
    print(job)

def release_existing_T(lockpath):
    parent=Path(lockpath).parent;lock=json.loads(Path(lockpath).read_text())
    if (parent/'submission.json').exists():raise ValueError('ALREADY_RELEASED')
    held=json.loads((parent/'held-inspection.json').read_text());job=held['job']
    inspection=call(['scontrol','show','job',job])
    for required in ('UserId=janghj','JobState=PENDING','JobHeldUser','NumCPUs=8','Requeue=0','gres/gpu=1','mem=59G',
                     'Command='+lock['source_root']+'/'+PACKAGE+'run.sbatch',str(lockpath)+' T'):
        if required not in inspection:raise ValueError('EXACT_HELD_RECHECK:'+required)
    for m in lock['execution']['members']:
        if sha(m['path'])!=m['sha256']:raise ValueError('SOURCE_DRIFT')
    write(parent/'held-inspection-r1.json',dict(job=job,inspection=inspection,
        earlier_control_failure='string formatter expected 60416M; actual59G = 60416MiB, unchanged resources',
        original=member(parent/'held-inspection.json'),control_source=member(Path(__file__))))
    call(['scontrol','release',job])
    write(parent/'submission.json',dict(job=job,args=held['arguments'],stage='T',M_registered=0,
        release_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),technical_or_pending_NOT_stop=True,
        narrow_control_repair='accept scheduler equivalent59G formatting; no job/source/resource mutation'))
    print(job)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','submit-T','release-existing-T']);p.add_argument('--worktree');p.add_argument('--attempt',default='attempt-v1');p.add_argument('--lock');a=p.parse_args()
    if a.action=='freeze':freeze(Path(a.worktree),a.attempt)
    elif a.action=='submit-T':submit_T(Path(a.lock))
    else:release_existing_T(Path(a.lock))

if __name__=='__main__':main()
