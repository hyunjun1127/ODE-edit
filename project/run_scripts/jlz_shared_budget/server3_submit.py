"""SH3 migration-only submission adapter; unchanged pilot/main/collector runtime."""
import argparse,getpass,json,os,re,shlex,shutil,subprocess,tarfile
from pathlib import Path
from project.run_scripts.jlz_writer_coupled.submit import command,_copy_once,_write_script,dependencies,dependency_members,expected_dependencies
from .common import *

SOURCES=['project/run_scripts/jlz_shared_budget','project/run_scripts/jlz_realization',
 'project/run_scripts/jlz_writer_coupled','project/run_scripts/jlz_pilot/prompts.py','project/run_scripts/jlz_pilot/__init__.py',
 'project/run_scripts/jlz_two_arm/baseline_pilot.py','project/run_scripts/jlz_two_arm/common.py','project/run_scripts/jlz_two_arm/__init__.py',
 'project/run_scripts/memit_history_lifelong/hparams.json',DESIGN,'docs/methods/jlz-v12-shared-budget.tex',ENVELOPE,EXCEPTION,
 'scripts/fixed_counterfact.py','scripts/check-slurm-resource-cap.sh','scripts/check-slurm-gpu-cap.sh','scripts/slurm_memory_policy.py',
 'servers/slurm-memory-policy.tsv','control/gpu-concurrency-policy.tsv',
 'messages/head/2026-10-04-jlz-v12-migrate-sh4-to-sh3.json',
 'control/experiment-exceptions/jlz-v12-migration-sh3-20261004.json',
 'audits/servers/server4/jlz-v12-shared-budget-bs100x20-20261004-v1/migration-to-s3-r1/cancellation.json']

MIGRATION='ODEEDIT-GH-SH4-SH3-JLZ-V12-MIGRATION-20261004-R1'
PYTHON='/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python'

def verify_source_cessation():
    p=ROOT/'audits/servers/server4/jlz-v12-shared-budget-bs100x20-20261004-v1/migration-to-s3-r1/cancellation.json'
    require(sha(p)=='74adc314f0b0a1f497aa1acefe415dd98867b9c0395d87369466cc2a150a3d22','CESSATION_SHA')
    r=json.loads(p.read_text());require(r['nonce']==MIGRATION,'CESSATION_NONCE')
    require([a['job'] for a in r['actions']]==['58174','58173'],'CESSATION_JOBS')
    require(r['queue_after']=='' and r['verification']['targets_terminal'] and r['verification']['allocated_targets']==0,'SOURCE_STILL_ACTIVE')
    for job in ('58173','58174'):
        d=r['details_after'][job]
        require('JobState=CANCELLED ' in d and 'UserId=janghj(1025)' in d and '/local/jlz-v12-shared-budget/' in d,'CESSATION_OWNER_SOURCE')
    return member(p)

def launcher(source,commit,module,args,cpu_only=False):
    argv=[PYTHON,'-B','-m',module,*args]
    lines=['#!/bin/bash','set -euo pipefail',
      'export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8',
      'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1',
      'export PYTHONPATH='+shlex.quote(str(LOCAL/'dependencies-r1')+':'+str(source)),
      'export ODEEDIT_SOURCE_COMMIT='+shlex.quote(commit),
      'export ODEEDIT_SLURM_CPU_ONLY='+('1' if cpu_only else '0'),
      *(['export CUDA_VISIBLE_DEVICES=""'] if cpu_only else []),
      'cd '+shlex.quote(str(source)), 'exec '+shlex.join(argv)]
    return '\n'.join(lines)+'\n'

NAMES=('pilot','V12_MAIN','collector')
CHAINS=('V12_MAIN',)

def resource_inventory(exclude=()):
    raw=command(['squeue','-h','-r','-w','ubuntu','-o','%i|%u|%j|%T|%b|%R']);jobs=[]
    for line in raw.splitlines():
        job,user,name,status,gres,reason=line.split('|',5)
        if job in exclude or user!=getpass.getuser():continue
        detail=command(['scontrol','show','job',job,'--oneliner'])
        tres=re.search(r'\bReqTRES=([^ ]+)',detail);gpu=re.search(r'gres/gpu=(\d+)',tres[1]) if tres else None
        if not gpu:continue
        cmd=re.search(r'\bCommand=([^ ]+)',detail)
        owned=bool(cmd and cmd[1].startswith('/data/janghj/ODE-edit/') and name.startswith(('odeedit_','bfode_','motivation_','session01_')))
        # Count unrelated owned allocation conservatively, without mutation.
        # Identification is resource admission only, never cancellation authority.
        jobs.append(dict(job=job,name=name,user=user,state=status,gpus=int(gpu[1]),reason=reason,command=cmd[1] if cmd else 'NOT_EXPOSED',resource_detail=detail))
    return dict(raw=raw,jobs=jobs,scope='resource-only; no result/log inspection; no job mutation')

def freeze(configuration,attempt,project_cap=None):
    require(attempt.parent==LOCAL and attempt.name.startswith('attempt-') and not attempt.exists(),'CREATE_ONCE_ATTEMPT')
    require(not command(['git','status','--porcelain','--',*SOURCES],cwd=ROOT),'COMMIT_BEFORE_FREEZE')
    c=json.loads(configuration.read_text());require(c['instruction_id']==NONCE and c['task_id']==TASK,'AUTHORITY')
    commit=command(['git','rev-parse','HEAD'],cwd=ROOT);tree=command(['git','rev-parse','HEAD^{tree}'],cwd=ROOT)
    attempt.mkdir();source=attempt/'source';source.mkdir();archive=attempt/'source.tar'
    command(['git','archive','--format=tar','--output='+str(archive),commit,*SOURCES],cwd=ROOT)
    with tarfile.open(archive) as f:
        members=f.getmembers();require(len({m.name for m in members})==len(members),'ARCHIVE_DUPLICATE')
        require(all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members),'ARCHIVE_SAFE')
        f.extractall(source,filter='data')
    if project_cap is None:_copy_once(configuration,attempt/'config.json')
    else:
        require(project_cap == 1,'PROJECT_CAP_OVERRIDE')
        c['resources']['project_cap']=project_cap
        write(attempt/'config.json',c)
    for name in NAMES:
        module='collect' if name=='collector' else 'run'
        args=['--attempt',str(attempt)]+(['--pilot'] if name=='pilot' else [])
        _write_script(attempt/(name+'.sh'),launcher(source,commit,'project.run_scripts.jlz_shared_budget.'+module,args,cpu_only=name=='collector'))
    write(attempt/'execution.lock.json',dict(instruction_id=NONCE,task_id=TASK,source_commit=commit,source_tree=tree,
        archive=member(archive),source_members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()],
        config_sha256=sha(attempt/'config.json'),runtime_sources=c['runtime']['source_members'],native_reference=c['native_reference'],
        dependency_sources=[member(p) for p in sorted((LOCAL/'dependencies-r1').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'],
        native_hparams=member(c['native_hparams']),launchers=[member(attempt/(n+'.sh')) for n in NAMES],
        migration_instruction=MIGRATION,source_cessation=verify_source_cessation(),owner=getpass.getuser(),host='ubuntu',session='01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3',resources=c['resources'],noCP=True,
        flow='resource-afterany -> pilot -> V12_MAIN (afterany fail coverage, typed technical READY required before model load); afterany collector'))

def verify_frozen(attempt):
    lock=json.loads((attempt/'execution.lock.json').read_text());c=json.loads((attempt/'config.json').read_text())
    require(lock['owner']==getpass.getuser() and lock['instruction_id']==NONCE and c['instruction_id']==NONCE,'OWNER_NONCE')
    require(sha(attempt/'config.json')==lock['config_sha256'],'CONFIG_SHA')
    for r in lock['source_members']+lock['dependency_sources']+lock['runtime_sources']+lock['native_reference']+lock['launchers']+[lock['archive'],lock['native_hparams']]:verify(r)
    for r in c['authority_members']:verify(r)
    verify(c['observer_identity']);verify(c['native_input_alignment'])
    for r in c['assets']:
        s=Path(r['path']).stat();require((s.st_size,s.st_ino,s.st_mtime_ns)==(r['bytes'],r['inode'],r['mtime_ns']),'ASSET_STAT')
    require(shutil.disk_usage(attempt).free>=c['resources']['reserve_bytes'],'STORAGE_RESERVE')
    return lock,c

def jobname(name):return 'odeedit_jlz_v12_s3_'+name
def wall(name,r):return r['qualification_wall'] if name=='pilot' else r['collector_wall'] if name=='collector' else r['wall']

def arguments(name,dep,attempt,r):
    argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s3','--nodelist=ubuntu','--nodes=1','--ntasks=1',
        '--cpus-per-task=8','--export=NONE','--no-requeue','--job-name='+jobname(name),'--chdir='+str(attempt/'source'),
        '--mem='+str(r['collector_host_mib'] if name=='collector' else r['host_mib'])+'M','--time='+wall(name,r),
        '--output='+str(attempt/(name+'-%j.out')),'--error='+str(attempt/(name+'-%j.err'))]
    if name!='collector':argv+=['--gres=gpu:1']
    if dep:argv+=['--dependency='+dep]
    return argv+[str(attempt/(name+'.sh'))]

def inspect(job,name,dep,argv,attempt,r):
    detail=command(['scontrol','show','job',job,'--oneliner']);gpu=name!='collector';script=attempt/(name+'.sh')
    for term in [f'JobId={job} ',f'JobName={jobname(name)} ','UserId='+getpass.getuser()+'(',
        'JobState=PENDING ','Reason=JobHeldUser ','Requeue=0 ','CPUs/Task=8 ','ReqTRES=cpu=8,',
        'ReqNodeList=ubuntu ','Partition=gpu ','QOS=lab_gpu_s3 ']:require(term in detail,'HELD:'+term)
    require(re.search(r'\bNumCPUs=8(?:-[0-9]+)? ',detail),'HELD_CPU_RANGE')
    require(('TresPerNode=gres/gpu:1' in detail) if gpu else 'gres/gpu' not in detail,'HELD_GPU')
    mem=r['host_mib'] if gpu else r['collector_host_mib']
    require(f'mem={mem}M' in detail or f'mem={mem//1024}G' in detail,'HELD_MEMORY')
    require('TimeLimit='+wall(name,r)+' ' in detail,'HELD_WALL')
    field=re.search(r'\bDependency=([^ ]+)',detail)
    actual=set() if field and field[1]=='(null)' else dependency_members(field[1]) if field else None
    require(actual==expected_dependencies(dep),'HELD_DEPENDENCY')
    require('Command='+str(script)+' ' in detail and 'WorkDir='+str(attempt/'source')+' ' in detail,'HELD_SOURCE')
    submit=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
    require(submit and shlex.split(submit[1])==argv,'HELD_FULL_ARGV')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==script.read_text().strip(),'HELD_SCRIPT_BYTES')
    return dict(job=job,name=name,argv=argv,detail=detail,launcher=member(script))

def reconciled_previous(previous):
    previous=previous.resolve()
    require(previous.parent==LOCAL and previous.name.startswith('attempt-'),'PRIOR_ATTEMPT_SCOPE')
    receipt=json.loads((previous/'submission.json').read_text())
    require(receipt['nonce']==NONCE,'PRIOR_NONCE')
    verified=[]
    for name,job in receipt['jobs'].items():
        require(name in NAMES and str(job).isdigit(),'PRIOR_MAPPING')
        detail=command(['scontrol','show','job',str(job),'--oneliner'])
        require('UserId='+getpass.getuser()+'(' in detail and
            'Command='+str(previous/(name+'.sh'))+' ' in detail,'PRIOR_OWNER_SOURCE')
        state=re.search(r'\bJobState=([^ ]+)',detail)
        require(state and state[1] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','COMPLETED'),'PRIOR_STILL_ACTIVE')
        verified.append(dict(job=job,name=name,state=state[1],detail=detail))
    require(next(r for r in verified if r['name']=='pilot')['state']!='COMPLETED','NO_REPEAT_QUALIFIED_PILOT')
    require(not list(previous.glob('main-*/batch-*/commit.json')),'NO_REPEAT_SCIENCE_COMMITS')
    return verified

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--supersedes',type=Path);p.add_argument('--project-cap',type=int,choices=(1,),default=1);args=p.parse_args()
    verify_source_cessation()
    attempt=args.attempt.resolve();before=resource_inventory()
    queue=command(['squeue','-h','-r','-u',getpass.getuser(),'-o','%i|%j|%T'])
    require('odeedit_jlz_v12_s3_' not in queue,'DUPLICATE_TASK_JOB')
    registrations=list(LOCAL.glob('attempt-*/submission.json'))+list(LOCAL.glob('attempt-*/submitted-*.json'))
    prior=None
    if args.supersedes:
        require(all(p.parent==args.supersedes.resolve() for p in registrations),'OTHER_REGISTRATION_NO_RETRY')
        prior=reconciled_previous(args.supersedes)
    else:require(not registrations,'EXISTING_REGISTRATION_RECONCILE_NO_RETRY')
    require(args.project_cap==1,'NO_CAP_INCREASE_OR_OVERRIDE')
    freeze(args.config.resolve(),attempt);lock,c=verify_frozen(attempt)
    if prior:write(attempt/'superseded-failure.json',dict(previous=str(args.supersedes.resolve()),jobs=prior,sourceKEEP=True,new_USER_recall=True))
    caps=Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text()
    local=int(next(s for s in caps.splitlines() if s.startswith('server3\t')).split('\t')[2])
    tracked=int(next(s for s in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if s.startswith('server3\t')).split('\t')[1])
    cap=min(args.project_cap,local,tracked);require(cap>=1,'NO_GPU_AUTHORITY')
    r=c['resources'];require(r['host_mib']==59392 and r['task_cap']==1,'TASK_RESOURCE_POLICY')
    helper=subprocess.run(['bash',str(ROOT/'scripts/check-slurm-resource-cap.sh'),'server3','1','59392M'],
        env=dict(os.environ,AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv'),capture_output=True,text=True)
    require(helper.returncode in (0,4),'RESOURCE_HELPER:'+helper.stdout+helper.stderr)
    partition=command(['scontrol','show','partition','gpu']);node=command(['scontrol','show','node','ubuntu'])
    require('MaxTime=30-00:00:00' in partition,'PARTITION_LIMIT_CHANGED')
    # Every v12 GPU job is serial; old scientific success is not a dependency.
    external=[j['job'] for j in before['jobs']];ids={};inspection=[];mapping={}
    wait_external=external if sum(j['gpus'] for j in before['jobs'])+1>cap else []
    for name in NAMES:
        parents=wait_external if name=='pilot' else list(ids.values()) if name=='collector' else [list(ids.values())[-1]]
        dep=dependencies(('afterany',parents));argv=arguments(name,dep,attempt,r)
        job=command(argv).split(';')[0];require(job.isdigit(),'SUBMISSION_ID')
        ids[name]=job;mapping[name]=dict(job=job,dependency=dep,argv=argv)
        write(attempt/('submitted-'+name+'.json'),dict(nonce=NONCE,status='HELD',**mapping[name]))
        inspection.append(inspect(job,name,dep,argv,attempt,r))
    after=resource_inventory(tuple(ids.values()));require({r['job'] for r in after['jobs']}<=set(external),'ADMISSION_RACE_KEEP_HELD')
    require(bool(wait_external) or sum(j['gpus'] for j in after['jobs'])+1<=cap,'ADMISSION_CAP_RACE_KEEP_HELD')
    verify_frozen(attempt)
    write(attempt/'held-inspection.json',dict(jobs=inspection,before=before,prerelease=after,project_cap=cap,task_cap=1,
        max_new_GPU_concurrent=1,external_resource_afterany=wait_external,partition=partition,node=node,
        helper=dict(code=helper.returncode,output=helper.stdout+helper.stderr,exit4_is_dependency_pending=True),
        all_held_before_release=True,no_old_job_mutation=True))
    for name in reversed(NAMES):
        result=command(['scontrol','release',ids[name]])
        write(attempt/('released-'+name+'.json'),dict(job=ids[name],command_succeeded=True,result=result))
    write(attempt/'submission.json',dict(nonce=NONCE,status='RELEASED',jobs=ids,mapping=mapping,
        source=lock['source_commit'],lock=member(attempt/'execution.lock.json'),held=member(attempt/'held-inspection.json'),
        actual_pilot='NOT_OBSERVED',main_initial='NOT_OBSERVED',monitoring_active=False,automatic_resume=False))
    print(json.dumps(dict(status='RELEASED',jobs=ids)))

if __name__=='__main__':main()
