"""V14 cold20 single lane and afterany collector; exact S3 admission/held inspection."""
import argparse,getpass,json,os,re,shlex,shutil,subprocess,tarfile
from pathlib import Path
from project.run_scripts.jlz_writer_coupled.submit import command,_copy_once,_write_script,dependencies,dependency_members,expected_dependencies

from .common import *

PYTHON='/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python'

def launcher(source,commit,module,args,cpu_only=False):
    argv=[PYTHON,'-B','-m',module,*args]
    lines=['#!/bin/bash','set -euo pipefail',
      'export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8',
      'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1',
      'export PYTHONPATH='+shlex.quote(str(OVERLAY)+':'+str(source)),
      'export ODEEDIT_SOURCE_COMMIT='+shlex.quote(commit),
      'export ODEEDIT_SLURM_CPU_ONLY='+('1' if cpu_only else '0'),
      *(['export CUDA_VISIBLE_DEVICES=""'] if cpu_only else []),
      'cd '+shlex.quote(str(source)), 'exec '+shlex.join(argv)]
    return '\n'.join(lines)+'\n'


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


SOURCES=['project/run_scripts/jlz_native_writer_aware','project/run_scripts/jlz_realized_subject',
 'project/run_scripts/jlz_realized_writer','project/run_scripts/jlz_shared_budget',
 'project/run_scripts/jlz_realization','project/run_scripts/jlz_writer_coupled',
 'project/run_scripts/jlz_pilot/prompts.py','project/run_scripts/jlz_pilot/__init__.py',
 'project/run_scripts/jlz_two_arm/baseline_pilot.py','project/run_scripts/jlz_two_arm/common.py','project/run_scripts/jlz_two_arm/__init__.py',
 'project/run_scripts/memit_history_lifelong/hparams.json',DESIGN,
 'plans/global/2026-10-04-jlz-v12-marginal-allocation',
 'plans/global/2026-10-04-jlz-portable-execution-v1/runtime-contract.json',
 ENVELOPE,'plans/global/2026-10-05-jlz-v13-realized-writer/method.tex',
 'scripts/fixed_counterfact.py','scripts/check-slurm-resource-cap.sh','scripts/check-slurm-gpu-cap.sh',
 'scripts/slurm_memory_policy.py','servers/slurm-memory-policy.tsv','control/gpu-concurrency-policy.tsv']
NAMES=('main','collector')

def freeze(configpath,attempt):
    require(attempt.parent==LOCAL and attempt.name.startswith('attempt-') and not attempt.exists(),'CREATE_ONCE_ATTEMPT')
    require(not command(['git','status','--porcelain','--',*SOURCES],cwd=ROOT),'UNCOMMITTED_SOURCE')
    c=json.loads(configpath.read_text());require(c['instruction_id']==NONCE,'NONCE')
    commit=command(['git','rev-parse','HEAD'],cwd=ROOT);tree=command(['git','rev-parse','HEAD^{tree}'],cwd=ROOT)
    attempt.mkdir();source=attempt/'source';source.mkdir();archive=attempt/'source.tar'
    command(['git','archive','--format=tar','--output='+str(archive),commit,*SOURCES],cwd=ROOT)
    with tarfile.open(archive) as t:
        members=t.getmembers();require(len(members)==len({m.name for m in members}),'DUPLICATE_SOURCE_MEMBER')
        require(all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members),'SAFE_SOURCE_ARCHIVE')
        t.extractall(source,filter='data')
    _copy_once(configpath,attempt/'config.json')
    for name in NAMES:
        _write_script(attempt/(name+'.sh'),launcher(source,commit,'project.run_scripts.jlz_native_writer_aware.'+('collect' if name=='collector' else 'run'),
                      ['--attempt',str(attempt)],cpu_only=name=='collector'))
    dependencies_root=Path('/data/janghj/ODE-edit/local/jlz-v12-shared-budget/20261004-v1/dependencies-r1')
    write(attempt/'execution.lock.json',dict(instruction_id=NONCE,task_id=TASK,source_commit=commit,source_tree=tree,
        archive=member(archive),source_members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()],
        config_sha256=sha(attempt/'config.json'),runtime_sources=c['runtime']['source_members'],native_reference=c['native_reference'],
        native_hparams=member(c['native_hparams']),dependency_sources=[member(p) for p in sorted(dependencies_root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts],
        launchers=[member(attempt/(n+'.sh')) for n in NAMES],owner=getpass.getuser(),host='ubuntu',
        session='01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3',resources=c['resources'],noCP=True,fit_count=20))

def verify_frozen(attempt):
    lock=json.loads((attempt/'execution.lock.json').read_text());c=json.loads((attempt/'config.json').read_text())
    require(lock['owner']==getpass.getuser() and lock['instruction_id']==NONCE and sha(attempt/'config.json')==lock['config_sha256'],'FROZEN_IDENTITY')
    for row in lock['source_members']+lock['runtime_sources']+lock['dependency_sources']+lock['native_reference']+lock['launchers']+[lock['archive'],lock['native_hparams']]+c['authority_members']:verify(row)
    for row in c['assets']:
        s=Path(row['path']).stat();require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_STAT')
    require(shutil.disk_usage(attempt).free>=c['resources']['reserve_bytes'],'STORAGE_RESERVE')
    require(c['resources']['host_peak_plan_GiB']['execution']<119,'SEALED_HOST_MEMORY_PLAN')
    return lock,c

def name(stage):return 'odeedit_jlz_v14b15_s3_'+stage
def argv(stage,dep,attempt,r):
    cpu=stage=='collector'
    result=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s3','--nodelist=ubuntu','--nodes=1','--ntasks=1',
        '--cpus-per-task=8','--export=NONE','--no-requeue','--job-name='+name(stage),'--chdir='+str(attempt/'source'),
        '--mem='+str(r['collector_host_mib'] if cpu else r['host_mib'])+'M','--time='+r['collector_wall' if cpu else 'wall'],
        '--output='+str(attempt/(stage+'-%j.out')),'--error='+str(attempt/(stage+'-%j.err'))]
    if not cpu:result+=['--gres=gpu:1']
    if dep:result+=['--dependency='+dep]
    return result+[str(attempt/(stage+'.sh'))]

def inspect(job,stage,dep,args,attempt,r):
    detail=command(['scontrol','show','job',job,'--oneliner']);cpu=stage=='collector'
    for term in [f'JobId={job} ',f'JobName={name(stage)} ','UserId='+getpass.getuser()+'(',
        'JobState=PENDING ','Reason=JobHeldUser ','Requeue=0 ','CPUs/Task=8 ','ReqTRES=cpu=8,',
        'ReqNodeList=ubuntu ','Partition=gpu ','QOS=lab_gpu_s3 ']:require(term in detail,'HELD:'+term)
    require(re.search(r'\bNumCPUs=8(?:-[0-9]+)? ',detail),'HELD_CPU')
    require('gres/gpu' not in detail if cpu else 'TresPerNode=gres/gpu:1' in detail,'HELD_GPU')
    mem=r['collector_host_mib' if cpu else 'host_mib'];wall=r['collector_wall' if cpu else 'wall']
    require(f'mem={mem}M' in detail or f'mem={mem//1024}G' in detail,'HELD_MEMORY')
    require('TimeLimit='+wall+' ' in detail,'HELD_WALL')
    field=re.search(r'\bDependency=([^ ]+)',detail)
    actual=set() if field and field[1]=='(null)' else dependency_members(field[1]) if field else None
    require(actual==expected_dependencies(dep),'HELD_DEPENDENCY')
    script=attempt/(stage+'.sh')
    require('Command='+str(script)+' ' in detail and 'WorkDir='+str(attempt/'source')+' ' in detail,'HELD_SOURCE')
    submit=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
    require(submit and shlex.split(submit[1])==args,'HELD_FULL_ARGV')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==script.read_text().strip(),'HELD_SCRIPT_SHA')
    return dict(job=job,stage=stage,argv=args,detail=detail,launcher=member(script))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True);a=p.parse_args()
    require(not list(LOCAL.glob('attempt-*/submitted-*.json')) and not list(LOCAL.glob('attempt-*/submission.json')),'PRIOR_REGISTRATION_NO_DUPLICATE')
    queue=command(['squeue','-h','-r','-u',getpass.getuser(),'-o','%i|%j|%T'])
    require('odeedit_jlz_v14b15_s3_' not in queue,'DUPLICATE_V14_JOB');before=resource_inventory()
    caps=Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text()
    local=int(next(s for s in caps.splitlines() if s.startswith('server3\t')).split('\t')[2])
    tracked=int(next(s for s in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if s.startswith('server3\t')).split('\t')[1])
    cap=min(1,local,tracked);require(cap>=1,'NO_CAP')
    helper=subprocess.run(['bash',str(ROOT/'scripts/check-slurm-resource-cap.sh'),'server3','1','121856M'],
        env=dict(os.environ,AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv'),capture_output=True,text=True)
    require(helper.returncode in (0,4),'RESOURCE_HELPER:'+helper.stdout+helper.stderr)
    partition=command(['scontrol','show','partition','gpu']);node=command(['scontrol','show','node','ubuntu'])
    require('MaxTime=30-00:00:00' in partition,'PARTITION_CHANGED')
    attempt=a.attempt.resolve();freeze(a.config.resolve(),attempt);lock,c=verify_frozen(attempt);r=c['resources']
    import_probe=command([PYTHON,'-c',
        'import project.run_scripts.jlz_native_writer_aware.run, project.run_scripts.jlz_native_writer_aware.collect; print("SOURCE_IMPORT_OK")'],
        cwd=attempt/'source',env=dict(os.environ,PYTHONPATH=str(OVERLAY)+':'+str(attempt/'source'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''))
    require(import_probe=='SOURCE_IMPORT_OK','SOURCE_CLOSURE_IMPORT')
    require(r['host_mib']==121856 and r['task_cap']==1 and r['wall']=='2-00:00:00','SEALED_RESOURCE')
    external=[j['job'] for j in before['jobs']]
    barrier=sorted(set(external))
    ids={};checks=[];mapping={}
    for stage in NAMES:
        parents=barrier if stage=='main' else [ids['main']];dep=dependencies(('afterany',parents));args=argv(stage,dep,attempt,r)
        job=command(args).split(';')[0];require(job.isdigit(),'JOB_ID');ids[stage]=job
        mapping[stage]=dict(job=job,dependency=dep,argv=args)
        write(attempt/('submitted-'+stage+'.json'),dict(nonce=NONCE,status='HELD',**mapping[stage]))
        checks.append(inspect(job,stage,dep,args,attempt,r))
    after=resource_inventory(tuple(ids.values()));require({j['job'] for j in after['jobs']}<=set(external),'ADMISSION_RACE_HELD')
    require(barrier or sum(j['gpus'] for j in after['jobs'])+1<=cap,'ADMISSION_CAP_HELD');verify_frozen(attempt)
    write(attempt/'held-inspection.json',dict(checks=checks,before=before,prerelease=after,cap=cap,task_cap=1,
        external_afterany=barrier,partition=partition,node=node,helper=dict(rc=helper.returncode,output=helper.stdout+helper.stderr),
        no_other_job_mutation=True,all_held_before_release=True,
        predecessor_science_PASS_required=False,source_import_probe=import_probe))
    for stage in reversed(NAMES):
        result=command(['scontrol','release',ids[stage]])
        write(attempt/('released-'+stage+'.json'),dict(job=ids[stage],success=True,result=result))
    write(attempt/'submission.json',dict(nonce=NONCE,status='RELEASED',jobs=ids,mapping=mapping,source=lock['source_commit'],
        lock=member(attempt/'execution.lock.json'),held=member(attempt/'held-inspection.json'),actual_GPU='NOT_OBSERVED',
        monitoring_active=False,automatic_resume=False))
    print(json.dumps(dict(status='RELEASED',jobs=ids)))

if __name__=='__main__':main()
