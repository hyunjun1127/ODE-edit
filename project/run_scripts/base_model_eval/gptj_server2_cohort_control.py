"""CPU input binding and exact held submission; no scheduler polling loop."""
import argparse
import getpass
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from .gptj_server2_cohort import NONCE, TASK, REVISION, specifications, read
from project.run_scripts.jlz_realization.common import require, sha, member, digest, write

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path('/mnt/raid5/janghj/ODE-edit/local/base-model-gptj-w0-cohort-curves')
MODEL = Path('/mnt/raid5/janghj/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots')/REVISION
DATA = Path('/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1')
ENVELOPE = 'messages/head/2026-10-07-w0-cohort-curves-server2.json'
SOURCES = ['project/run_scripts/'+d for d in ('base_model_eval','jlz_price_gptj','jlz_interference_l1',
    'jlz_realization','jlz_shared_budget','jlz_pilot','experiment_tracking')]
SOURCES += ['scripts/fixed_counterfact.py', ENVELOPE, 'plans/global/2026-10-07-w0-cohort-curves-rerun/contract.json', 'control/wandb-policy.json',
            'control/wandb-method-metric-schema.json', 'control/gpu-concurrency-policy.tsv', 'servers/slurm-memory-policy.tsv']


def command(argv):
    p = subprocess.run(argv, text=True, capture_output=True)
    require(p.returncode==0, 'COMMAND_FAILED:'+shlex.join(argv)+':'+p.stderr[:1000])
    return p.stdout.strip()


def prepare():
    """Reuse exact prior input/hash receipts + current stat, not its observations."""
    out=LOCAL/'preparation-r1'
    require(not out.exists(),'CREATE_ONCE_PREPARATION')
    require(sha(ROOT/ENVELOPE)=='341763e8ec6b9d21e355458b2393f0634e87f1b7aca0111d017b2776ae51e8e7','AUTHORITY')
    require(sha(ROOT/'plans/global/2026-10-07-w0-cohort-curves-rerun/contract.json')=='b2f57f86b399ce47ad1c8e5cf777fc99e86aa3bdc812b2b16730c4f19015445c','CONTRACT')
    previous=Path('/mnt/raid5/janghj/ODE-edit/local/base-model-gptj-w0/attempt-r3')
    c=read(previous/'config.json');prior=read(previous/'submission.json')
    require(prior['jobs']==dict(main='60147',collector='60148'),'PRIOR_JOBS')
    require(sha(previous/'config.json')==prior['config_sha256'],'PRIOR_CONFIG')
    for item in c['assets']+c['runtime']['members']+[c['observer_manifest'],c['input_source']]:
        s=Path(item['path']).stat()
        require((s.st_size,s.st_ino,s.st_mtime_ns)==(item['bytes'],item['inode'],item['mtime_ns']),'REUSED_INPUT_CHANGED')
    require(sha(c['observer_manifest']['path'])==c['observer_manifest']['sha256'],'TOKEN_MANIFEST')
    c.update(instruction_id=NONCE,task_id=TASK,reference_only=True,actual_model_edits=0,
        edits_axis_semantics='reference_cohort_progress',prior_input_receipt=member(previous/'config.json'),
        prior_observations_reused=False,prior_jobs=prior['jobs'],
        prior_terminal_observation='60147 COMPLETED 0:0 00:21:07;60148 COMPLETED 0:0 00:00:01; bounded sacct owner/source check')
    out.mkdir();write(out/'config.json',c)
    print(json.dumps(dict(config=str(out/'config.json'),input_reuse='PRIOR_FULL_SHA_PLUS_CURRENT_STAT',new_GPU=0)))


def inventory():
    raw=command(['squeue','-h','-r','-w','server2','-u',getpass.getuser(),'-o','%i|%j|%T|%b|%R'])
    rows=[]
    for line in raw.splitlines():
        job,name,status,gres,reason=line.split('|',4)
        d=command(['scontrol','show','job',job,'--oneliner'])
        req=re.search(r'\bReqTRES=([^ ]+)',d);gpu=re.search(r'gres/gpu=(\d+)',req[1]) if req else None
        if gpu:rows.append(dict(job=job,name=name,state=status,gpus=int(gpu[1]),reason=reason,detail=d))
        require(name!=TASK, 'DUPLICATE_TASK_JOB')
    return rows


def launcher(a,role,c):
    env=dict(PYTHONPATH=str(a/'source'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
        OPENBLAS_NUM_THREADS='6',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONHASHSEED='20261002')
    if role=='collector':env['CUDA_VISIBLE_DEVICES']=''
    return '#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())+\
        'cd '+shlex.quote(str(a/'source'))+'\nexec '+shlex.join([c['runtime']['python'],'-u','-m',
        'project.run_scripts.base_model_eval.gptj_server2_cohort','collect' if role=='collector' else 'run','--attempt',str(a)])+'\n'


def submit(config_path, attempt_name, cpu_receipt):
    require(re.fullmatch(r'attempt-[A-Za-z0-9-]+',attempt_name),'ATTEMPT_NAME')
    a=LOCAL/attempt_name;c=read(config_path)
    require(c['resources']['cpu']==6 and 0<c['resources']['host_mib']<=59392,'CURRENT_CPU_RAM_POLICY')
    require(not a.exists(),'CREATE_ONCE_ATTEMPT')
    require(not command(['git','status','--porcelain','--',*SOURCES]),'UNCOMMITTED_SOURCE')
    tests=read(cpu_receipt);require(tests['passed'],'CPU_CHECK')
    for p,h in tests['source_sha256'].items():require(sha(ROOT/p)==h,'CPU_TEST_SOURCE_CHANGED')
    require(not command(['squeue','-h','-u',getpass.getuser(),'--name='+TASK,'-o','%i']), 'DUPLICATE_TASK')
    before=inventory()
    tracked=int(next(l for l in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if l.startswith('server2\t')).split('\t')[1])
    local=int(next(l for l in Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if l.startswith('server2\t')).split('\t')[2])
    cap=min(2,tracked,local);require(cap>=1,'CAP_DISABLED')
    barrier=[r['job'] for r in before] if sum(r['gpus'] for r in before)+1>cap else []
    node=command(['scontrol','show','node','server2']);partition=command(['scontrol','show','partition','gpu'])
    qos=command(['sacctmgr','-n','-P','show','qos','lab_gpu_s2','format=Name,MaxWall,MaxTRESPerUser,MaxTRESPerJob'])
    hardware=command(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'])
    for item in c['assets']+c['runtime']['members']:
        s=Path(item['path']).stat();require((s.st_size,s.st_ino,s.st_mtime_ns)==(item['bytes'],item['inode'],item['mtime_ns']),'INPUT_CHANGED')
    require(shutil.disk_usage(LOCAL).free>8*1024**3,'DISK_RESERVE')
    commit=command(['git','rev-parse','HEAD']);a.mkdir();(a/'source').mkdir()
    archive=a/'source.tar';command(['git','archive','--format=tar','--output='+str(archive),commit,*SOURCES])
    with tarfile.open(archive) as tar:
        require(all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in tar.getmembers()),'SAFE_SOURCE')
        tar.extractall(a/'source',filter='data')
    write(a/'config.json',c)
    for role in ('main','collector'):
        with (a/(role+'.sh')).open('x') as f:f.write(launcher(a,role,c))
    lock=dict(instruction_id=NONCE,source_commit=commit,source_tree=command(['git','rev-parse','HEAD^{tree}']),
        config_sha256=sha(a/'config.json'),archive=member(archive),source_members=[dict(relative=str(p.relative_to(a/'source')),sha256=sha(p)) for p in sorted((a/'source').rglob('*')) if p.is_file()],
        resources=c['resources'],launchers=[member(a/(r+'.sh')) for r in ('main','collector')],noCP=True,owner=getpass.getuser(),session='01a0493a-074c-7f91-9a13-769116326fef')
    write(a/'execution.lock.json',lock);ids={};inspections=[]
    for role in ('main','collector'):
        dep='afterany:'+ids['main'] if role=='collector' else ('afterany:'+':'.join(barrier) if barrier else None)
        mem=c['resources']['collector_host_mib'] if role=='collector' else c['resources']['host_mib']
        argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s2','--nodelist=server2','--nodes=1','--ntasks=1',
            '--cpus-per-task=6','--mem='+str(mem)+'M','--time=04:00:00','--export=NONE','--no-requeue',
            '--job-name='+TASK,'--chdir='+str(a/'source'),'--output='+str(a/(role+'-%j.out')),'--error='+str(a/(role+'-%j.err'))]
        if role=='main':argv+=['--gres=gpu:1']
        if dep:argv+=['--dependency='+dep]
        argv+=[str(a/(role+'.sh'))]
        job=command(argv).split(';')[0];require(job.isdigit(),'JOB_ID');ids[role]=job
        write(a/('submitted-'+role+'.json'),dict(job=job,argv=argv,dependency=dep))
        d=command(['scontrol','show','job',job,'--oneliner'])
        for value in [f'JobId={job} ',f'JobName={TASK} ','UserId='+getpass.getuser()+'(','JobState=PENDING ','Reason=JobHeldUser ',
            'Requeue=0 ','CPUs/Task=6 ','ReqNodeList=server2 ','Partition=gpu ','QOS=lab_gpu_s2 ','TimeLimit=04:00:00 ',
            'Command='+str(a/(role+'.sh'))+' ','WorkDir='+str(a/'source')+' ']:require(value in d,'HELD:'+value)
        require(re.search(r'\bNumCPUs=6(?:-\d+)? ',d) and 'ReqTRES=cpu=6,' in d,'HELD_CPU')
        require(f'mem={mem}M' in d or f'mem={mem//1024}G' in d,'HELD_MEM')
        require(('gres/gpu' not in d) if role=='collector' else 'TresPerNode=gres/gpu:1' in d,'HELD_GPU')
        observed=re.search(r'\bDependency=([^ ]+)',d)[1]
        require(set(re.findall(r'(?:afterany:|:)(\d+)',observed))==(set(dep.split(':')[1:]) if dep else set()),'HELD_DEPENDENCY')
        require(not dep or observed.startswith('afterany:'),'HELD_DEPENDENCY_TYPE')
        line=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',d)
        require(line and shlex.split(line[1])==argv,'HELD_ARGV')
        require(command(['scontrol','write','batch_script',job,'-']).strip()==(a/(role+'.sh')).read_text().strip(),'HELD_SCRIPT')
        inspections.append(dict(job=job,role=role,detail=d,argv=argv))
    # Recheck resource admission only; exclude these exact newly held IDs.
    queue=command(['squeue','-h','-r','-w','server2','-u',getpass.getuser(),'-o','%i|%b'])
    other={l.split('|')[0] for l in queue.splitlines() if 'gpu' in l and l.split('|')[0] not in ids.values()}
    require(other<={r['job'] for r in before},'ADMISSION_RACE_KEEP_HELD')
    require(barrier or sum(r['gpus'] for r in before if r['job'] in other)+1<=cap,'CAP_KEEP_HELD')
    write(a/'held-inspection.json',dict(jobs=inspections,before=before,cap=cap,external_dependencies=barrier,node=node,partition=partition,qos=qos,hardware=hardware))
    for role in ('collector','main'):
        write(a/('released-'+role+'.json'),dict(job=ids[role],result=command(['scontrol','release',ids[role]]),released=True))
    snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%b|%N|%r'])
    receipt=dict(nonce=NONCE,status='SUBMISSION_HANDOFF',jobs=ids,source=commit,config_sha256=lock['config_sha256'],lock=member(a/'execution.lock.json'),
        dependencies=barrier,initial_snapshot=snapshot,resources=c['resources'],WandB='STARTUP_IN_RUNNER_NOT_OBSERVED',
        actual_evaluation='NOT_OBSERVED',monitoring_active=False,automatic_retry=False)
    write(a/'submission.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','submit'))
    p.add_argument('--config',type=Path,default=LOCAL/'preparation-r1/config.json')
    p.add_argument('--attempt-name',default='attempt-r1');p.add_argument('--cpu-receipt',type=Path,default=LOCAL/'cpu-tests.json')
    args=p.parse_args()
    LOCAL.mkdir(parents=True,exist_ok=True)
    if args.mode=='prepare':prepare()
    else:submit(args.config,args.attempt_name,args.cpu_receipt)
