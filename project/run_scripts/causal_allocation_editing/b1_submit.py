"""One qualification+B1 job and afterany CPU collector; held inspect then release."""
import argparse
import getpass
import json
import os
import re
import shlex
import shutil
import tarfile
from pathlib import Path
from . import ROOT, require, write, member, sha, verify
from .profile import B1_TASK as TASK, B1_NONCE as NONCE, check_horizon
from .b1_prepare import LOCAL
from .submit import SOURCES, command, dependency_ids

SESSION='01a0493a-074c-7f91-9a13-769116326fef'
ROLES=('main','collector')
B1_SOURCES=SOURCES+['messages/head/causal-allocation-editing-b1.json']

def launcher(source,commit,role,attempt,c):
    module='collect' if role=='collector' else 'run'
    env=dict(PYTHONPATH=str(source)+':'+c['dependency_overlay'],PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',CAUSAL_ALLOCATION_EDITING_SOURCE_COMMIT=commit)
    if role=='collector':env['CUDA_VISIBLE_DEVICES']=''
    args=[c['runtime']['python'],'-u','-m','project.run_scripts.causal_allocation_editing.'+module,'--attempt',str(attempt)]
    return '#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())+'cd '+shlex.quote(str(source))+'\nexec '+shlex.join(args)+'\n'

def arguments(role,dep,attempt,r):
    cpu=role=='collector'
    argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s2','--nodelist=server2',
        '--nodes=1','--ntasks=1','--cpus-per-task=8','--export=NONE','--no-requeue',
        '--job-name='+TASK,'--chdir='+str(attempt/'source'),
        '--mem='+str(r['collector_host_mib'] if cpu else r['host_mib'])+'M',
        '--time='+('04:00:00' if cpu else '08:00:00'),
        '--output='+str(attempt/(role+'-%j.out')),'--error='+str(attempt/(role+'-%j.err'))]
    if not cpu:argv+=['--gres=gpu:1']
    if dep:argv+=['--dependency='+dep]
    return argv+[str(attempt/(role+'.sh'))]

def inspect(job,role,dep,argv,attempt,r):
    detail=command(['scontrol','show','job',job,'--oneliner'])
    for term in [f'JobId={job} ',f'JobName={TASK} ','UserId='+getpass.getuser()+'(',
        'JobState=PENDING ','Reason=JobHeldUser ','Requeue=0 ','CPUs/Task=8 ',
        'ReqTRES=cpu=8,','ReqNodeList=server2 ','Partition=gpu ','QOS=lab_gpu_s2 ']:
        require(term in detail,'HELD:'+term)
    require(re.search(r'\bNumCPUs=8(?:-[0-9]+)? ',detail),'HELD_CPU')
    require('gres/gpu' not in detail if role=='collector' else 'TresPerNode=gres/gpu:1' in detail,'HELD_GPU')
    mem=r['collector_host_mib'] if role=='collector' else r['host_mib']
    require(f'mem={mem}M' in detail or (mem%1024==0 and f'mem={mem//1024}G' in detail),'HELD_MEMORY')
    require('TimeLimit='+('04:00:00' if role=='collector' else '08:00:00')+' ' in detail,'HELD_WALL')
    observed=re.search(r'\bDependency=([^ ]+)',detail)[1]
    require(dependency_ids(observed)==(set(dep.split(':')[1:]) if dep else set())
        and (not dep or observed.startswith('afterany:')),'HELD_DEPENDENCY')
    script=attempt/(role+'.sh')
    require('Command='+str(script)+' ' in detail and 'WorkDir='+str(attempt/'source')+' ' in detail,'HELD_PATH')
    submit=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
    require(submit and shlex.split(submit[1])==argv,'HELD_FULL_ARGV')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==script.read_text().strip(),'HELD_SCRIPT')
    return dict(job=job,role=role,argv=argv,detail=detail,launcher=member(script))

def inventory(exclude=()):
    raw=command(['squeue','-h','-r','-w','server2','-u',getpass.getuser(),'-o','%i|%j|%T|%b|%R'])
    jobs=[]
    for line in raw.splitlines():
        job,name,status,gres,reason=line.split('|',4)
        if job in exclude:continue
        detail=command(['scontrol','show','job',job,'--oneliner'])
        req=re.search(r'\bReqTRES=([^ ]+)',detail);gpu=re.search(r'gres/gpu=(\d+)',req[1]) if req else None
        if gpu:jobs.append(dict(job=job,name=name,status=status,gpus=int(gpu[1]),reason=reason,detail=detail))
    return jobs

def frozen(attempt):
    c=json.loads((attempt/'config.json').read_text());lock=json.loads((attempt/'execution.lock.json').read_text())
    check_horizon(c)
    require(lock['owner']==getpass.getuser() and lock['host']=='server2' and sha(attempt/'config.json')==lock['config_sha256'],'FROZEN_IDENTITY')
    for key in ('source_members','runtime_sources','native_reference','dependency_sources','launchers'):
        for row in lock[key]:verify(row)
    for row in [lock['archive'],lock['native_hparams']]+c['authority_members']:verify(row)
    for row in c['assets']:
        st=Path(row['path']).stat();require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_CHANGED')
    require(shutil.disk_usage(attempt).free>=c['resources']['reserve_bytes'],'DISK_RESERVE')
    return lock,c

def freeze(config,attempt):
    require(attempt==LOCAL/'attempt' and not attempt.exists(),'CREATE_ONCE_ATTEMPT')
    c=json.loads(config.read_text());check_horizon(c)
    require(c['attempt']==str(attempt),'ATTEMPT_BINDING')
    require(not command(['git','status','--porcelain','--',*B1_SOURCES],ROOT),'SOURCE_MUST_BE_COMMITTED')
    tests=json.loads(verify(c['cpu_preflight']).read_text());require(tests['passed'],'CPU_TESTS')
    commit=command(['git','rev-parse','HEAD'],ROOT);tree=command(['git','rev-parse','HEAD^{tree}'],ROOT)
    attempt.mkdir();source=attempt/'source';source.mkdir();archive=attempt/'source.tar'
    command(['git','archive','--format=tar','--output='+str(archive),commit,*B1_SOURCES],ROOT)
    with tarfile.open(archive) as tf:
        require(all((r.isfile() or r.isdir()) and not Path(r.name).is_absolute() and '..' not in Path(r.name).parts for r in tf.getmembers()),'SAFE_ARCHIVE')
        tf.extractall(source,filter='data')
    for row in tests['source']:
        verify(row);require(sha(source/Path(row['path']).relative_to(ROOT))==row['sha256'],'TESTED_SOURCE')
    write(attempt/'config.json',c)
    for role in ROLES:
        p=attempt/(role+'.sh');p.write_text(launcher(source,commit,role,attempt,c));p.chmod(0o755)
    write(attempt/'execution.lock.json',dict(instruction_id=NONCE,task_id=TASK,source_commit=commit,source_tree=tree,
        archive=member(archive),source_members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()],
        config_sha256=sha(attempt/'config.json'),runtime_sources=c['runtime']['source_members'],
        native_reference=c['native_reference'],native_hparams=member(c['native_hparams']),dependency_sources=c['dependency_sources'],
        launchers=[member(attempt/(role+'.sh')) for role in ROLES],owner=getpass.getuser(),host='server2',session=SESSION,
        resources=c['resources'],flow='single GPU qualification2 -> READY -> cold B1/onecommit/H5 -> CPU afterany collector',
        save_checkpoints=False,exact_resume='NOT_AVAILABLE'))
    return frozen(attempt)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args();attempt=LOCAL/'attempt'
    require(not list(LOCAL.glob('*/submitted-*.json')) and not list(LOCAL.glob('*/submission.json')),'NO_DUPLICATE_REGISTRATION')
    require(not command(['squeue','-h','-w','server2','-u',getpass.getuser(),'--name='+TASK,'-o','%i']),'TASK_ALREADY_QUEUED')
    before=inventory()
    local=int(next(x for x in Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if x.startswith('server2\t')).split('\t')[2])
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if x.startswith('server2\t')).split('\t')[1])
    cap=min(2,local,tracked);require(cap>=1,'CAP_UNKNOWN_OR_DISABLED')
    node=command(['scontrol','show','node','server2']);partition=command(['scontrol','show','partition','gpu'])
    lock,c=freeze(args.config.resolve(),attempt);r=c['resources']
    require(r['host_mib']==59392 and r['hard_host_mib']==60416 and r['task_cap']==1 and r['collector_host_mib']==24576 and r['wall']=='08:00:00','RESOURCE_LOCK')
    external=[j['job'] for j in before];barrier=external if sum(j['gpus'] for j in before)+1>cap else []
    ids={};held=[]
    for role in ROLES:
        dep='afterany:'+ids['main'] if role=='collector' else ('afterany:'+':'.join(barrier) if barrier else None)
        argv=arguments(role,dep,attempt,r);job=command(argv).split(';')[0];require(job.isdigit(),'JOB_ID')
        ids[role]=job;write(attempt/('submitted-'+role+'.json'),dict(nonce=NONCE,job=job,role=role,argv=argv,dependency=dep,status='HELD'))
        held.append(inspect(job,role,dep,argv,attempt,r))
    pre=inventory(tuple(ids.values()));require({j['job'] for j in pre}<=set(external),'ADMISSION_RACE_KEEP_HELD')
    require(barrier or sum(j['gpus'] for j in pre)+1<=cap,'CAP_RACE_KEEP_HELD');frozen(attempt)
    write(attempt/'held-inspection.json',dict(jobs=held,node=node,partition=partition,before=before,prerelease=pre,
        project_cap=cap,task_cap=1,external_dependencies=barrier,source_verified=True,other_job_mutations=0,
        naming_exception='USER exact task name; conservatively count all own server2 GPU jobs, no shared helper change'))
    for role in reversed(ROLES):
        result=command(['scontrol','release',ids[role]]);write(attempt/('released-'+role+'.json'),dict(job=ids[role],result=result,success=True))
    snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%b|%N|%r'])
    receipt=dict(nonce=NONCE,task_id=TASK,status='SUBMISSION_HANDOFF',jobs=ids,source=lock['source_commit'],
        lock=member(attempt/'execution.lock.json'),initial_snapshot=snapshot,resources=r,
        GPU_qualification='NOT_OBSERVED',W1='NOT_OBSERVED',monitoring_active=False,automatic_resume=False,automatic_retry=False)
    write(attempt/'submission.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
