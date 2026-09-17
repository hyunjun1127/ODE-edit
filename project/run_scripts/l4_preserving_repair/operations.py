"""Explicit held/inspected admission. No scheduling callback or poll loop."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import re
import shutil
from .common import ROOT,ARMS,save,identity,verify
from .control import command

def admission(folder):
    queue=command(['squeue','-h','-u','janghj','-t','RUNNING,PENDING,CONFIGURING,COMPLETING',
        '-o','%i|%j|%T|%b|%R'])
    reservations=[];groups={}
    for row in queue.splitlines():
        jid=row.split('|')[0]; detail=command(['scontrol','show','job',jid,'-o'])
        if 'server4' not in detail or 'gres/gpu' not in detail:continue
        fields=dict(re.findall(r'(\w+)=([^ ]+)',detail))
        requested=fields.get('ReqTRES','');g=re.search(r'(?:^|,)gres/gpu=(\d+)',requested)
        if not g:raise RuntimeError('GPU_RESERVATION_UNPARSED')
        per=int(g.group(1));group=fields.get('ArrayJobId',jid)
        throttle=int(fields.get('ArrayTaskThrottle','1'))
        if 'ArrayJobId' in fields and throttle<=0:raise RuntimeError('UNBOUNDED_ARRAY_CAP_UNKNOWN')
        groups[group]=max(groups.get(group,0),per*throttle)
        reservations.append(dict(row=row,detail=detail,concurrent_capacity=per*throttle))
    capacity=sum(groups.values())
    receipt=dict(time=datetime.now(timezone.utc).isoformat(),gpu_cap=2,resource_only=True,
        existing=reservations,concurrent_reserved_capacity=capacity,available=max(0,2-capacity))
    save(folder/'admission.json',receipt);return receipt

def inspect_submit(lock_path,folder,mode,extras,wall_hours,resource_paths=None):
    lock=json.loads(Path(lock_path).read_text());verify(lock)
    source=Path(lock['source_root']);script=source/'project/run_scripts/l4_preserving_repair/run.sbatch'
    name='odeedit_l4repair_pilot_s4' if mode=='technical' else 'odeedit_l4repair_twoarm_s4'
    args=['sbatch','--parsable','--hold','--no-requeue','--job-name='+name,
        '--time='+str(wall_hours)+':00:00','--output='+str(folder/'%A_%a.out'),
        '--error='+str(folder/'%A_%a.err'),*extras,str(script),str(source),str(lock_path),mode]
    if mode=='science':
        # Science is separately submitted per arm below; no dynamic callback.
        raise ValueError('USE_EXPLICIT_ARM_SUBMISSION')
    job=command(args).split(';')[0];assert job.isdigit()
    detail=command(['scontrol','show','job',job,'-o'])
    save(folder/'held.json',dict(job_id=job,args=args,inspection=detail,lock=identity(lock_path)))
    assert 'UserId=janghj(' in detail and 'NumCPUs=8' in detail
    assert 'MinMemoryNode=59G' in detail or 'MinMemoryNode=60416M' in detail
    assert 'ReqNodeList=server4' in detail and 'Requeue=0' in detail and 'JobState=PENDING' in detail
    assert str(script) in detail and 'gpu:rtx_pro_6000:1' in detail
    command(['scontrol','release',job])
    state=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%b|%R'])
    result=dict(job_id=job,last_observation=state,time=datetime.now(timezone.utc).isoformat(),
        technical_PASS=False,scientific_jobs_registered=[],agent_callback=False)
    save(folder/'released.json',result);print(json.dumps(result))

def submit_technical(lock_path):
    folder=ROOT/'submission-technical-v1';folder.mkdir(exist_ok=False)
    resource=admission(folder)
    if resource['available']<1:
        save(folder/'resource-hold.json',dict(status='HOLD_NO_CAPACITY',new_jobs=0));return
    disk=shutil.disk_usage(ROOT)
    assert disk.free>=48*(1<<30),'DISK_RESERVE'
    save(folder/'resource.lock.json',dict(gpus=1,cpus=8,mem_MiB=60416,wall_hours=12,
        free_bytes=disk.free,checkpoint=False,estimate='FIRST100_TECHNICAL_NOT_MEASURED',hour_cap=None,
        aggregate_admitted_capacity=resource['concurrent_reserved_capacity']+1))
    inspect_submit(lock_path,folder,'technical',[],12)

def submit_science(lock_path):
    lock=json.loads(Path(lock_path).read_text());verify(lock)
    ready_ref=identity(lock['common_ready']);ready=json.loads(Path(ready_ref['path']).read_text())
    assert ready['status']=='TECHNICAL_READY' and ready['execution_lock_sha256']==identity(lock_path)['sha256']
    folder=ROOT/'submission-science-v1';folder.mkdir(exist_ok=False)
    resource=admission(folder)
    if resource['available']<2:
        save(folder/'resource-hold.json',dict(status='HOLD_TWO_INDEPENDENT_SLOTS_NOT_AVAILABLE',new_science_jobs=0));return
    # Conservative measured pilot inclusive upper envelope, not a science budget.
    hours=max(12,math.ceil(ready['seconds']*10*2/3600))
    assert hours<=168,'SCHEDULER_WALL_REQUIRES_RESOURCE_DECISION'
    assert shutil.disk_usage(ROOT).free>=48*(1<<30),'DISK_RESERVE'
    jobs=[];source=Path(lock['source_root']);script=source/'project/run_scripts/l4_preserving_repair/run.sbatch'
    for arm in ARMS:
        rpath=ROOT/'twoarms'/arm/'attempt-v1'/'resource.lock.json'
        save(rpath,dict(arm=arm,technical_ready=ready_ref,gpus=1,cpus=8,mem_MiB=60416,wall_hours=hours,
            hour_cap=None,pilot_seconds=ready['seconds'],estimate='10x inclusive pilot x2 reserve; NOT science measurement',
            disk_checkpoint=False))
        args=['sbatch','--parsable','--hold','--no-requeue','--job-name=odeedit_l4repair_'+arm.lower().replace('-','')+'_s4',
            '--time='+str(hours)+':00:00','--output='+str(folder/(arm+'-%j.out')),'--error='+str(folder/(arm+'-%j.err')),
            str(script),str(source),str(lock_path),'science',arm,str(rpath)]
        job=command(args).split(';')[0];assert job.isdigit();detail=command(['scontrol','show','job',job,'-o'])
        save(folder/(arm+'-held.json'),dict(job_id=job,args=args,inspection=detail))
        assert 'UserId=janghj(' in detail and 'ReqNodeList=server4' in detail and 'Requeue=0' in detail
        assert 'NumCPUs=8' in detail and ('MinMemoryNode=59G' in detail or 'MinMemoryNode=60416M' in detail)
        assert 'gpu:rtx_pro_6000:1' in detail and 'JobState=PENDING' in detail and str(script) in detail
        jobs.append(dict(arm=arm,job_id=job,resource_lock=identity(rpath)))
    save(folder/'registered.json',dict(jobs=jobs,technical_ready=ready_ref,scientific_PASS=False,aggregate_capacity=2))
    for job in jobs:command(['scontrol','release',job['job_id']])
    state=command(['squeue','-h','-j',','.join(j['job_id'] for j in jobs),'-o','%i|%j|%T|%b|%R'])
    save(folder/'released.json',dict(jobs=jobs,last_observation=state,time=datetime.now(timezone.utc).isoformat(),agent_callback=False))
    print(state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['technical','science']);p.add_argument('--lock',required=True)
    a=p.parse_args();(submit_technical if a.mode=='technical' else submit_science)(a.lock)
