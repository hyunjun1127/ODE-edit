"""Bounded resource admission and held inspection. No polling or callbacks."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import re
from .common import ROOT,ARMS,save,identity,verify
from .control import PACKAGE,command


def fields(text):
    return dict(re.findall(r'(\w+)=([^\s]+)',text))


def admission(path):
    # Resource-only project queue, no other task output/log/result access.
    queue=command(['squeue','-h','-u','janghj','-t','RUNNING,PENDING,CONFIGURING,COMPLETING',
        '-o','%i|%j|%T|%b|%R'])
    seen={}
    for row in queue.splitlines():
        jid=row.split('|')[0]
        detail=command(['scontrol','show','job',jid,'-o']);f=fields(detail)
        if 'server4' not in (f.get('NodeList','')+f.get('ReqNodeList','')):continue
        match=re.search(r'(?:gres/gpu(?:\:[^=,]+)?=)(\d+)',f.get('AllocTRES','')+','+f.get('ReqTRES',''))
        if not match:continue
        per=int(match[1]);parent=f.get('ArrayJobId',f['JobId'])
        throttle=int(f.get('ArrayTaskThrottle','1'))
        capacity=per*throttle
        if parent in seen:capacity=max(seen[parent]['capacity'],capacity)
        seen[parent]=dict(job=parent,capacity=capacity,detail=detail,row=row)
    # Conservative concurrent-capacity sum; never assume another task has finished.
    result=dict(time=datetime.now(timezone.utc).isoformat(),gpu_cap=2,resource_only=True,
        existing=list(seen.values()),existing_capacity=sum(r['capacity'] for r in seen.values()),
        algorithm='CONSERVATIVE_SUM_ARRAY_THROTTLE; dependency may serialize new work',job_mutations=0)
    save(path,result);return result


def inspected_submit(mode,folder,extra,*,wall_hours,expected_dependency=None,throttle=None):
    source=ROOT/'source-v1';lock=ROOT/'execution.lock.json';script=source/(PACKAGE+'run.sbatch')
    name='odeedit_slz_v2_tech_s4' if mode=='technical' else 'odeedit_slz_v2_seq1000_s4'
    args=['sbatch','--parsable','--hold','--no-requeue','--job-name='+name,
        '--time='+str(wall_hours)+':00:00','--output='+str(folder/(mode+'-%A_%a.out')),
        '--error='+str(folder/(mode+'-%A_%a.err')),*extra,str(script),str(source),str(lock),mode]
    job=command(args).split(';')[0];assert job.isdigit()
    # Slurm site prolog may default requeue; scope is this newly held owned job only.
    command(['scontrol','update','JobId='+job,'Requeue=0'])
    detail=command(['scontrol','show','job',job,'-o']);f=fields(detail)
    held=save(folder/(mode+'-held.json'),dict(job_id=job,args=args,inspection=detail))
    assert f.get('UserId','').startswith('janghj(') and f.get('JobState')=='PENDING'
    assert f.get('NumCPUs')=='8' and f.get('MinMemoryNode') in ('59G','60416M')
    assert f.get('ReqNodeList')=='server4' and f.get('Requeue')=='0'
    assert f.get('Command')==str(script) and 'gpu:rtx_pro_6000:1' in detail
    assert f.get('TimeLimit') in (f'{wall_hours:02d}:00:00',f'{wall_hours//24}-{wall_hours%24:02d}:00:00')
    if expected_dependency:assert expected_dependency in f.get('Dependency','')
    if throttle:assert f.get('ArrayTaskThrottle')==str(throttle)
    command(['scontrol','release',job])
    observed=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%b|%R'])
    return dict(job_id=job,held=held,last_observation=observed,last_time=datetime.now(timezone.utc).isoformat(),
        actual_GPU_validation='NOT_OBSERVED',monitoring_policy='INITIAL_GATE_ONLY_OR_PENDING_HANDOFF')


def technical():
    lock=json.loads((ROOT/'execution.lock.json').read_text());verify(lock)
    folder=ROOT/'submission-technical-v1';folder.mkdir(exist_ok=False)
    adm=admission(folder/'admission.json');extra=[];dependency=None
    if adm['existing_capacity']>2:
        save(folder/'HOLD.json',dict(status='RESOURCE_HOLD',reason='EXISTING_CAPACITY_EXCEEDS2',new_submission=False));return
    if adm['existing_capacity']==2:
        dependency='afterany:'+':'.join(r['job'] for r in adm['existing']);extra=['--dependency='+dependency]
    result=inspected_submit('technical',folder,extra,wall_hours=12,expected_dependency=dependency)
    result.update(technical_only=True,science_registered=False,science_mapping=dict(enumerate(ARMS)),
        concurrent_capacity_upper_bound=adm['existing_capacity']+1 if not dependency else 2,
        previous49238_not_changed=True)
    save(folder/'released.json',result);print(json.dumps(result,ensure_ascii=False))


def science():
    lockpath=ROOT/'execution.lock.json';lock=json.loads(lockpath.read_text());verify(lock)
    ready_path=Path(lock['common_ready']);ready=json.loads(ready_path.read_text())
    assert ready['status']=='TECHNICAL_READY' and ready['execution_lock_sha256']==identity(lockpath)['sha256']
    # Requires actual coverage and model evidence, not file existence alone.
    assert ready.get('scientific_commits',0)==0
    folder=ROOT/'submission-science-v1';folder.mkdir(exist_ok=False)
    adm=admission(folder/'admission.json')
    if adm['existing_capacity']>2:
        save(folder/'HOLD.json',dict(status='RESOURCE_HOLD',reason='EXISTING_CAPACITY_EXCEEDS2'));return
    extra=[];dependency=None
    if adm['existing_capacity']==2:
        dependency='afterany:'+':'.join(r['job'] for r in adm['existing']);extra.append('--dependency='+dependency);throttle=2
    else:throttle=2-adm['existing_capacity']
    # The bounded common pilot is a conservative measured episode-cost reference,
    # not a scientific efficacy gate. Fixed12h is not inherited as GPU-hour budget.
    wall=max(12,math.ceil(float(ready['seconds'])*10*1.5/3600)+1)
    save(folder/'resource.lock.json',dict(technical_ready=identity(ready_path),wall_hours=wall,
        estimate_rule='10*actual_common_technical_wall*1.5 + rounding reserve; conservative, not measured science',
        hour_hardcap=None,mem_MiB=60416,cpus=8,gpus_per_task=1,array_throttle=throttle,cap=2))
    extra.append('--array=0-5%'+str(throttle))
    result=inspected_submit('science',folder,extra,wall_hours=wall,expected_dependency=dependency,throttle=throttle)
    result.update(mapping=dict(enumerate(ARMS)),all_six_registered=True,technical_ready=identity(ready_path),
        actual_science='NOT_OBSERVED',array_throttle=throttle,concurrent_capacity_upper_bound=2)
    save(folder/'released.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['technical','science']);a=p.parse_args();globals()[a.mode]()
