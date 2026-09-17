"""Main-only initial gate recall. One-shot observations, never a daemon.

Old frozen runtime/source/locks/receipts remain immutable. This orchestration
layer does not loosen any science or actual-technical check.
"""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
from .common import ROOT,TASK,ARMS,save,identity,sha,verify
from .control import command,git,PACKAGE
from .operations import admission,inspected_submit,fields

RESUME=ROOT/'resume-r1'
AUTH='messages/head/2026-09-17-sh4-slz-v2-main-initial-gate-override.md'
AUTH_SHA='6f2bf01cb126c95423dbe063daa83016d02b7d815229687833d2cb70bb358303'
TECH_JOB='49421'
TECH_ROOT=ROOT/'technical/attempt-v1'


def initialize(w):
    assert sha(w/AUTH)==AUTH_SHA
    p=RESUME/'authoritative'/AUTH;p.parent.mkdir(parents=True,exist_ok=True)
    with (w/AUTH).open('rb') as a,p.open('xb') as b:shutil.copyfileobj(a,b)
    lock=ROOT/'execution.lock.json'
    assert identity(lock)['sha256']=='a41cb76a25ab98b02c043397cd9cf4db0768e9ae312931b349008c3e9b303d18'
    return save(RESUME/'override-receipt.json',dict(instruction_id=TASK,
        nonce='ODEEDIT-GH-SH4-SLZV2-MAIN-GATE-RESUME-20260917-R1',time=datetime.now(timezone.utc).isoformat(),
        authoritative=identity(p),reading='FULL_READ',base=git(w,'rev-parse','HEAD'),
        preserved_execution=identity(lock),historical_pause=identity(ROOT/'resume-manifest.json'),
        actual_state='RESUMED_TECHNICAL_THROUGH_MAIN_INITIAL_GATE',technical_pending_is_stop=False,
        permitted_stops=['MAIN_INITIAL_VALID','MAIN_GPU_RESOURCE_PENDING_HANDOFF','UNRESOLVABLE_TYPED_BLOCKER'],
        require_all_main_registered_inspected_released=True,cap=2,no_disk_W_M=True,
        new_arm_or_science_authority=False,daemon=False,automatic_callback=False))


def observe_technical(label):
    # Exact task only; no previous repair or other scientific output is queried.
    accounting=command(['sacct','-X','-j',TECH_JOB,'--noheader','--parsable2',
        '--format=JobIDRaw,JobName,User,State,ExitCode,Start,End,Elapsed,AllocTRES,NodeList'])
    import subprocess
    p=subprocess.run(['scontrol','show','job',TECH_JOB,'-o'],text=True,capture_output=True)
    stages=[]
    for path in sorted(TECH_ROOT.glob('*/result.json')):
        value=json.loads(path.read_text())
        stages.append(dict(stage=value['stage'],status=value['status'],identity=identity(path)))
    ready=TECH_ROOT/'READY.json';failure=TECH_ROOT/'failure.json'
    evidence=dict(time=datetime.now(timezone.utc).isoformat(),job=TECH_JOB,accounting=accounting,
        scontrol=p.stdout if not p.returncode else dict(exit=p.returncode,error=p.stderr),
        stages=stages,READY=identity(ready) if ready.is_file() else None,
        failure=identity(failure) if failure.is_file() else None,
        scope='TECHNICAL_ONLY_NOT_MAIN_INITIAL_GATE',new_submission=False)
    ref=save(RESUME/'observations'/(label+'.json'),evidence)
    print(json.dumps(dict(receipt=ref,**evidence)));return evidence


def verify_ready(path,lockpath):
    from .technical import MANDATORY,NUMERICS
    ready=json.loads(path.read_text())
    assert ready['status']=='TECHNICAL_READY' and ready['execution_lock_sha256']==identity(lockpath)['sha256']
    assert ready['all_required_actual_stages_pass'] and ready['scientific_commits']==0
    assert ready['state_restored_to_cold'] and set(ready['required_actual_stages'])==set(MANDATORY)
    keyed={}
    for ref in ready['checks']:
        assert identity(ref['path'])==ref
        value=json.loads(Path(ref['path']).read_text());assert value['status']=='PASS'
        keyed[value['stage']]=value
    assert set(keyed)==set(MANDATORY)
    coverage=keyed['C45678_BOUNDED_SEARCH']['evidence']['coverage']
    assert coverage['completed_search_gate_vectors']>=7 and coverage['distinct_a4']>=2
    assert identity(ready['capsule']['path'])==ready['capsule']
    assert identity(ready['cost']['path'])==ready['cost']
    assert identity(ready['numerics']['path'])==ready['numerics']
    numeric=json.loads(Path(ready['numerics']['path']).read_text())
    assert all(numeric[k]==v for k,v in NUMERICS.items())
    return ready


def capacity_plan(existing,technical_job=TECH_JOB):
    # Own completed technical preparation must not permanently reduce the array
    # throttle. afterok serializes it even if READY precedes scheduler termination.
    other=[r for r in existing if r['job']!=technical_job]
    capacity=sum(r['capacity'] for r in other)
    if capacity>2:raise RuntimeError('EXISTING_PROJECT_CAPACITY_EXCEEDS2')
    dependencies=['afterok:'+technical_job]
    if capacity==2:
        dependencies.append('afterany:'+':'.join(r['job'] for r in other));throttle=2
    else:throttle=2-capacity
    return dict(throttle=throttle,dependencies=dependencies,other_capacity=capacity,
        simultaneous_upper_bound=2,technical_main_overlap=0)


def main_pending_classification(rows,node,*,ready,registered,released):
    """Only physical GPU exhaustion, not generic Resources/None, admits pause."""
    if not (ready and registered==6 and released):return 'CONTINUE_NOT_MAIN_READY'
    if any(r['state'] in ('RUNNING','CONFIGURING','COMPLETING') for r in rows):return 'CONTINUE_RUNNING_MAIN_INITIAL_GATE'
    if not rows or any(r['state']!='PENDING' for r in rows):return 'CONTINUE_EXACT_STATE_DIAGNOSIS'
    reasons=[r['reason'] for r in rows]
    if not all(x in ('Resources','ReqNodeNotAvail','Priority') or x.startswith('ReqNodeNotAvail,') for x in reasons):
        return 'CONTINUE_PENDING_REASON_NOT_GPU_EVIDENCE'
    if node.get('gpu_free')==0 and node.get('gpu_total',0)>0:
        return 'MAIN_GPU_RESOURCE_PENDING_HANDOFF'
    return 'CONTINUE_PENDING_GPU_EXHAUSTION_NOT_ESTABLISHED'


def verify_initial_gate(out,lockpath,ready_path):
    """Validate durable B1 selected/history5 -> B2 entry, not efficacy."""
    gatepath=out/'initial-gate.json'
    gate=json.loads(gatepath.read_text());commitref=gate['previous_commit']
    assert identity(commitref['path'])==commitref
    commit=json.loads(Path(commitref['path']).read_text())
    entry=json.loads((out/'B002/entry.json').read_text())
    historyref=commit['history'];assert identity(historyref['path'])==historyref
    history=json.loads(Path(historyref['path']).read_text())
    selectionref=commit['selection'];assert identity(selectionref['path'])==selectionref
    selection=json.loads(Path(selectionref['path']).read_text())
    lock=json.loads(lockpath.read_text())
    assert gate['status']=='INITIAL_VALID' and gate['actual_committed_B1']
    assert gate['technical_ready']==identity(ready_path)
    assert commit['source']==lock['source_head'] and commit['batch']==1
    assert commit['next_ordinal']==gate['next_ordinal']==100 and commit['next_batch']==2
    assert commit['received']==list(range(100)) and commit['received_count']==100
    assert commit['state']==entry['state']==gate['B2_entry']
    assert entry['previous_commit']==commitref and entry['ordinals']==[100,200]
    assert commit['history_appends']==gate['history_appends_B1']==5 and commit['inner_history_appends']==0
    assert [r['layer'] for r in history['rows']]==[4,5,6,7,8]
    assert all(r['history_append']==1 for r in history['rows']) and history['candidate_appends']==0
    assert selection['final_next_state_decided'] and selection['official_observer_access_before_seal']==0
    assert selection['selected']==commit['selected'] and commit['controller_counts']['commits']==1
    assert commit['checkpoint'] is None and gate['all_six_actual_PASS'] is False
    return dict(status='MAIN_INITIAL_VALID',arm=gate['arm'],gate=identity(gatepath),
        commit=commitref,B2_entry=identity(out/'B002/entry.json'),history=historyref,
        selection=selectionref,actual_B1_requests=100,history_appends=5,next_ordinal=100,
        all_six_actual_PASS=False,full1000_complete_claim=False,disk_W_M_checkpoints=False)


def observe_main(label):
    releasedpath=RESUME/'submission-science-r1/released.json'
    released=json.loads(releasedpath.read_text());job=released['job_id']
    assert released['all_six_registered'] and released['normal_held_inspection_release']
    ready_path=Path(json.loads((ROOT/'execution.lock.json').read_text())['common_ready'])
    verify_ready(ready_path,ROOT/'execution.lock.json')
    raw=command(['squeue','-r','-h','-j',job,'-o','%i|%T|%R'])
    rows=[]
    for line in raw.splitlines():
        jid,state,reason=line.split('|',2);assert jid.startswith(job+'_')
        rows.append(dict(job=jid,state=state,reason=reason))
    accounting=command(['sacct','-X','-j',job,'--noheader','--parsable2',
        '--format=JobID,JobName,User,State,ExitCode,Start,End,Elapsed,AllocTRES,NodeList'])
    node_raw=command(['scontrol','show','node','server4','-o']);nf=fields(node_raw)
    def gpu_count(value):
        match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',value)
        return int(match[1]) if match else None
    total=gpu_count(nf.get('CfgTRES',''));allocated=gpu_count(nf.get('AllocTRES',''))
    node=dict(gpu_total=total,gpu_allocated=allocated,
        gpu_free=None if total is None or allocated is None else total-allocated,
        state=nf.get('State'),allocated_memory=nf.get('AllocMem'),free_memory=nf.get('FreeMem'),
        cpu_allocation=nf.get('CPUAlloc'),cpu_total=nf.get('CPUTot'),resource_only=True)
    gates=[];failures=[]
    for arm in ARMS:
        out=ROOT/'arms'/arm/'attempt-v1/output'
        if (out/'initial-gate.json').is_file():
            gates.append(verify_initial_gate(out,ROOT/'execution.lock.json',ready_path))
        if (out/'failure.json').is_file():failures.append(identity(out/'failure.json'))
    representative=next((g for g in gates if g['arm']=='C45678'),None)
    preferred_active=any(r['job']==job+'_0' and r['state'] in ('RUNNING','CONFIGURING','COMPLETING') for r in rows)
    if representative is None and not preferred_active and gates:representative=gates[0]
    boundary='MAIN_INITIAL_VALID' if representative else main_pending_classification(rows,node,
        ready=True,registered=6,released=True)
    if failures and not gates:boundary='CONTINUE_MAIN_TECHNICAL_FAILURE_DIAGNOSIS'
    evidence=dict(time=datetime.now(timezone.utc).isoformat(),array=job,mapping=released['mapping'],
        registered=6,released=True,queue=rows,accounting=accounting,node=node,node_raw=node_raw,
        gate_evidence=gates,representative_gate=representative,
        preferred_C45678_running=preferred_active,failure_evidence=failures,boundary=boundary,
        scientific_completion_claim=False,no_job_mutation=True)
    ref=save(RESUME/'observations'/(label+'.json'),evidence)
    print(json.dumps(dict(receipt=ref,**evidence)));return evidence


def science():
    lockpath=ROOT/'execution.lock.json';lock=json.loads(lockpath.read_text());verify(lock)
    assert not (ROOT/'submission-science-v1').exists(),'DO_NOT_DUPLICATE_EXISTING_SCIENTIFIC_REGISTRATION'
    ready_path=Path(lock['common_ready']);ready=verify_ready(ready_path,lockpath)
    folder=RESUME/'submission-science-r1';folder.mkdir(exist_ok=False)
    adm=admission(folder/'admission.json');plan=capacity_plan(adm['existing'])
    cost=json.loads(Path(ready['cost']['path']).read_text())
    wall=max(12,math.ceil(ready['seconds']*10*1.5/3600)+1)
    free=shutil.disk_usage(ROOT).free;fs=os.statvfs(ROOT)
    assert free>=lock['storage']['reserve_bytes'],'SCIENCE_DISK_RESERVE_UNAVAILABLE'
    assert fs.f_favail>=100000,'SCIENCE_INODE_RESERVE_UNAVAILABLE'
    resource=save(folder/'resource.lock.json',dict(technical=identity(ready_path),technical_cost=ready['cost'],
        measured_technical_seconds=ready['seconds'],science_estimate_rule='10*common_pilot_seconds*1.5 plus round reserve; NOT_MEASURED_MAIN',
        gpus_per_job=1,cpus=8,mem_MiB=60416,wall_hours=wall,hour_hardcap=None,plan=plan,
        technical_peak_host_bytes=cost['peak_host_RSS_bytes'],disk_free_bytes=free,
        disk_reserve_bytes=lock['storage']['reserve_bytes'],free_inodes=fs.f_favail,
        no_disk_W_M_checkpoint=True))
    assert cost['peak_host_RSS_bytes']<60416*1024**2,'MEASURED_HOST_PEAK_EXCEEDS_RESOURCE'
    dependency=','.join(plan['dependencies'])
    result=inspected_submit('science',folder,['--dependency='+dependency,'--array=0-5%'+str(plan['throttle'])],
        wall_hours=wall,expected_dependency='afterok:'+TECH_JOB,throttle=plan['throttle'])
    result.update(mapping=dict(enumerate(ARMS)),all_six_registered=True,normal_held_inspection_release=True,
        technical_ready=identity(ready_path),resource_lock=resource,plan=plan,
        monitoring_policy='MAIN_INITIAL_GATE_OR_CONFIRMED_MAIN_GPU_RESOURCE_PENDING_ONLY',
        last_observation_is_not_automatically_pause=True)
    ref=save(folder/'released.json',result);print(json.dumps(dict(receipt=ref,**result)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','observe-technical','observe-main','science'])
    p.add_argument('--worktree',type=Path);p.add_argument('--label');a=p.parse_args()
    if a.action=='initialize':print(json.dumps(initialize(a.worktree)))
    elif a.action=='observe-technical':observe_technical(a.label)
    elif a.action=='observe-main':observe_main(a.label)
    else:science()
