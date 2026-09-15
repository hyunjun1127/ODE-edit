"""One bounded cap-safe held/inspect/release; never waits for job completion."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from .control import boundary,identity,save,sha,verify_dispatch

def command(argv,**kwargs):
    p=subprocess.run(argv,text=True,capture_output=True,**kwargs)
    if p.returncode:raise RuntimeError((argv,p.returncode,p.stdout,p.stderr))
    return p.stdout.strip()

def count_reservations(queue,describe):
    reservations=[]
    for line in queue.splitlines():
        if not line:continue
        jid=line.split('|')[0];rec=describe(jid)
        wants=re.search(r'\bReqNodeList=([^ ]+)',rec);node=re.search(r'\bNodeList=([^ ]+)',rec)
        explicit=[m.group(1) for m in (wants,node) if m and m.group(1) not in ('(null)','None')]
        if explicit and all('server4' not in x for x in explicit):continue
        g=re.search(r'\bgres/gpu=(\d+)',rec) or re.search(r'gres/gpu:[^= ,]+=(\d+)',rec) or re.search(r'TresPerNode=gres/gpu:[^: ,]+:(\d+)',rec)
        if 'gpu' in rec and not g:raise RuntimeError('UNRESOLVED_GPU_ADMISSION '+jid)
        if g:reservations.append(dict(job_id=jid,gpus=int(g.group(1)),record=rec))
    return reservations

def submit(worktree,attempt):
    w,a=Path(worktree).resolve(),Path(attempt).resolve();boundary(w)
    repair=(a/'repair.lock.json').exists()
    lp=a/('repair.lock.json' if repair else 'execution.lock.json');lock=json.loads(lp.read_text());verify_dispatch(lock['dispatch']['path'])
    if repair:
        from .repair_control import verify_repair
        verify_repair(lock['repair_dispatch']['path'])
        assert identity(lock['scientific_lock']['path'])==lock['scientific_lock']
        science=json.loads(Path(lock['scientific_lock']['path']).read_text())
        assert science['repair_pass_path']==str(Path(lock['output'])/'technical-PASS.json')
        assert not Path(science['output']).exists()
    assert lock['new_scientific_chains']==1 and lock['baseline_reruns']==0
    assert identity(lock['source_archive']['path'])==lock['source_archive']
    source=Path(lock['source_root']);shell=source/'project/run_scripts/bg_tw_reference/ep_tw'/('repair.sbatch' if repair else 'run.sbatch')
    for m in lock['members']:
        assert Path(m['path']).stat().st_size==m['bytes']
        if m['path'].startswith(str(source)+'/'):assert sha(m['path'])==m['sha256']
    assert not Path(lock['output']).exists(),'OUTPUT_EXISTS_NO_DUPLICATE_RUN'
    ctl=a/'submission-v1';ctl.mkdir(mode=0o700,exist_ok=False)
    command(['bash','-n',str(shell)])
    memory=command([sys.executable,str(w/'scripts/slurm_memory_policy.py'),'audit',str(shell)])
    env=dict(os.environ,AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    cap=command(['bash',str(w/'scripts/check-slurm-resource-cap.sh'),'server4','1','60416M'],env=env)
    queue=command(['squeue','-h','-u','janghj','-t','RUNNING,PENDING,CONFIGURING,COMPLETING','-o','%i|%j|%T|%b|%R'])
    reserves=count_reservations(queue,lambda jid:command(['scontrol','show','job',jid,'-o']))
    admitted=sum(r['gpus'] for r in reserves)
    assert admitted+1<=2,'AGGREGATE_CAP2_RESOURCE_HOLD'
    disk=shutil.disk_usage(a);fs=os.statvfs(a)
    assert disk.free>lock['cost_plan']['total_new_disk_reserve_bytes']+20*(1<<30) and fs.f_favail>1000,'DISK_SPACE_HOLD'
    resource=save(ctl/'resource-preflight.json',dict(time=datetime.now(timezone.utc).isoformat(),
        cap=cap,memory=memory,reservations=reserves,total_admitted_with_new=admitted+1,
        resource=lock['resource'],free_bytes=disk.free,free_inodes=fs.f_favail,
        disk_reserve_is_not_exclusive=True,estimates=lock['cost_plan'],
        node=command(['scontrol','show','node','server4','-o']),queue=queue))
    args=['sbatch','--parsable','--hold','--no-requeue',f'--output={ctl}/slurm-%j.out',f'--error={ctl}/slurm-%j.err',
        str(shell),str(source),str(lp),lock['output']]
    if repair:args += [lock['scientific_lock']['path'],science['output']]
    job=command(args).split(';')[0];assert job.isdigit()
    save(ctl/'held-submission.json',dict(job_id=job,args=args,lock=identity(lp),resource=resource))
    # Public scheduler may retain its default Requeue=1 despite --no-requeue.
    # This is the newly held task-owned job only, never an existing run.
    command(['scontrol','update',f'JobId={job}','Requeue=0'])
    held=command(['scontrol','show','job',job,'-o'])
    try:
        for field in ['JobName='+('odeedit_ep_tw1_repair_s4' if repair else 'odeedit_ep_tw1_s4'),'UserId=janghj(','JobState=PENDING','ReqNodeList=server4',
                      'MinMemoryNode=59G','NumCPUs=8','TimeLimit=12:00:00','Requeue=0','Dependency=(null)']:
            assert field in held,field
        assert 'gpu:rtx_pro_6000:1' in held and str(shell) in held
        save(ctl/'held-inspection.json',dict(status='HELD_REQUEST_VERIFIED',job_id=job,record=held))
    except BaseException:
        save(ctl/'held-inspection-failed.json',dict(status='HOLD_INSPECTION_FAILED_NO_RELEASE',job_id=job,record=held));raise
    command(['scontrol','release',job]);last=command(['scontrol','show','job',job,'-o'])
    assert 'Reason=JobHeldUser' not in last
    return save(ctl/'release-receipt.json',dict(job_id=job,last_observation=datetime.now(timezone.utc).isoformat(),
        last_record=last,source_archive=lock['source_archive'],source_head=lock['source_head'],lock=identity(lp),output=lock['output'],
        status='SUBMITTED_NOT_G0_PASS',scientific_chain_count=1,teacher_reused='47592_COMPLETE_NO_NEW_TEACHER',
        saved_episode_technical_then_conditional_science=repair,
        scientific_output=science['output'] if repair else lock['output'],
        gate='PENDING_GATE_NOT_RUN' if 'JobState=PENDING' in last else 'INITIAL_GATE_NOT_YET_OBSERVED',
        no_callback=True,no_automatic_resume=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--attempt',required=True)
    x=p.parse_args();print(json.dumps(submit(x.worktree,x.attempt)))
