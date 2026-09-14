"""Bounded teacher-only submission. No polling loop or scientific submission."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .control import boundary, identity, save, sha, verify_dispatch


def command(argv, **kwargs):
    p=subprocess.run(argv,text=True,capture_output=True,**kwargs)
    if p.returncode:raise RuntimeError((argv,p.returncode,p.stdout,p.stderr))
    return p.stdout.strip()


def submit_teacher(worktree, attempt):
    worktree,attempt=Path(worktree).resolve(),Path(attempt).resolve()
    boundary(worktree)
    lockpath=attempt/'teacher.lock.json';lock=json.loads(lockpath.read_text())
    assert lock['phase']=='W0_TEACHER192' and lock['scientific_chains_submitted']==0
    verify_dispatch(lock['dispatch']['path'])
    assert sha(lock['source_archive']['path'])==lock['source_archive']['sha256']
    source=Path(lock['source_root'])
    for m in lock['members']:
        assert Path(m['path']).stat().st_size==m['bytes']
        if m['path'].startswith(str(source)+'/'):assert sha(m['path'])==m['sha256']
    ctl=attempt/'teacher-submission-v1';ctl.mkdir(mode=0o700,exist_ok=False)
    shell=source/'project/run_scripts/bg_tw_reference/preparation.sbatch'
    command(['bash','-n',str(shell)])
    memory=command([sys.executable,str(worktree/'scripts/slurm_memory_policy.py'),'audit',str(shell)])
    env=dict(os.environ,AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    cap=command(['bash',str(worktree/'scripts/check-slurm-resource-cap.sh'),'server4','1','60416M'],env=env)
    # Existing official helper counts running capacity only. Count every owned
    # active/admitted-pending reservation as well, including nonstandard names.
    queue=command(['squeue','-h','-u','janghj','-t','RUNNING,PENDING,CONFIGURING,COMPLETING','-o','%i|%j|%T|%b|%R'])
    reservations=[]
    for line in queue.splitlines():
        jid=line.split('|')[0]
        rec=command(['scontrol','show','job',jid,'-o'])
        wanted=re.search(r'\bReqNodeList=([^ ]+)',rec)
        nodes=re.search(r'\bNodeList=([^ ]+)',rec)
        explicit=[m.group(1) for m in (wanted,nodes) if m and m.group(1) not in ('(null)','None')]
        if explicit and all('server4' not in x for x in explicit):continue
        g=re.search(r'\bgres/gpu=(\d+)',rec)
        if not g:g=re.search(r'gres/gpu:[^= ,]+=(\d+)',rec)
        if not g:
            t=re.search(r'TresPerNode=gres/gpu:[^: ,]+:(\d+)',rec)
            g=t
        if 'gpu' in rec and not g:raise RuntimeError('UNRESOLVED_GPU_RESERVATION '+jid)
        if g:reservations.append(dict(job_id=jid,gpus=int(g.group(1)),record=rec))
    assert sum(r['gpus'] for r in reservations)+1<=2, 'AGGREGATE_CAP2_PENDING_HOLD'
    disk=shutil.disk_usage(attempt);s=os.statvfs(attempt)
    assert disk.free>60*(1<<30) and s.f_favail>1000,'PREPARATION_CAPACITY_HOLD'
    resource=save(ctl/'resource-preflight.json',dict(time=datetime.now(timezone.utc).isoformat(),
        cap=cap,memory=memory,existing_reservations=reservations,total_admitted_with_teacher=sum(r['gpus'] for r in reservations)+1,
        free_bytes=disk.free,free_inodes=s.f_favail,estimated_task_reserve_bytes=60*(1<<30),
        reserve_is_not_filesystem_exclusive=True,hour_cap=None,teacher_only=True,
        node=command(['scontrol','show','node','server4','-o']),raw_queue=queue))
    args=['sbatch','--parsable','--hold',f'--output={ctl}/slurm-%j.out',f'--error={ctl}/slurm-%j.err',
        str(shell),str(source),str(lockpath),lock['output']]
    job=command(args).split(';')[0];assert job.isdigit()
    save(ctl/'held-submission.json',dict(job_id=job,args=args,source_archive=lock['source_archive'],lock=identity(lockpath),resource=resource))
    held=command(['scontrol','show','job',job,'-o'])
    try:
        for field in ['JobName=odeedit_bg1_teacher_s4','UserId=janghj(','JobState=PENDING',
                      'ReqNodeList=server4','MinMemoryNode=59G','NumCPUs=8','TimeLimit=02:00:00']:
            assert field in held,field
        assert 'gpu:rtx_pro_6000:1' in held and 'Dependency=(null)' in held
        assert str(shell) in held
        save(ctl/'held-inspection.json',dict(job_id=job,status='HELD_REQUEST_VERIFIED',record=held))
    except BaseException:
        save(ctl/'held-inspection-failed.json',dict(job_id=job,record=held,status='HOLD_INSPECTION_FAILED_NO_RELEASE'))
        raise
    command(['scontrol','release',job])
    last=command(['scontrol','show','job',job,'-o'])
    assert 'JobState=PENDING' in last or 'JobState=RUNNING' in last or 'JobState=CONFIGURING' in last
    assert 'Reason=JobHeldUser' not in last
    return save(ctl/'release-receipt.json',dict(job_id=job,last_observation=datetime.now(timezone.utc).isoformat(),
        record=last,source=lock['source_archive'],lock=identity(lockpath),output=lock['output'],
        status='PREPARATION_SUBMITTED_NOT_G0',scientific_job_ids=[],teacher_validity='NOT_YET_OBSERVED',
        no_completion_callback=True,agent_waits_for_first_teacher=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--attempt',required=True)
    a=p.parse_args();print(json.dumps(submit_teacher(a.worktree,a.attempt)))
