"""Submit/inspect/release once; no science polling, cancellation or retry."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from .config import ROOT,check_lock
from .program import require_program
from project.run_scripts.single_layer_edit_preserving_correction.common import write,member,sha


def inspection(text,command,source,lock_path):
    def field(name):
        match=re.search(r'(?:^|\s)'+name+r'=(.*?)(?=\s+[A-Za-z][A-Za-z0-9_]*=|$)',text)
        return shlex.split(match.group(1)) if match else []
    actual=field('Command');submitted=field('SubmitLine')
    expected=[str(source/'project/run_scripts/single_layer_mechanism_first/run.sbatch'),str(source),str(lock_path)]
    # Slurm on this host reports only the script in Command, and its actual
    # argument vector in SubmitLine. Both are controller-returned evidence.
    argv_bound=actual==expected or (actual==expected[:1] and submitted==command and submitted[-3:]==expected)
    return dict(owner='UserId=janghj(' in text,name='JobName=odeedit_slmf_S10_s4' in text,
        held='JobState=PENDING' in text and 'Reason=JobHeldUser' in text,
        GPU='gres/gpu=1' in text,CPU='NumCPUs=8' in text,
        memory=any(s in text for s in ('mem=60416M','mem=59G','MinMemoryNode=60416M')),
        node='ReqNodeList=server4' in text,requeue='Requeue=0' in text,
        script=str(source/'project/run_scripts/single_layer_mechanism_first/run.sbatch') in text,
        cwd=f'WorkDir={source}' in text,dependency='afterok:50974' in text,
        actual_full_argv=argv_bound,submitted_full_argv=command[-3:]==expected)


def other_capacity(queue_text):
    """Conservative simultaneous capacity, including compressed array rows.

    Only the exact dependency is excluded. An unresolved array cannot become
    one GPU merely because squeue compressed ten tasks into one text line.
    """
    admitted=[]
    for row in queue_text.splitlines():
        job,name,state,gres=row.split('|',3)
        if job=='50974':continue
        if '[' in job or '%' in job:raise RuntimeError('ADMISSION_ARRAY_CAPACITY_UNRESOLVED')
        if not re.fullmatch(r'\d+(?:_\d+)?',job):raise RuntimeError('ADMISSION_JOB_ID_UNRESOLVED')
        g=re.fullmatch(r'(?:gres/)?gpu(?::[^:,]+)?:(\d+)',gres)
        if not g:raise RuntimeError('ADMISSION_GPU_CAPACITY_UNRESOLVED')
        admitted.append(dict(job_id=job,name=name,state=state,GPU=int(g.group(1))))
    return admitted


def run(lock_path):
    lock_path=Path(lock_path);lock=json.loads(lock_path.read_text());check_lock(lock);require_program(lock)
    attempt=lock_path.parent;source=Path(lock['execution']['source'])
    if (attempt/'submission.json').exists():raise ValueError('DUPLICATE_PROGRAM_SUBMISSION')
    if lock['hook_dependency']['job_id']!='50974':raise ValueError('EXACT_PARENT_ALLOWLIST')
    for item in [lock['hook_dependency']['lock'],lock['execution']['archive']]:
        if Path(item['path']).stat().st_size!=item['bytes'] or sha(item['path'])!=item['sha256']:
            raise ValueError('SOURCE_LOCK_ARCHIVE_BINDING')
    # Resource-only one-shot admission; no task output/log/result is read.
    q=subprocess.check_output(['squeue','-h','-u','janghj','-w','server4','-o','%i|%j|%T|%b'],text=True)
    admitted=other_capacity(q)
    if sum(x['GPU'] for x in admitted)+1>2:raise RuntimeError('OTHER_ADMISSION_CAPACITY_BLOCK')
    stat=os.statvfs(ROOT);free=stat.f_bavail*stat.f_frsize
    if free<24*2**30 or stat.f_favail<10000:raise RuntimeError('INITIAL_STAGE_STORAGE_PREFLIGHT')
    write(attempt/'admission.json',dict(time_unix=time.time(),resource_only_queue=q.splitlines(),
        other_admitted=admitted,new_GPU=1,existing_dependency_nonoverlap='afterok:50974',project_cap=2,
        free_bytes=free,free_inodes=stat.f_favail,initial_required_bytes=24*2**30,
        later_stage_capacity_NOT_GUARANTEED=True,checkpoint_bytes=0,monitoring=False))
    (attempt/'logs').mkdir(exist_ok=False)
    command=['sbatch','--parsable','--hold','--job-name=odeedit_slmf_S10_s4','--dependency=afterok:50974',
        '--time=7-00:00:00',f'--output={attempt}/logs/%j.out',f'--error={attempt}/logs/%j.err',
        str(source/'project/run_scripts/single_layer_mechanism_first/run.sbatch'),str(source),str(lock_path)]
    job=subprocess.check_output(command,cwd=source,text=True).strip().split(';')[0]
    if not re.fullmatch(r'\d+',job):raise RuntimeError('SBATCH_ID_SCHEMA')
    write(attempt/'submission.json',dict(job_id=job,command=command,held=True,lock=member(lock_path),
        submission_controller=member(__file__),
        source=lock['execution']['commit'],maximum_batch=10,scientific_gates_required=True))
    observed=subprocess.check_output(['scontrol','show','job',job,'--oneliner'],text=True)
    checks=inspection(observed,command,source,lock_path)
    write(attempt/'held-inspection.json',dict(job_id=job,checks=checks,scontrol=observed,full_argv=command))
    if not all(checks.values()):raise RuntimeError('HELD_INSPECTION_FAILED_NOT_RELEASED')
    subprocess.run(['scontrol','release',job],check=True)
    state=subprocess.check_output(['scontrol','show','job',job,'--oneliner'],text=True)
    write(attempt/'release.json',dict(job_id=job,released=True,scontrol=state,scientific_gate='NOT_OBSERVED',
        full_T0='NOT_OBSERVED',new_state_checkpoint=False,monitoring_active=False,automatic_resume=False))
    print(json.dumps(dict(job_id=job,source=lock['execution']['commit'],released=True,
        dependency='afterok:50974',maximum_batch=10,output=lock['output'],monitoring_active=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True)
    run(p.parse_args().lock)
