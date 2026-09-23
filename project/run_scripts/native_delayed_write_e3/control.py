"""Register all science stages in one immutable GPU process and CPU afterany.

No chat-agent future submission is needed. No invalid downstream GPU dependency
exists: the internal DAG stops on the first failed atomic gate; collector records
the terminal boundary without executing science or overriding the failure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from .common import ROOT, INSTRUCTION, read, save, sha, file_record


def command(args):return subprocess.check_output(args,text=True).strip()


def submit(source):
    source=Path(source).resolve();attempt=ROOT/'attempt-v1'
    lock=read(attempt/'execution.lock.json')
    if (attempt/'submission-events.jsonl').exists():raise RuntimeError('EXISTING_REGISTRATION_JOURNAL_NO_DUPLICATE_SUBMIT')
    queue=command(['squeue','-h','-u','janghj','-w','server4','-o','%i|%j|%T|%b|%R'])
    if queue:raise RuntimeError('EXACT_ADMISSION_ROUTING_REQUIRED:'+queue)
    disk=shutil.disk_usage(ROOT);assert disk.free>=4*2**30,('BLOCKED_STORAGE',disk.free)
    assert command(['hostname'])=='server4' and command(['id','-un'])=='janghj'
    node=command(['scontrol','show','node','server4','--oneliner'])
    save(attempt/'admission.json',dict(epoch=time.time(),queue=queue,node=node,free_bytes=disk.free,
        free_inodes=os.statvfs(ROOT).f_favail,project_cap=2,task_gpu_cap=1,source=str(source),lock_sha256=sha(attempt/'execution.lock.json')))
    (attempt/'logs').mkdir(exist_ok=False)
    jobs={};inspections={};scripts={}
    launcher=source/'project/run_scripts/native_delayed_write_e3/launch.sh'
    with (attempt/'submission-events.jsonl').open('x') as journal:
        for phase in ('science','collector'):
            script=attempt/(phase+'.sh')
            args=[str(launcher),str(source),phase,str(attempt)]+([jobs['science']] if phase=='collector' else [])
            content='#!/usr/bin/env bash\nset -euo pipefail\nexec bash '+shlex.join(args)+'\n'
            with script.open('x') as f:f.write(content);f.flush();os.fsync(f.fileno())
            scripts[phase]=file_record(script)
            cmd=['sbatch','--parsable','--hold','--job-name=odeedit_delayed_E3_'+phase+'_s4',
                '--partition=gpu','--nodelist=server4','--nodes=1','--ntasks=1','--cpus-per-task='+('8' if phase=='science' else '4'),
                '--mem='+('60416M' if phase=='science' else '8192M'),'--time='+('1-00:00:00' if phase=='science' else '04:00:00'),
                '--export=NONE','--no-requeue','--output='+str(attempt/'logs'/(phase+'-%j.out')),
                '--error='+str(attempt/'logs'/(phase+'-%j.err'))]
            if phase=='science':cmd+=['--gres=gpu:1']
            else:cmd+=['--dependency=afterany:'+jobs['science'],'--kill-on-invalid-dep=yes']
            cmd.append(str(script))
            job=command(cmd).split(';')[0];assert job.isdigit();jobs[phase]=job
            journal.write(json.dumps(dict(phase=phase,job_id=job,argv=cmd,epoch=time.time()))+'\n');journal.flush();os.fsync(journal.fileno())
            check=command(['scontrol','show','job',job,'--oneliner']);inspections[phase]=check
            assert f'JobId={job} ' in check and 'UserId=janghj(' in check and 'JobState=PENDING' in check and 'Priority=0' in check
            assert 'ReqNodeList=server4' in check and 'Partition=gpu' in check and 'Requeue=0' in check
            assert 'NumCPUs='+('8' if phase=='science' else '4') in check
            assert ('mem=59G' in check or 'mem=60416M' in check) if phase=='science' else ('mem=8G' in check or 'mem=8192M' in check)
            if phase=='science':assert 'gres/gpu=1' in check and 'Dependency=(null)' in check
            else:assert re.search(r'Dependency=afterany:'+jobs['science']+r'(\(|\s)',check) and 'gres/gpu=' not in check
            stored=subprocess.check_output(['scontrol','write','batch_script',job,'-'],text=True)
            assert stored==content
            journal.write(json.dumps(dict(phase=phase,job_id=job,inspection='PASS',stored_script_sha256=hashlib.sha256(stored.encode()).hexdigest()))+'\n');journal.flush();os.fsync(journal.fileno())
        save(attempt/'held-inspection.json',dict(status='PASS',jobs=jobs,inspections=inspections,scripts=scripts,lock_sha256=sha(attempt/'execution.lock.json')))
        for phase,job in jobs.items():
            result=command(['scontrol','release',job]);journal.write(json.dumps(dict(job_id=job,action='release',result=result,epoch=time.time()))+'\n');journal.flush();os.fsync(journal.fileno())
    after=command(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%j|%T|%b|%E|%R'])
    receipt=dict(instruction_id=INSTRUCTION,status='REGISTERED_INSPECTED_RELEASED',jobs=jobs,after_queue=after,source=str(source),
        lock_sha256=sha(attempt/'execution.lock.json'),stage_mapping={s:jobs['collector'] if s=='G70' else jobs['science'] for s in lock['allowed_stages']},
        scheduler_dependency={'collector':'afterany:'+jobs['science']},internal_dependency='G00 -> G10 -> G20 -> G21 -> G30 -> G31 -> G40 -> G50 -> G51 -> G60; matching atomic PASS required',
        peak_task_GPU=1,peak_project_GPU_at_admission=1,actual_GPU_gates='NOT_OBSERVED',new_checkpoint=False,
        failure_collector='afterany CPU only, no science bypass/no dynamically submitted science/no invalid-dependency GPU jobs',epoch=time.time())
    print(save(attempt/'submission.json',receipt));print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args();submit(a.source)
