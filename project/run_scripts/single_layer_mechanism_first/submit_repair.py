"""One repaired immutable same-process program; no mutation of old jobs."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from .config import ROOT,check_lock
from .program import require_program
from .technical_repair import LENIENT_GATE_ID
from .submit_program import inspection,other_capacity
from project.run_scripts.single_layer_edit_preserving_correction.common import member,sha,write


def validate_repair_lock(lock):
    check_lock(lock);require_program(lock)
    if lock.get('hook_dependency') is not None or lock['hook_repair']['kind']!='T0_NATIVE_Z_HOOK_PARITY_REPAIR':
        raise ValueError('REPAIR_INLINE_NO_FAILED_DEPENDENCY')
    if lock['old_dependent_job']!='50983' or lock.get('old_dependent_job_mutation') is not False:
        raise ValueError('OLD_PENDING_PRESERVATION')
    if (lock['prior_failed_allocation_GPU_seconds']!=136 or lock['method_threshold_changes'] is not False
            or lock.get('technical_gate_role_changes') is not True
            or lock.get('hook_gate_policy',{}).get('id')!=LENIENT_GATE_ID):
        raise ValueError('REPAIR_COST_OR_THRESHOLD_LINEAGE')


def run(lock_path):
    lock_path=Path(lock_path);lock=json.loads(lock_path.read_text());validate_repair_lock(lock)
    attempt=lock_path.parent;source=Path(lock['execution']['source'])
    if (attempt/'submission.json').exists():raise ValueError('DUPLICATE_REPAIR_SUBMISSION')
    for item in [lock['execution']['archive']]+[v for v in lock['hook_repair'].values() if isinstance(v,dict) and 'path' in v]:
        if Path(item['path']).stat().st_size!=item['bytes'] or sha(item['path'])!=item['sha256']:
            raise ValueError('IMMUTABLE_REPAIR_INPUT_CHANGED')
    q=subprocess.check_output(['squeue','-h','-u','janghj','-w','server4','-o','%i|%j|%T|%b'],text=True)
    # Count all listed jobs conservatively, including the preserved blocked job.
    admitted=other_capacity(q,excluded_jobs=())
    if sum(x['GPU'] for x in admitted)+1>2:raise RuntimeError('REPAIR_PROJECT_CAP2')
    stat=os.statvfs(ROOT);free=stat.f_bavail*stat.f_frsize
    if free<24*2**30 or stat.f_favail<10000:raise RuntimeError('REPAIR_STORAGE_NO_WAIVER')
    write(attempt/'admission.json',dict(time_unix=time.time(),resource_only_queue=q.splitlines(),admitted=admitted,
        new_GPU=1,project_cap=2,free_bytes=free,free_inodes=stat.f_favail,new_checkpoint_bytes=0,
        old_job_cancelled=False,old_job_dependency_changed=False,monitoring=False))
    (attempt/'logs').mkdir(exist_ok=False)
    command=['sbatch','--parsable','--hold','--job-name=odeedit_slmf_repair_s4','--time=7-00:00:00',
        f'--output={attempt}/logs/%j.out',f'--error={attempt}/logs/%j.err',
        str(source/'project/run_scripts/single_layer_mechanism_first/run.sbatch'),str(source),str(lock_path)]
    job=subprocess.check_output(command,cwd=source,text=True).strip().split(';')[0]
    if not re.fullmatch(r'\d+',job):raise RuntimeError('REPAIR_SBATCH_ID')
    write(attempt/'submission.json',dict(job_id=job,command=command,held=True,source=lock['execution']['commit'],
        lock=member(lock_path),repair_of='50974',replaces_unstarted_program='50983',old_jobs_modified=False))
    text=subprocess.check_output(['scontrol','show','job',job,'--oneliner'],text=True)
    checks=inspection(text,command,source,lock_path,job_name='odeedit_slmf_repair_s4',dependency=None)
    write(attempt/'held-inspection.json',dict(job_id=job,checks=checks,scontrol=text))
    if not all(checks.values()):raise RuntimeError('REPAIR_HELD_INSPECTION_FAILED_NOT_RELEASED')
    subprocess.run(['scontrol','release',job],check=True)
    text=subprocess.check_output(['scontrol','show','job',job,'--oneliner'],text=True)
    write(attempt/'release.json',dict(job_id=job,released=True,scontrol=text,
        actual_repaired_T0='NOT_OBSERVED',save_checkpoints=False,monitoring_active=False,automatic_resume=False))
    print(json.dumps(dict(job_id=job,released=True,source=lock['execution']['commit'],output=lock['output'],
        T0_repaired_actual='NOT_OBSERVED',old_jobs_modified=False,monitoring_active=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True)
    run(p.parse_args().lock)
