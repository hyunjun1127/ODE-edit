"""Explicit one-job held admission; no retries, no old-job changes.

Admission and reviewed READY must be produced from real evidence. An ordinary
preparation manifest is not READY. Baseline cancellation does not call this CLI.
"""
import argparse
import os
import pwd
from pathlib import Path
import shlex
import subprocess
from official.experiments.prepare import read, file_sha, write_new, ROOT
from .run import validate


def canonical_cap():
    rows=[]
    for line in (ROOT.parent/'control/gpu-concurrency-policy.tsv').read_text().splitlines():
        fields=line.split()
        if fields and fields[0]=='server4':rows.append(int(fields[1]))
    if len(rows)!=1:raise ValueError('CANONICAL_SERVER4_CAP_IDENTITY')
    return rows[0]


def command(args, admission, script):
    cpus, memory = admission['cpus'], admission['memory_MiB']
    if not 1 <= cpus <= 8 or not 1 <= memory <= 60416:
        raise ValueError('SERVER4_CPU_MEMORY_LIMIT')
    if not 1 <= admission['effective_project_cap'] <= min(3,canonical_cap()) or not admission['combined_DAG_cap_pass']:
        raise ValueError('PROJECT_ADMISSION_NOT_VERIFIED')
    if not 1 <= admission['wall_hours'] <= 48:
        raise ValueError('WALL_LIMIT')
    deps = admission['dependency_job_ids']
    if any(not str(j).isdigit() or int(j) <= 0 for j in deps):
        raise ValueError('DEPENDENCY_ID')
    result = ['sbatch','--parsable','--hold','--partition=gpu','--qos='+admission['qos'],
        '--nodelist=server4','--nodes=1','--ntasks=1','--gres=gpu:1',
        '--cpus-per-task='+str(cpus),'--mem='+str(memory)+'M',
        '--time='+str(admission['wall_hours'])+':00:00','--export=NONE','--no-requeue',
        '--job-name=official-server4-'+read(args.config)['run_id'],
        '--chdir='+str(ROOT.parent),'--output='+str(args.output/'slurm-%j.out'),
        '--error='+str(args.output/'slurm-%j.err')]
    if deps:
        result.append('--dependency=afterany:'+':'.join(map(str,deps)))
    return result+[str(script)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','assets','ready','admission','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--qualification',action='store_true')
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--stop-after',type=int,choices=range(1,21),default=20)
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    from official.tracking.schema import identifier
    identifier(args.attempt)
    config, assets, ready, admission = map(read,(args.config,args.assets,args.ready,args.admission))
    config['_path']=str(args.config)
    validate(config, assets, ready, args.qualification)
    if args.qualification and (args.stop_after not in (2,3) or config['dataset']!='cf'):
        raise ValueError('QUALIFICATION_BUDGET')
    if not admission['node_QoS_memory_pass'] or admission['config_sha256'] != file_sha(args.config):
        raise ValueError('RESOURCE_BINDING')
    import time
    if not 0 <= time.time()-admission['observed_unix'] <= 300:
        raise ValueError('FRESH_ADMISSION_REQUIRED')
    if args.stop_after==20 and not ready.get('native_resume_and_parity_pass'):
        raise ValueError('ACTUAL_NATIVE_RESUME_PARITY_REQUIRED')
    args.output.mkdir(parents=True,exist_ok=True)
    registration=args.output/'registrations'/args.attempt
    registration.mkdir(parents=True,exist_ok=True)
    receipt=registration/'submission.json'
    if receipt.exists():
        raise ValueError('DUPLICATE_SUBMISSION_RECEIPT')
    if args.resume and not (args.output/'checkpoint/latest.json').is_file():
        raise ValueError('RESUME_CHECKPOINT_REQUIRED')
    if not args.resume and (args.output/'checkpoint/latest.json').exists():
        raise ValueError('EXISTING_CHAIN_REQUIRES_EXPLICIT_RESUME')
    argv=[assets['python'],'-m','official.runners.server4.run']
    for name in ('config','assets','ready','output'):
        argv+=['--'+name,str(getattr(args,name).resolve())]
    argv+=['--stop-after',str(args.stop_after),'--attempt',args.attempt]
    if args.resume:argv+=['--resume']
    if args.qualification:argv+=['--qualification']
    script=registration/'launch.sh'
    body='#!/bin/bash\nset -euo pipefail\n'
    body+='export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1\n'
    body+='export OMP_NUM_THREADS='+str(admission['cpus'])+' MKL_NUM_THREADS='+str(admission['cpus'])+'\n'
    body+='export PYTHONPATH='+shlex.quote(str(ROOT.parent))+'\n'
    body+='exec '+shlex.join(argv)+'\n'
    if script.exists() and script.read_text()!=body:raise ValueError('IMMUTABLE_LAUNCHER')
    if not script.exists():
        with script.open('x') as stream:stream.write(body)
    submit=command(args,admission,script)
    write_new(registration/'registration-plan.json',dict(argv=submit,ready=ready,admission=admission))
    if not args.submit:
        print('PLAN_ONLY_NOT_SUBMITTED');return
    out=subprocess.check_output(submit,text=True).strip()
    job=out.split(';')[0]
    if not job.isdigit():raise ValueError('SBATCH_ID_UNPARSEABLE_KEEP_HELD')
    # Persist actual ID before any inspection can fail.
    write_new(receipt,dict(job_id=job,state='HELD_NOT_RELEASED',argv=submit))
    held=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    fields=dict(piece.split('=',1) for piece in held.split() if '=' in piece)
    expected={'JobState':'PENDING','Requeue':'0','ReqNodeList':'server4',
              'Command':str(script),'CPUs/Task':str(admission['cpus'])}
    if any(fields.get(k)!=v for k,v in expected.items()) or not fields.get('UserId','').startswith(pwd.getpwuid(os.getuid()).pw_name+'('):
        raise ValueError('HELD_INSPECTION_MISMATCH_KEEP_HELD')
    if 'gres/gpu=1' not in fields.get('ReqTRES',''):
        raise ValueError('HELD_GPU_MISMATCH_KEEP_HELD')
    # Exact argv/resource/dependency readback must match the sealed admission.
    if '--export=NONE' not in held or '--mem='+str(admission['memory_MiB'])+'M' not in held:
        raise ValueError('HELD_MEMORY_EXPORT_MISMATCH_KEEP_HELD')
    deps=admission['dependency_job_ids']
    if deps and not all('afterany:'+str(j) in fields.get('Dependency','') for j in deps):
        raise ValueError('HELD_DEPENDENCY_MISMATCH_KEEP_HELD')
    write_new(registration/'held-inspection.json',dict(job_id=job,scontrol=held))
    subprocess.run(['scontrol','release',job],check=True)
    write_new(registration/'released.json',dict(job_id=job,status='RELEASED_NOT_COMPLETE'))
    print(job)


if __name__=='__main__':main()
