"""One-shot held/inspect/release for six immutable jobs, never a polling loop."""
import argparse
import fnmatch
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import time

from .runtime import file_sha, save


def command(args, cwd):
    result=subprocess.run(args,cwd=cwd,text=True,capture_output=True,timeout=60)
    if result.returncode:
        raise RuntimeError(f'{args[0]} exit={result.returncode}: {result.stderr} {result.stdout}')
    return result.stdout


def project_rows(text,patterns):
    rows=[]
    for line in text.splitlines():
        values=line.split('|')
        if len(values)!=7:raise ValueError('SQUEUE_SCHEMA')
        job,owner,name,state,nodes,tres,dependency=values
        if any(fnmatch.fnmatchcase(name,p) for p in patterns):
            rows.append(dict(job=job,owner=owner,name=name,state=state,nodes=nodes,tres=tres,dependency=dependency))
    return rows


def inspect(text,job,attempt):
    owner=pwd.getpwuid(os.getuid()).pw_name
    required=[f'JobId={job}',f'UserId={owner}(', 'JobName=odeedit_lowcost_seq10_s4',
              'ArrayTaskId=0-5', 'NumCPUs=8',
              'TimeLimit=1-00:00:00', 'ReqNodeList=server4', str(attempt/'source/project/run_scripts/low_cost_write_donor_pilot/sequential.sbatch')]
    for field in required:
        if field not in text:raise ValueError(('HELD_REQUEST_MISMATCH',field,text))
    if not ('ArrayTaskThrottle=2' in text or 'ArrayTaskId=0-5%2' in text):
        raise ValueError('HELD_ARRAY_THROTTLE')
    if not ('MinMemoryNode=59G' in text or 'MinMemoryNode=60416M' in text):
        raise ValueError('HELD_MEMORY_REQUEST')
    if 'gres/gpu:rtx_pro_6000=1' not in text and 'gres:gpu:rtx_pro_6000:1' not in text:
        raise ValueError('HELD_GPU_REQUEST')
    if 'JobState=PENDING' not in text or 'Reason=JobHeldUser' not in text:
        raise ValueError('NOT_EXPECTED_HELD_PENDING')


def submit(attempt):
    a=Path(attempt).resolve();w=Path(__file__).resolve().parents[3]
    lock=json.loads((a/'execution.lock.json').read_text())
    if (a/'submission-accepted.json').exists():raise FileExistsError('ALREADY_SUBMITTED_NO_DUPLICATE')
    command(['scripts/check-session-boundary.sh','01a04939-b5c7-7a03-ba2d-ef3343d62cfd'],w)
    cap=Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    row=[l.split('\t') for l in cap.read_text().splitlines() if l.startswith('server4\t')]
    assert len(row)==1 and row[0][1:4]==['server4','2','60416']
    patterns=row[0][4].split(',')
    sq=command(['squeue','-h','-w','server4','-o','%i|%u|%j|%T|%D|%b|%E'],w)
    existing=project_rows(sq,patterns)
    # New array may start only after all earlier project allocations/admissions
    # have terminated. This is applied only if such earlier work actually exists.
    deps=sorted(set(r['job'].split('_')[0] for r in existing))
    if any(not re.fullmatch(r'[0-9]+',d) for d in deps):raise ValueError('DEPENDENCY_ID_UNRESOLVED')
    cap_helper='DEFERRED_AFTER_EXISTING_ADMISSIONS' if deps else command(
        ['env',f'AGENT_GPU_CAPS_FILE={cap}','scripts/check-slurm-resource-cap.sh','server4','2','120832M'],w)
    partition=command(['scontrol','show','partition','gpu','-o'],w)
    host=command(['hostname'],w).strip()
    script=a/'source/project/run_scripts/low_cost_write_donor_pilot/sequential.sbatch'
    memory=command(['python3','scripts/slurm_memory_policy.py','audit',str(script)],w)
    for m in lock['members']:
        if str(script)==m['path']:assert file_sha(script)==m['sha256']
    output=a/'slurm';output.mkdir(mode=0o700,exist_ok=False)
    (a/'output').mkdir(mode=0o700,exist_ok=False)
    evidence=save(a/'admission.json',dict(observed_epoch=time.time(),host=host,session=lock['session'],
        cap_registry=dict(path=str(cap),sha256=file_sha(cap)),existing_project_rows=existing,
        tracked_cap_helper=cap_helper,
        scheduler_snapshot=sq,partition=partition,memory_audit=memory,
        admitted_array_throttle=2,per_process_gpus=1,mem='60416M',
        dependency_ids=deps,aggregate_cap_proof='existingnone+array%2<=2' if not deps else 'afterany all existing admissions then array%2<=2',
        unrelated_job_mutation=0))
    args=['sbatch','--parsable','--hold',f'--output={output}/%A_%a.out',f'--error={output}/%A_%a.err']
    if deps:args.append('--dependency=afterany:'+':'.join(deps))
    args += [str(script),str(a)]
    result=command(args,w).strip()
    job=result.split(';')[0]
    if not job.isdigit():raise ValueError(('SBATCH_RESPONSE',result))
    save(a/'submission-accepted.json',dict(job=job,command=args,admission=evidence,lock_sha256=file_sha(a/'execution.lock.json'),
        array_mapping=lock['array_mapping'],status='SUBMITTED_HELD_NOT_RELEASED',initial_gpu_gate='NOT_RUN'))
    held=command(['scontrol','show','job',job,'-o'],w)
    save(a/'held-inspection-raw.json',dict(job=job,output=held))
    inspect(held,job,a)
    save(a/'held-inspection.json',dict(job=job,status='REQUEST_SOURCE_RESOURCE_PASS',mem='60416M',array='0-5%2',dependency_ids=deps))
    command(['scontrol','release',job],w)
    after=command(['squeue','-h','-r','-j',job,'-o','%i|%u|%j|%T|%D|%b|%E'],w)
    released=project_rows(after,patterns)
    assert len(released)==6,'RELEASED_SIX_CELLS_NOT_OBSERVED'
    assert all(r['state'] in ['PENDING','RUNNING','CONFIGURING'] for r in released),'UNEXPECTED_INITIAL_STATE'
    receipt=save(a/'released-inspection.json',dict(job=job,cells=released,status='RELEASED',
        all_pending=all(r['state']=='PENDING' for r in released),initial_gpu_gate='NOT_YET_RUN',
        output_root=str(a/'output'),lock_sha256=file_sha(a/'execution.lock.json'),
        monitoring='PENDING_GATE_NOT_RUN_PAUSE' if all(r['state']=='PENDING' for r in released) else 'MINIMUM_INITIAL_GATE_ONLY'))
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);args=p.parse_args();submit(args.attempt)
