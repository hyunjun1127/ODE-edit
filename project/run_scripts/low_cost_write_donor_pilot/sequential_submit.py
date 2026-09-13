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
    if not any(s in text for s in ('gres/gpu:rtx_pro_6000=1','gres:gpu:rtx_pro_6000:1','TresPerNode=gres/gpu:rtx_pro_6000:1')):
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
    # Node-list filtering omits pending jobs without an allocated NodeList.
    # This deployment's server4 QOS admits those pending reservations as well.
    pending_sq=command(['squeue','-h','-q','lab_gpu_s4','-o','%i|%u|%j|%T|%D|%b|%E'],w)
    existing=list({r['job']:r for r in project_rows(sq,patterns)+project_rows(pending_sq,patterns)}.values())
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
        scheduler_snapshot=sq,pending_qos_snapshot=pending_sq,partition=partition,memory_audit=memory,
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


def release_existing(attempt):
    """Recover only held inspection; never submit a second job or edit source."""
    a=Path(attempt).resolve();w=Path(__file__).resolve().parents[3]
    command(['scripts/check-session-boundary.sh','01a04939-b5c7-7a03-ba2d-ef3343d62cfd'],w)
    accepted=json.loads((a/'submission-accepted.json').read_text())
    job=accepted['job']
    assert job.isdigit() and not (a/'released-inspection.json').exists()
    assert file_sha(a/'execution.lock.json')==accepted['lock_sha256']
    held=command(['scontrol','show','job',job,'-o'],w)
    inspect(held,job,a)
    save(a/'held-inspection-repair-v1.json',dict(job=job,status='EXACT_REQUEST_PASS',
        cause='Slurm26 TresPerNode uses gres/gpu:type:1; inspection accepted only equals or legacy colon',
        current_held_output=held,control_source=dict(path=__file__,sha256=file_sha(__file__)),
        runtime_archive_mutation=0,science_execution_before_release=0,duplicate_submit=0,resource_changes=0))
    command(['scontrol','release',job],w)
    text=command(['squeue','-h','-r','-j',job,'-o','%i|%u|%j|%T|%D|%b|%E'],w)
    rows=project_rows(text,['odeedit_lowcost_seq10_s4'])
    assert len(rows)==6 and all(r['state'] in ['PENDING','RUNNING','CONFIGURING'] for r in rows)
    receipt=save(a/'released-inspection.json',dict(job=job,cells=rows,status='RELEASED',
        all_pending=all(r['state']=='PENDING' for r in rows),initial_gpu_gate='NOT_YET_RUN',
        output_root=str(a/'output'),lock_sha256=accepted['lock_sha256'],
        monitoring='PENDING_GATE_NOT_RUN_PAUSE' if all(r['state']=='PENDING' for r in rows) else 'MINIMUM_INITIAL_GATE_ONLY'))
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--release-existing',action='store_true')
    args=p.parse_args();(release_existing if args.release_existing else submit)(args.attempt)
