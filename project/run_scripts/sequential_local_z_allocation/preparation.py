"""Create-once authority binding. No original evidence writers executed."""
import argparse
from datetime import datetime,timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .common import ROOT,COLD,TASK,ARMS,LAYERS,save,identity,sha

PREFIX='plans/global/2026-09-17-sequential-local-z-allocation-sh4-dispatch'
ENVELOPE='messages/head/2026-09-17-sh4-sequential-local-z-allocation-v2.md'
SESSION='01a04939-b5c7-7a03-ba2d-ef3343d62cfd'

def command(args,check=True):
    p=subprocess.run(args,text=True,capture_output=True)
    if check and p.returncode:raise RuntimeError(dict(args=args,exit=p.returncode,stderr=p.stderr))
    return dict(args=args,exit=p.returncode,stdout=p.stdout,stderr=p.stderr)

def initialize(w):
    assert command(['hostname'])['stdout'].strip()=='server4'
    assert os.environ.get('CODEX_THREAD_ID')==SESSION
    manifest=json.loads((w/PREFIX/'source-input-manifest.json').read_text())
    members=manifest['members']
    extra=[ENVELOPE,PREFIX+'/source-input-manifest.json',PREFIX+'/execution-envelope.json',
        PREFIX+'/gh-dispatch-checks.json','PROTOCOL.md','control/gpu-concurrency-policy.tsv',
        'servers/connection-inventory.md',manifest['reference_mapping'][0]['mirror']]
    for m in members:assert sha(w/m['path'])==m['sha256'] and (w/m['path']).stat().st_size==m['bytes']
    copies=[]
    for rel in [m['path'] for m in members]+extra:
        src=w/rel;dst=ROOT/'authoritative'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
        assert sha(src)==sha(dst)
        copies.append(dict(identity(dst),relative=rel,lines=len(src.read_bytes().splitlines())))
    previous=Path('/data/janghj/ODE-edit/local/l4-preserving-repair/20260917-v1/full-read-receipt.json')
    prior=json.loads(previous.read_text());prior_docs={r['repo_path']:r for r in prior['documents']}
    assert sha(w/'PROTOCOL.md')==prior_docs['PROTOCOL.md']['sha256']
    import scipy,scipy.optimize._cobyla_py,scipy.optimize._cobyla
    assert scipy.__version__=='1.15.3'
    dependencies=[identity(Path(x.__file__)) for x in (scipy,scipy.optimize._cobyla_py,scipy.optimize._cobyla)]
    # Exact old job, resource/state only. Never read its output or repair it.
    old=command(['scontrol','show','job','49238','-o'],check=False)
    if old['exit']:
        old=command(['sacct','-X','-j','49238','--noheader','--parsable2','--format=JobIDRaw,JobName,User,State,AllocTRES,NodeList'])
    old['purpose']='ADMISSION_ONLY_NO_REPAIR_RAW_OR_JOB_MUTATION'
    save(ROOT/'old49238-admission-initial.json',old)
    disk=shutil.disk_usage(ROOT)
    result=dict(instruction_id=TASK,session=SESSION,host='server4',time=datetime.now(timezone.utc).isoformat(),
        publication_base=command(['git','-C',str(w),'rev-parse','HEAD'])['stdout'].strip(),
        documents=copies,reading='FULL_READ new documents; exact prior FULL_READ reuse for identical PROTOCOL/native/cold7 prose',
        prior_full_read=identity(previous),python=sys.version,scipy=scipy.__version__,dependencies=dependencies,
        new_science=list(ARMS),batches=60,unique_requests=1000,arm_request_observations=6000,gpu_cap=2,
        old49238=identity(ROOT/'old49238-admission-initial.json'),old_repair_future_scope='STOPPED_NOT_RESUMED',
        checkpoint_policy='NO_DISK_W_M_CHECKPOINT; immutable structurally-shared RAM snapshots; scalar/target/hash evidence retained',
        exact_crash_resume='NOT_AVAILABLE',old_cold7_21CP='USER_DELETED_NOT_RECREATED',
        resource_estimate=dict(free_bytes=disk.free,planned_disk_reserve_GiB=64,
            snapshot_strategy='share unchanged layer buffers and all precommit memories; no full-model candidate copy',
            per_process_M_P_GiB=5*2*14336*14336*4/2**30,
            wall='TECHNICAL12H_RESERVE; SCIENCE_ESTIMATE_AFTER_MEASURED_PILOT',actual_model='NOT_RUN'),
        scientific_validation='NOT_RUN',initial_policy='INITIAL_GATE_ONLY_OR_PENDING_HANDOFF',
        session_helper='Prior stale-session/missing-worktree-config BLOCK preserved; direct environment/registry exact; helper not claimed PASS')
    save(ROOT/'full-read-m0.json',result);print(json.dumps(dict(receipt=identity(ROOT/'full-read-m0.json'),old_admission=old),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);a=p.parse_args();initialize(Path(a.worktree))
