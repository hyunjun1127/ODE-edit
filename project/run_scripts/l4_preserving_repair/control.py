"""CPU binding, source freeze and bounded held submission; never a daemon."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from .common import ROOT,COLD,TASK,ARMS,save,identity,sha,digest

PREFIX='project/run_scripts/l4_preserving_repair/'
MAN='plans/global/2026-09-17-l4-preserving-repair-sh4-dispatch/source-input-manifest.json'
ENV='messages/head/2026-09-17-sh4-l4-preserving-repair-twoarm.md'
SESSION='01a04939-b5c7-7a03-ba2d-ef3343d62cfd'

def command(args,**kw):
    r=subprocess.run(args,text=True,capture_output=True,**kw)
    if r.returncode: raise RuntimeError(dict(args=args,stdout=r.stdout,stderr=r.stderr,exit=r.returncode))
    return r.stdout.strip()

def git(w,*args): return command(['git','-C',str(w),*args])

def copied(source,target):
    p=Path(target);p.parent.mkdir(parents=True,exist_ok=True)
    with Path(source).open('rb') as inp,p.open('xb') as out:shutil.copyfileobj(inp,out)
    assert sha(source)==sha(p);return identity(p)

def initialize(w):
    assert command(['hostname'])=='server4'
    assert os.environ.get('CODEX_THREAD_ID')==SESSION
    assert git(w,'remote','get-url','origin')=='https://github.com/hyunjun1127/ODE-edit.git'
    inputs=json.loads((w/MAN).read_text())['copies']; refs=[]
    extra=[MAN,ENV,'PROTOCOL.md','control/gpu-concurrency-policy.tsv','servers/connection-inventory.md',
        'plans/global/2026-09-17-l4-preserving-repair-sh4-dispatch.json',
        'plans/global/2026-09-17-l4-preserving-repair-sh4-dispatch/dispatch-approved-cells.csv',
        'plans/global/fixed-counterfact-10k-policy.md']
    for row in inputs:
        assert sha(w/row['path'])==row['sha256'] and (w/row['path']).stat().st_size==row['bytes']
    for rel in [r['path'] for r in inputs]+extra:
        refs.append(dict(copied(w/rel,ROOT/'authoritative'/rel),repo_path=rel,
            lines=len((w/rel).read_bytes().splitlines()),reading='FULL_READ'))
    old=json.loads((COLD/'execution.lock.json').read_text())
    native=[]
    for rel in ('AlphaEdit/AlphaEdit_main.py','AlphaEdit/compute_z.py','AlphaEdit/compute_ks.py'):
        native.append(dict(identity(Path(old['blue_root'])/rel),reading='FULL_READ'))
    for rel in ('low_cost_write_donor_pilot/fitting.py','local_z_adaptive_allocation/model.py',
                'local_z_adaptive_allocation/policy.py','local_z_adaptive_allocation/runner.py',
                'local_z_adaptive_allocation/technical.py','bg_tw_reference/ep_tw/model_adapter.py'):
        native.append(dict(identity(w/'project/run_scripts'/rel),reading='FULL_READ_REUSED_FUNCTIONS_AND_NATIVE_CONNECTION'))
    result=dict(instruction_id=TASK,time=datetime.now(timezone.utc).isoformat(),host='server4',session=SESSION,
        source_head=git(w,'rev-parse','HEAD'),tree=git(w,'rev-parse','HEAD^{tree}'),documents=refs,sources=native,
        session_helper='BLOCK_STALE_ROOT_SESSION_AND_MISSING_WORKTREE_CONFIG; environment and tracked registry match; no helper PASS',
        scientific_arms=list(ARMS),cap=2,science_batches=20,unique_requests=1000,arm_requests=2000,
        checkpoint_override='최신 사용자: checkpoint는 저장하지 말고 진행하라.',
        disk_checkpoint=False,exact_restart='NOT_AVAILABLE',in_memory_rollback=True,
        GPU_actions=0,numerical_validation='NOT_RUN',monitoring='INITIAL_GATE_ONLY_OR_PENDING_HANDOFF')
    save(ROOT/'full-read-receipt.json',result)
    print(json.dumps(identity(ROOT/'full-read-receipt.json')))

def freeze(w):
    from scripts.fixed_counterfact import load_prefix
    from .numerical import POLICY
    from .geometry import DEFAULT_GEOMETRY_POLICY
    from .qp import DEFAULT_QP_POLICY
    from dataclasses import asdict
    assert not git(w,'status','--porcelain','--',PREFIX),'UNCOMMITTED_RUNTIME'
    cpu_ref=identity(ROOT/'cpu-checks-v1.json')
    cpu=json.loads(Path(cpu_ref['path']).read_text())
    assert cpu['status']=='PASS' and cpu['model_GPU_tests']==0
    for member in cpu['source_members']:
        assert sha(w/member['relative'])==member['sha256'],'CPU_CHECK_SOURCE_DRIFT'
    old=json.loads((COLD/'execution.lock.json').read_text())
    oldroot=Path(old['source_root']);source=ROOT/'source-v1';source.mkdir(exist_ok=False)
    relative=set();external=[]
    # Preserve the exact prior input closure. Prior scientific results are not inputs.
    for m in old['members']:
        p=Path(m['path'])
        if p.is_relative_to(oldroot):
            rel=str(p.relative_to(oldroot))
            if rel.startswith(('project/','scripts/')) and p.suffix=='.py':relative.add(rel)
        elif m.get('verification')=='PRIOR_FULL_SHA_STABLE_STAT':
            st=p.stat();assert st.st_size==m['bytes'] and [st.st_dev,st.st_ino,st.st_mtime_ns]==m['stat']
            external.append(m)
    relative.update(str(p.relative_to(w)) for p in (w/PREFIX).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    relative.update([ENV,MAN,'scripts/fixed_counterfact.py'])
    members=[];lineage=[]
    for rel in sorted(relative):
        p=w/rel if (w/rel).is_file() else oldroot/rel
        members.append(copied(p,source/rel));lineage.append(dict(relative=rel,input=identity(p)))
        (source/rel).chmod(0o400)
    archive=ROOT/'source-v1.tar'
    with archive.open('xb') as handle,tarfile.open(fileobj=handle,mode='w') as tar:
        for rel in sorted(relative):
            info=tar.gettarinfo(str(source/rel),arcname=rel)
            info.uid=info.gid=info.mtime=0;info.uname=info.gname='';info.mode=0o400
            with (source/rel).open('rb') as f:tar.addfile(info,f)
    records=load_prefix(old['dataset_root'],1000)
    capsule=identity(COLD/'technical/attempt-v1/cold-capsule.json')
    cold=json.loads(Path(capsule['path']).read_text())
    assert cold['seed']==20260916 and cold['teacher_regenerated'] is False
    disk=shutil.disk_usage(ROOT); reserve=48*(1<<30)
    assert disk.free>reserve,'STORAGE_RESERVE'
    keys=('blue_root','snapshot','projector','config4','historical_evaluator_root','helper_scripts_root','editor_sha256',
          'reference_root','teacher_manifest','dataset_root','torch','transformers','model_revision')
    lock={k:old[k] for k in keys}
    lock.update(instruction_id=TASK,source_head=git(w,'rev-parse','HEAD'),source_tree=git(w,'rev-parse','HEAD^{tree}'),
        source_root=str(source),source_archive=identity(archive),members=members+external+[capsule,identity(ROOT/'full-read-receipt.json'),cpu_ref],
        old_asset_lock=identity(COLD/'execution.lock.json'),cold_capsule=capsule,
        arms=list(ARMS),seed=20260916,gpu_cap=2,batches=10,batch_size=100,tf32_matmul=False,tf32_cudnn=False,
        records_digest=digest(records),prefix1000_ordered_root=old['prefix1000_ordered_root'],whole_ordered_root=old['whole_ordered_root'],
        batch_locks=old['batch_locks'],numerical=POLICY,geometry_policy=asdict(DEFAULT_GEOMETRY_POLICY),
        QP_policy=asdict(DEFAULT_QP_POLICY),technical_output=str(ROOT/'technical/attempt-v1'),
        common_ready=str(ROOT/'technical/attempt-v1/READY.json'),
        event_id_encoding=old['event_id_encoding'],state_policy=old['state_policy'],
        storage=dict(disk_checkpoint=False,selected_W_M_RNG_payload=False,checkpoint_override='USER_NO_CHECKPOINT',
            in_memory_rollback=True,exact_restart='NOT_AVAILABLE',Q_response_diagnostics=True,
            native_fit_weight_payload=False,free_bytes=disk.free,reserve_bytes=reserve,
            estimate='20 basis payloads <=14.1GB plus native target/response/JSON/technical diagnostics and 32GB reserve; not measured'),
        resource=dict(gpus=1,cpus=8,mem_MiB=60416,technical_wall_hours=12,hour_cap=None,
            science_wall='LOCK_AFTER_PILOT_MEASUREMENTS',queue='technical then two independent mains if READY and aggregate cap permits'),
        early_cohort_ordinals=[0,100],W0_reference=identity(COLD/'technical/attempt-v1/W0-first1000.json'),
        after_gate='MONITORING_PAUSED_AWAITING_USER',automatic_resume=False)
    save(ROOT/'execution.lock.json',lock);save(ROOT/'source-lineage.json',dict(source_head=lock['source_head'],members=lineage))
    print(json.dumps(identity(ROOT/'execution.lock.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','freeze']);p.add_argument('--worktree',required=True)
    a=p.parse_args();globals()[a.action](Path(a.worktree).resolve())
