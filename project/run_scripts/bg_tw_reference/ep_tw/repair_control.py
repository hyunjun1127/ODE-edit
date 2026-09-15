"""Create-once repair authority/input/source seals; no model or teacher rebuild."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tarfile
import subprocess
import sys
from datetime import datetime, timezone

from .control import REPO, boundary, git, identity, save, sha, verify_dispatch, DISPATCH

TASK='ODEEDIT-S06-EP-TW1-SAVED-EPISODE-FD-REPAIR-SH4-V1'
ENVELOPE='messages/head/2026-09-15-sh4-ep-tw1-saved-episode-fd-repair.md'
REPAIR_DISPATCH='plans/global/2026-09-15-ep-tw1-saved-episode-fd-repair-dispatch.json'
OLD=REPO/'local/ep-tw1-c4/20260915-v1/attempt-v1'
EPISODE=OLD/'scientific-v1/B001/native-targets-map.pt'
EPISODE_SHA='67b1aaaf765fca753e3399016967f62b2e0cf3f5b2f374874b1eb9d9aae88451'

def verify_repair(path):
    d=json.loads(Path(path).read_text())
    assert d['instruction_id']==TASK and d['prior_failed_job']=='47884'
    assert d['reuse']['native_target_calls_diagnostic']==d['reuse']['native_solves_diagnostic']==0
    assert d['diagnostic']['grid_factors']==[2.**-k for k in range(10)]
    assert d['diagnostic']['max_signed_grid_observations']==80
    assert d['diagnostic']['relative_derivative_tolerance']==.15
    assert d['diagnostic']['absolute_derivative_tolerance']==1e-7
    assert d['diagnostic']['relative_convergence_tolerance']==.15
    assert d['scientific']['new_replacement_chains']==1 and d['scientific']['conditional_after_technical_PASS']
    assert d['scientific']['baseline_reruns']==0 and not d['scientific']['quality_restoration'] and not d['scientific']['ODE']
    return d

def initialize(worktree,attempt):
    w,a=Path(worktree).resolve(),Path(attempt).resolve(); b=boundary(w)
    a.mkdir(parents=True,exist_ok=False,mode=0o700)
    verify_repair(w/REPAIR_DISPATCH);verify_dispatch(w/DISPATCH)
    docs=[ENVELOPE,REPAIR_DISPATCH,'PROTOCOL.md',DISPATCH,
        'audits/global/2026-09-15-ep-tw1-failure-review-ko.md',
        'audits/global/2026-09-15-ep-tw1-failure-review-checks.json',
        'audits/global/2026-09-15-ep-tw1-saved-episode-repair-dispatch-source-inventory.json',
        'plans/global/2026-09-15-edit-quality-preserving-tw-design-v3.md',
        'plans/global/2026-09-15-edit-quality-preserving-tw-contract.json',
        'plans/global/2026-09-15-bg-tw-reference-data-contract.json']
    expected={ENVELOPE:'16999f5cae1435f2a43c6b7174fd5d667d271dd12e019db2c0e56078f8ee5130',
        REPAIR_DISPATCH:'503753f4a324fdfb43486cd7aae42ac96448d30444c5bdc2fd8090b0bbd27bbb',
        docs[4]:'4c2241e0709a98d2d16ae401633bb7b4afe0018c8c67dac7855d637a85e88c89',
        docs[5]:'ac50462b1301faa13187cb7ac99e73cffdc9c04c410e9a06aa998583909ddef8'}
    copied=[]
    for rel in docs:
        p=w/rel;r=identity(p)
        if rel in expected:assert r['sha256']==expected[rel]
        dst=a/'authoritative'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        with p.open('rb') as src,dst.open('xb') as out:shutil.copyfileobj(src,out)
        dst.chmod(0o400);copied.append(dict(r,lines=len(p.read_bytes().splitlines()),copy=str(dst)))
    refs=[OLD/'execution.lock.json',OLD/'resume-manifest.json',OLD/'failure-diagnosis-r1/diagnostic-report-ko.md',
          OLD/'failure-diagnosis-r1/receipt.json',OLD/'failure-diagnosis-r1/cpu-evidence.json',OLD/'teacher-reuse-verification.json']
    assert sha(refs[0])=='306ad09b56d470ce47562bd3f36230d7f96fe258769dc0fb5b3dc192c33d954b'
    return save(a/'full-read-receipt.json',dict(instruction_id=TASK,time=datetime.now(timezone.utc).isoformat(),
        boundary=b,documents=copied,reuse=[identity(p) for p in refs],full_read_by_parent=True,
        actual_old_source='3fb0bfb27773ec9d016e76fcdc978dc996f010a7',
        new_actual_GPU_checks='NOT_RUN',old_scalar36_not_GPU_PASS=True,
        no_broadcast='NO_BROADCAST_NOT_REQUIRED',gpu_cap=2,gpu_hour_cap=None))

def freeze(worktree,attempt):
    from .repair_checks import RepairNumerics
    w,a=Path(worktree).resolve(),Path(attempt).resolve();b=boundary(w)
    assert not git(w,'status','--porcelain','--','project/run_scripts/bg_tw_reference/ep_tw'),'UNCOMMITTED_REPAIR_SOURCE'
    old=json.loads((OLD/'execution.lock.json').read_text())
    # Reuse exact old model/teacher/environment evidence, without reading their
    # multi-GB payloads again. Runtime stat checks and original source hashes remain.
    src=a/'source-v1';src.mkdir(mode=0o700,exist_ok=False)
    oldsrc=Path(old['source_root']);members=[];rels=[]
    for m in old['members']:
        p=Path(m['path'])
        if p.is_relative_to(oldsrc):rels.append(str(p.relative_to(oldsrc)))
        else:members.append(dict(m))
    rels += [str(p.relative_to(w)) for p in (w/'project/run_scripts/bg_tw_reference/ep_tw').glob('*.py')]
    rels += [ENVELOPE,REPAIR_DISPATCH,'project/run_scripts/bg_tw_reference/ep_tw/repair.sbatch']
    rels=sorted(set(rels))
    for rel in rels:
        p=w/rel;dst=src/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        with p.open('rb') as inp,dst.open('xb') as out:shutil.copyfileobj(inp,out)
        dst.chmod(0o400);members.append(identity(dst))
    check_paths=sorted(a.glob('cpu-checks*.json'),key=lambda p:p.stat().st_mtime_ns)
    assert check_paths,'CPU_CHECKS_REQUIRED'
    latest=json.loads(check_paths[-1].read_text())
    assert all(identity(m['path'])==m for m in latest['source_members']),'CPU_TESTED_SOURCE_DRIFT'
    for p in (a/'full-read-receipt.json',a/'authoritative/user-checks.md',*check_paths):
        members.append(identity(p))
    archive=a/'source-v1.tar'
    with archive.open('xb') as f,tarfile.open(fileobj=f,mode='w') as tar:
        for rel in rels:
            info=tar.gettarinfo(str(src/rel),arcname=rel);info.uid=info.gid=info.mtime=0;info.uname=info.gname='';info.mode=0o400
            with (src/rel).open('rb') as inp:tar.addfile(info,inp)
    assert EPISODE.stat().st_size==256307027 and sha(EPISODE)==EPISODE_SHA
    # Frozen stats bind reused heavy members after their prior exact full-SHA
    # checks. No claim of new full model/teacher rehash.
    stats=[]
    for m in members:
        p=Path(m['path']);st=p.stat();assert st.st_size==m['bytes']
        stats.append(dict(path=str(p),dev=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,bytes=st.st_size))
    sample=json.loads(Path(old['sample_lock']['path']).read_text())
    from scripts.fixed_counterfact import load_prefix
    rows=load_prefix(old['dataset_root'],1000)
    assert [r['case_id'] for r in rows]==sum([x['case_ids'] for x in old['batches']],[])
    assert sample['prefix1000_root']=='40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd'
    science=copy.deepcopy(old)
    science.update(instruction_id=TASK,source_head=b['source_head'],source_tree=b['source_tree'],source_root=str(src),
        source_archive=identity(archive),members=members,member_stats=stats,
        dispatch=identity(src/DISPATCH),repair_dispatch=identity(src/REPAIR_DISPATCH),
        prior_failed_lock=identity(OLD/'execution.lock.json'),
        repair_numerics=RepairNumerics().to_dict(),
        repair_pass_path=str(a/'technical-v1/technical-PASS.json'),
        output=str(a/'scientific-r2'),early_diagnostics=True,
        resource=dict(gpu=1,cpus=8,mem='60416M',wall='12:00:00',gpu_cap=2,gpu_hour_cap=None),
        cost_plan=dict(technical_estimate_seconds=[300,1800],science_estimate_hours=[2,8],
            wall_seconds=43200,total_new_disk_reserve_bytes=40*(1<<30),
            old_failed_gpu_seconds=473,teacher_reused_seconds=98,
            old_native_fit_seconds=286.54453050531447,old_fit_nested_in_failed_cost=True))
    science['diagnostic_budget']=dict(grid_signed_objective_observations=80,
        E_grid_groups=280,D_grid_documents=2560,baseline_extra_E_groups=14,baseline_extra_D_documents=128,
        direct_backward_E_groups=7,direct_backward_D_documents=64,
        residual_backward_E_groups=7,residual_backward_D_documents=64,
        observer_true_E_groups=7,functional_materialization_forwards=4,extra_rechecks=0,
        scientific_fresh_endpoint_policy='exact Vp/A/gE/gD reuse; otherwise same bounded endpoint checks once',
        model_W0_selfKL_reuse='prior full S64 actual forward; fresh scientific W0 selfKL remains')
    sref=save(a/'scientific.lock.json',science)
    tech=copy.deepcopy(science)
    tech.update(mode='SAVED_NATIVE_EPISODE_TECHNICAL_ONLY',scientific_lock=sref,
        output=str(a/'technical-v1'),saved_episode=identity(EPISODE),
        old_fit_receipt=identity(OLD/'scientific-v1/B001/native-fit.json'),
        old_cpu_receipt=identity(OLD/'failure-diagnosis-r1/cpu-evidence.json'),
        diagnostic_seed=2026091503,native_target_calls=0,native_solves=0,
        preserved_old_coarse=identity(OLD/'scientific-v1/failure.json'),
        reuse_scope='technical only; no post-fit RNG/original-gradient byte equality or continuation claim')
    return save(a/'repair.lock.json',tech)

def preflight(worktree,attempt):
    w,a=Path(worktree).resolve(),Path(attempt).resolve();b=boundary(w)
    checks=[]
    commands=[[sys.executable,'-m','unittest','discover','-s','project/run_scripts/bg_tw_reference/ep_tw','-t','.','-p','test_*.py'],
        ['bash','-n','project/run_scripts/bg_tw_reference/ep_tw/repair.sbatch'],
        [sys.executable,'scripts/slurm_memory_policy.py','audit','project/run_scripts/bg_tw_reference/ep_tw/repair.sbatch'],
        ['git','diff','--check']]
    for argv in commands:
        p=subprocess.run(argv,cwd=w,capture_output=True,text=True)
        checks.append(dict(command=argv,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
        if p.returncode:raise RuntimeError(checks[-1])
    changed=[]
    for p in sorted((w/'project/run_scripts/bg_tw_reference/ep_tw').glob('*')):
        if p.suffix in ('.py','.sbatch'):
            if p.suffix=='.py':compile(p.read_bytes(),str(p),'exec')
            changed.append(identity(p))
    original='3fb0bfb27773ec9d016e76fcdc978dc996f010a7'
    immutable=['project/run_scripts/bg_tw_reference/ep_tw/policy.py',
        'project/run_scripts/bg_tw_reference/ep_tw/ledger.py',
        'project/run_scripts/bg_tw_reference/native_map.py',
        'project/run_scripts/low_cost_write_donor_pilot/fitting.py']
    import hashlib
    same=[]
    for rel in immutable:
        before=subprocess.check_output(['git','show',original+':'+rel],cwd=w)
        assert hashlib.sha256(before).hexdigest()==sha(w/rel),('FORBIDDEN_SEMANTIC_DIFF',rel)
        same.append(identity(w/rel))
    destination=a/'cpu-checks.json';revision=1
    while destination.exists():
        revision+=1;destination=a/f'cpu-checks-v{revision}.json'
    return save(destination,dict(status='CPU_SOURCE_CHECKS_PASS_NOT_GPU_PASS',checks=checks,
        byte_compile=True,source_members=changed,unchanged_scientific_implementations=same,
        boundary=b,new_GPU_checks='NOT_RUN',diagnostic_native_fit_calls=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['initialize','preflight','freeze']);p.add_argument('--worktree',required=True);p.add_argument('--attempt',required=True)
    args=p.parse_args();print(json.dumps(globals()[args.command](args.worktree,args.attempt)))
