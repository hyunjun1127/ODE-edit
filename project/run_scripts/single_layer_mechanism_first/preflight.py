"""CPU-only create-once authority/input/resource sealing, no job submission."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from .config import ROOT, INSTRUCTION, AUTHORITY, NUMERIC
from project.run_scripts.single_layer_edit_preserving_correction.common import sha, member, write, digest

DOCS = [
    'project/proposals/2026-09-19-single-layer-mechanism-first-gh-instruction.md',
    'plans/global/2026-09-19-single-layer-mechanism-first-experiment-design-v1.md',
    'plans/global/2026-09-19-single-layer-mechanism-first-contract-v1.json',
    'plans/global/2026-09-19-single-layer-mechanism-first-cells-v1.csv',
    'audits/global/2026-09-19-single-layer-mechanism-first-dispatch/reference/mechanism-and-method-review.md',
    'audits/global/2026-09-19-single-layer-mechanism-first-dispatch/reference/en-capacity-scope-audit.md',
    'audits/global/2026-09-19-single-layer-mechanism-first-dispatch/authority-manifest.json',
    'messages/head/2026-09-19-sh4-single-layer-mechanism-first-z-hook.md',
    'PROTOCOL.md', 'servers/active/server4.md', 'servers/connection-inventory.md',
    'control/gpu-concurrency-policy.tsv']
PRIOR = Path('/data/janghj/ODE-edit/local/en-execution-reuse/20260919-v1/B1/attempt-v1/execution.ready.lock.json')


def authority(repo):
    receipts=[]
    for rel in DOCS:
        source=repo/rel; dest=ROOT/'authoritative'/rel
        content=source.read_bytes();dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if dest.read_bytes()!=content:raise ValueError('AUTHORITATIVE_BYTES_CHANGED:'+rel)
        else:
            with dest.open('xb') as f:f.write(content)
        receipts.append(dict(repo_path=rel,**member(dest),full_read=True))
    return receipts


def resource_snapshot():
    stat=os.statvfs(ROOT)
    q=subprocess.run(['squeue','-h','-u','janghj','-o','%i|%j|%T|%b|%D|%R'],check=True,capture_output=True,text=True)
    return dict(time_unix=time.time(),hostname=socket.gethostname(),
        free_bytes=stat.f_bavail*stat.f_frsize,free_inodes=stat.f_favail,
        resource_only_queue=q.stdout.splitlines(),new_GPU_jobs=0,
        user_cleanup_assumed=False,storage_waiver=False)


def build(repo, stage, attempt):
    old=json.loads(PRIOR.read_text())
    # Preserve exact asset lineage, not the old scientific scope/skip/routing.
    keep=['blue_root','config4','snapshot','projector','cold_capsule','P_star_basis','torch',
          'transformers','transformers_import','numpy','scipy','generated_ready','reference_inputs',
          'historical_evaluator_root','helper_scripts_root','prior_large_asset_binding','external_members',
          'editor_sha256','model_revision','model_config_sha256','model_weights_identity_sha256',
          'tokenizer_identity_sha256','dataset_root']
    lock={k:old[k] for k in keep}
    from scripts.fixed_counterfact import load_prefix
    records=load_prefix(lock['dataset_root'],1000)
    lock.update(instruction=INSTRUCTION,authority_commit=AUTHORITY,stage=stage,
        output=str(ROOT/stage/attempt/'output'),seed=20260916,
        records_digest=digest(records),sample_order=[r['case_id'] for r in records],
        numerical=NUMERIC,project_gpu_cap=2,task_gpu_cap=2,
        prior_skip_or_storage_waiver_inherited=False,save_checkpoints=False,disk_state_checkpoints=False,
        resources=dict(GPU=1,CPU=8,mem_MiB=60416,node='server4',export='NONE',requeue=0,
                       wall_hours=24,GPUhour_hardcap=None),
        prior_asset_lock=member(PRIOR),full_read=authority(repo),
        source_reference_only='17b5a133cd050ea94195e690aab33342e754965b',
        hook_reference=member('/data/janghj/tmp/dnm/hooking.py'),
        resource_preflight=resource_snapshot(),
        storage_plan=dict(B1_new_upper_GiB=24,CP_and_weights_GiB=0,
            derivative_factors_GiB=4,geometry_GiB=5,temporary_and_reports_GiB=15,
            existing_teacher_reused_not_copied=True,stage_recheck_required=True,
            estimate_not_reserved_or_measured=True),
        wall_plan=dict(T0_hours=2,B1_hours=8,configured_job_limit_hours=24,
            estimated_not_measured=True,S3_S10_reestimate_after_prior_stage=True),
        native_plan='FRESH_B1_SHARED_ONCE_WITH_VALIDATED_HOOK; own-entry fresh on later branches',
        later_submission='ONLY_AFTER_MECHANICAL_PRIOR_GATE',
        NO_BROADCAST_NOT_REQUIRED='server4 existing immutable inputs and server-local new outputs')
    if socket.gethostname()!='server4':raise ValueError('HOST_BOUNDARY')
    if lock['resource_preflight']['free_bytes'] < 24*2**30:
        raise OSError('STORAGE_PREFLIGHT_LESS_THAN_NEW_B1_24GiB_ESTIMATE')
    # The actual executable/archive closure is added by source freeze.
    lock['status']='CPU_INPUT_PLAN_NOT_EXECUTION_LOCK'
    return write(ROOT/'receipts'/f'{stage}-{attempt}-input-plan.json',lock)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--stage',choices=['T0','B1'],default='T0');p.add_argument('--attempt',default='attempt-v1')
    a=p.parse_args();print(json.dumps(build(a.repo,a.stage,a.attempt)))
