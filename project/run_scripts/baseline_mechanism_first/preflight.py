"""Small CPU/source receipt and first-wave plan; never submits or loads models."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from .contracts import (DESIGN_SHA, digest, execution_plan, member, save)


def prepare(worktree, root, input_lock):
    wt, root = Path(worktree).absolute(), Path(root).absolute()
    lock = json.loads(Path(input_lock).read_text())
    python = '/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
    commands = [
        [python, '-m', 'unittest', 'discover', '-s', 'project/run_scripts/baseline_mechanism_first/tests', '-t', '.', '-q'],
        [python, '-m', 'compileall', '-q', 'project/run_scripts/baseline_mechanism_first'],
        ['bash','-n','project/run_scripts/baseline_mechanism_first/launch_cold_l4.sh'],
        ['python3','scripts/slurm_memory_policy.py','audit'],
        ['bash','scripts/check-session-boundary.sh','01a04939-f93a-7b50-bca0-65438eab2062'],
        ['git','diff','--check']]
    gates=[]
    for cmd in commands:
        completed = subprocess.run(cmd,cwd=wt,text=True,capture_output=True)
        row=dict(command=cmd, exit_code=completed.returncode, stdout=completed.stdout,stderr=completed.stderr)
        gates.append(row)
        if completed.returncode:
            raise RuntimeError(row)
    count=int(re.search(r'Ran (\d+) tests',gates[0]['stderr']).group(1))
    files=[member(p) for p in sorted((wt/'project/run_scripts/baseline_mechanism_first').rglob('*'))
           if p.is_file() and p.suffix in {'.py','.md','.sh'}]
    authority=member(wt/'project/proposals/2026-09-12-baseline-mechanism-first-lifelong-editing-design.md',expected=DESIGN_SHA)
    source=dict(members=files,members_root=digest(files),base_head='f8d78c88db24cd5580c844b6a3621e48f9bc3734',
        base_tree='ed4ed74b2a45cf653925346bcd43ea0aa7c041ce',design=authority,
        original_execution_lock=lock['original_execution_lock'],input_lock=member(input_lock))
    audit=wt/'audits/servers/server1/2026-09-12-baseline-mechanism-first-e01'
    save(audit/'source-preflight-r1.json', source)
    save(audit/'cpu-preflight-r1.json',dict(status='CPU_AND_SOURCE_PASS_NOT_GPU_PASS',test_count=count,gates=gates,
        red_review=['Restorable endpoint metadata state/covariance corrected before execution',
                    'Historical companion path+SHA consumption binding corrected before execution'],
        preflight_old_input_lock_preserved=True,source_members_root=source['members_root'],
        model_load_count=0,GPU_count=0,Slurm_submit_count=0))
    plan=execution_plan()
    plan['first_wave']=dict(cell='AlphaEdit-L4-n00000',native_batches=1,evaluation_panels=['W0 Current100','Native Current100'],
        representative_FD='first canonical rewrite, fixed alpha=2^-8; no efficacy sweep',
        no_CP_equivalence_claim_before_exact_checkpoint_comparison=True,
        next_wave_requires_USER_recall=True)
    plan['resource']=dict(server='server1',node='devbox',project_cap=2,gpus=1,cpus=8,mem_mib=182272,
        wall_seconds=14400,GPU_hour_cap=None,inherit_other_budget=False,
        admission='active_and_admitted_pending_recount_before_submit_required')
    plan['cost_estimate']=dict(original_S4_L4_B001_target_seconds=270.08237051032484,
        original_S4_key_seconds=8.385147655382752,original_S4_solve_seconds=.15635457262396812,
        expected_S1_first_wave_hours_range=[.25,2.],scheduler_wall_limit_hours=4,
        range_status='PLANNING_ESTIMATE_NOT_MEASURED_HARDWARE_EXTRAPOLATION',
        scope_native_batches_max=155,full_scope_cost='UPDATE_AFTER_FIRST_S1_MEASUREMENT',
        host_memory_components_gib=dict(model_backup=30,projector_stack=3.83,history_and_copies=4,other_bound=40),
        GPU_model_FP32_gib=30,storage_first_wave_estimate_gib=5,
        general_spectrum_warm_continuation='NOT_INCLUDED_IN_FIRST_WAVE_ESTIMATE')
    save(wt/'runs/baseline-mechanism-first-e01-sh1-20260912-v1/first-wave-plan.json',plan)
    evidence=dict(scope='E0/E1 only', categories=[
        dict(kind='existing_completed_observations',status='SEALED_SINGLETON_CURRENT_SEEN_FULL_RECEIVED',
             receipt=member(root/'imports/server4-received-v1.json'), metric_regeneration=False),
        dict(kind='accessible_raw_source',status='1242_ALLOWLIST_MEMBERS_HASH_VERIFIED_15_LOCAL_REUSE'),
        dict(kind='new_existing_case_computation',status='CPU_MODULES_TESTED_NOT_YET_EXECUTED_FULL_POPULATION'),
        dict(kind='new_GPU_measurements',status='NOT_YET_RUN'),
        dict(kind='checkpoint_access',status='S2_PATH_OWNER_RESPONSE_PENDING_EXISTING_L4_WARM_LOCAL_AVAILABLE'),
        dict(kind='general_text_panel',status='NOT_YET_SEALED_NOT_IN_FIRST_NATIVE_GATE'),
        dict(kind='compute_z_internal_optimizer_metadata',status='NOT_OBSERVED_BY_INITIAL_WRAPPER')],
        imputation=False, scientific_promotion=False)
    save(wt/'experiment-reports/servers/server1/baseline-mechanism-first-e01-2026-09-12-v1/evidence-reuse-manifest.json',evidence)
    return dict(test_count=count,source_members_root=source['members_root'],input_sha=source['input_lock']['sha256'],
                model=0,GPU=0,submitted=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--root',required=True);p.add_argument('--input-lock',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.worktree,a.root,a.input_lock)))
