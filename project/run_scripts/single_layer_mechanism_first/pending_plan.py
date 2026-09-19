"""Latest USER queue-only/noCP override, without observing prior job results."""
import argparse
import json
from pathlib import Path
from .config import ROOT
from project.run_scripts.single_layer_edit_preserving_correction.common import write,member


def build(destination):
    old=ROOT/'receipts/T0-attempt-v1-input-plan.json'
    value=json.loads(old.read_text());value.pop('execution',None);value.pop('lock_identity',None)
    hook=ROOT/'T0/hook-attempt-v1/execution.lock.json'
    sealed=json.loads(hook.read_text())
    value.update(stage='PROGRAM',phase='GATED_PROGRAM',maximum_batch=10,
        scientific_gates_required=True,scheduler_writes_in_program=False,
        agent_monitoring_after_release=False,disk_state_checkpoints=False,save_checkpoints=False,
        storage_policy=dict(instruction='ODEEDIT-ALL-SH-DEFAULT-NO-CHECKPOINT-20260919-V1',
            publication='4d871dc6aeb328675d4912f4525ed23434bc0ad8',exceptions=[],
            checkpoint_equivalents=False,exact_resume='NOT_AVAILABLE'),
        in_memory_chain_state=True,exact_crash_resume='NOT_AVAILABLE_USER_NO_CP',
        user_override=dict(instructions=['실행기 구현 마치고 pending 까지만 걸어놔',
                                        'checkpoint 저장은 하지말고 진행해'],
            existing_jobs_cancelled=False,new_checkpoint_deletion_authority=False,
            scientific_threshold_changes=False,prior_noCP_waiver_inherited=False),
        hook_dependency=dict(job_id='50974',lock=member(hook),source=sealed['execution']['commit']),
        later_submission='NONE: ALL_CONDITIONAL_COMPUTATION_IN_ONE_DEPENDENT_JOB',
        status='CPU_PROGRAM_PLAN_NOT_ACTUAL_VALIDATION',
        output=str(ROOT/'PROGRAM/dependency-attempt-v1/output'))
    value['resources'].update(wall_hours=168,GPU=1)
    value['wall_plan']=dict(configured_job_limit_hours=168,estimate_not_measured=True,
        stage_cost_observed_in_program=True,user_GPUhour_hardcap=None,
        hard_limit_not_scientific_cost_gate=True)
    value['storage_plan']=dict(initial_T0_B1_estimate_GiB=24,S3_additional_estimate_GiB=32,
        S10_additional_estimate_GiB=112,stagewise_check=True,measured=False,exclusive_reserved=False,
        state_checkpoint_payload_bytes=0,keys_small_factors_ledgers_and_gradient_hashes_retained=True,
        full_gradient_direction_or_update_tensors_saved=False,
        existing_artifacts_deleted=False,no_storage_waiver=True,
        global_current_free_space_does_not_guarantee_conditional_S10=True)
    return write(destination,value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.output)))
