"""One-batch user recall; immutable serialization repair, no new hook fitting."""
import argparse
import json
from .config import ROOT
from .program import B1_ONLY_AUTHORITY, require_program
from .reuse_completed_hook import check_completed_evidence
from project.run_scripts.single_layer_edit_preserving_correction.common import member, write


def build(destination):
    prior=ROOT/'PROGRAM/cleanup-repair-r2'
    plan=json.loads((prior/'execution.lock.json').read_text())
    plan.pop('execution');plan.pop('lock_identity')
    plan.update(status='B1_ONLY_SERIALIZATION_REPAIR_PLAN',
        output=str(ROOT/'PROGRAM/b1-repair-r3/output'),
        maximum_batch=1,sequential_authorized=False,auto_continue=False,
        agent_monitoring_after_release=True,b1_only_authority=B1_ONLY_AUTHORITY,
        latest_user_quote='batch 1개만 우선 모니터링 계속 하면서 task 마무리해',
        submission_authority='USER_B1_ONLY_MONITOR_TO_COMPLETION_20260920',
        serialization_failure=member(prior/'output/failure.json'),
        prior_r2_lock=member(prior/'execution.lock.json'),
        prior_failed_allocation_GPU_seconds=575,
        prior_failed_allocations=[dict(job='50974',GPU_seconds=136),
            dict(job='51055',GPU_seconds=104),dict(job='51056',GPU_seconds=335)],
        remaining_full_T0='REQUIRED_BEFORE_B1; PARTIAL_PRIOR_EVIDENCE_NOT_FULL_PASS',
        monitoring='THIS_B1_ATTEMPT_UNTIL_TERMINAL_AND_CPU_REPORT',
        method_threshold_changes=False,
        repair_scope='NUMPY_BOOL_RECEIPT_SERIALIZATION_AND_USER_B1_STOP_ONLY',
        checkpoint_saved=False,save_checkpoints=False,disk_state_checkpoints=False,
        exact_resume='NOT_AVAILABLE',automatic_resume=False)
    require_program(plan);check_completed_evidence(plan)
    return write(destination,plan)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    print(json.dumps(build(p.parse_args().output)))
