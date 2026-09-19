"""Versioned USER FD-only exception; reuse completed checks, never mark FD PASS."""
import argparse
import json
from .config import ROOT
from .program import require_program
from .reuse_t0_user_waiver import FD_WAIVER_ID, validate
from project.run_scripts.single_layer_edit_preserving_correction.common import member, write


def build(destination):
    prior=ROOT/'PROGRAM/b1-repair-r3a'
    plan=json.loads((prior/'execution.lock.json').read_text())
    plan.pop('execution');plan.pop('lock_identity')
    plan.update(status='USER_AUTHORIZED_B1_WITH_FD_NOT_ESTABLISHED',
        output=str(ROOT/'PROGRAM/b1-fd-waiver-r4/output'),
        latest_user_quote='통과할테니 task 이어서 진행해',
        FD_user_waiver=dict(id=FD_WAIVER_ID,user_quote='통과할테니 task 이어서 진행해',
            question_context='FD 미확립을 명시한 채 방법 자체의 보호조건은 유지하고 B1만 진행',
            threshold_changed=False,FD_as_scientific_PASS=False,B1_only=True),
        completed_T0_reuse=dict(job_id='51057',
            execution_lock=member(prior/'execution.lock.json'),
            runtime=member(prior/'output/runtime-load.json'),
            checks=member(prior/'output/technical-decision/checks.json'),
            failure=member(prior/'output/failure.json')),
        full_numerical_validation='NOT_ESTABLISHED',
        prior_failed_allocation_GPU_seconds=1113,
        prior_failed_allocations=plan['prior_failed_allocations']+[dict(job='51057',GPU_seconds=538)],
        new_T0_GPU_calls=0,
        remaining_full_T0='REUSED_51057; FD_ONLY_USER_WAIVED_NOT_PASS',
        repair_scope='CONTROL_ONLY_USER_FD_WAIVER_NO_METHOD_GUARD_CHANGE')
    require_program(plan);validate(plan)
    return write(destination,plan)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    print(json.dumps(build(p.parse_args().output)))
