"""Immutable replacement for cancelled 50983, reusing valid 51055 hook evidence."""
import argparse
import json
from pathlib import Path
from .config import ROOT
from .reuse_completed_hook import check_completed_evidence
from project.run_scripts.single_layer_edit_preserving_correction.common import member,write


def build(destination,cancellation):
    prior=ROOT/'PROGRAM/hook-repair-r1';p=prior/'execution.lock.json'
    plan=json.loads(p.read_text());source=Path(plan['execution']['source'])/'project/run_scripts/single_layer_mechanism_first'
    plan.pop('execution',None);plan.pop('lock_identity',None)
    plan.update(output=str(ROOT/'PROGRAM/cleanup-repair-r2/output'),
        status='CLEANUP_REPAIR_PLAN_NOT_FULL_T0_PASS',
        replacement_authority='USER: 50983 도 그대로 돌리면 오류가 있을 것 같으니 수정해서 다시 올려야한다',
        submission_authority='USER: repair 해서 올려',
        old_dependent_job_mutation=True,old_job_cancel_receipt=member(cancellation),
        completed_hook_reuse=dict(job_id='51055',source='5ea4e4efad5a9420674641dd13a04d4651701a08',
            execution_lock=member(p),failure=member(prior/'output/failure.json'),
            runtime=member(prior/'output/runtime-load.json'),
            summary=member(prior/'output/technical-hook-repair/hook-summary.json'),
            source_members={name:member(source/name) for name in
                ('z_hook.py','z_hook_parity.py','config.py','model.py','technical.py','technical_repair.py','current.py')}),
        new_technical_z_request_calls=0,new_unhooked_native_reference_calls=0,
        prior_failed_allocations=[dict(job='50974',GPU_seconds=136),dict(job='51055',GPU_seconds=104)],
        prior_failed_allocation_GPU_seconds=240,remaining_full_T0='REQUIRED_NOT_RUN',
        hook_execution='REUSE_VALID_FIXED4_COMPONENT; FRESH_PROCESS_REMAINDER_T0_THEN_GATED_SCIENCE')
    check_completed_evidence(plan)
    return write(destination,plan)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cancellation',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.output,a.cancellation)))
